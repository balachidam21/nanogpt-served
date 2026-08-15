"""
PagedAttention, part 3 — PREEMPTION (what happens when the pool runs dry).

Part 1 built the allocator. Part 2 built the gather. Both assumed the happy path:
`allocate()` returns a block. This file is the unhappy path, and it is not an
error case — under load it is the STEADY STATE. An inference server is sized so
the pool is nearly full nearly always; that is what "capacity-bound, not
compute-bound" means in practice.

WHY THIS IS NOT CACHE EVICTION. A CPU cache evicts a line that nobody currently
needs — the data still lives in RAM, the eviction costs a refill later, and no
computation is undone. A KV block has no backing store and no cold entries:
every block belongs to a request that will need it on the very next decode step.
So there is nothing to "evict." You can only PREEMPT — stop a live request,
release its blocks, and put it back in the queue.

THE COST MODEL. Preempting is not free and its price is not constant:

    recompute cost of preempting request R  =  prompt_len(R) + generated(R)

because the generated TOKENS are never lost (they're just integers in a list) —
only their KV is. Resuming R means re-prefilling everything it holds. A request
that has been decoding for 400 steps costs 400+ tokens of recompute to bring
back. A request that just started costs almost nothing.

That asymmetry is the entire policy question, and it pulls against fairness:
the cheapest request to preempt is usually the one that arrived most recently,
and preempting newcomers repeatedly is exactly how you starve them.

(vLLM also offers SWAP — copy the blocks to CPU RAM and copy them back later,
trading PCIe bandwidth for recompute FLOPs. We implement RECOMPUTE only; it is
what vLLM defaults to for short sequences, and it needs no second memory pool.)

Run:  uv run python paged_scheduler.py
"""

import math
from dataclasses import dataclass, field

from paged_kv import BlockPool, BlockTable


@dataclass(eq=False)   # identity, not field-equality: `victim in self.running` must
                       # mean "this exact object", never "one that looks like it"
class Req:
    """One request. Note what survives a preemption and what doesn't:
    `generated` survives (the token ids are cheap), `table` does not (the KV)."""
    rid: int
    prompt_len: int
    max_new_tokens: int
    arrival: int = 0                      # lower = arrived earlier
    table: BlockTable = None              # None while waiting, live while running
    generated: int = 0                    # tokens produced so far (survives preemption)
    times_preempted: int = 0
    tokens_recomputed: int = 0

    @property
    def done(self):
        return self.generated >= self.max_new_tokens

    @property
    def context_len(self):
        """Tokens whose KV must exist for this request to take another step."""
        return self.prompt_len + self.generated

    def __repr__(self):
        return f"R{self.rid}"


class PagedScheduler:
    """Continuous batching, but admission is limited by BLOCKS, not by slots.

    `scheduler.py` capped the batch with `max_batch_size` — a made-up number
    standing in for "how much fits." Here the real constraint is visible: the
    pool. A request is admitted only if its KV fits, and a running request can
    be thrown out to make room for another.
    """

    def __init__(self, pool: BlockPool, max_batch_size):
        self.pool = pool
        self.max_batch_size = max_batch_size
        self.waiting = []      # FIFO queue; preempted requests go to the FRONT
        self.running = []      # kept in arrival order
        self.finished = []
        self.steps = 0
        self.preemptions = 0
        self.tokens_recomputed = 0
        self.trace = []

    # ------------------------------------------------------------------ #
    # THE POLICY — the only decision in this file, and it is yours.
    # ------------------------------------------------------------------ #
    def select_victim(self, requester: Req):
        """Pool is dry and `requester` needs one more block. Return the running
        request to preempt, or None if there is nothing to give.

        `requester` may itself be a legal answer.
        """
        return max(self.running, key=lambda r: r.max_new_tokens - r.generated)

    # ------------------------------------------------------------------ #
    # MECHANISM — given. Policy decides WHO; this decides what HAPPENS.
    # ------------------------------------------------------------------ #
    def _preempt(self, victim: Req):
        """Stop a running request: hand its blocks back, requeue it at the front.

        Front, not back: the request has already waited once and is older than
        everything behind it. Appending to the tail would punish it twice for a
        capacity problem that wasn't its fault.
        """
        freed = len(victim.table.block_ids)
        victim.table.release()
        victim.table = None
        self.running.remove(victim)
        self.waiting.insert(0, victim)
        victim.times_preempted += 1
        self.preemptions += 1
        self.trace.append(f"    preempt {victim} (ctx {victim.context_len}, freed {freed} blocks)")
        return freed

    def _prefill(self, req: Req):
        """Materialize KV for `context_len` tokens. Used both for a first
        admission and for resuming a preempted request — which is the whole
        point: to the pool, a resume IS a prefill, just a longer one."""
        need = req.context_len
        blocks_needed = math.ceil(need / self.pool.block_size)
        if self.pool.num_free < blocks_needed:
            return False
        table = BlockTable(pool=self.pool, rid=req.rid)
        for _ in range(need):
            assert table.append_token(), "checked free blocks above; must not fail"
        req.table = table
        if req.generated:                       # this was a RESUME, not a fresh start
            req.tokens_recomputed += need
            self.tokens_recomputed += need
        return True

    def _admit(self):
        """Fill free lanes from the head of the queue. Strict FIFO — we stop at
        the first request that doesn't fit rather than skipping past it. Skipping
        would raise throughput and let a large request starve forever."""
        while len(self.running) < self.max_batch_size and self.waiting:
            if not self._prefill(self.waiting[0]):
                break
            req = self.waiting.pop(0)
            self.running.append(req)
            self.running.sort(key=lambda r: r.arrival)
            self.trace.append(f"    admit {req} (ctx {req.context_len})")

    # ------------------------------------------------------------------ #
    def step(self):
        self.steps += 1
        self.trace.append(f"  step {self.steps}: free={self.pool.num_free}/{self.pool.num_blocks} blocks")
        self._admit()

        for req in list(self.running):
            if req not in self.running:
                continue                        # preempted earlier in this same step
            landed = False
            while True:
                if req.table.append_token():
                    landed = True
                    break
                victim = self.select_victim(req)
                if victim is None:
                    raise RuntimeError(
                        f"pool dry, {req} needs a block, select_victim returned None "
                        f"while {len(self.running)} requests are running — deadlock")
                if victim not in self.running:
                    raise RuntimeError(f"select_victim returned {victim}, which is not running")
                self._preempt(victim)
                if victim is req:
                    break                       # it preempted itself; no token this step
            if landed:
                req.generated += 1

        done = [r for r in self.running if r.done]
        for r in done:
            r.table.release()
            r.table = None
            self.running.remove(r)
            self.finished.append(r)
            self.trace.append(f"    finish {r}")

    def run(self, max_steps=500):
        while (self.waiting or self.running) and self.steps < max_steps:
            self.step()
        return self.report()

    # ------------------------------------------------------------------ #
    def report(self):
        return {
            "steps": self.steps,
            "served": len(self.finished),
            "preemptions": self.preemptions,
            "tokens_recomputed": self.tokens_recomputed,
            "per_request": {r.rid: (r.times_preempted, r.tokens_recomputed)
                            for r in sorted(self.finished, key=lambda r: r.rid)},
            "completion_order": [r.rid for r in self.finished],
        }


# ====================================================================== #
if __name__ == "__main__":
    BS, NB, MAXB = 4, 8, 4          # 8 blocks x 4 slots = 32 token slots, 4 lanes

    # Deliberately oversubscribed: peak demand is well past 32 slots, so the
    # pool WILL go dry and select_victim WILL be called.
    SPECS = [  # (prompt_len, max_new_tokens)
        (4, 12),
        (4, 4),
        (8, 10),
        (4, 6),
        (4, 3),
    ]

    def build():
        pool = BlockPool(NB, BS)
        sched = PagedScheduler(pool, max_batch_size=MAXB)
        for rid, (plen, mnt) in enumerate(SPECS):
            sched.waiting.append(Req(rid=rid, prompt_len=plen, max_new_tokens=mnt, arrival=rid))
        return sched

    print("=" * 72)
    print(f"pool: {NB} blocks x {BS} slots = {NB * BS} token slots | lanes: {MAXB}")
    total_demand = sum(p + m for p, m in SPECS)
    print(f"peak token demand if all ran to completion: {total_demand} slots "
          f"({total_demand / (NB * BS):.1f}x oversubscribed)")
    print("=" * 72)

    sched = build()
    rep = sched.run()

    print("\n".join(sched.trace))
    print("\n" + "-" * 72)
    print(f"  steps              {rep['steps']}")
    print(f"  served             {rep['served']}/{len(SPECS)}")
    print(f"  preemptions        {rep['preemptions']}")
    print(f"  tokens recomputed  {rep['tokens_recomputed']}   <- pure wasted FLOPs")
    print(f"  completion order   {rep['completion_order']}")
    print("\n  per request (times preempted, tokens recomputed):")
    for rid, (n, toks) in rep["per_request"].items():
        bar = "!" * n
        print(f"    R{rid}: preempted {n}x  recomputed {toks:>3} tokens  {bar}")
    print("-" * 72)
    print("  STARVATION CHECK: is the damage spread, or is one request paying for")
    print("  everyone else? A policy can look great on total recompute and still be")
    print("  unshippable if one request never finishes.")
