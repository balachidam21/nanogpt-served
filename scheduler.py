"""
Continuous batching — Sub-project (June 12).

The Jun-9 system-design mock surfaced this as a black box: "if requests arrive
and finish at different times, how does ONE model serve them all efficiently?"

STATIC batching: pick a batch, run it in LOCKSTEP, everyone leaves when the
SLOWEST finishes. Two wastes — finished requests keep riding the batch as dead
weight (idle slots), and no new request can start until the whole wave drains
(head-of-line blocking).

CONTINUOUS batching (vLLM's "iteration-level scheduling"): the batch is
re-decided EVERY decode step. A finished request leaves immediately; a waiting
request takes the freed slot the very next step. The GPU stays full; new work
doesn't wait on old.

TOY SCOPE — we RECOMPUTE the full context each step (no KV cache). Why throw
away the cache we built Jun 7? Because the module-level, single-sequence cache
(self.cache_k) does NOT compose with a batch whose membership and lengths change
every step. That mismatch is precisely what PagedAttention (vLLM) solves, and
it's the NEXT build. Today the spotlight is the SCHEDULER, so we trade speed for
a correct, readable batch path.

Run:  uv run python scheduler.py
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


@dataclass
class Request:
    """One generation request. `tokens` holds prompt + everything generated so
    far; we decode until `max_new_tokens` of them have been produced."""
    rid: int
    tokens: list                 # token ids: prompt first, generated appended
    max_new_tokens: int
    prompt_len: int = field(default=0)

    def __post_init__(self):
        self.prompt_len = len(self.tokens)

    @property
    def generated(self):
        return len(self.tokens) - self.prompt_len

    @property
    def done(self):
        return self.generated >= self.max_new_tokens


class ContinuousBatchingScheduler:
    """Serves a queue of Requests through one model. `run_static_baseline()` is
    the slow 'before'; `run()` (which calls your `step()`) is continuous batching."""

    def __init__(self, model, max_batch_size, device, temperature=1.0):
        self.model = model
        self.max_batch_size = max_batch_size
        self.device = device
        self.temperature = temperature
        self.waiting = []    # requests not yet admitted to the batch (the queue)
        self.running = []    # requests currently in the decode batch
        self.finished = []   # completed requests
        # stats (filled by run / run_static_baseline)
        self.steps = 0
        self.useful_slot_steps = 0   # slot-steps spent decoding a NOT-yet-done request
        self.total_slot_steps = 0    # every slot the batch occupied, useful or idle

    def add_request(self, req):
        self.waiting.append(req)

    # ------------------------------------------------------------------ #
    # The tensor plumbing — given a list of (ragged-length) requests, run
    # ONE forward pass and append one sampled token to each. This is the
    # "decode step" both schedulers share. You do NOT need to edit this.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _decode(self, batch):
        lens = [len(r.tokens) for r in batch]
        max_len = max(lens)
        # RIGHT-pad to a rectangle so it's one tensor. Real tokens sit at
        # positions 0..len-1; trailing pads are never attended to (the causal
        # mask blocks position i from seeing anything > i), and we read each
        # row's logits at its last REAL position — so the pads are harmless.
        inp = torch.zeros(len(batch), max_len, dtype=torch.long, device=self.device)
        for i, r in enumerate(batch):
            inp[i, : lens[i]] = torch.tensor(r.tokens, device=self.device)
        logits, _ = self.model(inp)                          # (B, max_len, vocab)
        rows = torch.arange(len(batch), device=self.device)
        last = torch.tensor([l - 1 for l in lens], device=self.device)
        step_logits = logits[rows, last] / self.temperature  # (B, vocab) — last REAL pos
        probs = F.softmax(step_logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,) one token / row
        for r, tok in zip(batch, nxt.tolist()):
            r.tokens.append(tok)

    # ------------------------------------------------------------------ #
    # The 'before': STATIC batching, for comparison. Process the queue in
    # fixed waves; run each wave lockstep until its SLOWEST finishes; idle
    # slots (finished requests) keep getting decoded as dead weight. You do
    # NOT need to edit this either — it's the baseline your step() beats.
    # ------------------------------------------------------------------ #
    def run_static_baseline(self):
        self.model.eval()
        queue = list(self.waiting)
        while queue:
            wave = [queue.pop(0) for _ in range(min(self.max_batch_size, len(queue)))]
            while not all(r.done for r in wave):
                self.total_slot_steps += len(wave)                  # whole wave occupies slots
                self.useful_slot_steps += sum(not r.done for r in wave)  # ...some are dead weight
                self._decode(wave)
                self.steps += 1
            self.finished += wave
        return self._stats("static")

    # ------------------------------------------------------------------ #
    # The 'after': CONTINUOUS batching. run() loops your step() until every
    # request is served.
    # ------------------------------------------------------------------ #
    def run(self):
        self.model.eval()
        while self.waiting or self.running:
            self.step()
        return self._stats("continuous")

    def step(self):
        """ONE iteration of continuous batching. This is the heart of the idea —
        you write it. See TODO(human) below."""
        # TODO(human): implement one iteration-level scheduling step.
        #
        # You have: self.waiting (queue), self.running (current batch),
        #           self.finished, self.max_batch_size, and the helpers
        #           self._decode(batch) (one forward → appends a token to each),
        #           plus the stat counters self.steps / self.useful_slot_steps /
        #           self.total_slot_steps.
        #
        # An iteration should, in some order you decide:
        #   1. ADMIT — while there's room in the batch (len(self.running) <
        #      self.max_batch_size) AND self.waiting is non-empty, move one
        #      waiting request into self.running. (This is what fills slots
        #      freed by last step's evictions — the thing static batching can't do.)
        #   2. DECODE — run one step on the current batch: self._decode(self.running).
        #      Bump self.steps; add len(self.running) to BOTH slot-step counters
        #      (in continuous batching every decoded slot is useful — why?).
        #   3. EVICT — move every finished request (r.done) out of self.running
        #      and into self.finished, so its slot is free for the next iteration.
        #
        # Think about ORDER: which of admit / decode / evict has to happen before
        # which for the batch to stay full and for no finished request to get
        # decoded twice? That ordering choice IS the design.
        # raise NotImplementedError("implement the continuous-batching step — see TODO(human)")
        # ADMIT: fill EVERY free lane (while, not if). Both the initial empty
        # batch and a lane freed by last step's eviction get filled to capacity.
        while len(self.running) < self.max_batch_size and self.waiting:
            self.running.append(self.waiting.pop(0))
        
        # DECODE: one forward over the current batch (membership won't change
        # between admit above and evict below, so len(self.running) is stable here).
        self._decode(self.running)
        self.steps += 1
        # ACCUMULATE (+=, not =). Every lane held a not-done request at decode
        # time (we evict at the end of each step), so every slot-step is useful.
        self.total_slot_steps += len(self.running)
        self.useful_slot_steps += len(self.running)

        # EVICT: finished requests leave, freeing their lane for the next step.
        done = [r for r in self.running if r.done]
        self.finished.extend(done)
        self.running = [r for r in self.running if not r.done]

    # ------------------------------------------------------------------ #
    def _stats(self, mode):
        util = (self.useful_slot_steps / self.total_slot_steps * 100) if self.total_slot_steps else 0.0
        return {
            "mode": mode,
            "steps": self.steps,
            "useful_slot_steps": self.useful_slot_steps,
            "total_slot_steps": self.total_slot_steps,
            "utilization_pct": util,
            "served": len(self.finished),
        }


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = GPTConfig()  # random weights — scheduling/utilization don't depend on training
    torch.manual_seed(1337)
    model = GPT(config).to(device)

    # 5 requests needing [2, 6, 3, 4, 2] tokens — the exact scenario from the
    # ASCII timeline. Batch size 3. Static should take 10 steps, continuous 6.
    SPECS = [(1, 2), (3, 6), (2, 3), (1, 4), (4, 2)]  # (prompt_len, max_new_tokens)

    def make_requests():
        torch.manual_seed(123)  # same prompts for both runs → fair comparison
        reqs = []
        for rid, (plen, mnt) in enumerate(SPECS):
            prompt = torch.randint(0, config.vocab_size, (plen,)).tolist()
            reqs.append(Request(rid=rid, tokens=prompt, max_new_tokens=mnt))
        return reqs

    sched_static = ContinuousBatchingScheduler(model, max_batch_size=3, device=device)
    for r in make_requests():
        sched_static.add_request(r)
    s = sched_static.run_static_baseline()

    sched_cont = ContinuousBatchingScheduler(model, max_batch_size=3, device=device)
    for r in make_requests():
        sched_cont.add_request(r)
    c = sched_cont.run()  # <-- needs your step()

    print(f"\n{'':>12} | {'steps':>5} | {'slot-steps':>10} | {'util':>6} | served")
    print("-" * 52)
    for st in (s, c):
        print(f"{st['mode']:>12} | {st['steps']:>5} | "
              f"{st['useful_slot_steps']:>3}/{st['total_slot_steps']:<6} | "
              f"{st['utilization_pct']:>5.0f}% | {st['served']}")
