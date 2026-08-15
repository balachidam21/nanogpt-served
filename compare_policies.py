"""
Preemption policies, scored side by side.

`paged_scheduler.py` holds ONE policy — the one you committed to. This file runs
several against the same workload so the trade-offs stop being arguments and
start being numbers.

Two workloads, because they expose different failures:

  CLOSED — 5 requests, nothing new arrives. Everything eventually drains, so a
           starving request still finishes once its rivals leave. Flatters any
           policy that starves somebody.

  OPEN   — short requests keep arriving. Nothing drains. This is what a real
           server looks like, and it is where starvation stops being cosmetic:
           a request that is always the cheapest victim is preempted forever.

Run:  uv run python compare_policies.py
"""

from paged_kv import BlockPool
from paged_scheduler import PagedScheduler, Req


# ---------------------------------------------------------------------- #
# The policies. Each takes (scheduler, requester) and returns one running Req.
# ---------------------------------------------------------------------- #
POLICIES = {
    # Yours: protect whoever is closest to finishing; preempt whoever has the
    # most work left. Shortest-remaining-first, applied to eviction.
    "most-remaining (yours)":
        lambda s, req: max(s.running, key=lambda r: r.max_new_tokens - r.generated),

    # vLLM's answer: preempt the most recently arrived. The point is not that
    # it's cheap — it's that ARRIVAL ORDER NEVER CHANGES, so the oldest request
    # is permanently safe and the queue drains in order.
    "newest-arrival (vLLM)":
        lambda s, req: s.running[-1],

    # The inverse, as a control. Should be catastrophic.
    "oldest-arrival":
        lambda s, req: s.running[0],

    # Pure cost minimization: throw away the least work.
    "cheapest-recompute":
        lambda s, req: min(s.running, key=lambda r: r.context_len),

    # Free the most blocks per preemption — fewest preemptions, priciest each.
    "most-blocks-held":
        lambda s, req: max(s.running, key=lambda r: r.context_len),

    # The asker gives up its own turn. Nobody else is disturbed.
    "self-preempt":
        lambda s, req: req,
}


def build(policy_fn, specs, num_blocks, block_size, max_batch):
    pool = BlockPool(num_blocks, block_size)
    sched = PagedScheduler(pool, max_batch_size=max_batch)
    sched.select_victim = lambda requester: policy_fn(sched, requester)
    for rid, (plen, mnt) in enumerate(specs):
        sched.waiting.append(Req(rid=rid, prompt_len=plen, max_new_tokens=mnt, arrival=rid))
    return sched


def score(name, specs, num_blocks, block_size, max_batch, cap):
    sched = build(POLICIES[name], specs, num_blocks, block_size, max_batch)
    try:
        rep = sched.run(max_steps=cap)
    except RuntimeError as e:
        return {"name": name, "outcome": f"DEADLOCK ({str(e)[:28]}…)"}

    stalled = bool(sched.waiting or sched.running)
    worst = max((r.times_preempted for r in sched.finished), default=0)
    # Anyone still unfinished at the cap has effectively been starved.
    stuck = [r for r in sched.waiting + sched.running]
    if stuck:
        worst = max(worst, max(r.times_preempted for r in stuck))
    return {
        "name": name,
        "outcome": "LIVELOCK" if stalled else "ok",
        "steps": rep["steps"],
        "served": rep["served"],
        "preemptions": rep["preemptions"],
        "recomputed": rep["tokens_recomputed"],
        "worst": worst,
        "stuck": [r.rid for r in stuck],
    }


def table(title, specs, num_blocks, block_size, max_batch, cap):
    total = sum(p + m for p, m in specs)
    print("\n" + "=" * 88)
    print(f"{title}   |   {len(specs)} requests, {num_blocks * block_size} token slots, "
          f"{total / (num_blocks * block_size):.1f}x oversubscribed")
    print("=" * 88)
    print(f"  {'policy':<24}{'steps':>7}{'served':>8}{'preempt':>9}{'recomp':>8}"
          f"{'worst-hit':>11}   outcome")
    print("  " + "-" * 84)
    for name in POLICIES:
        r = score(name, specs, num_blocks, block_size, max_batch, cap)
        if "steps" not in r:
            print(f"  {r['name']:<24}{'—':>7}{'—':>8}{'—':>9}{'—':>8}{'—':>11}   {r['outcome']}")
            continue
        note = r["outcome"]
        if r["stuck"]:
            note += f" — R{','.join(str(i) for i in r['stuck'])} never finished"
        print(f"  {r['name']:<24}{r['steps']:>7}{r['served']:>4}/{len(specs):<3}"
              f"{r['preemptions']:>9}{r['recomputed']:>8}{r['worst']:>9}x    {note}")


if __name__ == "__main__":
    BS, NB, MAXB = 4, 8, 4

    # CLOSED: the demo workload from paged_scheduler.py. Finite; always drains.
    CLOSED = [(4, 12), (4, 4), (8, 10), (4, 6), (4, 3)]
    table("CLOSED WORKLOAD — finite arrivals", CLOSED, NB, BS, MAXB, cap=200)

    # OPEN: short traffic, then ONE long request, then more short traffic.
    # Two design constraints, both learned the hard way:
    #   (1) the long request must FIT — 4+16=20 tokens = 5 of 8 blocks. A request
    #       larger than the pool can never finish under any policy, which looks
    #       like starvation in the results but is really an impossible request.
    #   (2) the long request must NOT be the oldest, or "most-remaining" and
    #       "oldest-arrival" pick the same victim and the comparison says nothing.
    OPEN = [(4, 3)] * 4 + [(4, 16)] + [(4, 3)] * 9   # R4 is the long one
    table("OPEN WORKLOAD — steady short traffic, one long request (R4)",
          OPEN, NB, BS, MAXB, cap=400)

    print("\n" + "-" * 88)
    print("  worst-hit = the single most-preempted request. It is the fairness number;")
    print("  'preempt' and 'recomp' are the efficiency numbers. They do not agree.")
    print("-" * 88)
