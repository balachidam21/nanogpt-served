"""
PagedAttention, part 2 — the GATHER (attention over scattered blocks).

Part 1 (`paged_kv.py`) built a page table with no MMU. `translate(37)` will tell
you "physical block 5, offset 5" — and then nothing happens. No tensor is ever
touched. This file is the part that actually goes and fetches.

THE PROBLEM. Standard attention is written against a contiguous cache:

    K = cache_k[:T]                      # (T, d) — one slice, one stride
    scores = q @ K.T / sqrt(d)

That single slice is doing a LOT of unearned work. It assumes position 0 and
position 37 are `37 * d * itemsize` bytes apart. Under paging that is false:
position 0 might live in physical block 5 and position 37 in physical block 2,
which sits EARLIER in memory. The logical sequence is contiguous; the physical
layout is confetti.

THE FIX. Put the whole KV region in ONE big tensor and address it by SLOT:

    pool_k : (num_blocks, block_size, d)      # every physical block, one tensor
    slot   = physical_block * block_size + offset

Then a request's keys are not a slice — they are a GATHER: `T` slot ids pulled
out of the pool in logical order. Contiguity gets rebuilt on the fly, per read,
instead of being paid for up front in reserved HBM.

WHY THIS IS THE PIECE THAT MATTERS. Everything paging buys you (the 84.6% -> 2.8%
you measured yesterday) is only real if attention can still run at full speed on
the scattered layout. If the gather is slow, you traded memory for latency and
lost. That is why vLLM does NOT do what this file does — it fuses the gather INTO
the attention kernel so the scattered K/V is read straight into SRAM and the
contiguous copy is never materialized in HBM at all. Materializing it, as here,
is the honest first version: correct, obvious, and measurably wasteful. Knowing
exactly what it wastes is the point.

SCOPE: single head, single request, decode step only (one query token attending
to T cached keys). No batching, no causal mask — decode only ever attends to the
past, so the mask is implicitly satisfied.

Run:  uv run python paged_attention.py
"""

import math
import torch

from paged_kv import BlockPool, BlockTable


# ---------------------------------------------------------------------- #
# The physical KV pool, now with actual bytes.
#
# `paged_kv.BlockPool` hands out block IDs and knows nothing about tensors.
# This wraps it with the storage those IDs point INTO. Same split vLLM uses:
# the block manager is integer bookkeeping, the cache is a slab of memory.
# ---------------------------------------------------------------------- #
class PagedKVCache:
    def __init__(self, num_blocks, block_size, d_head, dtype=torch.float32):
        self.pool = BlockPool(num_blocks, block_size)
        self.block_size = block_size
        self.d_head = d_head
        # In a real engine these two are allocated ONCE at server startup and
        # never grow. Their size is what caps how many requests fit — this
        # tensor IS the "KV memory wall" from the batch-economics session.
        self.pool_k = torch.zeros(num_blocks, block_size, d_head, dtype=dtype)
        self.pool_v = torch.zeros(num_blocks, block_size, d_head, dtype=dtype)

    def write_token(self, table: BlockTable, k, v):
        """Append one token's K and V for this request. One decode step = one call.

        Note the ordering: the block table decides WHERE first (allocating a
        physical block if the tail is full), and only then do we write bytes.
        Bookkeeping leads, memory follows.
        """
        if not table.append_token():
            return False  # pool exhausted — scheduler's problem (build (c))
        pos = table.num_tokens - 1                 # the slot we just claimed
        block, offset = table.translate(pos)
        self.pool_k[block, offset] = k
        self.pool_v[block, offset] = v
        return True


# ---------------------------------------------------------------------- #
# THE GATHER — the MMU. Turn a scattered request into a contiguous (T, d).
# ---------------------------------------------------------------------- #
def gather_kv_loop(pool_tensor, table: BlockTable):
    """VERSION 1 — the obvious one. Walk the positions, ask translate() each time.

    This is literally the page-table walk drawn T times. Position by position:
    "where does logical i live?" -> copy that row out. Then stack the rows into
    a matrix.

    Correct, readable, and issues T separate indexing operations.
    """
    rows = []
    for pos in range(table.num_tokens):
        block, offset = table.translate(pos)   # the page-table walk
        rows.append(pool_tensor[block, offset])          # (d,)  one token's vector
    return torch.stack(rows)                             # (T, d) rows stacked


def gather_kv(pool_tensor, table: BlockTable):
    """VERSION 2 — vectorized. Compute ALL T addresses at once, index ONCE.

    Same output as the loop, different cost. The loop's problem is not the copying
    (both versions copy exactly T*d numbers) — it is that every iteration is a
    round trip out to Python: bounds-check, build a small tensor, dispatch a
    kernel. T=2048 means 2048 of those. Here Python runs a fixed handful of
    operations no matter how long the sequence is.

    THE KEY IDEA — flatten the pool and address it by SLOT:

        pool_tensor : (num_blocks, block_size, d)
        flat        : (num_blocks * block_size, d)
        slot        = physical_block * block_size + offset

    Once every token has a single integer address, "gather" is one index.
    """
    num_blocks, block_size, d = pool_tensor.shape
    T = table.num_tokens
    if T == 0:
        return pool_tensor.new_zeros(0, d)                # (0, d) — nothing held yet

    # -- STEP 1: this request's physical blocks, in LOGICAL order. ------------
    # The list position is the logical index; the value is the physical address.
    blocks = torch.tensor(table.block_ids, dtype=torch.long)   # (L,)
    # print("BLOCKS:", blocks)

    # -- STEP 2: the first slot of each of those blocks. ----------------------
    base = blocks * block_size                                 # (L,)
    # print("BASE:" , base, base.shape)

    # -- STEP 3: every slot inside every block, via BROADCASTING. -------------
    # (L, 1) + (block_size,) -> (L, block_size). Torch stretches both operands
    # to the common shape: each row is one block's base + [0, 1, ..., bs-1].
    offsets = torch.arange(block_size)                         # (block_size,)
    slots = base[:, None] + offsets                            # (L, block_size)
    # print("SLOTS:", slots)

    # -- STEP 4: flatten to reading order, then cut the tail. -----------------
    # Row-major flatten walks block 0's slots, then block 1's, ... which IS
    # logical order. The slice drops the unfilled slots of the last block —
    # safe ONLY because the append-only invariant guarantees every interior
    # block is full, so all the slack is at the end.
    slots = slots.reshape(-1)[:T]                              # (T,)
    # print("SLOTS RESHAPE:", slots)

    # -- STEP 5: one index into the flattened pool. ---------------------------
    # .view (not .reshape) on purpose: it must be a free reinterpretation of the
    # same bytes, never a copy. It succeeds because pool_tensor is contiguous.
    flat = pool_tensor.view(num_blocks * block_size, d)        # (nb*bs, d)
    return flat[slots]                                         # (T, d)


def paged_attention(q, cache: PagedKVCache, table: BlockTable):
    """One decode step: query attends over the request's whole cached history.

    q: (d,) — the single new token's query
    """
    K = gather_kv(cache.pool_k, table)             # (T, d)
    V = gather_kv(cache.pool_v, table)             # (T, d)
    scores = (K @ q) / math.sqrt(cache.d_head)     # (T,)
    weights = torch.softmax(scores, dim=-1)        # (T,)
    return weights @ V                             # (d,)


def naive_attention(q, K, V, d_head):
    """Ground truth: the same math over a plain contiguous cache."""
    scores = (K @ q) / math.sqrt(d_head)
    return torch.softmax(scores, dim=-1) @ V


# ====================================================================== #
if __name__ == "__main__":
    torch.manual_seed(0)
    BS, D = 4, 8                      # tiny block size + head dim, readable traces

    # -- DEMO 1: correctness. Paged output must EQUAL the contiguous answer. --
    print("=" * 70)
    print(f"DEMO 1 — paged vs contiguous, block_size={BS}, d_head={D}")
    print("=" * 70)
    cache = PagedKVCache(num_blocks=16, block_size=BS, d_head=D)

    # Two requests interleaved, so request A's blocks end up NON-CONSECUTIVE.
    # This is the whole test: if gather_kv secretly assumes contiguity, the
    # interleaving is what exposes it.
    a = BlockTable(pool=cache.pool, rid=0)
    b = BlockTable(pool=cache.pool, rid=1)
    ref_ka, ref_va = [], []
    for step in range(9):
        k, v = torch.randn(D), torch.randn(D)
        cache.write_token(a, k, v)
        ref_ka.append(k)
        ref_va.append(v)
        if step % 2 == 0:                          # B grows too, stealing blocks
            cache.write_token(b, torch.randn(D), torch.randn(D))

    print(f"  {a.render()}")
    print(f"  {b.render()}")
    print(f"  A's physical blocks consecutive? {a.block_ids == list(range(a.block_ids[0], a.block_ids[0] + len(a.block_ids)))}")

    q = torch.randn(D)
    got = paged_attention(q, cache, a)
    want = naive_attention(q, torch.stack(ref_ka), torch.stack(ref_va), D)
    print(f"\n  paged   {got[:4].tolist()}")
    print(f"  naive   {want[:4].tolist()}")
    print(f"  max abs diff: {(got - want).abs().max().item():.2e}")
    assert torch.allclose(got, want, atol=1e-6), "GATHER IS WRONG"
    print("  ✓ identical")

    # -- DEMO 2: the gather in isolation — is logical order preserved? --
    print("\n" + "=" * 70)
    print("DEMO 2 — logical order survives physical scatter")
    print("=" * 70)
    K = gather_kv(cache.pool_k, a)
    print(f"  gathered shape {tuple(K.shape)}  (expected ({a.num_tokens}, {D}))")
    for pos in (0, 4, 8):
        blk, off = a.translate(pos)
        match = torch.equal(K[pos], cache.pool_k[blk, off])
        print(f"  logical pos {pos} -> block {blk}, offset {off}   row matches: {match}")

    # -- DEMO 3: loop vs vectorized — same answer, different cost. --
    print("\n" + "=" * 70)
    print("DEMO 3 — loop vs vectorized gather")
    print("=" * 70)
    K_loop = gather_kv_loop(cache.pool_k, a)
    K_vec = gather_kv(cache.pool_k, a)
    print(f"  loop shape {tuple(K_loop.shape)} · vectorized shape {tuple(K_vec.shape)}")
    print(f"  identical: {torch.equal(K_loop, K_vec)}")

    # Now at a realistic sequence length, where the Python overhead shows up.
    big = PagedKVCache(num_blocks=200, block_size=16, d_head=128)
    big_t = BlockTable(pool=big.pool, rid=0)
    for _ in range(2048):
        big.write_token(big_t, torch.randn(128), torch.randn(128))

    import time
    for name, fn in (("loop", gather_kv_loop), ("vectorized", gather_kv)):
        fn(big.pool_k, big_t)                       # warm up
        t0 = time.perf_counter()
        for _ in range(20):
            fn(big.pool_k, big_t)
        ms = (time.perf_counter() - t0) / 20 * 1000
        print(f"  T=2048, d=128  {name:>10}: {ms:7.3f} ms/gather")
    print("  Same bytes copied either way. The gap is Python dispatch, not memory.")

    # -- DEMO 4: what the copy costs. The reason vLLM fuses instead. --
    print("\n" + "=" * 70)
    print("DEMO 4 — the price of materializing the gather")
    print("=" * 70)
    T = a.num_tokens
    copied = 2 * T * D * cache.pool_k.element_size()   # K and V
    print(f"  per decode step, per request, per LAYER: {copied} bytes copied HBM->HBM")
    print(f"  scaled to a real model (T=2048, d_head=128, 32 layers, 64 reqs, fp16):")
    volume = 2 * 2048 * 128 * 2 * 32 * 64      # bytes of KV gathered
    traffic = 2 * volume                       # a copy crosses the bus TWICE: read + write
    print(f"    {volume / 1e9:.1f} GB gathered  ->  {traffic / 1e9:.1f} GB of HBM traffic")
    print("  ...at ~3 TB/s that is ~"
          f"{traffic / 3e12 * 1000:.1f} ms/step of bandwidth spent moving data that was")
    print("  already in HBM. This is why the gather gets FUSED into the kernel.")
