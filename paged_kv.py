"""
PagedAttention, part 1 — the ALLOCATOR (block pool + block table).

The question this answers: the Jun-12 scheduler had to throw away the KV cache
(see scheduler.py, line 17) because a single contiguous `cache_k` tensor cannot
serve a batch whose membership changes every step. Why not? Because a contiguous
buffer forces you to decide a request's MAX length at admit time and reserve it
all up front. Reserve 2048 slots for a request that generates 37 tokens and you
have burned 2011 slots of HBM on nothing — internal fragmentation. With ~1 GB of
KV cache per request, that waste is the thing that limits how many requests fit,
and KV memory is the wall that bites FIRST (before compute).

The fix is ONE LEVEL OF INDIRECTION. Contiguity was never a requirement of
attention — it was an artifact of how the allocator handed out memory. So:

  * Chop the KV region into fixed-size BLOCKS of `block_size` token-slots each.
    All blocks live in one shared POOL. Any block can go to any request.
  * Give each request a BLOCK TABLE: a list where logical block index -> physical
    block id. Logically the request sees one contiguous run of positions;
    physically its blocks are scattered anywhere in the pool.
  * Reading position `p` becomes two integer ops plus a table lookup:
        logical = p // block_size      -> which entry of my table
        offset  = p %  block_size      -> where inside that block
        physical = block_table[logical]
    That is a page-table walk. This is not *like* OS paging, it IS OS paging.

Why paging is EASIER here than in an OS: decode is APPEND-ONLY. A request never
writes into the middle of its own history, so once a block fills up it is sealed
forever. Only the TAIL block is ever writable. That invariant is what makes
waste bounded by construction rather than by tuning: every interior block is
100% full, so the ONLY waste is in the last partially-filled block —
at most `block_size - 1` slots per request, full stop.

SCOPE — this file is bookkeeping over integers ONLY. No torch, no tensors, no
attention. That is deliberate: paging is an allocation problem, and vLLM splits
it the same way (its block manager knows nothing about CUDA). Next builds:
  (a) the GATHER — an attention kernel that reads K/V across scattered blocks
  (b) POOL EXHAUSTION — what the scheduler does when allocate() returns None

Run:  uv run python paged_kv.py
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------- #
# The POOL. Owns every physical block; knows NOTHING about requests.
# In the real system each physical block is a slice of a big preallocated
# KV tensor of shape (num_blocks, block_size, ...). Here a block is just
# its integer id — the id is the whole point, the bytes are not.
# ---------------------------------------------------------------------- #
class BlockPool:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))

    @property
    def num_free(self):
        return len(self.free_blocks)

    @property
    def num_used(self):
        return self.num_blocks - self.num_free

    def allocate(self):
        """Hand out one physical block, or None if the pool is dry.

        Returning None rather than raising is a deliberate contract: running out
        of KV blocks is a NORMAL, expected condition under load, not a bug. It is
        a signal to the scheduler ('preempt someone'), which is build (b)."""
        if not self.free_blocks:
            return None
        return self.free_blocks.pop(0)

    def free(self, block_ids):
        """Return blocks to the pool. No zeroing — the next owner overwrites."""
        self.free_blocks.extend(block_ids)


# ---------------------------------------------------------------------- #
# The BLOCK TABLE. Belongs to ONE request; knows NOTHING about other
# requests. logical block index -> physical block id.
#
# Direction matters: mid-decode the kernel asks "where is the key for
# position 37 of THIS request?", so it needs logical -> physical. A
# physical -> request map would force a scan of the whole pool.
# ---------------------------------------------------------------------- #
@dataclass
class BlockTable:
    pool: BlockPool
    rid: int = 0
    block_ids: list = field(default_factory=list)  # logical index -> physical id
    num_tokens: int = 0                            # how many slots are actually filled

    # ------------------------------------------------------------------ #
    # THE APPEND PATH — one decode step produces exactly one new token, so
    # this is called once per request per step. It is the only place blocks
    # are ever allocated.
    # ------------------------------------------------------------------ #
    def append_token(self):
        """Make room for ONE more token. Return True on success, False if the
        pool is exhausted (caller must then preempt or queue).

        Contract you must preserve:
          - on success, num_tokens has grown by exactly 1
          - every interior block stays 100% full (the append-only invariant)
          - on failure, nothing is mutated (no half-allocated state)
        """
        # 1. Check if the current block has enough space for one token
        # 2. if there is space, allocate it to the existing block, increase num_token by 1.
        # 3. if there is no space, ask for a new block from BlockPool
        # 4. if there is a block returns add it block_ids and increase num_token by 1.
        # 5. else block pool is exhausted, return False
        if self.num_tokens % self.pool.block_size > 0:
            self.num_tokens += 1
            return True
        if self.pool.num_free > 0:
            new_block_id = self.pool.allocate()
            self.block_ids.append(new_block_id)
            self.num_tokens += 1
            return True
        else:
            return False

    # ------------------------------------------------------------------ #
    # THE READ PATH — the page-table walk. Two integer ops + a lookup.
    # vLLM uses block_size = 16 (a power of two) so // and % compile down to
    # a shift and a mask; no division in the attention hot path.
    # ------------------------------------------------------------------ #
    def translate(self, pos):
        """Position `pos` in this request's logical sequence -> (physical_block, offset)."""
        if not 0 <= pos < self.num_tokens:
            raise IndexError(f"position {pos} outside filled range [0, {self.num_tokens})")
        logical = pos // self.pool.block_size
        offset = pos % self.pool.block_size
        return self.block_ids[logical], offset

    # ------------------------------------------------------------------ #
    @property
    def slots_held(self):
        """Token-slots this request is OCCUPYING (whole blocks, filled or not)."""
        return len(self.block_ids) * self.pool.block_size

    @property
    def wasted_slots(self):
        """Slots held but unfilled. Lives entirely in the LAST block, because
        every interior block is full — that is the append-only invariant paying
        out. This is what must stay <= block_size - 1."""
        return self.slots_held - self.num_tokens

    def release(self):
        """Request finished (evicted by the scheduler) — give the blocks back."""
        self.pool.free(self.block_ids)
        self.block_ids = []
        self.num_tokens = 0

    def render(self):
        return f"R{self.rid}: logical{list(range(len(self.block_ids)))} -> physical{self.block_ids}"


# ====================================================================== #
if __name__ == "__main__":
    BS = 4  # tiny block size so the traces are readable; vLLM uses 16

    # -- DEMO 1: growth. A new block is allocated ONLY when the tail fills. --
    print("=" * 68)
    print(f"DEMO 1 — one request growing, block_size={BS}")
    print("=" * 68)
    pool = BlockPool(num_blocks=8, block_size=BS)
    a = BlockTable(pool=pool, rid=0)
    print(f"{'tokens':>6} | {'blocks':>6} | {'held':>4} | {'wasted':>6} | table")
    print("-" * 68)
    for _ in range(10):
        assert a.append_token(), "pool should not be dry this early"
        print(f"{a.num_tokens:>6} | {len(a.block_ids):>6} | {a.slots_held:>4} | "
              f"{a.wasted_slots:>6} | {a.render()}")
    print(f"\nwaste never exceeded block_size-1 = {BS - 1}  ->  bounded BY CONSTRUCTION")

    # -- DEMO 2: scatter + the page-table walk. --
    print("\n" + "=" * 68)
    print("DEMO 2 — three requests growing interleaved (physical ids scatter)")
    print("=" * 68)
    pool = BlockPool(num_blocks=12, block_size=BS)
    tables = [BlockTable(pool=pool, rid=i) for i in range(3)]
    for step in range(6):                      # round-robin, like a real decode loop
        for t in tables:
            t.append_token()
    for t in tables:
        print(f"  {t.render()}   ({t.num_tokens} tokens, {t.wasted_slots} wasted)")
    print("\n  Physical ids are interleaved — no request owns a contiguous run,")
    print("  and attention does not care. Reading R0 position 5:")
    phys, off = tables[0].translate(5)
    print(f"    5 // {BS} = {5 // BS}  -> logical block {5 // BS}")
    print(f"    block_table[{5 // BS}] = {phys}  -> physical block {phys}")
    print(f"    5 %  {BS} = {off}  -> offset {off} inside it")

    # -- DEMO 3: the payoff, at realistic numbers. --
    print("\n" + "=" * 68)
    print("DEMO 3 — naive reservation vs paged, block_size=16, max_seq_len=2048")
    print("=" * 68)
    REAL_BS, MAX_SEQ = 16, 2048
    lengths = [37, 900, 12]
    pool = BlockPool(num_blocks=4096, block_size=REAL_BS)
    paged = []
    for i, n in enumerate(lengths):
        t = BlockTable(pool=pool, rid=i)
        for _ in range(n):
            t.append_token()
        paged.append(t)

    naive_held = MAX_SEQ * len(lengths)
    used = sum(lengths)
    paged_held = sum(t.slots_held for t in paged)
    print(f"{'':>18} | {'held':>7} | {'used':>5} | {'wasted':>7} | waste %")
    print("-" * 68)
    print(f"{'naive (reserve max)':>18} | {naive_held:>7} | {used:>5} | "
          f"{naive_held - used:>7} | {(naive_held - used) / naive_held * 100:>5.1f}%")
    print(f"{'paged':>18} | {paged_held:>7} | {used:>5} | "
          f"{paged_held - used:>7} | {(paged_held - used) / paged_held * 100:>5.1f}%")
    print(f"\n  per-request waste: {[t.wasted_slots for t in paged]}  "
          f"(all <= {REAL_BS - 1})")
    print(f"  request 1 has {lengths[1]} tokens; {lengths[1]} % {REAL_BS} = "
          f"{lengths[1] % REAL_BS} -> its tail block is partial")

    # -- DEMO 4: the pool runs dry. Plants the next build. --
    print("\n" + "=" * 68)
    print("DEMO 4 — pool exhaustion (the condition build (b) has to answer)")
    print("=" * 68)
    pool = BlockPool(num_blocks=2, block_size=BS)
    t = BlockTable(pool=pool, rid=0)
    n = 0
    while t.append_token():
        n += 1
    print(f"  pool: {pool.num_blocks} blocks x {BS} slots = {pool.num_blocks * BS} slots")
    print(f"  request grew to {n} tokens, then append_token() returned False")
    print(f"  pool free: {pool.num_free}, used: {pool.num_used}")
    print("\n  A real scheduler now has to CHOOSE: preempt-and-recompute (drop a")
    print("  victim's blocks, redo its prefill later) or swap its blocks to CPU.")
    print("  That is a POLICY question, and it is the next build.")
