# Debug Prompt — Heap Corruption Crash

Paste this into a new Claude Code conversation. Working directory: `/Users/victor/Documents/Dev/poker-solver/`

Private repo: `ZHANGV25/poker-solver-dev`

---

## The Problem

The 6-player MCCFR blueprint solver crashes with `malloc(): corrupted top size` after ~500-700M iterations. This has happened on TWO different runs:

1. **Run 1:** c5.metal (96 cores), crashed at 702M iterations. Old code without regret ceiling fix.
2. **Run 2:** c6a.metal (192 cores), crashed at 575M iterations. NEW code WITH regret ceiling fix, int64 overflow protection, side pot fix, folded traverser guard.

The crash happens earlier with more cores (575M vs 702M), strongly suggesting a **thread race condition** that corrupts the heap. With 192 threads, corruption accumulates faster.

`malloc(): corrupted top size` means heap metadata was overwritten — some code is writing past the end of an allocated buffer, corrupting the header of an adjacent heap block.

**The regret overflow bug was also found and fixed (see below), but it is NOT the cause of this crash.** The crash persists with the fix applied.

Your job: Find the heap corruption bug. This is the ONLY thing blocking the solver from completing an 83-hour run.

```
UTG root strategies at 9.87B iterations:
  AA  (bucket 0):   raise 69% r0.5x / 31% r3x          — OK
  AKs (bucket 1):   call 100% (pure limp)               — Unusual
  KK  (bucket 25):  raise mix 32/21/48                   — OK
  QQ  (bucket 48):  raise 100% r2x                       — OK
  J3s (bucket 84):  raise 98% (38% r0.5x, 59% r3x)     — WRONG, should fold
  88  (bucket 120): raise 100% r2x                       — Aggressive but maybe OK
  63o (bucket 150): raise 85% r3x                        — WRONG, should fold
  33  (bucket 165): raise 100% r3x                       — Too aggressive UTG
  32s (bucket 166): raise 78% r3x, fold 22%              — WRONG, should fold >90%
  32o (bucket 167): raise 74% (26% r0.5x, 74% r3x)     — WRONG, should fold >95%
  22  (bucket 168): raise 100% r0.5x                     — WRONG, should fold often UTG
```

Trash hands (32o, 63o, J3s) are raising aggressively from UTG instead of folding. This might be:
1. A bug in the solver causing incorrect regret accumulation
2. Normal early-convergence behavior that will self-correct with more iterations
3. An issue with the action encoding, bucket mapping, or traversal logic

Your job: **investigate the C solver code** to determine if there's a bug, or if this is expected behavior at 10% of training.

---

## Current State

The solver instance `i-0f0b05bee28b0b088` (c6a.metal, 192 cores, 376 GB RAM) has crashed and the process is dead. You can SSH in to read logs and source code. The instance is still running (not terminated).

```bash
# Generate SSH key (if /tmp/ec2-temp-key doesn't exist)
rm -f /tmp/ec2-temp-key /tmp/ec2-temp-key.pub
ssh-keygen -t rsa -b 2048 -f /tmp/ec2-temp-key -N "" -q

# Push key and SSH in
aws ec2-instance-connect send-ssh-public-key \
    --instance-id i-0f0b05bee28b0b088 \
    --instance-os-user ec2-user \
    --ssh-public-key file:///tmp/ec2-temp-key.pub \
    --region us-east-1

ssh -i /tmp/ec2-temp-key -o StrictHostKeyChecking=no ec2-user@18.208.173.178
```

Key paths on the instance:
- Solver log: `/var/log/blueprint-unified.log`
- Source code: `/opt/poker-solver/src/mccfr_blueprint.c`
- Worker: `/opt/poker-solver/precompute/blueprint_worker_unified.py`
- Core dump may exist: check `ls /tmp/core*` or `coredumpctl list`

The instance has 376 GB RAM and the solver is dead, so you CAN compile and run test programs freely.

---

## Codebase

All source is at `/Users/victor/Documents/Dev/poker-solver/` (local) and `/opt/poker-solver/` (on instance).

### Key files to investigate:

**`src/mccfr_blueprint.c`** — The entire blueprint solver. Key functions:
- `traverse()` (~line 611): Main MCCFR traversal. Recursively walks the game tree for the traverser, sampling for opponents.
- `generate_actions()` (~line 447): Generates legal actions at each node. For UTG preflop: fold(0), call(1), raise 0.5x(2), raise 1x(3), raise 2x(4), raise 3x(5).
- `eval_showdown_n()` (~line 568): N-player showdown evaluation with side pots.
- `bp_solve()` (~line 1429): Main solve loop. Handles discount, pruning, snapshots, batching.
- `info_table_find_or_create()` (~line 275): Lock-free hash table lookup/insert with dedup.
- `regret_match()`: Converts cumulative regrets to strategy via positive-regret normalization.
- `canonicalize_board()` (~line 158): Suit-isomorphic board canonicalization for hash keys.
- `apply_discount()`: Multiplies all regrets by d = t/(t+1) for Linear CFR.

**`src/card_abstraction.c`** — Bucketing:
- `ca_preflop_classes()` (~line 379): Maps 1326 hands to 169 lossless classes. 0=AA, 1=AKs, 2=AKo, ..., 25=KK, ..., 48=QQ, ..., 165=33, 166=32s, 167=32o, 168=22.

**`src/mccfr_blueprint.h`** — Data structures:
- `BPInfoKey`: (player, street, bucket, board_hash, action_hash)
- `BPConfig`: int64 threshold fields (discount_stop, prune_start, snapshot_start, etc.)
- `BPSolver`: Full solver state

**`precompute/blueprint_worker_unified.py`** — Python orchestration:
- Calls bp_solve in sub-chunks of 2,147,483,647 (INT32_MAX)
- Handles checkpointing every 10B iterations
- Computes thresholds from Pluribus core-hours (93B total, 12400 core-hours)

---

## What to Investigate

The crash is `malloc(): corrupted top size` — heap metadata corruption. Something is writing past the end of an allocated buffer. Focus on these areas:

### 1. Arena allocator thread safety

`arena_alloc()` (~line 87) uses atomic fetch-add on `chunk->used` for the fast path, and a spinlock for new chunk allocation. With 192 threads:
- Could two threads get overlapping regions from the same chunk? (fetch-add should prevent this, but verify)
- Could a thread read a stale `g_arena.head` pointer while another thread is linking a new chunk?
- The `g_arena.head = c` assignment at line 136 — is this a release-store visible to all threads?

### 2. Hash table dedup race

`info_table_find_or_create()` (~line 275) has a dedup scan after CAS. The dedup scan checks `occupied[idx] == 1` — but what if a concurrent thread is still at state 2 (initializing)? The scan skips state-2 slots, potentially missing a duplicate. Two threads could insert the same key at different slots with different `num_actions`, and later a third thread writes `na` regrets to a buffer allocated for fewer.

Specifically: can `generate_actions()` return different action counts for the same info set key on different visits? If the game state is identical (same key), the actions should be identical. But verify this — any floating-point non-determinism in bet size calculation could change whether a raise size equals the all-in amount, changing the action count.

### 3. ensure_strategy_sum race

`ensure_strategy_sum()` (~line 323) uses CAS on `is->strategy_sum`. Two threads could both see NULL, both calloc, one wins CAS, loser frees. But what if a third thread is READING `is->strategy_sum` (in the regret update or snapshot) while it's being swapped? Could it read a freed pointer?

### 4. The regret update itself

Lines ~1054-1063 (with the new ceiling fix):
```c
int64_t tmp = (int64_t)is->regrets[a] + (int64_t)delta;
if (tmp < BP_REGRET_FLOOR) is->regrets[a] = BP_REGRET_FLOOR;
else if (tmp > BP_REGRET_CEILING) is->regrets[a] = BP_REGRET_CEILING;
else is->regrets[a] = (int)tmp;
```

This is Hogwild (no lock). Two threads writing to the same `is->regrets[a]` simultaneously could produce a torn write on 32-bit int — but on x86-64, aligned 4-byte writes are atomic. However, the READ of `is->regrets[a]` and the WRITE are not atomic together. Thread A reads regret=100, Thread B reads regret=100, both add delta, both write back — losing one update. This is expected in Hogwild and doesn't cause corruption, just noise. But verify nothing worse can happen.

### 5. Stack overflow from deep recursion

`traverse()` is deeply recursive (~50+ levels for a full preflop-through-river traversal). Each frame is ~6.5 KB (TraversalState struct). With `OMP_STACKSIZE=64m`, the stack can hold ~10,000 frames — should be enough. But with 192 threads, total stack = 192 × 64 MB = 12 GB. Verify the stack isn't overflowing into heap.

### 6. The canonicalize_board function

`canonicalize_board()` (~line 158) writes to a stack-allocated `int canon_out[5]`. Called from the traverse() regret update (~line 859). If `num_board > 5` somehow, it would write past the buffer. Verify num_board is always 0-5.

### 7. Strategy snapshot accumulation

`accumulate_snapshot()` iterates all info sets and calls `ensure_strategy_sum()` + writes to `strategy_sum`. If this runs concurrently with traversal threads (it runs in an `omp single` block between batches), are the barriers correct? Could a traversal thread read/write `strategy_sum` while the snapshot thread is allocating it?

## Recommended Approach

1. **Check if there's a core dump** on the instance (`coredumpctl list`, or `ls /tmp/core*`). If so, load it with gdb to find the exact crash location.

2. **Compile with AddressSanitizer** locally: `gcc -O1 -fsanitize=address -fopenmp -o test_asan src/mccfr_blueprint.c src/card_abstraction.c -I src -lm`. Run a small solve (1000 iterations, 4 threads). ASan detects heap overflows, use-after-free, and buffer overruns.

3. **Compile with ThreadSanitizer**: `gcc -O1 -fsanitize=thread -fopenmp -o test_tsan ...`. Run same small solve. TSan detects data races.

4. If sanitizers don't catch it locally (race conditions are timing-dependent), add manual bounds checks: before every `is->regrets[a]` access, assert `a < is->num_actions`. Before every `arena_alloc`, assert the returned pointer + size doesn't overlap the next allocation.

---

## Fixes Already Applied (in local code, verify on instance)

These fixes are in the local codebase and were uploaded to S3 before the crashed run:

1. **Regret ceiling** (`BP_REGRET_CEILING = 310000000`): Caps positive regrets at 310M to prevent int32 overflow for frequently-visited preflop info sets.
2. **int64 intermediate**: `int64_t tmp = (int64_t)is->regrets[a] + (int64_t)delta` prevents overflow during addition.
3. **Delta clamping**: Raw float delta clamped to ±2B before int cast.
4. **Folded traverser guard**: `eval_showdown_n` returns `-(invested)` immediately if `!active[traverser]`.
5. **Side pot handling**: Proper multi-level pot splitting for unequal all-ins.
6. **num_batches int64**: Prevents overflow in batch count calculation.
7. **Sub-chunking**: bp_solve called with INT32_MAX chunks, explicit ctypes.c_int guard.

## Possible outcomes

1. **Bug found in code**: Fix it, write a test, verify locally with ASan/TSan, restart.
2. **Bug found in compiler optimization**: Add `-fwrapv` or reduce to `-O2` and test.
3. **Need more data**: Recompile with ASan on the EC2 instance and run until crash — ASan will pinpoint the exact overflow location.
