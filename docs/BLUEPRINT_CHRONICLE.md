# Blueprint Solver Chronicle

Complete timeline of everything attempted, what worked, what failed, and why.
For bug-level detail, see `BLUEPRINT_BUGS.md`. This document is the narrative.

---

## Phase 1: Initial Build (pre-April 2026)

### What was built
- 6-player external-sampling MCCFR with Linear CFR discount and regret-based pruning
- OpenMP parallelism (192 cores on c7a.metal-48xl)
- Card abstraction: 169 lossless preflop buckets, 200 k-means postflop buckets per street
- Texture cache for flop bucket precomputation (saves 40 min per launch)
- S3 checkpoint/resume system
- Strategy export for the competition bot

### First runs — nothing converged
- Trash hands (32o, 63o) raising 74% from UTG
- Raw regret extraction showed ALL fold regrets negative for trash hands
- Root cause: board_hash in info set key inflated tree from ~100M to billions

---

## Phase 2: Core Bug Fixes (commits 7eca71d, 13e30db, e1f6ca5)

### Bug 1: board_hash in info set key
- **Symptom:** 27% miss rate, tree never plateaus
- **Fix:** Set board_hash = 0 (bucket abstracts the board)
- **Result:** Tree plateaued at 60-80M with 4 preflop sizes. Miss rate → 0%.
- **Status:** ✅ FIXED

### Bug 2: int32 regret overflow
- **Symptom:** Nonsensical strategies at 9.87B iterations (32o raising from UTG)
- **Fix:** BP_REGRET_CEILING = 310M, int64 intermediate arithmetic
- **Status:** ✅ FIXED

### Bug 3: Heap corruption from billions of small callocs
- **Symptom:** malloc(): corrupted top size crash at ~575M iterations
- **Fix:** Arena allocator for strategy_sum (same as regrets)
- **Status:** ✅ FIXED

### Bug 3b: eval_showdown_n division by zero
- **Symptom:** Extreme regret deltas from +inf return value when traverser folded
- **Fix:** Early return when !active[traverser]
- **Status:** ✅ FIXED

---

## Phase 3: Hash Table Sizing (April 4, 2026)

### Problem: 2B hash table fills at 75M iterations with 8 preflop sizes

Measured data:
| Config | Hash table | Filled at | Tree size | Miss rate | Speed |
|--------|-----------|-----------|-----------|-----------|-------|
| board_hash + 4 pre | 1.34B | 80M iters | infinite | 27% | 185K/s |
| no board_hash + 14 pre | 536M | 27M iters | 535M+ | ~20% | — |
| no board_hash + 8 pre | 1B | ~35M iters | 1B+ | — | — |
| no board_hash + 8 pre | 2B | ~75M iters | 2B+ | — | 76K/s |
| no board_hash + 4 pre | 8M test | 370K iters | 60-80M plateau | 0% | 360K/s |

### Investigation: Why does Pluribus have 665M while we have 2B+?

**Research findings (Pluribus supplementary materials p.12):**
- Pluribus uses "between 1 and 14" sizes per decision point — **tapered, not flat**
- Turn/river: 3 first-raise sizes, 2 subsequent
- 200 buckets postflop, 169 preflop — matches ours

**Analytical enumeration (tests/count_tiered.py):**
| Config | Preflop Nodes | × 169 = Preflop IS |
|--------|--------------|---------------------|
| 8 flat, max 4 raises | 43,806,293 | 7,403,263,517 |
| Tiered 8/3/2/1, max 4 | 2,283,138 | 385,850,322 |
| 4 flat, max 3 | 678,182 | 114,612,758 |

**Root cause:** 8 flat sizes at every preflop level × max 4 raises × 6 players = combinatorial explosion. The 4th raise level alone costs 38M extra nodes.

### Fix: Tiered preflop sizing (Pluribus-style)
- Open: 8 sizes [0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0]
- 3-bet: 3 sizes [0.7, 1.0, 2.5]
- 4-bet: 2 sizes [1.0, 4.0]
- 5-bet: all-in only [8.0]
- **19x reduction** in preflop tree (43.8M → 2.28M nodes)
- New API: `bp_set_preflop_tier(solver, level, sizes, num_sizes, max_raises)`
- **Status:** ✅ IMPLEMENTED

### int64 hash table support
- Changed table_size, num_entries, iteration counters to int64_t
- BPR4 checkpoint format (backward compatible with BPR2/BPR3)
- Fixed `BP_HASH_SIZE_3B` macro: was `((int)3000000000)` which overflows to -1.29B!
- Fixed bp_solve, all loop variables, TraversalState.iteration, bp_num_info_sets
- **Status:** ✅ FIXED

### Chosen config: 3B hash table on c7a.metal-48xl (376 GB)
- 180 GB metadata + ~30 GB arena = ~210 GB total
- Table never fills (1.67B entries at 2.6B iterations = 56% full)
- Speed: ~90K iter/s (cache-hostile due to 180 GB random access, but acceptable)
- **Status:** ✅ DEPLOYED

---

## Phase 4: Strategy Verification (April 4, 2026)

### Extract tool bug: wrong hash function
- `extract_all_preflop.c` used a hardcoded root hash — only found UTG (first to act)
- `extract_roots.c` written to find all 6 positions' "folds to me" root hashes
- **But used wrong hash_combine function:** `seed ^ (val * C1 + C2 + (seed << 12) + (seed >> 4))`
- Actual solver uses: `a ^= b + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2)`
- Spent ~2 hours debugging before finding the mismatch via debug DLL with fprintf
- **Lesson:** Always verify hash functions match between tools and solver
- **Status:** ✅ FIXED

### Verification at 20M iterations (200M table, 8 threads)
- All 6 positions: 169/169 buckets found
- Position gradient correct: UTG 87 fold → MP 83 → CO 68 → BTN 51 → SB 13
- AA raises 100% from all positions, 32o folds 100% from UTG
- Marginal hands noisy but directionally correct
- **Result:** ✅ SOLVER WORKS (at 20M iterations with small table)

### Verification at 2B iterations (3B table, 192 threads) — REGRESSION
- AA only raises 77% from UTG (was 100% at 20M)
- 32o shows 9.1/9.1/81.8 = uniform (1/11 each action)
- TT folds 100%, 99 folds 100% — should raise
- MP tighter than UTG (inverted position gradient)
- **Result:** ❌ WORSE than 20M run

### Stale checkpoint incident
- First tiered instance loaded OLD flat-8 checkpoint (536M entries, wrong tree structure)
- Had to move old checkpoints to `checkpoints_old_flat8/` on S3
- Later, the failed first run overwrote `regrets_latest.bin` with stale data
- Had to manually copy `regrets_2B.bin` → `regrets_latest.bin` to fix
- **Lesson:** Always verify checkpoint metadata (entries + iterations) matches expected config
- **Status:** ✅ RESOLVED

---

## Phase 5: Convergence Investigation (April 4-5, 2026)

### Raw regret dump reveals frozen regrets
At 2.0B iterations, raw regret dump showed:
- UTG 99: fold regret = +57,219 (tiny), all raise regrets -89M to -309M
- UTG TT: fold regret = +131K, all raise regrets -63M to -309M
- UTG AA: fold at regret floor (-310M), raise regrets +5M to +26M (correct but low)
- strategy_sum = NULL for all UTG root info sets

### Convergence trajectory (2.0B → 2.2B → 2.4B → 2.6B)

**What improved:**
- Position gradient corrected (MP now looser than UTG)
- 22's raise regret crossed zero (+4.9M at 2.2B, +10.5M at 2.6B) — escaped the trap
- AA/KK raise regrets growing steadily
- 32o fold regret strengthening

**What did NOT improve:**
- 99 fold regret: frozen at exactly +57,219 across ALL 4 checkpoints
- TT fold regret: frozen at +2.27M since 2.2B
- TT best raise regret: improved from -63M to -6.4M at 2.2B, then DEGRADED to -17.9M at 2.6B
- strategy_sum still NULL for UTG roots

### Diagnosis: Preflop Regret Lock-In

When a hand converges to 100% fold:
1. `node_value = fold_value` (100% weight on fold)
2. `regret[fold] += fold_value - fold_value = 0` ← **delta is exactly zero**
3. Fold regret freezes permanently
4. Raise regrets keep sinking (raise_value < fold_value because postflop hasn't converged)
5. Regret-based pruning kills actions below -300M — their subtrees stop being explored
6. Self-reinforcing: locked actions can't recover

**Affected hands:** 99, TT, A7s, A6s, A5s, K9s, 87s, 98s, and most suited connectors from UTG. These should raise in equilibrium but converged to fold too early.

**Unaffected:** 22 (escaped because raise regret crossed zero fast enough), AA/KK (never converged to pure fold), 32o (correctly folds, frozen but correct).

### Status: ❌ UNSOLVED — needs fix in next session

See `DEBUG_SESSION_2.md` for the full investigation prompt. Options include exempting preflop from pruning, raising the pruning threshold, or epsilon-greedy exploration. The diagnosis should be verified before implementing any fix (it could be a different root cause).

---

## Current State (end of April 4-5 session)

### What works
- Tiered preflop sizing (8/3/2/1) — 19x tree reduction
- int64 support throughout — handles 3B table and 46.5B iterations
- All 6 positions populated with correct position gradient
- Premiums (AA, KK, AKo) converge correctly
- Pure trash (32o, 72o) converge correctly
- Checkpoint/resume system works with BPR4 format
- Extract tools with correct hash function

### What's broken
- Preflop regret lock-in for marginal hands (99, TT, suited connectors)
- strategy_sum NULL for UTG root info sets (average strategy not computed)
- Instance self-terminated after first checkpoint (S3 upload may have failed with set -e)

### Resources on S3 (`s3://poker-blueprint-unified/`)
- `checkpoints/regrets_2B.bin` — 2.0B iterations, 66 GB, tiered config
- `checkpoints/regrets_latest.bin` — 2.6B iterations, 66 GB, tiered config
- `texture_cache.bin` — precomputed flop buckets (saves 40 min)
- `checkpoints_old_flat8/` — old flat-8-sizes checkpoints (archived, don't use)

### No running instances
All EC2 instances terminated. Next session should fix the lock-in bug before deploying.
