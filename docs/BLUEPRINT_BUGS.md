# Blueprint Solver Bug Report

Three critical bugs were found and fixed in the 6-player MCCFR blueprint solver.
All three must be fixed together — any one alone prevents convergence.

## Bug 1: board_hash in info set key (root cause of non-convergence)

**Commit:** `7eca71d`

**The problem:** The info set key was `(player, street, bucket, board_hash, action_hash)`.
Including `board_hash` alongside `bucket` defeats the purpose of card abstraction.
Every canonical board creates separate info sets, inflating the tree from ~60-100M
(Pluribus has 665M) to billions.

The 1.34B-slot hash table filled at **80M iterations** — just 0.17% into the 46.5B
target. After that, 27% of info set lookups failed and returned 0 instead of the
true game value. This systematically biased raise values positive for trash hands
(true value is negative, truncated to 0), preventing fold from accumulating
positive regret.

**Evidence:**
- Raw regret extraction at 2B iters showed ALL trash hands have negative fold
  regrets (63o: -497K, 33: -1.8M, 32o: -150K, 22: -1.2M)
- Miss rate measurement: 27.5% of lookups return 0 with a full table
- Tree size test: reducing postflop bet sizes from 3→1 made zero difference —
  the tree is dominated by `200 buckets × 175K canonical boards`, not bet sizes
- Removing board_hash: tree plateaus at ~50-80M info sets (0% miss rate in 512M table)

**How Pluribus does it:** Info set = `(player, street, bucket, action_sequence)`.
No board in the key. The bucket already abstracts the board's strategic impact
via EHS/k-means clustering. 665M total action sequences, 414M encountered.

**The fix:** `key.board_hash = 0` in traverse() and bp_get_strategy().

## Bug 2: int32 regret overflow (preflop strategy corruption)

**Commit:** `13e30db`

**The problem:** Preflop info sets are visited ~1.1B times in the undiscounted
phase (3.25B–9.87B iterations). Regret deltas of 100–10,000 per visit cause
cumulative regrets to exceed INT_MAX (2.1B), causing undefined behavior
(signed int overflow compiled with -O2, no -fwrapv).

**Evidence:** Observed at 9.87B iterations on the first run (c5.metal, 96 cores).
Trash hands showed nonsensical strategies (32o raising 74% from UTG).

**The fix:**
- Added `BP_REGRET_CEILING = 310M` (symmetric with existing -310M floor)
- int64 intermediate arithmetic: `int64_t tmp = (int64_t)regrets[a] + delta`
- Delta clamped to ±2B before int cast
- Compile with -O2 -fno-strict-aliasing (was -O3)

## Bug 3: Heap corruption from billions of small callocs

**Commit:** `e1f6ca5`

**The problem:** `accumulate_snapshot()` allocated `strategy_sum` via individual
`calloc()` calls for each info set — up to 1.34B calls of 24-32 bytes each.
This created massive heap fragmentation that interacted poorly with glibc's
thread-local arenas under 192-thread concurrency, leading to
`malloc(): corrupted top size`.

**Evidence:** Crash at ~575M iterations on c6a.metal (192 cores), earlier at
~702M on c5.metal (96 cores). Earlier with more cores = race condition signature.
Hash table at 100% capacity at crash time.

**The fix:** Moved strategy_sum from heap `calloc`/`free` to the arena allocator
(same as regrets). Eliminates all heap operations during solving.

**Secondary fixes in same commit:**
- `eval_showdown_n`: Added early return when traverser has folded
  (`!active[traverser]`). Previously returned +inf from division by zero
  (`pot / n_tied` where n_tied=0), generating extreme regret deltas.
- `na` guard: Clamp caller's action count to `is->num_actions` to prevent
  buffer overflow from hash collisions.

## Additional changes (Pluribus alignment)

**Commit:** `7eca71d`

- **14 preflop raise sizes** (was 4): `[0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2,
  1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0]`. Matches Pluribus "up to 14" for
  preflop where real-time search is not used.
- **Postflop subsequent raises**: Reduced to {1x pot, all-in} (was 3 sizes for
  all raises). First raise keeps {0.5x, 1x, all-in}. Matches Pluribus
  turn/river specification.
- **Hash table**: 1.34B → 512M slots. Tree is ~60-100M without board_hash.
- **BP_MAX_ACTIONS**: 8 → 16 (fold + call + 14 sizes).
- **ARENA_BLOCK_SIZE**: 32 → 64 bytes (fits 16 int regrets).
- **Texture cache**: Auto-loaded from `/tmp/texture_cache.bin` at init,
  downloaded from S3 at deploy time (saves 40 min precomputation).

## Diagnostic tools added

| Tool | Purpose |
|------|---------|
| `tests/extract_regrets.c` | Dump raw regret values for specific preflop classes from checkpoint |
| `tests/test_miss_counter.c` | Measure info set lookup miss rate after table fills |
| `tests/test_tree_size.c` | Measure game tree size under different bet/bucket configs |
| `tests/test_6p_preflop.c` | 6-player preflop-only MCCFR convergence test |
| `tests/test_preflop_convergence.c` | 2-player preflop MCCFR convergence test |
| `tests/test_asan_harness.c` | AddressSanitizer test harness |

## Investigation methodology

1. **Regret extraction** (not just strategies) revealed the solver "thought"
   raising was profitable — fold regrets were negative for ALL trash hands.
2. **32s vs 32o comparison**: Nearly identical hands with opposite strategies
   (fold 100% vs raise 100%). 2-player local test proved algorithm doesn't
   diverge them → noise, not a structural bug.
3. **Fold value analysis**: fold returns 0 for UTG (no investment). Negative
   fold regret means `avg(node_value) > 0` — the solver thinks 32o has
   positive EV from UTG. This is only possible if raise values are inflated.
4. **Miss rate measurement**: Direct instrumentation showed 27.5% of lookups
   returning 0. Each miss truncates a subtree that should return a negative
   value for trash hands.
5. **Tree size sweep**: Tested 1/2/3 postflop bet sizes — all identical tree
   growth. Tested removing board_hash — tree plateaus at ~50-80M. This proved
   board_hash was the multiplicative factor, not bet sizes.

## Bug 4: Hash table fills at 75M iterations — tree 19x larger than Pluribus

**Problem:** With 8 flat preflop raise sizes and max 4 raises, the game tree
produces ~7.4 BILLION preflop info sets (43.8M decision nodes × 169 buckets).
Pluribus has 665M action sequences total across all streets. Even a 2B hash
table fills at ~75M iterations (0.16% of the 46.5B target), causing:

1. New info set lookups return 0 — biases game values, prevents convergence
2. Only UTG root info sets exist — SB/BB/MP/CO/BTN entries were never created
   because the table filled before sampling reached their decision points
3. Cache thrashing — 250 GB table at 100% occupancy, 76K iter/s instead of 360K

**Root cause:** Flat sizing. Every preflop decision point (open, 3-bet, 4-bet,
5-bet) gets 8 raise options. With 6 players and max 4 raises, the combinatorial
explosion is massive. Pluribus uses "between 1 and 14" sizes per decision point,
**tapered by hand** — many for the open raise, progressively fewer for 3-bet/4-bet.

**Evidence — analytical enumeration (tests/count_tiered.py):**

| Config                  | Preflop Nodes | × 169 = Preflop IS | Lines to Flop |
|-------------------------|---------------|---------------------|---------------|
| 8 flat, max 4 (broken)  | 43,806,293    | 7,403,263,517       | 43,806,262    |
| Tiered 8/3/2/1, max 4   | 2,283,138     | 385,850,322         | 2,283,107     |
| 4 flat, max 3            | 678,182       | 114,612,758         | 678,151       |

The 4th raise level is the costliest — it reopens action for all 6 players with
8 choices each. Dropping from max 4 to max 3 raises alone gives an 8x reduction.

**The fix — tiered preflop sizing (Pluribus-style):**

```
Open raise (level 0): 8 sizes [0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0]
3-bet (level 1):      3 sizes [0.7, 1.0, 2.5]
4-bet (level 2):      2 sizes [1.0, 4.0]
5-bet (level 3):      1 size  [8.0] (essentially all-in)
Max raises: 4
```

This produces 2.28M preflop decision nodes (386M preflop IS) — a 19x reduction
from flat sizing while keeping all 8 open-raise sizes where strategic granularity
matters most. The open raise is the highest-value decision in 6-max poker.

**Implementation:**
- `bp_set_preflop_tier(solver, level, sizes, num_sizes, max_raises)` — new API
  to set per-raise-level bet size arrays
- `traverse()` selects the right size array based on `num_raises`
- `BPSolver` stores `preflop_tiered_sizes[4][BP_MAX_ACTIONS]` and
  `num_preflop_tiers` — falls back to flat `preflop_bet_sizes` when 0

**Deployed with 3B hash table** (c7a.metal-48xl, 376 GB RAM):
- 180 GB metadata, ~30 GB arena at estimated ~600M occupied entries
- 3B chosen for safety margin — never fills with tiered tree
- Tradeoff: larger table is cache-hostile (~87K iter/s vs 360K with cache-fitting
  table), but speed increases to 200-350K iter/s once pruning activates at ~790M
  iterations (95% of subtrees skipped)

## Bug 5: int32 overflow throughout solver — cannot run 46.5B iterations

**Problem:** Multiple `int` (32-bit) variables overflow at >2.1B:

| Variable | Location | Impact |
|----------|----------|--------|
| `bp_solve(int max_iterations)` | API | Can't request >2.1B iterations |
| `iter`, `batch_start`, `batch_end` | solve loop | Overflow during iteration |
| `TraversalState.iteration` | traversal state | Corrupts strategy accumulation timing |
| `BPInfoTable.table_size` | hash table | Can't create 3B table |
| `BP_HASH_SIZE_3B ((int)3B)` | constant | Wraps to -1.29B (silent!) |
| `bp_num_info_sets` return | API | Truncates count >2.1B |
| `written`/`loaded` in save/load | checkpoint I/O | Wrong counts |
| `BPConfig.hash_table_size` | config struct | Truncates 3B to negative |

**The fix:** Widened all iteration counters, hash table indices, and entry counts
to `int64_t`. Updated save format to BPR4 (int64 header fields), backward
compatible with BPR2/BPR3. Updated Python ctypes bindings to match.

## Diagnostic tools added (this session)

| Tool | Purpose |
|------|---------|
| `tests/count_preflop.py` | Enumerate exact preflop decision nodes for any flat config |
| `tests/count_tiered.py` | Enumerate preflop nodes for tiered configs (Pluribus-style) |
| `tests/enumerate_tree.py` | Full tree enumeration (preflop + postflop) |
| `tests/test_tiered.py` | Smoke test: verify tiered sizing works end-to-end |
| `tests/test_deploy_ready.py` | Pre-deploy validation: creation rate, save/load, int64, tiered vs flat |

## Bug 6: strategy_sum not accumulated for SB, UTG, CO (modular arithmetic aliasing)

**Problem:** The strategy_sum accumulation for preflop (round 1) only triggered when
`iteration % strategy_interval == 0` (strategy_interval = 10,000). Since the
traverser is `(iteration - 1) % 6`, and `gcd(6, 10000) = 2`, only players
1 (BB), 3 (MP), and 5 (BTN) were ever the traverser at multiples of 10,000.
Players 0 (SB), 2 (UTG), and 4 (CO) were permanently excluded — their
strategy_sum remained NULL forever.

**Evidence:** `dump_raw_regrets` showed `Strategy sum: NULL` for all UTG root
info sets across 4 checkpoints (2.0B–2.6B iterations). MP and BTN entries had
populated strategy_sum values.

**The fix:** Remove the interval check entirely. Accumulate strategy_sum on
every traverser visit for preflop:

```c
// Before (buggy):
if (street == 0 && (ts->iteration % s->config.strategy_interval) == 0) {
// After (fixed):
if (street == 0) {
```

This is cheap (preflop info sets are tiny) and ensures all 6 players accumulate.

## Bug 7: Dominant-action feedback loop (call trap / fold trap)

**Problem:** In external-sampling MCCFR, non-traversers sample actions according
to the current regret-matched strategy. When one action dominates (e.g., 80% call),
two self-reinforcing effects trap the solver:

1. **Frozen dominant regret:** When strategy ≈ [0%, 80%, 1%, 1%, ...] (call dominant),
   `node_value ≈ call_value`, so `regret[call] += call_value - call_value ≈ 0`.
   Call regret freezes at its current positive value — identical to the fold lock-in
   mechanism, but for call instead of fold.

2. **Stale opponent strategies:** During non-traverser iterations, the dominant player
   samples call 80% of the time. With 8 raise sizes each getting ~1%, opponent nodes
   at "player raised size X" get ~80x less training data than "player called" nodes.
   When the traverser explores raising, opponents respond with undertrained strategies,
   producing unreliable raise values that prevent raise regrets from growing.

This affects early positions (UTG, MP) severely because they face 5 opponents —
more opponent nodes to train, each getting less data. Late positions (BTN, SB) with
1-3 opponents converge correctly.

**Evidence (2.5B iterations, strategy_sum averages):**

| Position | Opponents | TT avg raise | 99 avg raise | Status |
|----------|-----------|-------------|-------------|--------|
| SB       | 1         | 94.7%       | 98.7%       | ✓ Works |
| BTN      | 2-3       | 84.5%       | 91.4%       | ✓ Works |
| CO       | 3-4       | 38.9%       | 58.4%       | Borderline |
| UTG      | 5         | 39.6%       | 38.5%       | ✗ Broken |
| MP       | 4-5       | 26.4%       | 23.6%       | ✗ Broken |

Additional failure modes at UTG:
- **Call trap:** TT raise declined monotonically: 63% → 41% → 34% → 12% → 38% avg
- **Fold locked:** 44/33/22 at 100% fold across all checkpoints (same as original bug)
- **Uniform stuck:** AKo, 88 show 9.1/9.1/81.8 (all regrets ≤ 0) after 2B+ iterations

**Root cause:** The combination of (a) many competing raise sizes fragmenting the
positive raise signal, and (b) external sampling's non-traverser using the current
strategy, which starves minority actions of opponent training data.

**The fix — Average Strategy Sampling (Lanctot et al., NIPS 2012):**

For preflop non-traverser sampling, use the accumulated average strategy
(strategy_sum) instead of the current regret-matched strategy. This preserves
historical action frequencies even when the current strategy concentrates on
one action, ensuring opponent nodes at all action subtrees receive adequate
training data.

```c
// Non-traverser sampling in traverse():
float *sample_strat = strategy;  // default: current regret-matched
float avg_strat[BP_MAX_ACTIONS];
if (street == 0 && is->strategy_sum) {
    float sum = 0;
    for (int a = 0; a < na; a++) sum += is->strategy_sum[a];
    if (sum > 0) {
        for (int a = 0; a < na; a++) avg_strat[a] = is->strategy_sum[a] / sum;
        sample_strat = avg_strat;
    }
}
int sampled = sample_action(sample_strat, na, ts->rng);
```

**Why not exempt preflop from pruning?** This was tried first. It broke the fold
lock-in (99/TT started raising) but revealed the call trap underneath — same
frozen-dominant-action mechanism, one level deeper. Pruning exemption is a
symptom fix; AS addresses the root cause. Pruning was reverted to match Pluribus.

**Why not reduce raise sizes?** Helps (concentrates signal), but doesn't fix the
feedback loop. Even with 2 raise sizes, call can starve raise nodes of opponent
training data. AS is orthogonal and can combine with fewer sizes if needed.

## Bug 8: Spurious 10x regret scaling factor

**Problem:** The regret delta was multiplied by 10.0f before casting to int:
```c
float raw_delta = (action_values[a] - node_value) * 10.0f;
```

This was added in commit `de36555` to "prevent float-to-int truncation" when chip
values were small. But with $50/$100 blinds and $10,000 stacks, typical deltas are
in the hundreds to thousands — truncation loses at most 1 chip, which is negligible.

The 10x multiplier causes:
1. Regrets hit ±310M ceiling/floor **10x faster**, saturating and losing information
2. Actions reach the -300M pruning threshold 10x sooner, getting pruned before
   proving their value
3. External-sampling noise amplified 10x, making all value estimates less reliable

Pluribus stores "regrets as 4-byte integers" with no scaling mentioned.

**The fix:** Remove the scaling factor:
```c
// Before:
float raw_delta = (action_values[a] - node_value) * 10.0f;
// After:
float raw_delta = action_values[a] - node_value;
```

## Diagnostic tools added (this session)

| Tool | Purpose |
|------|---------|
| `tests/dump_raw_regrets.c` | Dump raw regret + strategy_sum for specific hands (fixed bucket indices) |
| `tests/extract_roots.c` | Extract root strategies for all 6 positions, now shows strategy_sum average alongside regret-matched |
