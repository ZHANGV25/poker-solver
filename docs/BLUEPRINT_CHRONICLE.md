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

### Status: ✅ ROOT CAUSE IDENTIFIED — fixed in Phase 6

---

## Phase 6: Fixing the Lock-In — Attempt 1: Pruning Exemption (April 5, 2026)

### Hypothesis: Pruning prevents locked hands from recovering
The debug session diagnosed fold lock-in as: pure fold → fold regret frozen → raise
regrets sink below -300M → pruning kills them permanently. Fix: exempt preflop from
pruning.

### Also found: strategy_sum aliasing bug (Bug 6)
`iteration % 10000 == 0` combined with `traverser = (iter-1) % 6` created a phase
lock — only players 1,3,5 accumulated strategy_sum. Players 0 (SB), 2 (UTG), 4 (CO)
were permanently excluded. Fix: remove interval check, accumulate every visit.

### Local verification (2M iterations, 10M hash table, 1 thread)
- UTG 99: **100% RAISE** (was 100% FOLD). Lock-in broken.
- UTG TT: **100% RAISE**. All pocket pairs raising.
- strategy_sum populated for all players.
- 127/169 hands raising from UTG — too loose (tiny hash table, garbage postflop).
- **Result:** Pruning fix works locally. Deployed to EC2 for proper verification.

### EC2 run: Fresh start, 5B target, 500M checkpoints (c7a.metal-48xl)
Instance `i-02b8082868fdbe0c6`, ~90K iter/s, checkpoints at 500M/1B/1.5B/2B/2.5B.

**Convergence trajectory (UTG pocket pairs, regret-matched strategy):**

| Hand | 500M | 1B | 1.5B | 2B | Diagnosis |
|------|------|-----|------|-----|-----------|
| AA | 11%R | 59%R | 87%R | 83%R | ✓ Converging |
| KK | 1%R | 66%R | 75%R | 81%R | ✓ Converging |
| TT | 63%R | 41%R | 34%R | 12%R | ✗ Declining → call trap |
| 99 | 73%R | 40%R | 20%R | 0%R | ✗ 100% call at 2B |
| 88 | uniform | uniform | uniform | uniform | ✗ No convergence |
| 44-22 | fold | fold | fold | fold | ✗ Still locked |

**The pruning fix solved the fold lock-in but revealed a deeper problem:**
TT/99 moved from fold → call, not fold → raise. The call trap is the SAME
frozen-dominant-action mechanism, one level deeper: call regret freezes when
call dominates, raise regrets can't grow because opponent strategies at "player
raised" nodes are undertrained.

### Cross-position analysis confirms structural issue

Position comparison at 2.5B (strategy_sum averages, not noisy snapshot):

| Position | Opponents | TT avg raise | 99 avg raise |
|----------|-----------|-------------|-------------|
| SB | 1 | 94.7% | 98.7% |
| BTN | 2-3 | 84.5% | 91.4% |
| CO | 3-4 | 38.9% | 58.4% |
| UTG | 5 | 39.6% | 38.5% |
| MP | 4-5 | 26.4% | 23.6% |

BTN: 15/15 pocket pairs raise. UTG: 5/15 raise, 4 call-trapped, 4 fold-locked,
2 uniform-stuck. Perfectly correlated with opponent count.

### Root cause: External Sampling feedback loop

In ES-MCCFR, non-traversers sample from the CURRENT regret-matched strategy.
When call dominates (80%), raise subtrees get ~1% sampling each → opponents at
those nodes get 80x less training data → raise values are unreliable → raise
regrets stay negative → call stays dominant. Self-reinforcing loop.

With 8 open-raise sizes, each individual raise competes against a single
concentrated call action. Even if total raise EV > call EV, no single raise
size accumulates enough regret to overtake call.

### Pruning fix reverted
Exempting preflop from pruning deviates from Pluribus without fixing the root
cause. Reverted to standard Pluribus pruning (river + fold exemptions only).

---

## Phase 7: Fixing the Lock-In — Attempt 2: Average Strategy Sampling (April 5, 2026)

### Research: Lanctot et al., NIPS 2012
"Efficient Monte Carlo CFR in Games with Many Player Actions" — a variant of
MCCFR where non-traversers sample from the accumulated **average strategy**
instead of the current regret-matched strategy. Proven to converge faster in
games with many actions. 54% improvement over ES in no-limit hold'em.

### Why AS fixes the feedback loop
- Current strategy: 80% call → raise nodes get 1% each → stale opponents
- Average strategy: historical mix (e.g., 40% call, 60% raise spread) → raise
  nodes get meaningful sampling → opponents train properly → raise values improve
  → raise regrets can grow → convergence

We already maintain strategy_sum for preflop (Bug 6 fix). We just need to
USE it for non-traverser sampling.

### Implementation
One change in the non-traverser branch of `traverse()`:
```c
if (street == 0 && is->strategy_sum) {
    // normalize strategy_sum → avg_strat
    sample_strat = avg_strat;
}
```

Strategy_sum fix (Bug 6) retained — all players now accumulate correctly.
Pruning reverted to Pluribus-standard.

### Pluribus alignment check
- Pluribus paper does NOT mention AS explicitly
- But Pluribus hand-selected 1-14 raise sizes per decision point ("based on what
  earlier versions used with significant positive probability") — they avoided the
  fragmentation problem by design
- Pluribus opens to ~2.0-2.25 BB; our 8 sizes include irrelevant options like
  0.4x pot (min-raise) and 8.0x pot (overbet shove)
- AS is the algorithmic fix; reducing open sizes to 2-3 is the abstraction fix.
  Both address the same problem from different angles.

### Also found: 10x regret scaling factor (Bug 8)
Git blame traced `raw_delta * 10.0f` to commit `de36555` ("prevent float-to-int
truncation"). With $50/$100 blinds and $10K stacks, deltas are naturally in the
hundreds — truncation is negligible. The 10x causes regrets to hit the ±310M
ceiling/floor and -300M pruning threshold 10x faster, amplifies noise, and
accelerates premature pruning. Removed: `raw_delta = action_values[a] - node_value`.

### Status: DEPLOYED — made things worse (see Phase 8)

---

## Phase 8: AS + No-10x Run — Failed (April 5, 2026)

### EC2 run: AS + no-10x + standard pruning + fixed strategy_sum
Instance `i-016f5878b0e6aaf2e`, fresh start, 500M checkpoint.

**Result at 500M — WORSE than all previous runs:**

| Hand | Phase 6 (pruning fix, 10x) | Phase 7+8 (AS, no-10x) |
|------|---------------------------|------------------------|
| AA | 11% raise (improving) | **uniform (no signal)** |
| KK | 1.4% raise (improving) | **0% raise (100% call)** |
| TT | 63% raise | **0% raise (100% call)** |
| 99 | 73% raise | 79% raise |

BTN converged perfectly (all pairs raise 85-97%). UTG was broken.

### Root cause: AS bootstrap problem
AS makes non-traversers sample from strategy_sum. In early iterations,
strategy_sum is dominated by accumulated uniform strategy (all regrets start
at 0 → 1/11 each action). Opponents sample uniformly → all UTG action values
look similar → no regret differentiation → strategy stays uniform → more
uniform data in strategy_sum → self-reinforcing loop.

Every-visit strategy_sum accumulation worsened this: 83M uniform entries
drowned out later signal.

### Killed instance, reverted AS

---

## Phase 9: Return to Pluribus-Exact (April 5, 2026)

### Full audit: what deviated from Pluribus?

After 4 failed attempts, audited every algorithmic parameter against the paper:

| Parameter | Pluribus | Our code | Status |
|-----------|----------|----------|--------|
| Non-traverser sampling | Current σ | **AS (strategy_sum)** | ✗ Reverted |
| Regret scaling | None | **10x** | ✗ Removed |
| Strategy_sum interval | Every 10K | **Every visit** | ✗ Fixed: 10007 (coprime) |
| Pruning | River + fold | River + fold | ✓ |
| Discount | d=T/(T+1) | d=T/(T+1) | ✓ |
| PPot/NPot bucketing | K-means [EHS,PPot,NPot] | **All zeros (broken)** | ✗ Fixed |
| Turn bucketing | K-means | **30-sample EHS percentile** | ✗ Fixed: k-means centroids |
| River bucketing | K-means (but PPot/NPot=0) | 200-sample EHS percentile | ~ equivalent |

### Changes made
1. **Reverted AS** — non-traverser samples from current σ (standard ES)
2. **10x already removed** in Phase 7
3. **Strategy_sum interval** — changed from every-visit to every 10007 iterations
   (coprime with 6, avoids aliasing, matches Pluribus intent)
4. **Fixed PPot/NPot** in ca_compute_features — padding cards were identical to
   board completion cards, making current=final evaluation, PPot/NPot always 0
5. **Turn k-means centroids** — precomputed during init from 2000 sampled boards,
   nearest-centroid lookup during traversal
6. **River 200 samples** — up from 30
7. **Flop texture cache regenerated** with fixed PPot/NPot

### Monitoring improvement
Worker runs in 50M mini-chunks with lightweight probes between each.
`bp_get_strategy` reads directly from memory — no disk I/O. Probe uploaded
to `s3://probes/probe_latest.txt` (~100 bytes). Full checkpoint every 500M.

### Status: ✅ DEPLOYED — running on `i-0de1597e03994a548`

---

## Current State (end of April 5, 2026)

### What the code matches
Every algorithmic parameter now matches Pluribus exactly. Remaining known
gaps are action abstraction (our 8 open sizes vs Pluribus hand-selected)
and total compute (targeting 12,400 core-hours on 192-core metal instance).

### What's running
Instance `i-0de1597e03994a548` (c7a.metal-48xl), fresh start, 5B iteration
target, 50M probe interval, 500M full checkpoint interval.
Texture cache on S3 with fixed PPot/NPot k-means.

### Monitoring
```bash
# Quick probe (100 bytes, instant):
aws s3 cp s3://poker-blueprint-unified/probes/probe_latest.txt -

# Full summary (after 500M checkpoint):
aws s3 cp s3://poker-blueprint-unified/summaries/strategy_500M.txt -

# SSH log:
grep "Probe" /var/log/blueprint-unified.log | tail -5
```

### Key question this run answers
Does standard Pluribus-matching ES-MCCFR (no 10x, no AS, fixed bucketing)
converge correctly for UTG pocket pairs? If TT/99 show raise signal in
the first few 50M probes, the algorithm works and we just need compute time.
If not, the issue is in the action abstraction (8 open sizes).
