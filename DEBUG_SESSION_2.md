# Debug Session 2: Preflop Regret Lock-In (Pruning + Pure Strategy Trap)

Paste this into a new Claude Code conversation. Working directory: `/Users/victor/Documents/Dev/poker-solver/`

Private repo: `ZHANGV25/poker-solver-dev`

---

## Context

We're building a 6-player no-limit hold'em blueprint solver matching Pluribus (Brown & Sandholm, Science 2019). The solver runs external-sampling MCCFR with Linear CFR discount and regret-based pruning on EC2 (c7a.metal-48xl, 192 cores, 376 GB RAM).

**Previous bugs fixed (see `docs/BLUEPRINT_BUGS.md`):**
1. board_hash removed from info set key
2. int32 regret overflow fixed
3. Heap corruption fixed (arena allocator)
4. Hash table sizing: tiered preflop bet sizes (8/3/2/1) brought tree from 7.4B to 386M preflop IS
5. int64 overflow throughout (iteration counters, hash table indices)

**Current state:** The solver runs correctly, converges in the right direction, and all 6 positions show correct position gradient (UTG tight → BTN wide). At 2.6B iterations with a 3B hash table and 1.67B info sets, the macro patterns are right.

## The Bug: Preflop Regret Lock-In

**When a hand converges to 100% fold before the postflop has converged, its fold regret FREEZES and it can never recover.**

### Mechanism

In external-sampling MCCFR, the traverser's regret update is:
```
node_value = Σ σ[a] * action_value[a]
regret[a] += action_value[a] - node_value
```

When σ = [1, 0, 0, ...] (100% fold):
- `node_value = fold_value`
- `regret[fold] += fold_value - fold_value = 0` ← **FROZEN**
- `regret[raise] += raise_value - fold_value` ← keeps updating

The fold regret is permanently stuck. Meanwhile, regret-based pruning (threshold: -300M) stops exploring deeply negative actions, so those raise subtrees are never traversed and can never recover.

### Evidence

Raw regret trajectories from checkpoints at 2.0B, 2.2B, 2.4B, 2.6B iterations:

**UTG 99 — locked into fold, raise regrets sinking:**
```
Fold regret:  [+57219, +57219, +57219, +57219]  ← FROZEN (exact same int)
Best raise:   [-89M,  -105M,  -115M,  -126M]     ← getting WORSE
Worst raise:  [-309M, -309M,  -309M,  -309M]     ← at floor, pruned
```
99 from UTG should raise ~100%. Instead it folds 100%.

**UTG TT — same trap:**
```
Fold regret:  [+131K, +2.27M, +2.27M, +2.27M]   ← froze at 2.2B
Best raise:   [-63M,  -6.4M,  -11.8M, -17.9M]    ← improved then degraded
```

**UTG 22 — ESCAPED the trap (raise crossed zero):**
```
Fold regret:  [+228K, +7.5K,  -1.2M,  -3.9M]     ← going negative (good!)
Best raise:   [-8.2M, +4.9M,  +10.0M, +10.5M]    ← crossed zero, holding
```
22 escaped because its raise regret managed to cross zero before the lock-in solidified.

**UTG AA — not locked, but raise is undervalued:**
```
Fold regret:  [-310M, -310M,  -310M,  -310M]      ← correctly at floor
Best raise:   [+26M,  +31M,   +36M,   +27M]       ← oscillating, not growing fast
Raise %:      77%     87%     49%     78%           ← fluctuating (should be ~100%)
```

**strategy_sum is NULL for all UTG preflop root info sets** — snapshot accumulation doesn't start until 3.25B iterations (configurable), and the traverser-path accumulation (every 10K iters) doesn't seem to populate UTG entries. This means we only see the noisy current regret-matched strategy, not the smoother average.

### Affected hands

At 2.6B iterations, UTG folds 133/169 hands. Many of these are hands that should raise from UTG in equilibrium:
- All pocket pairs 22-TT (should raise, most are locked into fold)
- Suited connectors like 87s, 98s (should raise, locked into fold)
- Hands like A7s, A6s, A5s, K9s (should raise, locked into fold)

The position gradient is correct (UTG=133 fold > MP=127 > CO=110 > BTN=84 > SB=63), so the problem is specifically in the preflop root nodes converging to pure fold too early.

## Solver Parameters (Pluribus-matched)

```
Discount:  d = T/(T+1) every 40.7M iters, first 1.63B iters (40 events total)
Pruning:   after 790M iters, threshold -300M, 95% of iterations
Regret floor: -310M
Regret ceiling: +310M
Preflop tiers: 8 open / 3 three-bet / 2 four-bet / all-in five-bet
Max raises: 4 preflop, 3 postflop
Hash table: 3B slots (1.67B occupied at 2.6B iters = 56%)
Postflop: first raise [0.5x, 1.0x] + all-in, subsequent [1.0x] + all-in
Buckets: 169 preflop (lossless), 200 per postflop street
```

## Root Cause Analysis

The lock-in is caused by the interaction of three things:

1. **Early convergence to pure fold** — Before postflop values are accurate, marginal hands appear to have negative EV for all non-fold actions. The solver converges to 100% fold.

2. **Zero delta for dominant action** — Once at 100% fold, the regret update for fold is always exactly 0. The fold regret freezes.

3. **Regret-based pruning** — Actions with regret below -300M are never traversed. Their subtrees are never explored, so their values never improve. This makes the lock-in permanent.

## What to investigate / fix

### Option A: Exempt preflop from pruning
The simplest fix. Preflop info sets are a tiny fraction of the tree (~386M out of billions). Never pruning them means all preflop actions always get explored, preventing lock-in.

**Location:** `src/mccfr_blueprint.c` line 1038:
```c
if (ts->use_pruning && is->regrets[a] < BP_PRUNE_THRESHOLD) {
    if (ts->num_board < 5 && actions[a].type != ACT_FOLD) {
        action_values[a] = 0;
        continue;
    }
}
```
Add: `&& street > 0` to skip pruning on preflop. Or `&& ts->history_len > 6` to skip pruning for the first action round.

### Option B: Add regret floor for preflop actions
Instead of letting preflop raise regrets sink to -310M (where they get pruned), floor them at something like -50M. This keeps all actions above the pruning threshold and allows recovery.

### Option C: Epsilon-greedy exploration
Add a small exploration probability (e.g., 1%) that ignores the regret-matched strategy and samples uniformly. This ensures every action gets some exploration even when the strategy is pure.

### Option D: Different pruning threshold for preflop
Keep pruning but use a less aggressive threshold for preflop (e.g., -50M instead of -300M).

### Option E: Fix strategy_sum accumulation
The strategy_sum (average strategy) is NULL for UTG root info sets, which means the output strategy is the noisy current regret-matched one. Even if regrets oscillate, the average strategy should be smooth. Investigate why strategy_sum isn't being populated for UTG, and fix it. This might not fix the lock-in but would make the output strategies much better.

## Recommendation

Start with **Option A** (exempt preflop from pruning) — it's the smallest code change, directly addresses the root cause, and has minimal performance impact (preflop is a tiny fraction of the tree). Then also investigate **Option E** (strategy_sum) because the average strategy matters for the final output.

## Testing

After making changes, run the convergence trajectory test:
1. Fresh solve with the fix, extract at 1M, 5M, 20M, 100M milestones
2. Verify: 99 and TT should have positive raise regrets by 20M
3. Verify: fold regret should NOT be frozen (should increase/decrease)
4. Compare to the unfixed 2.6B checkpoint to confirm improvement

Use `tests/dump_raw_regrets.c` to check raw regrets and `tests/extract_roots.c` for strategy extraction.

## Key files

- `src/mccfr_blueprint.c` — solver (traverse(), generate_actions(), pruning logic at line 1038)
- `src/mccfr_blueprint.h` — data structures, constants
- `precompute/blueprint_worker_unified.py` — orchestration
- `precompute/launch_blueprint_unified.sh` — deploy script
- `tests/dump_raw_regrets.c` — raw regret extraction from checkpoints
- `tests/extract_roots.c` — root strategy extraction for all 6 positions
- `tests/quick_verify.py` — lightweight solve for testing
- `docs/BLUEPRINT_BUGS.md` — full bug history
- Checkpoint on S3: `s3://poker-blueprint-unified/checkpoints/regrets_2B.bin` (2.0B iters, 66 GB)
- Latest on S3: `s3://poker-blueprint-unified/checkpoints/regrets_latest.bin` (2.6B iters, 66 GB)

## SSH access (need to launch a new instance)

Use `precompute/launch_blueprint_unified.sh` or launch manually:
```bash
aws ec2 run-instances --instance-type c7a.metal-48xl --region us-east-1 ...
```

## Desired output

1. Code fix that prevents preflop regret lock-in
2. Verification that 99, TT, and similar hands converge to raise (not fold) from UTG
3. strategy_sum populating correctly for preflop roots
4. Fresh production run deployed with the fix
