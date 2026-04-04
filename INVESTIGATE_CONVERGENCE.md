# Investigation: Preflop Convergence Rate

Paste this into a new Claude Code conversation. Working directory: `/Users/victor/Documents/Dev/poker-solver/`

Private repo: `ZHANGV25/poker-solver-dev`

---

## The Question

The solver is at 2B iterations (of 46.5B target) and trash hands like 32o are still raising from UTG instead of folding. Is this:
1. Normal early-convergence behavior that will self-correct given enough iterations?
2. A sign that 46.5B iterations won't be enough?
3. A remaining bug?

The regret overflow is fixed (max regret = 997,504, well under the 310M ceiling). The strategies are bad because with 91.9% of info sets near-uniform, opponents don't defend against raises, making raising trash temporarily profitable.

## What We Know

- 2B iterations, 192 cores (c6a.metal), ~185K iter/s
- 46.5B total target (12,400 core-hours / 192 cores)
- Discount runs for first 1.63B iterations (35‰ of total), every 40.7M iters
- Pruning starts at 790M iterations (17‰ of total)
- 169 preflop classes, each visited ~2M times total, ~365K undiscounted
- Preflop avg|regret| = 134,504
- Max regret anywhere = 997,504
- Near-uniform = 91.9%

Preflop strategies at 2B (UTG root):
```
AA  (0):   raise mix 43/57 — correct
KK  (25):  call 46%, raise 54% — plausible (Pluribus limps)
32s (166): fold 100% — correct!
32o (167): raise 100% r0.5x — WRONG
63o (150): call 100% — WRONG
J3s (84):  raise 100% r2x — WRONG
22  (168): raise 100% r0.5x — WRONG
33  (165): raise 100% r3x — WRONG
```

Note: 32s folds 100% but 32o raises 100%. These should behave almost identically.

## What to Investigate

### 1. Is 32s vs 32o divergence a clue?

Bucket 166 (32s) folds 100%, bucket 167 (32o) raises 100%. These hands are nearly identical in strength. If they have completely opposite strategies, something is wrong with how they accumulate regrets. Check:
- Are they visiting the same info set nodes, or do they have different action_hashes?
- Do they have different starting stacks/pots when they act? (They shouldn't)
- Is there a hash collision putting 32o in the same slot as a strong hand?

### 2. Write a convergence rate estimator

Compute how many iterations each preflop class needs to converge. The key metric: when does fold_regret for 32o overtake raise_regret?

For each preflop class at UTG root:
- Parse the checkpoint, extract the actual regret values (not just strategies)
- Compute fold_regret / raise_regret ratio
- Estimate: at the current growth rate, when will fold dominate?

If fold_regret is growing (even slowly), convergence will happen — just need patience. If raise_regret is growing faster than fold_regret, convergence won't happen within 46.5B iterations.

### 3. Compare to Pluribus convergence rate

Pluribus: 12,400 core-hours, 64 cores, 665M info sets.
Us: 12,400 core-hours, 192 cores, 1.34B info sets.

Per info set, Pluribus had: 12,400 × 3600 / 665M = 67 core-seconds per info set.
Per info set, we have: 12,400 × 3600 / 1.34B = 33 core-seconds per info set.

We have HALF the compute budget per info set. Could this be why convergence is slower?

If so, we might need 2× more iterations (93B instead of 46.5B) — or reduce the number of info sets.

### 4. Run a small local test

Write a minimal 2-player preflop-only solver (no postflop) that runs fast:
- 2 players, preflop only (fold/call/raise), 169 classes
- Same MCCFR algorithm, same regret matching
- Run 10M iterations locally in seconds
- Check: does 32o converge to fold? How many iterations does it take?

If 32o converges to fold in the small test, the algorithm is correct and we just need more iterations for the 6-player version. If it doesn't converge, there's a bug in the MCCFR implementation.

### 5. Check if the traversal is correct

Read `traverse()` in `src/mccfr_blueprint.c` carefully. Verify:
- When UTG (traverser) raises with 32o and opponents call, does the payoff reflect that 32o usually loses at showdown?
- When UTG folds with 32o, does the payoff correctly return 0 (no loss)?
- Is the blind posting correct? UTG hasn't posted anything — folding should cost 0.
- Are opponents' sampled actions reasonable, or is there a bias toward folding that makes raises too profitable?

### 6. Check the regret values directly

Write a C tool that extracts the RAW regret values (not strategies) for specific preflop classes. For UTG root:
```
32o (bucket 167): fold_regret=???, call_regret=???, raise_regrets=???
32s (bucket 166): fold_regret=???, call_regret=???, raise_regrets=???
AA (bucket 0): fold_regret=???, call_regret=???, raise_regrets=???
```

This tells us exactly what the solver "thinks" about each action. If fold_regret for 32o is negative, there's a bug. If it's positive but small compared to raise, we just need more iterations.

## How to Access the Solver

The solver is running on `i-000794d25703e439d` (c6a.metal, 192 cores, 376 GB RAM, IP 54.167.241.124).

```bash
rm -f /tmp/ec2-temp-key /tmp/ec2-temp-key.pub
ssh-keygen -t rsa -b 2048 -f /tmp/ec2-temp-key -N "" -q

aws ec2-instance-connect send-ssh-public-key \
    --instance-id i-000794d25703e439d \
    --instance-os-user ec2-user \
    --ssh-public-key file:///tmp/ec2-temp-key.pub \
    --region us-east-1

ssh -i /tmp/ec2-temp-key -o StrictHostKeyChecking=no ec2-user@54.167.241.124
```

The checkpoint is at `/opt/blueprint_unified/regrets_latest.bin` (56 GB, BPR3 format). The instance has 376 GB RAM — you can read the checkpoint freely (260 GB headroom).

## Key Files

- `src/mccfr_blueprint.c` — full solver, traverse() is the core
- `src/mccfr_blueprint.h` — data structures
- `src/card_abstraction.c` — ca_preflop_classes() for bucket mapping
- `tests/check_convergence.c` — existing C checker (BPR3-aware)

## Checkpoint Format (BPR3)

```
Header: "BPR3"(4B) + table_size(int32) + num_entries(int32) + iterations_run(int64)
Per entry: player(int32) + street(int32) + bucket(int32) + board_hash(uint64) + 
           action_hash(uint64) + num_actions(int32) + regrets(int32[na]) + 
           has_sum(int32) + [strategy_sum(float32[na])]
```

Root action_hash (empty history) = 0xFEDCBA9876543210

## Desired Output

1. The raw regret values for 32o, 32s, AA, 72o, 22 at UTG root
2. Whether fold regret is growing or shrinking relative to raise
3. An estimate of how many total iterations are needed for preflop convergence
4. Whether a small local test confirms the algorithm converges correctly
5. A clear verdict: keep running, increase iterations, or fix a bug
