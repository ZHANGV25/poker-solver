# Debug Prompt — Preflop Strategy Anomaly

Paste this into a new Claude Code conversation. Working directory: `/Users/victor/Documents/Dev/poker-solver/`

---

## The Problem

The 6-player MCCFR blueprint solver is running on EC2 at 9.87B iterations (of 93B target). The preflop strategies look wrong:

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

## How to Access the Solver

The solver is running on EC2 instance `i-01a892bf1d1ec41e9` (c5.metal, 96 cores, 192 GB RAM).

```bash
# Generate SSH key (if /tmp/ec2-temp-key doesn't exist)
ssh-keygen -t rsa -b 2048 -f /tmp/ec2-temp-key -N "" -q

# Push key and SSH in
aws ec2-instance-connect send-ssh-public-key \
    --instance-id i-01a892bf1d1ec41e9 \
    --instance-os-user ec2-user \
    --ssh-public-key file:///tmp/ec2-temp-key.pub \
    --region us-east-1

ssh -i /tmp/ec2-temp-key -o StrictHostKeyChecking=no ec2-user@44.197.198.230
```

Key paths on the instance:
- Solver log: `/var/log/blueprint-unified.log`
- Checkpoint: `/opt/blueprint_unified/regrets_latest.bin` (62 GB, BPR3 format)
- Source code: `/opt/poker-solver/src/mccfr_blueprint.c`
- Worker: `/opt/poker-solver/precompute/blueprint_worker_unified.py`

**WARNING: Do NOT run any memory-heavy programs on this instance.** The solver uses 156 GB of 188 GB. Running the convergence checker or any program that reads the 62 GB file will OOM-kill the solver. Read source code and logs only.

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

### 1. Regret accumulation correctness

In `traverse()`, when the traverser acts, regrets are updated:
```c
regret[action] += value_of_action - weighted_average_value
```

Check:
- Is the regret delta computed correctly?
- Is it applied to the right info set?
- Does the traverser variable rotate correctly (each iteration, one player is traverser)?
- When UTG is traverser with 32o, and UTG raises, do opponents respond correctly (sample from their current strategies)?

### 2. Opponent sampling

When a non-traverser acts, they sample ONE action from their current strategy (regret-matched). Check:
- Is the sampling correct? Does it use the right info set for the opponent?
- If opponents are near-uniform (87% of info sets), they fold/call/raise roughly equally — is this causing UTG raises to be too profitable?
- Is there a bias in the sampling that favors folding?

### 3. Action encoding

For UTG preflop (to_call=100, pot=150 from blinds):
- Action 0: fold (lose 0 — haven't invested yet)
- Action 1: call 100 (limp into the pot)
- Actions 2-5: raise to various sizes

Check that generate_actions produces these correctly and that the action indices map to the right bet amounts throughout the traversal.

### 4. Preflop blind posting

The unified solver posts blinds before UTG acts:
- SB posts 50, BB posts 100
- UTG is the first voluntary actor
- Check that the pot, stacks, and invested arrays are set correctly before UTG acts
- Check that the action_hash at UTG's root node is the expected 0xFEDCBA9876543210 (empty history)

### 5. Discount and pruning timing

With 93B total iterations:
- discount_stop = 3,254,999,999 (~3.25B)
- discount_interval = 81,374,999 (~81M)
- prune_start = 1,580,999,999 (~1.58B)

At 9.87B iterations:
- 39 discounts were applied (confirmed in log)
- Pruning has been active since 1.58B

Check:
- Does the discount correctly down-weight early (pre-pruning) regrets?
- After discount stops at 3.25B, do regrets accumulate without any damping?
- Could the regrets from the unpruned exploration phase (0 to 1.58B) be dominating despite discount?

### 6. The num_batches fix

`bp_solve` was recently fixed to use int64 for num_batches calculation. Verify:
- Does the first sub-chunk (2,147,483,647 iterations) actually execute all batches?
- Does iter_offset correctly propagate between sub-chunks?
- Does discount_count fast-forward correctly on subsequent sub-chunks?

### 7. Comparison: what SHOULD happen

In a correctly converging solver at 10% of training:
- Premium hands (AA, KK, QQ) should raise from every position — check
- Trash hands (32o, 72o) should fold from early positions — FAILING
- Medium hands should have mixed strategies — partially check
- Strategies should be more converged for preflop (visited every iteration) than postflop (visited rarely)

If the solver is correct but just needs more iterations, the trash hand raise frequencies should decrease over the next 10B iterations. If there's a bug, they'll stay the same or increase.

---

## Quick diagnostic you can run (READ ONLY on the instance)

```bash
# Check the solver is still running
ps aux | grep blueprint | grep -v grep

# Check recent log output
tail -5 /var/log/blueprint-unified.log

# Check discount and snapshot history
grep -E 'discount|snapshot' /var/log/blueprint-unified.log | tail -20

# Check memory (should be ~156 GB used)
free -h

# Read the traverse function
grep -n 'static float traverse' /opt/poker-solver/src/mccfr_blueprint.c
```

DO NOT run check_convergence.c or any tool that reads the 62 GB checkpoint file on the solver instance. It will OOM-kill the solver.

---

## Possible outcomes

1. **Bug found**: Stop the solver, fix the bug, restart. We lose the current run but save 4+ days of bad compute.
2. **No bug found, expected behavior**: Let the solver continue. Check again at 20B iterations (~10 hours). If trash hand raise frequencies are decreasing, the solver is converging. If not, dig deeper.
3. **Inconclusive**: If you can't determine the root cause from code inspection, write a small standalone test that creates a minimal solver (2 players, small game tree) and verifies regret accumulation produces sensible strategies within a few thousand iterations.
