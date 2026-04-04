# Debug Session: Hash Table Sizing & Pluribus Architecture Research

Paste this into a new Claude Code conversation. Working directory: `/Users/victor/Documents/Dev/poker-solver/`

Private repo: `ZHANGV25/poker-solver-dev`

---

## Context

We're building a 6-player no-limit hold'em blueprint solver matching Pluribus (Brown & Sandholm, Science 2019). The solver runs external-sampling MCCFR with Linear CFR discount and regret-based pruning on EC2 (c7a.metal-48xl, 192 cores, 376 GB RAM).

**The solver works correctly now** — at 2B iterations, all trash hands (32o, 63o, 22) fold from UTG with positive fold regrets, and premium hands (AA, KK, QQ) raise. This is a major improvement from previous runs where 32o was raising 74%.

## What was fixed (already done)

1. **board_hash removed from info set key** — was inflating the tree from ~100M to billions. Info set key is now `(player, street, bucket, action_hash)`, matching Pluribus. The bucket abstracts the board.

2. **int32 regret overflow fixed** — added ceiling at 310M, int64 intermediate arithmetic.

3. **Heap corruption fixed** — strategy_sum moved from heap calloc to arena allocator.

4. **eval_showdown_n** — folded traverser guard added.

See `docs/BLUEPRINT_BUGS.md` for full details.

## Current problem: hash table fills too quickly

The solver creates unique info sets faster than the hash table can hold them. With 8 preflop raise sizes and the current game tree:

| Run | Preflop sizes | Hash table | Filled at | Final IS count |
|-----|--------------|------------|-----------|---------------|
| Old (board_hash) | 4 | 1.34B | 80M iters | infinite (never plateaus) |
| No board_hash, 14 pre | 14 | 536M | 27M iters | 535M+ |
| No board_hash, 8 pre | 8 | 1B | ~35M iters | ~1B+ |
| No board_hash, 8 pre | 8 | 2B | ~75M iters | ~2B+ |

The tree is finite (creation rate declines from ~38 to ~24 new IS/iter) but large — estimated 1.5-2.5B total info sets with 8 preflop sizes. Even our 2B hash table fills at ~75M iterations, which is 0.16% of the 46.5B target.

**When the table fills, new info set lookups return 0.** This biases game values and slows convergence, though the effect is less severe than before (the common paths are all in the table; only rare deep sequences miss).

**Pluribus had 665M action sequences (414M encountered) with "up to 14" preflop sizes.** Our tree with 8 sizes has ~2B+ info sets. Something about our tree structure is fundamentally different.

## The current solver instance

Running on `i-078cac5cf7cab2b59` (c7a.metal-48xl, 192 cores, 376 GB RAM, IP 54.157.222.245).

- 2B hash table (250 GB RAM at capacity)
- 8 preflop sizes: [0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0]
- 2 postflop first-raise sizes: [0.5, 1.0] (all-in added automatically)
- 1 postflop subsequent-raise size: [1.0] (all-in added automatically)
- Postflop max raises: 3
- Preflop max raises: 4
- 200 buckets per postflop street, 169 lossless preflop classes
- Checkpoint every 2B iterations to S3

**The strategies at 2B iterations look correct** (trash folds, premiums raise). The solver is functional. The question is whether the miss rate after the table fills will prevent full convergence at 46.5B iterations.

## What to investigate

### 1. Deep Pluribus research: how does Pluribus keep the tree at 665M?

Our tree is 3-4x larger than Pluribus despite having fewer preflop sizes (8 vs "up to 14"). The difference must be structural. Research:

- **Postflop max raises per round**: Pluribus says "at most three raise sizes for the FIRST raise" and "at most two for REMAINING raises." Does "remaining raises" mean there's a max of 2 raises per round? Or is it unlimited but with fewer size options?
- **Preflop bet sizes per decision point**: "Up to 14" is the max. But does UTG get 14 sizes for the open, then MP gets 14 sizes for the 3-bet? Or does the 3-bet have fewer? The paper says sizes were "decided by hand based on what raise sizes earlier versions decided to use with significant positive probabilities."
- **Total raises across the hand**: Is there a global cap on the number of raises (e.g., 4 total preflop, 3 per postflop round)?
- **Board abstraction**: We removed board_hash. Verify Pluribus truly doesn't include the board in the info set key. Check the Pluribus supplementary materials carefully.
- **Action sequence count**: 665M sequences with 414M encountered. How many per street? Per position?

Search for the Pluribus supplementary PDF, CMU lecture slides, Noam Brown's talks, and the original Science paper. Also search for ReBeL, Student of Games, and other follow-up work that might clarify the blueprint structure.

### 2. Count our action sequences analytically

Write a tool that enumerates all possible preflop action sequences for 6 players with our current config (8 sizes, max 4 raises). This is a finite enumeration:

- Start: UTG acts (fold/call/raise×8 = 10 actions)
- After UTG: MP acts (fold/call/raise×8 or fold/call if max raises reached)
- Continue through all 6 players
- Count terminal sequences (round done) and non-terminal (raise reopens)

Compare to Pluribus's preflop sequence count. If ours is 10x more, the preflop tree is the bottleneck and we need fewer sizes or a per-decision-point limit.

### 3. Measure the actual miss rate

The production solver doesn't have miss counters. Options:
- Add miss logging to the solver (requires restart)
- Estimate from iteration speed (misses are cheap → higher speed = more misses)
- Compile a diagnostic build on the instance alongside the running solver (different port/memory)

### 4. Test: would tiered preflop sizes fix the tree size?

Instead of 8 sizes at every preflop decision:
- Open raise (first raise): 8 sizes
- 3-bet: 4 sizes (e.g., [0.7, 1.0, 2.0, 4.0])
- 4-bet: 2 sizes (e.g., [1.0, 3.0])
- 5-bet: all-in only

Write a test that measures tree size with this configuration vs flat 8 sizes.

### 5. Decision: let it run or restart?

The solver is running and producing correct-looking strategies at 2B iterations. Options:
- **Let it run**: Accept the miss rate. If convergence looks good at the 4B and 6B checkpoints, keep going.
- **Restart with smaller tree**: Drop to 4-5 preflop sizes or implement tiered sizing. Guarantees no misses but loses compute time.
- **Restart with bigger instance**: r7a.metal-48xl has 768 GB RAM → could fit a 4B hash table. More expensive but solves the problem directly.

## Key files

- `src/mccfr_blueprint.c` — the solver (traverse(), generate_actions(), info_table_find_or_create())
- `src/mccfr_blueprint.h` — data structures, BP_MAX_ACTIONS=16
- `precompute/blueprint_worker_unified.py` — Python orchestration, bet size config
- `precompute/launch_blueprint_unified.sh` — deploy script
- `docs/BLUEPRINT_BUGS.md` — full bug report
- `tests/test_tree_size.c` — tree size measurement tool
- `tests/test_miss_counter.c` — miss rate measurement tool
- `tests/extract_regrets.c` — raw regret extraction from checkpoints

## SSH access

```bash
rm -f /tmp/ec2-temp-key /tmp/ec2-temp-key.pub
ssh-keygen -t rsa -b 2048 -f /tmp/ec2-temp-key -N "" -q
aws ec2-instance-connect send-ssh-public-key \
    --instance-id i-078cac5cf7cab2b59 \
    --instance-os-user ec2-user \
    --ssh-public-key file:///tmp/ec2-temp-key.pub \
    --region us-east-1
ssh -i /tmp/ec2-temp-key -o StrictHostKeyChecking=no ec2-user@54.157.222.245
```

## Desired output

1. Clear understanding of why Pluribus's tree is 665M vs our ~2B
2. A specific tree size target and the config changes to hit it
3. Recommendation: keep running, restart with changes, or scale up hardware
