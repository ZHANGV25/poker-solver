# Verification Suite — Master Prompt

Paste this entire file into a new Claude Code conversation.

---

## Context

You are working on a Pluribus-exact 6-player no-limit hold'em poker solver at `/Users/victor/Documents/Dev/poker-solver/`. The private repo is `ZHANGV25/poker-solver-dev`.

A blueprint solve is currently running on EC2 (c5.metal, 96 cores). It computes a Nash equilibrium strategy for the entire game via External-Sampling MCCFR. Target: 93 billion iterations (~130 hours of solving). Checkpoints are saved to `s3://poker-blueprint-unified/` every 10B iterations as `checkpoints/regrets_latest.bin` (BPR3 format, ~63 GB each).

**Your job:** Build a verification suite that proves the solver output is correct. Build ALL of the following tools, test them, and commit to git when done. Do NOT modify any files in `src/`, `python/`, or `precompute/` — those are the production solver. All your work goes in `tests/` and `verification/`.

---

## Codebase Layout

```
poker-solver/
├── src/
│   ├── mccfr_blueprint.c      — C MCCFR engine (hash table, traversal, regrets)
│   ├── mccfr_blueprint.h      — Data structures (BPInfoKey, BPInfoSet, BPConfig, BPSolver)
│   ├── card_abstraction.c     — EHS bucketing, k-means, 169 preflop classes
│   └── cuda/street_solve.cu   — GPU real-time solver
├── python/
│   ├── blueprint_v2.py        — BPS2/BPS3 loader
│   ├── hud_solver.py          — Runtime decision pipeline
│   ├── range_narrowing.py     — Bayesian range tracker
│   └── leaf_values.py         — Continuation values
├── precompute/
│   ├── blueprint_worker_unified.py — EC2 orchestration
│   └── watchdog.sh
├── tests/
│   ├── check_convergence.c    — Fast C checkpoint parser (already exists)
│   └── test_before_deploy.sh  — Pre-deploy integration tests (already exists)
└── verification/              — YOUR WORK GOES HERE
```

---

## Checkpoint Format (BPR3)

The regret checkpoint file (`regrets_latest.bin`) has this binary format:

```
Header (20 bytes):
  "BPR3"           — 4 bytes magic
  table_size       — int32 (1,342,177,280)
  num_entries       — int32
  iterations_run   — int64

Per entry (variable size):
  player           — int32 (0-5: SB,BB,UTG,MP,CO,BTN)
  street           — int32 (0=preflop, 1=flop, 2=turn, 3=river)
  bucket           — int32 (0-168 preflop, 0-199 postflop)
  board_hash       — uint64 (canonical board hash)
  action_hash      — uint64 (hash of action sequence)
  num_actions      — int32 (typically 2-7)
  regrets          — int32[num_actions] (cumulative regrets)
  has_strategy_sum — int32 (0 or 1)
  if has_strategy_sum:
    strategy_sum   — float32[num_actions]
```

Strategy is computed from regrets via regret matching:
```
strategy[a] = max(0, regret[a]) / sum(max(0, regret[all_actions]))
If all regrets <= 0: strategy = uniform (1/num_actions each)
```

---

## Preflop Bucket Mapping

169 lossless classes, ordered:
- 0=AA, 1=AKs, 2=AKo, 3=AQs, 4=AQo, 5=AJs, 6=AJo, ...
- 25=KK, 26=KQs, 27=KQo, ...
- 48=QQ, ...
- Pairs at positions: 0(AA), 25(KK), 48(QQ), 69(JJ), 88(TT), 105(99), 120(88), 133(77), 144(66), 153(55), 160(44), 165(33), 168(22)

---

## Action Encoding

At each decision point, actions are generated in order:
- 0: fold (if facing a bet)
- 1: check/call
- 2+: raises at 0.5x, 1x, 2x, 3x pot (preflop) or 0.5x, 1x, 2x pot (postflop)

The action_hash encodes the full action history. Root (empty history) has action_hash = 0xFEDCBA9876543210.

---

## What to Build

### 1. Convergence Trend Tool (`verification/convergence_trend.py`)

Downloads checkpoints from S3 (or reads local copies) and tracks convergence over time. Since checkpoints are 63 GB, spin up a temporary `t3.medium` EC2 instance with 100 GB disk to run the C checker (`tests/check_convergence.c`), upload results to S3, then terminate.

Build a Python script that:
- Launches a checker instance for each available checkpoint
- Collects the results (near-uniform %, dominant action %, avg|regret| per street, preflop strategies)
- Plots convergence curves over iterations (save as PNG)
- Can be run repeatedly as new checkpoints appear

The C checker already exists at `tests/check_convergence.c`. Read it to understand the output format, then parse it.

### 2. Strategy Consistency Checks (`verification/strategy_checks.py`)

Parse a checkpoint and verify poker-knowledge invariants. Write a C program for speed (63 GB file). Checks:

**Preflop:**
- AA (bucket 0) never folds from any position (fold frequency < 1%)
- 72o (bucket 167) folds >90% from UTG/MP/CO
- All pocket pairs (buckets 0,25,48,69,88,105,120,133,144,153,160,165,168) have fold frequency <50% from BTN
- No hand raises >80% from UTG with bucket >140 (weak hands shouldn't raise often UTG)

**Postflop (requires grouping by street):**
- River info sets: no hand with bucket >190 (near-nuts) folds >5%
- Flop info sets: average bet frequency < average check frequency (players check more than bet on the flop in general)
- Turn: bet frequency > flop bet frequency (value hands bet more as board develops)

**General:**
- No action has probability exactly 0.0 or 1.0 for >50% of info sets (would indicate degenerate strategies)
- Distribution of num_actions: most entries should have 3-6 actions, very few with 1-2

Output: PASS/FAIL for each check with the actual numbers.

### 3. Self-Play Simulator (`verification/self_play.py`)

Simulate hands using the blueprint strategies. This requires:

**Card dealing:** Standard 52-card deck, deal 2 cards to each of 6 players, deal board cards.

**Strategy lookup:** For each player's decision, compute their bucket (preflop: use the 169-class mapping from `card_abstraction.c`; postflop: compute EHS from Monte Carlo simulation, map to 0-199 bucket), compute the board_hash and action_hash, look up the info set in the checkpoint, compute the regret-matched strategy, and sample an action.

**The challenge:** Loading and querying a 63 GB hash table. Options:
- Load the full checkpoint into a Python dict keyed on (player, street, bucket, board_hash, action_hash). At 63 GB this needs a machine with ~80 GB RAM. Use an r5.2xlarge ($0.50/hr) temporarily.
- Or write the self-play in C for speed and memory efficiency.

**Metrics to collect over 100K+ hands:**
- Win rate per position (bb/100 hands) — should sum to ~0
- Showdown frequency — players should see showdown ~25-30% of hands
- Fold to first bet frequency — should be 40-60%
- Average pot size — sanity check
- Voluntary put money in pot (VPIP) by position — BTN should be highest, UTG lowest

**Output:** Table of metrics by position, plus a PASS/FAIL on whether the sum of win rates is within ±2 bb/100 of zero.

### 4. Best-Response Exploitability (`verification/best_response.py`)

Freeze 5 players at blueprint strategies. For the 6th player (hero), compute the best possible counter-strategy using the GPU re-solver or exhaustive search.

This is the hardest tool. Start with a simplified version:
- Pick 100 common preflop spots
- For each, enumerate hero's possible actions and compute EV against the blueprint opponents
- Find the action that maximizes EV
- The difference between the best-response EV and the blueprint EV is the exploitability at that spot

The full version would use the GPU solver to compute best responses across all streets, but the preflop-only version is a good start and can run without a GPU.

**Output:** Average exploitability in bb/100 across the sampled spots. Below 5 bb/100 = good. Below 1 bb/100 = excellent.

---

## AWS Resources

- S3 bucket: `poker-blueprint-unified` (us-east-1)
- Checkpoints at: `s3://poker-blueprint-unified/checkpoints/regrets_latest.bin`
- Checkpoint metadata: `s3://poker-blueprint-unified/checkpoint_meta.json`
- IAM profile for EC2: `poker-solver-profile`
- Security group: `poker-solver-sg`
- Key pair: `poker-solver-key`

To spin up a checker instance, use the pattern in `tests/check_convergence.c` — the C checker already runs on EC2. Your tools should follow the same pattern: launch cheap instance, download checkpoint, run analysis, upload results, self-terminate.

---

## Order of Implementation

1. **Convergence trend** — easiest, builds on existing C checker
2. **Strategy consistency checks** — C program, moderate complexity
3. **Self-play simulator** — most complex, needs full game logic
4. **Best-response exploitability** — hardest, start with preflop-only

Build and test each tool locally where possible (use small test data), then run on EC2 with real checkpoints. Commit each tool as you complete it.

---

## Important Notes

- Do NOT modify files in `src/`, `python/`, or `precompute/`
- All new code goes in `verification/` or `tests/`
- The checkpoint file is 63 GB — don't try to download it to a laptop. Use EC2.
- The solver is running on instance `i-01a892bf1d1ec41e9` (c5.metal). Do NOT SSH into it to run analysis — it will OOM. Use a separate instance.
- AWS credentials may expire. If AWS commands fail, ask the user to refresh with `! aws configure`.
- Run `./tests/test_before_deploy.sh` if you modify anything that could affect the solver.
