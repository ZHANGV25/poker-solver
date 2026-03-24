# Blueprint Precompute — What It Is and How It Works

## What the Blueprint IS

The blueprint is a **complete strategy for the entire game of 6-max NLHE**, computed
offline via self-play. It tells every player what to do (probability over actions)
at every decision point, for every possible hand, given every possible action history.

Pluribus computes ONE blueprint for the ENTIRE game. There is no per-flop or
per-scenario separation — it's a single monolithic MCCFR run over the full game tree.

### Key point: NO range narrowing during blueprint computation

The blueprint assumes **all players start with the full range** (all 1326 hands,
abstracted to 169 preflop / 200 per postflop street). There is no "opponent range"
or "scenario" during training. Range narrowing is a **real-time search** concept
that happens during actual play, NOT during blueprint computation.

During blueprint MCCFR:
- Each iteration, ONE random hand is sampled for each player from ALL possible hands
- The traversal goes through the FULL game: preflop → flop → turn → river → showdown
- All players start active, with full stacks
- The algorithm doesn't know or care about "scenarios" like "CO vs BB SRP"

### What about our scenario-based approach?

Our existing precompute (`solve_scenarios.py`) was designed for 2-player subgame
solving — each scenario (CO_vs_BB_srp, BTN_vs_SB_3bp, etc.) is solved independently
with specific starting ranges. This is NOT how Pluribus works.

**Pluribus approach** (correct, what we should do):
- Single MCCFR run, full 6-player game, preflop through river
- 169 preflop buckets, 200 postflop buckets per street
- All 665M action sequences in one hash table
- 64 cores, 8 days, <512GB RAM

**Our scenario approach** (what we had before):
- 27 separate 2-player solves per flop texture
- Each scenario has its own starting ranges
- 1,755 textures × 27 scenarios = ~47K solves
- This produces strategies conditional on ranges, NOT unconditional blueprint strategies

### The correct architecture

For a Pluribus-equivalent engine:

```
BLUEPRINT (offline, one-time):
  Single MCCFR run over full 6-max NLHE
  All 6 players, preflop to river
  169 preflop buckets × 200 flop/turn/river buckets
  Output: P(action | info_set) for every encountered info set
  ~400M info sets, ~128GB compressed

REAL-TIME SEARCH (online, per-decision):
  At each decision during play:
  1. Look up blueprint strategy for current state
  2. Narrow opponents' ranges via Bayesian update on observed actions
  3. Build subgame from START of current betting round
  4. Solve subgame with GPU (street_solve.cu)
  5. Use 4 continuation strategies at leaf nodes
  6. Play final-iteration strategy
```

## Can We Parallelize the Blueprint?

Yes, but not by splitting into scenarios. The parallelism is:

### Option A: Single large instance (Pluribus approach)
- One c5.18xlarge (72 vCPUs), OpenMP Hogwild
- 8 days, ~$576
- Simplest, proven to work

### Option B: Multiple instances with shared state (not practical)
- MCCFR needs shared regret tables
- Network latency between instances makes this infeasible
- Pluribus used shared memory on ONE server for this reason

### Option C: Split by flop texture (our approach — approximate)
- Run independent MCCFR for each flop texture
- Each instance solves a subset of flop textures
- This is an APPROXIMATION — it doesn't capture cross-texture learning
- But it's close enough for practical purposes
- Parallelizes perfectly across instances

**We use Option C** because:
1. It parallelizes across 20 EC2 instances
2. Each instance needs only ~8-16GB RAM (one texture at a time)
3. Cost: ~$50 per instance × 1-2 days = ~$1000-2000 total
4. Quality: slightly lower than Pluribus's single-run approach, but good enough

## How Option C Works (Our Implementation)

### Step 1: Partition work across instances

There are 1,755 unique flop textures (after suit isomorphism).
With 20 instances: ~88 textures per instance.

For each texture, we run full 6-player MCCFR from the flop through river:
- 6 players, all starting ranges (200 buckets each)
- Full flop → turn → river → showdown traversal
- External sampling: sample turn/river cards, sample opponent actions
- ~50K-200K iterations per texture

### Step 2: Card abstraction

Before solving, compute 200 EHS buckets for each street:
- Flop buckets: computed once per flop texture
- Turn buckets: re-computed for each turn card (47 possible)
- River buckets: re-computed for each river card (46 possible)

This is computed on-the-fly during MCCFR traversal — when a new
turn/river card is sampled, the bucket mapping for that board is cached.

### Step 3: Per-texture MCCFR solve

For each flop texture:
1. Generate all 1176 possible hands (52 - 3 flop cards, choose 2)
2. Compute flop EHS buckets (200 buckets)
3. Initialize 6-player MCCFR with these hands/buckets
4. Run 100K+ iterations of external-sampling MCCFR
5. Save strategy snapshots to disk periodically
6. Extract weighted-average strategy for all info sets

### Step 4: Aggregate

Upload all per-texture blueprints to S3.
Download to local machine, combine into BlueprintStore format.

## Action Abstraction for Blueprint

Following Pluribus:
- **Flop**: fold, check/call, bet 50% pot, bet 100% pot, all-in (5 actions)
- **Turn**: fold, check/call, bet 50% pot, bet 100% pot, all-in (5 actions)
- **River**: fold, check/call, bet 50% pot, bet 100% pot, all-in (5 actions)
- Max 3 raises per street

## What About Preflop?

Pluribus's blueprint includes preflop with 169 lossless hand classes.
We skip preflop in the blueprint and instead use:
- Fixed preflop ranges from GTO solutions (ranges.json)
- 27 scenarios define which hands are in each player's postflop range

This is a simplification vs Pluribus but acceptable because:
- Preflop play is relatively simple (few decision points)
- GTO preflop ranges are well-established
- The important part is postflop play (where search happens)

## Memory Estimates

Per flop texture (6-player, 200 buckets, full tree):
- Info sets: ~500K-2M (varies by texture complexity)
- Memory per info set: ~8 actions × 200 buckets × 4 bytes (int32) = 6.4 KB
- Total: ~3-13 GB per texture
- With lazy allocation (~60% encountered): ~2-8 GB per texture

Per instance (solving one texture at a time sequentially):
- Peak RAM: ~16 GB
- Instance type: c5.4xlarge (16 vCPU, 32 GB RAM, ~$0.70/hr spot)
- Or c5.9xlarge (36 vCPU, 72 GB RAM, ~$1.50/hr spot) for faster solving

Total across all textures:
- 1,755 textures × ~10 MB compressed output = ~17.5 GB total blueprint
- Upload to S3, download to local machine

## Timeline Estimate

With 20 × c5.4xlarge instances:
- 1,755 textures / 20 instances = ~88 textures per instance
- ~5 minutes per texture (100K iterations at ~20K iter/s with 16 cores)
- ~88 × 5 min = ~7.3 hours per instance
- Total wall time: ~8 hours
- Total cost: 20 instances × 8 hours × $0.70/hr = ~$112

With 20 × c5.9xlarge instances (2x faster):
- ~4 hours per instance
- Total cost: 20 × 4 × $1.50 = ~$120
