# Poker Solver Architecture — Pluribus Implementation

## Overview

Full Pluribus architecture for 6-max NLHE: unified preflop-through-river
blueprint via CPU MCCFR, real-time GPU subgame search during play.

## Components

### Blueprint Training (CPU, EC2)
**`mccfr_blueprint.c`** — Production N-player external-sampling MCCFR
- 2-6 player, OpenMP Hogwild parallelism
- Int32 regrets with -310M floor, -300M pruning threshold
- Linear CFR discount (first 35% of iterations)
- Card abstraction via hand-to-bucket mapping (200 buckets/street)
- Cumulative investment tracking for correct multi-street payoffs
- ~44K iter/s single-thread, scales linearly with cores

**`card_abstraction.c`** — Fast EHS computation + percentile bucketing
- Monte Carlo equity computation using hand_eval.h
- 1176 hands × 500 samples in ~2 seconds
- 200-bucket percentile assignment (k-means planned)
- 169 lossless preflop hand classes

### Runtime Search (GPU, local RTX 3060)
**`street_solve.cu`** — N-player single-street GPU solver
- 2-6 player subgame solving in one betting round
- Level-batched Linear CFR, regret-based pruning
- 4 continuation strategies per player at depth-limit leaves
- 67ms river, 236ms flop (200 hands, 200 iterations)

### Research (GPU)
**`gpu_mccfr.cu`** — Batch outcome-sampling MCCFR (novel)
- Each GPU thread = one complete game trajectory
- 17M trajectories/sec on RTX 3060
- Finding: not competitive with CPU for blueprint convergence
- Value: research contribution, code reusable for GPU equity computation

### Python Layer
- **`hud_solver.py`** — Street-by-street decision pipeline for ACR HUD
- **`gpu_mccfr.py`** — GPU MCCFR wrapper with EHS bucketing
- **`street_solver_gpu.py`** — ctypes wrapper for street_solve.dll
- **`blueprint_store.py`** — Binary storage format (LZMA compressed)
- **`range_narrowing.py`** — Bayesian range tracking (weighted average)
- **`off_tree.py`** — Pseudoharmonic interpolation for off-tree bets

### Precompute Infrastructure
- **`blueprint_worker.py`** — Per-EC2-instance texture solver
- **`launch_blueprint.sh`** — Parallel EC2 spot instance launcher (20 instances)
- **`solve_scenarios.py`** — 1,755 flop texture generation (suit isomorphism)

## Decision Pipeline

```
PREFLOP (TODO):
  6-player CFR+ over 169 hand classes → mixed-frequency ranges
  All positions solve simultaneously (UTG through BB)

POSTFLOP BLUEPRINT (EC2, offline):
  For each of 1,755 flop textures:
    Derive starting ranges from preflop strategy
    6-player MCCFR with 200 EHS buckets, ~1M iterations
    Store: P(action | bucket, position, history) per info set

REAL-TIME SEARCH (GPU, online):
  At each decision during play:
    1. Look up blueprint strategy
    2. Narrow opponent ranges via Bayesian update
    3. Build subgame from start of current betting round
    4. GPU solve with 4 continuation strategies at leaves
    5. Play final-iteration strategy
```

## Performance

| Component | Speed | Hardware |
|-----------|-------|----------|
| Blueprint (per texture) | ~3-5s @ 50K iter | i7-13700K, 1 thread |
| Blueprint (full, 1755 textures) | ~2-4 hrs estimated | 20× c5.4xlarge |
| GPU street solve (river) | 67ms | RTX 3060 |
| GPU street solve (flop) | 236ms | RTX 3060 |
| GPU MCCFR (research) | 17M traj/s | RTX 3060 |
| Card abstraction (EHS) | 2s / 1176 hands | i7-13700K |
