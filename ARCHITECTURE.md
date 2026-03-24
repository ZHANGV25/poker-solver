# Poker Solver Architecture — Pluribus Implementation

## Overview

Full Pluribus architecture for 6-max NLHE: N-player street-by-street GPU
solving with precomputed blueprint strategies from N-player MCCFR.

## Components

### Runtime (GPU, <300ms per decision)
**`street_solve.cu`** — N-player single-street GPU solver
- 2-6 player subgame solving in one betting round
- Level-batched Linear CFR, regret-based pruning
- External leaf values from blueprint continuation strategies
- 4 continuation strategies per player at depth-limit leaves
- Extracts final-iteration strategy (play) + weighted average (narrowing)

### Blueprint Training (CPU, EC2)
**`mccfr_blueprint.c`** — N-player external-sampling MCCFR
- Full flop→turn→river traversal with sampled chance/opponent actions
- Linear CFR discounting (DCFR α=β=γ=1)
- Hash table info set storage (~4M slots)
- Designed for parallel execution across CPU cores

### Python Layer
- **`hud_solver.py`** — Street-by-street decision pipeline for ACR HUD
- **`street_solver_gpu.py`** — ctypes wrapper for street_solve.dll
- **`leaf_values.py`** — Continuation values from blueprint (4 biased strategies)
- **`blueprint_store.py`** — Binary storage format (LZMA compressed)
- **`range_narrowing.py`** — Bayesian range tracking (weighted average)
- **`off_tree.py`** — Pseudoharmonic interpolation for off-tree bets

## Decision Pipeline

```
PREFLOP:  Load ranges from ranges.json → starting ranges

FLOP:
  First action → blueprint lookup (instant)
  Facing action → narrow villain range, GPU re-solve with turn leaf values

TURN:
  Narrow ranges from flop actions
  GPU re-solve with river equity leaf values

RIVER:
  Narrow ranges from turn actions
  GPU re-solve to showdown (no depth limit)
```

## Solver Configuration (matching Pluribus)

| Setting | Value |
|---------|-------|
| Algorithm | Linear CFR (DCFR α=β=γ=1) |
| Strategy for play | Final iteration |
| Strategy for narrowing | Weighted average |
| Bet sizes | [33%, 75%, 150%, all-in] |
| Max raises | 3 per street |
| Iterations | 200 per subgame |
| Leaf evaluation | 4 continuation strategies per remaining player |
| Off-tree mapping | Pseudoharmonic (Johanson 2013) |
| Pruning | Regret threshold -10000 (skip 95% of iterations) |

## Performance (RTX 3060, 200 hands, 200 iterations)

| Street | Time | Nodes |
|--------|------|-------|
| River | 67ms | 159 |
| Flop (cont strats) | 236ms | 1219 |
| Turn (cont strats) | 269ms | 1219 |
| 3-player river | 141ms | 1156 |

## Storage Estimates

| Config | Raw | Compressed |
|--------|-----|------------|
| 2-player, 27 scenarios | ~1.5 GB | ~200 MB |
| 6-player, 27 scenarios | ~47 GB | ~6 GB |
