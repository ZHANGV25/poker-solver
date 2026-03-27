# poker-solver

Exact Pluribus replica — 6-player unified preflop-through-river MCCFR blueprint + GPU real-time subgame search for 6-max No-Limit Hold'em.

## Goal

A commercial-grade poker AI matching Pluribus (Brown & Sandholm, Science 2019):
1. **Blueprint**: One unified 6-player MCCFR solve, preflop through river, on a single 72-core machine for 8 days
2. **Runtime**: GPU real-time subgame search with 4 continuation strategies, strategy freezing (A3), Bayesian range narrowing
3. **HUD**: Wire into ACR Poker for live play assistance

## Current Status

**Blueprint compute is RUNNING on EC2** (launched 2026-03-27, ~April 4 ETA).

| Milestone | Status |
|-----------|--------|
| Unified 6-player MCCFR engine | Done |
| 169 lossless preflop + 200 k-means postflop buckets | Done |
| Checkpoint/resume for spot instances | Done |
| EC2 pipeline (solver + watchdog) | Running |
| GPU N-player street solver | Done |
| A3 strategy freezing in GPU CFR | Done |
| Bayesian range narrowing (1326 hands) | Done |
| 4 continuation strategies (5x bias) | Done |
| Pseudoharmonic off-tree mapping | Done |
| HUD integration | Done (needs blueprint) |

See [GTO_GAPS.md](GTO_GAPS.md) for detailed tracking.

## Architecture

```
BLUEPRINT TRAINING (EC2, 8 days):
  bp_init_unified() → precompute 1,755 flop textures (k-means)
  → bp_solve() × N iterations (72-core OpenMP, Hogwild hash table)
  → bp_save_regrets() checkpoint to S3 every 1M iters
  → watchdog auto-relaunches on spot reclaim

RUNTIME (local GPU, <1 second):
  Preflop: blueprint lookup (169 classes)
  Flop:    GPU re-solve with narrowed ranges + 4 continuation strategies
  Turn:    GPU re-solve with equity leaf values
  River:   GPU re-solve to showdown
```

### Key Files

| Component | Files | Description |
|-----------|-------|-------------|
| **Blueprint engine** | `src/mccfr_blueprint.c` + `card_abstraction.c` | Production 6-player external-sampling MCCFR with bucket-in-key info sets, k-means bucketing, Linear CFR, pruning, arena allocator |
| **GPU solver** | `src/cuda/street_solve.cu` + `.cuh` | N-player single-street Linear CFR with exact showdowns, A3 strategy freezing, continuation strategies |
| **Blueprint worker** | `precompute/blueprint_worker_unified.py` | EC2 solver wrapper with checkpoint/resume, S3 upload |
| **Watchdog** | `precompute/watchdog.sh` | Auto-restart on spot reclaim + staleness detection |
| **HUD solver** | `python/hud_solver.py` | Full decision pipeline: blueprint → range narrowing → GPU re-solve |
| **Range narrowing** | `python/range_narrowing.py` | Bayesian updates over 1326 hands per player |
| **Leaf values** | `python/leaf_values.py` | N-player depth-limited continuation values |
| **Off-tree** | `python/off_tree.py` | Pseudoharmonic bet interpolation |

## Building

```bash
# Blueprint engine (requires both .c files + OpenMP)
make blueprint

# Or manually:
gcc -O2 -shared -fopenmp -o build/mccfr_blueprint.dll \
    src/mccfr_blueprint.c src/card_abstraction.c -I src -lm

# Street solver (requires CUDA)
nvcc -O2 -shared -o build/street_solve.dll src/cuda/street_solve.cu -I src
```

## Pluribus Parameter Match

| Parameter | Pluribus | Ours |
|-----------|----------|------|
| Players | 6 | 6 |
| Algorithm | External-sampling MCCFR | Same |
| Training | 8 days, 64 cores | 8 days, 72 cores |
| Preflop buckets | 169 lossless | 169 lossless |
| Postflop buckets | 200 k-means | 200 k-means |
| Discount | d=T/(T+1) every 10 min | Same (proportional) |
| Pruning | -300M threshold, 95% | Same |
| Regret floor | -310M (int32) | Same |
| Search | Linear CFR, 4 cont, 5x bias | Same (GPU) |
| Freezing | A3 (hero's past actions) | Same |
| Off-tree | Pseudoharmonic | Same |
| Beliefs | Uniform 1/1326 | Same |

## References

- Brown & Sandholm, "Superhuman AI for multiplayer poker", Science 2019
- See [pluribus_technical_details.md](pluribus_technical_details.md) for full parameter extraction
- See [REFERENCES.md](REFERENCES.md) for additional citations
