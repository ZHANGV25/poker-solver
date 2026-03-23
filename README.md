# poker-solver

Pluribus-style depth-limited DCFR poker solver for real-time 6-max NLHE.

## Architecture

```
PREFLOP:  Scenario selection → starting ranges (instant)
FLOP:     Precomputed blueprint lookup + Bayesian range narrowing (instant)
TURN:     Re-solve with narrowed ranges + 4 continuation strategies (~200ms)
RIVER:    Re-solve with narrowed ranges to showdown (~180ms)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design details.

### Components

| Component | File | Description |
|-----------|------|-------------|
| **C solver v2** | `src/solver_v2.c` | Linear CFR (Pluribus-style), final iteration strategy, leaf continuation values |
| **C solver v1** | `src/solver.c` | DCFR baseline, cross-validated against postflop-solver (Rust) |
| **Hand evaluation** | `src/hand_eval.h` | 7-card eval via 21× 5-card combinations |
| **CUDA solver** | `src/cuda/gpu_solver.cu` | GPU batch solver for EC2 precomputation (WIP) |
| **Python bindings** | `python/solver.py` | ctypes wrapper, range parsing |
| **Range narrowing** | `python/range_narrowing.py` | Bayesian updates for hero + villain |
| **Blueprint I/O** | `python/blueprint_io.py` | Read precomputed solutions (JSON + LZMA) |
| **Solver pool** | `python/solver_pool.py` | Thread pool for multi-table concurrent solving |
| **HUD interface** | `python/hud_solver.py` | High-level API for ACR HUD integration |
| **LZMA compression** | `python/compression.py` | 20x compression for blueprint data |
| **EC2 precompute** | `precompute/` | Batch solve + GPU/CPU deployment scripts |

## Quick Start

```bash
make all          # build solver v2 + DLL
make bench        # run benchmarks
make test         # run Python tests
```

## Performance (i7-13700K, single thread, solver v2, 500 iterations)

| Hands | River | Turn (w/ leaf eval) |
|-------|-------|---------------------|
| 40    | 35ms  | 48ms                |
| 60    | 104ms | 117ms               |
| 80    | 180ms | 203ms               |
| 100   | 301ms | 324ms               |

8 concurrent river solves: 103ms average, ~103ms wall time with threading.

## Pluribus Alignment

| Feature | Status |
|---------|--------|
| Linear CFR (DCFR α=1,β=1,γ=1) | ✅ Implemented |
| Final iteration strategy | ✅ Implemented |
| 4 continuation strategies at leaves | ✅ Implemented |
| Bayesian range narrowing | ✅ Implemented |
| Depth-limited to current street | ✅ Implemented |
| Unsafe search (no gadget) | ✅ Matches Pluribus |
| Subgame root at round start | ⬜ TODO |
| Multiway solving (>2 players) | ⬜ TODO (heads-up only) |

## Status

- [x] Solver v2 (Linear CFR, final iteration, leaf values)
- [x] Solver v1 (DCFR baseline, cross-validated)
- [x] Hand evaluation + precomputed strengths
- [x] Exploitability computation
- [x] Python bindings + range parser
- [x] Bayesian range narrowing (hero + villain)
- [x] Blueprint I/O (JSON + LZMA, 20x compression)
- [x] Multi-table solver pool
- [x] HUD integration interface
- [x] EC2 precompute pipeline
- [ ] CUDA GPU solver for fast precompute
- [ ] Turn strategy precomputation
- [ ] Subgame rooting at round start
