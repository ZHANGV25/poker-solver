# poker-solver

Custom DCFR poker solver for real-time depth-limited subgame solving. Pluribus-inspired architecture with Bayesian range narrowing.

## Architecture

```
Precompute (EC2):
  27 scenarios x 1,755 flop textures -> per-hand strategies
  Stored as JSON with suit isomorphism, LZMA-ready

Runtime (Desktop):
  Blueprint lookup -> Bayesian range narrowing -> DCFR re-solve
  Per-decision: <500ms for narrowed river, <2s for turn
```

### Components

- **C solver** (`src/solver.c`): DCFR with Brown's parameters, depth-limited leaf values
- **Python bindings** (`python/solver.py`): ctypes wrapper, range parsing
- **Range narrowing** (`python/range_narrowing.py`): Bayesian updates for hero + villain
- **Blueprint I/O** (`python/blueprint_io.py`): reads precomputed flop solutions
- **Solver pool** (`python/solver_pool.py`): thread pool for multi-table concurrent solving
- **HUD interface** (`python/hud_solver.py`): high-level API for ACR HUD integration
- **EC2 precompute** (`precompute/`): batch solve + deployment scripts

## Quick Start

```bash
# Build C solver
make all

# Run benchmarks
make bench

# Run unit tests
python tests/test_end_to_end.py

# Run integration tests (requires ACR HUD flop_solutions)
python tests/test_integration.py
```

## Performance (i7-13700K, single thread)

| Hands | 500 iter | Exploitability | Use Case |
|-------|----------|----------------|----------|
| 20    | 15ms     | 0.000%         | Tiny range |
| 40    | 38ms     | 0.000%         | Very narrow |
| 60    | 109ms    | 0.009%         | River (narrow) |
| 80    | 188ms    | 0.083%         | Turn (narrow) |
| 100   | 306ms    | 0.360%         | Moderate |
| 200   | 1,497ms  | 0.135%         | Full range |

Concurrent: 3 tables solved in 36ms wall time.

## EC2 Precompute

```bash
# Deploy to EC2 spot instances
cd precompute
./deploy.sh --all --workers 16

# Or solve a single scenario locally
python solve_scenarios.py --scenario CO_vs_BB_srp --workers 4
```

## Status

- [x] DCFR solver (C, O3-optimized)
- [x] Hand evaluation (7-card, 21-combo)
- [x] Precomputed hand strengths (213x speedup)
- [x] Exploitability computation (best-response)
- [x] Depth-limited leaf value support
- [x] Python bindings (ctypes, auto-compile)
- [x] Range narrowing (Bayesian, hero + villain)
- [x] Blueprint I/O (suit isomorphism, lazy load)
- [x] Solver pool (multi-table threading)
- [x] HUD integration interface
- [x] EC2 precompute pipeline
- [x] Cross-validated against postflop-solver (Rust)
- [ ] LZMA blueprint compression
- [ ] CUDA GPU acceleration
- [ ] Turn strategy precomputation
