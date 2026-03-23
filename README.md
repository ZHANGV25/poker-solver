# poker-solver

Custom DCFR poker solver for real-time depth-limited subgame solving.

## Architecture

- **C solver** (`src/solver.c`): DCFR with Brown's parameters, O(N*M) showdown with precomputed strengths
- **Python bindings** (`python/solver.py`): ctypes wrapper for integration
- **Range narrowing** (`python/range_narrowing.py`): Bayesian range updates for both players

## Quick Start

```bash
# Build
make all

# Run benchmarks
make bench

# Run Python tests
python tests/test_end_to_end.py
```

## Performance (i7-13700K, single thread)

| Hands | 500 iter | Exploitability |
|-------|----------|----------------|
| 20    | 17ms     | 0.000%         |
| 40    | 50ms     | 0.000%         |
| 60    | 130ms    | 0.009%         |
| 80    | 218ms    | 0.083%         |
| 100   | 341ms    | 0.360%         |

## Status

- [x] DCFR solver (C)
- [x] Hand evaluation (7-card)
- [x] Precomputed hand strengths
- [x] Exploitability computation
- [x] Python bindings (ctypes)
- [x] Range narrowing (Bayesian)
- [x] End-to-end pipeline test
- [ ] Depth-limited leaf values (4 continuation strategies)
- [ ] Multi-threading
- [ ] Blueprint I/O (LZMA compressed)
- [ ] CUDA GPU acceleration
