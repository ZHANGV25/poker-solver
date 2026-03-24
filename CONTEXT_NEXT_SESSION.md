# Poker Solver — Context & Status

**Last updated**: 2026-03-24
**Project**: `C:/Users/Victor/Documents/Projects/poker-solver/`
**GitHub**: https://github.com/ZHANGV25/poker-solver.git

---

## Current State: Pluribus Architecture Implemented

The solver implements the exact Pluribus approach for 6-max NLHE:
- N-player street-by-street GPU solving (runtime)
- N-player external-sampling MCCFR (blueprint training)
- 4 continuation strategies at depth-limit leaves
- Bayesian range narrowing with weighted-average strategies
- Pseudoharmonic off-tree bet mapping
- Regret-based pruning

**What's blocking**: No blueprint data exists yet. The precompute pipeline is built but hasn't been run. Without blueprints, flop/turn solves use zero leaf values.

---

## Architecture

```
OFFLINE (CPU, EC2):
  mccfr_blueprint.c — N-player external-sampling MCCFR
  Solves full flop→turn→river for each texture
  Extracts: weighted avg P(action|hand) + EVs at flop/turn roots
  Stores: BlueprintStore binary format (~6GB for 6-player, 27 scenarios)

ONLINE (GPU, local RTX 3060):
  street_solve.cu — N-player single-street GPU solver
  Solves current street only (67-270ms, 200 iterations)
  Leaf values from precomputed blueprint continuation strategies
  4 cont strats × N players at depth-limit leaves
```

---

## File Manifest

### CUDA Solvers
| File | Purpose | Status |
|------|---------|--------|
| `src/cuda/street_solve.cu` | N-player single-street GPU solver | DONE, compiled |
| `src/cuda/street_solve.cuh` | Header (SS_MAX_PLAYERS=6, SS_MAX_HANDS=200) | DONE |
| `src/cuda/flop_solve.cu` | 2-player full-tree GPU solver + extract_all | DONE, compiled |
| `src/cuda/flop_solve.cuh` | Header with extended FSOutput | DONE |

### C Solvers
| File | Purpose | Status |
|------|---------|--------|
| `src/mccfr_blueprint.c` | N-player external-sampling MCCFR | DONE, compiled |
| `src/mccfr_blueprint.h` | Header (BP_MAX_PLAYERS=6) | DONE |
| `src/solver_v2.c` | 2-player CPU multi-street solver (legacy) | Working |

### Python Layer
| File | Purpose | Status |
|------|---------|--------|
| `python/street_solver_gpu.py` | N-player GPU solver wrapper (ctypes) | DONE |
| `python/hud_solver.py` | HUD interface — street-by-street solving | DONE |
| `python/leaf_values.py` | Continuation leaf values from blueprint | DONE (2-player) |
| `python/blueprint_store.py` | Binary blueprint read/write | DONE |
| `python/blueprint_io.py` | JSON blueprint read (legacy) | Working |
| `python/range_narrowing.py` | Bayesian range tracker | Working |
| `python/off_tree.py` | Pseudoharmonic bet mapping | DONE |
| `python/solver.py` | CPU solver wrapper (legacy) | Working |

### Precompute
| File | Purpose | Status |
|------|---------|--------|
| `precompute/run_all.py` | GPU precompute orchestrator (2-player) | DONE |
| `precompute/solve_scenarios.py` | Texture generation + scenario loading | Working |

### Tests
| File | Purpose |
|------|---------|
| `tests/test_street_solve.py` | GPU solver + blueprint + integration tests |

### Build Artifacts
| File | Source |
|------|--------|
| `build/street_solve.dll` | street_solve.cu (N-player GPU) |
| `build/flop_solve.dll` | flop_solve.cu (2-player GPU + extract_all) |
| `build/mccfr_blueprint.dll` | mccfr_blueprint.c (N-player CPU) |
| `build/solver_v2.dll` | solver_v2.c (2-player CPU) |

---

## Benchmarks (RTX 3060, i7-13700K)

### Runtime Solving (street_solve.cu, 200 hands, 200 iter, 3 bet sizes)
| Street | Nodes | Time |
|--------|-------|------|
| River | 159 | **67ms** |
| Flop (cont strats) | 1219 | **236ms** |
| Turn (cont strats) | 1219 | **269ms** |

### Blueprint Training (mccfr_blueprint.c, CPU)
| Players | Hands | Iterations | Time | Info Sets |
|---------|-------|------------|------|-----------|
| 2 | 15+21 | 50K | 5.3s | 2.1M |
| 3 | 24+27+21 | 30K | 1.4s | 685K |

---

## GTO Gap Tracking

### Done
- [x] **A1** N-player GPU street solver (2-6 players, CUDA)
- [x] **A2** N-player MCCFR blueprint (external sampling, CPU)
- [x] **A5** Precompute pipeline built (needs EC2 run)
- [x] **B1** 3 bet sizes + all-in
- [x] **B6** Pseudoharmonic off-tree mapping
- [x] Regret-based pruning (5x river speedup)
- [x] Per-leaf pot-aware continuation values
- [x] IP strategy extraction (not just root player)
- [x] Action classification for continuation bias
- [x] Leaf value computation wired into HUD solver

### Remaining (priority order)
1. **Wire MCCFR into precompute pipeline** — Python wrapper for mccfr_blueprint.dll, update run_all.py
2. **Run precompute on EC2** — generates blueprint data, enables leaf values
3. **Generalize leaf_values.py to N players** — currently 2-player only
4. **A3: Strategy freezing** — CUDA frozen_mask enforcement (tracking in place)
5. **A4: Warm-start** — persist regrets between solves
6. **B5: Preflop ranges** — proper GTO frequencies (not semi-binary)
7. **C1: GPU equity kernel** — all 46 river cards for turn leaves (currently 12 sampled)
8. **Research: GPU-parallel MCCFR** — run many independent iterations on GPU simultaneously

### Not Pluribus (accepted tradeoffs)
- No exploitative opponent modeling (Pluribus doesn't either)
- 200 exact hands vs Pluribus's 200 lossy buckets (ours is better per-hand)
- float32 precision (acceptable)

---

## Hardware & Build

- **CPU**: i7-13700K, **RAM**: 64GB, **GPU**: RTX 3060 12GB
- **OS**: Windows 10 Pro, **Shell**: Git Bash (MSYS2)
- **CUDA**: 11.8, **NVCC**: requires MSVC BuildTools
- **GCC**: 14.2.0 (MSYS2 UCRT64)
- **Python**: 3.9 (`/c/Users/Victor/AppData/Local/Programs/Python/Python39/python.exe`)

### NVCC compile command:
```bash
MSVC_DIR="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207"
WINSDK_INC="/c/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0"
WINSDK_LIB="/c/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0"
export PATH="$MSVC_DIR/bin/Hostx64/x64:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin:$PATH"

nvcc -O2 --shared -o build/street_solve.dll src/cuda/street_solve.cu -I src/cuda \
  -allow-unsupported-compiler -ccbin "$MSVC_DIR/bin/Hostx64/x64" \
  -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH \
  --compiler-options "/MD /I\"$MSVC_DIR/include\" /I\"$WINSDK_INC/ucrt\" /I\"$WINSDK_INC/um\" /I\"$WINSDK_INC/shared\"" \
  -L"$MSVC_DIR/lib/x64" -L"$WINSDK_LIB/ucrt/x64" -L"$WINSDK_LIB/um/x64"
```

### GCC compile command:
```bash
gcc -O2 -shared -o build/mccfr_blueprint.dll src/mccfr_blueprint.c -I src -lm
```

### Running .exe from Git Bash:
```bash
/c/Users/Victor/AppData/Local/Programs/Python/Python39/python.exe -c "import subprocess; ..."
```

---

## Key References

- Pluribus paper: https://noambrown.github.io/papers/19-Science-Superhuman.pdf
- Pluribus supplementary: https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf
- Depth-limited solving: https://proceedings.neurips.cc/paper_files/paper/2018/file/34306d99c63613fad5b2a140398c0420-Paper.pdf
- Brown's thesis: http://reports-archive.adm.cs.cmu.edu/anon/2020/CMU-CS-20-132.pdf
- Noam Brown's reference solver: https://github.com/noambrown/poker_solver
