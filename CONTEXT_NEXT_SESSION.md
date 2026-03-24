# Poker Solver — Context & Status

**Last updated**: 2026-03-24 (end of session)
**Project**: `C:/Users/Victor/Documents/Projects/poker-solver/`
**GitHub**: https://github.com/ZHANGV25/poker-solver.git

---

## Current State: Blueprint Pipeline Ready, Preflop Integration Next

### What Was Accomplished This Session

1. **GPU MCCFR Research (Novel)**: Built batch outcome-sampling MCCFR on GPU.
   Ran 6-player with 10 equity buckets on RTX 3060 (18.8M nodes, 7.7GB VRAM).
   **Finding**: GPU outcome sampling is ~5-10x slower to converge than CPU external
   sampling per wall-clock second. The variance per trajectory is too high.
   **Conclusion**: GPU is for real-time subgame search, CPU for blueprint.

2. **Production MCCFR Blueprint Solver**: Rewrote `mccfr_blueprint.c` with:
   - OpenMP Hogwild parallelism (lock-free shared regret tables)
   - Int32 regrets with -310M floor (matches Pluribus)
   - Regret-based pruning at -300M threshold (95% of iterations)
   - Linear CFR discount (first 35% of iterations, then stop)
   - Card abstraction support (hand-to-bucket mapping per street)
   - **Fixed 3 critical payoff bugs** (cumulative investment tracking,
     raise double-counting, stack decrementing)

3. **Card Abstraction Pipeline**: Built `card_abstraction.c`:
   - Fast C-based EHS computation (1176 hands in 2s @ 500 samples)
   - Percentile bucketing into 200 buckets
   - 169 lossless preflop hand classes

4. **EC2 Launch Infrastructure**: Built parallel blueprint generation:
   - `precompute/blueprint_worker.py` — per-instance worker
   - `precompute/launch_blueprint.sh` — launches 20 EC2 spot instances
   - Tested locally: 6-player, 200 buckets, 50K iter in 3s/texture

5. **Pluribus Technical Details**: Complete extraction of every parameter
   from the Science 2019 supplementary materials.

### What Was Tried and Abandoned

- **GPU MCCFR for blueprints**: Outcome sampling on GPU produces 17M traj/s
  but convergence is poor for large trees (9M+ decision nodes). Each info set
  is visited ~0.001 times per trajectory. External sampling on CPU is ~1000x
  more sample-efficient per info set.
- **Linear CFR discount for GPU MCCFR**: Global discount destroys information
  at unvisited info sets. Switched to CFR+ (regret floor at 0). Still worse
  than CPU external sampling.

### Critical Bugs Found and Fixed

1. **`bets[]` reset between streets but `pot` accumulates**: Showdown payoff
   only subtracted last street's bet, not total investment. Made aggression
   appear cheap → always all-in convergence.
2. **Raise action `bets[ap] = mx + amount`**: `amount` includes `to_call`,
   so the bet was inflated. Fixed to `bets[ap] += amount`.
3. **Stack never decremented**: Players could bet full stack repeatedly.
   Fixed with per-player `stacks[]` that decrements on each action.

---

## Architecture (Updated)

```
PHASE 1 — PREFLOP BLUEPRINT (TODO: not yet built)
  6-player preflop CFR+ over 169 hand classes
  Solves all positions simultaneously (not pairwise)
  Outputs: mixed-frequency P(action | hand, position, history)
  Used to derive starting ranges for each flop texture

PHASE 2 — POSTFLOP BLUEPRINT (CPU, EC2, ready to run)
  mccfr_blueprint.c — 6-player external-sampling MCCFR
  card_abstraction.c — 200-bucket EHS abstraction
  Per flop texture: 6-player, 200 buckets, ~1M iterations
  1,755 textures parallelized across 20 EC2 instances
  Estimated: ~2-4 hours total, ~$15-30 cost

PHASE 3 — REAL-TIME SEARCH (GPU, local, working)
  street_solve.cu — N-player single-street GPU solver
  67ms river, 236ms flop (200 hands, 200 iterations, RTX 3060)
  4 continuation strategies at depth-limit leaves
  Blueprint leaf values feed into search
```

---

## File Manifest (Updated)

### CUDA Solvers
| File | Purpose | Status |
|------|---------|--------|
| `src/cuda/street_solve.cu` | N-player single-street GPU solver | DONE |
| `src/cuda/gpu_mccfr.cu` | Batch outcome-sampling MCCFR (research) | DONE (not for production) |
| `src/cuda/flop_solve.cu` | 2-player full-tree GPU solver | DONE |

### C Solvers
| File | Purpose | Status |
|------|---------|--------|
| `src/mccfr_blueprint.c` | **Production 6-player MCCFR** (OpenMP, int32 regrets, pruning) | DONE |
| `src/card_abstraction.c` | Fast EHS + percentile bucketing | DONE |
| `src/preflop_solver.c` | 2-player preflop CFR+ (legacy) | Working but needs 6-player upgrade |
| `src/solver_v2.c` | 2-player CPU multi-street solver | Working |

### Python Layer
| File | Purpose | Status |
|------|---------|--------|
| `python/gpu_mccfr.py` | GPU MCCFR wrapper (research) | DONE |
| `python/hud_solver.py` | HUD interface | DONE |
| `python/street_solver_gpu.py` | GPU solver wrapper | DONE |
| `python/blueprint_store.py` | Binary blueprint storage | DONE |
| `python/leaf_values.py` | Continuation leaf values | DONE (2-player) |

### Precompute
| File | Purpose | Status |
|------|---------|--------|
| `precompute/blueprint_worker.py` | Per-instance texture solver | DONE |
| `precompute/launch_blueprint.sh` | EC2 multi-instance launcher | DONE |
| `precompute/solve_scenarios.py` | 1,755 texture generation | Working |

### Build Artifacts
| File | Source |
|------|--------|
| `build/street_solve.dll` | street_solve.cu |
| `build/gpu_mccfr.dll` | gpu_mccfr.cu |
| `build/mccfr_blueprint.dll` | mccfr_blueprint.c (with OpenMP) |
| `build/card_abstraction.dll` | card_abstraction.c |

---

## Remaining Work (Priority Order)

### High Priority
1. **6-player preflop solver** — Extend preflop_solver.c from 2-player to 6-player.
   All positions solve simultaneously. 169 hand classes. Outputs mixed frequencies.
2. **Wire preflop → postflop** — Preflop strategy → starting ranges for each texture.
3. **Run EC2 blueprint generation** — 1,755 textures, 200 buckets, ~2M iterations each.
4. **Wire blueprint into GPU search** — leaf_values from blueprint into street_solve.cu.

### Medium Priority
5. **Per-street re-bucketing** — Currently flop EHS used for all streets.
   Pluribus re-buckets when turn/river dealt.
6. **Blueprint serialization** — Save to BlueprintStore binary format.
7. **Exploitability measurement** — Compare strategies vs postflop-solver (Rust).

### Lower Priority
8. **Strategy freezing (A3)** — CUDA frozen_mask enforcement.
9. **Warm-start (A4)** — Persist regrets between solves.
10. **Full turn/river equity kernel (C1)** — All 46 cards for turn leaves.

---

## Key Research Findings

### GPU vs CPU for Blueprint MCCFR
- GPU batch outcome sampling: 17M traj/s but ~1000x worse sample efficiency
- CPU external sampling: 44K iter/s single-thread but each iteration visits
  ~100+ info sets (vs GPU's ~10 per trajectory)
- **Verdict**: CPU wins for blueprints. GPU wins for real-time search.

### Exploitability Targets
- < 0.5% pot = PioSOLVER default (adequate)
- < 0.3% pot = high quality (GTO Wizard standard)
- < 0.1% pot = near-perfect
- Pluribus: not measured (intractable for 6-player), evaluated by win rate vs pros

### Commercial Solver Landscape
- PioSOLVER/GTO+: **2-player postflop only**, no preflop
- MonkerSolver: 6-player preflop, 3+ postflop with abstraction (RAM-hungry)
- GTO Wizard: 9-player preflop (2026), 3-way postflop (2025)
- **Nobody except Pluribus does full 6-player end-to-end**

### Player Count Limits
- `BP_MAX_PLAYERS = 6` in mccfr_blueprint.h (configurable)
- `SS_MAX_PLAYERS = 6` in street_solve.cuh
- `GM_MAX_PLAYERS = 6` in gpu_mccfr.cuh
- These are compile-time constants. Increasing to 9 is straightforward
  but tree size grows exponentially with players.

---

## Hardware & Build

- **CPU**: i7-13700K, **RAM**: 64GB, **GPU**: RTX 3060 12GB
- **OS**: Windows 10 Pro, **Shell**: Git Bash (MSYS2)
- **CUDA**: 11.8, **GCC**: 14.2.0 (MSYS2 UCRT64)
- **Python**: 3.9

### Compile commands:
```bash
# MCCFR blueprint (with OpenMP)
gcc -O2 -shared -fopenmp -static -o build/mccfr_blueprint.dll src/mccfr_blueprint.c -I src -lm

# Card abstraction
gcc -O2 -shared -o build/card_abstraction.dll src/card_abstraction.c -I src -lm

# GPU street solver (NVCC)
nvcc -O2 --shared -o build/street_solve.dll src/cuda/street_solve.cu ...

# GPU MCCFR (research)
nvcc -O2 --shared -o build/gpu_mccfr.dll src/cuda/gpu_mccfr.cu ...
```
