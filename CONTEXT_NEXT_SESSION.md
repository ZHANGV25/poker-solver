# Poker Solver — Context for Next Session

**Last updated**: 2026-03-24 (end of marathon session)
**Project**: `C:/Users/Victor/Documents/Projects/poker-solver/`
**GitHub**: https://github.com/ZHANGV25/poker-solver.git
**Mission**: Build a Pluribus-equivalent (or better) 6-max NLHE solver — faster real-time search via GPU, full 6-player blueprint, production HUD integration.

---

## What Exists Right Now

### Blueprint (110 GB on S3 — NEEDS RE-RUN)
- **S3**: `s3://poker-blueprint-2026/worker-{0..19}/` — 1,755 .bps files
- **Format**: Binary .bps (LZMA compressed strategies + JSON metadata)
- **Problem**: The uint8 strategy data is **mostly garbage** due to int32 regret overflow (`*1000` scaling). Fixed in code but not re-generated.
- **What IS correct**: The JSON metadata in each .bps file has valid `root_strategies` (extracted via `bp_get_strategy` before the buggy export)
- **What to do**: Re-run EC2 generation with fixed code (~$12, ~50 min). The fix: `*10` scaling instead of `*1000`, export uses `regret_match_int` directly.

### Code (all compiled, tested, pushed to GitHub)

| File | What | Status |
|------|------|--------|
| `src/mccfr_blueprint.c/.h` | Production 6-player MCCFR | DONE — OpenMP lock-free CAS, int32 regrets, pruning, card abstraction, cumulative investment tracking |
| `src/card_abstraction.c/.h` | Fast C EHS + 200-bucket percentile | DONE — 1176 hands in 2s |
| `src/cuda/street_solve.cu/.cuh` | N-player GPU subgame solver | DONE — 67ms river, 236ms flop |
| `src/cuda/gpu_mccfr.cu/.cuh` | GPU batch outcome-sampling MCCFR | DONE (research — not for production) |
| `python/blueprint_v2.py` | .bps loader with hash lookup | DONE — loads flop info sets, canonical board mapping |
| `python/hud_solver.py` | HUD pipeline with blueprint_v2 wired in | DONE — falls through v2 → old blueprint → GPU search |
| `python/street_solver_gpu.py` | GPU solver ctypes wrapper | DONE |
| `python/range_narrowing.py` | Bayesian range tracker | DONE |
| `python/off_tree.py` | Pseudoharmonic bet mapping | DONE |
| `precompute/blueprint_worker.py` | EC2 worker (solve + export .bps) | DONE |
| `precompute/launch_blueprint.sh` | EC2 fleet launcher | DONE |
| `build/*.dll` | All compiled for Windows | DONE |

### Build Commands
```bash
# MCCFR (OpenMP, static)
gcc -O2 -shared -fopenmp -static -o build/mccfr_blueprint.dll src/mccfr_blueprint.c -I src -lm

# Card abstraction
gcc -O2 -shared -o build/card_abstraction.dll src/card_abstraction.c -I src -lm

# GPU street solver
nvcc -O2 --shared -o build/street_solve.dll src/cuda/street_solve.cu -I src/cuda ...

# Linux (EC2) — add -fPIC
gcc -O2 -fPIC -shared -fopenmp -o build/mccfr_blueprint.so src/mccfr_blueprint.c -I src -lm -lpthread
```

### AWS Resources
- **S3 bucket**: `poker-blueprint-2026` (110 GB blueprint data)
- **Key pair**: `poker-solver-key` (PEM at `C:/Users/Victor/poker-solver-key.pem`)
- **Security group**: `poker-solver-sg`
- **IAM profile**: `poker-solver-profile` (role: `poker-solver-ec2-role`)
- **Region**: us-east-1

---

## What Needs to Happen (Priority Order)

### 1. Re-run Blueprint on EC2 (~$12, ~50 min)
The code is fixed. Just need to upload and launch:
```bash
# Upload fixed code
aws s3 sync src/ s3://poker-blueprint-2026/code/src/ --exclude "*.dll" --exclude "*.exe"
aws s3 sync precompute/ s3://poker-blueprint-2026/code/precompute/
aws s3 sync python/ s3://poker-blueprint-2026/code/python/ --exclude "__pycache__/*"

# Launch 20 on-demand c5.4xlarge instances
# (see precompute/launch_blueprint.sh for the full script)
# Workers: 500K iterations, 200 buckets, 6 players, OMP_STACKSIZE=16m
```
**Key fixes in the new code:**
- Regret delta: `(int)((action_values[a] - node_value) * 10.0f)` (was `*1000` → overflow)
- Export: always uses `regret_match_int` (was using empty `strategy_sum`)
- `strategy_interval = 1` (was 10000 — too infrequent)
- `#include <stddef.h>` for `size_t` (was missing → EC2 compile fail)

### 2. Download Blueprint (~3-5 GB flop-only subset)
The full 110 GB is on S3. For the HUD, only flop info sets are needed (~2-3 MB per texture × 1755 = ~4 GB). The loader (`blueprint_v2.py`) filters by street tag during read.

### 3. Wire Leaf Values into GPU Search
The GPU street solver (`street_solve.cu`) needs **continuation leaf values** at depth-limit boundaries. These come from the blueprint:
- **Flop search leaves** (at turn boundary): need blueprint P(action|bucket) at turn root → 4 continuation strategies (unmodified, fold×5, call×5, raise×5)
- **Turn search leaves** (at river boundary): need blueprint P(action|bucket) at river root

Currently `street_solve.cu` accepts `leaf_values` as input. The Python wrapper needs to compute these from the blueprint and feed them in.

### 4. End-to-End HUD Test
Run the full pipeline on an actual ACR table:
1. Detect game state (board, actions, positions) via Chrome DevTools Protocol
2. Look up blueprint strategy for first flop action
3. On opponent action: narrow ranges using blueprint P(action|bucket)
4. On hero's turn: GPU re-solve subgame from street root
5. Display recommended strategy

---

## Key Architecture Decisions Made

### Blueprint: Per-Texture, Not Unified
Pluribus runs ONE MCCFR over the entire game. We run **separate MCCFR per flop texture** (1,755 textures). This is an approximation but:
- Parallelizes perfectly across EC2 instances
- Each instance needs only 16-30 GB RAM (vs Pluribus's 512 GB)
- Cost: ~$12 vs Pluribus's ~$144

### GPU for Search, CPU for Blueprint
Attempted GPU batch outcome-sampling MCCFR (novel research). Finding: CPU external sampling converges ~5-10x faster per wall-clock. GPU is for real-time subgame search only.

### Card Abstraction: 200 EHS Buckets
Matches Pluribus. Fast C-based EHS computation (2s per texture). Percentile bucketing. All players share the same bucket pool per texture.

### Payoff Model: Cumulative Investment Tracking
Fixed 3 critical bugs: (1) bets[] reset between streets but pot accumulated, (2) raise double-counted to_call, (3) stack never decremented. Now uses `invested[]` array that persists across streets.

### Hash Table: Lock-Free CAS
Replaced `#pragma omp critical` (global lock → 15x slowdown) with per-slot atomic CAS. Three-state protocol: 0=empty, 1=ready, 2=initializing. Near-linear thread scaling to 16 cores.

---

## What We Tried and Abandoned

1. **GPU MCCFR for blueprints**: 17M traj/s but ~1000x worse sample efficiency than CPU external sampling. Outcome sampling variance too high for large game trees (9M+ info sets).

2. **Linear CFR discount as global post-hoc step**: Doesn't work — needs to be applied incrementally during training. Currently applies only once at the end (minimal effect).

3. **c5.2xlarge EC2 instances**: Only 16 GB RAM → OOM with 64M hash table. Need c5.4xlarge (32 GB) minimum.

4. **Spot instances for blueprint**: Too volatile — 16/20 reclaimed in one run. Use on-demand for reliability (~2x cost but no re-runs).

---

## Performance Numbers (Measured on EC2 c5.4xlarge)

| Metric | Value |
|--------|-------|
| MCCFR iter/s (1 thread) | 35-50K |
| MCCFR iter/s (16 threads, lock-free CAS) | 24-30K effective |
| EHS computation (1176 hands, 500 samples) | 1.8s |
| Info sets per texture (500K iter) | ~4.8M |
| Blueprint file size (LZMA) | ~60 MB per texture |
| GPU street solve (river, 200 hands) | 67ms |
| GPU street solve (flop, 200 hands) | 236ms |

---

## File Locations

```
C:/Users/Victor/Documents/Projects/poker-solver/    # main repo
├── src/
│   ├── mccfr_blueprint.c/.h     # production MCCFR
│   ├── card_abstraction.c/.h    # EHS + bucketing
│   ├── hand_eval.h              # 7-card evaluator
│   └── cuda/
│       ├── street_solve.cu/.cuh # GPU subgame solver
│       └── gpu_mccfr.cu/.cuh   # GPU MCCFR (research)
├── python/
│   ├── blueprint_v2.py          # .bps loader
│   ├── hud_solver.py            # HUD pipeline
│   ├── street_solver_gpu.py     # GPU solver wrapper
│   ├── range_narrowing.py       # Bayesian range tracker
│   └── off_tree.py              # pseudoharmonic mapping
├── precompute/
│   ├── blueprint_worker.py      # EC2 worker
│   ├── launch_blueprint.sh      # fleet launcher
│   └── solve_scenarios.py       # texture generation
├── build/                        # compiled DLLs
├── blueprint_output/             # downloaded .bps files (partial)
├── pluribus_technical_details.md # every Pluribus parameter
├── BLUEPRINT_PRECOMPUTE.md       # precompute architecture
└── ARCHITECTURE.md               # overall architecture

C:/Users/Victor/Documents/Projects/ACRPoker-Hud-PC/  # HUD app
├── solver/solver-cli/target/release/tbl-engine.exe  # Rust postflop-solver
├── solver/ranges.json                                # preflop ranges
└── src/cdp_reader.py                                # game state reader
```

---

## Prompt for Next Agent

You are continuing development of a Pluribus-equivalent 6-max NLHE poker solver. The project is at `C:/Users/Victor/Documents/Projects/poker-solver/`. Read `CONTEXT_NEXT_SESSION.md` for full context.

**Immediate task**: Re-run the EC2 blueprint generation with the fixed regret scaling code, verify the exported strategies are non-uniform, download the flop-only subset (~4 GB), and test the full HUD pipeline end-to-end (blueprint lookup → range narrowing → GPU search → strategy output).

**The specific bugs that were fixed but not re-deployed:**
1. `src/mccfr_blueprint.c` line 535: regret delta uses `*10.0f` (was `*1000.0f` → int32 overflow → all-zero regrets)
2. `src/mccfr_blueprint.c` export function: always uses `regret_match_int` (was using empty `strategy_sum` → uniform strategies)
3. `precompute/blueprint_worker.py`: `strategy_interval=1` (was 10000)
4. `src/mccfr_blueprint.h`: `#include <stddef.h>` for `size_t`

**After the blueprint is regenerated and verified:**
1. Wire blueprint continuation strategies into `street_solve.cu` leaf values
2. Test on actual ACR poker table via the HUD
3. Measure strategy quality vs Rust postflop-solver (tbl-engine.exe)

**Long-term goals:**
- Preflop solver (6-player simultaneous, not pairwise)
- Per-street re-bucketing (Pluribus re-buckets when turn/river dealt)
- Strategy freezing in GPU search (A3)
- Support 2-9 players (just change MAX_PLAYERS defines)
