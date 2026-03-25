# Poker Solver — Context for Next Session

**Last updated**: 2026-03-25 (blueprint re-run complete, convergence benchmarked)
**Project**: `C:/Users/Victor/Documents/Projects/poker-solver/`
**GitHub**: https://github.com/ZHANGV25/poker-solver.git
**Mission**: Build a Pluribus-equivalent (or better) 6-max NLHE solver — faster real-time search via GPU, full 6-player blueprint, production HUD integration.

---

## What Exists Right Now

### Blueprint (re-generated 2026-03-25, 1,755 .bps files on S3)
- **S3**: `s3://poker-blueprint-2026/worker-{0..19}/` — 1,755 .bps files (~140 MB each)
- **Format**: Binary .bps (LZMA-1 compressed strategies + JSON metadata)
- **Generated with**: 20× c5.4xlarge on-demand, 1M iterations per texture, 200 buckets, `*10` regret scaling
- **Root strategies (metadata JSON)**: CORRECT — 200/200 buckets non-uniform for every texture. Used via `bp_get_strategy` / `get_root_strategy()`.
- **Binary uint8 strategies (bulk export)**: ~99.8% uniform (1/N) for non-root info sets. This is expected — 6-player game tree has ~13M info sets per texture, 1M external-sampling iterations visits most nodes <10 times. NOT a code bug; verified via convergence benchmarking (see below).
- **Usable for**: First flop action lookup, root-level range narrowing, leaf value priors (uniform is a safe default for GPU search).
- **NOT usable for**: Deep-tree strategy lookup without GPU re-solve.

### Convergence Benchmark Results (AAA rainbow texture, c5.4xlarge)
| Iterations | Info Sets | Flop Non-Uniform % | Time |
|-----------|-----------|-------------------|------|
| 50K | 690K | 0.01% | 8.5s |
| 200K | 2.7M | 0.04% | 22s |
| 1M | 13.1M | 0.2% | 90s |
| 5M | >21M (OOM at 32GB) | N/A | N/A |

Root strategies via `bp_get_strategy` are non-uniform at ALL iteration counts and at 1-2 levels deep. The issue is only in the bulk binary export which iterates all 13M info sets (most unvisited).

### Key Finding: Per-Texture vs Pluribus Approach
Pluribus used ONE unified solve over the entire game tree for ~12,400 CPU-hours. Our per-texture isolation with ~0.5 CPU-hours each produces strong root strategies but weak deeper ones. This is a fundamental compute gap, not a code bug. The GPU real-time search (67ms river, 236ms flop) is designed to fill this gap — same architecture as Pluribus.

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

### 1. ~~Re-run Blueprint~~ DONE (2026-03-25)
Blueprint re-generated with fixed code. 1,755 .bps files on S3. Root strategies correct.
See "Convergence Benchmark Results" above for quality assessment.

### 2. Download Blueprint + Wire into HUD
- Download .bps files from S3 (or load on-demand via `blueprint_v2.py`)
- HUD should use `get_root_strategy()` for first flop action (reads metadata JSON — correct)
- Range narrowing at flop root: use root_strategies per bucket (correct)
- For deeper decisions: route to GPU real-time search

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

5. **More iterations for deeper convergence**: Tested 50K→1M→5M iterations. Flop non-uniform % barely grows (0.01%→0.2%). Tree grows faster than visits — 5M iterations creates 21M+ info sets and OOMs at 32 GB. Fundamental compute gap vs Pluribus.

6. **Fewer buckets (50, 10) for more visits per IS**: Marginal improvement (0.2%→6.3% at 10 buckets). Still 94% uniform. Tradeoff: less strategic precision per bucket.

7. **Regret scaling *50 instead of *10**: No improvement. Truncation is not the issue — visit count is.

8. **strategy_sum for all streets**: Actually WORSE — accumulates 1/N (uniform) at rarely-visited nodes, drowning out any signal from the few visits.

9. **c5.9xlarge (36 vCPU, 72 GB)**: Lock-free CAS hash table doesn't scale past 16 threads — 36 threads is slower than 16. No benefit.

10. **tmpfs /tmp on Amazon Linux 2023**: RAM-backed, only 16 GB. Caused OOM when accumulating .bps files. Fixed by incremental S3 upload + local delete, or writing to EBS.

---

## Performance Numbers (Measured on EC2 c5.4xlarge, 2026-03-25)

| Metric | Value |
|--------|-------|
| MCCFR iter/s (1 thread) | 35-50K |
| MCCFR iter/s (16 threads, lock-free CAS) | 27-30K effective |
| MCCFR iter/s (36 threads, c5.9xlarge) | 15-20K (WORSE — CAS contention) |
| EHS computation (1176 hands, 500 samples) | 1.8s |
| Info sets per texture (1M iter) | ~13M |
| Blueprint file size (LZMA-1) | ~140 MB per texture |
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
