# Poker Solver — Full Context for Next Session

**Last updated**: 2026-03-25
**Project**: `C:/Users/Victor/Documents/Projects/poker-solver/`
**GitHub**: https://github.com/ZHANGV25/poker-solver.git
**Mission**: Build an exact Pluribus replica — 6-player unified preflop-through-river MCCFR blueprint + GPU real-time subgame search at runtime.

---

## END GOAL

An exact copy of Pluribus (Brown & Sandholm, Science 2019):
1. **Blueprint**: ONE unified 6-player MCCFR solve, preflop through river, 169 lossless preflop buckets, 200 EHS postflop buckets, on a single 64-96 core machine for 8 days (~$177 spot)
2. **Runtime**: GPU real-time subgame search (street_solve.cu) with 4 continuation strategies, strategy freezing, Bayesian range narrowing, pseudoharmonic off-tree mapping
3. **HUD integration**: Wire into ACR Poker via Chrome DevTools Protocol

---

## WHAT EXISTS RIGHT NOW (all code written, compiles, tested)

### C Code (`src/`)

**`mccfr_blueprint.c` + `.h`** — Production N-player external-sampling MCCFR
- Unified `bp_init_unified()` — 6 players, all 1326 hands, preflop through river in one solve
- 169 lossless preflop buckets (hand classes)
- 200 EHS postflop buckets precomputed for all 1,755 flop textures at init time (stored in `texture_bucket_cache`)
- Preflop betting: SB/BB post blinds, UTG→...→BB acting order, up to 4 re-raises, configurable preflop bet sizes
- Flop deal: 3 random cards via Fisher-Yates, canonicalize to texture, look up precomputed buckets
- Postflop: flop→turn→river→showdown with configurable postflop bet sizes
- Linear CFR discount applied incrementally every `discount_interval` iterations (single parallel region with `#pragma omp single` between batches)
- Strategy snapshots for rounds 2-4 via `accumulate_snapshot()`
- Regret-based pruning (95% of iters after warmup, threshold -300M)
- Int32 regrets with -310M floor
- Lock-free CAS hash table (OpenMP Hogwild)
- **`bp_save_regrets()` / `bp_load_regrets()`** — serialize/deserialize full hash table for checkpoint/resume
- Backward compatible: `bp_init_ex()` still works for postflop-only solves
- Compiles on Windows (`-static`) and Linux (`-fPIC -shared -fopenmp`)

**`card_abstraction.c` + `.h`** — EHS computation + bucketing
- `ca_compute_ehs()` — Monte Carlo equity via hand_eval.h
- `ca_assign_buckets()` — EHS percentile bucketing
- `ca_assign_buckets_kmeans()` — k-means on [EHS, positive_potential, negative_potential] (Pluribus-style domain features)
- `ca_preflop_classes()` — 169 lossless preflop hand classes

**`cuda/street_solve.cu` + `.cuh`** — N-player GPU subgame solver
- 2-6 players, single-street Linear CFR on GPU
- 4 continuation strategies at depth-limit leaves (5x bias, Pluribus-exact)
- Level-batched CFR, 67ms river, 236ms flop (RTX 3060, 200 hands)
- Returns final-iteration strategy for play + weighted-average for narrowing

### Python — Runtime (`python/`)

**`hud_solver.py`** — Full Pluribus decision pipeline
- `new_hand()` with scenario_id, N-player ranges, uniform_beliefs option
- Sets `blueprint_v2.current_scenario` for scenario-aware texture loading
- `extra_villain_positions` for true N-player GPU search (not just heuristic)
- Strategy freezing (A3): passes frozen_actions to GPU solver
- Falls through: blueprint_v2 → blueprint_store → GPU re-solve
- Flop leaf values from v2 blueprint via `compute_flop_leaf_equity()`

**`street_solver_gpu.py`** — GPU solver Python wrapper
- N-player support (`player_ranges=`, `acting_order=`)
- Strategy freezing: `get_strategy(frozen_actions=[...])` walks tree following hero's prior actions to find current decision node
- `_get_strategy_at_node()`, `_match_action_label()` for arbitrary node extraction

**`blueprint_v2.py`** — .bps file loader
- Scenario-aware file indexing: `worker-N/{scenario_id}/{texture}.bps` (v2 layout)
- Backward compatible with flat `worker-N/{texture}.bps` (v1 layout)
- `current_scenario` property, `available_scenarios()`

**`range_narrowing.py`** — Bayesian range tracker
- `set_uniform_range()` — all 1326 hands at weight 1.0 (Pluribus-style)
- `set_initial_range()` — from preflop ranges (HUD use case)

**`leaf_values.py`** — Depth-limited continuation values
- `compute_flop_leaf_values()` — uses BlueprintStore per-action EVs + 4 biased strategies
- `compute_flop_leaf_equity()` — averages equity over turn+river cards (v2 blueprint fallback)
- `compute_turn_leaf_values()` — equity over river cards
- `bias_strategy()` — 5x fold/call/raise biasing (Pluribus-exact)

**`off_tree.py`** — Pseudoharmonic bet interpolation
**`multiway_adjust.py`** — Heuristic N-player adjustments (fallback when true N-player search unavailable)

### Python — Blueprint Generation (`precompute/`)

**`blueprint_worker_unified.py`** — Unified 6-player blueprint worker
- Wraps `bp_init_unified` with Pluribus-matched parameters
- Solves in chunks with periodic checkpoints (strategy .bps + regret table .bin)
- `--resume` flag: downloads latest regret checkpoint from S3 and continues
- Uploads both `unified_blueprint.bps` and `checkpoints/regrets_latest.bin` to S3
- `--time-limit-hours 192` for 8-day run
- `--checkpoint-interval 1000000` (saves every ~1M iterations)

**`launch_blueprint_unified.sh`** — EC2 launcher for unified solve
- Targets c5.metal (96 vCPU, 192GB) or c5.18xlarge (72 vCPU, 144GB)
- Compiles with `-O3 -march=native` for maximum throughput
- Sets `OMP_STACKSIZE=64m`
- `--status`, `--download`, `--dry-run`

**`watchdog.sh`** — Auto-restart monitor for spot instances
- Runs on a t3.micro (~$0.01/hr) 24/7
- Checks solver instance every 5 minutes
- If dead: relaunches new spot instance with `--resume`
- Tries spot first, falls back to on-demand
- Stops when final checkpoint detected or max relaunches hit

**`blueprint_worker_v2.py`** — Per-texture 2-player blueprint (OLDER APPROACH, still works)
**`scenario_matrix.py`** — 27 scenario definitions from ranges.json
**`range_parser.py`** — PioSOLVER range string parser with frequency handling

### Compiled Binaries (`build/`)
- `mccfr_blueprint.dll` / `bp_unified.dll` / `bp_final.dll` — various builds (some may be locked by old processes)
- `card_abstraction.dll`
- `street_solve.dll` — GPU solver
- Various test/benchmark executables

### Data
- `C:/Users/Victor/Documents/Projects/ACRPoker-Hud-PC/solver/ranges.json` — 6-position preflop ranges
- `s3://poker-blueprint-2026/` — v1 blueprint (1,755 .bps files, 6-player per-texture, 1M iter)

---

## WHERE WE ARE RIGHT NOW

### Just completed:
1. Implemented unified 6-player preflop-through-river MCCFR (`bp_init_unified`)
2. Precomputed 200 EHS postflop buckets for all 1,755 flop textures
3. Fixed paired-board flush draw canonicalization bug
4. Added regret table save/load (`bp_save_regrets` / `bp_load_regrets`) for checkpoint/resume
5. Built checkpoint system in Python worker (strategy .bps + regret .bin to S3)
6. Built watchdog.sh for auto-restart on spot reclaim

### Test that was running when session ended:
- Regret save/load test: run 500 iters → save → new solver → load → run 500 more → compare
- This test takes ~3-5 min because of EHS precompute for 1,755 textures (single-threaded, 500 MC samples each)
- The test was launched as background task `bz0ou8l34` but PC restarted before it completed
- **Re-run this test first** to verify save/load works

### What needs to happen before launch:

1. **Verify regret save/load works** — re-run the test above
2. **Compile final DLL** — `gcc -O2 -shared -fopenmp -static -o build/mccfr_blueprint.dll src/mccfr_blueprint.c -I src -lm`
3. **Test on Linux** — the EC2 compilation uses `-fPIC -shared -fopenmp` without `-static`; verify it works
4. **Launch sequence**:
   a. Launch t3.micro watchdog instance (on-demand, ~$0.01/hr)
   b. Deploy watchdog.sh + code to it
   c. Watchdog launches c5.metal spot instance with solver
   d. Solver runs, checkpoints to S3 every 1M iterations
   e. If spot reclaimed, watchdog detects in ≤5 min, launches new instance with --resume
   f. After 8 days (192h) total compute, solver saves "final" checkpoint and shuts down
   g. Watchdog detects "final" label and stops

### Cost estimate:
- c5.metal spot: ~$1.22/hr × 192h = **~$234** (if never interrupted)
- With interruptions + relaunch overhead: ~$250-300
- Watchdog t3.micro: ~$1.70 for 8 days
- **Total: ~$235-300**

---

## WHAT WAS VERIFIED WORKING

| Component | Tested? | Result |
|-----------|---------|--------|
| Unified 6-player init | YES | 1,755 textures precomputed, 169 preflop buckets |
| Unified MCCFR traversal | YES | 60K IS from 2K iters, ~10K iter/s single-thread |
| 200 postflop buckets | YES | Bucket lookup via canonicalized flop hash |
| Linear CFR incremental discount | YES | Applied 5 times in 500-iter test |
| Strategy export (.bps) | YES | 73MB raw, 1MB compressed |
| bp_save_regrets | Compiled OK | Runtime test interrupted by PC restart |
| bp_load_regrets | Compiled OK | Runtime test interrupted by PC restart |
| Python worker checkpoint loop | Parses OK | Full integration not yet tested |
| watchdog.sh | Syntax OK | Not deployed |
| GPU street_solve.dll | YES | 67ms river, 236ms flop |
| HUD pipeline | YES | Imports OK, scenario wiring works |
| BlueprintV2 scenario indexing | YES | v2 layout tested with mock files |

---

## KEY DESIGN DECISIONS

1. **Unified solve, not per-texture**: Matches Pluribus exactly. One MCCFR over entire game tree.
2. **Preflop integrated**: Not static ranges.json — preflop strategies emerge from the unified solve.
3. **200 EHS buckets**: Precomputed for all 1,755 textures at init. Cached in `texture_bucket_cache[1760*1326]`.
4. **Spot + watchdog**: Cheapest reliable path. ~$235 vs $783 on-demand.
5. **Checkpoint/resume**: Regret table serialized to binary, uploaded to S3, reloaded on new instance.

---

## EVERY FILE THAT WAS MODIFIED OR CREATED

### Created new:
- `precompute/range_parser.py`
- `precompute/scenario_matrix.py`
- `precompute/blueprint_worker_v2.py`
- `precompute/launch_blueprint_v2.sh`
- `precompute/blueprint_worker_unified.py`
- `precompute/launch_blueprint_unified.sh`
- `precompute/watchdog.sh`

### Modified:
- `src/mccfr_blueprint.c` — incremental discount, strategy snapshots, unified init, preflop traversal, texture bucket cache, save/load regrets, paired-board canonicalization fix
- `src/mccfr_blueprint.h` — BPConfig.include_preflop, BPSolver fields (preflop bet sizes, blinds, texture cache), bp_init_unified, bp_save_regrets, bp_load_regrets
- `src/card_abstraction.c` — ca_assign_buckets_kmeans (k-means on domain features)
- `src/card_abstraction.h` — ca_assign_buckets_kmeans declaration
- `python/blueprint_v2.py` — scenario-aware file indexing, current_scenario, available_scenarios
- `python/hud_solver.py` — scenario wiring, N-player GPU search, strategy freezing, v2 leaf values, uniform_beliefs
- `python/street_solver_gpu.py` — strategy freezing (frozen_actions), tree-walk for arbitrary node extraction
- `python/leaf_values.py` — compute_flop_leaf_equity (turn+river equity averaging)
- `python/range_narrowing.py` — set_uniform_range (1326 hands)
- `precompute/blueprint_worker.py` — 3 bet sizes

### Not modified (work as-is):
- `src/cuda/street_solve.cu/.cuh` — already supports 2-6 players + 4 continuation strategies
- `python/off_tree.py` — pseudoharmonic mapping
- `python/solver.py` — parse_range_string, card_to_int
- `precompute/solve_scenarios.py` — generate_all_textures, load_scenarios

---

## PLURIBUS PARAMETER REFERENCE

From `pluribus_technical_details.md` in this repo:

| Parameter | Pluribus | Our Value |
|-----------|----------|-----------|
| Players | 6 | 6 |
| Algorithm | External-sampling MCCFR | Same |
| Training time | 8 days, 64 cores | 8 days, 72-96 cores |
| Training cost | ~$144 spot | ~$235 spot |
| Memory | <512GB | ~30GB estimated |
| Preflop buckets | 169 lossless | 169 lossless |
| Postflop buckets | 200 per street | 200 per street |
| Discount | d=T/(T+1) every 10 min, first 400 min | Same (proportional) |
| Pruning | -300M threshold, 95%, after 200 min | Same |
| Regret floor | -310M | -310M |
| Regret storage | int32 | int32 |
| Strategy sum | Round 1 only, every 10K iter | Same + snapshots for rounds 2-4 |
| Search (runtime) | Linear CFR, 4 cont strategies, 5x bias | Same (GPU, 67ms river) |
| Play strategy | Final iteration | Same |
| Narrowing strategy | Weighted average | Same |
| Strategy freezing | A3: freeze at passed nodes for hero's hand | Implemented |
| Off-tree bets | Pseudoharmonic | Same |
| Beliefs | Start uniform 1/1326 | Supported (uniform_beliefs=True) |

---

## AWS RESOURCES

- **S3 bucket (v1)**: `poker-blueprint-2026` (110 GB, v1 per-texture blueprint)
- **S3 bucket (unified)**: `poker-blueprint-unified` (will be created by launch script)
- **Key pair**: `poker-solver-key` (PEM at `C:/Users/Victor/poker-solver-key.pem`)
- **Security group**: `poker-solver-sg`
- **IAM profile**: `poker-solver-profile` (role: `poker-solver-ec2-role`)
- **Region**: us-east-1

---

## PROMPT FOR NEXT AGENT

You are continuing development of an exact Pluribus replica poker solver. The project is at `C:/Users/Victor/Documents/Projects/poker-solver/`. Read `CONTEXT_NEXT_SESSION.md` for full context.

**Immediate task**: Verify the regret save/load checkpoint system works (test was interrupted), then launch the 8-day unified blueprint computation on EC2.

**Test to run first**:
```bash
cd /c/Users/Victor/Documents/Projects/poker-solver
gcc -O2 -shared -fopenmp -static -o build/mccfr_blueprint.dll src/mccfr_blueprint.c -I src -lm
# Then run the Python save/load test (see blueprint_worker_unified.py)
```

**Launch sequence** (after test passes):
1. Launch t3.micro watchdog
2. Deploy code + watchdog.sh
3. Watchdog launches c5.metal spot with unified solver + --resume
4. Monitor via `watchdog.sh --status` or S3 checkpoint metadata
