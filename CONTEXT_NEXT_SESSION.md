# Poker Solver — Full Context for Next Session

**Last updated**: 2026-04-05
**Project**: `/Users/victor/Documents/Dev/poker-solver/`
**GitHub**: https://github.com/ZHANGV25/poker-solver-dev (private)
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
- Int32 regrets with -310M floor, 2B ceiling (Pluribus has no explicit ceiling)
- **`bp_get_regrets()`** — extract raw integer regrets for diagnostics
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
- Adaptive checkpoint schedule: 200M, 400M, 600M, 2B, 4B, 6B, 8B, 10B, then every 10B
- Lightweight probes every 50M: F/C/R splits for 19 hands (pairs + key broadways/SCs), raw regrets for 5 diagnostic hands, UTG + BTN positions
- `--resume` flag: downloads latest regret checkpoint from S3 and continues
- `--time-limit-hours 192` for 8-day run (iteration count auto-computed)

**`launch_blueprint_unified.sh`** — EC2 launcher for unified solve
- Targets c7a.metal-48xl (192 vCPU, 376GB) with 3B hash table
- Compiles with `-O2 -march=native -fopenmp`
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

## WHERE WE ARE RIGHT NOW (2026-04-05)

### COMPUTE IS RUNNING:
- **Solver:** `i-09ab6b2e26d0f9007` (c7a.metal-48xl, 192 cores, us-east-1)
- **S3 bucket:** `poker-blueprint-unified` — probes every 50M, adaptive checkpoints
- **Target:** Time-limited (192h = 8 days), ~138B iterations at ~200K iter/s
- **Cost estimate:** ~$998 on-demand
- **Checkpoint schedule:** 200M, 400M, 600M, 2B, 4B, 6B, 8B, 10B, 20B, 30B, 40B

### Blueprint convergence debugging (April 4-5, 2026)

After 10 bugs found and fixed, the solver runs correctly but **early positions
(UTG, MP) don't converge** while late positions (BTN, SB) do. TT/99 from UTG
converge to 100% call instead of raise. Deep research into the Pluribus
supplementary materials, Noam Brown's thesis, and open-source implementations
found two remaining algorithmic deviations (Bugs 9-10):

**Bug 9 (CRITICAL): Pruned actions' regrets updated instead of left unchanged.**
Pluribus Algorithm 1 only updates regrets for explored actions. Our code updated
ALL actions, giving pruned actions delta = -node_value, pushing them deeper negative
on 95% of iterations. This is the root cause of the call trap at UTG.

**Bug 10: Regret ceiling too low (310M vs ~2B).** Pluribus has no explicit ceiling.
Our 310M ceiling caused dominant actions to saturate 7x too early, losing ordering
information when multiple raise sizes competed.

Both fixed. Current run has the fixes + expanded probes + adaptive checkpoints.

### All bugs found and fixed (10 total):

| # | Bug | Impact | Phase |
|---|-----|--------|-------|
| 1 | board_hash in info set key | Tree inflated to billions, 27% miss rate | Phase 2 |
| 2 | int32 regret overflow | Nonsensical strategies at 9.87B iters | Phase 2 |
| 3 | Heap corruption from billions of callocs | Crash at ~575M iters | Phase 2 |
| 4 | Hash table fills (flat 8 preflop sizes) | Tree 19x larger than Pluribus | Phase 3 |
| 5 | int32 overflow in iteration counters | Can't run >2.1B iterations | Phase 3 |
| 6 | strategy_sum aliasing (gcd(6,10000)=2) | UTG/SB/CO never accumulate avg strategy | Phase 5 |
| 7 | Call/fold trap feedback loop | Diagnosed but not root-caused until Bug 9 | Phase 6 |
| 8 | 10x regret scaling factor | Regrets hit ceiling/floor 10x too fast | Phase 7 |
| 9 | **Pruned regrets updated (should be unchanged)** | **Root cause of call trap** | Phase 10 |
| 10 | Regret ceiling 310M (should be ~2B) | Dominant actions saturate too early | Phase 10 |

See `docs/BLUEPRINT_BUGS.md` for full details, `docs/BLUEPRINT_CHRONICLE.md` for narrative.

### Compilation (CHANGED — requires both .c files):
```bash
gcc -O2 -shared -fopenmp -static -o build/mccfr_blueprint.dll src/mccfr_blueprint.c src/card_abstraction.c -I src -lm
```

### Key learnings from first run:
- K-means texture precompute: ~45-60 min single-threaded (one-time)
- 8M iters → 300M info sets, 1750-2950 iter/s on 72 cores
- Regret checkpoint at 300M IS = 10.9 GB
- .bps strategy export too slow for intermediate checkpoints (deferred to final)

### What to do when compute finishes (~April 3):
1. Download final blueprint: `aws s3 sync s3://poker-blueprint-unified/ blueprint_unified/`
2. Terminate both EC2 instances
3. Wire blueprint into runtime: update blueprint_v2.py to load unified .bps
4. Test full HUD pipeline: preflop blueprint → GPU re-solve → strategy output

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

| Parameter | Pluribus | Our Value | Status |
|-----------|----------|-----------|--------|
| Players | 6 | 6 | ✓ |
| Algorithm | External-sampling MCCFR | Same | ✓ |
| Training time | 8 days, 64 cores | 8 days, 192 cores | ✓ |
| Training cost | ~$144 spot | ~$998 on-demand | ✓ |
| Memory | <512GB | ~210GB (3B hash table) | ✓ |
| Preflop buckets | 169 lossless | 169 lossless | ✓ |
| Postflop buckets | 200 k-means [EHS,PPot,NPot] | 200 k-means [EHS,PPot,NPot] | ✓ |
| Discount | d=T/(T+1), 40 rounds, first 3.5% | Same (proportional) | ✓ |
| Pruning | -300M threshold, 95%, explored-only updates | Same (Bug 9 fixed) | ✓ |
| Regret floor | -310M | -310M | ✓ |
| Regret ceiling | None (implicit int32 max ~2.1B) | 2B (Bug 10 fixed) | ✓ |
| Regret storage | int32 | int32 (int64 intermediates) | ✓ |
| Strategy sum | Round 1 only, every 10K iter | Every 10007 iter (coprime w/ 6) | ✓ |
| Preflop sizes | 1-14 per decision (hand-tuned) | Tiered 8/3/2/1 | ~close |
| Postflop sizes | 0.5x, 1x, all-in first; 1x, all-in subsequent | Same | ✓ |
| Search (runtime) | Linear CFR, 4 cont strategies, 5x bias | Same (GPU, 67ms river) | ✓ |
| Play strategy | Final iteration | Same | ✓ |
| Strategy freezing | A3: freeze at passed nodes for hero's hand | Implemented | ✓ |
| Off-tree bets | Pseudoharmonic | Same | ✓ |

---

## AWS RESOURCES

- **S3 bucket (unified)**: `poker-blueprint-unified` — code, checkpoints, probes, final blueprint
- **S3 bucket (v1)**: `poker-blueprint-2026` (110 GB, old per-texture blueprint)
- **Current solver**: `i-09ab6b2e26d0f9007` (c7a.metal-48xl, 192 vCPU, 376 GB)
- **Key pair**: `poker-solver-key` (PEM at `~/poker-solver-key.pem`)
- **Security group**: `poker-solver-sg`
- **IAM profile**: `poker-solver-profile` (role: `poker-solver-ec2-role`, has S3 + EC2FullAccess)
- **Region**: us-east-1

### Monitoring
```bash
# Quick probe (strategy + raw regrets):
aws s3 cp s3://poker-blueprint-unified/probes/probe_latest.txt -

# List checkpoints:
aws s3 ls s3://poker-blueprint-unified/checkpoints/

# Instance status:
bash precompute/launch_blueprint_unified.sh --status

# SSH logs:
ssh -i ~/poker-solver-key.pem ec2-user@$(aws ec2 describe-instances \
    --instance-ids i-09ab6b2e26d0f9007 \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text) \
    'tail -30 /var/log/blueprint-unified.log'
```

---

## BUGS FOUND AND FIXED (BLUEPRINT ENGINE — 10 total)

See `docs/BLUEPRINT_BUGS.md` for full details.

| # | Bug | Root cause | Fix |
|---|-----|-----------|-----|
| 1 | board_hash in info set key | Tree inflated from ~665M to billions | Set board_hash=0, bucket abstracts board |
| 2 | int32 regret overflow | UB at 9.87B iters | int64 intermediates + BP_REGRET_CEILING |
| 3 | Heap corruption | Billions of tiny callocs | Arena allocator |
| 4 | Flat preflop sizing | 7.4B preflop IS (19x Pluribus) | Tiered 8/3/2/1 per raise level |
| 5 | int32 iteration counters | Can't run >2.1B iters | Widened to int64 throughout |
| 6 | strategy_sum aliasing | gcd(6,10000)=2, 3 players excluded | Use 10007 (coprime with 6) |
| 7 | Call/fold trap (diagnosed) | Dominant action freezes regret | See Bug 9 for root cause |
| 8 | 10x regret scaling | Regrets hit ceiling/floor 10x early | Removed scaling factor |
| 9 | **Pruned regrets updated** | 95% of iters push pruned actions negative | Only update explored actions |
| 10 | Regret ceiling 310M | 7x lower than Pluribus implicit max | Raised to 2B |

---

## PROMPT FOR NEXT AGENT

You are continuing development of a Pluribus-exact 6-max NLHE poker solver. The project is at `/Users/victor/Documents/Dev/poker-solver/`. Read `CONTEXT_NEXT_SESSION.md` for full context and `docs/BLUEPRINT_CHRONICLE.md` for the full debugging narrative.

**Blueprint compute with Bug 9+10 fixes is running on EC2.** Check status:

```bash
# Quick convergence check (F/C/R splits + raw regrets):
aws s3 cp s3://poker-blueprint-unified/probes/probe_latest.txt -

# List all checkpoints:
aws s3 ls s3://poker-blueprint-unified/checkpoints/

# Instance status:
bash precompute/launch_blueprint_unified.sh --status
```

**What to look for in probes:**
- `pruned=N/8` for TT/99/44 should DECREASE over time (raises un-pruning)
- `best_raise` should climb toward positive for TT/99
- BTN should show healthy raise rates (sanity check)
- If TT shows R>50% from UTG by 400M, the fix is working

**If compute is complete** (S3 has `checkpoint_meta.json` with `"checkpoint": "final"`):
1. Download blueprint: `aws s3 sync s3://poker-blueprint-unified/ blueprint_unified/`
2. Terminate EC2 instance
3. Wire unified blueprint into runtime:
   - Update `python/blueprint_v2.py` to load the unified `.bps` file
   - The blueprint covers all streets (preflop through river) in one file
   - Test: load blueprint → query preflop strategy for AA → should show mixed raise sizes
4. Test full HUD pipeline: preflop blueprint → GPU re-solve → strategy output
5. Integrate with ACR Poker HUD via Chrome DevTools Protocol

**If compute crashed:** Check the log for the error. Common past issues:
- OOM → check `dmesg | grep oom` → may need larger instance or reduced hash table
- Silent death → check if `snapshot_interval` overflowed int32 (fixed in latest code)
- Spot reclaim → watchdog should auto-relaunch; if not, manually restart

**Cost tracking:**
- On-demand c5.18xlarge: $3.06/hr × 176h remaining ≈ $539
- Can switch to spot (~$0.96/hr) when spot limits reset to save ~$370
- Watchdog t3.micro: $0.01/hr (negligible)

**Compilation** (both .c files required):
```bash
gcc -O2 -shared -fopenmp -o build/mccfr_blueprint.dll \
    src/mccfr_blueprint.c src/card_abstraction.c -I src -lm
```
