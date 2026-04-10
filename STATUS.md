# Poker Solver — Status & Forward Plan

**This is the single source of truth for the poker-solver repo.** Read this first. Everything else is either reference (frozen) or appendix (for one specific subsystem).

Last updated: 2026-04-10 (unified preflop→postflop flow shipped, Layer 3 rollout implemented, GPU segfault fixed, tree walker bug fixed)

---

## TL;DR

A 6-player No-Limit Hold'em MCCFR blueprint solver matching the Pluribus algorithm (Brown & Sandholm 2019), targeted at training/analysis use. **v2 is currently training on EC2** (~7.6% of 8B-iter target). All known v3 code fixes are committed but two of them are partial. The v3 deployment plan does **not** require a fresh training run — most v3 wins live in code that operates on v2's existing blueprint.

---

## Where we are right now

### v2 training run — PAUSED (as of 2026-04-09)

No EC2 instance is currently running. The v2 training was stopped due to AWS
budget ceiling; user has applied for AWS credits and is waiting on approval.
Latest checkpoints from the paused run live in S3:

| | |
|---|---|
| S3 bucket | `poker-blueprint-unified-v2` |
| Latest regrets | `checkpoints/regrets_1500M.bin` (60 GB, 1214M info sets) |
| Earlier snapshots | 200M / 400M / 600M / 900M / 1200M |
| Iter target when paused | 8,000,000,000 (only ~19% complete at 1500M) |
| Training config | 2B-slot hash table, 6-max, Pluribus-style tiered preflop sizes |

When credits land, resume training OR pivot to v3. See §"Immediate next steps"
for the full "when credits land" runbook.

### v3 commit status

All v3 work is in **two commits on `master`**: `a82219b` (Phase 1) and `48da71b` (Phase 2+3). They're behind v2's running code on the EC2 instance — v3 takes effect at the next training run, OR when the realtime path is redeployed.

| Phase | Description | File | Verified | Notes |
|---|---|---|---|---|
| 1.1 | CFR iters 200 → 2000 in realtime solver | `python/hud_solver.py:506` | ✅ | `DEFAULT_CFR_ITERATIONS = 2000` |
| 1.2 | Delete `multiway_adjust.py` heuristics | `python/multiway_adjust.py` (gone) | ✅ | Trust N-player CFR |
| 1.3 | Per-action EVs via σ̄-sampled walk | `src/mccfr_blueprint.c`, `precompute/export_v2.py`, `python/blueprint_v2.py`, `docs/PHASE_1_3_DESIGN.md` | 🟡 **PHASE A DONE, PHASE B DEFERRED** | Phase A (local validation) complete as of 2026-04-09: parallel mmap loader (20x speedup), synthetic ground-truth C test, Python CFR self-consistency verifier (0.003 chip max error), strengthened sentinels 4/5, Bug 6 Python hash mixer sync. Phase B (1.5B v2 EC2 run) deferred pending AWS credit approval. See §"Finish the Phase 1.3 export-tool work" below. |
| 2.1–2.8 | 8 defensive bug fixes | `src/mccfr_blueprint.c` | ✅ | All defensive — none change blueprint output in normal v2 conditions |
| 3.1 | Bug 7: texture lookup → hashmap | `src/mccfr_blueprint.c:268+` | ✅ | +5–15% throughput when applied |
| 3.2 | Bug 6: hash mixer → splitmix64 | `src/mccfr_blueprint.c:241–259` | ✅ | Eliminates the `max_probe = 4096` clustering. Requires fresh run to take effect (changes slot derivation). |

### What's deployed where

| System | Code branch | Blueprint version | Notes |
|---|---|---|---|
| EC2 v2 training | PAUSED (no instance running) | 1.5B iters checkpoint in S3 | Waiting on AWS credits to resume |
| Realtime solver (`hud_solver.py`) | v3 on master | reads `unified_blueprint_*.bps` | Phase 1.1 (2000 iters) + 1.2 active. Phase 1.3 equity-only fallback. |
| GPU postflop solver (`street_solver_gpu.py`) | v3 on master | n/a (real-time CFR, no blueprint) | 200 iterations, ~14s HU flop / ~3s turn / ~1.4s river on RTX 3060. |
| Rollout leaf values (`rollout_leaves.py`) | v3 on master | uniform-strategy fallback | Layer 3 implemented, disabled by default (USE_ROLLOUT_LEAVES=true). ~100x slower than equity-only in Python. |
| Export tool (`precompute/export_v2.py`) | unchanged | writes `schema_version: 2` | **Phase 1.3 step 2 not yet implemented** |
| nexusgto-api (FastAPI) | separate repo | preflop via JSON, postflop via GPU CFR | `POST /api/solve-all` returns both players' strategies. Accepts `ranges` for narrowed preflop weights. |
| nexusgto (Next.js) | separate repo | unified preflop→postflop flow at `/solver` | Preflop tree walk → board picker → GPU CFR solve → range grid. Verified e2e via Playwright. |

---

## Verified facts

> *This section captures the things that have been wrong in earlier docs. If you're tempted to repeat one of these claims without checking, re-read the verification.*

### 1. Pluribus is preflop-and-postflop, not preflop-only at training time

Pluribus's blueprint covers all 4 betting rounds. At **inference time** they use the blueprint only for the first betting round and use real-time search for rounds 2–4. But the blueprint *training* solves all 4 rounds and Pluribus's published 664,845,654 "action sequences" / 413,507,309 "encountered" counts cover the full multi-street tree.

**Source:** `pluribus_technical_details.md` §1.

### 2. Pluribus iter count is unknown

The Pluribus paper does NOT publish a raw iteration count. Anyone who quotes a number for "Pluribus iters" (including past versions of our docs) is making it up. What's actually published: 8 days wall clock, 64 cores, 12,400 core-hours, <512 GB RAM, ~$144 spot. Iteration count is approximately 2-4B based on hardware-throughput estimation.

**Source:** `pluribus_technical_details.md` §1, supplement p.13.

### 3. Hash insertion failures: 231 is negligible

`231 / 914M info sets = 0.0000253%`. Projected at 8B iters: ~5,500 failures = 0.0007% of info sets. The dashboard label "DATA LOSS — bump hash size" is alarmist for this magnitude.

**The real signal isn't 231 — it's `max_probe = 4096`.** That's the cap (`HASH_PROBE_LIMIT_INSERT` in `mccfr_blueprint.h`). Hitting the cap at only 45.7% load factor means the boost-style `hash_combine` is producing pathological clustering. **v3 Phase 3.2 (splitmix64 mixer) fixes this**, but it requires a fresh training run because it changes slot derivation for every key.

### 4. Pruning is identical to Pluribus

`BP_PRUNE_THRESHOLD = -300000000`, `BP_PRUNE_PROB = 0.95`, `prune_start_iter = iters * 17/1000` (= 1.7%, vs Pluribus's 200 min / 11520 min = 1.74%). Verified in `mccfr_blueprint.h` and `precompute/blueprint_worker_unified.py:199`.

**Implication:** "encountered info set" counts ARE apples-to-apples comparable to Pluribus's 413M.

### 5. The 5-bet jam line is in active equilibrium use — DO NOT cut PREFLOP_MAX_RAISES

At iter 400M, BB facing UTG 4-bet shows: AA jams 49%, KK 77%, QQ 99.9%, AKo 81%. The regrets back this up (QQ jam regret +631 vs call -98985). Removing the 5-bet jam tier (`PREFLOP_MAX_RAISES = 4 → 3`) would force these hands to flat-call 4-bets and lose significant EV.

**Verification:** `tests/query_5bet_jam.py` (and `/tmp/query_5bet_jam_remote.py` for the instance-side version that loads regrets).

### 6. Action_hash encoding is NOT inflating the tree

Hypothesis: the `compute_action_hash` representation (full action history per info set) is causing 1 logical info set to split into many. **REJECTED.** Walking the full preflop tree with three encodings:
- Action-hash (current): 1,125,132 preflop nodes
- Logical "FINE" (per-player invested + exact pot + active/has_acted bitmaps): 1,041,472 nodes — only **8% compression**
- Logical "MEDIUM" (collapses per-player investments — too aggressive): 7,641 nodes

The 8% gap is the only collapse possible without losing real strategic differentiation. **Don't change `compute_action_hash`.**

**Verification:** `tests/count_actionhash_vs_logical.py`.

### 7. UTG limping is correct, not undertraining

Pluribus's blueprint famously limps from early positions even with premium hands — a surprising finding from the paper that contradicts conventional opening theory. v2's UTG strategies showing AA limping ~60% is consistent with Pluribus-style equilibrium, NOT a convergence bug.

### 8. River bucket abstraction has mild degeneracy, not catastrophic

Initial sample of 5 river boards showed 1 board with `k_used = 110/200` and a single bucket holding 46 hands — looked like a major problem. **Expanded to 100 boards and the picture is much milder:**
- Mean k_used: 194.7 / 200 (97.4% utilized)
- 2% of boards have ≥10% buckets empty
- 0% have ≥25% empty
- 6% have a single "lumpy" bucket > 25 hands

The cause: PPot/NPot are zero on the river (no future cards), so our 3D feature space `[EHS, PPot, NPot]` collapses to 1D EHS. Hands with identical EHS are mathematically equivalent vs random opponent — k-means putting them together is correct EHS clustering, not a bug. **Net impact: ~5% of the tree is mildly suboptimal. Not worth blocking on.**

The Pluribus-style fix (EHS histograms + Earth Mover's Distance, per Brown & Ganzfried 2015) would be a v4 nice-to-have, not a v3 critical.

**Verification:** `/tmp/test_bucket_quality.py` (run on the v2 instance).

### 9. Per-info-set training quality is comparable to Pluribus

At v2's 8B iters, projected ~1.5–2.5B info sets in the hash table. Average visits/info-set works out to be in the same ballpark as Pluribus's estimated ~10–50 visits/info-set. We are NOT undertrained relative to them.

### 10. Tree-size comparison to Pluribus's 664M is ambiguous

Our enumerated full betting tree (deployed config: 3-2-1-1 preflop tiers, max raises 4 preflop / 3 postflop, 200 buckets) is **~51 billion info sets**. Pluribus reports 664M total. The 77x ratio depends on how "action sequence" is counted in their supplement, and the alternatives are mutually inconsistent (per-player counting, sparse bucket allocation, etc.). **Don't chase the 664M number** — it's not directly comparable to our enumerated count, and even our absolute-minimum config (1 bet size everywhere, max_raises 2) bottoms out at 1.34B = 2× Pluribus, suggesting a structural difference we can't reach with action-abstraction tuning alone.

**Verification:** `tests/count_full_tree.py` and `tests/sweep_config_tree.py` (sweep tested 7 alternative configs).

### 11. Cutting `POSTFLOP_MAX_RAISES` from 3→2 is barely a lever

The Pluribus supplement wording "at most two for the remaining raises in a round" weakly suggests they cap at 2 raises per round on rounds 2–4. We tried it: D config (deployed but with `POSTFLOP_MAX_RAISES = 2`) gives 44.0B info sets vs deployed 51B = **only 14% reduction**. Not worth losing the 3rd raise level.

The big lever in the sweep was **number of bet sizes per node**, not max raises — going from 2 first-bet sizes to 1 shrank the tree 8.5x. But that loses real strategic granularity.

**Verification:** `tests/sweep_config_tree.py` results in `/tmp/sweep_*.log`.

---

## What v3 actually does (and doesn't)

### What v3 fixes that requires no retraining

These are realtime-path / export-side changes. They take effect when you redeploy the realtime code or re-run the export tool against any existing checkpoint.

| Phase | Change | What it actually buys you |
|---|---|---|
| 1.1 | CFR iters 200 → 2000 | Largest single quality improvement available. Subgame strategies become noticeably more converged on mixed-strategy spots. Just redeploy the realtime code. |
| 1.2 | Delete `multiway_adjust.py` | Stops double-correcting on top of N-player CFR. No retraining needed. |
| 1.3 | Per-action EV variance reduction | **Currently broken — needs the export-tool work to actually function.** When complete: restores Pluribus's variance-reduction trick at flop/turn leaves. Single biggest realtime-path quality gap after Phase 1.1. |

### What v3 fixes that DOES require a fresh training run

| Phase | Change | Impact |
|---|---|---|
| 3.2 | Bug 6 hash mixer (splitmix64) | Eliminates the ~5,500 silently-dropped info sets at saturation = **0.0004% improvement**. Reduces `max_probe` from 4096 cap to ~50–200, speeds up lookups. |
| 3.1 | Bug 7 texture hashmap | **+5–15% throughput** — saves ~12–24 hours of wall time over a full 8B-iter training run. |
| 2.x | Defensive bug fixes | Zero behavioral impact in normal conditions. Future-proofing only. |

**Net: a fresh v3 training run is borderline pointless on its own.** The marginal improvement (1–3% blueprint quality + faster throughput) doesn't justify the cost ($55–75 + 3 days) given that:
1. The biggest realtime wins (Phase 1.1, 1.2) work with v2's blueprint as-is
2. The Phase 1.3 export work (the second-biggest win) gives per-action EVs **without retraining**
3. The fresh-run-only wins are cosmetic for blueprint quality

### What v3 does NOT fix

The strategy convergence issues you can see at iter 600M (zigzag pair-ladder, BTN 66 raising 7%, etc.) are from:
1. **Average strategy lag** — the average strategy lags the current strategy at low visit counts. v3 doesn't change this. Pluribus uses the **final-iteration** strategy for postflop precisely to avoid this lag.
2. **Low visit counts** on rare info sets — pruning suppresses 95% of branches. v3 doesn't change this.

What WOULD fix the zigzag:
- Just running v2 longer (linear improvement per iter)
- **Switching the export to use final-iteration strategy** (Pluribus's approach for postflop) — code change, no retraining
- Real-time search at inference time (T5.1 in [REALTIME_TODO.md](docs/REALTIME_TODO.md)) — handles the rare branches without needing them trained well

---

## What was completed 2026-04-10

1. **Unified preflop → postflop UX** — `/solver` is one continuous flow. After preflop action completes, inline BoardPicker appears. User picks flop cards, clicks "Solve Flop". GPU CFR runs with narrowed ranges from the preflop walk. Both players' strategies displayed in the same SolverShell. Verified end-to-end via Playwright.
2. **Layer 3 rollout-based leaf values** — `python/rollout_leaves.py` has full Pluribus-faithful betting simulation (EHS bucket lookup → blueprint σ̄ → 5x bias → action sampling). 56/56 tests pass. Disabled by default (Python ~100x slower than equity-only). Enable with `USE_ROLLOUT_LEAVES=true`.
3. **GPU segfault fix** — back-to-back solves no longer crash. Skipped explicit `solver.free()` in the API provider (GPU memory already freed inside `ss_solve_gpu`).
4. **Preflop tree walker bug fix** — `findNextToAct` in `solver-state.ts` now checks `committed < currentBet`, fixing action loop-back (e.g. UTG acting again after BB 3-bets).
5. **New API endpoint** — `POST /api/solve-all` returns both players' strategies from a single GPU solve. Accepts optional `ranges` field for narrowed preflop weights.
6. **Test coverage** — 164/164 vitest, 56/56 rollout, 5/5 API e2e, all passing.

## Immediate next steps

### 1. v2 training — PAUSED

The current v2 training run on EC2 is stopped as of early 2026-04-09 due to AWS budget
ceiling. User has applied for AWS credits and is waiting on approval before resuming.
Latest checkpoints on S3:

- `s3://poker-blueprint-unified-v2/checkpoints/regrets_1500M.bin` (60 GB, 1214M info sets)
- Earlier snapshots at 200M / 400M / 600M / 900M / 1200M iterations

When training resumes (or a fresh v3 run kicks off with credits), this section should be
updated with the new instance / target iter count / ETA.

**Not to be confused with:** the older 1.6B-iter v2 blueprint exported on 2026-04-07 that
lives locally at `blueprint_data/unified_blueprint.bps`. That file is from a DIFFERENT
training run (pre-Bug-6 fix, 946M info sets vs. the current run's 1214M at 1500M) and its
raw regrets are not in S3.

### 2. Finish the Phase 1.3 export-tool work

**Status as of 2026-04-09: Phase A (local validation) complete, Phase B (1.5B EC2 run) deferred.**

#### Phase A — local validation (complete, 6 commits on master)

| Commit | Scope |
|---|---|
| `199c243` | Phase 1.3 core: traverse_ev(), bp_compute_action_evs(), bp_export_action_evs(), BPInfoSet extensions, initial export_v2 + blueprint_v2 bindings |
| `c968fe2` | **Parallel mmap regret loader (BPR4)** — ~20x speedup vs. serial; duplicate-merge correctness validated via hand-crafted 5000-entry test. Drive-by Windows build compat. |
| `5e3f7fd` | Synthetic C ground-truth test (2p postflop toy game, Stage 1 sanity checks, test bundle format) |
| `ac5c834` | **Bug 6 Python hash mixer sync** (splitmix64). Fixes silent lookup failure in blueprint_v2.py on v3+ files. Regression guard via tests/test_hash_sync.c + .py. |
| `fd3d02a` | Strengthened sentinels: new sentinel 4 (AA/32o EV ordering at UTG open) and sentinel 5 (visit count distribution from new bp_get_ev_visit_stats C function) |
| `fe087fd` | Hash mixer backcompat — tag `"hash_mixer": "splitmix64"` on new exports; blueprint_v2.py warns on legacy files |
| `78b936a` | Python CFR self-consistency verifier — independently walks the toy game's action tree, validates 6 (info_set, action) pairs to 0.003 chip max error |

**What's validated locally:**
- Parallel loader produces bit-identical hash table state vs. serial loader on a trained toy checkpoint AND on a hand-crafted 5000-entry duplicate-heavy file (atomic regret fetch_add + strategy_sum CAS merge verified under 24-thread contention)
- Phase 1.3 EV walk math is self-consistent: Python walker reaches all 24 info sets the C code exports, validated (info_set, action) pairs match to 0.003 chip error, and sign-flip propagation across opponents in zero-sum is correct
- Python hash-combine matches C splitmix64 byte-exactly on 17 test vectors (bp_mix64 + hash_combine + compute_action_hash)
- Stage 1 C sanity checks (no NaN, bounded EVs, non-NULL action_evs allocation) pass on every synthetic run
- Windows MSYS2 build now works end-to-end (was broken by unconditional sys/mman.h / sched.h / unistd.h includes before this session)

**What's NOT yet validated (Phase B territory):**
- Full pipeline run against the 200M v2 raw regrets on this PC (27 GB downloaded to `blueprint_data/regrets_200M.bin`, pipeline run blocked waiting for CPU availability). This is the last Phase A task — ~10 min of 24-thread CPU pressure.
- Full pipeline run against the **1.5B** v2 raw regrets on EC2 (requires ~$5 r6i.8xlarge spend; user is waiting on AWS credit approval before authorizing)

#### Phase B — EC2 1.5B verification (deferred, requires AWS credits)

When credits land, execute:

1. Launch **`r6i.8xlarge`** (256 GB RAM, ~$2/hr, ~$5 total). Smaller instances OOM — the Phase 1.3 BPInfoSet extensions add ~8 GB of struct overhead on top of the existing hash table, which pushes the 1.5B v2 checkpoint over the r6i.4xlarge limit (documented from a prior failed attempt).
2. `poker-solver-key` in us-east-1, Ubuntu 22.04 AMI, sg `poker-solver-sg`.
3. SCP source tarball (or clone via GitHub auth — the public mirror is too stale).
4. Download caches to `/tmp/` BEFORE running:
   - `s3://poker-blueprint-unified-v2/texture_cache.bin` → `/tmp/texture_cache.bin`
   - Turn centroids need fresh compute (~7.5 min) because the S3 file has stale
     magic `TKC1` — C code expects `TCN1`. Copy the regenerated file to
     `/home/ubuntu/` so stop/start cycles don't lose it.
5. `EV_WALK_ITERS=1000000 python3 precompute/export_v2.py /path/to/regrets_1500M.bin
   /home/ubuntu/out '' 1073741824` for a smoke test. Scale to 50M for production.
6. **Regret load should now take ~1-2 minutes** (Tier 2 parallel loader), not the 25-30 min the old serial loader needed. This is the session-saver from Phase A.
7. Run `python3 precompute/verify_phase_1_3.py /home/ubuntu/out/unified_blueprint.bps` — all 5 strengthened sentinels should pass.
8. If sentinels pass: upload v3 .bps to S3 as `unified_blueprint_v3_1.5B.bps`.
9. **Terminate EC2** (not stop — EBS charges continue). Shred `~/.aws/credentials` before termination. IAM access key `AKIAQD7AYAUBM43NHZWY` for user `prithish` should be rotated; it was exposed in a prior session's terminal output.

**Gotcha about production blueprint file:** the blueprint file currently on this PC at
`blueprint_data/unified_blueprint.bps` (3.2 GB, 946M info sets, exported 2026-04-07 16:10 EDT
from iter 1.6B) is from an **older training run** — pre-Bug 6 fix. Its keys use the
boost-style mixer and do NOT match what current blueprint_v2.py computes. Any re-enable of
`USE_BLUEPRINT=true` in nexusgto-api against this legacy file will trip the loud warning
added in `fe087fd` and every get_strategy lookup will return None. The fix is to export a
fresh .bps from a current S3 checkpoint (200M-1500M) once Phase B finishes.

### 3. After v2 reaches 8B: re-export with the new tool

Once v2 finishes:
```
aws s3 cp s3://poker-blueprint-unified-v2/checkpoints/regrets_8000M.bin /tmp/
python3 precompute/export_v2.py /tmp/regrets_8000M.bin /tmp/v2_blueprint.bps
aws s3 cp /tmp/v2_blueprint.bps s3://poker-blueprint-unified-v2/
```

Verify the .bps loads and produces 4 distinct biased leaf values (not collapsed to one) via `python/leaf_values.py`.

### 4. Sentinel decision audit

Pull ~50 sentinel decisions from the v2.bps (UTG opens by hand, BB defense vs each position, BTN open range, 3-bet decisions, 4-bet decisions, common flop spots) and compare to a published 6-max GTO solver output (PioSolver / GTO Wizard). This is the cleanest direct quality measurement.

**No tooling exists for this yet.** It's a one-day write. Decision: do it manually for the first audit, then automate if useful.

### 5. Deploy nexusgto-api with v2.bps + v3 realtime code

The realtime code (Phase 1.1, 1.2, 1.3-after-export-fix) is already on `master`. Once v2.bps exists in S3, the API repo just needs to point at it. Standard deployment.

---

## Forward roadmap (Pluribus parity)

### Short-term (1–2 weeks)

| Item | Effort | Status | Notes |
|---|---|---|---|
| Layer 6 — Final-iteration strategy | ~2 hours | Not started | Return iter-N strategy instead of σ̄ for postflop. Fixes zigzag convergence artifacts. |
| Layer 3.3 — CUDA rollout port | 3–5 days | Not started | Port the Python rollout betting sim to CUDA. Required for real-time use (Python is ~100x slower). |
| Layer 4.A — Multi-way flop | 3–5 days | Not started | Fix `extract_leaf_info_from_tree` for variable `n_act`, add 2-raises depth limit. Unlocks 3+ player postflop. |

### Medium-term (2–6 weeks)

| Item | Effort | Status | Notes |
|---|---|---|---|
| **Layer 4.B — Multi-street GPU CFR** | 1–2 weeks | Not started | Port chance-node kernels from `flop_solve.cu` into `street_solve.cu`. Biggest CUDA work item. |
| **Layer 5 — Preflop re-solve** | ~1 week | Not started | >$100 off-tree bets trigger subgame re-solve from preflop root instead of pseudoharmonic snap. |
| Sentinel decision audit | ~3 days | Not started | Compare 50 solver decisions to PioSolver/GTO Wizard. First quality measurement. |

### Long-term (1+ quarters)

| Item | Notes |
|---|---|
| Card abstraction v2 (EHS histograms + EMD) | The Pluribus-style fix for the river bucket mild degeneracy. Probably unlocks the next quality improvement after T5.1. |
| Hand-tuned bet abstraction sweep | Empirically test which bet-size schemes give best performance vs. Pluribus reimplementations. Currently we use 3-2-1-1 preflop tiers based on heuristics, not measurement. |
| Vector-based MCCFR for small subgames | Pluribus uses this for real-time search. Roughly 2–4× per-core throughput on small subgames. |

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-07 | Pivoted from HUD to trainer use case | Latency budget relaxed from <100ms to seconds-to-minutes, unlocking Phase 1.1 and other realtime quality improvements |
| 2026-04-07 | Approved Phase 1-3 commits | Defensive + realtime fixes, none disturb running v2 |
| 2026-04-07 | v2 launched at 8B iter target with all v2 bug fixes | Slightly more buffer than Pluribus's 7B compute equivalent |
| 2026-04-07 | v2 hash table at 2B slots | Empirical projection ~1.05B entries at 8B → 52% load |
| 2026-04-08 | **Don't chase Pluribus's 664M number** | Comparison is ambiguous; even our minimum config can't reach it |
| 2026-04-08 | **Don't cut `PREFLOP_MAX_RAISES`** | 5-bet jam line is in active equilibrium use (verified via live solver query) |
| 2026-04-08 | **Don't cut `POSTFLOP_MAX_RAISES` either** | Only 14% tree reduction, not worth the strategic loss |
| 2026-04-08 | **Don't fix river bucket abstraction in v3** | Mild (2-6% of boards), not worth blocking |
| 2026-04-08 | **Skip the v3 retraining run** | Marginal quality improvement vs cost; Phase 1-3 realtime work delivers most of the benefit without retraining |
| 2026-04-08 | Final-iteration strategy export is higher priority than v3 retraining | Probably bigger blueprint quality impact |
| 2026-04-10 | API postflop uses 200 CFR iterations (not 2000) | 2000 would take ~140s per HU flop solve; 200 gives ~14s which is acceptable for interactive use. hud_solver keeps 2000 for standalone/batch. |
| 2026-04-10 | Skip solver.free() to fix segfault | GPU memory freed inside ss_solve_gpu; CPU struct leak (~50KB/solve) is negligible vs the CUDA runtime corruption from explicit teardown |
| 2026-04-10 | Layer 3 rollout disabled by default | Python rollout is ~100x slower than equity-only. Correct but impractical until CUDA port (Layer 3.3). Enable with USE_ROLLOUT_LEAVES=true for testing. |

---

## Open questions (for the human)

These are things I genuinely don't know and the human needs to decide:

1. **When does the trainer product launch?** This drives whether the next priority is sentinel audits + nexusgto polish (launch sooner) or T5.1 real-time search (launch later but stronger product).
2. **How important is a head-to-head benchmark vs Pluribus reimplementations?** The fedden/poker_ai repo has a Pluribus-style blueprint that could play against ours. Would be the cleanest "are we actually good?" measurement, but takes engineering work to integrate.
3. **What's the right card abstraction to invest in?** Our EHS+PPot+NPot k-means is functional but not Pluribus-grade. EHS histograms + EMD is the published "right answer" but expensive to implement.

---

## Related repos on disk

The full system is **3 repos**, all under `~/Documents/Dev/`:

| Repo | Stack | Role |
|---|---|---|
| `poker-solver` (this repo) | C + Python + CUDA | The MCCFR blueprint solver and realtime subgame search. The thing that produces and consumes the .bps blueprint file. |
| `nexusgto-api` | Python (FastAPI) | API layer that wraps the solver and serves to the frontend. `app/routers/`, `app/services/`. |
| `nexusgto` | Next.js (TypeScript) | Frontend at nexusgto.com. Calls nexusgto-api over HTTP. |

**When working on integration**, pull all three before reading code:
```
cd ~/Documents/Dev/poker-solver && git pull
cd ~/Documents/Dev/nexusgto-api && git pull
cd ~/Documents/Dev/nexusgto && git pull
```

The local copies can drift from origin without warning.

---

## Pointers to specialized docs

These are the **only** other docs you should read. Anything not listed here is either deleted or archived.

| File | Purpose | When to read |
|---|---|---|
| [`README.md`](README.md) | Project intro, build instructions | First time setting up the repo |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Component overview and decision pipeline | Understanding how the pieces fit together |
| [`pluribus_technical_details.md`](pluribus_technical_details.md) | Verbatim extraction of Pluribus paper supplement | When you need ground-truth Pluribus values. **Frozen — never edit.** |
| [`REFERENCES.md`](REFERENCES.md) | Citations | When you need a paper |
| [`docs/SOLVER_CONFIG.md`](docs/SOLVER_CONFIG.md) | Source of truth for every solver parameter | Before launching any training run, OR before changing any constant |
| [`docs/REALTIME_TODO.md`](docs/REALTIME_TODO.md) | Realtime / subgame solver backlog | For the realtime path forward plan |
| [`docs/V3_PLAN.md`](docs/V3_PLAN.md) | v3 execution plan (status: shipped Phase 1-3) | Historical for v3; future plans live in this STATUS.md |
| [`docs/BLUEPRINT_BUGS.md`](docs/BLUEPRINT_BUGS.md) | Solver bug log (Bugs 1–11) | When debugging a specific historical bug |
| [`docs/EXTRACTOR_BUGS.md`](docs/EXTRACTOR_BUGS.md) | Frontend extractor bug log | When debugging the frontend's blueprint consumption |
| [`docs/BLUEPRINT_CHRONICLE.md`](docs/BLUEPRINT_CHRONICLE.md) | Narrative timeline of every training run | Research / historical interest |
| [`COMMERCIALIZATION.md`](COMMERCIALIZATION.md) | Business strategy (separate concern from solver) | Product / business decisions |

---

*If you're confused about the project state and you're reading some other doc that contradicts this one, this STATUS.md is correct and the other doc is stale. If a fact in STATUS.md is wrong, fix it here first.*
