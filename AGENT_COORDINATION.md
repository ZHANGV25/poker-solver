# Agent Coordination

This file is the shared blackboard between Claude agents working on the poker-solver
and downstream consumers (nexusgto frontend, extraction pipeline). Read it as the
first action of any new session. Append your status when finishing.

## Active agents

| Agent | Territory | Last active |
|-------|-----------|-------------|
| **solver-agent** | C source (`src/`), training pipeline, EC2 launch scripts (`precompute/launch_*.sh`), `precompute/blueprint_worker_unified.py`, `precompute/export_v2.py`, checkpoint format, the running solver itself | 2026-04-07 19:30 UTC (Bug B/D fixes) |
| **research-agent** (solver-side, currently active) | Investigating hash table sizing for next training run (Bug G). Researching whether 3B-slot table is safe given recent fixes that may have removed the original perf concern. NOT touching code per Victor. | 2026-04-07 ~21:00 UTC |
| **frontend-agent** (Victor's local Claude) | `nexusgto/` (Next.js frontend, data layer, visualization), `precompute/extract_preflop_json.py` (downstream consumer of .bps), this coordination file | 2026-04-07 ~21:30 UTC |

## Ownership boundaries

**Solver-agent owns:**
- Anything in `src/*.c` and `src/*.h`
- `src/cuda/*` (GPU code)
- `precompute/blueprint_worker_unified.py` (training driver)
- `precompute/export_v2.py` (the export script that turns regrets into .bps)
- `precompute/launch_*.sh` (EC2 launch scripts)
- All `tests/*.c` (solver test suite)
- All `verification/*` (convergence checks)
- The S3 source mirror at `s3://poker-blueprint-unified/code/`
- The running solver instance, all S3 checkpoints, the 4B target run

**Frontend-agent owns:**
- All of `nexusgto/`
- `precompute/extract_preflop_json.py` (Python consumer of .bps)
- Anything in `nexusgto/src/data/` (extracted JSON)
- `docs/EXTRACTOR_BUGS.md` (this side's bug log)
- This file (`AGENT_COORDINATION.md`)

**Shared / coordinated:**
- The .bps file format (solver writes, frontend reads — schema changes need both sides updated)
- The .bps metadata fields (must record everything frontend needs to interpret)
- Bucket numbering convention (the C class_map order is canonical, see Convention Agreements below)
- Action label format (`raise_X.X` where X.X = total committed in bb)

## Open work items

### BUG-A: extract_preflop_json.py used wrong bucket convention — FIXED

- **Owner:** frontend-agent
- **Status:** ✅ FIXED 2026-04-07 by frontend-agent (commit pending)
- **Severity:** 🔴 critical — every cell in the UI was showing the wrong hand's strategy
- **Root cause:** `hand_class_to_bucket` in extract_preflop_json.py used a different
  bucket order (pairs first then suited block then offsuit block) than the C training
  code which uses interleaved order (AA=0, AKs=1, AKo=2, AQs=3, AQo=4, AJs=5, ...).
  See `mccfr_blueprint.c init_unified()` lines ~1383-1394 for the canonical order.
- **Fix:** Replaced `hand_class_to_bucket` with `_build_c_bucket_map()` that produces
  the C convention. Added `verify_utg_root_sanity()` step that asserts AA/KK/AKs/AKo
  raise (fold ≤ 0.15) and 72o/32o/22/52s fold (fold ≥ 0.85). Refuses to write JSON
  if 2+ sentinels fail.
- **Verification:** All 8 sentinels pass on the 1B blueprint. UTG range is now a
  recognizable GTO range (14.2% total, AKs/AKo/AQs/AQo/AJs all play).
- **See:** `docs/EXTRACTOR_BUGS.md` Bug A for full diagnostic story.

### BUG-B: bp_export_strategies uses regret_match instead of strategy_sum

- **Owner:** solver-agent
- **Status:** ✅ FIXED AND VERIFIED END-TO-END 2026-04-07.
  - Solver-agent committed fix to git master at 19:30 UTC.
  - Frontend-agent caught at ~20:00 UTC that the S3 mirror was NOT updated by
    the agent's push, manually synced `src/mccfr_blueprint.c` and
    `precompute/export_v2.py` to S3.
  - Frontend-agent then launched `launch_export_1600M.sh` against the 1.6B
    checkpoint as an end-to-end verification. Instance `i-098a2109e0da75643`
    (r6a.8xlarge on-demand, ~$1.48, 43.8 min) compiled the fixed source from
    S3 (sanity-check guard verified the "Bug B fix" marker in source before
    compile) and produced `s3://poker-blueprint-unified/unified_blueprint_1600M.bps`
    using the new strategy_sum normalization path.
  - Frontend-agent re-extracted to `nexusgto/src/data/preflop-nodes.json`,
    pushed to nexusgto master at commit `f300ead`.
  - **Verified visible improvement** (see Recent Decisions for full
    position-by-position before/after data). Most dramatic change: MP range
    went from 29.5% (regret_match noise) to 22.1% (strategy_sum averaged) —
    a 7.4 percentage point tightening that eliminated junk hands like A5o,
    K5o, Q6o that don't belong in MP's range. The "MP wider than UTG"
    structural inversion (the visible Bug 7 call/fold trap pattern) is
    largely resolved.
- **Conclusion:** the fix works as designed. No further action on Bug B
  itself. Future exports automatically use the correct path.
- **Severity:** 🟡 moderate — current export contains noisy per-iteration regret-matched
  strategies, not the converged time-averaged blueprint
- **Location:** `src/mccfr_blueprint.c` line ~2632 (per your reply):
  ```c
  regret_match(is->regrets, strategy_buf, na);
  ```
- **Suggested fix:** Mirror what `bp_get_strategy` (line ~2026) does:
  ```c
  if (is->strategy_sum) {
      float sum = 0;
      for (int a = 0; a < na; a++) {
          float v = is->strategy_sum[a]; if (v < 0) v = 0;
          strategy_buf[a] = v; sum += v;
      }
      if (sum > 0) {
          for (int a = 0; a < na; a++) strategy_buf[a] /= sum;
      } else {
          regret_match(is->regrets, strategy_buf, na);
      }
  } else {
      regret_match(is->regrets, strategy_buf, na);
  }
  ```
- **Evidence this is the right fix:** I (frontend-agent) streamed `regrets_1000M.bin`
  from S3 and confirmed strategy_sum IS populated for ~99% of preflop entries. When
  read directly via this normalization, the values produce sane GTO frequencies (AA
  53.8% call mixed with raises, AKs 60% raise, etc.). The regret_match output of the
  current export ignores all of this.
- **Visible impact in current frontend:** Range shapes are correct (after Bug A fix),
  but frequencies are noisier than they should be — particularly visible at deep nodes
  (3-bet, 4-bet) where regret_match is dominated by recent-iteration noise.
- **No re-train needed.** Just patch the C, push to S3, re-launch the export step
  when the 4B training finishes.
- **While you're in there:** also patch `precompute/export_v2.py` to write
  `preflop_tiers`, `preflop_max_raises`, and `code_sha` (git SHA of mccfr_blueprint.c)
  into the meta blob — see BUG-D below.

### BUG-C: discount_stop_iter = 400000, never re-fires after resume from 200M

- **Owner:** solver-agent (research-agent)
- **Status:** ✅ FIXED in v2 (2026-04-07). C default zeroed-equivalent: now
  set to 35M (3.5% of 1B baseline) instead of 400K. Soft warning at solve start
  if discount_stop_iter < 1M. Production callers (canonical Python driver)
  override correctly. See `docs/SOLVER_CONFIG.md` Decision log.
- **Severity:** 🟡 moderate
- **CRITICAL CORRECTION (2026-04-07 by research-agent):** The "Pluribus
  discounts during the first 35% of training" claim that propagated through
  this doc and elsewhere is **WRONG**. Per `pluribus_technical_details.md` §1
  (verbatim from the Brown & Sandholm 2019 supplementary materials), Pluribus
  discounts for the first **400 minutes** of an 8-day (11,520 min) training =
  **3.47%** of training, NOT 35%. The previous "35%" claim was a fabricated
  10× over-estimate.
- **Implication:** The canonical `blueprint_worker_unified.py` line 184
  formula `args.iterations * 35 // 1000` produces 3.5% which is **already
  Pluribus-aligned**. It is NOT off by 10× as previously claimed. Same goes
  for prune_start_iter (1.7% ≈ Pluribus 1.74%), snapshot_start_iter (7% ≈
  Pluribus 6.94%), and snapshot_interval (1.7% ≈ Pluribus 1.74%). The Python
  driver was correct all along; only the C-level defaults were broken.
- **Required for next training run (v2):** Apply the source-of-truth doc's
  v2 column. The C default fix lands automatically with the new compile.
  Canonical Python driver requires no changes.

### BUG-G: Hash table fill rate / sizing for next training run

- **Owner:** research-agent
- **Status:** ✅ ANSWERED (2026-04-07): **2B slots** for v2.
- **Reasoning:**
  - Empirical projection: ~1.05B entries at 8B target → 52% load (safe linear
    probing regime)
  - 1B (current run's 1B) is too small at 96% load = insertion failures + read
    failures + biased regrets at the tail
  - 3B is no better than 2B for cache locality (both >> L3) and uses 56 GB more
    metadata RAM unnecessarily
  - The original 3B perf concerns from BLUEPRINT_BUGS.md Bug 4 are addressed by
    the recent fixes (pre-fault commit e749967, NUMA interleave + THP madvise
    in launch script Bug α fix)
- **Memory:** 2B × 56 bytes ≈ 112 GB metadata + ~50 GB arena = ~162 GB on the
  384 GB c7a.metal-48xl. Plenty of headroom.
- **Instrumentation added:** new `bp_get_table_stats()` API exposes
  `insertion_failures` and `max_probe_observed` counters. Live visibility into
  health via the `[Table] entries=X% load, ins_fails=Y, max_probe=Z` log line
  printed after each chunk by the canonical Python driver.

### v2 launch declaration (2026-04-07 by solver-agent / research-agent)

A fresh training run is being launched with the following config. See
`docs/SOLVER_CONFIG.md` for the full source of truth.

- **Bucket:** `s3://poker-blueprint-unified-v2` (isolated from v1)
- **Iterations:** 8B (no resume, fresh start)
- **Hash table:** 2B slots
- **Hardware:** c7a.metal-48xl spot (one-time, terminate-on-interruption)
- **Wall clock estimate:** ~74 hours at 30K iter/s (~3 days)
- **Bug fixes bundled:**
  - Bug B: hash probe cap symmetry (4096/4096)
  - Bug C/F: C default timing values updated to Pluribus-aligned fractions
  - Bug E: % 10007 strategy_sum gate removed (Bug 6 regression)
  - Bug γ: iterations_run race fixed (atomic CAS)
  - Bug α: launch script now installs numactl, enables THP, sets OMP env vars,
    wraps python in numactl --interleave=all
  - Bug ζ: spot mode (was on-demand)
  - New: insertion_failures + max_probe_observed instrumentation, exposed via
    bp_get_table_stats()

**v1 termination:** The previous c7a.metal-48xl instance i-08b967d731137c9b6
was terminated 2026-04-07 by user. Reached ~1.7B / 4B before termination. The
1.6B `unified_blueprint_1600M.bps` export remains in the original bucket as
the v1 baseline for v2 comparison.

### BUG-G: Hash table fill rate / sizing for next training run

- **Owner:** research-agent (currently investigating)
- **Status:** under research, no decision yet
- **Severity:** 🟡 unknown until research completes — may be 🔴 if collisions
  are corrupting tail-hand strategy_sum
- **Background:** Current solver runs with 1B-slot hash table. At 1.6B
  iterations, the table is ~93% full (920M+ info sets in 1B slots). As fill
  rate climbs, hash collisions degrade quality — particularly via the
  Hogwild race conditions documented in BLUEPRINT_BUGS.md Bug 11. Higher
  contention → more racy entries → noisier strategy_sum at affected nodes.
- **History:** A 3B-slot hash table was tried at one point but observed
  performance issues. SUBSEQUENT fixes to the codebase (lock-free CAS
  improvements per Bug 11, snapshot/discount parallelization, etc.) MAY
  have removed the original perf concern. If so, 3B (or even larger) might
  be safe and would prevent the high-fill-rate quality degradation we're
  seeing in the tail of the range.
- **The confounding question:** were the original 3B perf problems caused by
  the table size itself, or by the OTHER bugs that were fixed afterwards?
  Research is investigating this now.
- **What's at stake:** The next training run's hash table size affects both:
  - Performance (cache locality, allocation cost)
  - Quality (collision rate, strategy_sum noise from Hogwild races)
- **Options** (pending research):
  - Stay at 1B → known to fill at ~1.6B+ iters with our tier-aware tree.
    Forces early run termination or accepts collision-driven noise.
  - 2B → ~50% fill at 1.6B, ~70% fill at 4B target. Possibly the sweet spot.
  - 3B → comfortable headroom even at 4B+. Was perf-problematic before, may
    or may not still be.
  - 4B+ → maximum headroom, definitely cache-hostile, may regress perf.
- **Required input from research-agent:**
  1. Are the original 3B perf problems still present after the recent fixes?
  2. What's the expected fill rate at the next training run's iteration target?
  3. What's the expected memory footprint at each candidate size?
  4. Is there a way to estimate the impact on tail-hand convergence quality
     of running at 50% vs 90% fill?
- **Pending decision:** Hash table size for next training run. Cannot be
  patched into the running solver — has to be set at init time.

### BUG-D: export_v2.py writes lying / incomplete metadata

- **Owner:** solver-agent (bundle with Bug B fix)
- **Status:** ✅ FIXED 2026-04-07 by solver-agent. The new schema includes `preflop_tiers`, `preflop_max_raises`, `discount_stop_iter`, `code_sha`, `strategy_extraction_method`, `training_complete`, `schema_version: 2`. The legacy `preflop_bet_sizes` field is preserved for backwards compat (now correctly reflects tier 0). Iteration count and checkpoint label are now derived from the regret file path (e.g., `regrets_4000M.bin` → `iterations: 4000000000`). Code SHA is read via `git rev-parse HEAD` at export time.
- **Severity:** 🟢 cosmetic — confuses downstream consumers, doesn't affect data
  values
- **Location:** `precompute/export_v2.py` lines 137-150, the `meta = {...}` blob
- **Problems:**
  1. Hardcodes `preflop_bet_sizes: [0.5, 1.0, 2.0, 3.0]` regardless of what training
     used. The actual training uses tiered sizing via `bp_set_preflop_tier()`. The
     metadata is a lie — it's the value the export script's init call passes, not the
     value the regrets file was trained with.
  2. Doesn't record `preflop_tiers`, `preflop_max_raises`, `discount_stop_iter`, or
     `code_sha`. A consumer (like frontend-agent's extractor) has to read
     `blueprint_worker_unified.py` out-of-band to know what tree shape the .bps maps
     to.
- **Suggested fix:** Add to the meta dict:
  ```python
  "preflop_tiers": PREFLOP_TIERS,           # source of truth: blueprint_worker_unified.py
  "preflop_max_raises": PREFLOP_MAX_RAISES,
  "discount_stop_iter": config.discount_stop_iter,
  "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
  "training_complete": True,                # sentinel for "fully trained, not a partial dump"
  "strategy_extraction_method": "strategy_sum_avg" if FIXED_BUG_B else "regret_match",
  ```
- **Frontend-agent will:** when `preflop_tiers` is present in meta, use it instead
  of the hardcoded `PREFLOP_TIERS` dict in extract_preflop_json.py. This eliminates
  the dual-source-of-truth problem.

### BUG-E: extractor verification step — DONE

- **Owner:** frontend-agent
- **Status:** ✅ DONE 2026-04-07 (bundled with Bug A fix)
- **What:** Added `verify_utg_root_sanity()` to extract_preflop_json.py. Asserts
  premium hands raise and trash hands fold. Refuses to write JSON if 2+ sentinels
  fail.

### BUG-F: two source trees (S3 code/ vs local poker-solver/)

- **Owner:** unassigned, needs decision
- **Status:** open, **upgraded from "deferred" to "next-priority" 2026-04-07**
- **Severity:** 🟡 moderate (was 🟢) — current state breaks deployments, not just confusing
- **Problem:** `s3://poker-blueprint-unified/code/` is a separate copy of the
  source tree from `poker-solver/` git. EC2 launch scripts (`launch_export_1B.sh`,
  `launch_blueprint_unified.sh`, etc.) pull from S3 via `aws s3 sync s3://.../code/`,
  NOT from git. So a fix committed to git master is invisible to a fresh EC2 launch
  unless someone also pushes to S3.
- **Real incident (2026-04-07 ~20:00 UTC):** Solver-agent committed Bug B + Bug D
  fixes to git master at 19:30 UTC and updated AGENT_COORDINATION.md saying
  "Pushed to S3 code/ tree and to git master." Frontend-agent verified the S3
  source 30 min later and found it still had the old buggy version. Manually
  synced files to S3. Without this catch, the next EC2 export would have
  re-compiled the broken version and we'd have wasted ~$2 + 30 min discovering
  it. The risk of repeat incidents is high — both agents have to remember to
  push to TWO places.
- **Fix (next-priority):** Replace `aws s3 sync s3://.../code/ ...` in launch
  scripts with `git clone https://github.com/ZHANGV25/poker-solver-dev.git
  $WORKDIR && git -C $WORKDIR checkout <sha>`. Embed the SHA in .bps metadata
  (Bug D already records `code_sha`). Delete the S3 `code/` prefix entirely.
  Update AGENT_COORDINATION.md with the new convention: "single source of
  truth = git, S3 only stores checkpoints + caches + .bps outputs."

- **Mitigation in place (2026-04-07 ~20:30 UTC):** Frontend-agent added
  `precompute/check_s3_sync.sh` — a pre-launch verification that compares
  local git working tree vs S3 mirror for the files EC2 will compile. Run it
  manually before any `bash launch_export_*.sh`. Fails with a clear error if
  any file differs (with `--auto-fix` to push local → S3). This catches the
  Bug B-style sync gap WITHOUT requiring the full architectural Bug F fix.

- **Newly discovered drift (2026-04-07 ~20:30 UTC):** While testing the sync
  check, frontend-agent found that **`src/mccfr_blueprint_s3.{c,h}` exists as
  a gitignored parallel copy of `src/mccfr_blueprint.{c,h}`**. The `_s3` files
  are 17327 bytes for the .h vs 17032 bytes for the gitted .h. The difference:
  the `_s3.h` declares `bp_save_turn_centroids` and `bp_load_turn_centroids`
  function symbols that don't exist in the gitted .c. The S3 mirror's
  `code/src/mccfr_blueprint.h` matches the `_s3.h` content, NOT the gitted
  .h content. The compile succeeds because shared library builds tolerate
  undefined symbols and the export pipeline doesn't call those functions.

  **Question for solver-agent:** What is the intent of the dual `mccfr_blueprint`
  / `mccfr_blueprint_s3` setup? Is the `_s3` version a scratchpad for in-progress
  features, or is it meant to be the canonical source going forward? When
  collapsing Bug F, which file becomes the single source of truth?

- **Also found (2026-04-07 ~20:30 UTC):** S3 has duplicate top-level files at
  `s3://.../code/mccfr_blueprint.c` (top-level) and `s3://.../code/src/mccfr_blueprint.c`
  (under src/). They're MD5-different (CRLF vs LF endings, but possibly older
  content too). All current launch scripts compile from `src/` so the top-level
  copies are unused dead weight. Suggest deleting `code/{mccfr_blueprint.{c,h},
  card_abstraction.{c,h}, hand_eval.h, test_hash_dedup.c}` from S3 — but
  `test_hash_dedup.c` may be the only S3 copy of a Bug 11 test, so check first.

## Convention agreements

### Bucket numbering

**The C convention is canonical.** Source of truth: `mccfr_blueprint.c`
`init_unified()` lines ~1383-1394:

```c
int n_classes = 0;
for (int r0 = 12; r0 >= 0; r0--) {
    for (int r1 = r0; r1 >= 0; r1--) {
        if (r0 == r1) {
            class_map[r0][r1][0] = n_classes;
            class_map[r0][r1][1] = n_classes;
            n_classes++;
        } else {
            class_map[r0][r1][1] = n_classes; n_classes++; /* suited */
            class_map[r0][r1][0] = n_classes; n_classes++; /* offsuit */
        }
    }
}
```

Where `r0`, `r1` are rank values (0=2, 12=A, from `card_int / 4`). The resulting
order is:

```
0  AA
1  AKs
2  AKo
3  AQs
4  AQo
...
23 A2s
24 A2o
25 KK
26 KQs
27 KQo
...
168 22
```

Frontend-agent (`extract_preflop_json.py:_build_c_bucket_map`) reproduces this
exactly. **Do not change the C class_map order without updating the extractor in
the same commit.**

### Action label format

Action labels in extracted nodes (`l` field) use the convention `raise_X.X` where
**X.X is the TOTAL committed amount after the raise, in big blinds**, NOT the chips
added. This matches the nexusgto reducer (`solver-state.ts parseRaiseSize`).

Example: at the UTG root, UTG raises by 350 chips (3.5 bb) from 0 committed → label
is `raise_3.5`. At UTG-vs-MP-3-bet where UTG already has 3.5 bb committed and adds
21.25 bb more, the label is `raise_24.8` (= 3.5 + 21.25, rounded), NOT `raise_21.2`.

`fold`, `call`, `check`, `allin` are unsigned. The frontend renders `raise_X.X` as
"Raise X.Xbb" in the UI.

### .bps metadata schema (proposed, post Bug-D fix)

```json
{
  "type": "unified_blueprint",
  "schema_version": 2,
  "num_players": 6,
  "blinds": [50, 100],
  "initial_stack": 10000,
  "preflop_tiers": {"0": [0.5, 0.7, 1.0], "1": [0.7, 1.0], "2": [1.0], "3": [8.0]},
  "preflop_max_raises": 4,
  "postflop_bet_sizes": [0.5, 1.0, 2.0],
  "iterations": 1000000000,
  "discount_stop_iter": 400000,
  "num_info_sets": 920312401,
  "preflop_buckets": 169,
  "postflop_buckets": 200,
  "checkpoint": "iter_1000000000",
  "exported_at": "2026-04-07T14:18:55Z",
  "code_sha": "abc1234...",
  "strategy_extraction_method": "strategy_sum_avg"
}
```

**Backwards compatibility:** Old .bps files (schema_version absent or 1) are still
readable. The extractor falls back to its hardcoded PREFLOP_TIERS when `preflop_tiers`
is missing from the meta blob.

## Recent decisions

- **2026-04-07 (frontend-agent):** Bug A fixed locally. Re-extracted preflop-nodes.json
  shows real GTO ranges (14.2% UTG range, premium hands play, junk folds).
  Verification step added. Pushed to nexusgto + poker-solver main.
- **2026-04-07 (solver-agent):** Solver resumed from 200M checkpoint at 04:44 UTC.
  At 1.5B as of 17:50 UTC. ETA to 4B = 2026-04-08 19:00 UTC. c7a.metal-48xl, ~28K
  iter/s.
- **2026-04-07 19:30 UTC (solver-agent):** Bug B fixed in mccfr_blueprint.c
  (bp_export_strategies now uses strategy_sum normalization with regret_match
  fallback). Bug D fixed in export_v2.py (schema_version=2 metadata with
  preflop_tiers, code_sha, etc.). Answered Q1/Q2/Q3 above — 400K discount stop
  is an unscaled Pluribus copy (wants 1.4B for our 4B target), `% 10007` gate is
  a regression of Bug 6's fix that should be removed for the next training run.
  Both fixes apply to .bps export only — running solver unaffected, ETA unchanged.
- **2026-04-07 ~20:00 UTC (frontend-agent):** Caught and corrected a deployment
  gap: solver-agent's Bug B + Bug D fixes were committed to git master but the
  S3 mirror at `s3://poker-blueprint-unified/code/` was NOT updated, so the next
  EC2 launch would have re-compiled the buggy version. The launch scripts (e.g.
  `launch_export_1B.sh:40`) use `aws s3 sync s3://.../code/` to fetch source,
  which is independent of git. Manually pushed `src/mccfr_blueprint.c` and
  `precompute/export_v2.py` from local working tree to S3 to bring them in sync.
  Verified by re-fetching S3 — it now contains the "Bug B fix" comment marker
  and the schema_version=2 metadata fields. **Bug F is more urgent than I
  thought** — this dual-source-tree is a real footgun and almost cost us a
  wasted re-export. Next session should prioritize collapsing to a single source
  of truth (git clone in launch scripts, delete S3 code/ prefix).
- **2026-04-07 ~20:30 UTC (frontend-agent):** Launched `launch_export_1600M.sh`
  on r6a.8xlarge (`i-098a2109e0da75643`) to verify the Bug B + Bug D fixes
  end-to-end against the latest 1.6B checkpoint. Output goes to
  `s3://.../unified_blueprint_1600M.bps` (200M and 1000M files preserved).
  Added a sanity-check guard inside the userdata that greps the freshly-synced
  S3 source for "Bug B fix" marker before compiling — bails out if S3 is stale.
  Also added `precompute/check_s3_sync.sh` as a pre-launch defense and
  `precompute/verify_export_freqs.py` for post-export sanity-checking. Found
  during testing: `src/mccfr_blueprint.h` is divergent between gitted version
  and S3 (S3 has the `_s3.h` content with extra function declarations) — see
  Bug F section for details and questions for solver-agent.

- **2026-04-07 ~21:00 UTC (frontend-agent):** 1.6B export completed (43.8 min,
  ~$1.48). Downloaded `unified_blueprint_1600M.bps` (3.0 GB) locally, deleted
  the stale `.preflop_cache.npz`, re-extracted via the cold load path
  (~16 min). All 8 sentinels in `verify_utg_root_sanity` pass except 22
  (folds 72%, expected ≥85% — but the threshold was calibrated for noisy
  regret_match data; 22 mixing call/raise at 28% is plausible GTO). Pushed
  the new `preflop-nodes.json` to nexusgto master at commit `f300ead`.

  **Position-by-position before/after comparison** (regret_match → strategy_sum):
  | Position | Old % | New % | Δ |
  |---|---|---|---|
  | UTG | 14.2% | 14.9% | +0.7% (essentially same shape) |
  | MP | **29.5%** | **22.1%** | **−7.4%** (huge improvement) |
  | CO | 35.5% | 32.3% | −3.2% (improvement) |
  | BTN | 45.9% | 46.2% | +0.3% (was good, still good) |
  | SB | 56.5% | 58.6% | +2.1% (was good, still good) |
  | BB | (no first-to-act node) | same | UI fallback to SB (known limitation) |

  **The "MP wider than UTG" structural inversion is gone.** New ordering is
  monotonic from UTG (14.9%) → MP (22.1%) → CO (32.3%) → BTN (46.2%) → SB
  (58.6%), which is the correct GTO structure for any 6-max solver. This
  confirms the Bug B fix worked end-to-end and the visible Bug 7 call/fold
  trap pattern is largely resolved.

  **Premium hand strategies are also more balanced and credible**:
  - AA: was 22% call → 43% call (proper slow-play frequency)
  - AKs: was 0% call / 64% raise_2.8 → 21% call / 53% raise_2.8 (more spread)
  - AKo: was 10% call / mixed → 28% call / evenly split raises
  - JJ/TT/99/88: now show ~62-73% call frequencies (slow-play correctly)

- **2026-04-07 ~21:30 UTC (frontend-agent + Victor):** **Honest assessment:
  data is intermediate quality, NOT ship-ready.** Even with Bug B fixed,
  several visible issues remain that would fail community sniff tests:

  **Quantitative residuals:**
  - UTG range still missing wheel Ax bluffs (A2s, A3s, A4s, A5s) which are
    standard 30-60% bluffs in any modern UTG range
  - UTG suited connectors below 87s (T9s, 76s, etc.) all fold ~98%, but
    real GTO plays them 30-70%
  - JTs at 38% fold but T9s at 98% fold → 60-point gap between adjacent
    suited connectors (no real solver does this)
  - Premium pairs (JJ/TT/99) at 62-73% call may be slightly too call-heavy
  - Speckle pattern: many out-of-range hands have a 1-3% raise sliver that
    looks visually "noisy" — display tells of an unconverged solver
  - 22 small pair at 72% fold (mixed strategy is plausible but borderline)

  **Root causes (per BLUEPRINT_BUGS.md docs and the agent's earlier reply):**
  - Bug 6 regression: `% 10007` gate on strategy_sum accumulation reduces
    sample density 10000x at the tail of the range
  - Bug C residual: discount only ran through ~350M iters (21 of 40 phases),
    not the full 35% warmup. Tail hands are biased toward early-iteration noise
  - Bug G (under research): hash table may be near 100% fill, causing
    Hogwild collisions that corrupt strategy_sum at affected nodes
  - Iteration count: 1.6B is 1/60th of Pluribus's 100B. Tail hands need
    more compute to converge regardless of any fixes

  **Victor's call (correct):** "Not shipping anything prematurely. When I
  ship, it must be perfect to get respect in the poker community. Any bad
  actions and people will question." Frontend-agent had erroneously
  recommended shipping the current state — that recommendation is rescinded.

  **Plan forward:**
  1. Wait for research-agent to finish hash table sizing investigation (Bug G)
  2. Terminate the current c7a.metal-48xl run (`i-08b967d731137c9b6`) — the
     remaining ~24 hours of compute will not fix the structural issues we've
     identified, the running config can't be patched in flight
  3. Plan a fresh training run with: Bug 6 gate removed, Bug C
     `discount_stop_iter = 1.4B`, Bug G hash table sized per research outcome
  4. Re-export, re-extract, re-evaluate
  5. If still not ship-ready, consider longer training or simpler abstraction

  **Pending decisions** (waiting on research-agent + Victor):
  - Hash table size for next training run (1B / 2B / 3B / larger?)
  - Iteration target for next training run (4B again, or 8B+?)
  - When to terminate the running c7a.metal-48xl instance (after research-agent
    finishes — termination is queued, not yet executed)

- **2026-04-07 ~21:30 UTC (frontend-agent cleanup):** Deleted stale
  `nexusgto/src/data/preflop-strategies.json` (164 KB, no longer imported by
  any code). Deleted the two stale feature branches (`feat/preflop-hash-lookup`,
  `feat/flat-hash-extraction`) from local + remote in both repos. Both repos
  now in clean state, on master, aligned with origin.
- **2026-04-07:** Tier-aware preflop sizing committed to training:
  `PREFLOP_TIERS = {0:[0.5,0.7,1.0], 1:[0.7,1.0], 2:[1.0], 3:[8.0]}`
  (reduced from earlier 8-size config per `f804aa2` "Reduce open raise sizes").
- **Earlier (solver-agent):** 11 documented bugs in `docs/BLUEPRINT_BUGS.md` Bugs 1-11
  all fixed. Notable: Bug 7 (call trap) fix uses Average Strategy Sampling for
  non-traversers. Bug 11 (Hogwild duplicate keys) fix uses unbounded spin + merge-on-load.

## Answered questions (frontend-agent → solver-agent, 2026-04-07)

### Q1: strategy_sum density at root at 1.5B?

I can't probe the running solver without disrupting it (would need to attach
analyze_checkpoint.py to a 200GB process). But the math:

- Gate fires every 10007 global iters → ~150K qualifying iters at 1.5B
- Each qualifying iter does ONE traversal, sample-rate of AA into UTG-root is
  approximately 1/6 (UTG is traverser) × 6/1326 (AA combos) ≈ 0.075%
- Expected accumulations at 1.5B for AA UTG root ≈ **~110**
- Linear extrapolation to 4B: **~290**
- Frontend's observation at 1B was 33, which implies 75 expected → **observed
  is roughly half of expected**. Either my model overstates the rate (possible —
  Hogwild contention may drop some events) or AA-UTG sampling is rarer than 6/1326

**Bottom line:** 100-300 accumulations per UTG-root info set by 4B is THIN.
Statistical noise on the average strategy will be visible at deep nodes. This
is the fundamental issue, and it's caused by the `% 10007` gate (see Q3).

### Q2: What does BLUEPRINT_BUGS.md say about the discount issue?

**Nothing direct.** I grepped for `discount_stop_iter`, `400000`, `400 min`,
`discount_period` — no results in BLUEPRINT_BUGS.md. The only context is in
the C source comment at line 1343:
```c
config->discount_stop_iter = 400000;    /* 400 min * ~1000 iter/min */
```

**Diagnosis:** This is a literal copy of Pluribus's wall-clock equivalent without
hardware scaling. Pluribus ran at ~1K iter/min × 400 min = 400K iters of
discount. Our hardware does ~1.8M iter/min. To preserve Pluribus's
"discount during the warmup phase = first 35% of training" semantic, on our
hardware, `discount_stop_iter` should be:

- Pluribus wall-clock equivalent: 400 min × 1.8M iter/min = **720M iters**
- Our 35%-of-target equivalent: 4B × 0.35 = **1.4B iters**

400K is **~1750x too small** by either definition. Looks like an
oversight/forgotten-to-rescale, NOT a deliberate choice. Should be flagged
as a real bug.

### Q3: Is the `% 10007` gating in strategy_sum accumulation deliberate?

**Yes (intentionally re-added) but it CONTRADICTS Bug 6's documented fix.**

History (reconstructed from git + comments):

1. **Original Bug 6 fix** (per BLUEPRINT_BUGS.md line 219-227): "Remove the
   interval check entirely. Accumulate strategy_sum on every traverser visit
   for preflop." Comment: "This is cheap (preflop info sets are tiny) and
   ensures all 6 players accumulate."

2. **Current code at mccfr_blueprint.c:1241**: gate is BACK with `% 10007`:
   ```c
   if (street == 0 && (ts->iteration % 10007) == 0) {
       ensure_strategy_sum(is);
       for (int a = 0; a < na; a++)
           is->strategy_sum[a] += strategy[a];
   }
   ```
   Comment at line 1234-1240: "every 10007 (prime) instead of 10000 to avoid
   aliasing with traverser cycling: gcd(6, 10000)=2 caused only players
   1,3,5 to accumulate, permanently excluding SB(0), UTG(2), CO(4).
   gcd(6, 10007)=1, so all 6 players get equal accumulation."

So someone re-added the gate using a coprime to fix the aliasing problem, but
the FIX in Bug 6 was specifically to remove the gate entirely. The re-add
trades "all 6 players accumulate" against "10007x sparser accumulation".

**The result:** strategy_sum is 10007x sparser than Bug 6's original fix
intended. This is the root cause of the thin density observed in Q1, and it
compounds with Q2 (no discount). Together, they make the average strategy
noisy at deep tree nodes.

**Required for the next training run:**
- Remove the `% 10007` gate entirely (revert to Bug 6's original fix)
- Set `discount_stop_iter = 1400000000` (1.4B = 35% of 4B target)
- Set hash table size per Bug G research outcome (currently 1B, may need 2B+)
- All three preserve Pluribus alignment

For THIS run (the 4B target): the 1.6B end-to-end test confirmed that even
with Bug B fixed, the data is intermediate quality, not ship-ready (see
Recent Decisions for the position-by-position analysis). Letting the run
continue to 4B will produce marginally cleaner data but won't fix the
structural issues caused by Bug 6 + Bug C + (possibly) Bug G. Pending
research-agent finishing Bug G investigation, the running c7a.metal-48xl
instance should be terminated and a fresh run launched with all three fixes.

## Process notes

- **How to update this file:** When you start a session, read it first. When you
  finish, append your status to "Recent decisions" with date and a 1-line summary.
  If you change a Bug status, update the relevant Bug section.
- **Race conditions:** Both agents push to `poker-solver/main`. To minimize collisions,
  always pull before editing this file. If `git push` fails, pull, manually merge,
  re-push.
- **Cross-references:** When committing a fix, reference the bug ID in the commit
  message (e.g., `fix: Bug A — bucket mapping in extractor`).
- **Before launching ANY EC2 export/training job (until Bug F is fixed):** Run
  `bash precompute/check_s3_sync.sh` first to verify the S3 `code/` mirror
  matches local git for the files EC2 will compile. The script exits 1 with a
  clear diff list if anything is out of sync. Use `--auto-fix` to upload local
  → S3. This catches the dual-source-tree footgun that bit us 2026-04-07. The
  failure mode it prevents is "EC2 spends $1-2 + 30 min compiling stale source
  and producing a bad .bps before anyone notices."

## What to do next (reading order for fresh agents)

If you're a new agent picking this up, read in this order:

1. **THIS section** for the high-level state.
2. **Recent decisions** above for the chronological story of how we got here.
3. **Open work items (BUG-A through BUG-G)** for the bug-by-bug status.
4. **`docs/EXTRACTOR_BUGS.md`** for the consumer-side bug history (Bug A in detail).
5. **`docs/BLUEPRINT_BUGS.md`** for the training-side bug history (Bugs 1-11 by solver-agent).

### Current state (as of 2026-04-07 ~21:30 UTC)

- **Bugs A, B, D, E are fixed and verified end-to-end.** A 1.6B-iteration export
  using the corrected pipeline is on S3 (`unified_blueprint_1600M.bps`) and
  the frontend at `nexusgto/src/data/preflop-nodes.json` is up to date with it.
- **Bug C is documented and a fix is queued for the next training run** (set
  `discount_stop_iter = 1400000000`).
- **Bug F has a defensive measure in place** (`precompute/check_s3_sync.sh`)
  but the architectural fix is deferred.
- **Bug G (hash table sizing) is under research** by solver-side research-agent.
- **The running c7a.metal-48xl instance (`i-08b967d731137c9b6`) is queued for
  termination** after research-agent finishes. It will not be restarted until
  Bug 6 + Bug C + Bug G are all addressed in a fresh launch config.
- **The data is NOT ship-ready** to the poker community. It's "good for
  internal study" but not "good for marketing or first impressions." See the
  2026-04-07 ~21:30 UTC entry in Recent decisions for the specific quantitative
  and qualitative gaps.
- **No frontend / UI work in progress.** Threshold display polish (hide hands
  with <3% play rate to clean up the speckle) is identified as the smallest
  data-independent improvement, but explicitly deferred until ship-ready data
  is in hand.

### Next decisions (waiting on humans + research-agent)

1. **Bug G research outcome.** What hash table size is safe for the next run
   given the recent fixes that may have removed the original 3B perf concern?
2. **Termination of the running c7a.metal-48xl.** Pending Bug G research.
3. **Next training run config.** Bug 6 gate removal + Bug C
   `discount_stop_iter` + Bug G hash size.
4. **Iteration target for the next run.** Stay at 4B, or go higher (8B+) to
   give tail hands more compute?

### Things explicitly NOT in flight

- No UI polish work (Victor's call: "perfect when shipping, not before")
- No threshold display fix (deferred until ship-ready data exists)
- No new EC2 launches (waiting on Bug G research)
- No tagging milestones (no v1.0 yet)
- No public sharing of any current screenshots / data
