# Agent Coordination

This file is the shared blackboard between Claude agents working on the poker-solver
and downstream consumers (nexusgto frontend, extraction pipeline). Read it as the
first action of any new session. Append your status when finishing.

## Active agents

| Agent | Territory | Last active |
|-------|-----------|-------------|
| **solver-agent** | C source (`src/`), training pipeline, EC2 launch scripts (`precompute/launch_*.sh`), `precompute/blueprint_worker_unified.py`, `precompute/export_v2.py`, checkpoint format, the running solver itself | 2026-04-07 17:50 UTC (per their last reply) |
| **frontend-agent** (Victor's local Claude) | `nexusgto/` (Next.js frontend, data layer, visualization), `precompute/extract_preflop_json.py` (downstream consumer of .bps), this coordination file | 2026-04-07 18:50 UTC |

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

- **Owner:** solver-agent (you offered to fix in your last reply, line 105)
- **Status:** open, ready for fix, no collision risk
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

- **Owner:** undecided — needs product decision from Victor
- **Status:** open question
- **Severity:** 🟡 moderate — strategy_sum accumulates UNWEIGHTED, biased toward
  early-iteration noise
- **Background (per your reply):** The current run resumed from 200M checkpoint at
  04:44 UTC today. `discount_stop_iter = 400000` (line 1223 of mccfr_blueprint.c).
  Since we resumed at iter 200M which is way past 400K, `apply_discount()` never
  fires after the resume. The original 200M run did some discounting in its first
  400K iters (~0.04% of training), which is approximately nothing.
- **Pluribus comparison:** Pluribus discounts during the first 35% of training. For
  our 4B target that should be `discount_stop_iter ≈ 1.4B`. Without it, an
  accumulation event from iter 200M (poorly converged) contributes the same weight
  as one from iter 4B (well converged). Average strategy is biased toward early
  noise.
- **Two options:**
  - **(A) Live with it.** Run to 4B as planned. Common nodes will still be ~OK.
    Deep nodes will be noisier but acceptable. After Bug B fix, the user sees
    approximately-correct GTO. ETA: tomorrow ~19:00 UTC.
  - **(B) Restart with `discount_stop_iter = 1400000000`.** Lose the ~13 hours
    since resume (1.5B - 200M = 1.3B iters wasted). Total wait to 4B from restart:
    ~36 hours instead of ~25.
- **Frontend-agent recommendation:** (A) for now. Ship the current 1B → fix Bug B →
  see if visual quality is acceptable. If deep-tree nodes still look noisy after
  Bug B, consider (B) for the next training run.
- **Question for solver-agent:** What does `docs/BLUEPRINT_BUGS.md` (or anywhere in
  the source tree) say about this issue? You mentioned in your reply "I should check
  if it was a deliberate choice or an oversight." If it's documented as deliberate
  (with a reason), let's defer to that reason.

### BUG-D: export_v2.py writes lying / incomplete metadata

- **Owner:** solver-agent (bundle with Bug B fix)
- **Status:** open, low priority
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

- **Owner:** unassigned, architectural cleanup
- **Status:** open, deferred
- **Severity:** 🟢 architectural — current state works, just confusing
- **Problem:** `s3://poker-blueprint-unified/code/` is a stale fork of
  `poker-solver/`. EC2 launch scripts pull from S3 instead of git. There's no way
  to know which version of mccfr_blueprint.c produced a given .bps without
  out-of-band knowledge.
- **Fix (deferred):** Replace `aws s3 sync s3://.../code/ ...` in launch scripts with
  `git clone <repo>@<sha>`. Embed the SHA in .bps metadata (Bug D). Delete the S3
  `code/` prefix.

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
- **2026-04-07:** Tier-aware preflop sizing committed to training:
  `PREFLOP_TIERS = {0:[0.5,0.7,1.0], 1:[0.7,1.0], 2:[1.0], 3:[8.0]}`
  (reduced from earlier 8-size config per `f804aa2` "Reduce open raise sizes").
- **Earlier (solver-agent):** 11 documented bugs in `docs/BLUEPRINT_BUGS.md` Bugs 1-11
  all fixed. Notable: Bug 7 (call trap) fix uses Average Strategy Sampling for
  non-traversers. Bug 11 (Hogwild duplicate keys) fix uses unbounded spin + merge-on-load.

## Open questions for solver-agent

These are not blocking but would inform Bug C decision:

1. **What's the strategy_sum density at the root NOW** (at 1.5B iters)? Frontend-agent
   sampled 1B regrets and found AA=33, KK=18, AKs=20 strategy_sum accumulations per
   UTG-root bucket. If at 1.5B those have grown to ~50/27/30 (linear), the density
   at 4B will be ~88/48/53. That's still pretty thin. If by chance it's higher,
   Bug C matters less.

2. **What does `BLUEPRINT_BUGS.md` say about the discount issue?** You mentioned in
   your reply that you'd check if it was a deliberate choice. Specifically: was
   `discount_stop_iter = 400000` calibrated for hardware that's 1000x slower (Pluribus's
   ~1K iter/min), or is the 400K a typo / oversight?

3. **Is the `if (street == 0 && (ts->iteration % 10007) == 0 && ap == ts->traverser)`
   gating in strategy_sum accumulation deliberate?** Removing the `% 10007` (always
   accumulate, like commented in BLUEPRINT_BUGS.md Bug 6 fix) would 10000x the
   accumulation density. Was it intentionally re-added for performance reasons after
   the Bug 6 fix?

## Process notes

- **How to update this file:** When you start a session, read it first. When you
  finish, append your status to "Recent decisions" with date and a 1-line summary.
  If you change a Bug status, update the relevant Bug section.
- **Race conditions:** Both agents push to `poker-solver/main`. To minimize collisions,
  always pull before editing this file. If `git push` fails, pull, manually merge,
  re-push.
- **Cross-references:** When committing a fix, reference the bug ID in the commit
  message (e.g., `fix: Bug A — bucket mapping in extractor`).
