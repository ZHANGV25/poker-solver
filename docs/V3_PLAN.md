# v3 Execution Plan

> **STATUS: Phase 1–3 SHIPPED.** All Phase 1–3 items are committed in `a82219b` and `48da71b`. Phase 1.3 is partially shipped (realtime side done, export-tool side still pending — see [`STATUS.md`](../STATUS.md#v3-commit-status)).
>
> For current state and forward plan, **read [`STATUS.md`](../STATUS.md)**, not this file. This document is preserved as the historical execution plan.

This is the canonical execution order for fixing every non-intentional deviation
from Pluribus, across both the blueprint solver and the realtime path. This doc
is the **WHEN** — for the **WHAT** of each item, see [`SOLVER_CONFIG.md`](SOLVER_CONFIG.md)
§11 (blueprint backlog) and [`REALTIME_TODO.md`](REALTIME_TODO.md) (realtime backlog).

**Use case context:** the realtime path was originally designed for sub-100ms response.
Now operating in trainer mode where seconds-to-minutes per re-solve is acceptable.
This unlocks several fixes that were previously blocked.

## Already done (intentional matches + completed v2 fixes)

These are confirmed Pluribus-aligned and tracked in `SOLVER_CONFIG.md`:

- All algorithm constants (BP_PRUNE_PROB=0.95, regret floor -310M, regret ceiling 2B, prune threshold -300M)
- Card abstraction (169 preflop, 200 per postflop street)
- Linear CFR discount formula
- Pluribus timing fractions (3.47% / 1.74% / 6.94% / 1.74%)
- External-sampling MCCFR core
- Bug B (probe symmetry), C/F (default config), E (% 10007 gate), γ (iterations_run race), α (launch perf), B/D (export bug + metadata) — fixed in v2
- Bug 11 (Hogwild duplicates) fix already in code

## Phase 1 — High impact, low cost (parallel with v2, ~1 day)  ✅ SHIPPED in a82219b

| # | Item | File:line | Status |
|---|---|---|---|
| 1.1 | **T1.1: bump CFR iterations 200 → 2000** in realtime solver | `python/hud_solver.py:506` (`DEFAULT_CFR_ITERATIONS`) | ✅ done |
| 1.2 | **T1.2: delete `python/multiway_adjust.py`** and remove callers | file deleted | ✅ done |
| 1.3 | **T2.1: per-action EVs for realtime leaf values** | `python/leaf_values.py` (+303 lines) | ⚠️ **PARTIAL** — realtime side done, but the export-tool side (`bp_export_regrets()` in `mccfr_blueprint.c` + `precompute/export_v2.py` schema_version bump) was NOT done. See [STATUS.md](../STATUS.md#v3-commit-status). |

## Phase 2 — Defense-in-depth fixes (parallel with v2, ~2-3 hours)  ✅ SHIPPED in 48da71b

These are small isolated bug fixes. None affect v2's output (defensive only)
but they need to be in the next-training-run code base.

All committed in 48da71b. Items 2.1–2.8 below match the order in V3_PLAN.md, not the order in the commit message.

| # | Item | File:line | Status |
|---|---|---|---|
| 2.1 | **Bug 1**: cap CALL against player stack | `mccfr_blueprint.c:1227-1232, 1305-1310` | ✅ done |
| 2.2 | **Bug 2**: lower-only `na` clamp on hash collision | `mccfr_blueprint.c:1186-1190` | ✅ done |
| 2.3 | **Bug 9**: NULL check after `arena_alloc` | `mccfr_blueprint.c:418-422` | ✅ done |
| 2.4 | **Bug 14**: texture cache double-allocation guard | `mccfr_blueprint.c:1626` | ✅ done |
| 2.5 | **Bug 10**: recompute `batch_size` at discount→post-discount boundary | `mccfr_blueprint.c:2143-2150` | ✅ done |
| 2.6 | **Bug 8**: `spin_until_ready` in `bp_get_strategy`/`bp_get_regrets` | `mccfr_blueprint.c:2392-2418, 2378-2424` | ✅ done |
| 2.7 | **F3**: `street == 0` filter in `apply_discount` | `mccfr_blueprint.c:1374-1378` | ✅ done |
| 2.8 | **F2**: remove dead `strategy_interval` field (or rename for ABI safety) | `mccfr_blueprint.h:102`, ~30 callers | ✅ done |

## Phase 3 — Quality / perf improvements (parallel with v2, ~3-4 hours)  ✅ SHIPPED in 48da71b

| # | Item | File:line | Status | Impact |
|---|---|---|---|---|
| 3.1 | **Bug 7**: replace texture lookup linear scan with hashmap | `mccfr_blueprint.c:268+` | ✅ done | 5-15% solver speedup |
| 3.2 | **Bug 6**: replace `hash_combine` with splitmix64 mixer for `action_hash` | `mccfr_blueprint.c:241-259` | ✅ done (splitmix64, not xxHash3 — same family, simpler implementation) | Eliminates `max_probe = 4096` clustering |

## Phase 4 — Wait for v2 to finish (~3 days, passive)

No work. Monitor v2 progress until it reaches 8B iters (~2026-04-11).

## Phase 5 — Verify after v2 finishes (~1 hour)

| # | Item |
|---|---|
| 5.1 | Verify the realtime solver works with the new (Phase 1.3) leaf-value code against v2's exported `unified_blueprint.bps` |
| 5.2 | Run on test spots, confirm 4 continuation strategies produce 4 distinct biased leaf values |
| 5.3 | Verify multiway path (Phase 1.2) works without heuristics in 3-way and 4-way scenarios |

## Phase 6 — Subgame depth analysis (research, ~3-5 days)

| # | Item | Effort |
|---|---|---|
| 6.1 | T3.1: build a multi-street CPU CFR for ground-truth comparison | ~2 days |
| 6.2 | Run on 30-50 representative spots | ~1-2 days |
| 6.3 | Measure single-street vs multi-street vs full-game L1 distance | ~1 day |

**Decision point at end of Phase 6:** if single-street stays within ~5% L1 of
full-game on most spots, skip Phase 7.1. Otherwise proceed.

## Phase 7 — Subgame depth and large gaps (size depends on Phase 6)

| # | Item | Effort | Conditional? |
|---|---|---|---|
| 7.1 | T5.1: multi-street GPU kernel | ~1-2 weeks | only if Phase 6 says it's needed |
| 7.2 | T4.1: preflop re-solve on large deviations | ~1 week | always |
| 7.3 | Realtime action set expansion (3 → 5-7 sizes) | ~1 day + tuning | always |

## Phase 8 — Optional / final cleanup

| # | Item | Notes |
|---|---|---|
| 8.1 | F1: literal Pluribus average strategy mechanism | **Recommended SKIP.** Our per-visit accumulation is mathematically lower-variance than the paper's. Implementing the paper's method makes the solver worse on every dimension except literal alignment. |
| 8.2 | Documentation cleanup — scrub stale "heads-up only" comments | Cosmetic |

---

## Summary

| Phase | When | Duration | Disturbs v2? |
|---|---|---|---|
| 1 | now | ~1 day | no |
| 2 | now (parallel) | ~2-3 hours | no |
| 3 | now (parallel) | ~3-4 hours | no |
| 4 | passive | ~3 days | n/a |
| 5 | after v2 | ~1 hour | no |
| 6 | after Phase 5 | ~3-5 days | no |
| 7 | after Phase 6 | 1-3 weeks | no |
| 8 | last | ~1-2 days | requires v3 retrain |

**Phases 1-3 can all be committed today and don't disturb the running v2.**
Total code change estimate: ~400-500 LOC across ~10 files.

## Decision log

| Date | Decision |
|---|---|
| 2026-04-07 | Approved Phases 1-3 for immediate execution. F1 (Phase 8.1) marked as recommended-skip. |
| 2026-04-07 | Phases 1-3 committed in `a82219b` and `48da71b`. Phase 1.3 partial (realtime side only). |
| 2026-04-08 | A fresh v3 retraining run is **not planned** — marginal blueprint quality improvement vs. cost. The Phase 1-3 realtime fixes work with v2's blueprint. See [`STATUS.md`](../STATUS.md). |
| 2026-04-08 | Subgame depth analysis (T3.1) and real-time multi-street search (T5.1) are the next high-value items. Tracked in [`REALTIME_TODO.md`](REALTIME_TODO.md), prioritized in [`STATUS.md`](../STATUS.md#forward-roadmap-next-month). |
