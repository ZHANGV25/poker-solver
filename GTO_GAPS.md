# GTO Gaps — Complete Tracking List

## Category A: Architectural Gaps (Pluribus features)

- [x] **A1** N-player street solver — extend street_solve.cu from 2 to N players
  - Status: DONE (2026-03-24) — exact N-player showdown (2026-03-26)
- [x] **A2** N-player blueprint — external-sampling MCCFR for 2-6 players
  - `src/mccfr_blueprint.c` + `card_abstraction.c` (must compile together)
  - Bucket-in-key info set architecture (2026-03-25), k-means bucketing, per-street buckets
  - Status: DONE — running on EC2 c5.18xlarge since 2026-03-26
- [x] **A3** Strategy freezing — freeze own strategy at passed nodes
  - `ss_apply_freeze` CUDA kernel + `frozen_action` field in SSTreeData
  - Python `set_frozen_actions()` walks tree and marks hero's decision nodes
  - Status: DONE (2026-03-26)
- [ ] **A4** Warm-start — persist regrets between solves
  - Status: NOT STARTED (nice-to-have, not blocking)
- [x] **A5** Run precompute — unified pipeline with blueprint_worker_unified.py
  - EC2 compute launched 2026-03-26, watchdog on t3.micro auto-relaunches
  - Status: RUNNING (~April 3 completion)
- [x] **A6** Card abstraction — 200-bucket k-means on [EHS, pos_potential, neg_potential]
  - `src/card_abstraction.c` + `.h`, k-means wired into blueprint init (2026-03-26)
  - Per-street bucket recomputation for turn/river
  - Status: DONE
- [x] **A7** 6-player preflop solver — unified preflop-through-river in mccfr_blueprint.c
  - 169 lossless preflop classes, correct blind posting and acting order
  - Status: DONE (integrated into unified solver, preflop_solver.c removed)

## Category B: Necessary Abstractions

- [x] **B1** Bet sizes — configurable, currently [0.5, 1.0] for blueprint
- [x] **B2** Max raises per street — 3
- [x] **B3** Max hands per player — 1326 (all combos), bucketed to 200
- [x] **B4** CFR iterations — configurable, ~1M+ needed for convergence
- [ ] **B5** Preflop ranges — currently semi-binary pairwise, need continuous 6-player
- [x] **B6** Off-tree bet mapping — pseudoharmonic

## Category C: Research (Completed)

- [x] **C1** GPU batch outcome-sampling MCCFR — novel research
  - Built, tested, benchmarked (17M traj/s on RTX 3060)
  - Finding: CPU external sampling is superior for blueprints
  - GPU MCCFR useful for research, not production blueprint
  - Status: DONE, documented as research finding

## Category D: Minor/Cosmetic

- [x] **D1** Weight floor 0.005 — acceptable
- [x] **D2** float32 precision — acceptable (int32 regrets in blueprint)
- [ ] **D3** No exploitability check — could add adaptive stopping
- [x] **D4** Normalization max=1.0 — mathematically equivalent

## Summary

| Category | Total | Done | Remaining |
|----------|-------|------|-----------|
| A: Architecture | 7 | 6 (A1-A3,A5-A7) | A4 (warm-start, nice-to-have) |
| B: Abstractions | 6 | 5 | B5 not started |
| C: Research | 1 | 1 | — |
| D: Minor | 4 | 3 | D3 |

## Critical Path to Working Blueprint
1. **A7**: 6-player preflop solver (outputs mixed frequencies for all positions)
2. **B5**: Wire preflop strategy → postflop starting ranges
3. **A5**: Run EC2 blueprint generation (1,755 textures × 6-player MCCFR)
4. Wire blueprint leaf values into GPU real-time search (street_solve.cu)
