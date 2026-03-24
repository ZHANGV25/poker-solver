# GTO Gaps — Complete Tracking List

## Category A: Architectural Gaps (Pluribus features)

- [x] **A1** N-player street solver — extend street_solve.cu from 2 to N players
  - Status: DONE (2026-03-24) — tested 2-player + 3-player river
- [x] **A2** N-player blueprint — external-sampling MCCFR for 2-6 players
  - `src/mccfr_blueprint.c` + `.h`, compiled to `build/mccfr_blueprint.dll`
  - Rewritten 2026-03-24: OpenMP, int32 regrets, pruning, card abstraction, payoff fixes
  - Status: DONE — production-ready for EC2
- [~] **A3** Strategy freezing — freeze own strategy at passed nodes
  - Python tracking infrastructure added
  - CUDA frozen_mask kernel change needed
  - Status: PARTIAL
- [ ] **A4** Warm-start — persist regrets between solves
  - Status: NOT STARTED
- [x] **A5** Run precompute — pipeline ready with blueprint_worker.py + launch_blueprint.sh
  - Status: READY TO RUN (needs preflop integration first)
- [x] **A6** Card abstraction — 200-bucket EHS percentile bucketing
  - `src/card_abstraction.c` + `.h`, fast C implementation
  - Status: DONE (2026-03-24)
- [ ] **A7** 6-player preflop solver — extend from 2-player to all positions simultaneously
  - Current: 2-player CFR+ over 169 classes (preflop_solver.c)
  - Needed: Full 6-position tree with all actions
  - Status: NOT STARTED

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
| A: Architecture | 7 | 4 (A1,A2,A5,A6) | A3 partial, A4/A7 not started |
| B: Abstractions | 6 | 5 | B5 not started |
| C: Research | 1 | 1 | — |
| D: Minor | 4 | 3 | D3 |

## Critical Path to Working Blueprint
1. **A7**: 6-player preflop solver (outputs mixed frequencies for all positions)
2. **B5**: Wire preflop strategy → postflop starting ranges
3. **A5**: Run EC2 blueprint generation (1,755 textures × 6-player MCCFR)
4. Wire blueprint leaf values into GPU real-time search (street_solve.cu)
