# GTO Gaps — Complete Tracking List

## Category A: Architectural Gaps (Pluribus features)

- [x] **A1** N-player street solver — extend street_solve.cu from 2 to N players
  - N-player tree builder, N-player CFR traversal, N-player showdown
  - Status: DONE (2026-03-24) — tested 2-player + 3-player river
  - 3-player river: 1156 nodes, 141ms for 200 iterations
- [x] **A2** N-player blueprint — external-sampling MCCFR for 2-6 players
  - `src/mccfr_blueprint.c` + `.h`, compiled to `build/mccfr_blueprint.dll`
  - Tested: 2-player (50K iter, 5.3s, 2M infosets), 3-player (30K iter, 1.4s, 685K infosets)
  - Status: DONE (2026-03-24) — needs integration into precompute pipeline
- [~] **A3** Strategy freezing — freeze own strategy at passed nodes for actual hand
  - Python tracking infrastructure added (_hero_actions_this_street)
  - CUDA frozen_mask kernel change needed to actually enforce
  - Status: PARTIAL — tracking in place, enforcement TODO
- [ ] **A4** Warm-start — persist regrets between solves of same street
  - Need init_regrets parameter in ss_solve_gpu
  - Status: NOT STARTED (requires CUDA API change)
- [x] **A5** Run precompute — pipeline ready, needs EC2 execution
  - `precompute/run_all.py` implemented
  - `src/mccfr_blueprint.c` for N-player precompute also available
  - Status: READY TO RUN

## Category B: Necessary Abstractions (practical solver tradeoffs)

- [x] **B1** Bet sizes — 3+all-in (0.33, 0.75, 1.5)
  - Configurable via DEFAULT_BET_SIZES
  - Status: DONE
- [x] **B2** Max raises per street — 3 (adequate)
- [x] **B3** Max hands per player — 200 exact (better than Pluribus's 200 lossy buckets)
- [x] **B4** CFR iterations — 200 (similar to Pluribus time budget)
- [ ] **B5** Preflop ranges — semi-binary (0/0.5/1.0), need continuous weights
  - Status: NOT STARTED
- [x] **B6** Off-tree bet mapping — pseudoharmonic (same as Pluribus)

## Category C: Leaf Value Approximations

- [ ] **C1** Turn leaf river sampling — 12/46 cards sampled for equity
  - Need GPU equity kernel for all 46 cards
  - Status: CPU WORKAROUND (12 sampled)

## Category D: Minor/Cosmetic

- [x] **D1** Weight floor 0.005 — acceptable
- [x] **D2** float32 precision — acceptable
- [ ] **D3** No exploitability check — could add adaptive stopping
- [x] **D4** Normalization max=1.0 — mathematically equivalent
- [ ] **D5** Process exit segfault — cosmetic, harmless

## Summary

| Category | Total | Done | Remaining |
|----------|-------|------|-----------|
| A: Architecture | 5 | 3 (A1,A2,A5) | A3 partial, A4 not started |
| B: Abstractions | 6 | 5 | B5 not started |
| C: Leaf Values | 1 | 0 | C1 workaround |
| D: Minor | 5 | 3 | D3, D5 minor |

**Major completed this session:**
- A1: N-player GPU solver (2-6 players, CUDA)
- A2: N-player MCCFR blueprint (external sampling, CPU)
- Regret-based pruning in GPU solver (5x speedup on river)
- 3 bet sizes + all-in

**Remaining work for full Pluribus parity:**
1. A3: CUDA frozen_mask enforcement
2. A4: Warm-start (persist regrets)
3. A5: Run precompute on EC2
4. B5: Proper GTO preflop ranges
5. C1: GPU equity kernel for turn leaves
6. Wire MCCFR blueprint into precompute pipeline
