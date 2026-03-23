# Concerns & Issues Log

## 2026-03-23: Initial Benchmark Results

### Issue 1: O(N*M) Showdown Is Catastrophic — FIXED
Precomputing hand strengths once in solver_init reduced 100-hand benchmark from
14,072ms to 66ms (213x speedup). Remaining O(N*M) with precomputed strengths is
fine for narrowed ranges (<100 hands). Upgrade to O(N+M) prefix-sum if needed.

### Issue 1 (original):
- With 40 hands: 112ms/100iter (894 iter/sec) — acceptable
- With 60 hands: 3228ms/100iter (31 iter/sec) — 29x slower than expected
- With 100 hands: 14072ms/100iter (7 iter/sec) — unusable
- **Root cause**: `compute_showdown_values()` calls `eval7()` for every (hero, opponent) pair
  at every showdown node on every iteration. eval7 is 21× eval5 calls per hand.
- **Fix needed**: Precompute all hand strengths once per board, sort, use O(N+M) prefix sums.
  Expected: O(N*M) → O(N+M) = ~100x speedup for 100-hand ranges.

### Issue 2: Malloc/Free Per Iteration
- `cfr_traverse()` allocates and frees reach arrays for every action at every node.
- With 57 nodes × 4 actions × 2 players × 100 iterations = ~45,000 malloc/free calls.
- **Fix needed**: Pre-allocate a scratch buffer and use stack-based allocation.

### Issue 2: Malloc/Free Per Iteration — FIXED
Replaced all malloc/free in cfr_traverse with stack-allocated arrays (MAX_HANDS).
No heap allocation in the hot loop.

### Issue 3: Strategy Convergence Not Validated — RESOLVED
- With 5 hands × 100 iterations: AhKh shows 51.5% check on a board where it has top pair
  top kicker. This might be correct (checking to trap) or might indicate a bug.
- Need to validate against a known solver (PioSOLVER or postflop-solver) on the same spot.
- **TODO**: Add exploitability computation to measure convergence.

### Issue 4: Fold Value Computation May Be Wrong — FIXED
- In `cfr_traverse` for fold nodes, the value computation uses `pot / 2` and `bets[traverser]`.
- Need to verify: when player X folds, player Y wins the pot. Y's CFV should be
  (pot - Y's contribution) * opponent_reach. X's CFV should be -(X's contribution) * opponent_reach.
- The current implementation may have sign/scaling issues.

### Issue 5: Leaf Values Not Implemented
- `NODE_LEAF` returns 0 for all hands. This makes the solver useless for non-river streets.
- The 4-continuation-strategy mechanism (Pluribus) is the next critical piece.
