# Poker Solver v2 Architecture

## Goal
Real-time depth-limited subgame solving for 6-max NLHE cash games.
Pluribus-inspired with Bayesian range narrowing at every street.

## Decision Pipeline

```
PREFLOP:  Scenario selection from ranges.json → starting ranges (instant)

FLOP (first to act):
  Blueprint lookup — exact per-texture, no abstraction. (instant)
  Narrow both ranges using blueprint P(action|hand).

FLOP (facing action):
  Narrow villain range using blueprint P(action|hand).
  Re-solve with narrowed ranges + 4 continuation strategies at leaves.
  (~268ms measured)

TURN:
  Narrow both ranges from flop actions.
  Compute river blueprint on-the-fly for this turn card (~200ms).
  Re-solve with narrowed ranges + 4 continuation strategies.
  (~400ms total)

RIVER:
  Narrow both ranges from flop + turn actions.
  Re-solve with narrowed ranges to showdown.
  (~186ms)
```

## Precomputed Data (EC2)

For each of 27 scenarios × 1,755 flop textures:
- **Flop root**: P(action|hand) for ~300 hands × 2 players × ~4 actions
- **Turn roots**: 47 turn cards × P(action|hand) for ~280 hands × 2 players

Storage: ~20 GB raw → ~1 GB LZMA compressed
EC2 cost: ~$25 (4-8 × c5.4xlarge spot, 12-25 hours)

## Solver Configuration (matching Pluribus)

- **Algorithm**: Linear CFR = DCFR(alpha=1, beta=1, gamma=1)
- **Strategy selection**: Final iteration (not average)
- **Subgame root**: Start of current betting round
- **Search depth**: End of current betting round
- **Leaf evaluation**: 4 continuation strategies (unmodified, fold×5, call×5, raise×5)
  evaluated against current narrowed ranges
- **Unsafe search**: No gadget game (mitigated by round-rooting + multi-strategy leaves)
- **Iterations**: Time-budgeted, 200-500 per subgame

## Measured Performance (solver v2, i7-13700K, single thread, 500 iter)

- River re-solve (80 hands): **180ms** (0.000% exploitability)
- Turn re-solve (80 hands): **203ms** (precomp 5ms + leaf 97ms + DCFR 102ms)
- 8 concurrent river solves: **103ms** wall time
- All decisions under 500ms

## Known Gaps

- Multiway pots: heads-up only. Heuristic adjustment for multiway (v2: real multiplayer solver).
- Off-tree bet sizes: map to nearest blueprint size for narrowing.
