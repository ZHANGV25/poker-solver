# Architecture

> See [`STATUS.md`](STATUS.md) for current state and forward plan. This doc is the static component overview.

## Big picture

The solver is split into two halves that share a `.bps` blueprint file:

```
   OFFLINE TRAINING (CPU, EC2)              REALTIME INFERENCE (CPU + GPU, local)
   ──────────────────────────────           ────────────────────────────────────
   precompute/blueprint_worker_unified.py   python/hud_solver.py
            │                                      │
            ▼                                      ▼
   src/mccfr_blueprint.c    ──── .bps ────► python/leaf_values.py
   src/card_abstraction.c                          │
            │                                      ▼
            ▼                              src/cuda/street_solve.cu
   regrets_*.bin (checkpoint)                      │
            │                                      ▼
            ▼                              strategy for the current decision
   precompute/export_v2.py
            │
            ▼
   .bps blueprint
```

The blueprint training runs once (~3 days on a 192-core box) and produces a static `.bps` file. The realtime inference loads that `.bps` and uses it for preflop lookups + as a leaf-value source for GPU re-solving on rounds 2–4.

## Components

### Blueprint training (CPU, EC2)

**`src/mccfr_blueprint.c`** — Production N-player external-sampling MCCFR.
- 2–6 player support, OpenMP Hogwild parallelism (lock-free atomic CAS on slot-occupied flags)
- 4-byte int regrets with -310M floor and `BP_REGRET_CEILING` (~2.1B) ceiling
- Linear CFR discounting on the first ~3.5% of training, 10-minute-equivalent intervals
- Negative-regret pruning (95% of iters skip actions with regret < -300M; never on river or actions leading to terminals)
- Card abstraction via 169 lossless preflop classes + 200-bucket k-means on flop/turn/river
- Cumulative-investment tracking for correct multi-street payoffs
- Per-thread arena allocator for slot regrets (no global lock contention)
- ~30K iter/s on 192-core c7a.metal-48xl

**`src/card_abstraction.c`** — Fast EHS computation + bucketing.
- Monte Carlo equity computation via `hand_eval.h` (7-card evaluator)
- 1326 hands × 200 samples ≈ 0.2 seconds for one board
- 169 lossless preflop classes (suit-isomorphic)
- 200-bucket k-means on `[EHS, PPot, NPot]` features for postflop streets
- Note: river effectively uses 1D EHS (PPot/NPot are zero with all 5 cards known) — see [STATUS.md §8](STATUS.md#8-river-bucket-abstraction-has-mild-degeneracy-not-catastrophic) for the implication.

**`precompute/blueprint_worker_unified.py`** — Training driver.
- Loads the C solver via ctypes
- Sets tiered preflop sizing (3-2-1-1)
- Periodic checkpoint to S3 (`regrets_*.bin` and `regrets_latest.bin`)
- Probe extraction for live monitoring (UTG/BTN strategies + raw regrets)
- Texture cache load/save (avoids 65-min flop bucket precompute)

### Realtime inference (CPU + GPU, local)

**`python/hud_solver.py`** — Decision pipeline for one hand.
- Loads the blueprint `.bps`
- For each villain action: narrows opponent ranges via Bayesian update (`range_narrowing.py`)
- For each hero decision:
  - Preflop: blueprint lookup
  - Postflop: GPU re-solve of the current street, with leaf values from the blueprint
- Off-tree opponent bets get pseudoharmonic interpolation (`off_tree.py`)

**`src/cuda/street_solve.cu`** — N-player single-street GPU solver.
- 2–6 player subgame solving in one betting round
- Level-batched Linear CFR with regret-based pruning
- 4 continuation strategies per player at depth-limit leaves (unmodified, fold-bias, call-bias, raise-bias)
- A3 strategy freezing (hero's already-chosen actions don't change)
- ~67ms river / ~236ms flop on RTX 3060

**`python/leaf_values.py`** — Depth-limited continuation values.
- Computes per-action EVs at the start of the next street, used as leaf values during GPU re-solve
- Uses Pluribus-style 4 continuation strategies with 5x bias multiplier
- Currently has a fallback path that uses equity-only values (when the blueprint .bps doesn't include per-action EVs — see Phase 1.3 status in [STATUS.md](STATUS.md#v3-commit-status))

**`python/range_narrowing.py`** — Bayesian range tracking.
- Maintains probability distributions over the 1326 possible hole-card combos for each opponent
- Updated via Bayes' rule on every observed action

**`python/off_tree.py`** — Pseudoharmonic interpolation.
- Maps opponent bet sizes to nearest blueprint-tree sizes (Ganzfried & Sandholm 2013)
- Used when opponent bets don't match any tree size exactly

### Export pipeline

**`precompute/export_v2.py`** — Converts trained regrets to a `.bps` file.
- Loads a `regrets_*.bin` checkpoint via the C solver
- Iterates the hash table, normalizing per-info-set strategies
- Writes the `.bps` format consumed by `python/blueprint_io.py`

## Decision pipeline (one hand)

```
PREFLOP — every betting decision:
  Look up the blueprint strategy at (player, action_history, hand_class)

POSTFLOP — every betting decision:
  1. Build the current subgame from the start of this betting round
  2. Initialize ranges from the narrowed opponent ranges
  3. Run GPU CFR for ~2000 iterations on the subgame
  4. Use blueprint as leaf-value source (depth-limited via 4 continuation strategies)
  5. Sample action from the resulting strategy
```

## Performance

| Component | Speed | Hardware |
|---|---|---|
| Blueprint training | ~30K iter/s | c7a.metal-48xl, 192 cores |
| Blueprint training | ~28K iter/s observed | (with hash table at 45% load) |
| Card abstraction (EHS) | ~0.2 s / 1326 hands | Modern CPU, 1 thread |
| GPU street re-solve (river) | ~67 ms | RTX 3060, 200 hands, 200 iters |
| GPU street re-solve (flop) | ~236 ms | RTX 3060, 200 hands, 200 iters |
| Realtime CFR (default) | 2000 iters per re-solve | trainer mode (no latency budget) |

## File-to-feature mapping

| Feature | Files |
|---|---|
| MCCFR core | `src/mccfr_blueprint.c`, `src/mccfr_blueprint.h` |
| Card abstraction | `src/card_abstraction.c`, `src/card_abstraction.h`, `src/hand_eval.h` |
| GPU subgame solver | `src/cuda/street_solve.cu`, `src/cuda/street_solve.cuh` |
| Training driver | `precompute/blueprint_worker_unified.py` |
| Export | `precompute/export_v2.py` |
| Blueprint I/O | `python/blueprint_io.py`, `python/blueprint_store.py` |
| Realtime decision pipeline | `python/hud_solver.py` |
| Range narrowing | `python/range_narrowing.py` |
| Leaf values | `python/leaf_values.py` |
| Off-tree mapping | `python/off_tree.py` |
| Solver pool / GPU wrapper | `python/solver_pool.py`, `python/street_solver_gpu.py` |
| Tests + analysis | `tests/*.py`, `tests/*.c`, `verification/*` |

## Cross-references

- Parameters and any Pluribus deviations: [`docs/SOLVER_CONFIG.md`](docs/SOLVER_CONFIG.md)
- Forward plan and current state: [`STATUS.md`](STATUS.md)
- Realtime backlog: [`docs/REALTIME_TODO.md`](docs/REALTIME_TODO.md)
- Pluribus paper extraction: [`pluribus_technical_details.md`](pluribus_technical_details.md)
