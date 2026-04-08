# Realtime solver TODO

Backlog for the realtime / subgame-solving side of the system. The blueprint
solver (`mccfr_blueprint.c` + `precompute/blueprint_worker_unified.py`) is
mostly Pluribus-aligned and tracked in [`SOLVER_CONFIG.md`](SOLVER_CONFIG.md).
This file tracks the realtime path (`python/hud_solver.py`,
`python/street_solver_gpu.py`, `python/leaf_values.py`,
`src/cuda/street_solve.cu`) and its known gaps vs Pluribus.

## Use case context

**Pivoted from HUD to trainer.** The realtime solver was originally written for
the ACR Poker HUD which needed sub-100ms response times. The current target use
case is a trainer / analysis tool where seconds-to-minutes per re-solve is
acceptable. **The latency budget is no longer load-bearing** for design choices.
Multiple gaps below were forced by the HUD latency budget and can be reopened
now that the constraint is gone.

## Priority 1 — easy wins from the latency relaxation

### T1.1 — Up `iterations` from 200 to 2000-5000

**Where:** `python/hud_solver.py:829`
```python
solver.solve(iterations=200)
```
**Why:** Linear CFR converges roughly as O(1/√T). 200 → 2000 is one extra log
unit of convergence quality. Pluribus runs 1-33 seconds per subgame, which is
thousands of iterations. We're capped at 200 because of the HUD latency budget,
which no longer applies.

**Change:** Make `iterations` a parameter with a sensible default like 2000.
Add a fast path (200) for any future low-latency callers, but make trainer
calls use the higher count.

**Effort:** 5 lines.

**Expected impact:** Largest single quality improvement available. Subgame
strategies become noticeably more converged on mixed-strategy spots
(bluff/value combos, mixed call/raise frequencies).

### T1.2 — Delete `python/multiway_adjust.py` and remove its callers

**Why:** The file is a relic from when the GPU solver was heads-up only. The
header comment says it explicitly:
> "Since the solver is heads-up only, when multiple players see the flop we
> apply heuristic adjustments to approximate multiway GTO behavior."

That comment is **outdated**. The GPU solver `src/cuda/street_solve.cu` is
N-player capable (`SS_MAX_PLAYERS = 6`), and `python/street_solver_gpu.py`
exposes a `player_ranges=[...]` interface that runs full N-player CFR.
`python/hud_solver.py:805-815` already wires up the N-player path.

Despite that, `hud_solver.py:561-562` and `:568-569` still apply
`adjust_multiway_strategy()` AFTER getting the N-player CFR result. This is:
- **Double-correcting** at best (CFR already gave you the multiway answer)
- **Contradictory** at worst (the heuristic might point the opposite direction
  from what CFR converged to)
- **Ungrounded** in any analysis — the multipliers (0.15 per extra opponent for
  value, 0.50 for bluffs, etc.) are arbitrary hand-tuned constants

**Change:**
1. Delete `python/multiway_adjust.py` entirely
2. Remove imports and call sites in `python/hud_solver.py` (5-6 lines)
3. Trust the N-player GPU solver. If it gives bad answers in some specific
   multiway situation, that's a bug in the solver, not something to paper over.

**Effort:** 30 minutes.

**Expected impact:** Cleaner, more principled multiway handling. May expose
real N-player CFR convergence issues that were previously masked by the
heuristics — those would be separate bugs to investigate.

## Priority 2 — Blueprint export fix (no restart needed)

### T2.1 — Extend .bps export with per-action EVs

**Where the breakage actually is:** the **blueprint training itself is fine** —
`mccfr_blueprint.c:1259-1270` does the standard MCCFR regret update
(`delta = action_values[a] - node_value`), regrets accumulate correctly, and
the strategy derived from regrets is Pluribus-aligned. The blueprint represents
the strategy, not EVs, and it doesn't need to.

The breakage is **only in the realtime leaf-value path**:
`python/leaf_values.py.compute_flop_leaf_equity` (called from
`hud_solver.py:754-779` on the v2 .bps path) computes leaf values as
"showdown equity from this position", which doesn't depend on what action
the player took at the leaf info set. So when the GPU CFR sits at a leaf
decision and asks "what's the value of each action?" it gets the SAME
number for every action.

This collapses Pluribus's 4 continuation strategies (unmodified, fold-bias,
call-bias, raise-bias) to a single equity value. The depth-limited solving
loses its main variance-reduction trick.

**Impact:** flop subgame strategies are slightly less accurate than Pluribus's
would be, particularly on close-call mixed-strategy spots. River play is
**unaffected** (river leaves are showdown which has exact values). Turn play
has a smaller effect (1 street to the leaf vs 2 for flop). The Brown & Sandholm
2018 NeurIPS paper showed depth-limited + continuation strategies stays within
~5% of full-solving exploitability; without continuation strategies the gap
grows to ~10-20%. For our flop solver, that's the gap we're paying.

The honest comment is in `python/hud_solver.py:740-751`:
> "Without full per-action turn EVs from the v2 blueprint, we compute per-turn-card
> equity directly. ... The 4 continuation strategy pairs all collapse to the same
> equity value since we lack per-action EVs to differentiate them. This is
> conservative — the GPU search refines leaf values via CFR iterations regardless."

**The fix path (no restart needed):**

The .bps export is a separate one-shot process from the running solver. The
running v2 just writes raw regret checkpoints periodically; the export tool
(`precompute/export_v2.py`) loads a checkpoint and produces a .bps. Modifying
the export tool doesn't affect the running solver.

**The right approach is a post-hoc tree walk at export time** that computes
per-action EVs from the converged blueprint strategies. Math: at every info
set `I`, `node_value(I) = sum_a strategy(I, a) * action_value(I, a)` where
`action_value(I, a) = expected immediate value + node_value(child(I, a))`.
This gives an **exact** computation of "EV of action a assuming both players
play the converged blueprint from this point on" — which is exactly what
the Pluribus continuation-strategy formula needs.

**Why post-hoc is strictly better than tracking-during-training:** if you
tracked EVs during training instead, you'd be averaging samples taken under
a *changing* strategy (early iterations use a near-uniform strategy, late
iterations use the near-converged strategy). The result is a noisy mix.
Post-hoc uses the *converged* strategy at every node, giving exact EVs.

**Three-step implementation:**

1. **C side:** add `bp_export_regrets()` companion to `bp_export_strategies()`
   in `src/mccfr_blueprint.c`. Output the same `(bucket, action)` indexing as
   strategies, but emitting `regrets[a]` instead of `strategy[a]`. Optionally
   also emit a precomputed per-action EV table from a single bottom-up tree
   walk during export. ~30-80 LOC depending on whether we precompute EVs in C
   or in Python. ~30 LOC.

2. **Export tool side:** extend `precompute/export_v2.py` to also call
   `bp_export_regrets()` (and optionally `bp_export_action_evs()`) and pack
   the new data alongside strategies in the .bps file. Bump `schema_version`
   from 2 to 3. ~20 LOC.

3. **Realtime solver side:** in `python/leaf_values.py`, add a path that reads
   per-action EVs from the v2 .bps blueprint and plugs into the existing
   `bias_strategy()` and `compute_*_leaf_values()` machinery. ~50 LOC.

**Effort:** ~1 day.

**No restart needed.** The running v2 solver just writes raw regret checkpoints.
The export tool runs after v2 finishes (or right now from any existing
checkpoint). Modifying the export tool and re-running it against the final
v2 8B checkpoint produces the corrected .bps. The running solver process
doesn't need to know.

**When to apply:** after v2 reaches 8B target naturally. Don't terminate v2.

## Priority 3 — Subgame depth analysis

### T3.1 — Compare single-street vs multi-street solving on representative spots

**Status:** **never been done.** I grepped `BLUEPRINT_CHRONICLE.md` for
"subgame", "depth", "single-street" — zero hits. The single-street choice was
inherited from when the HUD was the primary use case and never revisited.

The Pluribus paper (Brown & Sandholm 2019, Supp. p. 6) uses **variable subgame
depth**:
- Round 1: rest-of-round, leaves at start of round 2
- Round 2 with ≥3 players: rest-of-round OR 2 raises ahead, whichever earlier
- Otherwise: **rest-of-game (no depth limit)**

We always solve exactly one street.

**What proper analysis would look like:**

1. Build or borrow a multi-street CPU CFR solver. Could use the existing
   blueprint solver in postflop-only mode with a fixed flop, or extend
   `street_solver_gpu.py` to handle chance nodes (board card deals) inside the
   subgame.

2. Pick 30-50 representative spots covering: HU and 3-way; flop, turn, river;
   pot sizes from small to large; equity distributions from polar to merged.

3. Solve each spot three ways:
   - Single-street + 4 continuation strategies (current approach)
   - Two-street (current + next street)
   - Full-game (rest-of-hand)

4. Measure pairwise strategy distance (L1 norm over the per-hand action
   distributions at the root) between single-street and full-game. Same for
   two-street vs full-game.

5. Decide whether the deviation is acceptable for the trainer use case. If the
   single-street strategies are within ~5% of full-game everywhere, we can
   keep the single-street design. If they deviate by 15%+ in common spots,
   we should add multi-street solving.

**Effort:** ~3-5 days for the analysis. Implementation of multi-street solving
(if needed) is another 1-2 weeks.

**This is a research task.** Worth doing before investing in either deeper
subgames or longer iteration counts.

## Priority 4 — Preflop re-solve on large deviations

### T4.1 — Re-solve from root when opponent bets off-tree by a large margin

**What Pluribus does** (Supp. p. 5):
> "If the opponent's action is more than $100 off any blueprint raise size and
> there are no more than 4 players remaining, then Pluribus performs real-time
> search starting from the root of the current betting round."

**What we do:** `python/off_tree.py` does pseudoharmonic interpolation
(Ganzfried & Sandholm 2013) to map the opponent's actual size to the nearest
two tree sizes. This is approximate but cheap. We never re-solve from the
preflop root.

**Why it matters:** preflop is the most game-defining street. Pseudoharmonic
interpolation is fine for small deviations (opponent opens to 2.5x when our
tree has 2x and 3x — easy to interpolate). It breaks down for large deviations
(opponent shoves preflop, opponent opens to 5x when our tree has 1x/1.5x). Real
exploits live in those large-deviation cases.

**Change:**

1. Add an off-tree-detection check in `hud_solver.py.on_villain_action`: if
   the bet size is more than X bb off any tree size AND we're in round 1, mark
   the situation as "needs preflop re-solve".

2. Add a `_resolve_preflop_root()` method to `HUDSolver` that builds a
   preflop subgame including the opponent's actual bet size as a new branch,
   loads the blueprint as the leaf-value source for postflop nodes, and runs
   CFR on the preflop subgame. This needs a preflop CFR solver — we don't have
   one yet (the GPU `street_solver_gpu.py` is a single-postflop-street solver).

**Effort:** ~1 week. The preflop subgame solver is the hard part. The
detection logic is trivial.

**Defer until** T1, T2, T3 are done.

## Priority 5 — Multi-street subgame solving (if T3.1 says it's needed)

### T5.1 — Extend the GPU solver to handle chance nodes inside subgames

**Why it matters:** see T3.1. If the analysis shows single-street is too
inaccurate, this is the structural fix.

**The constraint that's NOT GPU memory:** the GPU has plenty of memory and
parallelism. The constraint is **algorithmic**: the current `street_solve.cu`
kernel deliberately avoids chance nodes (board card deals) because they
multiply the tree's branching factor by 49 (turn) or 48 (river). A
multi-street kernel needs either:
- **Explicit per-card branches** (52x branching, blows up tree size)
- **Reach-probability accumulation across cards** (Monte Carlo style — sample
  some cards each iteration, accumulate over many iterations)

Pluribus uses both depending on subgame size.

**Effort:** 1-2 weeks. Not worth doing until T3.1 confirms the benefit.

## Priority 6 — Cosmetic / cleanup

### T6.1 — Remove the heads-up-only assumption from comments

After T1.2 deletes `multiway_adjust.py`, scrub remaining "the solver is
heads-up only" claims from `hud_solver.py` and other docs. The solver has been
N-player capable since `street_solve.cu` was written; the comments lag.

### T6.2 — Replace pseudoharmonic with explicit re-solve for round 1 large deviations

After T4.1 ships, `python/off_tree.py` becomes pseudoharmonic-only-for-small-
deviations. The threshold (e.g., $100 off any tree size, per Pluribus) is the
boundary. Document this in `off_tree.py`.

---

## Decision log

| Date | Item | Decision |
|---|---|---|
| 2026-04-07 | Realtime use case | Pivoted from HUD to trainer. Latency budget relaxed from <100ms to seconds-to-minutes. |
| 2026-04-07 | T1.1 iteration count | Approved in principle; bump from 200 to 2000-5000 once trainer-mode is the only consumer |
| 2026-04-07 | T1.2 multiway heuristics | Approved removal. The heuristics double-correct on top of N-player CFR and are based on arbitrary tuning constants, not principled analysis. |
| 2026-04-07 | T2.1 export per-action EVs | Approved. Hot-fix path (no solver restart) is identified: extend export tool only, modify C export functions, re-export from the final v2 8B checkpoint. |
| 2026-04-07 | T3.1 subgame depth analysis | Approved as research task. Never been done. Should precede T5.1. |
| 2026-04-07 | T4.1 preflop re-solve | Approved as priority 4. Defer until T1-T3 are done. |
| 2026-04-07 | T5.1 multi-street kernel | Pending T3.1 outcome. |
