# Postflop Pluribus Gap Analysis

**Status**: 2026-04-09 initial audit. Updated after discovery of the two-blueprint-pipeline split.

**Who should read this**: anyone about to work on postflop inference, real-time search, or wiring the API to the solver. Read it BEFORE writing code — several design assumptions held earlier in the session turned out to be wrong.

---

## Executive summary

**The headline finding is not "Pluribus vs us" — it's that we have two completely separate blueprint pipelines and the realtime solver reads the one that Phase 1.3 does not touch.**

### The two pipelines

| Pipeline | File format | Players | Scope | Producer | Consumer |
|---|---|---|---|---|---|
| **1. Unified MCCFR** | `BPS3` / `BPR3` (`unified_blueprint.bps`) | 6 (full table) | Whole 4-street tree, bucket-keyed | `precompute/blueprint_worker_unified.py` calling `src/mccfr_blueprint.c` (CPU, OpenMP Hogwild) | `precompute/export_v2.py`, `python/blueprint_v2.py`, frontend's `preflop-nodes.json` extraction |
| **2. Per-scenario GPU flop solver** | `BPST0002` (per-texture files in `flop_blueprints/{scenario}/`) | 2 (hero/villain) | Per-scenario, per-texture, per-hand strategies + per-action EVs baked in | `precompute/run_all.py` calling `flop_solve.dll` (CUDA) | `python/blueprint_store.py`, `python/hud_solver.py`, `python/leaf_values.py` |

**What `hud_solver.py` actually reads for postflop inference**: pipeline 2 (`BlueprintStore`). Per-scenario, per-texture, 2-player. `num_players = 2` hardcoded (hud_solver.py:121).

**What Phase 1.3 added per-action EVs to**: pipeline 1 (`BPS3` trailing `BPR3` section).

**Conclusion**: Phase 1.3's per-action EV export in `precompute/export_v2.py` is architecturally orthogonal to `hud_solver.py`. The hud_solver never reads the file Phase 1.3 writes to. `leaf_values.py:55` has a comment confirming this:

> ".bps export does not currently store [per-action EVs] (the binary BlueprintStore format does)"

This was written before Phase 1.3 landed, but it's still true: Phase 1.3 added EVs to the `.bps` file, but `hud_solver.py` doesn't use that file for postflop — it uses `BlueprintStore`.

### What this means

- **Phase 1.3 work is still valuable** for any consumer that wants per-action EVs from the unified 6-player blueprint. That currently means: nothing in production. The `nexusgto` frontend's `preflop-nodes.json` only uses σ̄ (quantized strategies), not EVs. The realtime solver uses a different pipeline.
- **If we want Pluribus-style depth-limited subgame solving with biased leaf values in `hud_solver.py`**, we need per-action EVs **in the BPST format**, not the BPS3 format. The BPST format already has them (see `blueprint_store.py:20-28`, "Per-action EV for each player: float32[num_flop_actions][num_hands]"). The question is whether they're being **correctly generated** by `run_all.py` / `flop_solve.dll` and whether `leaf_values.py` is **correctly consuming** them.
- **The 2-player vs 6-player split is a major architectural decision** that's not well-documented anywhere. The unified blueprint is 6-player because training is 6-player. But the realtime solver is 2-player because the GPU flop solver is 2-player. This means **postflop inference collapses to heads-up automatically** regardless of how many players are at the table. That might be intentional (postflop often is heads-up), but it's worth flagging as a Pluribus divergence: Pluribus real-time search is N-player throughout.
- **`run_all.py` is a precompute step** that must be run BEFORE any postflop inference works. It generates the per-scenario BPST files. Has this been run recently? Do we have current BPST files anywhere? Need to check.

### Biggest blocking gap

**#11 (API wiring): `GPUPostflopProvider.solve()` returns mock data.** One line, line 124 of `nexusgto-api/app/services/postflop_provider.py`:

```python
logger.warning("GPU solver not yet wired, using mock fallback")
return MockPostflopProvider().solve(query)
```

This is the single chokepoint. Everything else in the postflop pipeline could be perfect and the API would still return canned data. Wiring this to `HUDSolver.get_strategy()` is ~200 lines of adapter code (schema translation: `PostflopQuery` → `HUDSolver.new_hand()` → action replay → response building).

---

## Pluribus components — what Pluribus does vs. what we have

### 1. Blueprint abstraction for inference

**Pluribus** (`pluribus_technical_details.md` §1, §4):
> "Pluribus uses its blueprint only for the first round of play. Real-time search is used for all subsequent rounds."

Blueprint is preflop-only at inference. Rounds 2-4 (flop, turn, river) all go through real-time subgame search.

**Us**:
- `hud_solver.py:368-394` — if current street is flop and `blueprint_v2` is loaded, it pulls **flop strategies** from the unified BPS3 blueprint. Not just preflop.
- `hud_solver.py:96-112` — also loads per-scenario `BlueprintStore` for postflop flop/turn roots.
- `leaf_values.py` — uses `BlueprintStore.get_turn_strategy()` and `BlueprintStore.get_turn_action_evs()` as continuation strategies at flop-subgame leaves.

**Status**: **PARTIAL + DIVERGENT.** We load the blueprint at multiple postflop streets, which is richer than Pluribus's preflop-only approach. This isn't necessarily wrong — Pluribus's blueprint is preflop-only because their postflop search is depth-limited multi-street and uses the blueprint AT the leaves. Our approach reads the blueprint AT decision nodes too. The end user experience may be similar; the mechanism is different.

**Gap**: documentation mostly. Decide and document whether we want Pluribus-faithful (blueprint only at leaves) or our current approach (blueprint read directly at turn/river roots too).

---

### 2. Depth-limited subgame solving

**Pluribus** (§4 p.6):
> "The depth of the subgame varies by betting round. For round 1 [preflop], the subgame extends to the end of round 1. For round 2 [flop] with more than 2 players, the subgame extends until either the end of round 2 or 2 raises have occurred, whichever is later. For round 2 with 2 players, or for rounds 3-4, the subgame extends to the end of the game."

Variable depth based on player count and street. For heads-up flop play, **Pluribus solves the whole rest of the game** (flop + turn + river + showdown). For 3+ players on flop, depth is "rest-of-round or 2 raises."

**Us**:
- `hud_solver.py` calls `street_solver_gpu.py` which is **single-street only**. No multi-street traversal inside the subgame.
- At the end of the current street, `leaf_values.py` computes **leaf values** from the blueprint instead of solving deeper.
- This is the "depth = 1 street" version of depth-limited subgame solving.

**Status**: **PARTIAL.** We have single-street depth-limited solving with blueprint leaf values, which IS a form of depth-limited subgame solving. But Pluribus's heads-up flop solve goes turn+river+showdown, which is 3x deeper. For bigger stacks / more complex spots this matters a lot.

**Gap**: multi-street GPU CFR. Tracked as T5.1 in `docs/REALTIME_TODO.md`. ~1-2 weeks estimated, and should be preceded by T3.1 (measure whether multi-street actually helps enough to justify it).

---

### 3. The 4 biased continuation strategies at leaves

**Pluribus** (§4 p.6):
> "At the leaf nodes of the depth-limited subgame, each opponent is modeled as playing one of four strategies. The four strategies are: (1) the approximate blueprint strategy itself; (2) a biased version that folds with 5x the normal frequency; (3) a biased version that calls with 5x the normal frequency; (4) a biased version that raises with 5x the normal frequency. The traverser's strategy is computed as a best response to the mixture."

Four continuation strategies, each a hand-level adjustment of σ̄ where one action category's probability is multiplied by 5 and the others are renormalized. **16 leaf values** per leaf (4 strategies × 4 "which bias is active", since the opponent's bias choice is also hidden from the traverser).

**Us**: Exists in code.
- `leaf_values.py:152-181` — `bias_strategy(strat, bias_type, categories, factor)` implements the 5x bias logic
- `leaf_values.py:46-88` (approximately) — computes the 16-per-leaf values
- **Status of the per-action EV input**: this is where Phase 1.3 and the blueprint format split matters
  - `leaf_values.py` calls `blueprint_store.get_turn_action_evs(board, tc, p)` (line ~331). This reads from the **BPST format**, which already has per-action EVs baked in per `blueprint_store.py:20-28`.
  - Phase 1.3 added per-action EVs to the **BPS3 format**, which `leaf_values.py` does not read for postflop.
  - **So the 4-biases + 16-leaf-values path works TODAY via the BPST format**, IF the BPST files exist AND were generated correctly with per-action EVs.

**Status**: **HAVE (conditionally).** The logic is there. The inputs are there (in BPST format). The binding is there. **But the BPST files must be freshly generated by `run_all.py`** for any current scenario, which is a separate precompute step from training the unified blueprint.

**Gap**:
- Verify BPST files exist locally or in S3 for the scenarios the product needs
- Verify `run_all.py` + `flop_solve.dll` still builds and runs on current codebase
- Verify the per-action EVs in the BPST format match the biased-leaf-values consumer's expectations (schema, ordering)

---

### 4. Real-time search trigger conditions

**Pluribus** (§4 p.5):
> "Real-time search is used on rounds 2, 3, and 4 [flop, turn, river]."

Always search postflop. On round 1 (preflop), blueprint is used directly UNLESS a large bet triggers re-solve (see item 5).

**Us**:
- `hud_solver.py:863-877` — `get_strategy()` dispatches to GPU subgame solve on postflop, blueprint on preflop
- Postflop search is gated on GPU availability via `GPU_ENABLED` flag in the API config
- If GPU not available, falls back to... what? Let me check. Actually I didn't trace this end to end — could be blueprint-only, could be error, could be the mock.

**Status**: **PARTIAL.** We have the search-postflop-always logic, but gated on GPU availability, and the fallback path isn't clearly traced.

**Gap**: trace the GPU-unavailable fallback. Document it. Probably also: test that GPU_ENABLED=true actually gets the GPU subgame solver running locally or on the server.

---

### 5. Preflop re-solve on large deviations (T4.1)

**Pluribus** (§4 p.5):
> "If an opponent bets an off-tree amount of more than $100, Pluribus re-solves the strategy from the start of the betting round (if there are at most 4 players) with the opponent's bet added to the action abstraction."

Only triggers if:
1. Opponent bet is **off-tree** (not in the blueprint's action abstraction)
2. Deviation magnitude > **$100** at Pluribus's stakes (so big relative to pot/blinds)
3. **≤ 4 players** remaining (Pluribus doesn't re-solve 5+ player spots)

When triggered: re-solve from the preflop root with the off-tree action injected, then play from that point.

**Us**:
- `python/off_tree.py` implements pseudoharmonic interpolation to snap off-tree bets to the nearest abstracted size (good, matches Pluribus's smaller-deviation handling per §4 p.5)
- **No re-solve trigger.** For any deviation, we just snap via pseudoharmonic.

**Status**: **PARTIAL / MISSING.** We have the small-deviation path (pseudoharmonic snap), not the large-deviation path (re-solve from root).

**Gap**: T4.1 in REALTIME_TODO.md. ~1 week. For a training-tool product (not a real-time bot), small-deviation snapping is probably acceptable. Re-solve triggers matter more for live play.

---

### 6. Pseudoharmonic bet interpolation

**Pluribus** (§4 p.5):
> "...the bet is mapped to the nearest action in the abstraction via pseudoharmonic mapping." — with the formula: `f(x) = (B - x)(1 + A) / ((B - A)(1 + x))` or similar, mapping (A, B) bracket to [0, 1] weights.

**Us**: `python/off_tree.py:75-87` implements the exact formula. Used as fallback for all off-tree deviations.

**Status**: **HAVE_FULL.**

**Gap**: none on the mechanism. The gap is that we use pseudoharmonic where Pluribus uses re-solve (item 5 above).

---

### 7. Vector-based MCCFR

**Pluribus** (§4 p.6):
> Pluribus uses a "vectorized form of Monte Carlo CFR" for real-time search, where CFR runs simultaneously on all hands in the player's range at each decision point. ~2-4x speedup on small subgames.

**Us**:
- `src/cuda/gpu_mccfr.cu` — CUDA kernels for CFR iterations
- The GPU path IS vectorized in the sense that each CUDA thread handles one hand — a form of data-parallel vectorization across the hand range
- Haven't deeply verified the parity to Pluribus's specific technique

**Status**: **HAVE** (architectural intent matches). Not verified to be faithful to Pluribus's specific vectorization pattern.

**Gap**: audit the GPU kernels against the Pluribus paper's description, probably in a future pass.

---

### 8. Bet abstraction at subgame root

**Pluribus** (§4 Algorithm 2 line 6):
> When the subgame is constructed, the action abstraction starts with the blueprint's abstraction. If the opponent has taken an off-tree action, `AddAction(opponent_action)` expands the tree, then search proceeds.

Dynamic tree expansion: the abstraction grows to include off-tree opponent actions.

**Us**:
- `DEFAULT_BET_SIZES = [0.33, 0.75, 1.5]` plus auto-appended all-in in the realtime solver
- Off-tree opponent actions use pseudoharmonic narrowing (item 5/6), not dynamic tree expansion
- No `AddAction` mechanism

**Status**: **PARTIAL / DIVERGENT.** Pluribus adds the off-tree action to the subgame tree and re-solves. We interpolate. Different approach, similar practical effect on small deviations.

**Gap**: dynamic tree expansion would require restructuring the GPU solver's tree building to accept mid-solve additions. ~1-2 weeks. Low priority for training tool.

---

### 9. Turn/river bucket abstraction at inference time

**Pluribus** (§2):
> Blueprint uses 200 buckets per street. Real-time search on turn/river uses **500 buckets** (finer abstraction than blueprint, because search has more compute per info set).

**Us**:
- Blueprint: 200 buckets (matches)
- Realtime: `leaf_values.py` operates per-hand (1326 lossless) for the flop leaves it computes. Much finer than Pluribus's 500 bucket abstraction.
- For turn/river decision nodes in the subgame, it's not clear whether we bucket at all inside the GPU kernels. Needs source read.

**Status**: **HAVE+** (we're more precise than Pluribus, or at least not less).

**Gap**: verify the GPU subgame solver doesn't bucket turn/river decisions inside the subgame tree (should be lossless per-hand).

---

### 10. Multi-street solving

**Pluribus** (§4 p.6):
> For 2-player flop play (or rounds 3-4 at any player count), Pluribus solves **to the end of the game** — multi-street, through the river, including chance nodes for turn/river cards.

**Us**:
- `hud_solver.py`'s GPU subgame solver is **single-street only**
- Street transitions (flop→turn, turn→river) hit leaf values, not chance nodes
- This is the "depth = 1 street" approach

**Status**: **MISSING for Pluribus parity.** Tracked as T5.1.

**Gap**: multi-street GPU CFR. Large effort (~1-2 weeks). Should be preceded by T3.1 (exploitability measurement of single-street vs multi-street on representative spots) to confirm it's worth the cost.

---

### 11. API wiring to real postflop solver

**This is the single biggest product blocker, not a Pluribus gap.**

**Current state** (`nexusgto-api/app/services/postflop_provider.py:119-124`):
```python
def solve(self, query: PostflopQuery) -> RangeStrategy:
    self._ensure_initialized()
    # TODO: Wire HUDSolver.new_hand() + get_strategy() when GPU is available
    # For now, fall back to mock
    logger.warning("GPU solver not yet wired, using mock fallback")
    return MockPostflopProvider().solve(query)
```

**Status**: **MISSING.**

**What needs to happen**:
1. Map `PostflopQuery` (position, board, action_history, stack_depth) to `HUDSolver.new_hand()` arguments. HUDSolver's init takes (hero_pos, villain_pos, scenario_type, num_players=2, pot_bb, stack_bb).
2. Replay `action_history` via `HUDSolver.on_villain_action()` / `on_hero_action()` to bring the solver to the current state.
3. Call `HUDSolver.get_strategy()` to get the hero's σ̄ at the current decision point.
4. Convert the output (per-hand strategy) to `RangeStrategy` (list of `HandStrategy` for the 169 hand grid, with action frequencies).
5. Error paths: what happens if the scenario isn't loaded, the board is unfamiliar, the GPU is unavailable.

**Effort**: ~2-3 days for someone who knows both the `HUDSolver` and API schemas. Less if they're already familiar.

---

### 12. Frontend expectations for postflop

**Frontend** (`nexusgto/src/app/(app)/study/postflop/` — not yet verified this path exists):

The frontend currently uses a **static JSON** for preflop (`preflop-nodes.json`). Postflop **probably** uses the API, but I did not verify this in the audit. If there's a static postflop artifact bundled in the frontend, that's a third pipeline to understand.

**Gap**: verify postflop data path in the frontend. If it's API-driven, great. If it's static, we have yet another JSON to regenerate + yet another hash mixer / format to audit.

---

### 13. Per-action EV wiring (Phase 1.3 status update)

**What Phase 1.3 did**:
- Added `bp_compute_action_evs()` and `bp_export_action_evs()` in `src/mccfr_blueprint.c`
- Added BPR3 trailing section to the `.bps` output format
- Updated `precompute/export_v2.py` to call the new functions
- Updated `python/blueprint_v2.py` to parse the BPR3 section

**What Phase 1.3 did NOT do**:
- Did not touch `python/blueprint_store.py` (BPST format)
- Did not touch `python/hud_solver.py`
- Did not touch `python/leaf_values.py`

**Why this matters**: `leaf_values.py:331` calls `blueprint_store.get_turn_action_evs(...)` (via the `BlueprintStore` class), not `blueprint_v2.get_turn_action_evs(...)`. The BPST format **already** has per-action EVs baked in (see `blueprint_store.py` header comment lines 20-28). Phase 1.3 added per-action EVs to a **different file** from the one `leaf_values.py` actually reads.

**Is this a problem?**

It depends on what's running at inference time:
- **If `run_all.py` has been run recently and BPST files exist with good per-action EVs**, then leaf values work TODAY without any Phase 1.3 integration. The 4-biases work. Pluribus-style leaf values work.
- **If BPST files are stale, missing, or lack per-action EVs**, nothing works regardless of Phase 1.3's state in the unified pipeline.
- **If we want to use the unified 6-player blueprint as the continuation strategy source** (instead of the per-scenario 2-player GPU solve), Phase 1.3 is the right primitive, but `leaf_values.py` / `hud_solver.py` would need refactoring to read from `BlueprintV2` instead of `BlueprintStore`.

**Actions needed**:
1. **Check if BPST files exist.** Look in `C:/Users/Victor/Documents/Projects/poker-solver/flop_blueprints/` (or wherever `_bp_store_dir` defaults to).
2. **If they exist, check their age.** Were they generated with current code or stale?
3. **Run `precompute/run_all.py --scenario <something> --iterations 100` as a smoke test** to verify the GPU precompute pipeline still works.
4. **Decide the architectural question**: does the product want per-scenario 2-player BPST (current) or unified 6-player BPS3 with Phase 1.3 EVs (requires significant refactor of leaf_values)?

---

## Priority recommendations

Ordered by "what blocks postflop working" most.

### CRITICAL — blocking user-visible postflop output

1. **Wire `GPUPostflopProvider.solve()` to `HUDSolver`** (#11)
   - **Effort**: 2-3 days
   - **Files**: `nexusgto-api/app/services/postflop_provider.py`, possibly `nexusgto-api/app/models/postflop.py`
   - **Unblocks**: any real postflop output from the API. Until this is done, everything else is theoretical.

2. **Verify BPST file pipeline is alive** (#13 preamble)
   - **Effort**: 1 day (run a precompute, check outputs, debug)
   - **Files**: `precompute/run_all.py`, `python/blueprint_store.py`, any existing `flop_blueprints/` directory
   - **Unblocks**: confirming that the leaf-values + 4-biases path has valid inputs. If BPST is broken, `hud_solver` has nothing to serve.

### HIGH — blocking Pluribus-faithful postflop

3. **Decide architectural question: BPST (2-player per-scenario) vs BPS3+Phase1.3 (6-player unified)**
   - **Effort**: design work, ~1 day of reading + 1 day of decision
   - **Output**: written decision doc, possibly in `docs/`
   - **Why**: this determines whether Phase 1.3's EV export is load-bearing for postflop or just scaffolding. Major architectural decision that should not be made by accident.

4. **Multi-street GPU subgame solver** (T5.1, #10)
   - **Effort**: 1-2 weeks
   - **Files**: `src/cuda/gpu_mccfr.cu`, `src/cuda/gpu_solver.cu`, `python/street_solver_gpu.py`, `python/hud_solver.py`
   - **Gate**: T3.1 (exploitability analysis) should precede this to confirm it's worth the cost
   - **Impact**: brings us to Pluribus parity on depth-limited search for heads-up postflop

### MEDIUM — Pluribus parity nice-to-haves

5. **Preflop re-solve on large deviations** (T4.1, #5)
   - **Effort**: ~1 week
   - **Files**: realtime solver path, `off_tree.py`, preflop CFR driver
   - **Impact**: matters more for live play than training use

6. **Dynamic tree expansion for off-tree actions** (#8)
   - **Effort**: 1-2 weeks
   - **Files**: GPU subgame tree builder
   - **Impact**: smoother handling of non-standard opponent bets; low priority for training tool

### LOW — defer

7. **Final-iteration strategy export** (referenced in STATUS.md §3, not in this audit)
   - Pluribus uses final-iteration strategy for postflop (not the weighted average) to avoid σ̄ lag. Our unified blueprint exports σ̄. This is a small change if needed.

---

## Open questions for the human

1. **Does `flop_blueprints/` exist locally?** If yes, with data from when? If no, we need to run `run_all.py` from scratch — and that depends on `flop_solve.dll` which is a CUDA build I haven't verified.

2. **Is the product's postflop target heads-up only (2-player) or multi-way?** The current realtime solver hardcodes `num_players = 2`. Pluribus supports up to 6 in real-time search (with depth reductions for 3+ players).

3. **Do we want Pluribus-faithful depth-limited search with multi-street solving, OR is "single-street + blueprint leaf values" good enough for a training tool?** Multi-street is 1-2 weeks of work and the difference might be invisible to users practicing on typical spots.

4. **What's the intended deployment architecture?** The API needs to load BlueprintStore files at startup. Those are gigabytes. Cold start time and memory cost matter for deployment decisions.

5. **Does the BPST format's per-action EVs work correctly with the current `leaf_values.py` / 4-biases logic?** The code exists but I didn't verify it runs end-to-end without a live BPST file to test against.

---

## Files referenced

| File | Purpose | Relevance |
|---|---|---|
| `pluribus_technical_details.md` | Ground truth for Pluribus algorithm | §1, §2, §4 cited throughout |
| `src/mccfr_blueprint.c` | 6-player MCCFR blueprint training (CPU) | Produces BPS3 via `export_v2.py` |
| `src/cuda/gpu_mccfr.cu` | GPU CFR for subgames | Used by `street_solver_gpu.py` |
| `python/blueprint_v2.py` | BPS3 reader | Unified blueprint consumer |
| `python/blueprint_store.py` | BPST reader | **Per-scenario flop blueprint consumer — what `hud_solver` uses** |
| `python/hud_solver.py` | Realtime solver entry point | Consumer of both blueprints; 2-player |
| `python/leaf_values.py` | Depth-limited leaf eval + 4 biases | Reads BPST, not BPS3 |
| `python/off_tree.py` | Pseudoharmonic bet interpolation | Used for small deviations |
| `python/street_solver_gpu.py` | Single-street GPU subgame solver | Called by hud_solver |
| `precompute/run_all.py` | GPU flop-solve precompute pipeline | **Produces BPST files** |
| `precompute/export_v2.py` | BPS3 exporter with Phase 1.3 EVs | Produces unified blueprint |
| `docs/PHASE_1_3_DESIGN.md` | Phase 1.3 design | Design for BPS3 per-action EVs |
| `docs/REALTIME_TODO.md` | Real-time solver backlog | T3.1, T4.1, T5.1 |
| `nexusgto-api/app/services/postflop_provider.py` | API postflop provider | **Returns mock today; blocking** |
| `nexusgto-api/app/services/dependencies.py` | Provider singleton wiring | Routes to mock vs GPU provider |
| `nexusgto-api/app/routers/solve.py` | `/api/solve` endpoint | Entry point for postflop API requests |

---

*This audit was done without running the code in-process. Several claims (BPST files exist, 4-biases path works end-to-end, GPU subgame solver runs) are code-verified but not execution-verified. A real smoke test — running `run_all.py` with a trivial scenario and verifying BPST output, then running `hud_solver.get_strategy()` end-to-end — is the next step before any refactor work.*
