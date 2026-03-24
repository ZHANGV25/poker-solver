# Poker Solver: Context Document for Next Session

## Prompt for Next Session

```
Implement the Pluribus hybrid architecture for the poker solver. We have a working GPU full-tree solver (flop→turn→river in 6.6s for 80 hands). Now we need to switch to the street-by-street approach:

1. OFFLINE PRECOMPUTE: Use the GPU solver (flop_solve.cu) to generate blueprint data for all 27 scenarios × 1755 textures. Store P(action|hand) at every decision node (flop root + turn roots per card + river roots per runout).

2. STREET-BY-STREET ONLINE SOLVE: Instead of solving 546K nodes at once, solve only the current street (~30-50 nodes) on GPU in <1 second. Use precomputed next-street strategies as leaf values.

3. RANGE NARROWING BETWEEN STREETS: After each observed action, multiply villain's hand weights by P(action|hand) from the most recent solve's weighted average strategy.

4. OFF-TREE BET HANDLING: Implement pseudoharmonic mapping when opponent bets a size not in our action abstraction.

5. MULTIPLE BET SIZES: Support 2-3 bet sizes per street (33%, 75%, all-in) now that each street's tree is small.

Project is at C:/Users/Victor/Documents/Projects/poker-solver/
Read CONTEXT_NEXT_SESSION.md for full details before starting.
```

---

## Project Location & Repository

- **Path**: `C:/Users/Victor/Documents/Projects/poker-solver/`
- **GitHub**: https://github.com/ZHANGV25/poker-solver.git
- **Branch**: master
- **Latest commit**: `634484a` — "Pluribus-aligned multi-street GPU solver"
- **ACR HUD repo** (consumer of this solver): `C:/Users/Victor/Documents/Projects/ACRPoker-Hud-PC/`

## Hardware

- **CPU**: i7-13700K
- **RAM**: 64GB
- **GPU**: RTX 3060 12GB (3584 CUDA cores, 360 GB/s bandwidth)
- **OS**: Windows 10 Pro
- **Compiler**: GCC 14.2.0 (MSYS2 UCRT64), NVCC 11.8
- **NVCC requires**: MSVC BuildTools cl.exe + flags `-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`

### NVCC compile command (copy-paste ready):
```bash
MSVC_DIR="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207"
WINSDK_INC="/c/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0"
WINSDK_LIB="/c/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0"
export PATH="$MSVC_DIR/bin/Hostx64/x64:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin:$PATH"

nvcc -O2 --shared -o build/flop_solve.dll src/cuda/flop_solve.cu -I src/cuda \
  -allow-unsupported-compiler -ccbin "$MSVC_DIR/bin/Hostx64/x64" \
  -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH \
  --compiler-options "/MD /I\"$MSVC_DIR/include\" /I\"$WINSDK_INC/ucrt\" /I\"$WINSDK_INC/um\" /I\"$WINSDK_INC/shared\"" \
  -L"$MSVC_DIR/lib/x64" -L"$WINSDK_LIB/ucrt/x64" -L"$WINSDK_LIB/um/x64"
```

### GCC compile command:
```bash
gcc -O2 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function -static -I src bench/test_phase1.c src/solver_v2.c -o build/test_phase1.exe -lm
```

### Running Windows .exe from Git Bash:
```bash
# Direct execution fails (exit code 127). Use Python subprocess:
/c/Users/Victor/AppData/Local/Programs/Python/Python39/python.exe -c "
import subprocess
r = subprocess.run(['C:/path/to/exe'], capture_output=True, text=True, timeout=120)
print(r.stdout)
print('RC:', r.returncode)
"
# Or use msys2 python:
/c/msys64/ucrt64/bin/python3.exe -c "..."
```

---

## What Exists Now

### Core C Solver (`src/solver_v2.c`, 987 lines)
- Multi-street tree builder: flop → turn → river via CHANCE nodes
- Linear CFR (regret × t/(t+1) discount each iteration)
- Final iteration strategy for play, weighted average for narrowing
- `sv2_get_strategy_at_node()` — query strategy at any tree depth
- `sv2_get_average_strategy()` — for Bayesian belief updating
- CPU multi-street solve works but is slow for flop (90s for 4 hands)

### GPU Full-Tree Solver (`src/cuda/flop_solve.cu`, 1286 lines)
- Materializes complete flop→turn→river tree on GPU (546K nodes for 1 bet size)
- Level-batched Linear CFR: one kernel per BFS level
- CUDA graph capture for near-zero launch overhead
- Shared memory for showdown opponent data
- **Performance with 80 hands**: 58ms/iter, 100 iters = 6.6s, 300 iters = 17.4s
- Tree build on CPU: 33ms
- Hand strength precompute on GPU: ~2s for 181K showdown nodes
- Uses 3.8GB GPU memory at 80 hands

### GPU Equity Evaluator (`src/cuda/flop_accel.cu`)
- Parallel equity rollout across all turn × river runouts
- 80 hands × 2352 runouts in 97ms
- Standalone — not currently integrated into the solver

### Python Layer
- `python/solver.py` — `StreetSolver` class (ctypes wrapper for solver_v2.dll)
- `python/hud_solver.py` — High-level HUD interface with multiway adjustments
- `python/range_narrowing.py` — Bayesian range tracker
- `python/blueprint_io.py` — Read precomputed flop solutions (suit isomorphism)
- `python/multiway_adjust.py` — Heuristic adjustments for 3+ player pots
- `python/preflop_solver.py` — Analytical preflop range approximation
- `python/solver_pool.py` — Thread pool for concurrent solves

### Precompute Pipeline
- `precompute/solve_scenarios.py` — Uses Rust tbl-engine for flop solutions
- `precompute/gpu_precompute.py` — GPU batch orchestrator (skeleton)
- Existing flop solutions in ACR HUD: `solver/flop_solutions/` (4 of 27 scenarios complete)

### Other C/CUDA Files
- `src/solver.c/h` — Original v1 solver (CFR+, single-street only)
- `src/hand_eval.h` — 7-card evaluation via 21× eval5
- `src/preflop_solver.c` — 169-class CFR solver (basic)
- `src/cuda/gpu_solver.cu/cuh` — Original GPU solver (not multi-street)

---

## Benchmarks (measured on this machine)

### GPU Full-Tree Solver (flop_solve.cu)
| Hands | 50 iter | 100 iter | 150 iter | 200 iter | 300 iter | ms/iter |
|-------|---------|----------|----------|----------|----------|---------|
| 8     | 886ms   | 1,500ms  | 2,241ms  | 3,123ms  | 4,546ms  | 14.9ms  |
| 40    | 1,454ms | 2,639ms  | 3,564ms  | 4,600ms  | 6,713ms  | 22.4ms  |
| 80    | 3,746ms | 6,568ms  | 9,294ms  | 12,151ms | 17,425ms | 58.1ms  |

### Convergence (8 hands on Qs As 2d, % = check/bet75/raise)
- 100 iter: KK 3/1/96, QQ 16/2/82, AK 80/21/0, 65 100/0/0 — strong hands converged
- 150 iter: KK 1/0/99, QQ 15/4/81, AK 91/9/0, JT 84/16/0 — all within 10%
- 300 iter: KK 8/0/92, QQ 17/0/83, AK 100/0/0, JT 96/4/0 — fully converged

### CPU Multi-Street Solver (solver_v2.c)
- River 4 hands/1000 iter: 700ms
- Turn 4 hands/50 iter: 7.2s (deals 48 river cards)
- Flop 4 hands/5 iter: 92s (deals 49 turn × 46 river)

### Theoretical GPU limits
- Bandwidth floor: ~7.5ms/iter (360 GB/s, ~2.7GB data)
- Current: 58ms/iter (80 hands) — bottleneck is O(N²) showdown kernel
- 181K showdown nodes × 80² = 1.16B comparisons per traverser

---

## What Pluribus Does (from the paper, verified)

### Architecture
1. **Offline blueprint** (12 days, 64 cores): External-sampling MCCFR with Linear CFR. 200-bucket lossy card abstraction. Stored as ~128GB.
2. **Real-time search** (1-33s per decision, 2 CPUs): Subgame from START of current betting round to END of round. Lossless abstraction for current round, 500-bucket for later rounds.

### Key Algorithm Details

**Linear CFR**: Regrets discounted by t/(t+1) each iteration. Strategy sum weighted by t. This is DCFR with α=1, β=1, γ=1. Pluribus uses this for both blueprint and real-time search.

**Strategy for play**: Final iteration (not average). Pluribus plays the strategy from the last CFR iteration.

**Strategy for belief updating**: Weighted average strategy. When narrowing ranges via Bayes' rule, uses the iteration-weighted average P(action|hand), not the final iteration.

**Leaf nodes (depth-limited solving)**: At the depth limit (end of betting round), BOTH players simultaneously choose among 4 continuation strategies. Modeled as sequential decision nodes where neither observes the other's choice. The 4 strategies are:
1. Unmodified blueprint for remainder of game
2. Fold-biased: multiply P(fold) by 5× at all future decision points, renormalize
3. Call-biased: multiply P(call) by 5× at all future decision points, renormalize
4. Raise-biased: multiply P(raise) by 5× at all future decision points, renormalize

CFR handles both players' choices naturally — their regrets over the 4 options converge.

**Belief updating**: P(hand | action) ∝ P(action | hand) × P(hand). The σ used for narrowing = output of most recent search (weighted average), or blueprint if no search has been done yet. Updated at start of each new betting round.

**Off-tree bets**: Pseudoharmonic mapping to nearest blueprint bet size. Formula: map bet b to the two nearest tree sizes b_lo and b_hi. Weight = (b - b_lo) / (b_hi - b_lo), applied to the probabilities. (Exact formula in Johanson et al. 2013.)

**Action abstraction**: 1-14 bet sizes per decision point depending on situation. Pluribus limits itself to a few sizes but the opponents can bet any amount.

**Frozen strategy during re-solve**: When Pluribus re-solves after an opponent's off-tree action, it freezes its own strategy ONLY for the specific hand it actually holds at nodes already passed through. Other hands' strategies at those information sets are free to vary. Opponent strategies are never frozen.

### What We Should Copy
1. **Street-by-street solving** with leaf values from next-street blueprint
2. **Range narrowing** between streets using weighted average strategy
3. **Off-tree bet mapping** via pseudoharmonic interpolation
4. **Multiple bet sizes** (2-3 per street, since per-street trees are small)
5. **Offline precompute** of blueprint strategies for all textures

### What We Should NOT Copy
- **6-player self-play** — not needed, solve 2-player and use heuristics for multiway
- **Lossy card abstraction** — we solve exact hands (better for the hands we cover)
- **128GB blueprint** — we store per-texture solutions instead

---

## The Hybrid Architecture to Implement

### Phase A: Offline Precompute (GPU, ~40 min for all scenarios)

For each of 27 scenarios × 1755 flop textures:
1. Run `fs_solve_gpu()` with 200 iterations, 80 hands per player
2. Extract and store:
   - **Flop root strategy**: P(action|hand) for each hand at the flop root node
   - **Turn root strategies**: For each of 49 turn cards, P(action|hand) at the turn root
   - **River root strategies**: For each of 49×46 runouts, P(action|hand) at the river root
   - These are the **weighted average** strategies (for narrowing)

Storage estimate: 27 × 1755 × (flop + 49 turn + 2254 river) × 80 hands × 4 actions × 4 bytes
= 47K × 2304 × 80 × 4 × 4 ≈ 140GB raw. Need compression or sampling.

**Optimization**: Only store flop + turn strategies. River is solved at runtime (fast).
= 47K × 50 × 80 × 4 × 4 ≈ 3GB raw → ~300MB LZMA compressed. Feasible.

### Phase B: Street-by-Street Online Solve

```
PREFLOP:
  Look up ranges from preflop solver output
  → hero_range, villain_range (with fractional weights)

FLOP (hero's turn to act):
  1. Look up blueprint P(action|hand) for this texture
  2. If villain already acted: narrow villain range using P(action|hand)
  3. Build flop-only betting tree (~30-50 nodes, 2-3 bet sizes)
  4. Leaf values = precomputed turn root EVs (average over turn cards)
     - For each leaf, compute: EV[hand] = avg over turn cards of
       sum over opponent hands of (turn_blueprint_EV[hand] × opp_weight[o])
  5. Run GPU CFR on flop tree: 200 iters in <100ms
  6. Return hero's strategy

TURN (hero's turn to act):
  1. Narrow villain range using flop actions (blueprint P(action|hand))
  2. Build turn-only betting tree (~30-50 nodes)
  3. Leaf values = precomputed river root EVs (average over river cards)
  4. Run GPU CFR: 200 iters in <100ms
  5. Return hero's strategy

RIVER (hero's turn to act):
  1. Narrow villain range using flop+turn actions
  2. Build river-only betting tree (~30-50 nodes)
  3. Terminal values = showdown evaluation
  4. Run GPU CFR: 200 iters in <100ms
  5. Return hero's strategy
```

Total time per decision: **<500ms** (vs 6.6s now)

### Phase C: Off-Tree Bet Handling

When villain bets a size not in our tree (e.g., 55% pot when tree has 33% and 75%):
1. Compute pseudoharmonic weight: w = (b - b_lo) / (b_hi - b_lo)
2. Interpolate villain's range narrowing between the two tree actions
3. Continue solving from that point

### Phase D: Integration with ACR HUD

Wire the solver into `cdp_ui.py`:
1. `hud_solver.py` calls the street-by-street solver
2. Blueprint data loaded from disk (lazy, cached per scenario)
3. Range narrowing state maintained per hand via `RangeNarrower`
4. Results displayed in the tkinter overlay

---

## Specific Task List

### Task 1: Blueprint Storage Format
- Design on-disk format for precomputed strategies
- Need: scenario_id/texture_key → {flop_strategies, turn_strategies[49]}
- Each strategy = per-hand action probabilities (weighted average)
- Consider: LZMA compression, memory-mapped access, lazy loading
- Output: `python/blueprint_store.py` with read/write functions

### Task 2: Offline Precompute Pipeline
- Modify `flop_solve.cu` to extract strategies at flop AND turn root nodes
- Write `precompute/run_all.py` that iterates all 27 scenarios × 1755 textures
- Each solve: build tree → GPU solve → extract flop+turn strategies → save
- Batch multiple textures if GPU memory allows
- Target: complete in <1 hour on the RTX 3060

### Task 3: Single-Street GPU Solver
- New `src/cuda/street_solve.cu`: solve a single street's betting tree
- Input: hands, weights, bet_sizes, leaf_values (from blueprint)
- Much smaller tree (~50 nodes vs 546K) → much faster
- 200 iterations should complete in <100ms
- Support 2-3 bet sizes (33%, 75%, all-in)

### Task 4: Leaf Value Computation from Blueprint
- Given precomputed turn root strategies, compute flop leaf values
- For each flop leaf: average over 49 turn cards of expected value
- EV computed using the 4 continuation strategy approach (both players choose)
- Or simpler: just use the raw weighted-average EV from the turn root solve

### Task 5: Range Narrowing Pipeline
- Wire `RangeNarrower` into the per-decision flow
- After each observed action, update weights using P(action|hand)
- P(action|hand) comes from: blueprint (if no re-solve yet) or weighted average of most recent solve
- Ensure narrowing happens at street boundaries, not mid-solve

### Task 6: Off-Tree Bet Mapping
- Implement pseudoharmonic action mapping in `python/solver.py`
- When villain bets off-tree: find nearest two tree sizes, interpolate
- Apply interpolated narrowing to villain's range
- Small self-contained function, <50 lines

### Task 7: Multiple Bet Sizes
- Update street tree builder to support [0.33, 0.75, all-in] bet sizes
- With single-street solving, tree stays small (~50-100 nodes)
- Verify GPU memory and performance with expanded tree

### Task 8: End-to-End Integration Test
- Simulate a full hand: preflop ranges → flop action → narrow → turn action → narrow → river solve
- Verify strategies are reasonable at each step
- Measure total time (should be <1s per decision)

---

## Key Files to Read First

1. `src/cuda/flop_solve.cu` — The GPU solver. Understand `fs_build_tree` (tree construction), batched kernels (Part 3B), and `fs_solve_gpu` (the CFR loop).
2. `src/solver_v2.c` — The CPU solver with multi-street CHANCE nodes. Shows how cfr_traverse and cfr_chance work.
3. `python/hud_solver.py` — The high-level interface that the HUD calls.
4. `python/blueprint_io.py` — How blueprint data is read from disk.
5. `python/range_narrowing.py` — Bayesian range tracking.
6. `precompute/solve_scenarios.py` — Existing precompute pipeline (uses Rust solver).

---

## Paper References (key URLs)

- **Pluribus paper**: https://noambrown.github.io/papers/19-Science-Superhuman.pdf
- **Pluribus supplementary (Algorithm 1 & 2 pseudocode)**: https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf
- **Depth-limited solving**: https://proceedings.neurips.cc/paper_files/paper/2018/file/34306d99c63613fad5b2a140398c0420-Paper.pdf
- **DCFR paper**: https://arxiv.org/pdf/1809.04040
- **Brown's thesis (comprehensive, 200 pages)**: http://reports-archive.adm.cs.cmu.edu/anon/2020/CMU-CS-20-132.pdf
- **CMU lecture on Pluribus**: https://www.cs.cmu.edu/~sandholm/cs15-888F24/Lecture_14_Pluribus_and_depth-limited_subgame_solving.pdf
- **GPUGT paper**: https://arxiv.org/html/2408.14778v1
- **Noam Brown's reference solver**: https://github.com/noambrown/poker_solver

---

## Known Issues / Gotchas

1. **Windows exe execution**: Can't run .exe from Git Bash directly (exit code 127). Must use Python subprocess or cmd.exe.
2. **NVCC + MSVC version**: CUDA 11.8 doesn't officially support MSVC 14.44. Must use `-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`.
3. **Stack overflow in C solver**: solver_v2.c uses heap-allocated scratch buffers (malloc/free per recursion) because MAX_HANDS_V2=1326 VLAs overflow the Windows stack.
4. **Tree node realloc safety**: When building trees, `realloc` can move the nodes array. Never hold pointers across `alloc_node` calls — use indices. The `expand_chance_nodes` function snapshots node data before expanding for this reason.
5. **GPU memory at 80 hands**: The full-tree solver uses 3.8GB. Single-street solve would use <100MB. Don't try to fit the full tree with more bet sizes — it won't fit.
6. **Python path**: Use `/c/Users/Victor/AppData/Local/Programs/Python/Python39/python.exe` for Python 3.9 on Windows. MSYS2 python at `/c/msys64/ucrt64/bin/python3.exe` also works.
7. **Unicode in Python output**: Use ASCII-only print statements. CP1252 encoding can't handle box-drawing characters.
