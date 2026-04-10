# Next Session Context

Last updated: 2026-04-10

Read `~/.claude/projects/C--Users-Victor/memory/postflop_wiring_session.md` for the full context across the 2026-04-09 and 2026-04-10 sessions.

## Where we left off

The unified preflop → postflop flow is **fully working end-to-end** and verified via Playwright:

1. `/solver` loads 6-player preflop ranges from the 1.5B-iter blueprint JSON
2. User walks the preflop action tree (e.g. CO raise → BTN fold → SB fold → BB call)
3. "Preflop Complete — Deal the Flop" card appears with an inline BoardPicker
4. User picks 3-5 board cards → "Solve Flop" button appears
5. GPU CFR runs (200 iterations, ~14s HU flop on RTX 3060) with narrowed ranges
6. Postflop strategies display in the same SolverShell/RangeGrid for both players

Layer 3 rollout-based leaf values are implemented and tested (56/56) but disabled by default (`USE_ROLLOUT_LEAVES=true` to enable). The Python rollout is ~100x slower than equity-only — needs Layer 3.3 CUDA port for production use.

## What to do next

### Priority 1: Layer 4.A — Multi-way flop
Fix `extract_leaf_info_from_tree` for variable `n_act` per leaf. Add 2-raises depth limit for 3+ player flops per Pluribus supplement §4. Currently the API returns 400 for 3+ players.

### Priority 2: Layer 4.B — Multi-street GPU CFR
Port chance-node kernels from `flop_solve.cu` into `street_solve.cu` for HU flop→end-of-game and turn→end-of-game. Biggest CUDA work item (~1-2 weeks).

### Priority 3: Layer 6 — Final-iteration strategy
Return the last-iteration strategy instead of σ̄ for postflop decisions. Fixes convergence artifacts (zigzag pair-ladder) on rare lines. ~2 hours.

### Priority 4: Layer 5 — Preflop re-solve
Trigger a subgame re-solve from the preflop root when an opponent bets >$100 off any tree size. Currently uses pseudoharmonic interpolation for all off-tree bets.

### Remaining bugs
- C walker bug in `bp_compute_action_evs` — only visits 169 preflop info sets out of 1.21B. Blocks per-action EV display in frontend.

### What was fixed in the 2026-04-10 session
- Unified preflop → postflop flow (Priority 1 from prior session) — DONE
- Layer 3 rollout-based leaf values — DONE (betting sim, array builder, wiring)
- GPU segfault on consecutive solves — FIXED (skip explicit free())
- Frontend preflop tree walker loop-back bug — FIXED (findNextToAct checks committed < currentBet)
- Action-hash test failures — FIXED (setHashMixer("boost") in tests)
- All tests passing: 164/164 vitest, 56/56 rollout, 5/5 API e2e

## How to start the stack

```bash
# Terminal 1 — API server
cd C:/Users/Victor/Documents/Projects/nexusgto-api
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Frontend
cd C:/Users/Victor/Documents/Projects/nexusgto
npm run dev
```

Then visit `http://localhost:3000/solver` — walk preflop, deal flop, solve.

## Performance (RTX 3060, 200 CFR iterations)

| Street | Hands | Time |
|--------|-------|------|
| HU Flop | 169 | ~14s |
| HU Turn | 169 | ~3s |
| HU River | 169 | ~1.4s |

## EC2

All instances terminated. No running costs. v3 .bps in S3 at `s3://poker-blueprint-unified-v2/unified_blueprint_v3_1.5B.bps` and locally at `blueprint_data_v3/unified_blueprint_v3_1.5B.bps` (14.66 GB).
