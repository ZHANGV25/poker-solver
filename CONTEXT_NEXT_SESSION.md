# Next Session Context

Last updated: 2026-04-10

Read `~/.claude/projects/C--Users-Victor/memory/postflop_wiring_session.md` for the full context across the 2026-04-09 and 2026-04-10 sessions.

## Where we left off

The unified preflop → postflop flow is **fully working end-to-end** (verified via Playwright). But the API is stateless — each solve is independent. The next step is to make it session-based with Pluribus-faithful mechanics.

## THE PLAN: Session-Based Pluribus API

### What's being built

Server-side hand sessions that track state from preflop through river with:
- **Bayesian range narrowing**: when a player acts, `P(hand|action) ∝ P(action|hand) × P(hand)`
- **Strategy freezing** (Pluribus A3): on re-solve, hero's already-taken actions are frozen
- **Off-tree bet interpolation**: pseudoharmonic instead of snap-to-nearest
- **Cross-street continuity**: pot, stacks, board, and narrowed ranges carry across streets

### Phase 1: Session Models & Store (nexusgto-api)

New files:
- `app/models/session.py` — `HandSession` dataclass, request/response models
- `app/services/session_store.py` — In-memory store + Pluribus pipeline
  - Import `RangeNarrower` from poker-solver
  - On create: initialize per-position ranges from preflop weights, expand 169 hand classes → 1326 combos
  - On action: Bayesian update via `RangeNarrower.update()`
  - On deal: solve with narrowed ranges, store strategies for next narrowing
  - 30-min auto-expiry, max 100 concurrent sessions

### Phase 2: Session Endpoints (nexusgto-api)

New file: `app/routers/session.py`
- `POST /api/session` — Create from preflop state (positions, range weights, pot, stacks)
- `POST /api/session/{id}/deal` — Deal cards, trigger GPU solve with narrowed ranges, return strategies
- `POST /api/session/{id}/action` — Player acts, Bayesian narrowing, update pot/stacks
- `GET /api/session/{id}` — Get current state
- `DELETE /api/session/{id}` — Clean up

Modified: `app/main.py` (register router), `app/services/dependencies.py` (store singleton)

### Phase 3: Narrowing & Freezing Integration (nexusgto-api + poker-solver)

The Pluribus-critical piece:
- **On action**: Extract P(chosen_action|hand) from last solve's strategy → `RangeNarrower.update()`
- **Range conversion**: 1326 combos (narrowing) ↔ 169 hand classes (solver). Expand on create, collapse before solve.
- **Freezing**: Track per-position actions this street. Pass to `set_frozen_actions()` on solve. Reset on street change.
- **Off-tree**: Call `interpolate_narrowing()` from `off_tree.py` for non-tree bet amounts.

### Phase 4: Frontend Session Integration (nexusgto)

- API client: `createSession()`, `dealStreet()`, `takeSessionAction()`, `getSession()`
- Solver page: preflop complete → createSession → deal → display strategies → show postflop action buttons → user picks action → narrowing → deal next street → repeat
- Reuse ActionBar for postflop actions (check, bet 33%, bet 75%, bet 150%, all-in)

### Phase 5: Testing

- Unit: session store, range narrowing, 169↔1326 conversion
- API: full session flow via curl (create → deal → action → deal → verify ranges narrow)
- Verify narrowed ranges produce different strategies than uniform

## Key files to read before starting

### Existing Pluribus mechanics (poker-solver/python/):
- `range_narrowing.py` — `RangeNarrower` class with Bayesian updates
- `off_tree.py` — `pseudoharmonic_map()`, `interpolate_narrowing()`
- `hud_solver.py` — Reference implementation (full pipeline, not used by API)

### Current API (nexusgto-api/app/):
- `services/postflop_provider.py` — `GPUPostflopProvider` with `solve()` and `solve_all()`
- `models/postflop.py` — `PostflopQuery` with `ranges` field
- `routers/solve.py` — Stateless endpoints

### Frontend (nexusgto/src/):
- `app/solver/page.tsx` — Current unified flow
- `lib/preflop-data.ts` — `extractRangeWeights()`
- `lib/api.ts` — API client

## Performance (RTX 3060, 200 CFR iterations, HU)

| Street | Time | Bottleneck |
|--------|------|------------|
| Flop | ~15s | Leaf equity computation |
| Turn | ~2.3s | Leaf equity (smaller) |
| River | ~0.4s | Pure CFR (showdown) |

200 iterations is fully converged (zero diff vs 1000/2000).

## How to start the stack

```bash
cd C:/Users/Victor/Documents/Projects/nexusgto-api && uvicorn app.main:app --reload --port 8000
cd C:/Users/Victor/Documents/Projects/nexusgto && npm run dev
```
