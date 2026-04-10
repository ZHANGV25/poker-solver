# Next Session Context

Read `~/.claude/projects/C--Users-Victor/memory/postflop_wiring_session.md` for the full context from the 2026-04-09 sprint.

## Where we left off

The postflop API is live on the local RTX 3060. Preflop serves from a fresh 1.5B-iter blueprint JSON. Both work end-to-end through the frontend. But the UX is disconnected — preflop and postflop are separate pages with separate inputs. The postflop solver uses uniform ranges instead of the narrowed ranges from preflop action.

## What to do next

### Priority 1: Unified preflop → postflop flow

Turn `/solver` into one continuous session. After all preflop actions resolve, show a board card picker. User deals the flop. The narrowed ranges from the preflop walk feed into the GPU CFR solver as starting ranges. Same position bar, same range grid, same action buttons — just backed by GPU CFR now. Turn and river continue the same way.

Key pieces:
- `nexusgto/src/lib/preflop-data.ts` already walks the preflop tree and produces per-position `RangeStrategy`. Extract the per-hand frequencies as range weights.
- `nexusgto-api/app/services/postflop_provider.py` `GPUPostflopProvider.solve()` currently builds a uniform 169-hand range. Change it to accept explicit range weights from the client.
- `PostflopQuery` in `nexusgto-api/app/models/postflop.py` needs a `ranges` field (optional dict of position → hand weights).
- `solver-state.ts` or a new `use-full-hand-session.ts` hook manages the transition from preflop JSON lookups to postflop API calls.

### Priority 2: Pluribus parity (Layers 3-6)

- **Layer 3**: Rollout-based leaf values. `python/rollout_leaves.py` is scaffolded. Needs the betting simulation loop that walks turn+river using biased blueprint σ̄ at each decision.
- **Layer 4.A**: Multi-way flop. Fix `extract_leaf_info_from_tree` for variable `n_act` per leaf. Add 2-raises depth limit for 3+ player flops.
- **Layer 4.B**: Multi-street GPU CFR. Port chance-node kernels from `flop_solve.cu` into `street_solve.cu`.
- **Layer 5**: Preflop re-solve trigger on >$100 off-tree bets.
- **Layer 6**: Final-iteration strategy (not σ̄) for postflop decisions.

### Priority 3: Bug fixes

- Frontend preflop tree walker stops showing action buttons when action loops back to an earlier position (e.g., CO after BB folds). Node exists in JSON, UI just doesn't continue the walk.
- C walker bug in `bp_compute_action_evs` — only visits 169 info sets. Blocks per-action EV display.

## How to start the stack

```bash
# Terminal 1
cd C:/Users/Victor/Documents/Projects/nexusgto-api
uvicorn app.main:app --reload --port 8000

# Terminal 2
cd C:/Users/Victor/Documents/Projects/nexusgto
npm run dev
```

Then visit `http://localhost:3000/solver` (preflop) or `http://localhost:3000/solver/postflop` (postflop).

## EC2

All instances terminated. No running costs. The v3 .bps is in S3 at `s3://poker-blueprint-unified-v2/unified_blueprint_v3_1.5B.bps` and downloaded locally at `poker-solver/blueprint_data_v3/unified_blueprint_v3_1.5B.bps`.
