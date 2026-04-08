# precompute/

> See [`../STATUS.md`](../STATUS.md) for project state. This README maps the precompute scripts to current vs legacy.

This directory has two generations of scripts. Use the **current** ones; the **legacy** ones predate the unified blueprint approach and reference an external HUD project that's no longer in scope.

## Current — unified blueprint pipeline

These are the scripts the v2/v3 training pipeline uses.

| Script | Role |
|---|---|
| `blueprint_worker_unified.py` | The training driver. Loads `mccfr_blueprint.so` via ctypes, sets the deployed config (3-2-1-1 preflop tiers, max raises 4/3, 2B hash slots), runs MCCFR, checkpoints to S3. |
| `launch_blueprint_unified.sh` | EC2 launch script. Provisions a c7a.metal-48xl spot instance, builds the solver, launches the training run. **Use this for new training runs.** |
| `export_v2.py` | Converts a `regrets_*.bin` checkpoint into the `.bps` blueprint file consumed by `python/blueprint_io.py`. Currently writes `schema_version: 2`. Phase 1.3 (per-action EVs) needs this updated to schema_version 3 — see [`../STATUS.md#v3-commit-status`](../STATUS.md#v3-commit-status). |
| `extract_preflop_json.py` | Pulls preflop strategies out of a `.bps` for the frontend (nexusgto). |
| `check_s3_sync.sh` | Sanity check that the source tree on S3 matches the local source tree before launch. |
| `verify_export_freqs.py` | Spot-check tool: pulls strategies for sentinel hands from a `.bps` and validates they look reasonable. |
| `launch_export_*.sh` | Launches a small EC2 instance to run `export_v2.py` against an S3-stored checkpoint (used to produce `.bps` files without burning capacity on the training instance). |

## Legacy — pre-unified per-flop pipeline

These scripts are from an older approach where each flop texture was solved separately, then assembled into a per-flop strategy library. They reference an external `ACRPoker-Hud-PC/solver/ranges.json` file that no longer exists in the active workflow. **They are dead code in the current pipeline.** Kept for historical reference and because some of them have useful one-off utilities embedded in them.

| Script | Status |
|---|---|
| `launch_blueprint_v2.sh` | LEGACY — superseded by `launch_blueprint_unified.sh` |
| `launch_blueprint.sh` | LEGACY |
| `run_all.py` | LEGACY — old per-scenario solver pipeline |
| `gpu_precompute.py` | LEGACY — old GPU per-flop precompute |
| `solve_scenarios.py` | LEGACY — per-scenario solver |
| `scenario_matrix.py` | LEGACY — scenario matrix builder |
| `range_parser.py` | LEGACY — parses an external ranges.json |
| `deploy_gpu.sh` | LEGACY — old GPU deployment |
| `blueprint_worker.py` | LEGACY — pre-unified per-texture worker |

If you need to read or modify these to extract some shared logic, fine — but **do not extend them** for new functionality. New work goes in the current pipeline above.
