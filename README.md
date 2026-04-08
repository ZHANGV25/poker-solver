# poker-solver

A 6-player No-Limit Hold'em MCCFR blueprint solver matching the Pluribus algorithm (Brown & Sandholm, *Science* 2019), plus GPU real-time subgame search for the postflop streets. Built for training/analysis use, with the goal of being **the first true 6-player postflop solver** (commercial alternatives cap at 3 players postflop).

> 📖 **Read [`STATUS.md`](STATUS.md) first.** It is the single source of truth for project state, what's running, what's committed, what's next, and pointers into the rest of the docs.

## Components

- **Blueprint engine** (`src/mccfr_blueprint.c` + `src/card_abstraction.c`) — Production 6-player external-sampling MCCFR with bucket-in-key info sets, k-means card abstraction, Linear CFR, regret pruning, Hogwild parallelism, arena allocator.
- **GPU street solver** (`src/cuda/street_solve.cu`) — N-player single-street Linear CFR with exact showdowns, A3 strategy freezing, 4 continuation strategies at depth-limited leaves.
- **Realtime decision pipeline** (`python/hud_solver.py` + `python/leaf_values.py` + `python/range_narrowing.py` + `python/off_tree.py`) — End-to-end inference: blueprint lookup → range narrowing → GPU re-solve → off-tree mapping.
- **Training driver** (`precompute/blueprint_worker_unified.py`) — EC2 wrapper for the C solver with checkpoint/resume and S3 upload.
- **Export tool** (`precompute/export_v2.py`) — Converts a trained `regrets.bin` checkpoint into the `.bps` blueprint file consumed by the realtime path.

## Building

```bash
# Blueprint engine (requires OpenMP)
make blueprint

# Or manually:
clang -O2 -shared -fPIC -fopenmp -o build/mccfr_blueprint.so \
    src/mccfr_blueprint.c src/card_abstraction.c -I src -lm

# GPU street solver (requires CUDA)
nvcc -O2 -shared -o build/street_solve.so src/cuda/street_solve.cu -I src
```

## Pluribus parameter alignment

| Parameter | Pluribus | Ours |
|-----------|----------|------|
| Players | 6 | 6 |
| Algorithm | External-sampling MCCFR + Linear CFR + pruning | Same |
| Preflop buckets | 169 lossless | 169 lossless |
| Postflop buckets | 200 k-means | 200 k-means |
| Pruning threshold | -300M | -300M |
| Pruning probability | 95% | 95% |
| Regret floor | -310M | -310M |
| Discount formula | `d = (T/10)/(T/10+1)` | Same |
| Linear CFR discount window | First 3.47% of training | Same |
| Pruning start | 1.74% of training | 1.74% |
| Strategy interval | 10,000 iters | 10,000 iters |

For the full parameter matrix and any deviations, see [`docs/SOLVER_CONFIG.md`](docs/SOLVER_CONFIG.md).
For the Pluribus paper extraction, see [`pluribus_technical_details.md`](pluribus_technical_details.md).

## Where everything is

```
poker-solver/
├── STATUS.md                     ← single source of truth — read first
├── README.md                     ← you are here
├── ARCHITECTURE.md               ← component overview
├── pluribus_technical_details.md ← Pluribus paper extract (frozen reference)
├── REFERENCES.md                 ← citations
├── COMMERCIALIZATION.md          ← business strategy (separate concern)
├── src/
│   ├── mccfr_blueprint.c         ← the C solver (3000 LOC)
│   ├── mccfr_blueprint.h         ← algorithm constants
│   ├── card_abstraction.c        ← EHS computation + k-means bucketing
│   └── cuda/street_solve.cu      ← GPU N-player single-street CFR
├── python/
│   ├── hud_solver.py             ← realtime decision pipeline
│   ├── leaf_values.py            ← depth-limited continuation values
│   ├── range_narrowing.py        ← Bayesian range tracking
│   ├── off_tree.py               ← pseudoharmonic interpolation
│   └── ...
├── precompute/
│   ├── blueprint_worker_unified.py  ← C solver wrapper for EC2 training
│   ├── export_v2.py              ← regrets.bin → .bps conversion
│   └── launch_*.sh               ← EC2 launch scripts
├── tests/
│   ├── enumerate_tree.py         ← betting-tree enumeration
│   ├── sweep_config_tree.py      ← config sweep
│   ├── count_actionhash_vs_logical.py
│   ├── check_convergence.c       ← regret checkpoint analysis
│   └── ...                       ← many one-off analysis scripts (kept on purpose)
├── verification/
│   └── ...                       ← convergence checks
└── docs/
    ├── SOLVER_CONFIG.md          ← parameter source of truth
    ├── REALTIME_TODO.md          ← realtime/subgame backlog
    ├── V3_PLAN.md                ← v3 execution plan (Phase 1-3 shipped)
    ├── BLUEPRINT_BUGS.md         ← solver bug log
    ├── EXTRACTOR_BUGS.md         ← frontend extractor bug log
    └── BLUEPRINT_CHRONICLE.md    ← narrative training history
```

## References

- Brown & Sandholm, *"Superhuman AI for multiplayer poker"*, **Science** 2019. [Paper](https://www.science.org/doi/10.1126/science.aay2400) · [Supplement](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf)
- See [`REFERENCES.md`](REFERENCES.md) for the full citation list.
