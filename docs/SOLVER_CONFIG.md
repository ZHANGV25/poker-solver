# Solver Configuration — Source of Truth

This is the canonical reference for every parameter in our blueprint solver, what
Pluribus uses for the same parameter, and the rationale for any deviation.

**Authoritative external references:**
- [Brown & Sandholm 2019, Science](https://www.science.org/doi/10.1126/science.aay2400) — main paper
- [Supplementary materials PDF](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf) — full algorithm details
- [`pluribus_technical_details.md`](../pluribus_technical_details.md) — local extracted Pluribus paper details (verbatim from supplementary)

**Authoritative internal references:**
- [`docs/BLUEPRINT_BUGS.md`](BLUEPRINT_BUGS.md) — historical bug log (Bugs 1-11)
- [`docs/BLUEPRINT_CHRONICLE.md`](BLUEPRINT_CHRONICLE.md) — narrative history of training runs
- [`AGENT_COORDINATION.md`](../AGENT_COORDINATION.md) — live coordination state, BUG-A through BUG-G
- [`src/mccfr_blueprint.h`](../src/mccfr_blueprint.h) — algorithm constants
- [`precompute/blueprint_worker_unified.py`](../precompute/blueprint_worker_unified.py) — training driver
- [`precompute/launch_blueprint_unified.sh`](../precompute/launch_blueprint_unified.sh) — EC2 launch

**Maintenance contract:** Every config change to the solver, training driver, or
launch script must add a one-line entry to the [Decision log](#decision-log) below
with the date, parameter, old value, new value, and reason. **Check this doc
before every training launch.**

---

## 1. Algorithm constants

These are compile-time constants in `src/mccfr_blueprint.h` and are part of the
core MCCFR algorithm. All match Pluribus exactly.

| Param | Pluribus | v1 | v2 | Source | Status |
|---|---|---|---|---|---|
| `BP_PRUNE_PROB` | 0.95 (95% iters use pruning) | 0.95 | 0.95 | pluribus_technical_details.md §1 | ✓ matches |
| `BP_PRUNE_THRESHOLD` | -300,000,000 | -300M | -300M | pluribus_technical_details.md §1 ("C = -300M") | ✓ matches |
| `BP_REGRET_FLOOR` | -310,000,000 | -310M | -310M | pluribus_technical_details.md §1 | ✓ matches |
| `BP_REGRET_CEILING` | implicit int32 max (~2.1B) | 2,000,000,000 | 2,000,000,000 | BLUEPRINT_BUGS.md Bug 10 | ✓ matches (Pluribus has no explicit ceiling) |
| `BP_MAX_ACTIONS` | 14 sizes + fold + call = 16 max | 16 | 16 | BLUEPRINT_BUGS.md Bug 4 | ✓ matches |
| Linear CFR discount formula | `d = (T/10)/(T/10+1)` | `d = T/(T+1)` per discount step | same | mccfr_blueprint.c:2241-2243 | ✓ algebraically equivalent |
| External-sampling MCCFR | yes | yes | yes | pluribus_technical_details.md §1 | ✓ matches |

## 2. Timing parameters (CRITICAL — earlier docs were wrong)

Pluribus measures all timing in **wall-clock minutes**, not iteration counts. Per
[`pluribus_technical_details.md`](../pluribus_technical_details.md) §1: *"Iterations
are measured in time (minutes), not count, for the pruning/discounting thresholds."*

The canonical Python driver computes these as **fractions of `args.iterations`**,
which gives the correct Pluribus alignment because Pluribus's wall-clock thresholds
are themselves fractions of the 8-day training (11,520 minutes total).

| Param | Pluribus value | Pluribus fraction | Python driver formula | v2 result (8B target) | Status |
|---|---|---|---|---|---|
| `discount_stop_iter` | first 400 minutes | 400/11520 = **3.47%** | `args.iter * 35 // 1000` = 3.5% | 280M | ✓ matches |
| `discount_interval` | every 10 minutes | 10/11520 = 0.087% | `discount_stop // 40` = 0.087% | 7M | ✓ matches |
| `prune_start_iter` | after 200 minutes | 200/11520 = **1.74%** | `args.iter * 17 // 1000` = 1.7% | 136M | ✓ matches |
| `snapshot_start_iter` | after 800 minutes | 800/11520 = **6.94%** | `args.iter * 7 // 100` = 7% | 560M | ✓ matches |
| `snapshot_interval` | every 200 minutes | 200/11520 = **1.74%** | `args.iter * 17 // 1000` = 1.7% | 136M | ✓ matches |
| `strategy_interval` | 10,000 iterations (Pluribus's only iter-count threshold) | n/a | hardcoded 10000 | 10000 | ✓ matches (and dead code after Bug 6 fix; field kept for ABI) |

**Important:** The C-level defaults in `bp_default_config()` (`mccfr_blueprint.c:1340-1352`)
were calibrated to Pluribus's hardware speed (~1000 iter/min) and are 30× too small
on our hardware. They are zeroed in v2 with an init assert that forces explicit Python override.
This is **Bug C/F** in the v2 fix list. The Python driver values above are correct;
only the C defaults were broken.

**Earlier "35% of training" claim was a fabrication.** AGENT_COORDINATION.md cited
"Pluribus discounts during the first 35% of training" — this is wrong. The actual
fraction is 3.47%. Anyone reading this doc should disregard the "35%" claim wherever
it appears in older docs.

**Earlier "Pluribus did 100B iterations" claim was a fabrication.** The paper does
not specify an iteration count. Pluribus iterations are unknown.

## 3. Card abstraction

| Round | Pluribus | v1 | v2 | Status |
|---|---|---|---|---|
| Preflop (round 1) | 169 lossless classes | 169 | 169 | ✓ matches |
| Flop (round 2) | 200 buckets (k-means on EHS+features) | 200 | 200 | ✓ matches |
| Turn (round 3) | 200 buckets (same method) | 200 | 200 | ✓ matches |
| River (round 4) | 200 buckets (same method) | 200 | 200 | ✓ matches |

Source: pluribus_technical_details.md §2.

## 4. Action abstraction (deliberate deviations)

Pluribus uses "between 1 and 14 raise sizes per decision point, hand-picked by
what earlier Pluribus versions used with significant positive probability"
([pluribus_technical_details.md §3](../pluribus_technical_details.md)). This is
the most subjective part of the design.

| Round | Pluribus | v1 | v2 | Rationale |
|---|---|---|---|---|
| **Preflop level 0 (open)** | "fine-grained, up to 14" | 3 sizes [0.5, 0.7, 1.0] | 3 sizes [0.5, 0.7, 1.0] | Compute budget — BLUEPRINT_BUGS.md Bug 4 documents tree-size explosion with flat 8 sizes |
| Preflop level 1 (3-bet) | (within "fine-grained") | 2 sizes [0.7, 1.0] | 2 sizes [0.7, 1.0] | Same |
| Preflop level 2 (4-bet) | (within "fine-grained") | 1 size [1.0] | 1 size [1.0] | Same |
| Preflop level 3 (5-bet) | (within "fine-grained") | 1 size [8.0] (≈all-in) | 1 size [8.0] | Same |
| **Flop first raise** | "more coarse" | 2 sizes [0.5, 1.0] | 2 sizes [0.5, 1.0] | User confirmed sizing is correct, already reduced |
| Flop subsequent raise | (within above) | [1.0] | [1.0] | Per BLUEPRINT_BUGS.md Bug 4 |
| **Turn first raise** | exactly [0.5x, 1x, all-in] | 2 sizes [0.5, 1.0] | 2 sizes [0.5, 1.0] | **Deviation:** missing all-in. User confirmed v2 keeps current sizing. |
| Turn subsequent raise | exactly [1x, all-in] | 1 size [1.0] | 1 size [1.0] | Same — missing all-in |
| **River first raise** | exactly [0.5x, 1x, all-in] | 2 sizes [0.5, 1.0] | 2 sizes [0.5, 1.0] | Same — missing all-in |
| River subsequent raise | exactly [1x, all-in] | 1 size [1.0] | 1 size [1.0] | Same — missing all-in |

**Acknowledged deviation:** Postflop turn/river raise sizes do not include the
all-in option that Pluribus uses. User decision (2026-04-07): the current sizing
is correct and already reduced; do not add all-in for v2. Revisit if v2 produces
data that lacks credible bluff/value combinations on later streets.

## 5. Game parameters

| Param | Pluribus | v1/v2 | Status |
|---|---|---|---|
| Players | 6 | 6 | ✓ matches |
| Small blind | $50 | $50 | ✓ matches |
| Big blind | $100 | $100 | ✓ matches |
| Starting stack | $10,000 (100bb) | $10,000 | ✓ matches |
| Min raise | $100 | $100 | ✓ matches |

Source: pluribus_technical_details.md §6.

## 6. Hash table

| Param | Pluribus | v1 (custom run_solver.py) | v2 | Source |
|---|---|---|---|---|
| Slots | not specified (sized for ~665M action sequences) | 1B (BUG: custom override) | **2B** | BUG-G research, AGENT_COORDINATION.md |
| Per-slot size | not specified | ~56 bytes | ~56 bytes | BPInfoKey + BPInfoSet + occupied |
| Total metadata | not specified | ~56 GB | ~112 GB | computed |
| Arena (entries) | "lazy allocation" | ~50 GB | ~50 GB | per-thread sliced |
| Total RAM | < 512 GB | ~106 GB | ~162 GB | fits c7a.metal-48xl 384 GB |
| Probe strategy | not specified | linear, 4096 insert / 1024 read (asym) | linear, 4096 insert / 4096 read (sym) | Bug B fix |
| Lock strategy | not specified | Hogwild lock-free CAS + unbounded spin | same | BLUEPRINT_BUGS.md Bug 11 |
| Insertion failure handling | not specified | silent return -1 → fake EV=0 (Bug F) | counted via `insertion_failures` field, exposed via `bp_get_table_stats()` | new instrumentation |

**v2 sizing rationale (2B slots):**
- Empirical projection: ~1.05B entries at 8B target → 52% load
- Linear probing is comfortable up to ~70% load
- 2B is the BUG-G research answer per AGENT_COORDINATION.md, after user pushed back on 3B for cache-hostility concerns
- All sizes (1B/2B/3B) are equally L3-hostile (L3 is 384 MB, hash table is ≥56 GB regardless), so 3B does not actually win on cache; 2B saves 56 GB metadata RAM

## 7. Hardware

| Param | Pluribus | v1/v2 |
|---|---|---|
| Cores | 64 (four 16-core Xeon E5-8860 v3) | 192 (c7a.metal-48xl, AMD EPYC 9R14 Zen 4) |
| RAM | < 512 GB used (3 TB available) | 384 GB |
| Wall-clock | 8 days | v1: ~24h to 4B (terminated at 1.7B); v2: ~74h to 8B |
| Cores × hours | 12,400 | v1 used: ~2,500 (terminated); v2 will use: ~14,200 |
| Cost | ~$144 spot | v1 (terminated): ~$30 spot; v2 estimate: ~$110 spot |

**v2 compute is ~15% more than Pluribus's compute budget.** This is intentional
buffer — our action abstraction is different (more conservative postflop), our
hardware per-core is different, and we don't know Pluribus's exact iter rate. A
small buffer is safer than under-budgeting.

Source: pluribus_technical_details.md §1, §5.

## 8. Per-run targets

| Param | Pluribus | v1 (terminated) | v2 |
|---|---|---|---|
| Target iter count | unknown | 4,000,000,000 | 8,000,000,000 |
| Resume from | n/a (fresh) | 200M checkpoint | fresh (no resume) |
| S3 bucket | n/a | poker-blueprint-unified | poker-blueprint-unified-v2 |
| Spot | n/a | yes (`sir-fcr7ec9p`) | yes |
| Watchdog | n/a | none | none (per-chunk S3 upload only) |

## 9. Pluribus features we explicitly do NOT implement

These are real Pluribus features that are out of scope for the blueprint training run:

1. **Real-time search** ([pluribus_technical_details.md §4](../pluribus_technical_details.md)). Pluribus uses Monte Carlo Linear CFR / vector-based Linear CFR for subgame solving on rounds 2-4. Our solver only computes the blueprint; subgame solving happens elsewhere (or not at all).
2. **Continuation strategies (4 of them, with 5x bias multiplier)**. Used at depth-limited subgame leaves during real-time search. Not relevant to blueprint training.
3. **Pseudo-harmonic action translation**. Used during real-time play to map off-tree opponent bets to nearest blueprint size. Not relevant to blueprint training.
4. **Compressed strategy storage (< 128 GB for live play)**. We export to .bps which is ~3 GB. We don't need Pluribus's compression.

## 10. Additional algorithmic deviations

### Average Strategy Sampling for non-traverser preflop (Bug 7 fix)

**Pluribus uses pure external-sampling MCCFR.** Our solver uses external sampling
for the traverser but **Average Strategy Sampling** (Lanctot et al., NIPS 2012)
for non-traverser preflop nodes.

**Why we deviate:** [BLUEPRINT_BUGS.md Bug 7](BLUEPRINT_BUGS.md) documents the
"call trap" — at our compute scale, dominant call actions starve raise nodes of
opponent training data, freezing the strategy. ASS preserves historical action
frequencies even when the current strategy concentrates, ensuring all subtrees
get adequate training. Pluribus doesn't see this bug at their compute scale.

**Status:** kept for v2. Removing it would re-introduce the call trap.

### Tiered preflop sizing 3/2/1/1 (Bug 4 fix)

**Pluribus uses up to 14 hand-picked sizes per decision point.** Our solver uses
3/2/1/1 tiered sizing (3 open sizes, 2 3-bet sizes, 1 4-bet size, 1 5-bet size).

**Why we deviate:** [BLUEPRINT_BUGS.md Bug 4](BLUEPRINT_BUGS.md) documents the
tree-size explosion at flat 8 sizes (7.4B preflop info sets, 19× larger than
Pluribus). Tiered keeps the tree at ~386M preflop info sets while preserving
strategic granularity at the most important decision (open raise).

**Status:** kept for v2. User confirmed sizing is correct.

---

## Decision log

| Date | Param / change | Old | New | Reason |
|---|---|---|---|---|
| 2026-04-07 | Hash table size (v2) | 1B (custom) / 3B (canonical default) | **2B** | BUG-G research; 2B = 52% load at 8B target, plenty safe; 3B is no more cache-friendly than 2B since both >> L3 |
| 2026-04-07 | Iter target (v2) | 4B | **8B** | More buffer beyond Pluribus's 7B compute equivalent; ~3 days wall clock |
| 2026-04-07 | Discount C default | 400000 | 0 + assert | Bug C/F: literal Pluribus 1000-iter/min copy is wildly wrong on our hardware |
| 2026-04-07 | All other timing C defaults | 200000-800000 | 0 + assert | Bug F: same root cause as Bug C |
| 2026-04-07 | `% 10007` strategy_sum gate | active | removed | Bug E: Bug 6 regression; original Bug 6 fix removes the gate entirely |
| 2026-04-07 | Hash probe cap symmetry | insert 4096 / read 1024 | 4096/4096 | Bug B: read-side asymmetry made existing entries silently invisible |
| 2026-04-07 | `iterations_run` write | non-atomic, tid 0 only | atomic store RELEASE | Bug γ: minor data race on resume offset |
| 2026-04-07 | Hash table instrumentation | none | `insertion_failures` + `max_probe_observed` + `bp_get_table_stats()` API | Defense in depth; observable failure modes |
| 2026-04-07 | Launch script perf optimizations | missing | numactl install + THP enable + OMP env vars + numactl --interleave=all | Bug α: canonical script omitted all 6 optimizations from current run; would be ~50% slower |
| 2026-04-07 | Launch script spot mode | on-demand | spot one-time terminate | Bug ζ: cost reduction |
| 2026-04-07 | S3 bucket isolation | shared with v1 | poker-blueprint-unified-v2 | Bug θ: avoid checkpoint collisions |
| 2026-04-07 | Pluribus correction | "35% of training" / "100B iters" | 3.47% of training / unknown iters | AGENT_COORDINATION.md was wrong; pluribus_technical_details.md is the source of truth |
| 2026-04-07 | Postflop all-in option | not added to v2 | not added to v2 | User decision: current sizing is correct, already reduced |

---

## How to use this doc

1. **Before launching any new training run**, read this entire file. Verify the
   v2 column matches what you intend to launch. Update any rows where v2 has
   changed since the last run.
2. **Before changing any solver constant or training driver value**, update the
   appropriate row in the relevant section AND add a row to the Decision log
   with the date and reason.
3. **When a bug is fixed**, update the bug's Status row above and add a Decision
   log entry. Cross-reference the BLUEPRINT_BUGS.md or AGENT_COORDINATION.md
   entry where appropriate.
4. **When in doubt about Pluribus's behavior**, the source of truth is
   [`pluribus_technical_details.md`](../pluribus_technical_details.md), which is
   the local extracted version of the Brown & Sandholm 2019 supplementary
   materials. Do NOT trust other docs (including this one, AGENT_COORDINATION.md,
   or BLUEPRINT_BUGS.md) over the paper extract for Pluribus values.
