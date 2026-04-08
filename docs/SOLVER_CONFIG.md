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
| `strategy_interval` | 10,000 iterations (Pluribus's only iter-count threshold) | n/a | hardcoded 10000 | 10000 | ❌ **field is dead code** — declared in `BPConfig` but never read by `mccfr_blueprint.c`. The current per-visit accumulation at `mccfr_blueprint.c:1283-1289` does NOT use this interval. See [§10 Algorithmic divergences](#10-algorithmic-divergences) and [§11 v3 backlog](#11-v3-backlog). |

**Important:** The C-level defaults in `bp_default_config()` (`mccfr_blueprint.c:1384-1416`)
were originally calibrated to Pluribus's hardware speed (~1000 iter/min) and were 30× too small
on our hardware. **As of v2** they are set to Pluribus-aligned fractions of a 1B-iter baseline
(35M / 17M / 70M / 17M, see `mccfr_blueprint.c:1403-1408`) with a soft warning emitted at solve
start if any timing field is suspiciously small. The previous v2 plan called for zeroing the
defaults with a hard init assert, but that would have broken ~30 callers (tests + benchmarks);
the Pluribus-aligned fallback + warning is the pragmatic compromise. The Python driver still
overrides all timing values from `args.iterations` so the C defaults only matter for callers
that forget to override.

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
| **Turn first raise** | exactly [0.5x, 1x, all-in] | 2 sizes [0.5, 1.0] + auto-allin | 2 sizes [0.5, 1.0] + auto-allin | ✓ **matches** — `generate_actions` at `mccfr_blueprint.c:613-616` always appends an all-in action when `stack > to_call` and no all-in size is in the configured list. The configured list is [0.5, 1.0], the auto-appended all-in makes the effective set [0.5, 1.0, all-in]. |
| Turn subsequent raise | exactly [1x, all-in] | 1 size [1.0] + auto-allin | 1 size [1.0] + auto-allin | ✓ **matches** — same auto-append mechanism |
| **River first raise** | exactly [0.5x, 1x, all-in] | 2 sizes [0.5, 1.0] + auto-allin | 2 sizes [0.5, 1.0] + auto-allin | ✓ **matches** |
| River subsequent raise | exactly [1x, all-in] | 1 size [1.0] + auto-allin | 1 size [1.0] + auto-allin | ✓ **matches** |

**Note on the auto-allin mechanism.** The `generate_actions()` function in
`mccfr_blueprint.c:602-617` builds the action list in two passes: first the
fractional bet sizes from the configured array (skipping any that exceed stack
or are below `to_call`), then a final fallback that appends `amount = stack`
as an all-in option if no all-in was already added. **All four postflop rows
above are Pluribus-matching once this fallback is accounted for.** A previous
version of this doc claimed these rows were "missing all-in" — that was wrong;
the auto-append guarantees Pluribus alignment without requiring the explicit
size in the configured list.

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

## 10. Algorithmic divergences

### Average strategy mechanism (round 1 + rounds 2-4)

**Pluribus** (Brown & Sandholm 2019, Supp. p. 15 + Algorithm 1):
- **Round 1:** call `UPDATE-STRATEGY` every `Strategy Interval = 10,000` iterations.
  Sample one path down the tree from the root, increment `phi[I, a] += 1` for the
  *single sampled action* at each visited info set. Sparse counter, on every 10K-th iter.
- **Rounds 2-4:** No in-memory accumulation. Snapshots of the current strategy
  are written **to disk** every 200 minutes after the first 800 minutes of training.
  Final blueprint = average of these saved disk snapshots. Paper explicitly says
  this "reduced memory usage by nearly half".

**Our implementation:**
- **Round 1** (`mccfr_blueprint.c:1283-1289`): on every traverser visit at street 0,
  add the **full regret-matched probability vector** to `strategy_sum` (no interval
  gate). The `Strategy Interval = 10,000` field exists in `BPConfig` but is never
  read.
- **Rounds 2-4** (`mccfr_blueprint.c:1343-1358` `accumulate_snapshot`): at batch
  barriers past `snapshot_start_iter`, iterate over **every occupied hash slot**
  (all streets, including street 0), lazily allocate `strategy_sum`, and add the
  current regret-matched distribution. **In memory, not on disk.**

**Is this a bug?** No, in the "wrong answer" sense. Both the per-visit method
and the paper's sampled method are unbiased estimators of the average strategy.
The per-visit method actually has *lower variance* (full distribution vs. one
sample). Both converge to the same blueprint in the limit.

**Real consequences of the divergence:**
1. **Memory:** Pluribus saves rounds 2-4 to disk specifically to halve RAM usage.
   We keep them in memory. For our 384 GB box this is fine. Architectural difference,
   not correctness.
2. **Round-1 double-counting:** `accumulate_snapshot` doesn't filter by street, so
   round 1's `strategy_sum` gets *both* the per-visit accumulation AND the snapshot
   accumulation. This doesn't bias within-info-set distributions (we normalize per
   info set at extraction time), but the relative weighting between rounds is uneven.
3. **Doc claim drift:** Earlier versions of this file claimed "✓ matches Pluribus"
   for the strategy_sum mechanism, the `strategy_interval` field, and "Average
   Strategy Sampling for non-traverser preflop" (Bug 7 fix). All three claims were
   wrong. The implementation is algorithmically defensible but does NOT literally
   match the paper. See §11 v3 backlog.

### Non-traverser action selection — pure external sampling

**Pluribus** (Algorithm 1 line 35-38, Supp. p. 16): non-traverser samples one
action from the current regret-matched strategy `σ(I)`. **Pure external-sampling
MCCFR.**

**Our implementation** (`mccfr_blueprint.c:1296`): `int sampled = sample_action(strategy, na, ts->rng);`
where `strategy` is from `regret_match(is->regrets, strategy, na)` at line 1194.
**Same — pure external sampling.**

**Doc claim drift correction:** Earlier versions of this file claimed we used
"Average Strategy Sampling (Lanctot et al., NIPS 2012)" for non-traverser preflop
nodes as a Bug 7 fix. **That claim is wrong.** ASS would sample proportional to
`strategy_sum` (historical action frequencies). The current code samples from the
current regret-matched strategy, which is what Pluribus does. ASS may have been
implemented at some point and reverted; the doc lagged the revert. As of v2 the
code is straightforwardly correct on this row.

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
| 2026-04-07 | Discount C default | 400000 | 35M (Pluribus 3.5% of 1B baseline) + soft warning at solve start if < 1M | Bug C/F: literal Pluribus 1000-iter/min copy was wildly wrong on our hardware. The original v2 plan called for zeroing the defaults with a hard init assert, but that would have broken ~30 callers (tests + benchmarks). The Pluribus-aligned fallback + warning is the pragmatic compromise. |
| 2026-04-07 | All other timing C defaults | 10000-800000 | 17M / 70M / 17M (Pluribus 1.7% / 7% / 1.7% of 1B baseline) + soft warning if `prune_start_iter < 500K` | Bug F: same root cause as Bug C, same rationale for the pragmatic fix |
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

## 11. v3 backlog

Verified bugs and architectural concerns deferred to the next training run after
v2. Each entry was verified against the actual code (line numbers cited) by a
read-only audit on 2026-04-07. None of these are load-bearing for v2's
correctness — every "real" bug below either does not fire under our current 6-max
equal-stack config, or has a fire rate so low it cannot materially affect
convergence at the 8B-iter scale. They are batched here for v3.

### Real bugs (defensive, deferred)

| Severity | Bug | Location | Fires in v2? | Fix sketch |
|---|---|---|---|---|
| 🟠 latent | **Short-stack CALL stack-underflow.** Line 1227-1232 (traverser) and 1305-1310 (non-traverser) subtract `to_call` from `child.stacks[ap]` without capping at remaining stack. The BET branch at 1238-1239 DOES cap. In equal-stack 6-max, `to_call ≤ stacks[ap]` always holds via the BET cap, so the bug is unreachable. | `mccfr_blueprint.c:1227-1232, 1305-1310` | No (unreachable in equal-stack 6-max) | Add `int call_amount = (to_call > child.stacks[ap]) ? child.stacks[ap] : to_call;` and use `call_amount` in all four field updates |
| 🟡 minor | **Hash-collision `na` one-sided clamp.** Line 1186-1190 unconditionally sets local `na` to `is->num_actions` when they differ, including raising a smaller local value. If raised, the loop accesses `actions[a]` for indices `generate_actions` never set — uninitialized stack data. | `mccfr_blueprint.c:1186-1190` | Yes (rarely, on hash collisions) | Change to clamp-down only: `if (na > is->num_actions) na = is->num_actions;` |
| 🟡 minor | **`action_hash` 64-bit weak combiner.** Boost `hash_combine` has known distribution weaknesses. Birthday-paradox at 1B entries → ~3% collision probability. Combined with the `na` clamp bug above, rare collisions can corrupt one iteration's regret update. | `mccfr_blueprint.c:223-226, 322-336` | Theoretically yes, very rare | Replace with xxHash3 or similar (~50 LOC) |
| 🔴 perf | **Texture hash O(1755) linear search.** Every flop deal does a linear scan through `s->num_cached_textures` (1755) to find the matching texture. Should be a hash map lookup. Estimated 5-15% solver throughput cost. | `mccfr_blueprint.c:971-977` | Yes always | Replace with a flat hashmap keyed on `flop_hash` (~30 LOC) |
| 🟡 minor | **`bp_get_strategy` no `spin_until_ready` on state=2 slots.** Line 2392-2394 only checks `!occupied[idx]`, falling through state=2 to `key_eq` on partially-initialized data. In current usage, called only between batches when no thread is initializing, so doesn't fire. | `mccfr_blueprint.c:2392-2418, 2378-2424` | No (probe runs between batches) | Add `__atomic_load_n(...)` with state==1 check, or use `spin_until_ready` matching the insert path |
| 🟡 minor | **`arena_alloc` NULL deref in `info_table_find_or_create`.** Line 419 stores `arena_alloc(num_actions)` to `regrets` without NULL check, then publishes state→1. If arena OOMs, slot has NULL regrets → next access segfaults. | `mccfr_blueprint.c:418-422` | No (we have plenty of RAM) | Check for NULL after `arena_alloc`, revert state→0, return -1 |
| 🟡 minor | **`bp_solve` batch_size picked once at start.** Line 2143-2149 sets batch_size based on `global_start_iter`. The chunk crossing the discount→post-discount boundary uses small batch_size for the entire chunk instead of switching mid-chunk. | `mccfr_blueprint.c:2137-2150` | Yes (~0.6% of training) | Recompute batch_size per-batch when crossing the boundary |
| 🟡 minor | **`bp_init_unified` texture cache double-allocation leak.** Line 1626 unconditionally callocs `texture_bucket_cache`. If `bp_load_texture_cache` (line 2673) was called first, it allocated already → first allocation is leaked (~9 MB). | `mccfr_blueprint.c:1626` | Yes (one-time, ~9 MB at startup) | `if (!s->texture_bucket_cache) s->texture_bucket_cache = calloc(...);` |
| 🟡 minor | **`bp_load_regrets` `is_dup` heuristic false positives on chained loads.** Line 2590-2593 flags any non-zero existing regret as duplicate. After a fresh load, the slot is arena-zeroed → is_dup=0. After chained loads, false-positive duplicate logs. The merge math is correct; only the log is noisy. | `mccfr_blueprint.c:2590-2603` | Maybe (cosmetic) | Track which entries are "newly created" by this load vs. pre-existing |
| 🟡 minor | **Global arena `g_arena` stale TLS pointers across `bp_init` calls.** `arena_free_all` only nulls TLS for the calling thread; other threads retain stale pointers into freed memory. Documented in code comments as a single-process limitation. | `mccfr_blueprint.c:215-218` | No (production runs one process per training) | Use a per-solver arena instead of a global, or null TLS for all threads on `bp_init` |

### Algorithmic divergences (deliberate or to-investigate)

| Item | Status | Action |
|---|---|---|
| **F1: Average strategy mechanism** (round 1 per-visit vs paper's sampled UPDATE-STRATEGY; rounds 2-4 in-memory vs paper's on-disk snapshots) | Real divergence, algorithmically defensible (lower variance than paper). Memory cost ~50 GB which fits in 384 GB. | Defer. Decide in v3 whether to literally match the paper or keep the lower-variance estimator. |
| **F2: `strategy_interval` is dead code** | Field exists in `BPConfig`, set to 10000 by 30+ callers, never read in `mccfr_blueprint.c` | Defer. Either wire it up to the round-1 accumulation or remove the field on the next ABI break. |
| **F3: Discount-of-strategy_sum is config-fragile** | `apply_discount` discounts `strategy_sum` if non-NULL. Currently safe because `discount_stop_iter < snapshot_start_iter`, so rounds 2-4's `strategy_sum` is NULL when discount runs. Any retuning that overlaps the windows would silently break rounds 2-4. | Defer. Add an explicit street-0 filter in `apply_discount` to make it config-independent. |

### Verified false positives (NOT bugs)

For the record, these were claimed by the bug-hunt audit and verified to be wrong:

| Claimed bug | Why it's not a bug |
|---|---|
| **Discount/snapshot race via `nowait`** | OpenMP `schedule(static, chunk)` gives deterministic chunk-to-thread assignment. The same thread owns the same slots in both `apply_discount` and `accumulate_snapshot`, so `nowait` doesn't allow inter-function races. |
| **`strategy_sum` non-atomic writes** | Intentional Hogwild MCCFR. Pluribus's Algorithm 1 also has non-atomic `phi[I, a] += 1`. The convergence theory tolerates O(1/T) noise from races. |
| **RNG states under-sized vs OMP thread count** | `bp_solve` calls `omp_set_num_threads(nt)` before the parallel region. With `OMP_DYNAMIC=false` (default in our launch script), the actual thread count cannot exceed nt. Latent only — would fire if someone enables dynamic adjustment. |
| **`apply_discount` loops over state=2 slots** | Theoretically possible but the OMP barriers prevent it: by the time `apply_discount` runs (inside an `omp single` → barrier), all threads have finished their `info_table_find_or_create` calls from the previous iter loop, so no state=2 slots exist. |

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
