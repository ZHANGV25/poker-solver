# Phase 1.3 Design: Per-Action EV Extraction Without Retraining

Status: **design finalized 2026-04-08, implementation in progress**
Author: solver-agent session (PC dev env)
Targets: v2 1.5B checkpoint (and forward to 8B once training completes)

---

## TL;DR

`leaf_values.py` needs per-action EVs `EV(I, a, hand)` to do Pluribus-style
depth-limited variance reduction. Today those calls return `None` and the
code falls back to a pot-size/equity approximation. The v2 blueprint
checkpoint does **not** store these EVs directly, but it stores enough
information (regrets + strategy_sum, per info set) to reconstruct them
via a **post-hoc exhaustive tree walk** using the average strategy. **No
retraining required.**

This document specifies the math, the algorithm, the storage format, and
the verification plan.

---

## 1. What the consumer wants

Consumer contract is in `python/leaf_values.py:329-351`:

```python
strat       = blueprint_store.get_turn_strategy(board_cards_str, tc_str, p)
action_evs  = blueprint_store.get_turn_action_evs(board_cards_str, tc_str, p)
# strat.shape      = [num_buckets_at_turn_node, num_actions]
# action_evs.shape = [num_actions,              num_buckets_at_turn_node]

# Consumer builds 4 biased strategies and computes EV per bias:
biased = [bias_strategy(strat, bt, cats, 5.0) for bt in range(4)]
ev_per_bias = np.zeros((4, strat.shape[0]), dtype=np.float64)
for s in range(4):
    for a in range(num_actions):
        ev_per_bias[s, :] += biased[s][:, a] * action_evs[a, :]
```

So `action_evs[a, b]` must equal:

> **The expected value in chips for player `p`, across all hands in bucket
> `b` at the turn-root info set, if player `p` takes action `a` at that
> info set and from then on everyone (player `p` included) plays the
> average blueprint strategy.**

Two deliberate simplifications are built into this:

1. **Per-bucket, not per-hand.** The consumer indexes by bucket, which
   means we can compute EVs at bucket granularity and skip per-hand card
   sampling at every downstream node. The hand-level detail only matters
   at the hand's own info set (via the bucket mapping) and at terminal
   showdowns (via eval7). Everything in between operates on buckets.
2. **"Average strategy" at every downstream node.** Not the regret-matched
   current strategy, not a sampled strategy. The **average strategy from
   `strategy_sum`**, which is what Pluribus uses for inference and what
   the depth-limited solver's opponent model expects.

This contract holds for turn, flop, and river roots — `get_flop_action_evs`
and `get_river_action_evs` have identical shapes with the corresponding
bucket counts (200 postflop, 169 preflop).

---

## 2. The math

### 2.1 Counterfactual value under the average strategy

Standard CFR notation. Let σ̄ be the average blueprint strategy, recovered
from `strategy_sum` normalized per info set. Define the counterfactual
value of info set `I` under σ̄ as:

```
v̄(I) = Σ_{z ∈ Z, I ⊑ z} π̄_{-i}(z) · u_i(z) / π̄_{-i}(I)
```

where:
- `Z` = terminal histories
- `π̄_{-i}(z)` = product of all players' (except `i`'s) reach probabilities
  from root to `z` under σ̄
- `u_i(z)` = utility to player `i` at terminal `z`, in chips
- The denominator normalizes so we get **expected chips per unit of
  opponent reach**, not a sum over all reach paths

And the **per-action counterfactual value**:

```
v̄(I, a) = Σ_{z ∈ Z, I·a ⊑ z} π̄_{-i}(z) · u_i(z) / π̄_{-i}(I)
```

Since `π̄_{-i}(z)` is only affected by opponent actions (not player `i`'s),
and `I·a` fixes player `i`'s action to `a`, the denominator stays the
same. These are the values we export.

**Key property:** `v̄(I) = Σ_a σ̄(I, a) · v̄(I, a)`. This gives us a cheap
consistency check: compute `v̄(I)` bottom-up, then `v̄(I, a)` for each
action by the recursion below, and verify they satisfy the identity.

### 2.2 Recursive definition

Let `walk(I)` return `v̄(I)` — a scalar per hand (or per bucket) at info
set `I`. We compute it bottom-up from terminal nodes:

```
walk(I):
    if terminal(I):
        if showdown:
            return eval_showdown_n(traverser=i, ...)   # per hand
        if fold:
            return pot_won or -invested                 # constant per hand
        if chance node (street transition):
            avg over next-street cards c:
                return walk(I with card c added to board)
    else:
        σ̄ = normalized strategy_sum at I
        v = 0
        for action a in legal_actions(I):
            v_a = walk(I·a)                             # recursive child EV
            v += σ̄[a] * v_a
        return v

walk_action(I, a) = walk(I·a)                           # child subtree EV
```

**Critical subtlety — acting player.** At a decision node, the acting
player is NOT necessarily the traverser `i` whose EV we're computing:

- If acting player **is** the traverser, `v̄(I, a)` is the traverser's EV
  conditional on **choosing** action `a`. The traverser explores all
  actions; the regret identity `Σ_a σ̄[a] · v̄(I, a) = v̄(I)` holds.
- If acting player **is NOT** the traverser, we compute `v̄(I)` as the
  σ̄-weighted average over the opponent's actions — same formula — and
  we don't split out per-action EVs for the opponent (we don't export
  those; we only export EVs at the traverser's decision nodes).

### 2.3 What "per bucket" means concretely

The blueprint's info sets are keyed by `(player, street, bucket,
action_hash)`. A single bucket at the turn node covers many hand
combinations that share similar EHS features. When we compute
`v̄(I, a)` at a turn info set:

- For each action, we recurse into a child subtree (river cards, more
  betting, showdown)
- At showdowns, we need specific hands to evaluate — so we sample a
  representative set of hands from the bucket, compute per-hand EVs, and
  average them to get the bucket-level EV
- For the consumer's `action_evs[a, bucket]` return value, the bucket
  index `b` in the output matches the bucket index in the blueprint —
  the consumer's own "hand → bucket" mapping is the same as the one
  used during training, so indexing is consistent

**For hands that share a bucket, they get the same EV.** This is an
abstraction-level approximation the consumer already accepts — see
`leaf_values.py:367-390` where the consumer maps per-hand indices to
bucket indices by iterating through the blueprint's hand list.

### 2.4 Chance nodes between streets

When the current subtree reaches "end of current street, transition to
next street," we insert a **chance node** that averages over all
non-conflicting next-street cards:

```
walk(I at end of street k) = (1/|C_valid|) · Σ_{c ∈ C_valid} walk(I with c)
```

where `C_valid` = 52 - (cards already in board) - (cards in traverser's
hand at this info set — but since we're per-bucket and hands within a
bucket share a card constraint, we iterate over bucket-representative
hands).

**Optimization:** the number of chance cards is small (47 turns, 46
rivers). The number of traverser hand combinations in a bucket is also
bounded (at most ~100 hands per bucket in flop/turn, fewer at river).
Chance-node compute is dominated by the child subtree walks, which are
amortized via **memoization** (§3).

---

## 3. Algorithm: σ̄-sampled MCCFR-style walk

**Revised design (2026-04-08):** the original "bottom-up exhaustive walk"
approach hit a fundamental problem — it needs per-bucket opponent ranges
at every decision node, which requires either a bucket-vs-bucket equity
table we don't have, or per-hand enumeration that's computationally
infeasible. Instead, we take the simpler and more faithful approach:

**Re-run `traverse()`-style MCCFR iterations, but sampling actions from
σ̄ (the average blueprint strategy) instead of regret-matched σ, and
accumulate `action_values[a]` into a new per-info-set EV accumulator
instead of using them for regret updates.**

### 3.1 Why this works

Look at `mccfr_blueprint.c:1361-1363` inside the existing `traverse()`:

```c
action_values[a] = traverse(&child, next_order, acting_order, num_in_order);
node_value += strategy[a] * action_values[a];
```

**The per-action EVs are already computed on every iteration.** They're
used transiently for the regret update and then discarded. Our job is:

1. Change `strategy = regret_match(is->regrets)` to
   `strategy = normalize(is->strategy_sum)` so that opponents sample
   actions from σ̄ instead of the current regret-matched strategy
2. Write `action_values[a]` into a new accumulator `is->action_evs[a]`
   and maintain a visit counter per info set
3. After N iterations, `action_evs[a] / visit_count` is the
   sample-averaged EV of action `a` for the acting player under σ̄

This is MCCFR external sampling but with the sampling distribution fixed
to σ̄ instead of the iteratively-updated regret-matched strategy, and
with a different accumulation target (EVs instead of regrets).

### 3.2 Why it gives the RIGHT values

Under σ̄, the sampled `action_values[a]` is an unbiased estimator of
`v̄(I, a)` — the counterfactual value of action `a` at info set `I`
under the average strategy. This is because:

- The card-sampling at chance nodes is uniform (same as training)
- The action-sampling at opponent nodes uses σ̄ (the blueprint's average
  strategy — what we're trying to measure EVs against)
- At the traverser node, we exhaustively enumerate actions (same as
  training), so we get one EV per action per visit

Averaging over N independent samples converges to the true `v̄(I, a)`
at rate O(1/√N). For N=10M iterations across 1.2B info sets, most info
sets get at least ~1 visit (external sampling + pruning means visit
density is heavily skewed toward the in-equilibrium parts of the tree —
which is exactly where we need accurate EVs).

### 3.3 Iteration count budget

Training runs 1.5B iterations to converge the blueprint. For EV
extraction, we need much less — we're not training, we're measuring.
Target: **50M iterations**, single-threaded ~15 minutes, multi-threaded
~2-3 minutes on an 8-core EC2 instance.

Fallback for rarely-visited info sets: if `visit_count[I] < 5`, mark
those EVs as "low-confidence" in the exported data. Consumer can
fallback to the equity approximation at those info sets.

### 3.4 Memoization is unnecessary

Under this design, there's no tree walk — each iteration is a single
linear top-to-bottom path through the game. Memoization happens
implicitly via accumulation across iterations: info sets visited more
often get more samples and more accurate EVs.

### 3.5 Per-info-set output

Each `BPInfoSet` gains `float *action_evs` (arena-allocated, length
`num_actions`) and an atomic `int visit_count`. On each traverser visit
during the EV walk:

```c
for (int a = 0; a < na; a++)
    __atomic_fetch_add(&is->action_evs[a], action_values[a], relaxed);
__atomic_fetch_add(&is->visit_count, 1, relaxed);
```

At export time, divide by `visit_count` to get the sample-averaged EV.

**Memory cost:**
- 1.2B info sets × 4 actions × 4 bytes (float32) = ~19 GB for EVs
- 1.2B info sets × 4 bytes (int32 counter) = ~5 GB for counters
- Total additional: **~24 GB** on top of the 60 GB checkpoint load
- Fits in a 96 GB or 128 GB EC2 instance, or tight in 64 GB (may need
  a bigger box than originally planned)

### 3.6 Non-decision nodes

Showdowns, folds, and chance nodes are computed inline per iteration,
same as training. No storage cost.

---

## 4. Storage format (schema v3)

The existing .bps file format (from `export_v2.py`) is:

```
[4B "BPS3"] [8B u64 compressed_strategies_size] [4B u32 meta_size]
[compressed_strategies (LZMA)]
[meta_json]
```

**Schema v3 adds a trailing optional section** after `meta_json`:

```
[4B "BPR3"] [8B u64 compressed_action_evs_size]
[compressed_action_evs (LZMA)]
```

Where the uncompressed action_evs blob is:

```
[4B "BPR3"] [4B u32 num_entries]
Per entry:
  [1B player] [1B street] [2B bucket]
  [8B action_hash] [1B num_actions]
  [4 * num_actions B: float32 EV per action]
```

**Total size budget:**
- 1.2B info sets × (16 bytes key + 4×4 bytes EV) ≈ **38 GB uncompressed**
- LZMA compression on EV float32 data ≈ 30-40% of original ≈ **12-15 GB**
  compressed on disk, plus the existing strategy blob (~300 MB)

The schema v3 file will be meaningfully larger than v2 (~15 GB vs
~500 MB), which is fine for S3 storage but bad for the browser/Vercel
path. **We do NOT ship the full v3 file to the frontend** — the
frontend continues consuming the strategies-only v2 extraction. The v3
file is consumed server-side by nexusgto-api / the realtime solver.

**Backward compatibility:** a v2 reader that stops after `meta_json`
ignores the trailing BPR3 section. A v3 reader that finds no BPR3
trailing section falls back to `get_turn_action_evs() → None` and the
consumer goes down the existing equity path. No breakage either way.

---

## 5. Metadata changes

Add to the JSON metadata in `export_v2.py`:

```json
{
    "schema_version": 3,
    "has_action_evs": true,
    "action_evs_compute_method": "posthoc_tree_walk_avg_strategy",
    "action_evs_walk_iterations": <iters at which strategy_sum was snapshotted>,
    ...
}
```

---

## 6. Implementation plan

### 6.1 C side (new functions in `mccfr_blueprint.c`)

1. `static float traverse_ev(TraversalState *ts, int acting_order_idx,
   const int *acting_order, int num_in_order)` — a sibling to the
   existing `traverse()`. Identical structure except:
   - Strategy computation at every decision node uses σ̄ from
     `strategy_sum`, not regret matching
   - Opponents still sample one action (external sampling), but sample
     from σ̄ instead of regret-matched σ
   - Traverser exhaustively enumerates actions (same as training)
   - Accumulates per-action EVs into `is->action_evs[a]` via
     `__atomic_fetch_add` (Hogwild-style, same pattern as regret updates)
   - Increments `is->ev_visit_count` atomically
   - Does NOT update regrets or strategy_sum — read-only except for
     the EV accumulators
2. `int bp_compute_action_evs(BPSolver *s, int64_t num_iterations)` —
   top-level driver. Runs `num_iterations` of `traverse_ev()` across
   all 6 players and all threads. Parallelized with OpenMP. Default
   count: 50M.
3. `BP_EXPORT int bp_export_action_evs(const BPSolver *s, unsigned char *buf,
   size_t buf_size, size_t *bytes_written)` — parallel to
   `bp_export_strategies`. Emits the BPR3 binary blob with EV values
   = `action_evs[a] / ev_visit_count` for each info set. Info sets
   with `ev_visit_count == 0` are skipped (consumer falls back to
   equity approximation).

### 6.2 BPInfoSet extension

```c
typedef struct {
    int num_actions;
    int *regrets;
    float *strategy_sum;
    float *action_evs;     /* NEW: [num_actions], NULL until ensure_action_evs */
    int ev_visit_count;    /* NEW: visit counter for EV averaging */
} BPInfoSet;
```

Allocated lazily by `ensure_action_evs()` on first visit during the
EV walk, arena-backed (same pattern as `ensure_strategy_sum`).

### 6.3 Python side

1. `precompute/export_v2.py`:
   - Call `bp_compute_action_evs` after the strategies export
   - Call `bp_export_action_evs` to get the BPR3 blob
   - LZMA-compress and append to the .bps file
   - Bump `schema_version` to 3
2. `python/blueprint_v2.py`:
   - Extend `_load_bps3` to peek for a trailing `"BPR3"` magic after the
     existing sections
   - If present, parse the BPR3 blob into a parallel table keyed by
     `(board_hash=0, action_hash, player, street) → np.array[num_buckets, num_actions]`
   - Expose `get_turn_action_evs(board, tc, p)` and siblings matching
     the consumer contract shape

### 6.4 Ordering: the walk IS the hot path

The walk must process ~1.2B info sets. Single-threaded at ~1 µs per
info set = **20 minutes**. With 8 threads = **2.5 minutes**. Realistic
target with memoization hits: **under 10 minutes on a CPU box**.

We run the walk on EC2 right after `bp_load_regrets` completes, inside
the same export run. No separate EC2 trip.

---

## 7. Ground-truth verification

### 7.1 Synthetic test case

Build a tiny 2-player 2-street toy game in C with known strategies and
payoffs:

- 2 players, 2 buckets per street, 2 actions per decision (fold/bet)
- Hand-compute the exact `v̄(I, a)` for the root info set
- Run `bp_compute_action_evs` on the toy game
- Assert exported values match to within FP noise (< 1e-5 relative error)

### 7.2 Real-data sentinel check

After running against the 1.5B v2 checkpoint on EC2:

1. Pick a specific info set: **UTG open-raise decision at 100bb stacks,
   bucket 0 (= AA equivalence class in the 169-class preflop lossless
   abstraction)**. This is the most well-trained info set in the entire
   blueprint.
2. Extract the average strategy (should be ~95% 1.0x pot raise)
3. Extract per-action EVs: `EV(raise 1.0x) should be clearly positive
   (AA is value-betting), EV(fold) should be 0, EV(limp/call) should
   be positive but smaller than the raise`
4. Also check the regret-based prior: `action_evs[a] - action_evs_avg ≈
   regrets[a] / iterations_run`. This is the CFR consistency check —
   since regrets are the time-weighted sum of EV differences, the
   average EV difference should match the exported EVs modulo
   discount/snapshot noise.

### 7.3 End-to-end leaf_values.py check

Load the new .bps file in `leaf_values.py`, run `compute_flop_leaf_values`
on a specific flop spot, verify that:
- `get_turn_action_evs` returns non-None
- The 4 biased leaf values are genuinely different (not collapsed)
- The magnitudes are within the same order of magnitude as the pot

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Walk is slower than 10 min on 1.2B info sets | Profile early on a smaller checkpoint (200M), scale estimates, add OpenMP parallelism if needed |
| Memoization collisions produce wrong EVs | Memoization key IS the info set key, so collisions ARE strategic equivalences; verify via consistency check `Σ_a σ̄[a] · v̄(I, a) = v̄(I)` at every stored info set |
| Average strategy has zero-visit nodes → fallback to uniform produces garbage EVs | Accept. Those info sets are pruned branches the blueprint never explored; their EVs are low-quality but marked as such via a visit-count metadata field. Consumer can filter them out. |
| Chance node averaging over cards is biased if traverser hand conflicts | Enumerate only non-conflicting cards at each chance node (same logic as `traverse()` uses at line 976-979) |
| Tree walk memory explodes via unbounded recursion | Walk depth is bounded by the game tree depth (~40 for 6-player 4-street), so stack is fine. Memoization table is bounded by the number of distinct info sets (1.2B) which fits in arena |
| Export BPR3 section > 20 GB on disk | Acceptable for server-side consumption. Don't ship to browser. |

---

## 9. What's out of scope

- **Hand-level (not bucket-level) per-action EVs.** The consumer uses
  per-bucket already. Refining to per-hand would blow up memory 100x.
- **Exposing action EVs to the frontend.** Server-side only. Browser
  continues using the compact strategies-only preflop-nodes.json.
- **Retraining v3 to add runtime accumulation.** Not needed — this
  approach is fully post-hoc.
- **Rebuilding the v2 checkpoint format.** We only add a trailing
  section; existing v2 consumers are unaffected.

---

## 10. Deliverables checklist

- [ ] `src/mccfr_blueprint.c`: `walk_info_set_ev`, `bp_compute_action_evs`,
      `bp_export_action_evs`
- [ ] `src/mccfr_blueprint.h`: function declarations, `BPInfoSet`
      extension
- [ ] `precompute/export_v2.py`: call new functions, bump schema, append
      BPR3 section, update metadata
- [ ] `python/blueprint_v2.py`: load BPR3 section, expose
      `get_turn_action_evs` / `get_flop_action_evs` /
      `get_river_action_evs`
- [ ] `tests/test_phase_1_3_synthetic.py`: toy game ground-truth check
- [ ] EC2 run against 1.5B v2 checkpoint, upload v3.bps to S3
- [ ] Sentinel verification on the UTG/AA test case
- [ ] Local smoke test: `compute_flop_leaf_values` with new .bps
- [ ] STATUS.md update marking Phase 1.3 complete
- [ ] Commit to master
