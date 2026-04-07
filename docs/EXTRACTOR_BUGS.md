# Extractor Bug Report

This documents bugs in the consumer side of the blueprint pipeline — specifically
`precompute/extract_preflop_json.py`, which reads `.bps` files and produces the
`preflop-nodes.json` consumed by the nexusgto frontend.

For training-side bugs, see `docs/BLUEPRINT_BUGS.md` (Bugs 1-11).

This file tracks bugs the frontend-agent owns and fixes. Some entries also flag
upstream issues that affect data quality even when the extractor itself is correct.

## Bug A: bucket mapping mismatch — every cell showed the wrong hand's strategy

**Severity:** 🔴 critical
**Status:** ✅ FIXED 2026-04-07
**Affected versions:** all versions of `extract_preflop_json.py` prior to 2026-04-07

### The problem

The C training code (`mccfr_blueprint.c init_unified()` lines ~1383-1394) builds
its bucket → hand class mapping by iterating rank pairs from highest to lowest:

```c
int n_classes = 0;
for (int r0 = 12; r0 >= 0; r0--) {     // r0 = high rank, 12 = Ace
    for (int r1 = r0; r1 >= 0; r1--) { // r1 = low rank
        if (r0 == r1) {                // pair: 1 class
            class_map[r0][r1][0] = n_classes;
            class_map[r0][r1][1] = n_classes;
            n_classes++;
        } else {
            class_map[r0][r1][1] = n_classes; n_classes++; // suited
            class_map[r0][r1][0] = n_classes; n_classes++; // offsuit
        }
    }
}
```

This produces the bucket order:

```
0   AA
1   AKs
2   AKo
3   AQs
4   AQo
5   AJs
6   AJo
7   ATs
8   ATo
9   A9s
10  A9o
...
24  A2o
25  KK
26  KQs
27  KQo
...
168 22
```

The .bps file stores per-bucket strategy arrays indexed by these C bucket IDs.

The Python extractor's `hand_class_to_bucket` used a different convention — pairs
grouped first (buckets 0-12), then a suited block (13-90), then an offsuit block
(91-168):

```python
# OLD (buggy):
def hand_class_to_bucket(hand):
    if len(hand) == 2:
        return RANKS.index(hand[0])  # AA=0, KK=1, ..., 22=12
    r0, r1 = RANKS.index(hand[0]), RANKS.index(hand[1])
    is_suited = hand.endswith("s")
    offset = 13 if is_suited else 91
    for i in range(13):
        for j in range(i + 1, 13):
            if i == r0 and j == r1:
                return offset
            offset += 1
    return 0
```

When the extractor read `strats[hand_class_to_bucket("AKs")]` it got `strats[13]`,
but in C bucket 13 is **A7s**, not AKs. So the AKs cell in the rendered range grid
showed A7s's strategy. Every cell except AA (where both conventions agree at 0)
displayed the wrong hand.

### The visible symptom

Looking at the frontend:
- AA cell: correct (both conventions = 0)
- AKs cell: showed A7s's data → "100% fold" because A7s actually folds UTG
- KK cell: showed AKs's data → looked OK because AKs raises (which is what KK should do)
- "UTG plays" included Q5o, J5s, 95o, 64o, 92s, 82o (random weak hands)
- "UTG folds" included AKs, AKo, AQs, AQo, AJs, AJo (every premium except AA)
- Range: 30/169 hands, totally non-monotonic — looked like "random noise" rather
  than a CFR strategy

### Why I almost shipped this

When the frontend rendered, I (frontend-agent) verified with Playwright that the
UI was working — buttons clickable, range grid populated, action labels correct,
deep-tree navigation working. **What I didn't verify is whether the displayed
ranges were poker-correct.** A "AA from UTG raises ≥80%" assertion would have
caught this immediately.

I also misread my own diagnostic. When I streamed the regrets file directly and
got values like `bucket 1: AA-like premium frequencies`, I labeled bucket 1 as "KK"
in my report (because my Python convention says KK=1) and thought the data was
broken. It wasn't — bucket 1 IS AKs in the C convention, and the strategies were
correct. The bucket-label mapping in my own diagnostic script had the same bug as
the production extractor.

### The fix

`_build_c_bucket_map()` reproduces the C iteration exactly. `hand_class_to_bucket`
is now a simple dict lookup into the pre-built map. Sentinel values verified
against direct streaming of regrets_1000M.bin from S3:

```python
assert hand_class_to_bucket('AA') == 0
assert hand_class_to_bucket('AKs') == 1
assert hand_class_to_bucket('AKo') == 2
assert hand_class_to_bucket('AQs') == 3
assert hand_class_to_bucket('22') == 168
```

### Verification

The extractor now runs `verify_utg_root_sanity()` after Step 3, before writing
the JSON. It asserts:

| Hand | Constraint | Reason |
|------|------------|--------|
| AA   | fold ≤ 0.15 | premium pair never folds |
| KK   | fold ≤ 0.15 | premium pair never folds |
| AKs  | fold ≤ 0.15 | AKs never folds UTG |
| AKo  | fold ≤ 0.20 | AKo never folds UTG |
| 72o  | fold ≥ 0.95 | trash always folds |
| 32o  | fold ≥ 0.95 | trash always folds |
| 22   | fold ≥ 0.85 | small pair folds UTG 6-max |
| 52s  | fold ≥ 0.85 | weak suited folds UTG 6-max |

If 2 or more sentinels fail, the extractor raises `ValueError` and refuses to
write the JSON file. (1 failure is allowed as noise tolerance.)

After the fix, all 8 sentinels pass on the 1B blueprint:

```
  AA        0.000  <= 0.15       OK  (premium pair never folds)
  KK        0.000  <= 0.15       OK  (premium pair never folds)
  AKs       0.000  <= 0.15       OK  (AKs never folds UTG)
  AKo       0.000  <= 0.20       OK  (AKo never folds UTG)
  72o       1.000  >= 0.95       OK  (trash always folds UTG)
  32o       1.000  >= 0.95       OK  (trash always folds UTG)
  22        1.000  >= 0.85       OK  (small pair folds UTG 6-max)
  52s       1.000  >= 0.85       OK  (weak suited folds UTG 6-max)
  All 8 sentinels passed.
```

### Investigation methodology

What worked:
1. **Direct binary streaming** of regrets_1000M.bin from S3 with byte-level entry
   parsing. Read strategy_sum directly, normalized, and printed per-bucket
   strategies for known UTG root entries. Showed that the underlying training data
   was correct — the bug had to be downstream of the .bps file.
2. **Reading the C bucket map construction code** (`init_unified()`). Traced
   `for r0 in 12..0, for r1 in r0..0` and realized my Python `hand_class_to_bucket`
   produced a different ordering.
3. **Spot-decoding diagnostic buckets back to hand names** (e.g., bucket 1 = AKs,
   not KK as I'd assumed). This confirmed the .bps was internally consistent and
   the bug was purely in the Python extractor's bucket convention.

### Lessons

1. **A working UI is not a verified UI.** Playwright + screenshots prove the page
   renders. They don't prove the data is correct. Always include domain-specific
   sanity checks in the pipeline (Bug E: `verify_utg_root_sanity()` is now part of
   the extractor).
2. **Conventions need to be written down.** Both the C code and the Python extractor
   "knew" how buckets were numbered, but they disagreed silently. The
   `AGENT_COORDINATION.md` Convention Agreements section now documents the canonical
   bucket order so future consumers don't drift.
3. **Don't trust your own diagnostic when both directions use the same broken
   convention.** When my diagnostic streamed strategy_sum and printed "bucket 1"
   data, I labeled it "KK" using the same broken Python convention, then reported
   "the data is broken" instead of "my labels are broken." Cross-validating against
   a known-good external source (or trusting your eyes — KK strategies for an
   "AKs-bucket" should look like AKs, not KK) avoids this.

## Bug B: .bps contains regret-matched strategies, not strategy_sum averages

**Severity:** 🟡 moderate
**Status:** open, owned by solver-agent (see `AGENT_COORDINATION.md` BUG-B)
**Affects:** all .bps files exported with current `bp_export_strategies`

### The problem

The C function `bp_export_strategies` (in `src/mccfr_blueprint.c`, around line 2632)
calls `regret_match(is->regrets, strategy_buf, na)` to compute the per-bucket
strategy values it writes to the .bps. This produces the **per-iteration
regret-matched strategy** at the moment training was paused, NOT the
**time-averaged strategy** `strategy_sum / sum(strategy_sum)` which is the actual
CFR-converged Nash approximation.

There's a `bp_get_strategy` function in the same file (~line 2026) that uses
strategy_sum correctly (with regret_match fallback when strategy_sum is null).
The bulk export should use the same logic.

### Visible impact in the frontend

Even after Bug A is fixed, the frequencies in each cell are noisier than they
should be. Common nodes (UTG root with premium hands) are approximately right
because regret_match at 1B iterations is "mostly converged" for well-sampled info
sets. Deep nodes (3-bet, 4-bet) are noticeably noisier — see the screenshot of
CO facing UTG raise + MP 3-bet, where the range includes some marginal hands like
Q3s, J2s, K5s that shouldn't be in a tight 4-bet defending range.

### Why this is the solver-agent's job

Fixing this requires:
1. Patching the C file in the source tree the solver agent owns
2. Pushing to `s3://poker-blueprint-unified/code/`
3. Re-running the export step (which the solver-agent will do anyway when 4B
   training completes)

Frontend-agent could write a Python streamer that bypasses the C export entirely
(read regrets_1000M.bin directly, normalize strategy_sum, build the JSON). That
works as a backup if the solver-agent fix is delayed, but the C fix is the right
long-term answer because it's where every future export goes through.

### Workaround until fixed

Frontend continues to ship the regret_match data after the Bug A fix. The
displayed ranges are functionally correct (right hands in right cells) even if
frequencies have residual noise. After Bug B is fixed and a new export happens,
the frontend re-extracts (4 seconds via cache) and ships the cleaner data.
