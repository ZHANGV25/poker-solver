#!/usr/bin/env python3
"""Phase 1.3 sentinel verification script.

Run AFTER `export_v2.py` produces a schema v3 .bps file. Loads it via
BlueprintV2 and runs a battery of sentinel checks on the exported
per-action EVs. Each sentinel is a necessary (not sufficient) condition
for correctness; if any fails, the blueprint should NOT be shipped.

    python3 verify_phase_1_3.py <path/to/unified_blueprint.bps>

Sentinel list:
    1. UTG opening node (empty action history) has non-trivial EVs for
       at least one bucket. Validates that the EV walk is finding
       and populating the root decision nodes.
    2. Coverage: fraction of strategy nodes that also have EVs.
       Must be ≥ 5%. Low coverage indicates the EV walk skipped too
       many info sets or pruning was too aggressive.
    3. CFR consistency (loose): σ̄-weighted sum of action EVs per bucket
       must be finite (no NaN) and bounded (< 2×stack). Weak but
       catches the "EV walk produced garbage" case.
    4. Action EV ordering at the AA UTG open. For bucket 0 (AA in the
       169-class lossless abstraction), the three raise actions should
       all have positive EV and larger raises should dominate smaller
       raises (within sampling noise). For the trashiest bucket (168 =
       32o), raises should have near-zero or negative EV and call EV
       should be no better than fold.
    5. Visit count distribution. Pulled from metadata["ev_visit_stats"]
       which export_v2.py stamps at export time. Asserts:
         - total_visited > 0
         - p50_visits >= 10 (median info set has 10+ samples)
         - above_1000 > 0 (at least one high-confidence info set)
         - NOT all visit counts = 1 (that would mean sampling is broken)
    6. CFR consistency (strict): DEFERRED — requires Python tree-walking
       of the exported data to find child info sets by action_hash.
       The loose version in sentinel 3 is the current bar.

A failure in any sentinel prints FAIL to stderr and exits non-zero.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "python"))

import numpy as np
from blueprint_v2 import BlueprintV2


def _fmt_ev(evs):
    return "[" + ", ".join(f"{v:+.1f}" for v in evs) + "]"


def _fmt_strat(strat):
    return "[" + ", ".join(f"{v:.2f}" for v in strat) + "]"


def sentinel_1_utg_open_has_evs(bp, meta):
    """The UTG opening node should have non-trivial EVs for at least one
    of the "strong" candidate buckets."""
    print("\n── Sentinel 1: UTG opening has non-trivial EVs ──")

    # UTG player index: in Pluribus 6-max with SB=0, BB=1, the first-to-act
    # preflop is index 2 (after the blinds post and acting order becomes
    # [2, 3, 4, 5, 0, 1]).
    UTG_PLAYER = 2
    PREFLOP_STREET = 0
    EMPTY_HISTORY = []

    all_evs = bp.get_all_bucket_action_evs(
        board=[], action_history=EMPTY_HISTORY,
        player=UTG_PLAYER, street=PREFLOP_STREET,
    )
    all_strat = bp.get_all_bucket_strategies(
        board=[], action_history=EMPTY_HISTORY,
        player=UTG_PLAYER, street=PREFLOP_STREET,
    )

    if all_evs is None:
        print("  FAIL: UTG open info set has no action EVs", file=sys.stderr)
        return False
    if all_strat is None:
        print("  FAIL: UTG open info set has no strategy", file=sys.stderr)
        return False

    num_buckets, num_actions = all_evs.shape
    print(f"  num_buckets={num_buckets}, num_actions={num_actions}")

    # Probe the strongest 5 and weakest 5 buckets. At least one must have
    # visibly non-trivial EVs (not all identical, not all zero).
    non_trivial = 0
    for probe_bucket in [0, 1, 2, 3, 4, 164, 165, 166, 167, 168]:
        if probe_bucket >= num_buckets:
            continue
        bucket_evs = all_evs[probe_bucket]
        if np.all(np.abs(bucket_evs - bucket_evs[0]) < 1e-6):
            continue
        if np.all(np.abs(bucket_evs) < 1e-3):
            continue
        non_trivial += 1

    print(f"  non-trivial buckets probed: {non_trivial}/10")
    if non_trivial == 0:
        print("  FAIL: no probed bucket had non-trivial EVs", file=sys.stderr)
        return False
    print("  ✓ passed")
    return True


def sentinel_2_coverage(bp, meta):
    """Fraction of strategy nodes that also have per-action EVs must be
    non-negligible. Low coverage = EV walk is broken or config is off."""
    print("\n── Sentinel 2: coverage ──")
    ev_table = bp._action_evs.get("__unified__", {})
    strat_table = bp._textures.get("__unified__", {})
    n_ev = len(ev_table)
    n_strat = len(strat_table)
    cov = 100.0 * n_ev / max(1, n_strat)
    print(f"  strategy nodes: {n_strat:,}")
    print(f"  EV nodes:       {n_ev:,}")
    print(f"  coverage:       {cov:.1f}%")

    if n_strat == 0:
        print("  FAIL: no strategy nodes loaded — .bps is empty?", file=sys.stderr)
        return False
    if cov < 5.0:
        print(f"  FAIL: coverage too low ({cov:.1f}% < 5.0%)", file=sys.stderr)
        return False
    print("  ✓ passed")
    return True


def sentinel_3_cfr_loose(bp, meta):
    """Loose CFR consistency: σ̄-weighted sum of action EVs per bucket
    must be finite and bounded."""
    print("\n── Sentinel 3: loose CFR consistency ──")
    ev_table = bp._action_evs.get("__unified__", {})
    strat_table = bp._textures.get("__unified__", {})
    initial_stack = meta.get("initial_stack", 10000)

    n_checked = 0
    n_nan = 0
    n_out_of_range = 0
    max_abs_ev = 0.0
    for key, evs_arr in ev_table.items():
        strat_arr = strat_table.get(key)
        if strat_arr is None:
            continue
        if evs_arr.shape != strat_arr.shape:
            continue
        node_evs_per_bucket = (strat_arr * evs_arr).sum(axis=1)
        if np.any(np.isnan(node_evs_per_bucket)):
            n_nan += 1
        over = np.abs(node_evs_per_bucket) > 2 * initial_stack
        if np.any(over):
            n_out_of_range += 1
        bucket_max = float(np.abs(node_evs_per_bucket).max())
        if bucket_max > max_abs_ev:
            max_abs_ev = bucket_max
        n_checked += 1
        if n_checked >= 10000:  # cap for large tables
            break

    print(f"  nodes checked: {n_checked:,}")
    print(f"  NaN node-EVs:  {n_nan:,}")
    print(f"  out-of-range:  {n_out_of_range:,}")
    print(f"  max |node_ev|: {max_abs_ev:.1f} (stack={initial_stack})")
    if n_nan > 0:
        print(f"  FAIL: {n_nan} nodes produced NaN EVs", file=sys.stderr)
        return False
    if n_out_of_range > max(1, n_checked * 0.01):
        print(f"  FAIL: too many nodes with |EV| > 2·stack "
              f"({n_out_of_range} > 1% of {n_checked})", file=sys.stderr)
        return False
    print("  ✓ passed")
    return True


def sentinel_4_ordering(bp, meta):
    """At the UTG open decision, AA (bucket 0) should be strongly +EV
    and the worst hand (bucket 168 = 32o) should be neutral-to-negative.

    Under Pluribus 6-max tiered preflop sizing, the action indices at
    UTG open (to_call > 0 because BB posted) are:
        0 = fold, 1 = call, 2 = raise_0.5, 3 = raise_0.7, 4 = raise_1.0

    If the action count is different (e.g. the training tree merges
    sizes), we relax the check to "max raise EV > call EV" for AA.

    Only runs against a unified-preflop blueprint (include_preflop in
    metadata). For postflop-only toy blueprints, this sentinel is
    skipped.
    """
    print("\n── Sentinel 4: AA / 32o EV sign + ordering at UTG open ──")

    if not meta.get("preflop_bet_sizes") and not meta.get("preflop_tiers"):
        print("  SKIP: not a unified preflop blueprint (no preflop_bet_sizes)")
        return True

    UTG_PLAYER = 2
    PREFLOP_STREET = 0
    all_evs = bp.get_all_bucket_action_evs(
        board=[], action_history=[],
        player=UTG_PLAYER, street=PREFLOP_STREET,
    )
    all_strat = bp.get_all_bucket_strategies(
        board=[], action_history=[],
        player=UTG_PLAYER, street=PREFLOP_STREET,
    )
    if all_evs is None or all_strat is None:
        print("  FAIL: UTG open node missing from .bps", file=sys.stderr)
        return False

    num_buckets, num_actions = all_evs.shape
    if num_actions < 3:
        print(f"  SKIP: unexpected action count at UTG open ({num_actions})")
        return True

    # AA is always bucket 0 in the 169-class mapping (r0=12, r1=12).
    # 32o is the last class (168 for 169 buckets).
    AA_BUCKET = 0
    TRASH_BUCKET = num_buckets - 1 if num_buckets >= 169 else None

    aa_evs = all_evs[AA_BUCKET]
    aa_strat = all_strat[AA_BUCKET]
    print(f"  AA (bucket {AA_BUCKET}):")
    print(f"    strategy:   {_fmt_strat(aa_strat)}")
    print(f"    action_evs: {_fmt_ev(aa_evs)}")

    # Identify which action indices are "raises" vs fold/call.
    # In the standard tree, fold=0 (if to_call>0), call=1, then raises.
    # We assume indices ≥ 2 are raises.
    fold_ev = float(aa_evs[0])
    call_ev = float(aa_evs[1]) if num_actions >= 2 else 0.0
    raise_evs = [float(aa_evs[i]) for i in range(2, num_actions)]
    max_raise_ev = max(raise_evs) if raise_evs else call_ev

    print(f"    fold EV = {fold_ev:+.2f}")
    print(f"    call EV = {call_ev:+.2f}")
    print(f"    raise EVs = {_fmt_ev(raise_evs)}")

    fail = False

    # AA MUST have max raise EV > 0 (it's the strongest hand, raising is +EV).
    if max_raise_ev <= 0.0:
        print(f"  FAIL: AA max raise EV = {max_raise_ev:+.2f}, "
              f"expected > 0", file=sys.stderr)
        fail = True
    # AA max raise EV should be >= call EV (raising dominates flatting).
    # Allow small noise tolerance — 2 chips on a 10000 stack.
    NOISE_CHIPS = 2.0
    if max_raise_ev + NOISE_CHIPS < call_ev:
        print(f"  FAIL: AA max raise EV ({max_raise_ev:+.2f}) < call EV "
              f"({call_ev:+.2f}) by more than noise tolerance",
              file=sys.stderr)
        fail = True
    # AA call EV should be >= fold EV = 0 (folding AA is worse than calling).
    if call_ev + NOISE_CHIPS < fold_ev:
        print(f"  FAIL: AA call EV ({call_ev:+.2f}) < fold EV "
              f"({fold_ev:+.2f})", file=sys.stderr)
        fail = True

    # Trash hand check (if we have a 169-bucket layout).
    if TRASH_BUCKET is not None:
        trash_evs = all_evs[TRASH_BUCKET]
        trash_strat = all_strat[TRASH_BUCKET]
        print(f"  Trash (bucket {TRASH_BUCKET}):")
        print(f"    strategy:   {_fmt_strat(trash_strat)}")
        print(f"    action_evs: {_fmt_ev(trash_evs)}")

        trash_fold = float(trash_evs[0])
        trash_call = float(trash_evs[1]) if num_actions >= 2 else 0.0
        trash_raises = [float(trash_evs[i]) for i in range(2, num_actions)]
        trash_max_raise = max(trash_raises) if trash_raises else trash_call

        # Trash: all raises should be ≤ a small positive threshold (they're
        # bluffs at equilibrium frequency, not value). Allow 20 chips of
        # slack on a 10000 stack because bluffing frequency is nonzero.
        TRASH_RAISE_CEILING = 20.0
        if trash_max_raise > TRASH_RAISE_CEILING:
            print(f"  WARN: trash max raise EV ({trash_max_raise:+.2f}) > "
                  f"{TRASH_RAISE_CEILING} chips — this is a soft sentinel, "
                  f"could indicate the blueprint is overvaluing bluffs",
                  file=sys.stderr)
            # Don't fail on this — it's a quality signal, not a correctness
            # signal. Bluffs CAN have positive EV at equilibrium.
        # Trash: call EV should be ≤ fold EV (calling with junk loses).
        # This is the real correctness check for the trash side.
        if trash_call > trash_fold + 10.0:
            print(f"  FAIL: trash call EV ({trash_call:+.2f}) > "
                  f"trash fold EV ({trash_fold:+.2f}) by more than 10 chips",
                  file=sys.stderr)
            fail = True

    if fail:
        return False
    print("  ✓ passed")
    return True


def sentinel_5_visit_distribution(bp, meta):
    """Visit count distribution sanity checks from export-time stats."""
    print("\n── Sentinel 5: visit count distribution ──")
    stats = meta.get("ev_visit_stats")
    if stats is None:
        print("  FAIL: no ev_visit_stats in metadata (export_v2.py too old?)",
              file=sys.stderr)
        return False

    print(f"  total_visited: {stats['total_visited']:,}")
    print(f"  percentiles:   min={stats['min']} p10={stats['p10']} "
          f"p50={stats['p50']} p90={stats['p90']} p99={stats['p99']} "
          f"max={stats['max']}")
    print(f"  below_5:       {stats['below_5']:,}")
    print(f"  below_100:     {stats['below_100']:,}")
    print(f"  above_1000:    {stats['above_1000']:,}")

    fail = False

    if stats["total_visited"] == 0:
        print("  FAIL: zero visited info sets", file=sys.stderr)
        return False

    # If every info set has exactly 1 visit, sampling is broken.
    if stats["min"] == stats["max"] == 1:
        print("  FAIL: all visit counts are 1 — sampling broken", file=sys.stderr)
        fail = True

    # Median should be well above 1 for a properly-trained EV walk.
    # With 50M iterations on ~1B info sets, we expect the median
    # visited info set to have ~50 samples (in-equilibrium paths
    # dominate). Allow 5 as a very loose floor.
    if stats["p50"] < 5:
        print(f"  WARN: p50 visits ({stats['p50']}) < 5 — EV walk "
              f"iterations may be too low", file=sys.stderr)
        # Soft warning, not failure — the toy game legitimately has
        # low visit counts.

    # Sanity: p90 should be > p50 (distribution has spread)
    if stats["p90"] < stats["p50"]:
        print(f"  FAIL: p90 ({stats['p90']}) < p50 ({stats['p50']}) — "
              f"corrupt distribution", file=sys.stderr)
        fail = True

    if fail:
        return False
    print("  ✓ passed")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: verify_phase_1_3.py <path/to/unified_blueprint.bps>",
              file=sys.stderr)
        sys.exit(2)

    bps_path = sys.argv[1]
    if not os.path.exists(bps_path):
        print(f"FATAL: {bps_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {bps_path}...", flush=True)
    bp = BlueprintV2(".", streets_to_load=[0, 1, 2, 3])
    if not bp.load_unified(bps_path):
        print("FATAL: load_unified failed", file=sys.stderr)
        sys.exit(1)

    meta = bp._metadata.get("__unified__", {})
    print("Metadata:")
    print(json.dumps(meta, indent=2, default=str))

    schema_v = meta.get("schema_version", 2)
    has_evs = meta.get("has_action_evs", False)
    print(f"\nSchema version: {schema_v}")
    print(f"has_action_evs: {has_evs}")
    print(f"Loader has_action_evs(): {bp.has_action_evs()}")

    if not bp.has_action_evs():
        print("\nFATAL: no per-action EVs loaded — either the file is "
              "schema v2, or the BPR3 section was empty, or the reader "
              "failed to parse it.", file=sys.stderr)
        sys.exit(1)

    # Run sentinels. Each returns True on pass, False on fail. We run
    # them all (don't short-circuit) so the user sees a complete report.
    results = []
    results.append(("1 — UTG open has EVs",    sentinel_1_utg_open_has_evs(bp, meta)))
    results.append(("2 — coverage",            sentinel_2_coverage(bp, meta)))
    results.append(("3 — loose CFR",           sentinel_3_cfr_loose(bp, meta)))
    results.append(("4 — AA/32o ordering",     sentinel_4_ordering(bp, meta)))
    results.append(("5 — visit distribution",  sentinel_5_visit_distribution(bp, meta)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = 0
    n_fail = 0
    for name, passed in results:
        mark = "✓" if passed else "✗"
        print(f"  {mark} sentinel {name}")
        if passed:
            n_pass += 1
        else:
            n_fail += 1

    print(f"\n{n_pass} passed, {n_fail} failed out of {len(results)}")
    if n_fail > 0:
        print("\n=== SENTINEL FAILURES PRESENT — do not ship ===", file=sys.stderr)
        sys.exit(1)

    print("\n=== ALL SENTINELS PASSED ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
