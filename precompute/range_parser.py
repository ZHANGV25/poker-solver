#!/usr/bin/env python3
"""Parse PioSOLVER range strings, handle frequencies, filter by board.

Wraps parse_range_string() from python/solver.py and adds:
  - Board-card removal (dead card elimination)
  - Frequency-based deterministic combo selection
  - Combo counting / summary utilities

Usage:
    from precompute.range_parser import get_range_hands

    hands = get_range_hands("AA,AKs,AQo:0.5", board_ints=[48, 36, 20])
    # Returns list of (card0, card1) tuples — board-blocked combos removed,
    # fractional combos resolved deterministically.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

from solver import parse_range_string, card_to_int, int_to_card

# ── Core API ──────────────────────────────────────────────────────────────


def get_range_hands(range_str, board_ints):
    """Parse range string and return board-filtered hands with frequency handling.

    Args:
        range_str: PioSOLVER-format range string (e.g. "AA,AKs,AQo:0.5")
        board_ints: list of int card indices on the board (0-51)

    Returns:
        list of (card0, card1) tuples, sorted. Fractional combos are resolved
        deterministically: for weight W and N total combos of that hand class,
        the first round(N * W) combos (sorted by card index) are included.
    """
    raw = parse_range_string(range_str)
    if not raw:
        return []

    board_set = set(board_ints)

    # Group combos by their hand class to handle fractional weights correctly.
    # A hand class is (rank0, rank1, suited/offsuit/pair) — all combos with
    # the same weight from a single range entry belong together.
    #
    # parse_range_string already expands groups into individual combos with
    # the same weight, so we can group by weight to reconstruct classes.
    # However, two different hand groups could share the same weight, so we
    # group by (rank_pair, type, weight) to be safe.

    # First pass: remove board-blocked combos
    unblocked = [(c0, c1, w) for c0, c1, w in raw
                 if c0 not in board_set and c1 not in board_set]

    # Separate full-weight and fractional combos
    result = []
    fractional = {}  # (r0, r1, type_key) -> [(c0, c1, weight)]

    for c0, c1, w in unblocked:
        if w >= 1.0:
            result.append((c0, c1))
        else:
            # Classify combo for deterministic fractional selection
            r0, r1 = c0 >> 2, c1 >> 2
            s0, s1 = c0 & 3, c1 & 3
            if r0 == r1:
                type_key = 'pair'
            elif s0 == s1:
                type_key = 'suited'
            else:
                type_key = 'offsuit'
            key = (r0, r1, type_key, w)
            fractional.setdefault(key, []).append((c0, c1))

    # Resolve fractional combos: include first round(N * W) combos
    for key, combos in fractional.items():
        w = key[3]
        combos.sort()  # deterministic ordering by card index
        n_include = round(len(combos) * w)
        result.extend(combos[:n_include])

    result.sort()
    return result


def count_range_combos(range_str):
    """Count total combos in a range string (no board filtering).

    Returns (full_combos, fractional_combos, effective_total).
    """
    raw = parse_range_string(range_str)
    full = sum(1 for _, _, w in raw if w >= 1.0)
    frac = [(c0, c1, w) for c0, c1, w in raw if w < 1.0]
    effective_frac = sum(w for _, _, w in frac)
    return full, len(frac), full + effective_frac


def range_summary(range_str, board_ints=None):
    """Return a summary dict for a range string.

    Args:
        range_str: PioSOLVER-format range string
        board_ints: optional board cards to filter

    Returns:
        dict with keys: total_raw, total_after_filter, hands (list of card strings)
    """
    raw = parse_range_string(range_str)
    total_raw = len(raw)

    if board_ints is not None:
        hands = get_range_hands(range_str, board_ints)
    else:
        hands = [(c0, c1) for c0, c1, w in raw if w >= 1.0]

    return {
        "total_raw": total_raw,
        "total_after_filter": len(hands),
        "hands": [(int_to_card(c0), int_to_card(c1)) for c0, c1 in hands],
    }


# ── CLI test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Quick self-test
    print("=== Range Parser Self-Test ===\n")

    # Test 1: Basic parsing
    hands = get_range_hands("AA,KK,AKs", board_ints=[])
    print(f"AA,KK,AKs -> {len(hands)} combos (expect 16: 6+6+4)")
    assert len(hands) == 16, f"Expected 16, got {len(hands)}"

    # Test 2: Board filtering
    # Board: Ah Kd 7s = [50, 43, 23]
    Ah, Kd, s7 = card_to_int("Ah"), card_to_int("Kd"), card_to_int("7s")
    board = [Ah, Kd, s7]
    hands = get_range_hands("AA", board_ints=board)
    print(f"AA on AhKd7s -> {len(hands)} combos (expect 3: 6 - 3 using Ah)")
    assert len(hands) == 3, f"Expected 3, got {len(hands)}"

    # Test 3: Frequency handling
    # AQo has 12 combos, at 0.5 -> should include 6
    hands = get_range_hands("AQo:0.5", board_ints=[])
    print(f"AQo:0.5 -> {len(hands)} combos (expect 6: 12 * 0.5)")
    assert len(hands) == 6, f"Expected 6, got {len(hands)}"

    # Test 4: Mixed with board
    hands = get_range_hands("AA,AKs,AQo:0.5", board_ints=board)
    print(f"AA,AKs,AQo:0.5 on AhKd7s -> {len(hands)} combos")

    # Test 5: Full range from ranges.json
    ranges_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               '..', 'ACRPoker-Hud-PC', 'solver', 'ranges.json')
    if os.path.exists(ranges_path):
        with open(ranges_path) as f:
            ranges = json.load(f)

        rfi_utg = ranges["rfi"]["UTG"]
        hands_unfiltered = get_range_hands(rfi_utg, board_ints=[])
        hands_filtered = get_range_hands(rfi_utg, board_ints=board)
        print(f"\nUTG RFI: {len(hands_unfiltered)} combos unfiltered, "
              f"{len(hands_filtered)} after AhKd7s removal")

        bb_call = ranges["vs_rfi"]["BB_vs_UTG"]["call"]
        hands_bb = get_range_hands(bb_call, board_ints=board)
        print(f"BB vs UTG call: {len(hands_bb)} combos after AhKd7s removal")

        # Count combos for all positions
        print("\n=== Combo counts (no board filter) ===")
        for pos, rfi_str in ranges["rfi"].items():
            full, frac, eff = count_range_combos(rfi_str)
            print(f"  {pos} RFI: {full} full + {frac} fractional = {eff:.0f} effective")

    print("\nAll tests passed.")
