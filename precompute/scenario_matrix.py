#!/usr/bin/env python3
"""Build the 27 preflop-filtered scenarios from ranges.json.

12 SRP scenarios + 15 3BP scenarios (no 4BP — ranges.json lacks vs_4bet call ranges).

Each scenario defines:
  - OOP/IP ranges (PioSOLVER strings)
  - Pot and stack sizes (in chips, 1BB = 100 chips)
  - Position labels and scenario type

Usage:
    from precompute.scenario_matrix import build_scenario_matrix
    scenarios = build_scenario_matrix("/path/to/ranges.json")
    # Returns dict: scenario_id -> scenario dict
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

from precompute.range_parser import get_range_hands, count_range_combos

# ── Constants ─────────────────────────────────────────────────────────────

SCALE = 100  # chips per BB

# SRP: open 2.5BB, call 2.5BB, blinds 1.5BB = 6.5BB pot, 97.5BB effective
SRP_POT = 650      # 6.5 BB in chips
SRP_STACK = 9750    # 97.5 BB in chips

# 3BP: open 2.5BB, 3bet ~9BB, call 9BB, blinds 1.5BB ≈ 20BB pot, 82BB effective
TBP_POT = 2000     # 20 BB in chips
TBP_STACK = 8200    # 82 BB in chips

# Postflop position order: leftmost is most OOP
POST_ORDER = ["SB", "BB", "UTG", "MP", "CO", "BTN"]


def _pos_index(pos):
    """Return postflop position index (lower = more OOP)."""
    return POST_ORDER.index(pos)


# ── Build scenario matrix ─────────────────────────────────────────────────

def build_scenario_matrix(ranges_path):
    """Build all 27 scenarios from ranges.json.

    Returns dict of scenario_id -> {
        oop_range: str,
        ip_range: str,
        starting_pot: int (chips),
        effective_stack: int (chips),
        oop_pos: str,
        ip_pos: str,
        opener: str,
        defender: str,
        scenario_type: "srp" | "3bp",
    }
    """
    with open(ranges_path) as f:
        ranges = json.load(f)

    scenarios = {}

    rfi_ranges = ranges.get("rfi", {})
    vs_rfi = ranges.get("vs_rfi", {})
    vs_3bet = ranges.get("vs_3bet", {})

    for opener in POST_ORDER:
        rfi = rfi_ranges.get(opener)
        if not rfi:
            continue

        for defender in POST_ORDER:
            if defender == opener:
                continue

            vs_key = f"{defender}_vs_{opener}"
            vs_entry = vs_rfi.get(vs_key)
            if not vs_entry:
                continue

            # Determine OOP/IP by postflop position
            o_idx = _pos_index(opener)
            d_idx = _pos_index(defender)

            # ── SRP: opener RFI vs defender call ──
            if vs_entry.get("call"):
                if o_idx < d_idx:
                    oop_range, ip_range = rfi, vs_entry["call"]
                    oop_pos, ip_pos = opener, defender
                else:
                    oop_range, ip_range = vs_entry["call"], rfi
                    oop_pos, ip_pos = defender, opener

                sid = f"{oop_pos}_vs_{ip_pos}_srp"
                if sid not in scenarios:
                    scenarios[sid] = {
                        "oop_range": oop_range,
                        "ip_range": ip_range,
                        "starting_pot": SRP_POT,
                        "effective_stack": SRP_STACK,
                        "oop_pos": oop_pos,
                        "ip_pos": ip_pos,
                        "opener": opener,
                        "defender": defender,
                        "scenario_type": "srp",
                    }

            # ── 3BP: opener RFI, defender 3bets, opener calls ──
            if vs_entry.get("3bet"):
                vs3 = vs_3bet.get(opener)
                if vs3 and vs3.get("call"):
                    if o_idx < d_idx:
                        oop_range = vs3["call"]
                        ip_range = vs_entry["3bet"]
                        oop_pos, ip_pos = opener, defender
                    else:
                        oop_range = vs_entry["3bet"]
                        ip_range = vs3["call"]
                        oop_pos, ip_pos = defender, opener

                    sid = f"{oop_pos}_vs_{ip_pos}_3bp"
                    if sid not in scenarios:
                        scenarios[sid] = {
                            "oop_range": oop_range,
                            "ip_range": ip_range,
                            "starting_pot": TBP_POT,
                            "effective_stack": TBP_STACK,
                            "oop_pos": oop_pos,
                            "ip_pos": ip_pos,
                            "opener": opener,
                            "defender": defender,
                            "scenario_type": "3bp",
                        }

    return scenarios


def scenario_summary(scenarios, board_ints=None):
    """Print summary table for all scenarios.

    If board_ints given, shows post-board-filter combo counts.
    """
    srp = {k: v for k, v in scenarios.items() if v["scenario_type"] == "srp"}
    tbp = {k: v for k, v in scenarios.items() if v["scenario_type"] == "3bp"}

    print(f"Total scenarios: {len(scenarios)} ({len(srp)} SRP + {len(tbp)} 3BP)\n")

    for label, group in [("SRP", srp), ("3BP", tbp)]:
        print(f"=== {label} ({len(group)}) ===")
        print(f"{'Scenario':<25} {'OOP pos':<8} {'IP pos':<8} "
              f"{'OOP combos':>11} {'IP combos':>11} {'Pot':>6} {'Stack':>7}")
        print("-" * 85)

        for sid in sorted(group):
            s = group[sid]
            if board_ints is not None:
                oop_n = len(get_range_hands(s["oop_range"], board_ints))
                ip_n = len(get_range_hands(s["ip_range"], board_ints))
            else:
                _, _, oop_n = count_range_combos(s["oop_range"])
                _, _, ip_n = count_range_combos(s["ip_range"])

            pot_bb = s["starting_pot"] / SCALE
            stack_bb = s["effective_stack"] / SCALE
            print(f"{sid:<25} {s['oop_pos']:<8} {s['ip_pos']:<8} "
                  f"{oop_n:>11.0f} {ip_n:>11.0f} {pot_bb:>5.1f}BB {stack_bb:>5.1f}BB")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Find ranges.json
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     '..', 'ACRPoker-Hud-PC', 'solver', 'ranges.json'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'data', 'ranges.json'),
    ]
    ranges_path = None
    for c in candidates:
        if os.path.exists(c):
            ranges_path = c
            break

    if len(sys.argv) > 1:
        ranges_path = sys.argv[1]

    if not ranges_path or not os.path.exists(ranges_path):
        print("ERROR: ranges.json not found. Pass path as argument.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ranges from: {ranges_path}\n")
    scenarios = build_scenario_matrix(ranges_path)

    # No board filter — show raw combo counts
    scenario_summary(scenarios)

    # With a sample board (Ts 7h 2d)
    from solver import card_to_int
    board = [card_to_int("Ts"), card_to_int("7h"), card_to_int("2d")]
    print("\n=== With board Ts 7h 2d (filtered) ===\n")
    scenario_summary(scenarios, board_ints=board)

    # Verify expected counts
    srp_count = sum(1 for s in scenarios.values() if s["scenario_type"] == "srp")
    tbp_count = sum(1 for s in scenarios.values() if s["scenario_type"] == "3bp")
    print(f"\nExpected: 12 SRP + 15 3BP = 27 total")
    print(f"Got:      {srp_count} SRP + {tbp_count} 3BP = {len(scenarios)} total")

    if len(scenarios) != 27:
        print("\nWARNING: Scenario count mismatch!")
        print("Scenarios found:")
        for sid in sorted(scenarios):
            s = scenarios[sid]
            print(f"  {sid}: opener={s['opener']}, defender={s['defender']}")
