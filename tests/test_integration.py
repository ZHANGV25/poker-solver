"""Integration test: full pipeline with real blueprint data.

Tests the complete flow from blueprint lookup through range narrowing
to runtime river solving, using actual flop solution files from the
ACR HUD project.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from solver import card_to_int, int_to_card, parse_range_string, SCALE
from range_narrowing import RangeNarrower
from blueprint_io import Blueprint, texture_key
from solver_pool import SolverPool

# Path to actual flop solutions
ACR_DIR = os.environ.get("ACR_PROJECT_DIR",
    r"C:\Users\Victor\Documents\Projects\ACRPoker-Hud-PC")
SOLUTIONS_DIR = os.path.join(ACR_DIR, "solver", "flop_solutions")
RANGES_PATH = os.path.join(ACR_DIR, "solver", "ranges.json")


def test_blueprint_loading():
    """Test loading blueprint data from real flop solutions."""
    print("Test 1: Blueprint Loading")

    if not os.path.isdir(SOLUTIONS_DIR):
        print("  SKIP: flop_solutions not found at", SOLUTIONS_DIR)
        return False

    bp = Blueprint(SOLUTIONS_DIR)
    print(f"  Found {len(bp.available_scenarios)} scenarios")
    for name, count in sorted(bp.available_scenarios.items())[:5]:
        print(f"    {name}: {count} textures")

    # Test texture key generation
    tk, suit_map = texture_key(["Qs", "As", "2d"])
    print(f"\n  Board Qs As 2d -> texture: {tk}")
    print(f"  Suit map: {suit_map}")

    # Test action probabilities
    probs = bp.get_action_probs("CO_vs_BB_srp", ["Qs", "As", "2d"],
                                 "oop", "Check")
    if probs:
        print(f"  P(Check|hand) for OOP: {len(probs)} hands loaded")
        # Show a few
        for (c0, c1), p in list(probs.items())[:3]:
            print(f"    {int_to_card(c0)}{int_to_card(c1)}: P(Check)={p:.2f}")
    else:
        print("  WARNING: no action probs returned (scenario may lack per-hand data)")

    print("  PASS")
    return True


def test_range_narrowing_with_blueprint():
    """Test Bayesian narrowing using real blueprint probabilities."""
    print("\nTest 2: Range Narrowing with Blueprint")

    if not os.path.isdir(SOLUTIONS_DIR):
        print("  SKIP: flop_solutions not found")
        return False

    bp = Blueprint(SOLUTIONS_DIR)

    # Scenario: CO opens, BB calls (CO_vs_BB_srp)
    # After flop Qs As 2d, villain (CO) bets
    import json
    if os.path.exists(RANGES_PATH):
        with open(RANGES_PATH) as f:
            ranges = json.load(f)
        co_range_str = ranges.get("rfi", {}).get("CO", "")
        bb_range_str = ranges.get("vs_rfi", {}).get("BB_vs_CO", {}).get("call", "")
    else:
        co_range_str = "AA,KK,QQ,JJ,TT,99,AKs,AQs,AJs,ATs,AKo,AQo,KQs"
        bb_range_str = "JJ,TT,99,88,77,66,AQs,AJs,ATs,KQs,KJs,QJs,JTs,T9s,98s,87s,76s,AQo,AJo,KQo"

    co_hands = parse_range_string(co_range_str)
    bb_hands = parse_range_string(bb_range_str)
    print(f"  CO range: {len(co_hands)} combos")
    print(f"  BB range: {len(bb_hands)} combos")

    narrower = RangeNarrower()
    narrower.set_initial_range("villain", co_hands)  # CO is villain
    narrower.set_initial_range("hero", bb_hands)      # BB is hero

    # Villain bets on flop Qs As 2d
    bet_probs = bp.get_action_probs("CO_vs_BB_srp", ["Qs", "As", "2d"],
                                     "oop", "Bet")
    if bet_probs:
        narrower.update("villain", "bet", bet_probs)
        narrowed = narrower.get_weighted_hands("villain")
        print(f"  After villain bets: {narrower.get_hand_count('villain')} effective hands")
        # Show top 5 hands
        print("  Top villain hands (most likely to bet):")
        for c0, c1, w in narrowed[:5]:
            print(f"    {int_to_card(c0)}{int_to_card(c1)}: weight={w:.3f}")
        print("  Bottom villain hands (least likely to bet):")
        for c0, c1, w in narrowed[-3:]:
            print(f"    {int_to_card(c0)}{int_to_card(c1)}: weight={w:.3f}")
    else:
        print("  WARNING: no bet probs available, skipping narrowing test")

    print("  PASS")
    return True


def test_runtime_river_solve():
    """Test runtime river solving with narrowed ranges."""
    print("\nTest 3: Runtime River Solve")

    # Build narrowed ranges (simulated)
    board = ["Qs", "As", "2d", "7h", "4c"]
    board_ints = [card_to_int(c) for c in board]

    # Narrowed OOP range (BB after defending and calling flop)
    oop_range_str = "JJ,TT,99,AQs,AJs,KQs,QJs,JTs,T9s,98s"
    ip_range_str = "AA,KK,QQ,AKs,AQs,AKo,AQo,KQs"

    oop_hands = parse_range_string(oop_range_str)
    ip_hands = parse_range_string(ip_range_str)

    # Filter board blockers
    blocked = set(board_ints)
    oop_hands = [(c0, c1, w) for c0, c1, w in oop_hands
                 if c0 not in blocked and c1 not in blocked]
    ip_hands = [(c0, c1, w) for c0, c1, w in ip_hands
                if c0 not in blocked and c1 not in blocked]

    print(f"  Narrowed ranges: OOP={len(oop_hands)} IP={len(ip_hands)}")

    # Solve
    try:
        pool = SolverPool(max_workers=1)
    except FileNotFoundError:
        print("  SKIP: solver DLL not found")
        return False

    pot = int(10 * SCALE)
    stack = int(50 * SCALE)

    t0 = time.time()
    req_id = pool.submit(
        board=board_ints,
        oop_hands=oop_hands,
        ip_hands=ip_hands,
        pot=pot, stack=stack,
        bet_sizes=[0.33, 0.75],
        iterations=500,
    )

    result = pool.wait(req_id, timeout=30)
    elapsed = (time.time() - t0) * 1000

    if result is None:
        print("  FAIL: solve timed out")
        pool.shutdown()
        return False

    print(f"  Solved in {result['time_ms']:.0f}ms")
    print(f"  Exploitability: {result['exploit_pct']:.4f}% of pot")
    print(f"  Hands: OOP={result['num_hands'][0]} IP={result['num_hands'][1]}")

    # Show strategies
    oop_strats = result['strategies'].get(0, {})
    print(f"\n  OOP strategies ({len(oop_strats)} hands):")
    for hand, actions in list(oop_strats.items())[:5]:
        act_str = " ".join(f"{k}={v*100:.0f}%" for k, v in actions.items())
        print(f"    {hand}: {act_str}")

    pool.shutdown()
    print("  PASS")
    return True


def test_concurrent_solves():
    """Test solving multiple tables concurrently."""
    print("\nTest 4: Concurrent Solves (3 tables)")

    try:
        pool = SolverPool(max_workers=3)
    except FileNotFoundError:
        print("  SKIP: solver DLL not found")
        return False

    boards = [
        ["Qs", "As", "2d", "7h", "4c"],
        ["Kh", "Td", "5s", "3c", "8h"],
        ["Jc", "9d", "6h", "2s", "Ah"],
    ]

    # Simple ranges for speed
    range_str = "AA,KK,QQ,JJ,TT,99,AKs,AQs,AJs,KQs"
    hands = parse_range_string(range_str)

    request_ids = []
    t0 = time.time()

    for board in boards:
        board_ints = [card_to_int(c) for c in board]
        blocked = set(board_ints)
        filtered = [(c0, c1, w) for c0, c1, w in hands
                     if c0 not in blocked and c1 not in blocked]

        req_id = pool.submit(
            board=board_ints,
            oop_hands=filtered,
            ip_hands=filtered,
            pot=1000, stack=5000,
            iterations=200,
        )
        request_ids.append(req_id)

    # Wait for all
    results = []
    for req_id in request_ids:
        result = pool.wait(req_id, timeout=30)
        results.append(result)

    elapsed = (time.time() - t0) * 1000

    all_ok = True
    for i, result in enumerate(results):
        if result is None:
            print(f"  Table {i+1}: TIMEOUT")
            all_ok = False
        elif 'error' in result:
            print(f"  Table {i+1}: ERROR: {result['error']}")
            all_ok = False
        else:
            print(f"  Table {i+1}: {result['time_ms']:.0f}ms, "
                  f"exploit={result['exploit_pct']:.3f}%")

    print(f"  Total wall time: {elapsed:.0f}ms (concurrent)")
    pool.shutdown()

    if all_ok:
        print("  PASS")
    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("  Poker Solver - Integration Tests")
    print("=" * 60)

    results = []
    results.append(("Blueprint Loading", test_blueprint_loading()))
    results.append(("Range Narrowing + Blueprint", test_range_narrowing_with_blueprint()))
    results.append(("Runtime River Solve", test_runtime_river_solve()))
    results.append(("Concurrent Solves", test_concurrent_solves()))

    print("\n" + "=" * 60)
    all_pass = all(r for _, r in results)
    for name, passed in results:
        status = "PASS" if passed else "FAIL/SKIP"
        print(f"  {name}: {status}")
    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
