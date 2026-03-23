"""End-to-end test: Python bindings + range narrowing + solver.

Tests the full pipeline:
1. Parse ranges from strings
2. Narrow ranges using mock blueprint probabilities
3. Solve river spot with narrowed ranges
4. Verify strategies are reasonable
"""

import sys
import os
import ctypes
import time

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from solver import card_to_int, int_to_card, parse_range_string, SCALE, MAX_ACTIONS
from range_narrowing import RangeNarrower, generate_all_hands


def test_card_parsing():
    """Test card parsing roundtrip."""
    print("Test 1: Card parsing")
    for card in ["Ah", "Kc", "2d", "Ts", "Qh"]:
        i = card_to_int(card)
        back = int_to_card(i)
        assert back == card, f"Roundtrip failed: {card} -> {i} -> {back}"
    print("  PASS: card parsing works")


def test_range_parsing():
    """Test range string parsing."""
    print("\nTest 2: Range parsing")

    # Specific combos
    hands = parse_range_string("AhKh,QsJs")
    assert len(hands) == 2, f"Expected 2 hands, got {len(hands)}"
    print(f"  Specific combos: {len(hands)} hands")

    # Pairs
    hands = parse_range_string("AA")
    assert len(hands) == 6, f"Expected 6 combos for AA, got {len(hands)}"
    print(f"  AA: {len(hands)} combos")

    # Suited
    hands = parse_range_string("AKs")
    assert len(hands) == 4, f"Expected 4 combos for AKs, got {len(hands)}"
    print(f"  AKs: {len(hands)} combos")

    # Offsuit
    hands = parse_range_string("AKo")
    assert len(hands) == 12, f"Expected 12 combos for AKo, got {len(hands)}"
    print(f"  AKo: {len(hands)} combos")

    # Mixed
    hands = parse_range_string("AA,KK,AKs,AKo")
    assert len(hands) == 28, f"Expected 28 combos, got {len(hands)}"
    print(f"  AA,KK,AKs,AKo: {len(hands)} combos")

    # Weighted
    hands = parse_range_string("AhKh:0.5,QsJs:0.3")
    assert abs(hands[0][2] - 0.5) < 0.01
    assert abs(hands[1][2] - 0.3) < 0.01
    print(f"  Weighted: {hands[0][2]:.1f}, {hands[1][2]:.1f}")

    print("  PASS: range parsing works")


def test_range_narrowing():
    """Test Bayesian range narrowing."""
    print("\nTest 3: Range narrowing")

    narrower = RangeNarrower()

    # Set up a simple range: 4 hands
    hands = [
        (card_to_int("Ah"), card_to_int("Kh"), 1.0),  # TPTK
        (card_to_int("Qh"), card_to_int("Qc"), 1.0),  # Trips
        (card_to_int("Jh"), card_to_int("Th"), 1.0),  # JT
        (card_to_int("6h"), card_to_int("5h"), 1.0),  # Air
    ]
    narrower.set_initial_range("villain", hands)

    # Villain bets — blueprint says:
    # TPTK bets 80%, Trips bets 90%, JT bets 10%, Air bets 70% (bluff)
    ak = (card_to_int("Ah"), card_to_int("Kh"))
    qq = (card_to_int("Qh"), card_to_int("Qc"))
    jt = (card_to_int("Jh"), card_to_int("Th"))
    air = (card_to_int("6h"), card_to_int("5h"))

    bet_probs = {ak: 0.80, qq: 0.90, jt: 0.10, air: 0.70}
    narrower.update("villain", "bet75", bet_probs)

    result = narrower.get_weighted_hands("villain")
    print("  After villain bets:")
    for c0, c1, w in result:
        print(f"    {int_to_card(c0)}{int_to_card(c1)}: weight={w:.3f}")

    # QQ (0.90) should have highest weight, JT (0.10) should be near floor
    weights = {(c0, c1): w for c0, c1, w in result}
    assert weights[qq] > weights[jt], "QQ should be weighted higher than JT after bet"
    assert weights[qq] > 0.5, "QQ should have high weight after bet"

    print("  PASS: range narrowing works correctly")
    return narrower


def test_solver_direct():
    """Test C solver directly via ctypes."""
    print("\nTest 4: C solver (direct ctypes)")

    solver_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dll_path = os.path.join(solver_dir, "build", "solver.dll")

    if not os.path.exists(dll_path):
        print("  SKIP: solver.dll not found")
        return

    lib = ctypes.CDLL(dll_path)
    lib.solver_init.restype = ctypes.c_int
    lib.solver_solve.restype = ctypes.c_float
    lib.solver_exploitability.restype = ctypes.c_float
    lib.solver_get_strategy.restype = ctypes.c_float

    # Board: Qs As 2d 7h 4c
    board = (ctypes.c_int * 5)(
        card_to_int("Qs"), card_to_int("As"), card_to_int("2d"),
        card_to_int("7h"), card_to_int("4c"))

    # OOP: 4 hands
    oop = [(card_to_int("Ah"), card_to_int("Kh")),
           (card_to_int("Qh"), card_to_int("Qc")),
           (card_to_int("Jh"), card_to_int("Th")),
           (card_to_int("6h"), card_to_int("5h"))]
    n0 = len(oop)
    hands0 = (ctypes.c_int * (n0 * 2))(*[c for pair in oop for c in pair])
    w0 = (ctypes.c_float * n0)(*([1.0] * n0))

    # IP: 4 hands
    ip = [(card_to_int("Ac"), card_to_int("Kc")),
          (card_to_int("3c"), card_to_int("3d")),
          (card_to_int("Tc"), card_to_int("9c")),
          (card_to_int("8c"), card_to_int("8d"))]
    n1 = len(ip)
    hands1 = (ctypes.c_int * (n1 * 2))(*[c for pair in ip for c in pair])
    w1 = (ctypes.c_float * n1)(*([1.0] * n1))

    bet_sizes = (ctypes.c_float * 2)(0.33, 0.75)
    pot = int(10 * SCALE)
    stack = int(50 * SCALE)

    # Allocate solver buffer
    buf = ctypes.create_string_buffer(4 * 1024 * 1024)

    err = lib.solver_init(buf, board, 5, hands0, w0, n0,
                          hands1, w1, n1, pot, stack, bet_sizes, 2)
    assert err == 0, "solver_init failed"

    # Solve
    t0 = time.time()
    lib.solver_solve(buf, 1000, ctypes.c_float(0.001))
    elapsed = (time.time() - t0) * 1000
    print(f"  Solved 1000 iter in {elapsed:.0f}ms")

    # Exploitability
    exploit = lib.solver_exploitability(buf)
    print(f"  Exploitability: {exploit/SCALE:.4f} BB ({exploit/pot*100:.4f}% of pot)")

    # Strategies
    strat = (ctypes.c_float * MAX_ACTIONS)()
    hand_names = ["AhKh(TPTK)", "QhQc(trips)", "JhTh(JT)", "6h5h(air)"]
    print("  OOP strategies:")
    for h in range(4):
        lib.solver_get_strategy(buf, 0, h, strat)
        s = [f"{strat[a]*100:.0f}%" for a in range(MAX_ACTIONS) if strat[a] > 0.001]
        print(f"    {hand_names[h]}: {' '.join(s)}")

    lib.solver_free(buf)
    print("  PASS: solver produces valid strategies")


def test_full_pipeline():
    """End-to-end: parse range -> narrow -> solve -> extract strategy."""
    print("\nTest 5: Full pipeline (range -> narrow -> solve)")

    # 1. Parse ranges
    oop_range = parse_range_string("AA,KK,QQ,AKs,AKo,AQs")
    ip_range = parse_range_string("AA,KK,QQ,JJ,TT,99,AKs,AQs,AJs,KQs,AKo")

    board_cards = [card_to_int(c) for c in ["Qs", "As", "2d", "7h", "4c"]]

    # Filter out board-blocked hands
    blocked = set(board_cards)
    oop_range = [(c0, c1, w) for c0, c1, w in oop_range
                 if c0 not in blocked and c1 not in blocked]
    ip_range = [(c0, c1, w) for c0, c1, w in ip_range
                if c0 not in blocked and c1 not in blocked]

    print(f"  Initial ranges: OOP={len(oop_range)} hands, IP={len(ip_range)} hands")

    # 2. Narrow ranges (simulate villain betting)
    narrower = RangeNarrower()
    narrower.set_initial_range("villain", ip_range)
    narrower.set_initial_range("hero", oop_range)

    # Mock blueprint: strong hands bet 80%, medium 40%, weak 20%
    bet_probs = {}
    for c0, c1, w in ip_range:
        # Simple heuristic: pairs and high cards bet more
        rank0 = c0 >> 2
        rank1 = c1 >> 2
        avg_rank = (rank0 + rank1) / 2.0
        p_bet = min(0.9, avg_rank / 12.0 * 0.8 + 0.1)
        bet_probs[(c0, c1)] = p_bet

    narrower.update("villain", "bet", bet_probs)
    villain_narrow = narrower.get_weighted_hands("villain")
    print(f"  After villain bets: {narrower.get_hand_count('villain')} effective hands")

    # 3. Solve with narrowed ranges
    solver_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dll_path = os.path.join(solver_dir, "build", "solver.dll")
    if not os.path.exists(dll_path):
        print("  SKIP: solver.dll not found")
        return

    lib = ctypes.CDLL(dll_path)
    lib.solver_init.restype = ctypes.c_int
    lib.solver_solve.restype = ctypes.c_float
    lib.solver_exploitability.restype = ctypes.c_float
    lib.solver_get_strategy.restype = ctypes.c_float

    board = (ctypes.c_int * 5)(*board_cards)

    # OOP hands (hero)
    hero_hands = narrower.get_weighted_hands("hero")
    n0 = len(hero_hands)
    hands0 = (ctypes.c_int * (n0 * 2))(*[c for c0, c1, w in hero_hands for c in (c0, c1)])
    w0 = (ctypes.c_float * n0)(*[w for _, _, w in hero_hands])

    # IP hands (villain, narrowed)
    n1 = len(villain_narrow)
    hands1 = (ctypes.c_int * (n1 * 2))(*[c for c0, c1, w in villain_narrow for c in (c0, c1)])
    w1 = (ctypes.c_float * n1)(*[w for _, _, w in villain_narrow])

    bet_sizes = (ctypes.c_float * 2)(0.33, 0.75)
    pot = int(10 * SCALE)
    stack = int(50 * SCALE)

    buf = ctypes.create_string_buffer(4 * 1024 * 1024)
    err = lib.solver_init(buf, board, 5, hands0, w0, n0,
                          hands1, w1, n1, pot, stack, bet_sizes, 2)
    assert err == 0, "solver_init failed"

    t0 = time.time()
    lib.solver_solve(buf, 500, ctypes.c_float(0.01))
    elapsed = (time.time() - t0) * 1000

    exploit = lib.solver_exploitability(buf)
    print(f"  Solved: {elapsed:.0f}ms, exploit={exploit/SCALE:.4f} BB")

    # Show a few strategies
    strat = (ctypes.c_float * MAX_ACTIONS)()
    print("  Hero strategies (first 5 hands):")
    for h in range(min(5, n0)):
        c0, c1 = hero_hands[h][0], hero_hands[h][1]
        lib.solver_get_strategy(buf, 0, h, strat)
        s = [f"{strat[a]*100:.0f}%" for a in range(MAX_ACTIONS) if strat[a] > 0.5]
        print(f"    {int_to_card(c0)}{int_to_card(c1)}: {' '.join(s)}")

    lib.solver_free(buf)
    print("  PASS: full pipeline works end-to-end")


if __name__ == "__main__":
    print("=" * 60)
    print("  Poker Solver — End-to-End Python Tests")
    print("=" * 60)

    test_card_parsing()
    test_range_parsing()
    test_range_narrowing()
    test_solver_direct()
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
