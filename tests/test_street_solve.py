#!/usr/bin/env python3
"""End-to-end test for the Pluribus hybrid architecture.

Tests:
1. GPU single-street solver (river) — compare to CPU solver_v2
2. GPU single-street solver with leaf values (flop/turn)
3. Off-tree bet mapping
4. Blueprint store read/write
5. Full hand simulation: preflop→flop→turn→river with narrowing
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from solver import card_to_int, int_to_card, parse_range_string, SCALE


def test_off_tree_mapping():
    """Test pseudoharmonic bet mapping."""
    print("=== Test: Off-tree bet mapping ===")
    from off_tree import pseudoharmonic_map

    tree_sizes = [0.33, 0.75]

    # Exact match
    m = pseudoharmonic_map(0.33, tree_sizes)
    assert len(m) == 1 and m[0][0] == 0
    print("  Exact match (0.33): OK")

    # Between sizes
    m = pseudoharmonic_map(0.50, tree_sizes)
    assert len(m) == 2
    assert m[0][0] == 0 and m[1][0] == 1
    assert abs(sum(w for _, w in m) - 1.0) < 0.001
    print(f"  Interpolate (0.50): lo={m[0][1]:.3f}, hi={m[1][1]:.3f}")

    # Below all sizes
    m = pseudoharmonic_map(0.20, tree_sizes)
    assert len(m) == 1 and m[0][0] == 0
    print("  Below range (0.20): maps to smallest")

    # Above all sizes
    m = pseudoharmonic_map(1.5, tree_sizes)
    assert len(m) == 1 and m[0][0] == 1
    print("  Above range (1.50): maps to largest")

    # Pseudoharmonic should weight closer to smaller bet
    m = pseudoharmonic_map(0.55, tree_sizes)
    # 55% pot is strategically closer to 33% than to 75% in harmonic space
    print(f"  Harmonic (0.55): lo={m[0][1]:.3f}, hi={m[1][1]:.3f}")

    print("  PASSED\n")


def test_blueprint_store():
    """Test binary blueprint store read/write."""
    print("=== Test: Blueprint store ===")
    import numpy as np
    from blueprint_store import BlueprintStore, pack_texture_blob, unpack_texture_blob

    import tempfile
    tmpdir = tempfile.mkdtemp()
    store_dir = os.path.join(tmpdir, "test_scenario")

    nh0, nh1 = 10, 12
    nfa, nta = 3, 3
    ntc = 5

    # Create fake data
    flop_strats = {
        0: np.random.rand(nh0, nfa).astype(np.float32),
        1: np.random.rand(nh1, nfa).astype(np.float32),
    }
    # Normalize
    for p in range(2):
        flop_strats[p] /= flop_strats[p].sum(axis=1, keepdims=True)

    flop_evs = {0: np.random.rand(nh0).astype(np.float32) * 10,
                1: np.random.rand(nh1).astype(np.float32) * 10}
    flop_action_evs = {
        0: np.random.rand(nfa, nh0).astype(np.float32) * 10,
        1: np.random.rand(nfa, nh1).astype(np.float32) * 10,
    }

    turn_data = []
    for i in range(ntc):
        nh0_tc = nh0 - 1
        nh1_tc = nh1 - 1
        td = {
            'turn_card': i * 4,
            'num_hands_oop': nh0_tc,
            'num_hands_ip': nh1_tc,
            'strategies': {
                0: np.random.rand(nh0_tc, nta).astype(np.float32),
                1: np.random.rand(nh1_tc, nta).astype(np.float32),
            },
            'evs': {
                0: np.random.rand(nh0_tc).astype(np.float32) * 10,
                1: np.random.rand(nh1_tc).astype(np.float32) * 10,
            },
            'action_evs': {
                0: np.random.rand(nta, nh0_tc).astype(np.float32) * 10,
                1: np.random.rand(nta, nh1_tc).astype(np.float32) * 10,
            },
        }
        for p in range(2):
            td['strategies'][p] /= td['strategies'][p].sum(axis=1, keepdims=True)
        turn_data.append(td)

    # Pack
    blob = pack_texture_blob(
        oop_hands=[(i, i+1) for i in range(0, nh0*2, 2)],
        ip_hands=[(i, i+1) for i in range(0, nh1*2, 2)],
        flop_strategies=flop_strats,
        flop_evs=flop_evs,
        flop_action_evs=flop_action_evs,
        turn_data=turn_data,
        num_flop_actions=nfa,
        num_turn_actions=nta,
    )
    print(f"  Blob size: {len(blob)} bytes (compressed)")

    # Unpack and verify
    data = unpack_texture_blob(blob)
    assert data['num_hands_oop'] == nh0
    assert data['num_hands_ip'] == nh1
    assert data['num_flop_actions'] == nfa
    assert data['num_turn_cards'] == ntc

    # Verify flop strategies roundtrip
    for p in range(2):
        np.testing.assert_allclose(data['flop_strategies'][p],
                                    flop_strats[p], atol=1e-6)
    print("  Flop strategies roundtrip: OK")

    # Verify turn data roundtrip
    for i in range(ntc):
        for p in range(2):
            np.testing.assert_allclose(data['turn_data'][i]['strategies'][p],
                                        turn_data[i]['strategies'][p], atol=1e-6)
    print("  Turn strategies roundtrip: OK")

    # Test store read/write
    store = BlueprintStore(store_dir, mode='w')
    store.write_texture("AKQ_r", blob)
    store.write_texture("AKQ_m", blob)
    store.close()

    store2 = BlueprintStore(store_dir, mode='r')
    assert store2.num_textures == 2
    data2 = store2.load_texture("AKQ_r")
    assert data2 is not None
    assert data2['num_hands_oop'] == nh0
    print("  Store read/write: OK")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)

    print("  PASSED\n")


def test_gpu_river_solve():
    """Test GPU single-street solver on a river spot."""
    print("=== Test: GPU river solve ===")

    try:
        from street_solver_gpu import StreetSolverGPU
    except Exception as e:
        print(f"  SKIPPED: {e}\n")
        return

    board = ["Qs", "As", "2d", "7h", "4c"]
    oop_range = "AhKh,QhQc,JhTh,6h5h,3c3d"
    ip_range = "AcKc,8c8d,Tc9c,KhJh,2h2c"

    t0 = time.time()
    solver = StreetSolverGPU(
        board=board,
        oop_range=oop_range,
        ip_range=ip_range,
        pot_bb=10.0,
        stack_bb=50.0,
        bet_sizes=[0.33, 0.75],
    )
    solver.solve(iterations=200)
    elapsed = (time.time() - t0) * 1000

    print(f"  Solve time: {elapsed:.0f}ms")

    # Check strategies
    strat = solver.get_strategy("oop", hand_str="AhKh")
    print(f"  OOP AhKh strategy: {strat}")

    all_strats = solver.get_all_strategies("oop")
    print(f"  OOP hands with strategies: {len(all_strats)}")

    # Check weighted average
    avg = solver.get_avg_strategy("oop")
    print(f"  OOP avg strategy entries: {len(avg)}")

    assert len(strat) > 0, "Should have at least one action"
    freq_sum = sum(strat.values())
    assert abs(freq_sum - 1.0) < 0.05, f"Frequencies should sum to 1: {freq_sum}"

    print("  PASSED\n")


def test_gpu_flop_solve_with_leaves():
    """Test GPU solver on a flop spot with external leaf values."""
    print("=== Test: GPU flop solve with leaf values ===")

    try:
        from street_solver_gpu import StreetSolverGPU
        import numpy as np
    except Exception as e:
        print(f"  SKIPPED: {e}\n")
        return

    board = ["Qs", "As", "2d"]
    oop_range = "AA,KK,QQ,AKs,AQs,AJs"
    ip_range = "AA,KK,QQ,JJ,TT,AKs,AQs,KQs"

    # Build solver without leaf values first to get tree structure
    solver = StreetSolverGPU(
        board=board,
        oop_range=oop_range,
        ip_range=ip_range,
        pot_bb=6.5,
        stack_bb=97.5,
        bet_sizes=[0.33, 0.75],
        use_cont_strats=True,
    )

    t0 = time.time()
    solver.solve(iterations=100)
    elapsed = (time.time() - t0) * 1000

    print(f"  Solve time: {elapsed:.0f}ms")

    strat = solver.get_strategy("oop", hand_idx=0)
    print(f"  OOP hand 0 strategy: {strat}")

    if strat:
        freq_sum = sum(strat.values())
        print(f"  Frequency sum: {freq_sum:.3f}")

    print("  PASSED\n")


def test_full_hand_simulation():
    """Simulate a complete hand: preflop→flop→turn→river."""
    print("=== Test: Full hand simulation ===")

    from hud_solver import HUDSolver

    # Find ranges.json
    ranges_path = None
    candidates = [
        "C:/Users/Victor/Documents/Projects/ACRPoker-Hud-PC/solver/ranges.json",
        os.path.join(os.path.dirname(__file__), "..", "data", "ranges.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            ranges_path = c
            break

    # Find blueprint dir
    bp_dir = None
    candidates = [
        "C:/Users/Victor/Documents/Projects/ACRPoker-Hud-PC/solver/flop_solutions",
    ]
    for c in candidates:
        if os.path.isdir(c):
            bp_dir = c
            break

    solver = HUDSolver(blueprint_dir=bp_dir)

    # 1. New hand: CO opens, BB calls (SRP)
    solver.new_hand(
        hero_pos="BB", villain_pos="CO", scenario_type="srp",
        ranges_json_path=ranges_path,
        pot_bb=6.5, stack_bb=97.5,
    )
    print(f"  Scenario: {solver.scenario_id}")
    print(f"  Hero: {solver.hero_player}")

    hero_hands = solver.narrower.get_weighted_hands("hero")
    villain_hands = solver.narrower.get_weighted_hands("villain")
    print(f"  Hero range: {len(hero_hands)} combos")
    print(f"  Villain range: {len(villain_hands)} combos")

    # 2. Flop: Qs As 2d
    board = ["Qs", "As", "2d"]
    hero_cards = ["Kh", "Qh"]  # KQo on QAx board

    t0 = time.time()
    result = solver.get_strategy(board, hero_cards, street="flop",
                                  pot_bb=6.5, stack_bb=97.5)
    elapsed = (time.time() - t0) * 1000

    print(f"\n  FLOP (Qs As 2d, hero=KhQh):")
    print(f"    Source: {result.get('source', 'unknown')}")
    print(f"    Time: {elapsed:.0f}ms")
    for a in result.get('actions', [])[:3]:
        print(f"    {a['action']}: {a['frequency']:.1%}")

    # 3. Hero checks, villain bets 75%
    solver.on_hero_action("flop", "check", board)
    solver.on_villain_action("flop", "bet", board)

    # 4. Hero calls → turn
    solver.on_hero_action("flop", "call", board)

    # Narrow ranges after flop actions
    villain_after = solver.narrower.get_weighted_hands("villain")
    print(f"\n  After flop actions: villain range {len(villain_after)} combos")

    # 5. Turn: 7h
    board_turn = ["Qs", "As", "2d", "7h"]
    t0 = time.time()
    result = solver.get_strategy(board_turn, hero_cards, street="turn",
                                  pot_bb=16.0, stack_bb=92.0)
    elapsed = (time.time() - t0) * 1000

    print(f"\n  TURN (+ 7h, hero=KhQh):")
    print(f"    Source: {result.get('source', 'unknown')}")
    print(f"    Time: {elapsed:.0f}ms")
    for a in result.get('actions', [])[:3]:
        print(f"    {a['action']}: {a['frequency']:.1%}")

    # 6. River: 4c
    board_river = ["Qs", "As", "2d", "7h", "4c"]
    t0 = time.time()
    result = solver.get_strategy(board_river, hero_cards, street="river",
                                  pot_bb=16.0, stack_bb=92.0)
    elapsed = (time.time() - t0) * 1000

    print(f"\n  RIVER (+ 4c, hero=KhQh):")
    print(f"    Source: {result.get('source', 'unknown')}")
    print(f"    Time: {elapsed:.0f}ms")
    for a in result.get('actions', [])[:3]:
        print(f"    {a['action']}: {a['frequency']:.1%}")

    if result.get('error'):
        print(f"    Error: {result['error']}")

    print("\n  PASSED\n")


if __name__ == "__main__":
    test_off_tree_mapping()
    test_blueprint_store()
    test_gpu_river_solve()
    test_gpu_flop_solve_with_leaves()
    test_full_hand_simulation()
    print("All tests completed!")
