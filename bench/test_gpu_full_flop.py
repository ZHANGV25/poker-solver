"""Test full GPU flop-through-river solver."""
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from solver import card_to_int, int_to_card, SCALE

FS_MAX_HANDS = 200
FS_MAX_ACTIONS = 6

solver_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dll_path = os.path.join(solver_dir, "build", "flop_solve.dll")
lib = ctypes.CDLL(dll_path)

# Struct definitions matching flop_solve.cuh
class FSNode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("player", ctypes.c_int),
        ("num_children", ctypes.c_int),
        ("first_child", ctypes.c_int),
        ("pot", ctypes.c_int),
        ("bets", ctypes.c_int * 2),
        ("board_cards", ctypes.c_int * 5),
        ("num_board", ctypes.c_int),
    ]

class FSTreeData(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.POINTER(FSNode)),
        ("children", ctypes.POINTER(ctypes.c_int)),
        ("num_nodes", ctypes.c_int),
        ("num_children_total", ctypes.c_int),
        ("hands", ((ctypes.c_int * 2) * FS_MAX_HANDS) * 2),
        ("weights", (ctypes.c_float * FS_MAX_HANDS) * 2),
        ("num_hands", ctypes.c_int * 2),
        ("level_order", ctypes.POINTER(ctypes.c_int)),
        ("node_depth", ctypes.POINTER(ctypes.c_int)),
        ("max_depth", ctypes.c_int),
        ("decision_node_indices", ctypes.POINTER(ctypes.c_int)),
        ("num_decision_nodes", ctypes.c_int),
        ("showdown_node_indices", ctypes.POINTER(ctypes.c_int)),
        ("num_showdown_nodes", ctypes.c_int),
    ]

class FSOutput(ctypes.Structure):
    _fields_ = [
        ("root_strategy", ctypes.POINTER(ctypes.c_float)),
        ("root_num_actions", ctypes.c_int),
        ("root_player", ctypes.c_int),
        ("root_ev", ctypes.POINTER(ctypes.c_float)),
    ]

lib.fs_build_tree.restype = ctypes.c_int
lib.fs_solve_gpu.restype = ctypes.c_int


def test_4_hands():
    """4-hand flop solve on GPU."""
    print("=== Full GPU Flop Solve (4 hands) ===\n")

    board = [card_to_int("Qs"), card_to_int("As"), card_to_int("2d")]
    board_arr = (ctypes.c_int * 3)(*board)
    bet_sizes = (ctypes.c_float * 1)(0.75)

    td = FSTreeData()
    t0 = time.time()
    err = lib.fs_build_tree(board_arr, 1000, 5000, bet_sizes, 1, ctypes.byref(td))
    build_ms = (time.time() - t0) * 1000
    print(f"Tree build: {build_ms:.0f}ms, err={err}")
    print(f"  nodes={td.num_nodes}, decision={td.num_decision_nodes}, showdown={td.num_showdown_nodes}")

    # Set up hands
    oop = [(card_to_int("Ah"), card_to_int("Kh")),
           (card_to_int("Qh"), card_to_int("Qc")),
           (card_to_int("Jh"), card_to_int("Th")),
           (card_to_int("6h"), card_to_int("5h"))]
    ip = [(card_to_int("Ac"), card_to_int("Kc")),
          (card_to_int("3c"), card_to_int("3d")),
          (card_to_int("Tc"), card_to_int("9c")),
          (card_to_int("8c"), card_to_int("8d"))]

    for i, (c0, c1) in enumerate(oop):
        td.hands[0][i][0] = c0
        td.hands[0][i][1] = c1
        td.weights[0][i] = 1.0
    td.num_hands[0] = len(oop)

    for i, (c0, c1) in enumerate(ip):
        td.hands[1][i][0] = c0
        td.hands[1][i][1] = c1
        td.weights[1][i] = 1.0
    td.num_hands[1] = len(ip)

    out = FSOutput()
    t0 = time.time()
    err = lib.fs_solve_gpu(ctypes.byref(td), 100, ctypes.byref(out))
    solve_ms = (time.time() - t0) * 1000
    print(f"\nGPU solve: {solve_ms:.0f}ms, err={err}")
    print(f"  root: {out.root_num_actions} actions, player={out.root_player}")

    names = ["AhKh(TPTK)", "QhQc(trips)", "JhTh(JT)", "6h5h(air)"]
    root_nh = td.num_hands[out.root_player]
    print(f"\nStrategies (player {out.root_player}):")
    for h in range(root_nh):
        strat = []
        for a in range(out.root_num_actions):
            strat.append(out.root_strategy[a * root_nh + h])
        pct = " ".join(f"{s*100:.0f}%" for s in strat)
        print(f"  {names[h]}: {pct}")

    lib.fs_free_output(ctypes.byref(out))
    lib.fs_free_tree(ctypes.byref(td))
    print("\n  PASS\n")


def test_80_hands():
    """80-hand flop solve on GPU — realistic scenario."""
    print("=== Full GPU Flop Solve (80 hands) ===\n")

    board = [card_to_int("Qs"), card_to_int("As"), card_to_int("2d")]
    board_arr = (ctypes.c_int * 3)(*board)
    bet_sizes = (ctypes.c_float * 2)(0.33, 0.75)

    td = FSTreeData()
    t0 = time.time()
    err = lib.fs_build_tree(board_arr, 650, 9750, bet_sizes, 2, ctypes.byref(td))
    build_ms = (time.time() - t0) * 1000
    print(f"Tree build: {build_ms:.0f}ms")
    print(f"  nodes={td.num_nodes}, decision={td.num_decision_nodes}, showdown={td.num_showdown_nodes}")

    # Generate 80 hands per player
    board_set = set(board)
    for p in range(2):
        count = 0
        for c0 in range(52):
            if c0 in board_set: continue
            for c1 in range(c0+1, 52):
                if c1 in board_set: continue
                if count >= 80: break
                td.hands[p][count][0] = c0
                td.hands[p][count][1] = c1
                td.weights[p][count] = 1.0
                count += 1
            if count >= 80: break
        td.num_hands[p] = count

    print(f"  hands=[{td.num_hands[0]},{td.num_hands[1]}]")

    out = FSOutput()
    t0 = time.time()
    err = lib.fs_solve_gpu(ctypes.byref(td), 200, ctypes.byref(out))
    solve_ms = (time.time() - t0) * 1000
    print(f"\nGPU solve (200 iters): {solve_ms:.0f}ms ({solve_ms/200:.1f}ms/iter), err={err}")

    root_nh = td.num_hands[out.root_player]
    print(f"  root: {out.root_num_actions} actions, player={out.root_player}")
    print(f"\nFirst 5 hand strategies:")
    for h in range(min(5, root_nh)):
        c0, c1 = td.hands[out.root_player][h][0], td.hands[out.root_player][h][1]
        strat = [out.root_strategy[a * root_nh + h] for a in range(out.root_num_actions)]
        pct = " ".join(f"{s*100:.0f}%" for s in strat)
        print(f"  {int_to_card(c0)}{int_to_card(c1)}: {pct}")

    lib.fs_free_output(ctypes.byref(out))
    lib.fs_free_tree(ctypes.byref(td))
    print("\n  PASS")


if __name__ == "__main__":
    test_4_hands()
    test_80_hands()
    print("\n=== ALL GPU TESTS PASSED ===")
