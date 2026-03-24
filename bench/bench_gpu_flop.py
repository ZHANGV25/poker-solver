"""Benchmark: GPU full flop-through-river solver at various iteration counts."""
import ctypes, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from solver import card_to_int, int_to_card

FS_MAX_HANDS = 200

class FSNode(ctypes.Structure):
    _fields_ = [('type',ctypes.c_int),('player',ctypes.c_int),('num_children',ctypes.c_int),
        ('first_child',ctypes.c_int),('pot',ctypes.c_int),('bets',ctypes.c_int*2),
        ('board_cards',ctypes.c_int*5),('num_board',ctypes.c_int)]

class FSTreeData(ctypes.Structure):
    _fields_ = [('nodes',ctypes.POINTER(FSNode)),('children',ctypes.POINTER(ctypes.c_int)),
        ('num_nodes',ctypes.c_int),('num_children_total',ctypes.c_int),
        ('hands',((ctypes.c_int*2)*FS_MAX_HANDS)*2),('weights',(ctypes.c_float*FS_MAX_HANDS)*2),
        ('num_hands',ctypes.c_int*2),('level_order',ctypes.POINTER(ctypes.c_int)),
        ('node_depth',ctypes.POINTER(ctypes.c_int)),('max_depth',ctypes.c_int),
        ('decision_node_indices',ctypes.POINTER(ctypes.c_int)),('num_decision_nodes',ctypes.c_int),
        ('showdown_node_indices',ctypes.POINTER(ctypes.c_int)),('num_showdown_nodes',ctypes.c_int)]

class FSOutput(ctypes.Structure):
    _fields_ = [('root_strategy',ctypes.POINTER(ctypes.c_float)),('root_num_actions',ctypes.c_int),
        ('root_player',ctypes.c_int),('root_ev',ctypes.POINTER(ctypes.c_float))]

lib = ctypes.CDLL('build/flop_solve.dll')
lib.fs_build_tree.restype = ctypes.c_int
lib.fs_solve_gpu.restype = ctypes.c_int


def setup_hands(td, n_hands):
    """Generate n_hands per player, avoiding board cards."""
    board_set = set()
    for i in range(3):
        # Read board from node 0
        pass
    # Just use first n_hands combos not blocked by Qs As 2d
    blocked = {card_to_int('Qs'), card_to_int('As'), card_to_int('2d')}
    for p in range(2):
        count = 0
        for c0 in range(52):
            if c0 in blocked: continue
            for c1 in range(c0+1, 52):
                if c1 in blocked: continue
                if count >= n_hands: break
                td.hands[p][count][0] = c0
                td.hands[p][count][1] = c1
                td.weights[p][count] = 1.0
                count += 1
            if count >= n_hands: break
        td.num_hands[p] = count


def run_solve(n_hands, iters, board_str=None):
    """Build tree, solve, return (time_ms, ms_per_iter, strategies_sample)."""
    if board_str is None:
        board_str = ['Qs', 'As', '2d']
    board = [card_to_int(c) for c in board_str]
    bs = (ctypes.c_float * 1)(0.75)

    td = FSTreeData()
    lib.fs_build_tree((ctypes.c_int * 3)(*board), 650, 9750, bs, 1, ctypes.byref(td))
    setup_hands(td, n_hands)

    out = FSOutput()
    t0 = time.time()
    lib.fs_solve_gpu(ctypes.byref(td), iters, ctypes.byref(out))
    ms = (time.time() - t0) * 1000

    # Extract first 5 strategies
    nh = td.num_hands[out.root_player]
    na = out.root_num_actions
    strats = []
    for h in range(min(8, nh)):
        c0 = td.hands[out.root_player][h][0]
        c1 = td.hands[out.root_player][h][1]
        s = [out.root_strategy[a * nh + h] for a in range(na)]
        strats.append((int_to_card(c0) + int_to_card(c1), s))

    lib.fs_free_output(ctypes.byref(out))
    lib.fs_free_tree(ctypes.byref(td))
    return ms, ms / iters, strats


print("=" * 70)
print("  GPU Flop Solver Benchmark — RTX 3060")
print("  Board: Qs As 2d | Bet sizes: 75% pot | 1 raise per street")
print("=" * 70)

# Warm up GPU
print("\nWarming up GPU...")
run_solve(4, 10)

# ── 8-hand benchmarks ────────────────────────────────
print("\n-- 8 HANDS --")
for iters in [50, 100, 150, 200, 300, 500]:
    ms, per, strats = run_solve(8, iters)
    print(f"  {iters:4d} iters: {ms:7.0f}ms total, {per:5.1f}ms/iter")

# ── 40-hand benchmarks ───────────────────────────────
print("\n-- 40 HANDS --")
for iters in [50, 100, 150, 200, 300]:
    ms, per, strats = run_solve(40, iters)
    print(f"  {iters:4d} iters: {ms:7.0f}ms total, {per:5.1f}ms/iter")

# ── 80-hand benchmarks ───────────────────────────────
print("\n-- 80 HANDS --")
for iters in [50, 100, 150, 200, 300]:
    ms, per, strats = run_solve(80, iters)
    print(f"  {iters:4d} iters: {ms:7.0f}ms total, {per:5.1f}ms/iter")

# ── Strategy convergence at 80 hands ─────────────────
print("\n-- CONVERGENCE CHECK (80 hands) --")
print("  Showing first 5 hands at root (check / bet75 / raise):")
for iters in [100, 200, 300]:
    ms, per, strats = run_solve(80, iters)
    print(f"\n  {iters} iterations ({ms:.0f}ms):")
    for name, s in strats[:5]:
        pct = " ".join(f"{x*100:5.1f}%" for x in s)
        print(f"    {name}: {pct}")

print("\n" + "=" * 70)
print("  BENCHMARK COMPLETE")
print("=" * 70)
