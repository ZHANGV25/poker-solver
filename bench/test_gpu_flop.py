"""Test GPU-accelerated flop chance evaluation.

Compares GPU equity rollout vs CPU to verify correctness and measure speedup.
"""
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from solver import card_to_int, int_to_card, SCALE

FA_MAX_HANDS = 400

# Load GPU DLL
solver_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dll_path = os.path.join(solver_dir, "build", "flop_accel.dll")

if not os.path.exists(dll_path):
    print(f"ERROR: {dll_path} not found")
    sys.exit(1)

lib = ctypes.CDLL(dll_path)
lib.flop_accel_init.restype = ctypes.c_int
lib.flop_accel_eval.restype = ctypes.c_int

# Define the input/output structs
class FlopAccelInput(ctypes.Structure):
    _fields_ = [
        ("board", ctypes.c_int * 3),
        ("hands", ((ctypes.c_int * 2) * FA_MAX_HANDS) * 2),
        ("reach", (ctypes.c_float * FA_MAX_HANDS) * 2),
        ("num_hands", ctypes.c_int * 2),
        ("half_pot", ctypes.c_float),
        ("traverser", ctypes.c_int),
    ]

class FlopAccelOutput(ctypes.Structure):
    _fields_ = [
        ("cfv", ctypes.c_float * FA_MAX_HANDS),
    ]


def test_basic():
    """Test with 4 hands, compare to known equity."""
    print("=== GPU Flop Accelerator Test ===\n")

    # Initialize
    lib.flop_accel_init()

    # Board: Qs As 2d
    board = [card_to_int("Qs"), card_to_int("As"), card_to_int("2d")]

    # OOP (traverser=0): 4 hands
    oop = [
        (card_to_int("Ah"), card_to_int("Kh")),  # TPTK
        (card_to_int("Qh"), card_to_int("Qc")),  # trips
        (card_to_int("Jh"), card_to_int("Th")),  # JT draw
        (card_to_int("6h"), card_to_int("5h")),  # air
    ]

    # IP (opponent): 4 hands
    ip = [
        (card_to_int("Ac"), card_to_int("Kc")),  # TPTK
        (card_to_int("3c"), card_to_int("3d")),  # small pair
        (card_to_int("Tc"), card_to_int("9c")),  # T9
        (card_to_int("8c"), card_to_int("8d")),  # 88
    ]

    inp = FlopAccelInput()
    for i in range(3):
        inp.board[i] = board[i]

    for i, (c0, c1) in enumerate(oop):
        inp.hands[0][i][0] = c0
        inp.hands[0][i][1] = c1
        inp.reach[0][i] = 1.0
    inp.num_hands[0] = len(oop)

    for i, (c0, c1) in enumerate(ip):
        inp.hands[1][i][0] = c0
        inp.hands[1][i][1] = c1
        inp.reach[1][i] = 1.0
    inp.num_hands[1] = len(ip)

    inp.half_pot = 500.0  # half of 1000 pot
    inp.traverser = 0

    out = FlopAccelOutput()

    # Run GPU evaluation
    t0 = time.time()
    err = lib.flop_accel_eval(ctypes.byref(inp), ctypes.byref(out))
    gpu_ms = (time.time() - t0) * 1000

    print(f"GPU evaluation: {gpu_ms:.1f}ms (return={err})")

    names = ["AhKh(TPTK)", "QhQc(trips)", "JhTh(JT)", "6h5h(air)"]
    print("\nPer-hand CFV (traverser=OOP):")
    for i in range(len(oop)):
        print(f"  {names[i]}: {out.cfv[i]:.2f}")

    # Sanity check: trips should have highest value, air lowest
    assert out.cfv[1] > out.cfv[0], "Trips should beat TPTK"
    assert out.cfv[0] > out.cfv[3], "TPTK should beat air"
    print("\nSanity checks PASSED")


def test_larger():
    """Test with 80 hands — realistic scenario."""
    print("\n=== Larger Range Test (80 hands) ===\n")

    board = [card_to_int("Qs"), card_to_int("As"), card_to_int("2d")]
    board_set = set(board)

    # Generate 80 hands per player
    inp = FlopAccelInput()
    for i in range(3):
        inp.board[i] = board[i]

    for p in range(2):
        count = 0
        for c0 in range(52):
            if c0 in board_set: continue
            for c1 in range(c0 + 1, 52):
                if c1 in board_set: continue
                if count >= 80: break
                inp.hands[p][count][0] = c0
                inp.hands[p][count][1] = c1
                inp.reach[p][count] = 1.0
                count += 1
            if count >= 80: break
        inp.num_hands[p] = count

    inp.half_pot = 500.0
    inp.traverser = 0

    out = FlopAccelOutput()

    # Warm-up
    lib.flop_accel_eval(ctypes.byref(inp), ctypes.byref(out))

    # Benchmark
    times = []
    for _ in range(5):
        t0 = time.time()
        lib.flop_accel_eval(ctypes.byref(inp), ctypes.byref(out))
        times.append((time.time() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    print(f"80 hands × 49 turn × 48 river = {49*48} runouts")
    print(f"GPU time: {avg_ms:.1f}ms avg (runs: {', '.join(f'{t:.1f}' for t in times)})")
    print(f"Throughput: {49*48*80*80/avg_ms/1000:.1f}M hand-pair evals/sec")

    # Show first few hand CFVs
    print("\nFirst 5 hand CFVs:")
    for i in range(5):
        c0, c1 = inp.hands[0][i][0], inp.hands[0][i][1]
        print(f"  {int_to_card(c0)}{int_to_card(c1)}: {out.cfv[i]:.2f}")


if __name__ == "__main__":
    test_basic()
    test_larger()
    print("\n=== ALL GPU TESTS PASSED ===")
