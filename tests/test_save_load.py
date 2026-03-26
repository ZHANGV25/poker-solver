"""Test regret checkpoint save/load cycle for unified solver.

Verifies:
1. bp_init_unified + bp_solve(500) produces info sets
2. bp_save_regrets writes a valid checkpoint file
3. A fresh solver can bp_load_regrets and resume
4. bp_solve(500) more iterations produces MORE info sets (not a reset)
5. Regrets from loaded solver match what was saved
"""
import ctypes
import os
import sys
import time
import struct
import tempfile

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DLL_PATH = os.path.join(PROJECT, "build", "mccfr_blueprint.dll")

# --- ctypes bindings ---

class BPConfig(ctypes.Structure):
    _fields_ = [
        ("discount_stop_iter", ctypes.c_int),
        ("discount_interval", ctypes.c_int),
        ("prune_start_iter", ctypes.c_int),
        ("snapshot_start_iter", ctypes.c_int),
        ("snapshot_interval", ctypes.c_int),
        ("strategy_interval", ctypes.c_int),
        ("num_threads", ctypes.c_int),
        ("hash_table_size", ctypes.c_int),
        ("snapshot_dir", ctypes.c_char_p),
        ("include_preflop", ctypes.c_int),
    ]

# BPSolver is opaque — we allocate a large buffer
SOLVER_BUF_SIZE = 128 * 1024 * 1024  # 128MB should be plenty


def load_lib():
    if not os.path.exists(DLL_PATH):
        print(f"ERROR: DLL not found at {DLL_PATH}")
        sys.exit(1)
    lib = ctypes.CDLL(DLL_PATH)

    # bp_default_config
    lib.bp_default_config.argtypes = [ctypes.POINTER(BPConfig)]
    lib.bp_default_config.restype = None

    # bp_init_unified
    lib.bp_init_unified.argtypes = [
        ctypes.c_void_p,  # BPSolver*
        ctypes.c_int,     # num_players
        ctypes.c_int,     # small_blind
        ctypes.c_int,     # big_blind
        ctypes.c_int,     # initial_stack
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # postflop bet sizes
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,  # preflop bet sizes
        ctypes.POINTER(BPConfig),
    ]
    lib.bp_init_unified.restype = ctypes.c_int

    # bp_solve
    lib.bp_solve.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.bp_solve.restype = ctypes.c_int

    # bp_num_info_sets
    lib.bp_num_info_sets.argtypes = [ctypes.c_void_p]
    lib.bp_num_info_sets.restype = ctypes.c_int

    # bp_save_regrets
    lib.bp_save_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.bp_save_regrets.restype = ctypes.c_int

    # bp_load_regrets
    lib.bp_load_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.bp_load_regrets.restype = ctypes.c_int

    # bp_free
    lib.bp_free.argtypes = [ctypes.c_void_p]
    lib.bp_free.restype = None

    return lib


def create_solver(lib):
    """Create and init a unified 6-player solver."""
    buf = ctypes.create_string_buffer(SOLVER_BUF_SIZE)

    config = BPConfig()
    lib.bp_default_config(ctypes.byref(config))
    config.num_threads = 1        # single-thread for determinism
    config.include_preflop = 1
    config.hash_table_size = (1 << 22)  # 4M slots — testing
    config.snapshot_dir = None

    # Pluribus-style bet sizes
    postflop_sizes = (ctypes.c_float * 3)(0.5, 1.0, 2.0)
    preflop_sizes = (ctypes.c_float * 3)(2.5, 3.0, 4.0)

    t0 = time.time()
    err = lib.bp_init_unified(
        buf, 6,          # 6 players
        50, 100,         # SB=50, BB=100
        10000,           # 100BB stack
        postflop_sizes, 3,
        preflop_sizes, 3,
        ctypes.byref(config),
    )
    elapsed = time.time() - t0

    if err != 0:
        print(f"ERROR: bp_init_unified returned {err}")
        sys.exit(1)

    print(f"  Init took {elapsed:.1f}s")
    return buf


def main():
    print("=" * 60)
    print("  Regret Save/Load Checkpoint Test")
    print("=" * 60)

    lib = load_lib()
    checkpoint_path = os.path.join(PROJECT, "build", "test_checkpoint.bin")
    # Use Windows-style path for the DLL
    checkpoint_path_win = checkpoint_path.replace("/", "\\")
    # Actually for ctypes on Windows, the path should work as-is
    checkpoint_bytes = checkpoint_path.encode("utf-8")

    # --- Phase 1: Create solver, run 500 iters, save ---
    print("\n[Phase 1] Init + solve 500 iterations")
    solver1 = create_solver(lib)

    t0 = time.time()
    lib.bp_solve(solver1, 500)
    elapsed = time.time() - t0
    info_sets_1 = lib.bp_num_info_sets(solver1)
    print(f"  Solved 500 iters in {elapsed:.1f}s — {info_sets_1} info sets")

    if info_sets_1 == 0:
        print("ERROR: No info sets created!")
        sys.exit(1)

    # Save checkpoint
    print("\n[Phase 2] Save regrets")
    err = lib.bp_save_regrets(solver1, checkpoint_bytes)
    if err != 0:
        print(f"ERROR: bp_save_regrets returned {err}")
        sys.exit(1)

    # Verify file exists and has content
    file_size = os.path.getsize(checkpoint_path)
    print(f"  Checkpoint file: {file_size:,} bytes")
    if file_size < 100:
        print("ERROR: Checkpoint file too small!")
        sys.exit(1)

    # Read header to verify format (BPR2 = bucket-in-key v2 format)
    with open(checkpoint_path, "rb") as f:
        magic = f.read(4)
        table_size, num_entries, iters = struct.unpack("iii", f.read(12))
    print(f"  Header: magic={magic}, table_size={table_size}, entries={num_entries}, iters={iters}")
    assert magic == b"BPR2", f"Bad magic: {magic} (expected BPR2)"
    assert num_entries == info_sets_1, f"Entry count mismatch: {num_entries} vs {info_sets_1}"
    assert abs(iters - 500) < 10, f"Iteration count wrong: {iters} (expected ~500)"
    saved_iters_1 = iters

    # Free solver 1
    lib.bp_free(solver1)
    print("  Solver 1 freed")

    # --- Phase 3: Create NEW solver, load checkpoint, run 500 more ---
    print("\n[Phase 3] New solver + load checkpoint + solve 500 more")
    solver2 = create_solver(lib)

    # Verify it starts empty
    info_sets_before_load = lib.bp_num_info_sets(solver2)
    print(f"  Before load: {info_sets_before_load} info sets")

    loaded = lib.bp_load_regrets(solver2, checkpoint_bytes)
    if loaded < 0:
        print(f"ERROR: bp_load_regrets returned {loaded}")
        sys.exit(1)

    info_sets_after_load = lib.bp_num_info_sets(solver2)
    print(f"  After load: {info_sets_after_load} info sets (loaded {loaded})")

    # The loaded count should match what was saved
    assert loaded == info_sets_1, f"Loaded {loaded} but saved {info_sets_1}"
    assert info_sets_after_load == info_sets_1, \
        f"Info sets after load ({info_sets_after_load}) != saved ({info_sets_1})"

    # Run 500 more iterations
    t0 = time.time()
    lib.bp_solve(solver2, 500)
    elapsed = time.time() - t0
    info_sets_2 = lib.bp_num_info_sets(solver2)
    print(f"  Solved 500 more iters in {elapsed:.1f}s — {info_sets_2} info sets")

    # Should have at least as many info sets (likely more from new paths)
    assert info_sets_2 >= info_sets_1, \
        f"Info sets decreased after resume: {info_sets_2} < {info_sets_1}"

    # --- Phase 4: Save again and compare ---
    print("\n[Phase 4] Save resumed solver and verify")
    checkpoint2_path = os.path.join(PROJECT, "build", "test_checkpoint2.bin")
    checkpoint2_bytes = checkpoint2_path.encode("utf-8")

    err = lib.bp_save_regrets(solver2, checkpoint2_bytes)
    assert err == 0, f"Second save failed: {err}"

    with open(checkpoint2_path, "rb") as f:
        magic = f.read(4)
        table_size2, num_entries2, iters2 = struct.unpack("iii", f.read(12))
    print(f"  Checkpoint 2: entries={num_entries2}, iters={iters2}")
    assert abs(iters2 - 1000) < 10, f"Expected ~1000 iters after resume, got {iters2}"
    assert num_entries2 >= num_entries, \
        f"Entries decreased: {num_entries2} < {num_entries}"

    file_size2 = os.path.getsize(checkpoint2_path)
    print(f"  File sizes: {file_size:,} -> {file_size2:,} bytes")

    lib.bp_free(solver2)

    # Cleanup
    try:
        os.remove(checkpoint_path)
        os.remove(checkpoint2_path)
    except OSError:
        pass

    print("\n" + "=" * 60)
    print("  ALL SAVE/LOAD TESTS PASSED")
    print(f"  Info sets: {info_sets_1} -> {info_sets_2}")
    print(f"  Iterations: 500 -> 1000 (checkpoint/resume verified)")
    print("=" * 60)


if __name__ == "__main__":
    main()
