#!/usr/bin/env python3
"""
Convergence trajectory test: run solver at increasing iteration counts,
extract root strategies at each milestone, and compare.
"""
import ctypes
import os
import sys
import time
import json

SO_PATH = sys.argv[1] if len(sys.argv) > 1 else "build/mccfr_blueprint.so"
THREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
HASH_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000_000

MILESTONES = [1_000_000, 5_000_000, 20_000_000, 100_000_000, 500_000_000]

bp = ctypes.CDLL(SO_PATH)
bp.bp_default_config.restype = None
bp.bp_init_unified.restype = ctypes.c_int
bp.bp_set_preflop_tier.restype = ctypes.c_int
bp.bp_set_preflop_tier.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
]
bp.bp_solve.restype = ctypes.c_int
bp.bp_solve.argtypes = [ctypes.c_void_p, ctypes.c_int64]
bp.bp_num_info_sets.restype = ctypes.c_int64
bp.bp_num_info_sets.argtypes = [ctypes.c_void_p]
bp.bp_save_regrets.restype = ctypes.c_int
bp.bp_save_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]


class BPConfig(ctypes.Structure):
    _fields_ = [
        ("discount_stop_iter", ctypes.c_int64),
        ("discount_interval", ctypes.c_int64),
        ("prune_start_iter", ctypes.c_int64),
        ("snapshot_start_iter", ctypes.c_int64),
        ("snapshot_interval", ctypes.c_int64),
        ("strategy_interval", ctypes.c_int64),
        ("num_threads", ctypes.c_int),
        ("hash_table_size", ctypes.c_int64),
        ("snapshot_dir", ctypes.c_char_p),
        ("include_preflop", ctypes.c_int),
        ("postflop_num_buckets", ctypes.c_int),
    ]


config = BPConfig()
bp.bp_default_config(ctypes.byref(config))
config.num_threads = THREADS
config.include_preflop = 1
config.hash_table_size = HASH_SIZE

# Match production discount parameters
config.discount_stop_iter = 1_627_499_999
config.discount_interval = 40_687_499
config.prune_start_iter = 790_499_999
config.snapshot_start_iter = 3_254_999_999
config.snapshot_interval = 790_499_999
config.strategy_interval = 10000

buf = (ctypes.c_char * 524288)()
solver = ctypes.cast(buf, ctypes.c_void_p)

postflop = (ctypes.c_float * 2)(0.5, 1.0)
preflop = (ctypes.c_float * 8)(0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0)

print(f"Initializing: {HASH_SIZE:,} table, {THREADS} threads", flush=True)
ret = bp.bp_init_unified(solver, 6, 50, 100, 10000,
                          postflop, 2, preflop, 8, ctypes.byref(config))
assert ret == 0

tiers = {0: [0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0],
         1: [0.7, 1.0, 2.5], 2: [1.0, 4.0], 3: [8.0]}
for level, sizes in tiers.items():
    c = (ctypes.c_float * len(sizes))(*sizes)
    bp.bp_set_preflop_tier(solver, level, c, len(sizes), 4)

total_done = 0
t0 = time.time()

for milestone in MILESTONES:
    if milestone > HASH_SIZE * 3:
        print(f"\nSkipping {milestone:,} (would overflow {HASH_SIZE:,} table)", flush=True)
        continue

    chunk = milestone - total_done
    if chunk <= 0:
        continue

    print(f"\n{'='*60}", flush=True)
    print(f"Running to {milestone:,} iterations (+{chunk:,})...", flush=True)
    bp.bp_solve(solver, chunk)
    total_done = milestone
    elapsed = time.time() - t0
    n_is = bp.bp_num_info_sets(solver)
    speed = total_done / elapsed if elapsed > 0 else 0
    print(f"Done: {n_is:,} IS, {elapsed:.0f}s, {speed:.0f} iter/s", flush=True)

    # Save checkpoint
    ckpt = f"/tmp/regrets_{milestone//1000000}M.bin"
    bp.bp_save_regrets(solver, ckpt.encode())
    print(f"Saved {ckpt}", flush=True)

    # Run extraction
    print(f"Extracting root strategies...", flush=True)
    os.system(f"/tmp/extract_roots {ckpt} > /tmp/extract_{milestone//1000000}M.txt 2>&1")

    # Parse and print summary
    with open(f"/tmp/extract_{milestone//1000000}M.txt") as ef:
        lines = ef.readlines()

    # Find key hands for each position
    key_hands = {
        "AA": 0, "KK": 25, "AKo": 2, "TT": 86, "55": 155,
        "87s": 131, "32o": 167, "72o": 159, "J7o": 93, "K8o": 52
    }

    print(f"\n--- Milestone: {milestone:,} iters, {n_is:,} IS ---")
    print(f"{'Hand':<6}", end="")

    # Find position headers and summaries
    positions = []
    for i, line in enumerate(lines):
        if "found]" in line:
            parts = line.strip().split()
            pos_name = parts[1]  # "open" or "vs"
            found = line.split("[")[1].split("/")[0]
            positions.append((pos_name, found))
        if "Summary:" in line and "fold" in line:
            # Extract fold/call/raise counts
            pass

    # Print position summaries
    for i, line in enumerate(lines):
        if "Summary:" in line:
            print(f"  {line.strip()}", flush=True)

    # Extract specific hands for UTG
    print(f"\nUTG key hands at {milestone:,} iters:")
    for line in lines:
        stripped = line.strip()
        for hand_name in ["AA ", "KK ", "TT ", "55 ", "32o", "72o", "87s", "J7o", "K8o"]:
            if stripped.startswith(hand_name) and "NOT FOUND" not in stripped:
                print(f"  {stripped}")
                break
        if "UTG open" in stripped:
            in_utg = True
        elif "========" in stripped and "UTG" not in stripped:
            in_utg = False

print(f"\n{'='*60}")
print(f"Total time: {time.time()-t0:.0f}s")
print("DONE", flush=True)
