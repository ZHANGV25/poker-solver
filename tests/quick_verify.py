#!/usr/bin/env python3
"""
Quick verification: run a small tiered solve and extract preflop strategies.
Designed to run ALONGSIDE the main solver on the same instance.
Uses a small hash table (200M = ~12 GB) and runs 50M iterations.
"""
import ctypes
import os
import sys
import time

SO_PATH = sys.argv[1] if len(sys.argv) > 1 else "build/mccfr_blueprint.so"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
HASH_SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 10_000_000  # tiny table

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

buf = (ctypes.c_char * 524288)()
solver = ctypes.cast(buf, ctypes.c_void_p)

postflop = (ctypes.c_float * 2)(0.5, 1.0)
preflop = (ctypes.c_float * 8)(0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0)

print(f"Initializing (200M table, {THREADS} threads)...", flush=True)
ret = bp.bp_init_unified(solver, 6, 50, 100, 10000,
                          postflop, 2, preflop, 8, ctypes.byref(config))
assert ret == 0

tiers = {0: [0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0],
         1: [0.7, 1.0, 2.5], 2: [1.0, 4.0], 3: [8.0]}
for level, sizes in tiers.items():
    c = (ctypes.c_float * len(sizes))(*sizes)
    bp.bp_set_preflop_tier(solver, level, c, len(sizes), 4)

print(f"Running {ITERS:,} iterations...", flush=True)
t0 = time.time()
bp.bp_solve(solver, ITERS)
elapsed = time.time() - t0
n = bp.bp_num_info_sets(solver)
print(f"Done: {n:,} IS in {elapsed:.0f}s ({ITERS/elapsed:.0f} iter/s)", flush=True)

# Save checkpoint for extraction
out = "/tmp/verify_regrets.bin"
bp.bp_save_regrets(solver, out.encode())
print(f"Saved to {out}", flush=True)
