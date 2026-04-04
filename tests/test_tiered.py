#!/usr/bin/env python3
"""Quick smoke test: verify tiered preflop sizing works end-to-end."""
import ctypes
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'precompute'))

BUILD_DIR = os.path.join(os.path.dirname(__file__), '..', 'build')
SO_PATH = os.path.join(BUILD_DIR, 'mccfr_blueprint.so')

bp = ctypes.CDLL(SO_PATH)
bp.bp_default_config.restype = None
bp.bp_init_unified.restype = ctypes.c_int
bp.bp_set_preflop_tier.restype = ctypes.c_int
bp.bp_set_preflop_tier.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
]
bp.bp_solve.restype = ctypes.c_int
bp.bp_solve.argtypes = [ctypes.c_void_p, ctypes.c_int]
bp.bp_num_info_sets.restype = ctypes.c_int
bp.bp_num_info_sets.argtypes = [ctypes.c_void_p]

# Config struct (must match BPConfig in header)
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
config.num_threads = 1
config.include_preflop = 1
config.hash_table_size = 4_000_000  # small table for test

buf = (ctypes.c_char * 524288)()
solver = ctypes.cast(buf, ctypes.c_void_p)

# Flat preflop sizes (passed to init, overridden by tiers)
postflop = (ctypes.c_float * 2)(0.5, 1.0)
preflop = (ctypes.c_float * 8)(0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0)

ret = bp.bp_init_unified(solver, 6, 50, 100, 10000,
                          postflop, 2, preflop, 8, ctypes.byref(config))
assert ret == 0, f"init failed: {ret}"

# Set tiered preflop sizes
tiers = {
    0: [0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0],
    1: [0.7, 1.0, 2.5],
    2: [1.0, 4.0],
    3: [8.0],
}
for level, sizes in tiers.items():
    c_sizes = (ctypes.c_float * len(sizes))(*sizes)
    ret = bp.bp_set_preflop_tier(solver, level, c_sizes, len(sizes), 4)
    assert ret == 0, f"set_tier {level} failed: {ret}"

print("Running 10K iterations with tiered preflop...")
ret = bp.bp_solve(solver, 10000)
assert ret == 0, f"solve failed: {ret}"

n_is = bp.bp_num_info_sets(solver)
print(f"Info sets created: {n_is:,}")
print(f"Expected: much less than flat-8 would create at 10K iters")

# Sanity: with tiered sizing and small table, should create reasonable IS count
assert n_is > 1000, f"Too few info sets: {n_is}"
assert n_is < 4_000_000, f"Table overflow: {n_is}"

print("PASS: Tiered preflop sizing works correctly")
