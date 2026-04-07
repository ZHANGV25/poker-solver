#!/usr/bin/env python3
"""Quick debug: test bp_init_unified with 1B hash size."""
import ctypes
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

bp = ctypes.CDLL('./build/mccfr_blueprint.so')
bp.bp_default_config.restype = None
bp.bp_init_unified.restype = ctypes.c_int
bp.bp_num_info_sets.restype = ctypes.c_int
bp.bp_load_regrets.restype = ctypes.c_int
bp.bp_load_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
bp.bp_export_strategies.restype = ctypes.c_int
bp.bp_export_strategies.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t)
]
bp.bp_free.restype = None

config = BPConfig()
bp.bp_default_config(ctypes.byref(config))
config.num_threads = 1
config.include_preflop = 1
config.hash_table_size = 1 << 30

print(f"hash_table_size = {config.hash_table_size:,}")
print(f"sizeof BPConfig = {ctypes.sizeof(BPConfig)}")

# Use a larger buffer - 2MB should be more than enough for BPSolver struct
buf = (ctypes.c_char * (2 * 1024 * 1024))()
solver = ctypes.cast(buf, ctypes.c_void_p)
print(f"solver buffer allocated: 2 MB")

c_post = (ctypes.c_float * 3)(0.5, 1.0, 2.0)
c_pre = (ctypes.c_float * 4)(0.5, 1.0, 2.0, 3.0)

print("Calling bp_init_unified...", flush=True)
ret = bp.bp_init_unified(solver, 6, 50, 100, 10000, c_post, 3, c_pre, 4, ctypes.byref(config))
print(f"bp_init_unified returned {ret}", flush=True)

if ret == 0:
    n_is = bp.bp_num_info_sets(solver)
    print(f"Info sets after init: {n_is}")

    regret_path = "/opt/blueprint_unified/regrets_200M.bin"
    if os.path.exists(regret_path):
        print(f"Loading regrets from {regret_path}...", flush=True)
        n = bp.bp_load_regrets(solver, regret_path.encode())
        print(f"Loaded {n:,} entries", flush=True)
        n_is = bp.bp_num_info_sets(solver)
        print(f"Info sets after load: {n_is:,}", flush=True)

        # Try export
        needed = ctypes.c_size_t(0)
        bp.bp_export_strategies(solver, None, 0, ctypes.byref(needed))
        print(f"Export buffer needed: {needed.value / 1024 / 1024:.1f} MB", flush=True)
    else:
        print(f"No regret file at {regret_path}")

    bp.bp_free(solver)
    print("Done.")
else:
    print("Init failed!")
