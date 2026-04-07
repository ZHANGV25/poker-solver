#!/usr/bin/env python3
"""One-shot BPS export with benchmarks. Minimal script — no imports beyond stdlib."""
import ctypes
import json
import lzma
import os
import struct
import subprocess
import sys
import time

NUM_PLAYERS = 6
SMALL_BLIND = 50
BIG_BLIND = 100
INITIAL_STACK = 10000
PREFLOP_BET_SIZES = [0.5, 1.0, 2.0, 3.0]
POSTFLOP_BET_SIZES = [0.5, 1.0, 2.0]


class BPConfig(ctypes.Structure):
    _fields_ = [
        ("discount_stop_iter", ctypes.c_int),
        ("discount_interval", ctypes.c_int),
        ("prune_start_iter", ctypes.c_int),
        ("snapshot_start_iter", ctypes.c_int),
        ("snapshot_interval", ctypes.c_int),
        ("strategy_interval", ctypes.c_int),
        ("num_threads", ctypes.c_int),
        ("hash_table_size", ctypes.c_int64),  # int64_t on EC2 build
        ("snapshot_dir", ctypes.c_char_p),
        ("include_preflop", ctypes.c_int),
        ("postflop_num_buckets", ctypes.c_int),
    ]


def main():
    regret_file = sys.argv[1] if len(sys.argv) > 1 else "/opt/blueprint_unified/regrets_200M.bin"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/opt/blueprint_unified"
    s3_bucket = sys.argv[3] if len(sys.argv) > 3 else "poker-blueprint-unified"
    hash_size = int(sys.argv[4]) if len(sys.argv) > 4 else (1 << 30)

    timings = {}
    t_total = time.time()

    print("=" * 60)
    print("  BPS Export Tool v2 — Benchmarked")
    print("=" * 60, flush=True)

    # Load DLL
    t0 = time.time()
    bp = ctypes.CDLL("./build/mccfr_blueprint.so")
    bp.bp_default_config.restype = None
    bp.bp_init_unified.restype = ctypes.c_int
    bp.bp_num_info_sets.restype = ctypes.c_int
    bp.bp_export_strategies.restype = ctypes.c_int
    bp.bp_export_strategies.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t)
    ]
    bp.bp_load_regrets.restype = ctypes.c_int
    bp.bp_load_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    bp.bp_free.restype = None
    timings["dll_load"] = time.time() - t0
    print(f"[{timings['dll_load']:.2f}s] DLL loaded", flush=True)

    # Init solver
    t0 = time.time()
    config = BPConfig()
    bp.bp_default_config(ctypes.byref(config))
    config.num_threads = 1
    config.include_preflop = 1
    config.hash_table_size = hash_size
    config.postflop_num_buckets = 200

    buf = (ctypes.c_char * (2 * 1024 * 1024))()
    solver = ctypes.cast(buf, ctypes.c_void_p)

    c_post = (ctypes.c_float * len(POSTFLOP_BET_SIZES))(*POSTFLOP_BET_SIZES)
    c_pre = (ctypes.c_float * len(PREFLOP_BET_SIZES))(*PREFLOP_BET_SIZES)

    print(f"Calling bp_init_unified (hash_size={hash_size:,})...", flush=True)
    ret = bp.bp_init_unified(
        solver, NUM_PLAYERS,
        SMALL_BLIND, BIG_BLIND, INITIAL_STACK,
        c_post, len(POSTFLOP_BET_SIZES),
        c_pre, len(PREFLOP_BET_SIZES),
        ctypes.byref(config)
    )
    if ret != 0:
        print(f"FATAL: bp_init_unified returned {ret}")
        sys.exit(1)
    timings["init"] = time.time() - t0
    print(f"[{timings['init']:.2f}s] Solver initialized", flush=True)

    # Load regrets
    t0 = time.time()
    file_gb = os.path.getsize(regret_file) / (1024 ** 3)
    print(f"Loading regrets ({file_gb:.1f} GB)...", flush=True)
    n_loaded = bp.bp_load_regrets(solver, regret_file.encode("utf-8"))
    timings["load_regrets"] = time.time() - t0
    n_is = bp.bp_num_info_sets(solver)
    print(f"[{timings['load_regrets']:.2f}s] Loaded {n_loaded:,} entries ({n_is:,} info sets)", flush=True)

    if n_loaded <= 0:
        print("FATAL: no entries loaded")
        bp.bp_free(solver)
        sys.exit(1)

    # Query export size
    t0 = time.time()
    needed = ctypes.c_size_t(0)
    bp.bp_export_strategies(solver, None, 0, ctypes.byref(needed))
    strat_size = needed.value
    print(f"Export buffer: {strat_size / (1024**2):.1f} MB", flush=True)

    # Export strategies
    strat_buf = (ctypes.c_char * strat_size)()
    written = ctypes.c_size_t(0)
    print("Exporting strategies...", flush=True)
    bp.bp_export_strategies(solver, strat_buf, strat_size, ctypes.byref(written))
    strat_data = bytes(strat_buf[:written.value])
    timings["export_strategies"] = time.time() - t0
    print(f"[{timings['export_strategies']:.2f}s] Exported {written.value / (1024**2):.1f} MB", flush=True)

    # Free solver before compress (reclaim ~60 GB)
    bp.bp_free(solver)
    del buf, strat_buf
    print("Solver freed, starting compression...", flush=True)

    # LZMA compress
    t0 = time.time()
    compressed = lzma.compress(strat_data, preset=1)
    timings["lzma_compress"] = time.time() - t0
    print(f"[{timings['lzma_compress']:.2f}s] Compressed: {len(compressed) / (1024**2):.1f} MB", flush=True)

    # Write BPS3 file
    meta = {
        "type": "unified_blueprint",
        "num_players": NUM_PLAYERS,
        "blinds": [SMALL_BLIND, BIG_BLIND],
        "initial_stack": INITIAL_STACK,
        "preflop_bet_sizes": PREFLOP_BET_SIZES,
        "postflop_bet_sizes": POSTFLOP_BET_SIZES,
        "iterations": 200000000,
        "num_info_sets": n_is,
        "preflop_buckets": 169,
        "postflop_buckets": 200,
        "checkpoint": "iter_200000000",
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")

    os.makedirs(output_dir, exist_ok=True)
    bps_path = os.path.join(output_dir, "unified_blueprint.bps")
    with open(bps_path, "wb") as f:
        f.write(b"BPS3")
        f.write(struct.pack("<QI", len(compressed), len(meta_bytes)))
        f.write(compressed)
        f.write(meta_bytes)

    bps_mb = os.path.getsize(bps_path) / (1024 * 1024)
    print(f"Wrote {bps_path} ({bps_mb:.1f} MB)", flush=True)

    # Upload to S3
    if s3_bucket:
        t0 = time.time()
        s3_uri = f"s3://{s3_bucket}/unified_blueprint.bps"
        print(f"Uploading to {s3_uri}...", flush=True)
        subprocess.run(["aws", "s3", "cp", bps_path, s3_uri, "--quiet"], check=True)
        timings["s3_upload"] = time.time() - t0
        print(f"[{timings['s3_upload']:.2f}s] Uploaded", flush=True)

    timings["total"] = time.time() - t_total

    # Benchmark summary
    print()
    print("=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    for step, secs in timings.items():
        print(f"  {step:25s}  {secs:8.2f}s")
    print(f"  {'TOTAL':25s}  {timings['total']:8.2f}s  ({timings['total']/60:.1f} min)")
    print("=" * 60, flush=True)

    bench_path = os.path.join(output_dir, "export_benchmark.json")
    with open(bench_path, "w") as f:
        json.dump(timings, f, indent=2)
    if s3_bucket:
        subprocess.run(["aws", "s3", "cp", bench_path, f"s3://{s3_bucket}/export_benchmark.json", "--quiet"], check=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
