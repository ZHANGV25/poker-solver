#!/usr/bin/env python3
"""One-shot BPS export: load regrets checkpoint, export strategies, upload to S3.

This script does NOT run any MCCFR iterations — it only:
  1. Initializes the solver (allocates hash table + precomputes texture buckets)
  2. Loads regrets from a checkpoint file
  3. Exports strategies to BPS3 format (regret-matched, uint8-quantized)
  4. Uploads the .bps file to S3

Usage:
    python3 export_bps_only.py \
        --regret-file /opt/blueprint_unified/regrets_200M.bin \
        --output-dir /opt/blueprint_unified \
        --s3-bucket poker-blueprint-unified \
        --build-dir build
"""

import argparse
import ctypes
import json
import lzma
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

# ── Constants (must match training run exactly) ─────────────────
NUM_PLAYERS = 6
SMALL_BLIND = 50
BIG_BLIND = 100
INITIAL_STACK = 10000
PREFLOP_BET_SIZES = [0.5, 1.0, 2.0, 3.0]
POSTFLOP_BET_SIZES = [0.5, 1.0, 2.0]

# ── DLL loading ─────────────────────────────────────────────────

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


def load_bp_dll(build_dir):
    """Load the MCCFR blueprint shared library."""
    for ext in ['so', 'dll']:
        path = os.path.join(build_dir, f'mccfr_blueprint.{ext}')
        if os.path.exists(path):
            bp = ctypes.CDLL(path)
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
            return bp
    raise FileNotFoundError(f"mccfr_blueprint shared library not found in {build_dir}")


def s3_upload(local_path, s3_uri, max_attempts=3, delay=5):
    """Upload to S3 with retries."""
    import subprocess
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(
                ["aws", "s3", "cp", local_path, s3_uri, "--quiet"],
                check=True, capture_output=True,
            )
            return
        except subprocess.CalledProcessError as e:
            if attempt < max_attempts:
                print(f"  S3 upload attempt {attempt} failed, retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description="Export BPS strategies from regret checkpoint")
    parser.add_argument("--regret-file", required=True, help="Path to regrets .bin file")
    parser.add_argument("--output-dir", default="/opt/blueprint_unified")
    parser.add_argument("--s3-bucket", default="")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--hash-size", type=int, default=1 << 30,
                        help="Hash table slots (default: 1B = 1<<30)")
    args = parser.parse_args()

    timings = {}
    t_total = time.time()

    # ── Step 1: Load DLL ────────────────────────────────────────
    print("=" * 60)
    print("  BPS Export Tool — Benchmarked")
    print("=" * 60)

    t0 = time.time()
    bp_lib = load_bp_dll(args.build_dir)
    timings['dll_load'] = time.time() - t0
    print(f"[{timings['dll_load']:.2f}s] DLL loaded")

    # ── Step 2: Initialize solver (allocates hash table + texture cache) ──
    t0 = time.time()
    config = BPConfig()
    bp_lib.bp_default_config(ctypes.byref(config))
    config.num_threads = 1  # no solving, just export
    config.include_preflop = 1
    config.hash_table_size = args.hash_size

    buf = (ctypes.c_char * (2 * 1024 * 1024))()  # 2MB — BPSolver struct is ~250KB but needs headroom
    solver = ctypes.cast(buf, ctypes.c_void_p)

    c_postflop = (ctypes.c_float * len(POSTFLOP_BET_SIZES))(*POSTFLOP_BET_SIZES)
    c_preflop = (ctypes.c_float * len(PREFLOP_BET_SIZES))(*PREFLOP_BET_SIZES)

    ret = bp_lib.bp_init_unified(
        solver, NUM_PLAYERS,
        SMALL_BLIND, BIG_BLIND, INITIAL_STACK,
        c_postflop, len(POSTFLOP_BET_SIZES),
        c_preflop, len(PREFLOP_BET_SIZES),
        ctypes.byref(config)
    )
    if ret != 0:
        print(f"FATAL: bp_init_unified returned {ret}")
        sys.exit(1)
    timings['init'] = time.time() - t0
    print(f"[{timings['init']:.2f}s] Solver initialized (hash_size={args.hash_size:,})")

    # ── Step 3: Load regret checkpoint ──────────────────────────
    t0 = time.time()
    regret_path = args.regret_file
    if not os.path.exists(regret_path):
        print(f"FATAL: regret file not found: {regret_path}")
        sys.exit(1)

    file_size_gb = os.path.getsize(regret_path) / (1024 ** 3)
    print(f"Loading regrets from {regret_path} ({file_size_gb:.1f} GB)...")

    n_loaded = bp_lib.bp_load_regrets(solver, regret_path.encode('utf-8'))
    timings['load_regrets'] = time.time() - t0
    n_is = bp_lib.bp_num_info_sets(solver)
    print(f"[{timings['load_regrets']:.2f}s] Loaded {n_loaded:,} entries, {n_is:,} info sets")

    if n_loaded <= 0:
        print("FATAL: no entries loaded from regret file")
        bp_lib.bp_free(solver)
        sys.exit(1)

    # ── Step 4: Export strategies to BPS3 ───────────────────────
    t0 = time.time()

    # First call: query required buffer size
    needed = ctypes.c_size_t(0)
    bp_lib.bp_export_strategies(solver, None, 0, ctypes.byref(needed))
    strat_size = needed.value
    print(f"Strategy buffer size: {strat_size / (1024*1024):.1f} MB")

    # Second call: export
    strat_buf = (ctypes.c_char * strat_size)()
    written = ctypes.c_size_t(0)
    bp_lib.bp_export_strategies(solver, strat_buf, strat_size, ctypes.byref(written))
    strat_data = bytes(strat_buf[:written.value])
    timings['export_strategies'] = time.time() - t0
    print(f"[{timings['export_strategies']:.2f}s] Exported {written.value / (1024*1024):.1f} MB raw strategies")

    # ── Step 5: LZMA compress + write BPS3 ──────────────────────
    t0 = time.time()
    compressed = lzma.compress(strat_data, preset=1)
    timings['lzma_compress'] = time.time() - t0
    print(f"[{timings['lzma_compress']:.2f}s] LZMA compressed: {len(compressed) / (1024*1024):.1f} MB")

    # Build metadata
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
    meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')

    os.makedirs(args.output_dir, exist_ok=True)
    bps_path = os.path.join(args.output_dir, "unified_blueprint.bps")
    with open(bps_path, 'wb') as f:
        f.write(b'BPS3')
        f.write(struct.pack('<QI', len(compressed), len(meta_bytes)))
        f.write(compressed)
        f.write(meta_bytes)

    bps_size_mb = os.path.getsize(bps_path) / (1024 * 1024)
    print(f"Wrote {bps_path} ({bps_size_mb:.1f} MB)")

    # ── Step 6: Upload to S3 ────────────────────────────────────
    if args.s3_bucket:
        t0 = time.time()
        s3_uri = f"s3://{args.s3_bucket}/unified_blueprint.bps"
        print(f"Uploading to {s3_uri}...")
        s3_upload(bps_path, s3_uri)
        timings['s3_upload'] = time.time() - t0
        print(f"[{timings['s3_upload']:.2f}s] Uploaded to S3")

    # ── Cleanup ─────────────────────────────────────────────────
    bp_lib.bp_free(solver)
    timings['total'] = time.time() - t_total

    # ── Benchmark summary ───────────────────────────────────────
    print()
    print("=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    for step, secs in timings.items():
        print(f"  {step:25s}  {secs:8.2f}s")
    print(f"  {'':25s}  {'':8s}")
    print(f"  {'TOTAL':25s}  {timings['total']:8.2f}s  ({timings['total']/60:.1f} min)")
    print("=" * 60)

    # Write benchmark to file for retrieval
    bench_path = os.path.join(args.output_dir, "export_benchmark.json")
    with open(bench_path, 'w') as f:
        json.dump(timings, f, indent=2)
    if args.s3_bucket:
        s3_upload(bench_path, f"s3://{args.s3_bucket}/export_benchmark.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
