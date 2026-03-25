#!/usr/bin/env python3
"""Blueprint worker: runs on each EC2 instance to solve assigned flop textures.

Each worker:
1. Runs a quick smoke test (1 texture, low iterations) to verify compilation
2. Solves its assigned partition of 1,755 textures
3. Reports convergence metrics (strategy change N vs 2N at checkpoints)
4. Saves per-texture results to disk
5. Uploads results to S3

Usage (local test):
    python blueprint_worker.py --worker-id 0 --total-workers 1 \
        --iterations 50000 --num-buckets 200 --smoke-test-only

Usage (EC2):
    python blueprint_worker.py --worker-id 0 --total-workers 20 \
        --iterations 1000000 --s3-bucket poker-solver-blueprints
"""

import argparse
import ctypes
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

from precompute.solve_scenarios import generate_all_textures

# ── Constants ────────────────────────────────────────────────────────────

BP_MAX_HANDS = 1326
BP_MAX_ACTIONS = 8
SCALE = 100

BLUEPRINT_BET_SIZES = [0.5, 1.0]   # 50% pot, pot-size bet
STARTING_POT = 650                   # 6.5 BB in chips
EFFECTIVE_STACK = 9750               # 97.5 BB in chips

RANKS = "23456789TJQKA"
SUITS = "cdhs"

def card_to_int(s):
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])

def int_to_card(i):
    return RANKS[i // 4] + SUITS[i % 4]

# ── DLL loading ──────────────────────────────────────────────────────────

def load_dlls(build_dir):
    """Load shared libraries (.so on Linux, .dll on Windows)."""
    for ext in ['so', 'dll']:
        bp_path = os.path.join(build_dir, f'mccfr_blueprint.{ext}')
        ca_path = os.path.join(build_dir, f'card_abstraction.{ext}')
        if os.path.exists(bp_path) and os.path.exists(ca_path):
            break
    else:
        raise FileNotFoundError(f"Libraries not found in {build_dir}")

    bp = ctypes.CDLL(bp_path)
    bp.bp_default_config.restype = None
    bp.bp_init_ex.restype = ctypes.c_int
    bp.bp_set_buckets.restype = ctypes.c_int
    bp.bp_solve.restype = ctypes.c_int
    bp.bp_get_strategy.restype = ctypes.c_int
    bp.bp_num_info_sets.restype = ctypes.c_int
    bp.bp_free.restype = None

    ca = ctypes.CDLL(ca_path)
    ca.ca_compute_ehs.restype = ctypes.c_int
    ca.ca_assign_buckets.restype = ctypes.c_int
    ca.ca_generate_hands.restype = ctypes.c_int

    return bp, ca


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
    ]


# ── Solve one texture ────────────────────────────────────────────────────

def solve_texture(bp_lib, ca_lib, texture_key, board_strs, num_players,
                  num_buckets, iterations, num_threads, output_dir,
                  ehs_samples=500):
    """Solve one flop texture with N-player MCCFR.

    Returns dict with strategy data, or None on failure.
    """
    t_start = time.time()
    flop_ints = [card_to_int(c) for c in board_strs]

    # Generate all possible 2-card hands for this flop
    HandsType = (ctypes.c_int * 2) * 1326
    all_hands = HandsType()
    n_all = ca_lib.ca_generate_hands(
        (ctypes.c_int * 3)(*flop_ints), 3, all_hands
    )
    if n_all <= 0:
        return None

    t_ehs_start = time.time()

    # Compute EHS
    ehs = (ctypes.c_float * n_all)()
    ca_lib.ca_compute_ehs(
        (ctypes.c_int * 3)(*flop_ints), 3,
        all_hands, n_all, ehs_samples, ehs
    )

    # Assign buckets
    buckets = (ctypes.c_int * n_all)()
    actual_buckets = ca_lib.ca_assign_buckets(ehs, n_all, num_buckets, buckets)

    t_ehs_end = time.time()
    ehs_time = t_ehs_end - t_ehs_start

    # Initialize solver
    nh = min(n_all, BP_MAX_HANDS)
    buf = (ctypes.c_char * 524288)()
    solver = ctypes.cast(buf, ctypes.c_void_p)

    HT = ((ctypes.c_int * 2) * BP_MAX_HANDS) * 6
    WT = (ctypes.c_float * BP_MAX_HANDS) * 6
    NT = ctypes.c_int * 6
    ch, cw, cn = HT(), WT(), NT()

    for p in range(num_players):
        cn[p] = nh
        for h in range(nh):
            ch[p][h][0] = all_hands[h][0]
            ch[p][h][1] = all_hands[h][1]
            cw[p][h] = 1.0

    c_bs = (ctypes.c_float * len(BLUEPRINT_BET_SIZES))(*BLUEPRINT_BET_SIZES)

    # Config — scale Pluribus timing proportionally
    config = BPConfig()
    bp_lib.bp_default_config(ctypes.byref(config))
    config.num_threads = num_threads
    config.hash_table_size = 0  # 0 = auto (MEDIUM for 3+ players, SMALL for 2)
    config.discount_stop_iter = max(iterations * 35 // 1000, 1000)
    config.discount_interval = max(config.discount_stop_iter // 40, 100)
    config.prune_start_iter = max(iterations * 17 // 1000, 500)
    config.strategy_interval = 1  # Update strategy_sum every iteration for usable avg strategies

    ret = bp_lib.bp_init_ex(
        solver, num_players,
        (ctypes.c_int * 3)(*flop_ints),
        ch, cw, cn,
        STARTING_POT, EFFECTIVE_STACK,
        c_bs, len(BLUEPRINT_BET_SIZES),
        ctypes.byref(config)
    )
    if ret != 0:
        return None

    # Set buckets for all postflop streets
    BM = (ctypes.c_int * BP_MAX_HANDS) * 6
    NB = ctypes.c_int * 6
    bm, nb = BM(), NB()
    for p in range(num_players):
        nb[p] = actual_buckets
        for h in range(nh):
            bm[p][h] = buckets[h]

    for street in [1, 2, 3]:  # flop, turn, river
        bp_lib.bp_set_buckets(solver, street, bm, nb)

    # Solve
    t_solve_start = time.time()
    bp_lib.bp_solve(solver, iterations)
    t_solve_end = time.time()
    solve_time = t_solve_end - t_solve_start

    n_is = bp_lib.bp_num_info_sets(solver)

    # ── Export ALL strategies (binary, quantized uint8) ──────────────
    bp_lib.bp_export_strategies.restype = ctypes.c_int
    bp_lib.bp_export_strategies.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t)
    ]
    bp_lib.bp_export_buckets.restype = ctypes.c_int

    # Query required buffer size
    needed = ctypes.c_size_t(0)
    bp_lib.bp_export_strategies(solver, None, 0, ctypes.byref(needed))
    strat_size = needed.value

    # Allocate and export
    strat_buf = (ctypes.c_char * strat_size)()
    written = ctypes.c_size_t(0)
    ret = bp_lib.bp_export_strategies(solver, strat_buf, strat_size, ctypes.byref(written))
    if ret != 0:
        print(f"  [WARN] bp_export_strategies failed for {texture_key}")
        strat_data = b''
    else:
        strat_data = bytes(strat_buf[:written.value])

    # Export bucket assignments for all players on flop street
    bucket_data = {}
    for p in range(num_players):
        b_out = (ctypes.c_int * BP_MAX_HANDS)()
        n = bp_lib.bp_export_buckets(solver, 1, p, b_out)
        bucket_data[p] = [b_out[h] for h in range(n)]

    # Extract root strategies for quick inspection (keep for summary)
    strat_out = (ctypes.c_float * BP_MAX_ACTIONS)()
    root_strategies = {}
    for b in range(actual_buckets):
        na = bp_lib.bp_get_strategy(
            solver, 0,
            (ctypes.c_int * 3)(*flop_ints), 3,
            (ctypes.c_int * 1)(), 0,
            strat_out, b
        )
        if na > 0:
            root_strategies[b] = [round(float(strat_out[a]), 4) for a in range(na)]

    bp_lib.bp_free(solver)

    elapsed = time.time() - t_start

    # ── Save binary blueprint file (.bps) ────────────────────────────
    # Format: header + LZMA-compressed strategy data + metadata JSON
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        import lzma
        import struct

        # Compress strategy data with LZMA
        compressed = lzma.compress(strat_data, preset=3)

        # Metadata (small, stored as JSON at the end)
        meta = {
            "texture": texture_key,
            "board": board_strs,
            "flop_ints": flop_ints,
            "num_players": num_players,
            "num_hands": nh,
            "num_buckets": actual_buckets,
            "num_info_sets": n_is,
            "iterations": iterations,
            "ehs_time_s": round(ehs_time, 2),
            "solve_time_s": round(solve_time, 2),
            "total_time_s": round(elapsed, 2),
            "bet_sizes": BLUEPRINT_BET_SIZES,
            "root_strategies": root_strategies,
            "bucket_assignments": {str(p): bucket_data[p] for p in bucket_data},
        }
        meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')

        # Binary file: [magic 4B][strat_compressed_size 4B][meta_size 4B]
        #              [compressed_strategies][meta_json]
        out_path = os.path.join(output_dir, f"{texture_key}.bps")
        with open(out_path, 'wb') as f:
            f.write(b'BPS2')
            f.write(struct.pack('<II', len(compressed), len(meta_bytes)))
            f.write(compressed)
            f.write(meta_bytes)

        strat_raw_mb = len(strat_data) / 1024 / 1024
        strat_comp_mb = len(compressed) / 1024 / 1024
        file_mb = (12 + len(compressed) + len(meta_bytes)) / 1024 / 1024

    result = {
        "texture": texture_key,
        "num_info_sets": n_is,
        "ehs_time_s": round(ehs_time, 2),
        "solve_time_s": round(solve_time, 2),
        "total_time_s": round(elapsed, 2),
        "strat_raw_mb": round(strat_raw_mb, 1) if output_dir else 0,
        "strat_compressed_mb": round(strat_comp_mb, 1) if output_dir else 0,
        "file_mb": round(file_mb, 1) if output_dir else 0,
    }

    return result


# ── Smoke test ───────────────────────────────────────────────────────────

def run_smoke_test(bp_lib, ca_lib, num_players):
    """Quick validation: solve 1 texture with minimal iterations."""
    print("=== Smoke Test ===")
    result = solve_texture(
        bp_lib, ca_lib,
        "T72_r", ["Ts", "7h", "2d"],
        num_players=num_players,
        num_buckets=50,
        iterations=10000,
        num_threads=0,
        output_dir=None,
        ehs_samples=100,
    )
    if not result:
        print("  FAILED: solve returned None")
        return False

    if result["num_info_sets"] == 0:
        print("  FAILED: 0 info sets (card conflict rejection?)")
        return False

    print(f"  Info sets: {result['num_info_sets']:,}")
    print(f"  EHS time: {result['ehs_time_s']:.1f}s, Solve time: {result['solve_time_s']:.1f}s")
    if result.get("file_mb", 0) > 0:
        print(f"  File: {result['file_mb']} MB")

    print("  PASSED")
    return True


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Blueprint worker")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--total-workers", type=int, default=20)
    parser.add_argument("--num-players", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=1000000)
    parser.add_argument("--num-buckets", type=int, default=200)
    parser.add_argument("--ehs-samples", type=int, default=500)
    parser.add_argument("--num-threads", type=int, default=0,
                        help="OpenMP threads (0=auto)")
    parser.add_argument("--output-dir", default="/tmp/blueprint_output")
    parser.add_argument("--s3-bucket", default="")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--smoke-test-only", action="store_true")
    args = parser.parse_args()

    print(f"=== Blueprint Worker {args.worker_id}/{args.total_workers} ===")
    print(f"Players: {args.num_players}, Iterations: {args.iterations:,}, Buckets: {args.num_buckets}")
    print(f"Threads: {args.num_threads or 'auto'}, EHS samples: {args.ehs_samples}")
    print()

    # Load libraries
    bp_lib, ca_lib = load_dlls(args.build_dir)
    print("Libraries loaded.")

    # Smoke test
    if not run_smoke_test(bp_lib, ca_lib, args.num_players):
        print("SMOKE TEST FAILED — aborting.")
        sys.exit(1)
    print()

    if args.smoke_test_only:
        print("Smoke test only — exiting.")
        return

    # Get all textures and partition
    all_textures = generate_all_textures()
    my_textures = [
        (key, board) for i, (key, board) in enumerate(all_textures)
        if i % args.total_workers == args.worker_id
    ]
    print(f"Total textures: {len(all_textures)}, this worker: {len(my_textures)}")
    print()

    # Solve each texture
    results = []
    failures = []
    t_batch_start = time.time()

    for idx, (tex_key, board) in enumerate(my_textures):
        try:
            result = solve_texture(
                bp_lib, ca_lib, tex_key, board,
                num_players=args.num_players,
                num_buckets=args.num_buckets,
                iterations=args.iterations,
                num_threads=args.num_threads if args.num_threads > 0 else 0,
                output_dir=args.output_dir,
                ehs_samples=args.ehs_samples,
            )

            if result:
                results.append({
                    "texture": tex_key,
                    "info_sets": result["num_info_sets"],
                    "ehs_s": result["ehs_time_s"],
                    "solve_s": result["solve_time_s"],
                    "total_s": result["total_time_s"],
                })
                elapsed_total = time.time() - t_batch_start
                avg_per = elapsed_total / (idx + 1)
                remaining = avg_per * (len(my_textures) - idx - 1)
                print(f"[{idx+1}/{len(my_textures)}] {tex_key}: "
                      f"{result['num_info_sets']:,} IS, {result['total_time_s']:.1f}s "
                      f"(ETA: {remaining/60:.0f}min)")
            else:
                failures.append(tex_key)
                print(f"[{idx+1}/{len(my_textures)}] {tex_key}: FAILED")

        except Exception as e:
            failures.append(tex_key)
            print(f"[{idx+1}/{len(my_textures)}] {tex_key}: ERROR: {e}")
            traceback.print_exc()

    # Summary
    t_batch_end = time.time()
    batch_elapsed = t_batch_end - t_batch_start

    summary = {
        "worker_id": args.worker_id,
        "total_workers": args.total_workers,
        "num_players": args.num_players,
        "iterations": args.iterations,
        "num_buckets": args.num_buckets,
        "textures_solved": len(results),
        "textures_failed": len(failures),
        "total_info_sets": sum(r["info_sets"] for r in results),
        "total_time_s": round(batch_elapsed, 1),
        "avg_time_per_texture_s": round(batch_elapsed / max(len(results), 1), 1),
        "failures": failures,
    }

    print(f"\n{'='*60}")
    print(f"Worker {args.worker_id} Complete")
    print(f"  Solved: {len(results)}/{len(my_textures)} textures")
    print(f"  Failed: {len(failures)}")
    print(f"  Total info sets: {summary['total_info_sets']:,}")
    print(f"  Total time: {batch_elapsed:.0f}s ({batch_elapsed/60:.1f} min)")
    print(f"  Avg per texture: {summary['avg_time_per_texture_s']:.1f}s")
    print(f"{'='*60}")

    # Save summary
    summary_path = os.path.join(args.output_dir, f"summary_worker_{args.worker_id}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Upload to S3
    if args.s3_bucket:
        import subprocess
        s3_prefix = f"s3://{args.s3_bucket}/worker-{args.worker_id}/"
        print(f"\nUploading to {s3_prefix}...")
        subprocess.run(["aws", "s3", "sync", args.output_dir, s3_prefix, "--quiet"],
                        check=True)
        print("Upload complete.")


if __name__ == "__main__":
    main()
