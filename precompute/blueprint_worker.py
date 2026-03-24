#!/usr/bin/env python3
"""Blueprint worker: runs on each EC2 instance to solve assigned flop textures.

Each worker:
1. Downloads its texture assignment from S3
2. For each texture: compute EHS buckets, run 6-player MCCFR, save results
3. Uploads results to S3

Usage:
    python blueprint_worker.py --worker-id 0 --total-workers 20 \
        --s3-bucket poker-solver-blueprints --iterations 100000
"""

import argparse
import ctypes
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

from precompute.solve_scenarios import generate_all_textures

# ── Constants ────────────────────────────────────────────────────────────

BP_MAX_HANDS = 1326
BP_MAX_ACTIONS = 8
NUM_PLAYERS = 6
SCALE = 100

# Pluribus-style bet sizes for postflop blueprint
FLOP_BET_SIZES = [0.5, 1.0]       # 50% pot, pot
TURN_BET_SIZES = [0.5, 1.0]
RIVER_BET_SIZES = [0.5, 1.0]
# Simplified: use same for all streets in blueprint (Pluribus uses 1-14 preflop, 3 post)
BLUEPRINT_BET_SIZES = [0.5, 1.0]

# Starting conditions (100bb deep, 6-max)
STARTING_POT = 650     # ~6.5 BB in chips (SRP average)
EFFECTIVE_STACK = 9750  # ~97.5 BB

# Card encoding
RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_to_int(s):
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])


def int_to_card(i):
    return RANKS[i // 4] + SUITS[i % 4]


# ── DLL loading ──────────────────────────────────────────────────────────

def load_dlls(build_dir):
    """Load mccfr_blueprint.dll and card_abstraction.dll."""
    bp_path = os.path.join(build_dir, 'mccfr_blueprint.so')
    ca_path = os.path.join(build_dir, 'card_abstraction.so')

    # Try .dll on Windows, .so on Linux
    if not os.path.exists(bp_path):
        bp_path = os.path.join(build_dir, 'mccfr_blueprint.dll')
    if not os.path.exists(ca_path):
        ca_path = os.path.join(build_dir, 'card_abstraction.dll')

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


# ── Config struct (must match BPConfig in C) ─────────────────────────────

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


# ── Solve one flop texture ───────────────────────────────────────────────

def solve_texture(bp_lib, ca_lib, texture_key, board_strs, num_buckets,
                  iterations, num_threads, output_dir):
    """Solve one flop texture with 6-player MCCFR.

    Returns dict with strategy data, or None on failure.
    """
    t_start = time.time()

    # Parse board
    flop_ints = [card_to_int(c) for c in board_strs]

    # Generate all possible hands for this flop
    HandsType = (ctypes.c_int * 2) * 1326
    all_hands = HandsType()
    n_all = ca_lib.ca_generate_hands(
        (ctypes.c_int * 3)(*flop_ints), 3, all_hands
    )

    if n_all <= 0:
        print(f"  [WARN] No hands for texture {texture_key}")
        return None

    # Compute EHS for flop
    ehs = (ctypes.c_float * n_all)()
    ca_lib.ca_compute_ehs(
        (ctypes.c_int * 3)(*flop_ints), 3,
        all_hands, n_all, 500,  # 500 samples per hand
        ehs
    )

    # Assign buckets
    buckets = (ctypes.c_int * n_all)()
    actual_buckets = ca_lib.ca_assign_buckets(ehs, n_all, num_buckets, buckets)

    # Cap hands at BP_MAX_HANDS per player
    nh = min(n_all, BP_MAX_HANDS)

    # Initialize solver
    buf = (ctypes.c_char * 524288)()  # large buffer for BPSolver struct
    solver = ctypes.cast(buf, ctypes.c_void_p)

    HT = ((ctypes.c_int * 2) * BP_MAX_HANDS) * 6
    WT = (ctypes.c_float * BP_MAX_HANDS) * 6
    NT = ctypes.c_int * 6
    ch, cw, cn = HT(), WT(), NT()

    # All 6 players get the same hand pool (Pluribus: all players can hold any hand)
    for p in range(NUM_PLAYERS):
        cn[p] = nh
        for h in range(nh):
            ch[p][h][0] = all_hands[h][0]
            ch[p][h][1] = all_hands[h][1]
            cw[p][h] = 1.0

    bs = BLUEPRINT_BET_SIZES
    c_bs = (ctypes.c_float * len(bs))(*bs)

    # Config
    config = BPConfig()
    bp_lib.bp_default_config(ctypes.byref(config))
    config.num_threads = num_threads
    config.hash_table_size = (1 << 22)  # 4M slots per texture (enough for ~2M info sets)

    # Scale Pluribus timing to our iteration count
    # Pluribus: 400 min discount out of ~11500 min total
    # Our ratio: discount for first 3.5% of iterations
    config.discount_stop_iter = max(iterations * 35 // 1000, 1000)
    config.discount_interval = max(config.discount_stop_iter // 40, 100)
    config.prune_start_iter = max(iterations * 17 // 1000, 500)
    config.strategy_interval = 10000

    ret = bp_lib.bp_init_ex(
        solver, NUM_PLAYERS,
        (ctypes.c_int * 3)(*flop_ints),
        ch, cw, cn,
        STARTING_POT, EFFECTIVE_STACK,
        c_bs, len(bs),
        ctypes.byref(config)
    )
    if ret != 0:
        print(f"  [ERROR] bp_init_ex failed for {texture_key}")
        return None

    # Set buckets for flop street (street=1)
    BM = (ctypes.c_int * BP_MAX_HANDS) * 6
    NB = ctypes.c_int * 6
    bm, nb = BM(), NB()
    for p in range(NUM_PLAYERS):
        nb[p] = actual_buckets
        for h in range(nh):
            bm[p][h] = buckets[h]

    bp_lib.bp_set_buckets(solver, 1, bm, nb)

    # Also set same buckets for turn/river (approximate — ideally re-bucket per board)
    bp_lib.bp_set_buckets(solver, 2, bm, nb)
    bp_lib.bp_set_buckets(solver, 3, bm, nb)

    # Solve
    bp_lib.bp_solve(solver, iterations)

    n_is = bp_lib.bp_num_info_sets(solver)
    elapsed = time.time() - t_start

    # Extract root strategies for all buckets
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
            root_strategies[b] = [float(strat_out[a]) for a in range(na)]

    bp_lib.bp_free(solver)

    # Save result
    result = {
        "texture": texture_key,
        "board": board_strs,
        "flop_ints": flop_ints,
        "num_hands": nh,
        "num_buckets": actual_buckets,
        "num_info_sets": n_is,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "root_strategies": root_strategies,
        "bet_sizes": bs,
    }

    out_path = os.path.join(output_dir, f"{texture_key}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Blueprint worker")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--total-workers", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--num-buckets", type=int, default=200)
    parser.add_argument("--num-threads", type=int, default=0,
                        help="OpenMP threads (0=auto)")
    parser.add_argument("--output-dir", default="/tmp/blueprint_output")
    parser.add_argument("--s3-bucket", default="")
    parser.add_argument("--build-dir", default="build")
    args = parser.parse_args()

    print(f"=== Blueprint Worker {args.worker_id}/{args.total_workers} ===")
    print(f"Iterations: {args.iterations}, Buckets: {args.num_buckets}")
    print(f"Threads: {args.num_threads or 'auto'}")

    # Load DLLs
    bp_lib, ca_lib = load_dlls(args.build_dir)
    print("DLLs loaded.")

    # Get all textures and partition
    all_textures = generate_all_textures()
    print(f"Total textures: {len(all_textures)}")

    # Assign textures to this worker (round-robin)
    my_textures = [
        (key, board) for i, (key, board) in enumerate(all_textures)
        if i % args.total_workers == args.worker_id
    ]
    print(f"Worker {args.worker_id} assigned {len(my_textures)} textures")

    # Solve each texture
    results = []
    for idx, (tex_key, board) in enumerate(my_textures):
        print(f"\n[{idx+1}/{len(my_textures)}] Solving {tex_key} ({' '.join(board)})")

        result = solve_texture(
            bp_lib, ca_lib,
            tex_key, board,
            args.num_buckets,
            args.iterations,
            args.num_threads if args.num_threads > 0 else 0,
            args.output_dir,
        )

        if result:
            results.append({
                "texture": tex_key,
                "info_sets": result["num_info_sets"],
                "elapsed": result["elapsed_seconds"],
            })
            print(f"  Done: {result['num_info_sets']:,} info sets in {result['elapsed_seconds']:.1f}s")
        else:
            print(f"  FAILED")

    # Summary
    print(f"\n=== Worker {args.worker_id} Complete ===")
    print(f"Solved: {len(results)}/{len(my_textures)} textures")
    total_is = sum(r["info_sets"] for r in results)
    total_time = sum(r["elapsed"] for r in results)
    print(f"Total info sets: {total_is:,}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Upload to S3 if specified
    if args.s3_bucket:
        import subprocess
        s3_prefix = f"s3://{args.s3_bucket}/worker-{args.worker_id}/"
        print(f"\nUploading to {s3_prefix}...")
        subprocess.run([
            "aws", "s3", "sync",
            args.output_dir, s3_prefix,
            "--quiet"
        ], check=True)
        print("Upload complete.")


if __name__ == "__main__":
    main()
