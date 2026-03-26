#!/usr/bin/env python3
"""Blueprint v2 worker: preflop-filtered 2-player solves.

Key differences from blueprint_worker.py (v1):
  - 2 players instead of 6 — per-player hand arrays from preflop ranges
  - 27 scenarios × 1,755 textures = 47,385 work items (partitioned across workers)
  - Adaptive bucket count: min(200, max_hands // 3), identity if max_hands ≤ 50
  - 20M iterations default (vs 1M in v1)
  - 4M hash table slots (vs auto/64M)
  - Incremental S3 upload per solve

Usage (local test):
    python blueprint_worker_v2.py --worker-id 0 --total-workers 1 \
        --iterations 100000 --ranges /path/to/ranges.json --smoke-test-only

Usage (EC2):
    python blueprint_worker_v2.py --worker-id 0 --total-workers 30 \
        --iterations 20000000 --ranges /path/to/ranges.json \
        --s3-bucket poker-blueprint-v2
"""

import argparse
import ctypes
import json
import lzma
import os
import struct
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

from precompute.solve_scenarios import generate_all_textures
from precompute.scenario_matrix import build_scenario_matrix
from precompute.range_parser import get_range_hands

# ── Constants ─────────────────────────────────────────────────────────────

BP_MAX_HANDS = 1326
BP_MAX_ACTIONS = 8
NUM_PLAYERS = 2

BLUEPRINT_BET_SIZES = [0.5, 1.0, 2.0]  # 50% pot, pot, 2x pot (Pluribus: up to 14 first round, 3 later)

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_to_int(s):
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])


def int_to_card(i):
    return RANKS[i // 4] + SUITS[i % 4]


# ── DLL loading (reused from blueprint_worker.py) ────────────────────────

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


# ── Per-player EHS + bucketing ────────────────────────────────────────────

def compute_player_ehs_buckets(ca_lib, flop_ints, player_hands, num_buckets,
                               ehs_samples=500, use_kmeans=True):
    """Compute EHS and assign buckets for a specific player's hand set.

    Args:
        ca_lib: card_abstraction library
        flop_ints: [c0, c1, c2] board card ints
        player_hands: list of (card0, card1) tuples
        num_buckets: target bucket count
        ehs_samples: Monte Carlo samples per hand
        use_kmeans: if True, use k-means on [EHS, pos_potential, neg_potential]
                    (Pluribus-style, per Johanson et al. 2013).
                    If False, fall back to EHS percentile bucketing.

    Returns:
        (ehs_array, bucket_array, actual_buckets) — ctypes arrays
    """
    n = len(player_hands)

    # Pack hands into ctypes array
    HandsType = (ctypes.c_int * 2) * n
    c_hands = HandsType()
    for i, (c0, c1) in enumerate(player_hands):
        c_hands[i][0] = c0
        c_hands[i][1] = c1

    # Compute EHS (always needed for metadata)
    ehs = (ctypes.c_float * n)()
    ca_lib.ca_compute_ehs(
        (ctypes.c_int * 3)(*flop_ints), 3,
        c_hands, n, ehs_samples, ehs
    )

    buckets = (ctypes.c_int * n)()

    if use_kmeans and hasattr(ca_lib, 'ca_assign_buckets_kmeans'):
        # Pluribus-style k-means on [EHS, positive_potential, negative_potential]
        ca_lib.ca_assign_buckets_kmeans.restype = ctypes.c_int
        actual = ca_lib.ca_assign_buckets_kmeans(
            (ctypes.c_int * 3)(*flop_ints), 3,
            c_hands, n, num_buckets, ehs_samples, buckets
        )
    else:
        # Fallback: EHS percentile bucketing
        actual = ca_lib.ca_assign_buckets(ehs, n, num_buckets, buckets)

    return ehs, buckets, actual


# ── Solve one (scenario, texture) pair ────────────────────────────────────

def solve_texture_v2(bp_lib, ca_lib, scenario_id, scenario, texture_key,
                     board_strs, iterations, num_threads, output_dir,
                     ehs_samples=500):
    """Solve one flop texture for one scenario with 2-player preflop-filtered MCCFR.

    Returns dict with strategy data, or None on failure.
    """
    t_start = time.time()
    flop_ints = [card_to_int(c) for c in board_strs]

    # Get per-player hands from preflop ranges (board-filtered)
    oop_hands = get_range_hands(scenario["oop_range"], flop_ints)
    ip_hands = get_range_hands(scenario["ip_range"], flop_ints)

    if not oop_hands or not ip_hands:
        return None

    player_hands = [oop_hands, ip_hands]
    max_hands = max(len(oop_hands), len(ip_hands))

    # Adaptive bucket count
    if max_hands <= 50:
        num_buckets = max_hands  # identity bucketing
    else:
        num_buckets = min(200, max_hands // 3)
    num_buckets = max(num_buckets, 2)  # at least 2 buckets

    # Compute EHS and buckets per player
    t_ehs_start = time.time()
    player_ehs = []
    player_buckets = []
    actual_buckets_list = []

    for p in range(NUM_PLAYERS):
        ehs, bkts, actual = compute_player_ehs_buckets(
            ca_lib, flop_ints, player_hands[p], num_buckets, ehs_samples
        )
        player_ehs.append(ehs)
        player_buckets.append(bkts)
        actual_buckets_list.append(actual)

    # Use the max actual bucket count across players
    actual_buckets = max(actual_buckets_list)
    t_ehs_end = time.time()
    ehs_time = t_ehs_end - t_ehs_start

    # Initialize solver with per-player hand arrays
    buf = (ctypes.c_char * 524288)()
    solver = ctypes.cast(buf, ctypes.c_void_p)

    HT = ((ctypes.c_int * 2) * BP_MAX_HANDS) * 6
    WT = (ctypes.c_float * BP_MAX_HANDS) * 6
    NT = ctypes.c_int * 6
    ch, cw, cn = HT(), WT(), NT()

    for p in range(NUM_PLAYERS):
        nh = len(player_hands[p])
        cn[p] = nh
        for h in range(nh):
            ch[p][h][0] = player_hands[p][h][0]
            ch[p][h][1] = player_hands[p][h][1]
            cw[p][h] = 1.0

    c_bs = (ctypes.c_float * len(BLUEPRINT_BET_SIZES))(*BLUEPRINT_BET_SIZES)

    # Config
    config = BPConfig()
    bp_lib.bp_default_config(ctypes.byref(config))
    config.num_threads = num_threads
    config.hash_table_size = 4 * 1024 * 1024  # 4M slots (~160MB) instead of 64M
    config.discount_stop_iter = max(iterations * 35 // 1000, 1000)
    config.discount_interval = max(config.discount_stop_iter // 40, 100)
    config.prune_start_iter = max(iterations * 17 // 1000, 500)
    config.strategy_interval = 1

    ret = bp_lib.bp_init_ex(
        solver, NUM_PLAYERS,
        (ctypes.c_int * 3)(*flop_ints),
        ch, cw, cn,
        scenario["starting_pot"], scenario["effective_stack"],
        c_bs, len(BLUEPRINT_BET_SIZES),
        ctypes.byref(config)
    )
    if ret != 0:
        return None

    # Set per-player buckets for all postflop streets
    BM = (ctypes.c_int * BP_MAX_HANDS) * 6
    NB = ctypes.c_int * 6
    bm, nb = BM(), NB()
    for p in range(NUM_PLAYERS):
        nb[p] = actual_buckets_list[p]
        nh = len(player_hands[p])
        for h in range(nh):
            bm[p][h] = player_buckets[p][h]

    for street in [1, 2, 3]:
        bp_lib.bp_set_buckets(solver, street, bm, nb)

    # Solve
    t_solve_start = time.time()
    bp_lib.bp_solve(solver, iterations)
    t_solve_end = time.time()
    solve_time = t_solve_end - t_solve_start

    n_is = bp_lib.bp_num_info_sets(solver)

    # Export strategies (binary, quantized uint8)
    bp_lib.bp_export_strategies.restype = ctypes.c_int
    bp_lib.bp_export_strategies.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t)
    ]
    bp_lib.bp_export_buckets.restype = ctypes.c_int

    needed = ctypes.c_size_t(0)
    bp_lib.bp_export_strategies(solver, None, 0, ctypes.byref(needed))
    strat_size = needed.value

    strat_buf = (ctypes.c_char * strat_size)()
    written = ctypes.c_size_t(0)
    ret = bp_lib.bp_export_strategies(solver, strat_buf, strat_size, ctypes.byref(written))
    if ret != 0:
        print(f"  [WARN] bp_export_strategies failed for {scenario_id}/{texture_key}")
        strat_data = b''
    else:
        strat_data = bytes(strat_buf[:written.value])

    # Export bucket assignments
    bucket_data = {}
    for p in range(NUM_PLAYERS):
        b_out = (ctypes.c_int * BP_MAX_HANDS)()
        n = bp_lib.bp_export_buckets(solver, 1, p, b_out)
        bucket_data[p] = [b_out[h] for h in range(n)]

    # Extract root strategies
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

    # Save .bps file
    strat_raw_mb = 0
    strat_comp_mb = 0
    file_mb = 0

    if output_dir:
        scenario_dir = os.path.join(output_dir, scenario_id)
        os.makedirs(scenario_dir, exist_ok=True)

        compressed = lzma.compress(strat_data, preset=1)

        meta = {
            "texture": texture_key,
            "board": board_strs,
            "flop_ints": flop_ints,
            "scenario_id": scenario_id,
            "scenario_type": scenario["scenario_type"],
            "oop_pos": scenario["oop_pos"],
            "ip_pos": scenario["ip_pos"],
            "num_players": NUM_PLAYERS,
            "oop_num_hands": len(oop_hands),
            "ip_num_hands": len(ip_hands),
            "num_buckets": actual_buckets,
            "num_info_sets": n_is,
            "iterations": iterations,
            "starting_pot": scenario["starting_pot"],
            "effective_stack": scenario["effective_stack"],
            "ehs_time_s": round(ehs_time, 2),
            "solve_time_s": round(solve_time, 2),
            "total_time_s": round(elapsed, 2),
            "bet_sizes": BLUEPRINT_BET_SIZES,
            "root_strategies": root_strategies,
            "bucket_assignments": {str(p): bucket_data[p] for p in bucket_data},
            "oop_hands": [(int_to_card(c0), int_to_card(c1)) for c0, c1 in oop_hands],
            "ip_hands": [(int_to_card(c0), int_to_card(c1)) for c0, c1 in ip_hands],
        }
        meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')

        out_path = os.path.join(scenario_dir, f"{texture_key}.bps")
        with open(out_path, 'wb') as f:
            f.write(b'BPS2')
            f.write(struct.pack('<II', len(compressed), len(meta_bytes)))
            f.write(compressed)
            f.write(meta_bytes)

        strat_raw_mb = len(strat_data) / 1024 / 1024
        strat_comp_mb = len(compressed) / 1024 / 1024
        file_mb = (12 + len(compressed) + len(meta_bytes)) / 1024 / 1024

    # Count non-uniform root strategies
    non_uniform = 0
    for b, strat in root_strategies.items():
        if len(strat) > 1:
            max_freq = max(strat)
            if max_freq < 0.99:
                non_uniform += 1
    total_root = len(root_strategies)
    pct = (non_uniform / total_root * 100) if total_root > 0 else 0

    return {
        "texture": texture_key,
        "scenario_id": scenario_id,
        "oop_hands": len(oop_hands),
        "ip_hands": len(ip_hands),
        "num_buckets": actual_buckets,
        "num_info_sets": n_is,
        "ehs_time_s": round(ehs_time, 2),
        "solve_time_s": round(solve_time, 2),
        "total_time_s": round(elapsed, 2),
        "strat_raw_mb": round(strat_raw_mb, 1),
        "strat_compressed_mb": round(strat_comp_mb, 1),
        "file_mb": round(file_mb, 1),
        "root_non_uniform_pct": round(pct, 1),
    }


# ── Build work items ──────────────────────────────────────────────────────

def build_work_items(scenarios, textures):
    """Build flat list of (scenario_id, texture_key, board) work items.

    Returns sorted list for deterministic partitioning across workers.
    """
    items = []
    for sid in sorted(scenarios):
        for tex_key, board in textures:
            items.append((sid, tex_key, board))
    return items


# ── Smoke test ────────────────────────────────────────────────────────────

def run_smoke_test(bp_lib, ca_lib, scenarios):
    """Quick validation: solve 1 scenario × 1 texture with low iterations."""
    print("=== Smoke Test (v2: 2-player, preflop-filtered) ===")

    # Pick first SRP scenario
    sid = None
    for s in sorted(scenarios):
        if scenarios[s]["scenario_type"] == "srp":
            sid = s
            break
    if not sid:
        sid = sorted(scenarios)[0]

    scenario = scenarios[sid]
    print(f"  Scenario: {sid}")
    print(f"  OOP: {scenario['oop_pos']}, IP: {scenario['ip_pos']}")
    print(f"  Pot: {scenario['starting_pot']}, Stack: {scenario['effective_stack']}")

    result = solve_texture_v2(
        bp_lib, ca_lib, sid, scenario,
        "T72_r", ["Ts", "7h", "2d"],
        iterations=10000,
        num_threads=0,
        output_dir=None,
        ehs_samples=100,
    )
    if not result:
        print("  FAILED: solve returned None")
        return False

    if result["num_info_sets"] == 0:
        print("  FAILED: 0 info sets")
        return False

    print(f"  OOP hands: {result['oop_hands']}, IP hands: {result['ip_hands']}")
    print(f"  Buckets: {result['num_buckets']}")
    print(f"  Info sets: {result['num_info_sets']:,}")
    print(f"  EHS: {result['ehs_time_s']:.1f}s, Solve: {result['solve_time_s']:.1f}s")
    print(f"  Root non-uniform: {result['root_non_uniform_pct']:.0f}%")
    print("  PASSED")
    return True


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Blueprint v2 worker (preflop-filtered)")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--total-workers", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=20000000)
    parser.add_argument("--ehs-samples", type=int, default=500)
    parser.add_argument("--num-threads", type=int, default=0,
                        help="OpenMP threads (0=auto)")
    parser.add_argument("--output-dir", default="/tmp/blueprint_v2_output")
    parser.add_argument("--s3-bucket", default="")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--ranges", required=True,
                        help="Path to ranges.json")
    parser.add_argument("--smoke-test-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-uploaded S3 keys")
    args = parser.parse_args()

    print(f"=== Blueprint v2 Worker {args.worker_id}/{args.total_workers} ===")
    print(f"Players: {NUM_PLAYERS}, Iterations: {args.iterations:,}")
    print(f"Threads: {args.num_threads or 'auto'}, EHS samples: {args.ehs_samples}")
    print()

    # Load libraries
    bp_lib, ca_lib = load_dlls(args.build_dir)
    print("Libraries loaded.")

    # Load scenarios
    scenarios = build_scenario_matrix(args.ranges)
    print(f"Scenarios: {len(scenarios)}")

    # Smoke test
    if not run_smoke_test(bp_lib, ca_lib, scenarios):
        print("SMOKE TEST FAILED — aborting.")
        sys.exit(1)
    print()

    if args.smoke_test_only:
        print("Smoke test only — exiting.")
        return

    # Build and partition work items
    textures = generate_all_textures()
    all_items = build_work_items(scenarios, textures)
    my_items = [
        item for i, item in enumerate(all_items)
        if i % args.total_workers == args.worker_id
    ]
    print(f"Total work items: {len(all_items)}, this worker: {len(my_items)}")
    print()

    # Check for already-completed items (resume support)
    existing_keys = set()
    if args.resume and args.s3_bucket:
        print("Checking S3 for completed items...")
        try:
            result = subprocess.run(
                ["aws", "s3", "ls",
                 f"s3://{args.s3_bucket}/worker-{args.worker_id}/",
                 "--recursive"],
                capture_output=True, text=True, timeout=60
            )
            for line in result.stdout.strip().split("\n"):
                if line.strip() and ".bps" in line:
                    # Extract scenario_id/texture_key from path
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        key = parts[-1]  # worker-N/scenario/texture.bps
                        existing_keys.add(key)
        except Exception as e:
            print(f"  Warning: S3 resume check failed: {e}")

        if existing_keys:
            before = len(my_items)
            my_items = [
                (sid, tex, board) for sid, tex, board in my_items
                if f"worker-{args.worker_id}/{sid}/{tex}.bps" not in existing_keys
            ]
            print(f"  Skipping {before - len(my_items)} already-completed items")
    print()

    # Solve
    results = []
    failures = []
    t_batch_start = time.time()

    for idx, (scenario_id, tex_key, board) in enumerate(my_items):
        try:
            result = solve_texture_v2(
                bp_lib, ca_lib, scenario_id, scenarios[scenario_id],
                tex_key, board,
                iterations=args.iterations,
                num_threads=args.num_threads if args.num_threads > 0 else 0,
                output_dir=args.output_dir,
                ehs_samples=args.ehs_samples,
            )

            if result:
                results.append(result)

                # Upload to S3 and delete local
                if args.s3_bucket and args.output_dir:
                    bps_path = os.path.join(args.output_dir, scenario_id,
                                            f"{tex_key}.bps")
                    if os.path.exists(bps_path):
                        s3_dest = (f"s3://{args.s3_bucket}/"
                                   f"worker-{args.worker_id}/"
                                   f"{scenario_id}/{tex_key}.bps")
                        subprocess.run(
                            ["aws", "s3", "cp", bps_path, s3_dest, "--quiet"],
                            check=False
                        )
                        os.remove(bps_path)

                elapsed_total = time.time() - t_batch_start
                avg_per = elapsed_total / (idx + 1)
                remaining = avg_per * (len(my_items) - idx - 1)
                print(f"[{idx+1}/{len(my_items)}] {scenario_id}/{tex_key}: "
                      f"OOP={result['oop_hands']} IP={result['ip_hands']} "
                      f"{result['num_info_sets']:,} IS, "
                      f"{result['total_time_s']:.1f}s, "
                      f"root_nu={result['root_non_uniform_pct']:.0f}% "
                      f"(ETA: {remaining/3600:.1f}h)")
            else:
                failures.append(f"{scenario_id}/{tex_key}")
                print(f"[{idx+1}/{len(my_items)}] {scenario_id}/{tex_key}: FAILED")

        except Exception as e:
            failures.append(f"{scenario_id}/{tex_key}")
            print(f"[{idx+1}/{len(my_items)}] {scenario_id}/{tex_key}: ERROR: {e}")
            traceback.print_exc()

    # Summary
    t_batch_end = time.time()
    batch_elapsed = t_batch_end - t_batch_start

    avg_is = (sum(r["num_info_sets"] for r in results) / len(results)
              if results else 0)
    avg_nu = (sum(r["root_non_uniform_pct"] for r in results) / len(results)
              if results else 0)

    summary = {
        "worker_id": args.worker_id,
        "total_workers": args.total_workers,
        "num_players": NUM_PLAYERS,
        "iterations": args.iterations,
        "work_items_total": len(all_items),
        "work_items_this_worker": len(my_items),
        "solved": len(results),
        "failed": len(failures),
        "avg_info_sets": round(avg_is),
        "avg_root_non_uniform_pct": round(avg_nu, 1),
        "total_time_s": round(batch_elapsed, 1),
        "avg_time_per_item_s": round(batch_elapsed / max(len(results), 1), 1),
        "failures": failures[:100],  # cap to avoid huge JSON
    }

    print(f"\n{'='*60}")
    print(f"Worker {args.worker_id} Complete")
    print(f"  Solved: {len(results)}/{len(my_items)}")
    print(f"  Failed: {len(failures)}")
    print(f"  Avg info sets: {avg_is:,.0f}")
    print(f"  Avg root non-uniform: {avg_nu:.1f}%")
    print(f"  Total time: {batch_elapsed:.0f}s ({batch_elapsed/3600:.1f}h)")
    print(f"  Avg per item: {summary['avg_time_per_item_s']:.1f}s")
    print(f"{'='*60}")

    # Save summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir,
                                f"summary_worker_{args.worker_id}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Upload summary + log to S3
    if args.s3_bucket:
        s3_prefix = f"s3://{args.s3_bucket}/logs/"
        subprocess.run(
            ["aws", "s3", "cp", summary_path,
             f"{s3_prefix}summary_worker_{args.worker_id}.json", "--quiet"],
            check=False
        )
        print("Summary uploaded to S3.")


if __name__ == "__main__":
    main()
