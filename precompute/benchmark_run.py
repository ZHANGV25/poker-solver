#!/usr/bin/env python3
"""Benchmark script that runs ON the EC2 instance.

Measures:
1. Compilation time
2. EHS computation speed
3. MCCFR iteration throughput (single-thread + all cores)
4. Strategy convergence at different iteration counts
5. Memory usage per texture
6. Extrapolated full-blueprint cost

Outputs benchmark_results.json to /tmp/benchmark_output/
"""

import ctypes
import json
import os
import resource
import sys
import time

sys.path.insert(0, '/tmp/poker-solver')
sys.path.insert(0, '/tmp/poker-solver/python')

from precompute.blueprint_worker import (
    load_dlls, solve_texture, BPConfig, card_to_int,
    BP_MAX_HANDS, BP_MAX_ACTIONS, BLUEPRINT_BET_SIZES,
    STARTING_POT, EFFECTIVE_STACK, RANKS, SUITS
)
from precompute.solve_scenarios import generate_all_textures

OUTPUT_DIR = "/tmp/benchmark_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_mem_mb():
    """Current RSS in MB (Linux only)."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except:
        return 0

def benchmark_ehs(ca_lib, flop_ints, n_hands_list, sample_counts):
    """Benchmark EHS computation at different hand counts and sample sizes."""
    print("\n=== Benchmark: EHS Computation ===")
    HandsType = (ctypes.c_int * 2) * 1326
    all_hands = HandsType()
    n_all = ca_lib.ca_generate_hands((ctypes.c_int * 3)(*flop_ints), 3, all_hands)

    results = []
    for n_samples in sample_counts:
        ehs = (ctypes.c_float * n_all)()
        t0 = time.time()
        ca_lib.ca_compute_ehs(
            (ctypes.c_int * 3)(*flop_ints), 3,
            all_hands, n_all, n_samples, ehs
        )
        t1 = time.time()
        elapsed = t1 - t0
        results.append({
            "hands": n_all, "samples": n_samples,
            "time_s": round(elapsed, 3),
            "hands_per_sec": round(n_all / elapsed),
        })
        print(f"  {n_all} hands x {n_samples} samples: {elapsed:.2f}s ({n_all/elapsed:.0f} hands/s)")

    return results


def benchmark_mccfr(bp_lib, ca_lib, flop_ints, num_players, num_buckets,
                     iteration_counts, thread_counts):
    """Benchmark MCCFR at different iteration counts and thread counts."""
    print(f"\n=== Benchmark: {num_players}-Player MCCFR ===")

    # Generate hands and buckets once
    HandsType = (ctypes.c_int * 2) * 1326
    all_hands = HandsType()
    n_all = ca_lib.ca_generate_hands((ctypes.c_int * 3)(*flop_ints), 3, all_hands)

    ehs = (ctypes.c_float * n_all)()
    ca_lib.ca_compute_ehs((ctypes.c_int * 3)(*flop_ints), 3, all_hands, n_all, 500, ehs)

    buckets = (ctypes.c_int * n_all)()
    actual_buckets = ca_lib.ca_assign_buckets(ehs, n_all, num_buckets, buckets)

    nh = min(n_all, BP_MAX_HANDS)
    results = []

    for num_threads in thread_counts:
        for iterations in iteration_counts:
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

            config = BPConfig()
            bp_lib.bp_default_config(ctypes.byref(config))
            config.num_threads = num_threads
            config.hash_table_size = (1 << 22)
            config.discount_stop_iter = max(iterations * 35 // 1000, 1000)
            config.discount_interval = max(config.discount_stop_iter // 40, 100)
            config.prune_start_iter = max(iterations * 17 // 1000, 500)
            config.strategy_interval = 10000

            bp_lib.bp_init_ex(
                solver, num_players,
                (ctypes.c_int * 3)(*flop_ints),
                ch, cw, cn, STARTING_POT, EFFECTIVE_STACK,
                c_bs, len(BLUEPRINT_BET_SIZES),
                ctypes.byref(config)
            )

            BM = (ctypes.c_int * BP_MAX_HANDS) * 6
            NB = ctypes.c_int * 6
            bm, nb = BM(), NB()
            for p in range(num_players):
                nb[p] = actual_buckets
                for h in range(nh):
                    bm[p][h] = buckets[h]
            for street in [1, 2, 3]:
                bp_lib.bp_set_buckets(solver, street, bm, nb)

            mem_before = get_mem_mb()
            t0 = time.time()
            bp_lib.bp_solve(solver, iterations)
            t1 = time.time()
            mem_after = get_mem_mb()

            n_is = bp_lib.bp_num_info_sets(solver)
            elapsed = t1 - t0
            ips = iterations / elapsed if elapsed > 0 else 0

            bp_lib.bp_free(solver)

            r = {
                "players": num_players, "threads": num_threads,
                "iterations": iterations, "info_sets": n_is,
                "time_s": round(elapsed, 2), "iter_per_s": round(ips),
                "mem_delta_mb": round(mem_after - mem_before, 1),
            }
            results.append(r)
            print(f"  {num_players}P, {num_threads}T, {iterations:>8,} iter: "
                  f"{n_is:>8,} IS, {elapsed:>6.1f}s, {ips:>8,.0f} iter/s, "
                  f"mem +{mem_after-mem_before:.0f}MB")

    return results


def benchmark_convergence(bp_lib, ca_lib, flop_ints, num_players, num_buckets):
    """Measure strategy convergence: L1 distance between N and 2N iterations."""
    print(f"\n=== Benchmark: Convergence ({num_players}P) ===")

    iteration_pairs = [(50000, 100000), (100000, 200000), (500000, 1000000)]
    results = []

    for iter_n, iter_2n in iteration_pairs:
        # Solve at N
        r_n = solve_texture(bp_lib, ca_lib, "T72_r", ["Ts", "7h", "2d"],
                            num_players=num_players, num_buckets=num_buckets,
                            iterations=iter_n, num_threads=0,
                            output_dir=None, ehs_samples=500)

        # Solve at 2N
        r_2n = solve_texture(bp_lib, ca_lib, "T72_r", ["Ts", "7h", "2d"],
                             num_players=num_players, num_buckets=num_buckets,
                             iterations=iter_2n, num_threads=0,
                             output_dir=None, ehs_samples=500)

        if r_n and r_2n:
            # Compute L1 distance between root strategies
            dists = []
            s_n = r_n["root_strategies"]
            s_2n = r_2n["root_strategies"]
            for b in s_n:
                if b in s_2n and len(s_n[b]) == len(s_2n[b]):
                    d = sum(abs(a - b_) for a, b_ in zip(s_n[b], s_2n[b]))
                    dists.append(d)
            avg_l1 = sum(dists) / len(dists) if dists else float('inf')

            r = {
                "iter_n": iter_n, "iter_2n": iter_2n,
                "info_sets_n": r_n["num_info_sets"],
                "info_sets_2n": r_2n["num_info_sets"],
                "l1_distance": round(avg_l1, 4),
                "time_n_s": r_n["total_time_s"],
                "time_2n_s": r_2n["total_time_s"],
            }
            results.append(r)
            print(f"  {iter_n:>8,} vs {iter_2n:>8,}: L1={avg_l1:.4f}, "
                  f"IS={r_2n['num_info_sets']:,}, time={r_2n['total_time_s']:.1f}s")

    return results


def benchmark_full_texture_batch(bp_lib, ca_lib, num_players, num_buckets,
                                  iterations, num_textures=10):
    """Solve a batch of textures to get realistic per-texture timing."""
    print(f"\n=== Benchmark: {num_textures} Texture Batch ({num_players}P, {iterations:,} iter) ===")

    all_textures = generate_all_textures()
    # Sample evenly across all textures
    step = max(len(all_textures) // num_textures, 1)
    sample = [(key, board) for i, (key, board) in enumerate(all_textures) if i % step == 0][:num_textures]

    results = []
    for i, (key, board) in enumerate(sample):
        r = solve_texture(bp_lib, ca_lib, key, board,
                          num_players=num_players, num_buckets=num_buckets,
                          iterations=iterations, num_threads=0,
                          output_dir=None, ehs_samples=500)
        if r:
            results.append({
                "texture": key,
                "info_sets": r["num_info_sets"],
                "ehs_s": r["ehs_time_s"],
                "solve_s": r["solve_time_s"],
                "total_s": r["total_time_s"],
            })
            print(f"  [{i+1}/{num_textures}] {key}: {r['num_info_sets']:,} IS, "
                  f"{r['total_time_s']:.1f}s (EHS={r['ehs_time_s']:.1f}s + solve={r['solve_time_s']:.1f}s)")

    return results


def main():
    print("=" * 60)
    print("  EC2 Blueprint Benchmark")
    print("=" * 60)
    print(f"CPUs: {os.cpu_count()}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    bp_lib, ca_lib = load_dlls("build")
    print("Libraries loaded.")

    flop_ints = [card_to_int("Ts"), card_to_int("7h"), card_to_int("2d")]
    all_results = {"timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                   "cpus": os.cpu_count()}

    # 1. EHS speed
    all_results["ehs"] = benchmark_ehs(ca_lib, flop_ints, [1176],
                                        [100, 500, 1000])

    # 2. MCCFR throughput: vary threads
    thread_options = [1, 2, 4, 8, 16] if os.cpu_count() >= 16 else [1, os.cpu_count()]
    all_results["mccfr_threads"] = benchmark_mccfr(
        bp_lib, ca_lib, flop_ints,
        num_players=6, num_buckets=200,
        iteration_counts=[100000],
        thread_counts=thread_options
    )

    # 3. MCCFR throughput: vary iterations (all cores)
    all_results["mccfr_scale"] = benchmark_mccfr(
        bp_lib, ca_lib, flop_ints,
        num_players=6, num_buckets=200,
        iteration_counts=[100000, 500000, 1000000],
        thread_counts=[0]  # 0 = auto (all cores)
    )

    # 4. Convergence
    all_results["convergence_6p"] = benchmark_convergence(
        bp_lib, ca_lib, flop_ints, num_players=6, num_buckets=200
    )

    # 5. Batch of 10 textures at production settings
    all_results["batch_6p"] = benchmark_full_texture_batch(
        bp_lib, ca_lib, num_players=6, num_buckets=200, iterations=1000000
    )

    # 6. Extrapolate full blueprint cost
    if all_results["batch_6p"]:
        avg_time = sum(r["total_s"] for r in all_results["batch_6p"]) / len(all_results["batch_6p"])
        total_1755 = avg_time * 1755
        spot_price_hr = 0.28  # c5.4xlarge spot estimate

        all_results["extrapolation"] = {
            "avg_time_per_texture_s": round(avg_time, 1),
            "total_1755_textures_s": round(total_1755),
            "total_1755_textures_hrs": round(total_1755 / 3600, 2),
            "cost_1_instance": round(total_1755 / 3600 * spot_price_hr, 2),
            "cost_20_instances": round(total_1755 / 3600 / 20 * 20 * spot_price_hr, 2),
            "wall_time_20_instances_min": round(total_1755 / 60 / 20, 1),
        }

        print(f"\n{'='*60}")
        print(f"  EXTRAPOLATION: Full 6-Player Blueprint")
        print(f"{'='*60}")
        print(f"  Avg per texture:       {avg_time:.1f}s")
        print(f"  Total (1755 textures): {total_1755/3600:.1f} hrs on 1 instance")
        print(f"  With 20 instances:     {total_1755/60/20:.1f} min wall time")
        print(f"  Estimated cost:        ${total_1755/3600/20*20*spot_price_hr:.2f}")

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
