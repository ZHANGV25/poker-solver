#!/usr/bin/env python3
"""Benchmark: measure strategy convergence at different tree depths and iteration counts.

Tests a single texture (AAA rainbow) at 50K, 200K, 1M, 5M, 10M iterations.
For each, checks:
  - Root node strategy (player 0, no actions)
  - 1-deep nodes (player 1, after player 0 acts)
  - 2-deep nodes (player 2, after P0+P1 act)
  - Overall % of non-uniform flop strategies in binary export
"""

import ctypes
import struct
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

from precompute.blueprint_worker import BPConfig, BP_MAX_ACTIONS, BP_MAX_HANDS

BUILD_DIR = os.environ.get("BUILD_DIR", "build")

def load_libs():
    ext = ".so" if sys.platform != "win32" else ".dll"
    bp = ctypes.CDLL(os.path.join(BUILD_DIR, f"mccfr_blueprint{ext}"))
    ca = ctypes.CDLL(os.path.join(BUILD_DIR, f"card_abstraction{ext}"))
    bp.bp_default_config.restype = None
    bp.bp_default_config.argtypes = [ctypes.POINTER(BPConfig)]
    bp.bp_init_ex.restype = ctypes.c_int
    bp.bp_solve.restype = ctypes.c_int
    bp.bp_get_strategy.restype = ctypes.c_int
    bp.bp_num_info_sets.restype = ctypes.c_int
    bp.bp_free.restype = None
    bp.bp_set_buckets.restype = ctypes.c_int
    bp.bp_export_strategies.restype = ctypes.c_int
    bp.bp_export_strategies.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t)
    ]
    ca.ca_generate_hands.restype = ctypes.c_int
    ca.ca_compute_ehs.restype = None
    ca.ca_assign_buckets.restype = ctypes.c_int
    return bp, ca


def run_test(bp_lib, ca_lib, iterations, flop_ints, num_buckets=200):
    """Solve one texture, export, and measure strategy quality."""
    # Generate hands + EHS + buckets
    HandsType = (ctypes.c_int * 2) * 1326
    all_hands = HandsType()
    n_all = ca_lib.ca_generate_hands(
        (ctypes.c_int * 3)(*flop_ints), 3, all_hands
    )
    ehs = (ctypes.c_float * n_all)()
    ca_lib.ca_compute_ehs(
        (ctypes.c_int * 3)(*flop_ints), 3, all_hands, n_all, 500, ehs
    )
    buckets = (ctypes.c_int * n_all)()
    ab = ca_lib.ca_assign_buckets(ehs, n_all, num_buckets, buckets)

    nh = min(n_all, BP_MAX_HANDS)

    # Init solver
    buf = (ctypes.c_char * 524288)()
    solver = ctypes.cast(buf, ctypes.c_void_p)
    HT = ((ctypes.c_int * 2) * BP_MAX_HANDS) * 6
    WT = (ctypes.c_float * BP_MAX_HANDS) * 6
    NT = ctypes.c_int * 6
    ch, cw, cn = HT(), WT(), NT()
    for p in range(6):
        cn[p] = nh
        for h in range(nh):
            ch[p][h][0] = all_hands[h][0]
            ch[p][h][1] = all_hands[h][1]
            cw[p][h] = 1.0

    config = BPConfig()
    bp_lib.bp_default_config(ctypes.byref(config))
    config.num_threads = 0  # auto
    config.hash_table_size = 0
    config.strategy_interval = 1
    config.discount_stop_iter = max(iterations * 35 // 1000, 1000)
    config.discount_interval = max(config.discount_stop_iter // 40, 100)
    config.prune_start_iter = max(iterations * 17 // 1000, 500)

    BETS = [0.5, 1.0, 2.0]
    c_bs = (ctypes.c_float * len(BETS))(*BETS)
    bp_lib.bp_init_ex(
        solver, 6, (ctypes.c_int * 3)(*flop_ints),
        ch, cw, cn, 6, 200, c_bs, len(BETS), ctypes.byref(config)
    )
    BM = (ctypes.c_int * BP_MAX_HANDS) * 6
    NB = ctypes.c_int * 6
    bm, nb = BM(), NB()
    for p in range(6):
        nb[p] = ab
        for h in range(nh):
            bm[p][h] = buckets[h]
    for st in [1, 2, 3]:
        bp_lib.bp_set_buckets(solver, st, bm, nb)

    # Solve
    t0 = time.time()
    bp_lib.bp_solve(solver, iterations)
    solve_time = time.time() - t0
    n_is = bp_lib.bp_num_info_sets(solver)

    # Check root strategies via bp_get_strategy
    strat_out = (ctypes.c_float * BP_MAX_ACTIONS)()
    root_uniform = 0
    root_diff = 0
    for b in range(min(ab, 200)):
        na = bp_lib.bp_get_strategy(
            solver, 0,
            (ctypes.c_int * 3)(*flop_ints), 3,
            (ctypes.c_int * 1)(), 0,
            strat_out, b
        )
        if na >= 3:
            probs = [float(strat_out[a]) for a in range(na)]
            if all(abs(p - probs[0]) < 0.02 for p in probs):
                root_uniform += 1
            else:
                root_diff += 1

    # Export binary strategies
    needed = ctypes.c_size_t(0)
    bp_lib.bp_export_strategies(solver, None, 0, ctypes.byref(needed))
    strat_buf = (ctypes.c_char * needed.value)()
    written = ctypes.c_size_t(0)
    bp_lib.bp_export_strategies(solver, strat_buf, needed.value, ctypes.byref(written))
    sd = bytes(strat_buf[:written.value])

    bp_lib.bp_free(solver)

    # Analyze binary export
    ne = struct.unpack_from("<I", sd, 4)[0]
    off = 12
    by_street = {}  # street -> {uniform, diff, total}
    for i in range(ne):
        p = sd[off]; st = sd[off + 1]; off += 18
        na = sd[off]; nh2 = struct.unpack_from("<H", sd, off + 1)[0]; off += 3

        if st not in by_street:
            by_street[st] = {"uniform": 0, "diff": 0, "total": 0, "na1": 0}

        if na <= 1:
            by_street[st]["na1"] += 1
        else:
            # Sample buckets 0, 42, 100, 199
            for h in [0, 42, 100, min(199, nh2 - 1)]:
                if h >= nh2:
                    continue
                probs = [sd[off + h * na + a] for a in range(na)]
                is_uniform = all(abs(x - probs[0]) <= 1 for x in probs)
                by_street[st]["total"] += 1
                if is_uniform:
                    by_street[st]["uniform"] += 1
                else:
                    by_street[st]["diff"] += 1

        off += na * nh2

    return {
        "iterations": iterations,
        "info_sets": n_is,
        "solve_time_s": round(solve_time, 1),
        "root_diff": root_diff,
        "root_uniform": root_uniform,
        "by_street": by_street,
    }


def main():
    bp_lib, ca_lib = load_libs()
    flop = [36, 32, 8]  # As Ah Ad (rainbow trips)

    # Test at different iteration counts
    for iters in [50_000, 200_000, 1_000_000, 5_000_000, 10_000_000]:
        print(f"\n{'='*60}")
        print(f"  {iters:,} iterations")
        print(f"{'='*60}")

        result = run_test(bp_lib, ca_lib, iters, flop)

        print(f"  Time: {result['solve_time_s']}s")
        print(f"  Info sets: {result['info_sets']:,}")
        print(f"  Root (bp_get_strategy): {result['root_diff']} diff / "
              f"{result['root_diff'] + result['root_uniform']} total")

        for st in sorted(result["by_street"].keys()):
            d = result["by_street"][st]
            total = d["diff"] + d["uniform"]
            pct = 100 * d["diff"] / max(total, 1)
            street_name = {1: "FLOP", 2: "TURN", 3: "RIVER"}.get(st, f"ST{st}")
            print(f"  {street_name} binary: {d['diff']} diff / {total} checked "
                  f"({pct:.1f}% non-uniform) [na=1 skipped: {d['na1']}]")

        sys.stdout.flush()


if __name__ == "__main__":
    main()
