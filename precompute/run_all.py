#!/usr/bin/env python3
"""GPU precompute pipeline: solve all textures and extract blueprint strategies.

Uses the CUDA flop_solve.dll with fs_solve_gpu_extract_all() to:
1. Build full flop→turn→river tree per texture
2. Solve with Linear CFR (GPU-accelerated)
3. Extract weighted-average strategies at flop root + all turn roots
4. Extract per-hand CFV and per-action EVs
5. Store in binary BlueprintStore format

Usage:
    python run_all.py --scenario CO_vs_BB_srp --iterations 100
    python run_all.py --all --iterations 100

EC2 usage:
    python run_all.py --all --iterations 100 --output-dir /mnt/data/blueprints
"""

import argparse
import ctypes
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.dirname(__file__))

from solve_scenarios import generate_all_textures, load_scenarios
from solver import card_to_int, parse_range_string
from blueprint_store import BlueprintStore, pack_texture_blob

# ── Constants matching flop_solve.cuh ────────────────────────────────────
FS_MAX_HANDS = 200
FS_MAX_ACTIONS = 6
SCALE = 100


# ── ctypes structure mirrors ─────────────────────────────────────────────

class FSNode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("player", ctypes.c_int),
        ("num_children", ctypes.c_int),
        ("first_child", ctypes.c_int),
        ("pot", ctypes.c_int),
        ("bets", ctypes.c_int * 2),
        ("board_cards", ctypes.c_int * 5),
        ("num_board", ctypes.c_int),
    ]


class FSTreeData(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.POINTER(FSNode)),
        ("children", ctypes.POINTER(ctypes.c_int)),
        ("num_nodes", ctypes.c_int),
        ("num_children_total", ctypes.c_int),

        ("hands", ((ctypes.c_int * 2) * FS_MAX_HANDS) * 2),
        ("weights", (ctypes.c_float * FS_MAX_HANDS) * 2),
        ("num_hands", ctypes.c_int * 2),

        ("level_order", ctypes.POINTER(ctypes.c_int)),
        ("node_depth", ctypes.POINTER(ctypes.c_int)),
        ("max_depth", ctypes.c_int),

        ("decision_node_indices", ctypes.POINTER(ctypes.c_int)),
        ("num_decision_nodes", ctypes.c_int),
        ("showdown_node_indices", ctypes.POINTER(ctypes.c_int)),
        ("num_showdown_nodes", ctypes.c_int),
    ]


class FSOutput(ctypes.Structure):
    _fields_ = [
        ("root_strategy", ctypes.POINTER(ctypes.c_float)),
        ("root_num_actions", ctypes.c_int),
        ("root_player", ctypes.c_int),
        ("root_ev", ctypes.POINTER(ctypes.c_float)),

        # Extended fields for extract_all
        ("all_avg_strategies", ctypes.POINTER(ctypes.c_float)),
        ("all_cfv", ctypes.POINTER(ctypes.c_float)),
        ("max_hands", ctypes.c_int),

        ("turn_root_indices", ctypes.POINTER(ctypes.c_int)),
        ("turn_root_cards", ctypes.POINTER(ctypes.c_int)),
        ("num_turn_roots", ctypes.c_int),
    ]


# ── GPU solver loading ──────────────────────────────────────────────────

def load_gpu_solver():
    """Load the flop_solve DLL."""
    solver_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_path = os.path.join(solver_dir, "build", "flop_solve.dll")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(solver_dir, "build", "flop_solve.so")
    if not os.path.exists(lib_path):
        print("ERROR: flop_solve library not found in build/", file=sys.stderr)
        sys.exit(1)

    lib = ctypes.CDLL(lib_path)
    lib.fs_build_tree.restype = ctypes.c_int
    lib.fs_solve_gpu.restype = ctypes.c_int
    lib.fs_solve_gpu_extract_all.restype = ctypes.c_int
    lib.fs_free_tree.restype = None
    lib.fs_free_output.restype = None
    return lib


# ── Solve a single texture ──────────────────────────────────────────────

def solve_texture(lib, board_ints, oop_hands, ip_hands,
                  starting_pot, effective_stack,
                  bet_sizes, iterations):
    """Solve one texture and extract all strategies.

    Returns:
        dict with flop strategies, turn strategies, EVs, action EVs
        or None on failure
    """
    tree = FSTreeData()
    board_arr = (ctypes.c_int * 3)(*board_ints)
    bet_arr = (ctypes.c_float * len(bet_sizes))(*bet_sizes)

    pot = int(starting_pot * SCALE)
    stack = int(effective_stack * SCALE)

    err = lib.fs_build_tree(board_arr, pot, stack, bet_arr, len(bet_sizes),
                            ctypes.byref(tree))
    if err != 0:
        return None

    # Populate hands
    board_set = set(board_ints)
    filtered = [[], []]
    for p, hands in enumerate([oop_hands, ip_hands]):
        for c0, c1, w in hands:
            if c0 in board_set or c1 in board_set:
                continue
            idx = len(filtered[p])
            if idx >= FS_MAX_HANDS:
                break
            tree.hands[p][idx][0] = c0
            tree.hands[p][idx][1] = c1
            tree.weights[p][idx] = w
            filtered[p].append((c0, c1, w))
        tree.num_hands[p] = len(filtered[p])

    output = FSOutput()
    err = lib.fs_solve_gpu_extract_all(ctypes.byref(tree), iterations,
                                        ctypes.byref(output))
    if err != 0:
        lib.fs_free_tree(ctypes.byref(tree))
        return None

    max_h = output.max_hands
    nh0 = tree.num_hands[0]
    nh1 = tree.num_hands[1]
    N = tree.num_nodes

    # Extract flop root strategy (weighted average)
    root_na = output.root_num_actions
    root_player = output.root_player

    flop_strategies = {}
    flop_evs = {}
    flop_action_evs = {}

    # Root node is node 0
    for p in range(2):
        nh = nh0 if p == 0 else nh1
        if tree.nodes[0].player == p:
            # This player acts at root
            strat = np.zeros((nh, root_na), dtype=np.float32)
            for h in range(nh):
                for a in range(root_na):
                    strat[h, a] = output.all_avg_strategies[
                        0 * FS_MAX_ACTIONS * max_h + a * max_h + h]
            flop_strategies[p] = strat

            # Per-hand CFV at root
            ev = np.zeros(nh, dtype=np.float32)
            for h in range(nh):
                ev[h] = output.all_cfv[0 * max_h + h]
            flop_evs[p] = ev

            # Per-action EV at root: CFV at each child node
            action_evs = np.zeros((root_na, nh), dtype=np.float32)
            for a in range(root_na):
                child_idx = tree.children[tree.nodes[0].first_child + a]
                for h in range(nh):
                    action_evs[a, h] = output.all_cfv[child_idx * max_h + h]
            flop_action_evs[p] = action_evs
        else:
            # Other player: uniform strategy (they don't act at root)
            flop_strategies[p] = np.ones((nh, 1), dtype=np.float32)
            flop_evs[p] = np.zeros(nh, dtype=np.float32)
            flop_action_evs[p] = np.zeros((1, nh), dtype=np.float32)

    # Extract turn root strategies
    turn_data = []
    for ti in range(output.num_turn_roots):
        turn_node_idx = output.turn_root_indices[ti]
        turn_card = output.turn_root_cards[ti]
        turn_node = tree.nodes[turn_node_idx]
        turn_na = turn_node.num_children
        turn_player = turn_node.player

        # Count valid hands for this turn card
        tc_set = board_set | {turn_card}
        nh0_tc = sum(1 for c0, c1, w in filtered[0]
                     if c0 not in tc_set and c1 not in tc_set)
        nh1_tc = sum(1 for c0, c1, w in filtered[1]
                     if c0 not in tc_set and c1 not in tc_set)

        td = {
            'turn_card': turn_card,
            'num_hands_oop': nh0_tc,
            'num_hands_ip': nh1_tc,
            'strategies': {},
            'evs': {},
            'action_evs': {},
        }

        for p in range(2):
            nh = nh0 if p == 0 else nh1
            nh_tc = nh0_tc if p == 0 else nh1_tc
            hands = filtered[p]

            if turn_player == p:
                # Extract strategy for valid hands only
                strat = np.zeros((nh_tc, turn_na), dtype=np.float32)
                ev = np.zeros(nh_tc, dtype=np.float32)
                action_ev = np.zeros((turn_na, nh_tc), dtype=np.float32)

                hi = 0
                for h in range(nh):
                    c0, c1, w = hands[h]
                    if c0 in tc_set or c1 in tc_set:
                        continue
                    for a in range(turn_na):
                        strat[hi, a] = output.all_avg_strategies[
                            turn_node_idx * FS_MAX_ACTIONS * max_h + a * max_h + h]
                    ev[hi] = output.all_cfv[turn_node_idx * max_h + h]

                    for a in range(turn_na):
                        child_idx = tree.children[turn_node.first_child + a]
                        action_ev[a, hi] = output.all_cfv[child_idx * max_h + h]
                    hi += 1

                td['strategies'][p] = strat
                td['evs'][p] = ev
                td['action_evs'][p] = action_ev
            else:
                td['strategies'][p] = np.ones((nh_tc, 1), dtype=np.float32)
                td['evs'][p] = np.zeros(nh_tc, dtype=np.float32)
                td['action_evs'][p] = np.zeros((1, nh_tc), dtype=np.float32)

        turn_data.append(td)

    # Sort turn data by card id for consistent storage
    turn_data.sort(key=lambda x: x['turn_card'])

    # Determine num_turn_actions (from the first turn root)
    num_turn_actions = 1
    if turn_data:
        for td in turn_data:
            for p in range(2):
                na = td['strategies'][p].shape[1]
                if na > num_turn_actions:
                    num_turn_actions = na

    result = {
        'oop_hands': filtered[0],
        'ip_hands': filtered[1],
        'flop_strategies': flop_strategies,
        'flop_evs': flop_evs,
        'flop_action_evs': flop_action_evs,
        'turn_data': turn_data,
        'num_flop_actions': root_na,
        'num_turn_actions': num_turn_actions,
    }

    lib.fs_free_output(ctypes.byref(output))
    lib.fs_free_tree(ctypes.byref(tree))

    return result


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPU precompute pipeline")
    parser.add_argument("--scenario", help="Solve a specific scenario")
    parser.add_argument("--all", action="store_true", help="Solve all scenarios")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Linear CFR iterations per texture")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ranges", default=None)
    parser.add_argument("--bet-sizes", default="0.75",
                        help="Comma-separated bet size fractions")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit textures per scenario (0=all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip scenarios with existing blueprint data")
    args = parser.parse_args()

    # Find ranges
    ranges_path = args.ranges
    if ranges_path is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "data", "ranges.json"),
            "C:/Users/Victor/Documents/Projects/ACRPoker-Hud-PC/solver/ranges.json",
        ]
        for c in candidates:
            if os.path.isfile(c):
                ranges_path = c
                break

    if ranges_path is None:
        print("ERROR: ranges.json not found. Use --ranges.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "blueprints")

    bet_sizes = [float(x.strip()) for x in args.bet_sizes.split(",")]

    # Load solver
    lib = load_gpu_solver()
    print(f"Loaded GPU solver")

    # Load scenarios
    scenarios = load_scenarios(ranges_path)
    print(f"Loaded {len(scenarios)} scenarios from {ranges_path}")

    if args.scenario:
        if args.scenario not in scenarios:
            print(f"ERROR: scenario '{args.scenario}' not found")
            print(f"Available: {sorted(scenarios.keys())}")
            sys.exit(1)
        scenarios = {args.scenario: scenarios[args.scenario]}
    elif not args.all:
        print("Specify --scenario SCENARIO_ID or --all")
        print(f"Available: {sorted(scenarios.keys())}")
        return

    textures = generate_all_textures()
    if args.limit > 0:
        textures = textures[:args.limit]

    print(f"Textures: {len(textures)}, iterations: {args.iterations}, "
          f"bet sizes: {bet_sizes}")
    print()

    total_start = time.time()

    for s_idx, (scenario_id, scenario) in enumerate(sorted(scenarios.items())):
        store_dir = os.path.join(output_dir, scenario_id)

        if args.resume and os.path.exists(os.path.join(store_dir, 'index.bin')):
            print(f"[{s_idx+1}/{len(scenarios)}] {scenario_id} — skipped (exists)")
            continue

        oop_hands = parse_range_string(scenario['oop_range'])
        ip_hands = parse_range_string(scenario['ip_range'])

        print(f"[{s_idx+1}/{len(scenarios)}] {scenario_id}: "
              f"OOP={len(oop_hands)} hands, IP={len(ip_hands)} hands")

        store = BlueprintStore(store_dir, mode='w')

        done = 0
        failed = 0
        t0 = time.time()

        for tex_key, board_strs in textures:
            board_ints = [card_to_int(c) for c in board_strs]

            result = solve_texture(
                lib, board_ints, oop_hands, ip_hands,
                scenario['starting_pot'], scenario['effective_stack'],
                bet_sizes, args.iterations)

            if result is None:
                failed += 1
                continue

            # Pack and store
            blob = pack_texture_blob(
                oop_hands=result['oop_hands'],
                ip_hands=result['ip_hands'],
                flop_strategies=result['flop_strategies'],
                flop_evs=result['flop_evs'],
                flop_action_evs=result['flop_action_evs'],
                turn_data=result['turn_data'],
                num_flop_actions=result['num_flop_actions'],
                num_turn_actions=result['num_turn_actions'],
            )
            store.write_texture(tex_key, blob)
            done += 1

            if done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(textures) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(textures)} ({rate:.1f}/s, "
                      f"ETA {remaining:.0f}s, {failed} failed)")

        store.close()
        elapsed = time.time() - t0
        print(f"  Done: {done} solved, {failed} failed in {elapsed:.0f}s")

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
