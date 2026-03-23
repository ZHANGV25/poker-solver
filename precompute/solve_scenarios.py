#!/usr/bin/env python3
"""EC2 precompute: solve all flop textures for all scenarios.

Optimized for batch execution on EC2 GPU/CPU instances.
Produces per-hand strategies at all flop AND turn decision nodes.

Usage:
    python solve_scenarios.py --scenario CO_vs_BB_srp --workers 4
    python solve_scenarios.py --all --workers 16  # solve everything
    python solve_scenarios.py --scenario CO_vs_BB_srp --save-turn  # include turn data

Output:
    flop_solutions/{scenario_id}/{texture_key}.json

EC2 deployment:
    See deploy.sh for automated instance setup and execution.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Texture generation (suit isomorphism) ────────────────────────────────────

RANKS = "23456789TJQKA"
RANK_VALUE = {r: i for i, r in enumerate(RANKS)}


def generate_all_textures():
    """Generate all 1,755 strategically unique flop textures.

    Returns list of (texture_key, canonical_board) tuples.
    """
    textures = []

    for r0 in range(12, -1, -1):
        for r1 in range(r0, -1, -1):
            for r2 in range(r1, -1, -1):
                ranks = [RANKS[r0], RANKS[r1], RANKS[r2]]
                rank_str = "".join(ranks)

                if r0 == r1 == r2:
                    # Trips: only rainbow
                    board = [ranks[0] + "s", ranks[1] + "h", ranks[2] + "d"]
                    textures.append((rank_str + "_r", board))
                elif r0 == r1 or r1 == r2:
                    # Paired: rainbow and flush-draw
                    board_r = [ranks[0] + "s", ranks[1] + "h", ranks[2] + "d"]
                    textures.append((rank_str + "_r", board_r))
                    # Flush draw (paired cards same suit)
                    if r0 == r1:
                        board_fd = [ranks[0] + "s", ranks[1] + "s", ranks[2] + "h"]
                    else:
                        board_fd = [ranks[0] + "s", ranks[1] + "h", ranks[2] + "h"]
                    textures.append((rank_str + "_fd", board_fd))
                else:
                    # Unpaired: rainbow, monotone, 3 flush draws
                    board_r = [ranks[0] + "s", ranks[1] + "h", ranks[2] + "d"]
                    textures.append((rank_str + "_r", board_r))
                    board_m = [ranks[0] + "s", ranks[1] + "s", ranks[2] + "s"]
                    textures.append((rank_str + "_m", board_m))
                    board_fd12 = [ranks[0] + "s", ranks[1] + "s", ranks[2] + "h"]
                    textures.append((rank_str + "_fd12", board_fd12))
                    board_fd13 = [ranks[0] + "s", ranks[1] + "h", ranks[2] + "s"]
                    textures.append((rank_str + "_fd13", board_fd13))
                    board_fd23 = [ranks[0] + "h", ranks[1] + "s", ranks[2] + "s"]
                    textures.append((rank_str + "_fd23", board_fd23))

    return textures


# ── Scenario definitions ─────────────────────────────────────────────────────

def load_scenarios(ranges_path):
    """Load scenario definitions from ranges.json.

    Returns dict of scenario_id -> {oop_range, ip_range, starting_pot, effective_stack}.
    """
    with open(ranges_path) as f:
        ranges = json.load(f)

    scenarios = {}

    # Single raised pots (SRP): opener raised ~2.5 BB, defender called
    # Pot = 2.5 + 2.5 + 1.5 (blinds) = ~6.5 BB for most positions
    # Effective stack = ~97.5 BB (100 - 2.5 open)
    srp_pot = 6.5
    srp_stack = 97.5

    # 3-bet pots: opener raised, defender 3-bet (~9 BB), opener called
    # Pot = ~20 BB, effective stack = ~82 BB
    tbp_pot = 20.0
    tbp_stack = 82.0

    # Build all position matchups
    positions = ["UTG", "MP", "CO", "BTN", "SB", "BB"]

    for opener in positions:
        rfi = ranges.get("rfi", {}).get(opener)
        if not rfi:
            continue
        for defender in positions:
            if defender == opener:
                continue

            # SRP: opener RFI, defender calls
            vs_key = "{}_vs_{}".format(defender, opener)
            vs_entry = ranges.get("vs_rfi", {}).get(vs_key)
            if vs_entry and vs_entry.get("call"):
                # Determine OOP/IP
                post_order = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
                o_idx = post_order.index(opener) if opener in post_order else 99
                d_idx = post_order.index(defender) if defender in post_order else 99
                if o_idx < d_idx:
                    oop_range, ip_range = rfi, vs_entry["call"]
                    oop_pos, ip_pos = opener, defender
                else:
                    oop_range, ip_range = vs_entry["call"], rfi
                    oop_pos, ip_pos = defender, opener

                sid = "{}_vs_{}_srp".format(oop_pos, ip_pos)
                scenarios[sid] = {
                    "oop_range": oop_range,
                    "ip_range": ip_range,
                    "starting_pot": srp_pot,
                    "effective_stack": srp_stack,
                    "opener": opener,
                    "defender": defender,
                }

            # 3BP: opener RFI, defender 3-bets, opener calls
            if vs_entry and vs_entry.get("3bet"):
                vs3 = ranges.get("vs_3bet", {}).get(opener)
                if vs3 and vs3.get("call"):
                    o_idx = post_order.index(opener) if opener in post_order else 99
                    d_idx = post_order.index(defender) if defender in post_order else 99
                    if o_idx < d_idx:
                        oop_range = vs3["call"]
                        ip_range = vs_entry["3bet"]
                        oop_pos, ip_pos = opener, defender
                    else:
                        oop_range = vs_entry["3bet"]
                        ip_range = vs3["call"]
                        oop_pos, ip_pos = defender, opener

                    sid = "{}_vs_{}_3bp".format(oop_pos, ip_pos)
                    scenarios[sid] = {
                        "oop_range": oop_range,
                        "ip_range": ip_range,
                        "starting_pot": tbp_pot,
                        "effective_stack": tbp_stack,
                        "opener": opener,
                        "defender": defender,
                    }

    return scenarios


# ── Solver invocation ────────────────────────────────────────────────────────

def solve_one(args):
    """Solve a single texture for a scenario.

    Args:
        tuple of (solver_bin, texture_key, board, scenario, output_path, settings)

    Returns:
        (texture_key, success, elapsed_seconds, error_msg)
    """
    solver_bin, tex_key, board, scenario, output_path, settings = args

    if os.path.exists(output_path) and settings.get("resume", True):
        # Check if existing file has per-hand data
        try:
            with open(output_path) as f:
                existing = json.load(f)
            if existing.get("hands") and len(existing["hands"]) > 0:
                return (tex_key, True, 0, "skipped (already solved)")
        except Exception:
            pass

    solver_input = {
        "board": board,
        "oop_range": scenario["oop_range"],
        "ip_range": scenario["ip_range"],
        "starting_pot": scenario["starting_pot"],
        "effective_stack": scenario["effective_stack"],
        "hero_hand": [board[0], board[1]],  # dummy, overridden by all_hands
        "hero_position": "oop",
        "max_iterations": settings.get("max_iterations", 200),
        "target_exploitability": settings.get("target_exploitability", 0.02),
        "all_hands": True,
        "bet_sizes_oop": settings.get("bet_sizes_oop", "33%, 75%, a"),
        "bet_sizes_ip": settings.get("bet_sizes_ip", "33%, 75%, a"),
        "raise_sizes_oop": settings.get("raise_sizes_oop", "2.5x, a"),
        "raise_sizes_ip": settings.get("raise_sizes_ip", "2.5x, a"),
    }

    t0 = time.time()
    try:
        proc = subprocess.run(
            [solver_bin],
            input=json.dumps(solver_input),
            capture_output=True,
            text=True,
            timeout=settings.get("timeout", 600),
        )
        if proc.returncode != 0:
            return (tex_key, False, time.time() - t0, proc.stderr[:200])

        result = json.loads(proc.stdout)
        result["board"] = board
        result["solver_input"] = solver_input

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f)

        return (tex_key, True, time.time() - t0, None)
    except subprocess.TimeoutExpired:
        return (tex_key, False, time.time() - t0, "timeout")
    except Exception as e:
        return (tex_key, False, time.time() - t0, str(e)[:200])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute flop solutions")
    parser.add_argument("--scenario", help="Solve a specific scenario")
    parser.add_argument("--all", action="store_true", help="Solve all scenarios")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--target-exploitability", type=float, default=0.02)
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-solve timeout in seconds")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-solve already completed textures")
    parser.add_argument("--solver-bin", default=None,
                        help="Path to solver binary (default: auto-detect)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ../flop_solutions)")
    parser.add_argument("--ranges", default=None,
                        help="Path to ranges.json")
    parser.add_argument("--bet-sizes", default="33%, 75%, a",
                        help="Bet sizes for solver")

    args = parser.parse_args()

    # Find solver binary
    solver_bin = args.solver_bin
    if solver_bin is None:
        # Try to find tbl-engine in the ACR project
        acr_dir = os.environ.get("ACR_PROJECT_DIR", "")
        candidates = [
            os.path.join(acr_dir, "solver/solver-cli/target/release/tbl-engine"),
            os.path.join(acr_dir, "solver/solver-cli/target/release/tbl-engine.exe"),
            "tbl-engine",
            "tbl-engine.exe",
        ]
        for c in candidates:
            if os.path.isfile(c):
                solver_bin = c
                break
        if solver_bin is None:
            print("ERROR: solver binary not found. Use --solver-bin.", file=sys.stderr)
            sys.exit(1)

    # Find ranges
    ranges_path = args.ranges
    if ranges_path is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "data", "ranges.json"),
            os.path.join(os.environ.get("ACR_PROJECT_DIR", ""),
                         "solver", "ranges.json"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                ranges_path = c
                break
        if ranges_path is None:
            print("ERROR: ranges.json not found. Use --ranges.", file=sys.stderr)
            sys.exit(1)

    # Output directory
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "flop_solutions")

    # Load scenarios
    scenarios = load_scenarios(ranges_path)
    print(f"Loaded {len(scenarios)} scenarios from {ranges_path}")

    # Filter scenarios
    if args.scenario:
        if args.scenario not in scenarios:
            print(f"ERROR: scenario '{args.scenario}' not found")
            print(f"Available: {sorted(scenarios.keys())}")
            sys.exit(1)
        scenarios = {args.scenario: scenarios[args.scenario]}
    elif not args.all:
        print("Specify --scenario SCENARIO_ID or --all")
        print(f"Available: {sorted(scenarios.keys())}")
        sys.exit(0)

    # Generate textures
    textures = generate_all_textures()
    print(f"Generated {len(textures)} unique flop textures")

    settings = {
        "max_iterations": args.max_iterations,
        "target_exploitability": args.target_exploitability,
        "timeout": args.timeout,
        "resume": not args.no_resume,
        "bet_sizes_oop": args.bet_sizes,
        "bet_sizes_ip": args.bet_sizes,
        "raise_sizes_oop": "2.5x, a",
        "raise_sizes_ip": "2.5x, a",
    }

    # Build work items
    total_scenarios = len(scenarios)
    for s_idx, (scenario_id, scenario) in enumerate(sorted(scenarios.items())):
        out_dir = os.path.join(output_dir, scenario_id)
        os.makedirs(out_dir, exist_ok=True)

        work = []
        for tex_key, board in textures:
            out_path = os.path.join(out_dir, tex_key + ".json")
            work.append((solver_bin, tex_key, board, scenario, out_path, settings))

        print(f"\n[{s_idx+1}/{total_scenarios}] Solving {scenario_id} "
              f"({len(work)} textures, {args.workers} workers)")

        done = 0
        failed = 0
        skipped = 0
        t0 = time.time()

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(solve_one, w): w[1] for w in work}
            for future in as_completed(futures):
                tex = futures[future]
                tex_key, success, elapsed, error = future.result()
                done += 1
                if not success:
                    if "skipped" in (error or ""):
                        skipped += 1
                    else:
                        failed += 1
                        print(f"  FAIL: {tex_key}: {error}", file=sys.stderr)

                if done % 100 == 0 or done == len(work):
                    elapsed_total = time.time() - t0
                    rate = (done - skipped) / elapsed_total if elapsed_total > 0 else 0
                    remaining = (len(work) - done) / rate if rate > 0 else 0
                    print(f"  Progress: {done}/{len(work)} "
                          f"(skip={skipped}, fail={failed}, "
                          f"{rate:.1f} solve/s, ETA {remaining:.0f}s)")

        elapsed_total = time.time() - t0
        print(f"  Done: {done - skipped - failed} solved, "
              f"{skipped} skipped, {failed} failed in {elapsed_total:.0f}s")


if __name__ == "__main__":
    main()
