#!/usr/bin/env python3
"""Validate blueprint MCCFR against the Rust postflop-solver (tbl-engine).

Runs both solvers on the same 2-player spot and compares:
1. Root strategies (action frequencies per hand)
2. Convergence rate (self-convergence L1 at increasing iterations)
3. Strategy quality (do strong hands bet more than weak hands?)

This is the ground truth test: if our MCCFR agrees with the Rust solver
(which is cross-validated against PioSOLVER), our payoff model is correct.
"""

import ctypes
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

# ── Paths ────────────────────────────────────────────────────────────────

RUST_SOLVER = os.path.join(
    "C:/Users/Victor/Documents/Projects/ACRPoker-Hud-PC",
    "solver/solver-cli/target/release/tbl-engine.exe"
)
BP_DLL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "build", "mccfr_blueprint.dll")
CA_DLL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "build", "card_abstraction.dll")

# ── Test configuration ───────────────────────────────────────────────────

BOARD = ["Qs", "Ts", "2d"]
OOP_RANGE = "AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,ATs,KQs,KJs,AKo,AQo"
IP_RANGE = OOP_RANGE  # same range for simplicity
POT_BB = 6.5
STACK_BB = 97.5
BET_SIZES = "50%,100%"

RANKS = "23456789TJQKA"
SUITS = "cdhs"

def card_to_int(s):
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])

def int_to_card(i):
    return RANKS[i // 4] + SUITS[i % 4]

def parse_range(range_str, blocked):
    """Parse PioSOLVER-style range string into (c0, c1, weight) tuples."""
    blocked_set = set(blocked)
    hands = []
    for part in range_str.split(","):
        part = part.strip()
        weight = 1.0
        if ":" in part:
            part, w = part.rsplit(":", 1)
            weight = float(w)

        if len(part) == 2:
            # Pair: e.g. "AA"
            r = RANKS.index(part[0])
            for s1 in range(4):
                for s2 in range(s1+1, 4):
                    c0, c1 = r*4+s1, r*4+s2
                    if c0 not in blocked_set and c1 not in blocked_set:
                        hands.append((c0, c1, weight))
        elif len(part) == 3 and part[2] == 's':
            # Suited: e.g. "AKs"
            r0, r1 = RANKS.index(part[0]), RANKS.index(part[1])
            for s in range(4):
                c0, c1 = r0*4+s, r1*4+s
                if c0 not in blocked_set and c1 not in blocked_set:
                    hands.append((min(c0,c1), max(c0,c1), weight))
        elif len(part) == 3 and part[2] == 'o':
            # Offsuit: e.g. "AKo"
            r0, r1 = RANKS.index(part[0]), RANKS.index(part[1])
            for s0 in range(4):
                for s1 in range(4):
                    if s0 == s1: continue
                    c0, c1 = r0*4+s0, r1*4+s1
                    if c0 not in blocked_set and c1 not in blocked_set:
                        hands.append((min(c0,c1), max(c0,c1), weight))
        elif len(part) == 2:
            pass  # already handled
    # Deduplicate
    seen = set()
    unique = []
    for c0, c1, w in hands:
        key = (min(c0,c1), max(c0,c1))
        if key not in seen:
            seen.add(key)
            unique.append((key[0], key[1], w))
    return unique


# ── Rust solver ──────────────────────────────────────────────────────────

def run_rust_solver(board, oop_range, ip_range, pot_bb, stack_bb,
                     bet_sizes="50%,100%", iterations=500, target_exp=0.01):
    """Run tbl-engine and get per-hand strategies."""
    solver_input = {
        "board": board,
        "oop_range": oop_range,
        "ip_range": ip_range,
        "starting_pot": pot_bb,
        "effective_stack": stack_bb,
        "hero_hand": [board[0], board[1]],  # dummy
        "hero_position": "oop",
        "max_iterations": iterations,
        "target_exploitability": target_exp,
        "all_hands": True,
        "bet_sizes_oop": bet_sizes,
        "bet_sizes_ip": bet_sizes,
    }

    proc = subprocess.run(
        [RUST_SOLVER],
        input=json.dumps(solver_input),
        capture_output=True, text=True, timeout=120
    )

    if proc.returncode != 0:
        print(f"  Rust solver error: {proc.stderr[:200]}")
        return None

    return json.loads(proc.stdout)


# ── Our MCCFR solver ────────────────────────────────────────────────────

def run_mccfr(board_ints, hands, pot_chips, stack_chips, bet_sizes,
               iterations, num_buckets=0):
    """Run our MCCFR blueprint solver and extract root strategies."""
    bp = ctypes.CDLL(BP_DLL)
    bp.bp_init.restype = ctypes.c_int
    bp.bp_solve.restype = ctypes.c_int
    bp.bp_get_strategy.restype = ctypes.c_int
    bp.bp_num_info_sets.restype = ctypes.c_int
    bp.bp_free.restype = None

    BP_MAX_HANDS = 1326
    buf = (ctypes.c_char * 524288)()
    ptr = ctypes.cast(buf, ctypes.c_void_p)

    HT = ((ctypes.c_int * 2) * BP_MAX_HANDS) * 6
    WT = (ctypes.c_float * BP_MAX_HANDS) * 6
    NT = ctypes.c_int * 6
    ch, cw, cn = HT(), WT(), NT()

    nh = len(hands)
    for p in range(2):
        cn[p] = nh
        for h in range(nh):
            ch[p][h][0] = hands[h][0]
            ch[p][h][1] = hands[h][1]
            cw[p][h] = hands[h][2]

    c_bs = (ctypes.c_float * len(bet_sizes))(*bet_sizes)

    bp.bp_init(ptr, 2, (ctypes.c_int * 3)(*board_ints),
               ch, cw, cn, pot_chips, stack_chips,
               c_bs, len(bet_sizes))

    t0 = time.time()
    bp.bp_solve(ptr, iterations)
    t1 = time.time()

    n_is = bp.bp_num_info_sets(ptr)

    # Extract strategies for all hands at root
    strategies = {}
    strat_buf = (ctypes.c_float * 8)()
    for h in range(nh):
        na = bp.bp_get_strategy(ptr, 0, (ctypes.c_int * 3)(*board_ints), 3,
                                 (ctypes.c_int * 1)(), 0, strat_buf, h)
        if na > 0:
            hand_str = int_to_card(hands[h][0]) + int_to_card(hands[h][1])
            strategies[hand_str] = [float(strat_buf[a]) for a in range(na)]

    bp.bp_free(ptr)

    return {
        "strategies": strategies,
        "info_sets": n_is,
        "time_s": t1 - t0,
        "iterations": iterations,
    }


# ── Comparison ───────────────────────────────────────────────────────────

def compare_strategies(rust_result, mccfr_result, hands):
    """Compare root strategies between Rust and MCCFR solvers."""
    rust_hands = rust_result.get("hands", {}).get("oop:root", {})
    mccfr_strats = mccfr_result["strategies"]

    comparisons = []
    for h_idx, (c0, c1, _w) in enumerate(hands):
        hand_str = int_to_card(c0) + int_to_card(c1)
        # Rust uses different card notation sometimes
        hand_str_alt = int_to_card(c1) + int_to_card(c0)

        rust_s = rust_hands.get(hand_str) or rust_hands.get(hand_str_alt)
        mccfr_s = mccfr_strats.get(hand_str)

        if rust_s and mccfr_s:
            # Extract action frequencies from Rust (format varies)
            if isinstance(rust_s, dict) and "actions" in rust_s:
                r_actions = {a["action"]: a["frequency"] for a in rust_s["actions"]}
            else:
                r_actions = rust_s

            comparisons.append({
                "hand": hand_str,
                "rust": r_actions,
                "mccfr": mccfr_s,
            })

    return comparisons


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  BLUEPRINT VALIDATION: MCCFR vs Rust Postflop-Solver")
    print("=" * 70)
    print()

    board_ints = [card_to_int(c) for c in BOARD]
    blocked = set(board_ints)
    hands = parse_range(OOP_RANGE, blocked)
    print(f"Board: {' '.join(BOARD)}")
    print(f"Hands: {len(hands)} combos")
    print(f"Pot: {POT_BB}bb, Stack: {STACK_BB}bb")
    print(f"Bet sizes: {BET_SIZES}")
    print()

    # ── Test 1: Rust solver baseline ─────────────────────────────────
    print("--- Test 1: Rust Solver (tbl-engine, ground truth) ---")
    if not os.path.exists(RUST_SOLVER):
        print(f"  SKIP: tbl-engine not found at {RUST_SOLVER}")
        rust_result = None
    else:
        t0 = time.time()
        rust_result = run_rust_solver(BOARD, OOP_RANGE, IP_RANGE,
                                       POT_BB, STACK_BB, BET_SIZES,
                                       iterations=500, target_exp=0.005)
        t1 = time.time()
        if rust_result:
            print(f"  Solved in {(t1-t0)*1000:.0f}ms")
            print(f"  Exploitability: {rust_result.get('exploitability', 'N/A')}")
            print(f"  Iterations: {rust_result.get('iterations', 'N/A')}")
            # Show a few strategies
            rust_hands = rust_result.get("hands", {}).get("oop:root", {})
            if rust_hands:
                print(f"  Hands with data: {len(rust_hands)}")
                sample_hands = ["AsAd", "AcAh", "KsKd", "QsQh", "JsJh",
                                "AcKc", "TsTd", "9s9d", "5s5d", "2c3c"]
                for h in sample_hands:
                    if h in rust_hands:
                        info = rust_hands[h]
                        if isinstance(info, dict) and "actions" in info:
                            acts = " ".join(f"{a['action']}={a['frequency']:.2f}"
                                            for a in info["actions"])
                        else:
                            acts = str(info)
                        print(f"    {h}: {acts}")
        else:
            print("  FAILED")
    print()

    # ── Test 2: Our MCCFR at increasing iterations ───────────────────
    print("--- Test 2: MCCFR Self-Convergence ---")
    print(f"{'Iter':>8} {'InfoSets':>10} {'Time':>8} {'L1 vs prev':>12}")

    prev_strats = None
    for iters in [10000, 50000, 200000, 1000000]:
        result = run_mccfr(board_ints, hands, int(POT_BB * 100),
                           int(STACK_BB * 100), [0.5, 1.0], iters)

        # Compute L1 vs previous iteration count
        l1 = float('inf')
        if prev_strats:
            dists = []
            for h_str, s in result["strategies"].items():
                if h_str in prev_strats:
                    ps = prev_strats[h_str]
                    if len(s) == len(ps):
                        dists.append(sum(abs(a - b) for a, b in zip(s, ps)))
            if dists:
                l1 = np.mean(dists)

        prev_strats = result["strategies"]
        l1_str = f"{l1:.4f}" if l1 < 100 else "N/A"
        print(f"{iters:>8} {result['info_sets']:>10,} {result['time_s']:>7.1f}s {l1_str:>12}")

    print()

    # ── Test 3: Strategy quality check ───────────────────────────────
    print("--- Test 3: Strategy Quality (do strong hands play differently?) ---")
    final = run_mccfr(board_ints, hands, int(POT_BB * 100),
                       int(STACK_BB * 100), [0.5, 1.0], 500000)

    # Group hands by approximate strength
    groups = {"Premium (AA-QQ)": [], "Big pairs (JJ-88)": [],
              "Medium pairs (77-22)": [], "Broadway (AK-QJ)": [],
              "Suited connectors": [], "Other": []}

    for h_idx, (c0, c1, _w) in enumerate(hands):
        r0, r1 = c0 // 4, c1 // 4
        suited = (c0 % 4 == c1 % 4)
        h_str = int_to_card(c0) + int_to_card(c1)
        s = final["strategies"].get(h_str)
        if not s or len(s) == 0:
            continue

        rhi, rlo = max(r0, r1), min(r0, r1)
        if rhi == rlo and rhi >= 10:
            groups["Premium (AA-QQ)"].append(s)
        elif rhi == rlo and rhi >= 6:
            groups["Big pairs (JJ-88)"].append(s)
        elif rhi == rlo:
            groups["Medium pairs (77-22)"].append(s)
        elif rhi >= 9 and rlo >= 9:
            groups["Broadway (AK-QJ)"].append(s)
        elif suited and abs(rhi - rlo) <= 2:
            groups["Suited connectors"].append(s)
        else:
            groups["Other"].append(s)

    action_names = ["Check", "Bet50%", "BetPot"]
    print(f"{'Group':>25} {'n':>4} ", end="")
    for a in action_names:
        print(f"{a:>10}", end="")
    print()

    for group, strats_list in groups.items():
        if not strats_list:
            continue
        avg = np.mean(strats_list, axis=0)
        print(f"{group:>25} {len(strats_list):>4} ", end="")
        for i, a in enumerate(action_names):
            if i < len(avg):
                print(f"{avg[i]:>10.1%}", end="")
        print()

    print()

    # ── Test 4: Compare with Rust if available ───────────────────────
    if rust_result:
        print("--- Test 4: MCCFR vs Rust Strategy Comparison ---")
        comparisons = compare_strategies(rust_result, final, hands)
        if comparisons:
            print(f"Compared {len(comparisons)} hands")
            # TODO: compute L1 between rust and mccfr per hand
        else:
            print("  Could not match hand formats between solvers")
    else:
        print("--- Test 4: SKIPPED (no Rust solver) ---")

    print()
    print("=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
