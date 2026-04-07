#!/usr/bin/env python3
"""Extract preflop strategies from unified_blueprint.bps into static JSON.

Traces the preflop action tree to find each position's decision nodes:
- "open": all earlier positions folded → this position is first to open
- "vs-open": an earlier position raised → this position faces a raise
- "vs-3bet": opened, got 3-bet → facing 3-bet
- "vs-4bet": 3-bet, got 4-bet → facing 4-bet

Output: static JSON for the Next.js frontend (no API server needed).
"""

import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
from blueprint_v2 import BlueprintV2, _compute_action_hash

RANKS = "AKQJT98765432"
POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
POS_TO_PLAYER = {"UTG": 2, "MP": 3, "CO": 4, "BTN": 5, "SB": 0, "BB": 1}
PLAYER_TO_POS = {v: k for k, v in POS_TO_PLAYER.items()}

# Preflop acting order
ACTING_ORDER = [2, 3, 4, 5, 0, 1]


def generate_hand_grid():
    hands = []
    for row in range(13):
        for col in range(13):
            r1, r2 = RANKS[row], RANKS[col]
            if row == col:
                hands.append(f"{r1}{r2}")
            elif col < row:
                hands.append(f"{r2}{r1}s")
            else:
                hands.append(f"{r1}{r2}o")
    return hands


def hand_class_to_bucket(hand):
    if len(hand) == 2:
        return RANKS.index(hand[0])
    r0, r1 = RANKS.index(hand[0]), RANKS.index(hand[1])
    is_suited = hand.endswith("s")
    offset = 13 if is_suited else 91
    for i in range(13):
        for j in range(i + 1, 13):
            if i == r0 and j == r1:
                return offset
            offset += 1
    return 0


def build_action_labels(num_actions, raise_sizes_bb):
    """Build action labels for a node based on number of actions.

    Action 0 is always fold, action 1 is always call/check.
    Actions 2+ are raise sizes from the tiered bet schedule.
    The last action is typically all-in (8x = effectively all-in at 100bb).
    """
    labels = []
    if num_actions >= 1:
        labels.append("fold")
    if num_actions >= 2:
        labels.append("call")

    n_raises = num_actions - 2
    if n_raises <= 0:
        return labels

    # Use the provided raise sizes; if more raises than sizes, the last is all-in
    sizes_to_use = raise_sizes_bb[:n_raises] if len(raise_sizes_bb) >= n_raises else raise_sizes_bb
    for i, sz in enumerate(sizes_to_use):
        if sz >= 50:  # >50bb is effectively all-in
            labels.append("allin")
        else:
            labels.append(f"raise_{sz:.1f}")
    # Pad if we ran out of sizes
    while len(labels) < num_actions:
        labels.append("allin")

    return labels


def compute_raise_sizes_for_node(facing_bet_bb, pot_bb, tier_multipliers, stack_bb=100):
    """Compute the actual raise sizes (in bb) for a given node.

    For preflop opens at 100bb, blinds 0.5/1.0, pot=1.5bb:
        0.5x pot raise = call(1bb) + 0.5*(1.5+1) = 2.25bb total
        0.7x pot raise = call(1bb) + 0.7*(1.5+1) = 2.75bb total
        1.0x pot raise = call(1bb) + 1.0*(1.5+1) = 3.5bb total
    Simplified: raise to size_x * pot + to_call.
    """
    sizes = []
    to_call = facing_bet_bb
    for mult in tier_multipliers:
        # Total raise amount (in chips above current bet)
        raise_amount = to_call + mult * (pot_bb + to_call)
        total_bet = facing_bet_bb + raise_amount
        if total_bet >= stack_bb * 0.9:
            sizes.append(stack_bb)  # all-in
        else:
            sizes.append(round(total_bet, 1))
    return sizes


def build_range_strategy(strats, hand_grid, pos, raise_sizes_bb):
    """Convert [169, na] array to RangeStrategy keeping raise sizes separate."""
    na = strats.shape[1]
    labels = build_action_labels(na, raise_sizes_bb)

    strategies = []
    for hand in hand_grid:
        bucket = hand_class_to_bucket(hand)
        probs = strats[bucket] if bucket < len(strats) else None

        if probs is None:
            strategies.append({"hand": hand, "actions": [{"action": "fold", "frequency": 1.0}]})
            continue

        actions = []
        for i in range(min(na, len(labels))):
            freq = round(float(probs[i]), 4)
            if freq > 0.005:
                actions.append({"action": labels[i], "frequency": freq})

        if not actions:
            actions = [{"action": "fold", "frequency": 1.0}]

        strategies.append({"hand": hand, "actions": actions})

    return {"position": pos, "street": "preflop", "strategies": strategies}


def main():
    bps_path = sys.argv[1] if len(sys.argv) > 1 else "blueprint_data/unified_blueprint.bps"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading {bps_path} (preflop only)...", flush=True)
    t0 = time.time()

    bp = BlueprintV2(".", streets_to_load=[0])
    if not bp.load_unified(bps_path):
        print(f"Failed to load {bps_path}")
        sys.exit(1)

    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    table = bp._textures.get("__unified__")
    if not table:
        print("No unified table")
        sys.exit(1)

    print(f"Total preflop nodes: {len(table)}", flush=True)
    hand_grid = generate_hand_grid()

    # Build lookup: (player, action_hash) -> strats
    lookup = {}
    for (bh, ah, player, street), strats in table.items():
        lookup[(player, ah)] = strats

    result = {}

    # ── OPEN scenario: trace fold path to each position ──
    # UTG root: action_history = [] (everyone acts for first time)
    # MP opens: UTG folded → action_history = [0]
    # CO opens: UTG fold, MP fold → [0, 0]
    # BTN opens: UTG, MP, CO fold → [0, 0, 0]
    # SB opens: UTG, MP, CO, BTN fold → [0, 0, 0, 0]
    # BB faces limp/open: depends on what SB does

    fold_paths = {
        "UTG": [],          # First to act
        "MP":  [0],         # UTG folded
        "CO":  [0, 0],      # UTG, MP folded
        "BTN": [0, 0, 0],   # UTG, MP, CO folded
        "SB":  [0, 0, 0, 0],  # All folded to SB
        "BB":  [0, 0, 0, 0, 0],  # All folded to BB (SB folds or completes)
    }

    # Open raise sizes (in BB total bet) for the standard 100bb spot.
    # Tier 0 multipliers from blueprint: [0.5, 0.7, 1.0]
    # Plus 8x raise = effectively all-in
    OPEN_RAISE_SIZES = compute_raise_sizes_for_node(
        facing_bet_bb=1.0, pot_bb=1.5, tier_multipliers=[0.5, 0.7, 1.0]
    ) + [100.0]  # all-in

    # 3-bet sizes when facing an open
    THREEBET_SIZES = compute_raise_sizes_for_node(
        facing_bet_bb=2.5, pot_bb=4.0, tier_multipliers=[0.7, 1.0]
    ) + [100.0]  # all-in

    print("\n=== OPEN RANGES ===", flush=True)
    for pos in POSITIONS:
        player = POS_TO_PLAYER[pos]
        ah = _compute_action_hash(fold_paths[pos])
        strats = lookup.get((player, ah))

        if strats is not None:
            range_strat = build_range_strategy(strats, hand_grid, pos, OPEN_RAISE_SIZES)
            n_play = sum(1 for h in range_strat["strategies"]
                        if any(a["action"] != "fold" for a in h["actions"]))
            print(f"  {pos}: {n_play}/169 hands play, {strats.shape[1]} actions", flush=True)

            for check_hand in ["AA", "AKs", "AKo", "T9s"]:
                for h in range_strat["strategies"]:
                    if h["hand"] == check_hand:
                        acts = ", ".join(f"{a['action']}={a['frequency']}" for a in h["actions"])
                        print(f"    {check_hand}: {acts}")
        else:
            print(f"  {pos}: NOT FOUND (action_hash=0x{ah:016X})", flush=True)
            range_strat = _empty_range(pos, hand_grid)

        result[pos] = {"open": range_strat}

    # ── VS-OPEN scenario: earlier positions fold, one raises, hero faces it ──
    print("\n=== VS-OPEN RANGES ===", flush=True)
    for pos_idx, pos in enumerate(POSITIONS):
        player = POS_TO_PLAYER[pos]
        acting = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
        my_order_idx = acting.index(pos)

        best_strats = None
        opener = None
        for opener_idx in range(my_order_idx):
            path = [0] * opener_idx + [2]
            path += [0] * (my_order_idx - opener_idx - 1)
            ah = _compute_action_hash(path)
            s = lookup.get((player, ah))
            if s is not None:
                best_strats = s
                opener = acting[opener_idx]

        if best_strats is not None:
            range_strat = build_range_strategy(best_strats, hand_grid, pos, THREEBET_SIZES)
            n_play = sum(1 for h in range_strat["strategies"]
                        if any(a["action"] != "fold" for a in h["actions"]))
            print(f"  {pos} vs {opener} open: {n_play}/169 hands play", flush=True)
            result[pos]["vs-open"] = range_strat
        else:
            print(f"  {pos}: no vs-open found", flush=True)
            result[pos]["vs-open"] = _empty_range(pos, hand_grid)

    # Write output
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'nexusgto',
            'src', 'data', 'preflop-strategies.json'
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, separators=(',', ':'))

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nWrote {output_path} ({size_kb:.0f} KB)")
    print(f"Total time: {time.time() - t0:.1f}s")


def _empty_range(pos, hand_grid):
    return {
        "position": pos,
        "street": "preflop",
        "strategies": [{"hand": h, "actions": [{"action": "fold", "frequency": 1.0}]} for h in hand_grid],
    }


if __name__ == "__main__":
    main()
