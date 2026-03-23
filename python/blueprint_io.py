"""Blueprint I/O — read precomputed flop/turn strategies for range narrowing.

Loads precomputed solutions from the ACR HUD's flop_solutions directory
and extracts P(action|hand) for Bayesian range narrowing.

Usage:
    bp = Blueprint("path/to/flop_solutions")

    # Get action probabilities for range narrowing
    probs = bp.get_action_probs("CO_vs_BB_srp", board=["Qs","As","2d"],
                                 player="oop", action="bet")
    # probs = {(card0, card1): P(bet|hand), ...}

    # Get continuation values for depth-limited solving
    leaf_vals = bp.get_continuation_values("CO_vs_BB_srp",
                                            board=["Qs","As","2d"],
                                            player="oop")
"""

import json
import lzma
import os
import sys
from typing import Dict, List, Optional, Tuple

# Import from sibling module
try:
    from solver import card_to_int, int_to_card
except ImportError:
    from python.solver import card_to_int, int_to_card

# ── Suit isomorphism (from precompute_flops.py) ─────────────────────────────

RANKS = "23456789TJQKA"
RANK_VALUE = {r: i for i, r in enumerate(RANKS)}
SUITS = "cdhs"


def texture_key(board_cards):
    """Compute the canonical texture key for a flop.

    Args:
        board_cards: list of 3 card strings, e.g. ["Qs", "As", "2d"]

    Returns:
        tuple of (texture_key_string, suit_map) where suit_map maps
        actual suits to canonical suits for hand remapping.
    """
    ranks = [c[0] for c in board_cards]
    suits = [c[1] for c in board_cards]

    # Sort by rank descending
    ranked = sorted(zip(ranks, suits), key=lambda x: -RANK_VALUE[x[0]])
    ranks = [r for r, s in ranked]
    suits = [s for r, s in ranked]

    rank_str = "".join(ranks)

    # Determine suit pattern
    s0, s1, s2 = suits
    if s0 == s1 == s2:
        pattern = "_m"
        suit_map = {s0: "s"}  # all map to spades
    elif s0 != s1 and s1 != s2 and s0 != s2:
        pattern = "_r"
        suit_map = {s0: "s", s1: "h", s2: "d"}  # rainbow: s, h, d
    else:
        # Two cards share a suit (flush draw)
        if s0 == s1:
            pattern = "_fd12"
            suit_map = {s0: "s", s2: "h"}
        elif s0 == s2:
            pattern = "_fd13"
            suit_map = {s0: "s", s1: "h"}
        else:  # s1 == s2
            pattern = "_fd23"
            suit_map = {s1: "s", s0: "h"}

    return rank_str + pattern, suit_map


def remap_hand(hand_str, suit_map):
    """Remap a hand's suits to match the canonical board.

    Args:
        hand_str: e.g. "AhKd"
        suit_map: dict mapping actual suits to canonical suits

    Returns:
        Remapped hand string, e.g. "AsKh"
    """
    r0, s0 = hand_str[0], hand_str[1]
    r1, s1 = hand_str[2], hand_str[3]

    new_s0 = suit_map.get(s0, s0)
    new_s1 = suit_map.get(s1, s1)

    return r0 + new_s0 + r1 + new_s1


# ── Blueprint class ──────────────────────────────────────────────────────────

class Blueprint:
    """Read precomputed flop solutions for range narrowing and leaf values."""

    def __init__(self, solutions_dir):
        """
        Args:
            solutions_dir: path to flop_solutions/ directory containing
                           scenario subdirectories with .json files
        """
        self.solutions_dir = solutions_dir
        self._cache = {}  # (scenario, texture_key) -> dict
        self._scenarios = self._scan_scenarios()

    def _scan_scenarios(self):
        """Find available scenario directories."""
        scenarios = {}
        if not os.path.isdir(self.solutions_dir):
            return scenarios
        for name in os.listdir(self.solutions_dir):
            path = os.path.join(self.solutions_dir, name)
            if os.path.isdir(path):
                # Count files (.json or .json.lzma)
                n = len([f for f in os.listdir(path)
                         if f.endswith(".json") or f.endswith(".json.lzma")])
                if n > 0:
                    scenarios[name] = n
        return scenarios

    def _load_solution(self, scenario_id, tex_key):
        """Load a single solution file (cached). Supports JSON and LZMA."""
        cache_key = (scenario_id, tex_key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try LZMA first (compressed), then JSON
        lzma_path = os.path.join(self.solutions_dir, scenario_id,
                                  tex_key + ".json.lzma")
        json_path = os.path.join(self.solutions_dir, scenario_id,
                                  tex_key + ".json")

        data = None
        if os.path.exists(lzma_path):
            with open(lzma_path, 'rb') as f:
                data = json.loads(lzma.decompress(f.read()))
        elif os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
        else:
            return None

        self._cache[cache_key] = data
        return data

    def get_action_probs(self, scenario_id, board, player, action,
                         action_path=None):
        """Get P(action|hand) for each hand in a player's range.

        Used for Bayesian range narrowing.

        Args:
            scenario_id: e.g. "CO_vs_BB_srp"
            board: list of 3 card strings (flop)
            player: "oop" or "ip"
            action: action to get probability for, e.g. "Check", "Bet", "Call"
            action_path: optional prior action path, e.g. ["check", "bet"]

        Returns:
            dict mapping (card0_int, card1_int) -> float probability
        """
        tex_key, suit_map = texture_key(board)
        data = self._load_solution(scenario_id, tex_key)
        if data is None:
            return {}

        hands_data = data.get("hands")
        if not hands_data:
            return {}

        # Build node key
        if action_path:
            node_key = "{}:{}".format(player, ":".join(action_path))
        else:
            node_key = "{}:root".format(player)

        # Try exact key, then fallback to simple position key
        node = hands_data.get(node_key)
        if node is None:
            node = hands_data.get(player)
        if node is None:
            return {}

        # Build inverse suit map for remapping hand strings back to actual suits
        inv_suit_map = {v: k for k, v in suit_map.items()}

        probs = {}
        for hand_str, strat in node.items():
            # Remap from canonical suits back to actual suits
            actual_hand = remap_hand(hand_str, inv_suit_map)

            c0 = card_to_int(actual_hand[:2])
            c1 = card_to_int(actual_hand[2:])
            key = (min(c0, c1), max(c0, c1))

            # Find matching action
            p = 0.0
            for act_info in strat.get("actions", []):
                act_name = act_info.get("action", "")
                if action.lower() in act_name.lower():
                    p += act_info.get("frequency", 0.0)

            probs[key] = p

        return probs

    def get_continuation_values(self, scenario_id, board, player,
                                action_path=None):
        """Get per-hand continuation EVs for depth-limited leaf values.

        Args:
            scenario_id: e.g. "CO_vs_BB_srp"
            board: list of 3 card strings (flop)
            player: "oop" or "ip"
            action_path: optional prior action path

        Returns:
            dict mapping (card0_int, card1_int) -> float EV
        """
        tex_key, suit_map = texture_key(board)
        data = self._load_solution(scenario_id, tex_key)
        if data is None:
            return {}

        hands_data = data.get("hands")
        if not hands_data:
            return {}

        if action_path:
            node_key = "{}:{}".format(player, ":".join(action_path))
        else:
            node_key = "{}:root".format(player)

        node = hands_data.get(node_key)
        if node is None:
            node = hands_data.get(player)
        if node is None:
            return {}

        inv_suit_map = {v: k for k, v in suit_map.items()}

        values = {}
        for hand_str, strat in node.items():
            actual_hand = remap_hand(hand_str, inv_suit_map)
            c0 = card_to_int(actual_hand[:2])
            c1 = card_to_int(actual_hand[2:])
            key = (min(c0, c1), max(c0, c1))
            values[key] = strat.get("ev", 0.0)

        return values

    def get_all_action_probs(self, scenario_id, board, player,
                              action_path=None):
        """Get full strategy (all action probabilities) per hand.

        Returns:
            dict mapping (card0, card1) -> {action_name: frequency, ...}
        """
        tex_key, suit_map = texture_key(board)
        data = self._load_solution(scenario_id, tex_key)
        if data is None:
            return {}

        hands_data = data.get("hands")
        if not hands_data:
            return {}

        if action_path:
            node_key = "{}:{}".format(player, ":".join(action_path))
        else:
            node_key = "{}:root".format(player)

        node = hands_data.get(node_key)
        if node is None:
            node = hands_data.get(player)
        if node is None:
            return {}

        inv_suit_map = {v: k for k, v in suit_map.items()}

        result = {}
        for hand_str, strat in node.items():
            actual_hand = remap_hand(hand_str, inv_suit_map)
            c0 = card_to_int(actual_hand[:2])
            c1 = card_to_int(actual_hand[2:])
            key = (min(c0, c1), max(c0, c1))

            actions = {}
            for act_info in strat.get("actions", []):
                act_name = act_info.get("action", "")
                freq = act_info.get("frequency", 0.0)
                if freq > 0.001:
                    actions[act_name] = freq
            result[key] = actions

        return result

    @property
    def available_scenarios(self):
        return dict(self._scenarios)
