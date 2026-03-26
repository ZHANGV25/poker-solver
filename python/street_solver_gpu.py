"""Python wrapper for the N-player CUDA single-street solver.

Wraps street_solve.dll for Pluribus-style depth-limited GPU solving.
Supports 2-6 players in a single betting round.

Usage (2-player, backward compatible):
    solver = StreetSolverGPU(
        board=["Qs", "As", "2d"],
        oop_range=[(0, 1, 1.0), ...],
        ip_range=[(2, 3, 1.0), ...],
        pot_bb=6.5, stack_bb=97.5,
    )
    solver.solve(iterations=200)
    strategy = solver.get_strategy(player=0, hand_idx=5)

Usage (N-player):
    solver = StreetSolverGPU(
        board=["Qs", "As", "2d", "7h", "4c"],
        player_ranges=[oop_hands, mp_hands, ip_hands],  # 3 players
        pot_bb=16.0, stack_bb=92.0,
    )
"""

import ctypes
import os
import platform
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Constants matching street_solve.cuh
SS_MAX_PLAYERS = 6
SS_MAX_HANDS = 200
SS_MAX_ACTIONS = 8
SS_MAX_BOARD = 5
SCALE = 100

try:
    from solver import card_to_int, int_to_card, parse_range_string
except ImportError:
    from python.solver import card_to_int, int_to_card, parse_range_string

# ── DLL loading ──────────────────────────────────────────────────────────

_LIB = None

def _get_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    solver_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_name = "street_solve.dll" if platform.system() == "Windows" else "street_solve.so"
    lib_path = os.path.join(solver_dir, "build", lib_name)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"street_solve not found at {lib_path}")
    _LIB = ctypes.CDLL(lib_path)
    _LIB.ss_build_tree.restype = ctypes.c_int
    _LIB.ss_solve_gpu.restype = ctypes.c_int
    _LIB.ss_free_tree.restype = None
    _LIB.ss_free_output.restype = None
    return _LIB

# ── ctypes structure mirrors (must match street_solve.cuh exactly) ───────

class SSNode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("player", ctypes.c_int),
        ("num_children", ctypes.c_int),
        ("first_child", ctypes.c_int),
        ("pot", ctypes.c_int),
        ("bets", ctypes.c_int * SS_MAX_PLAYERS),
        ("board_cards", ctypes.c_int * 5),
        ("num_board", ctypes.c_int),
        ("leaf_idx", ctypes.c_int),
        ("fold_player", ctypes.c_int),
        ("num_players", ctypes.c_int),
        ("active_players", ctypes.c_int * SS_MAX_PLAYERS),
    ]

class SSTreeData(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.POINTER(SSNode)),
        ("children", ctypes.POINTER(ctypes.c_int)),
        ("num_nodes", ctypes.c_int),
        ("num_children_total", ctypes.c_int),
        ("num_players_field", ctypes.c_int),  # renamed to avoid conflict
        ("hands", ((ctypes.c_int * 2) * SS_MAX_HANDS) * SS_MAX_PLAYERS),
        ("weights", (ctypes.c_float * SS_MAX_HANDS) * SS_MAX_PLAYERS),
        ("num_hands", ctypes.c_int * SS_MAX_PLAYERS),
        ("board", ctypes.c_int * SS_MAX_BOARD),
        ("num_board", ctypes.c_int),
        ("leaf_values", ctypes.POINTER(ctypes.c_float)),
        ("num_leaves", ctypes.c_int),
        ("level_order", ctypes.POINTER(ctypes.c_int)),
        ("node_depth", ctypes.POINTER(ctypes.c_int)),
        ("max_depth", ctypes.c_int),
        ("decision_node_indices", ctypes.POINTER(ctypes.c_int)),
        ("num_decision_nodes", ctypes.c_int),
        ("showdown_node_indices", ctypes.POINTER(ctypes.c_int)),
        ("num_showdown_nodes", ctypes.c_int),
        ("leaf_node_indices", ctypes.POINTER(ctypes.c_int)),
        ("num_leaf_nodes", ctypes.c_int),
        ("fold_node_indices", ctypes.POINTER(ctypes.c_int)),
        ("num_fold_nodes", ctypes.c_int),
        ("starting_pot", ctypes.c_int),
        ("effective_stack", ctypes.c_int),
        ("is_river", ctypes.c_int),
    ]

class SSOutput(ctypes.Structure):
    _fields_ = [
        ("root_strategy", ctypes.POINTER(ctypes.c_float)),
        ("root_num_actions", ctypes.c_int),
        ("root_player", ctypes.c_int),
        ("root_ev", ctypes.POINTER(ctypes.c_float)),
        ("avg_strategies", ctypes.POINTER(ctypes.c_float)),
        ("avg_strategy_node_ids", ctypes.POINTER(ctypes.c_int)),
        ("num_avg_nodes", ctypes.c_int),
        ("num_players_out", ctypes.c_int),
        ("max_hands", ctypes.c_int),
    ]

# ── Action label helpers ─────────────────────────────────────────────────

def _build_labels_at_node(tree_data, node_idx):
    """Build action labels for a decision node."""
    node = tree_data.nodes[node_idx]
    labels = []
    pot = node.pot
    acting = node.player
    acting_bet = node.bets[acting] if acting >= 0 else 0

    # Max bet among all players
    mx = 0
    for p in range(SS_MAX_PLAYERS):
        if node.bets[p] > mx:
            mx = node.bets[p]
    to_call = mx - acting_bet if acting >= 0 else 0

    for i in range(node.num_children):
        child_idx = tree_data.children[node.first_child + i]
        child = tree_data.nodes[child_idx]

        if child.type == 1:  # FOLD
            labels.append("Fold")
        elif child.type == 2 or child.type == 4:  # SHOWDOWN or LEAF
            labels.append("Call" if to_call > 0 else "Check")
        elif child.type == 0:  # DECISION
            # Determine bet amount from the change in acting player's bet
            child_bet = child.bets[acting] if acting >= 0 else 0
            bet_amount = child_bet - acting_bet

            if bet_amount == 0:
                labels.append("Check")
            elif pot > 0:
                pct = int(round(100 * bet_amount / pot))
                if to_call > 0:
                    labels.append(f"Raise {pct}%")
                else:
                    labels.append(f"Bet {pct}%")
            else:
                labels.append("Bet" if to_call == 0 else "Raise")
        else:
            labels.append(f"Action {i}")

    return labels

# ── Main solver class ────────────────────────────────────────────────────

class StreetSolverGPU:
    """N-player GPU-accelerated single-street solver.

    Supports 2-6 players in a single betting round with depth-limited
    leaf values and continuation strategy expansion (Pluribus approach).
    """

    def __init__(self, board, pot_bb, stack_bb,
                 # 2-player interface (backward compat)
                 oop_range=None, ip_range=None,
                 # N-player interface
                 player_ranges=None,
                 acting_order=None,
                 # Common options
                 bet_sizes=None, leaf_values=None,
                 use_cont_strats=True, leaf_value_fn=None):
        """
        Args:
            board: list of card strings or ints
            pot_bb: pot in BB
            stack_bb: effective stack in BB
            oop_range/ip_range: 2-player mode (backward compat)
            player_ranges: list of [(c0,c1,w), ...] per player (N-player mode)
            acting_order: list of player indices in acting order (default: [0,1,...,N-1])
            bet_sizes: bet fractions (default [0.33, 0.75, 1.5])
            leaf_values: np.array or None
            use_cont_strats: expand leaves with continuation strategies
            leaf_value_fn: callback to compute leaf values from tree
        """
        self.lib = _get_lib()

        # Parse board
        if isinstance(board[0], str):
            self.board_cards = [card_to_int(c) for c in board]
        else:
            self.board_cards = list(board)
        self.num_board = len(self.board_cards)
        self.is_river = (self.num_board == 5)

        # Build player ranges
        board_set = set(self.board_cards)

        if player_ranges is not None:
            # N-player mode
            self.num_players = len(player_ranges)
            self.player_hands = []
            for p_range in player_ranges:
                if isinstance(p_range, str):
                    hands = parse_range_string(p_range)
                else:
                    hands = list(p_range)
                # Filter + trim
                hands = [(c0, c1, w) for c0, c1, w in hands
                         if c0 not in board_set and c1 not in board_set]
                if len(hands) > SS_MAX_HANDS:
                    hands.sort(key=lambda x: -x[2])
                    hands = hands[:SS_MAX_HANDS]
                self.player_hands.append(hands)
        elif oop_range is not None and ip_range is not None:
            # 2-player backward compat
            self.num_players = 2
            self.player_hands = []
            for p_range in [oop_range, ip_range]:
                if isinstance(p_range, str):
                    hands = parse_range_string(p_range)
                else:
                    hands = list(p_range)
                hands = [(c0, c1, w) for c0, c1, w in hands
                         if c0 not in board_set and c1 not in board_set]
                if len(hands) > SS_MAX_HANDS:
                    hands.sort(key=lambda x: -x[2])
                    hands = hands[:SS_MAX_HANDS]
                self.player_hands.append(hands)
        else:
            raise ValueError("Provide oop_range+ip_range or player_ranges")

        # Backward compat aliases
        if self.num_players >= 1:
            self.oop_hands = self.player_hands[0]
        if self.num_players >= 2:
            self.ip_hands = self.player_hands[1]

        if bet_sizes is None:
            bet_sizes = [0.33, 0.75, 1.5]
        self.bet_sizes = bet_sizes

        if acting_order is None:
            acting_order = list(range(self.num_players))
        self.acting_order = acting_order

        pot = int(pot_bb * SCALE)
        stack = int(stack_bb * SCALE)

        # Build tree
        board_arr = (ctypes.c_int * self.num_board)(*self.board_cards)
        bet_arr = (ctypes.c_float * len(bet_sizes))(*bet_sizes)
        order_arr = (ctypes.c_int * len(acting_order))(*acting_order)

        self._tree = SSTreeData()
        use_cs = 1 if (use_cont_strats and not self.is_river) else 0

        err = self.lib.ss_build_tree(
            board_arr, self.num_board,
            pot, stack,
            bet_arr, len(bet_sizes),
            self.num_players, order_arr,
            1 if self.is_river else 0,
            use_cs,
            ctypes.byref(self._tree))
        if err != 0:
            raise RuntimeError("ss_build_tree failed")

        # Populate hands and weights
        for p in range(self.num_players):
            self._tree.num_hands[p] = len(self.player_hands[p])
            for i, (c0, c1, w) in enumerate(self.player_hands[p]):
                self._tree.hands[p][i][0] = c0
                self._tree.hands[p][i][1] = c1
                self._tree.weights[p][i] = w

        max_h = max(len(h) for h in self.player_hands)
        self._max_hands = max_h

        # Compute leaf values via callback if provided
        if leaf_values is None and leaf_value_fn is not None and self._tree.num_leaves > 0:
            leaf_values = leaf_value_fn(
                self._tree, self.player_hands,
                max_h, int(pot_bb * SCALE))

        # Upload leaf values
        if leaf_values is not None and self._tree.num_leaves > 0:
            expected = self._tree.num_leaves * self.num_players * max_h
            flat = leaf_values.astype(np.float32).flatten()
            if len(flat) != expected:
                raise ValueError(
                    f"leaf_values mismatch: {len(flat)} vs {expected} "
                    f"(leaves={self._tree.num_leaves}, players={self.num_players}, "
                    f"max_hands={max_h})")
            arr = (ctypes.c_float * len(flat))(*flat)
            self._leaf_arr = arr
            self._tree.leaf_values = ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))
        elif not self.is_river and self._tree.num_leaves > 0:
            print("[WARNING] Zero leaf values — no blueprint data.", file=sys.stderr)
            n = self._tree.num_leaves * self.num_players * max_h
            arr = (ctypes.c_float * n)()
            self._leaf_arr = arr
            self._tree.leaf_values = ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))

        self._output = SSOutput()
        self._solved = False
        self._freed = False

    def solve(self, iterations=200):
        err = self.lib.ss_solve_gpu(
            ctypes.byref(self._tree), iterations, ctypes.byref(self._output))
        if err != 0:
            raise RuntimeError("ss_solve_gpu failed")
        self._solved = True

    def get_strategy(self, player, hand_str=None, hand_idx=None,
                     frozen_actions=None):
        """Get strategy for a specific hand at this player's CURRENT decision.

        Implements Pluribus A3 (strategy freezing): when hero has already
        acted at a node earlier this street, the strategy at that node is
        frozen to 100% on the previously-chosen action FOR HERO'S HAND ONLY.
        Other hands at that info set remain free.

        The solver walks the tree following the frozen action sequence to find
        hero's actual current decision node, then returns the solver's strategy
        at that node.

        Args:
            player: int (0..N-1) or "oop"(0) / "ip"(1) for 2-player
            hand_str: e.g., "AhKh"
            hand_idx: index into player's hand array
            frozen_actions: list of action labels that hero took at previous
                decision nodes this street, in order. The solver follows these
                actions through the tree to find the current decision node.
                If None or empty, returns strategy at hero's first decision.

        Returns:
            dict {action_label: frequency}
        """
        if not self._solved:
            raise RuntimeError("Must call solve() first")

        player_idx = self._resolve_player(player)
        if hand_idx is None:
            hand_idx = self._find_hand(player_idx, hand_str)

        max_h = self._max_hands

        # If no frozen actions, return strategy at hero's first decision
        if not frozen_actions:
            if self._output.root_player == player_idx:
                root_nh = len(self.player_hands[player_idx])
                root_na = self._output.root_num_actions
                labels = _build_labels_at_node(self._tree, 0)
                result = {}
                for a in range(min(root_na, len(labels))):
                    freq = self._output.root_strategy[a * root_nh + hand_idx]
                    if freq > 0.001:
                        result[labels[a]] = float(freq)
                return result
            else:
                return self._get_strategy_at_player_node(player_idx, hand_idx)

        # Walk the tree following frozen actions to find the current node.
        # frozen_actions contains ALL actions (both hero and opponent) that
        # led to the current state, in sequence. We follow each action
        # through the tree regardless of whose decision node it is.
        frozen_idx = 0
        current_node = 0  # start at root

        while frozen_idx < len(frozen_actions) and current_node < self._tree.num_nodes:
            node = self._tree.nodes[current_node]
            if node.type != 0:  # not a decision node
                break

            # Follow the next frozen action at this node (any player's node)
            frozen_label = frozen_actions[frozen_idx]
            labels = _build_labels_at_node(self._tree, current_node)

            matched = False
            for c in range(node.num_children):
                if c < len(labels) and self._match_action_label(
                        labels[c], frozen_label):
                    child_idx = self._tree.children[node.first_child + c]
                    current_node = child_idx
                    frozen_idx += 1
                    matched = True
                    break

            if not matched:
                # Frozen action not found in tree — stop walking
                break

        # Now current_node should be at hero's next decision (or we exhausted frozen actions)
        node = self._tree.nodes[current_node]
        if node.type == 0 and node.player == player_idx:
            # Found hero's current decision node — extract avg strategy
            return self._get_strategy_at_node(current_node, player_idx, hand_idx)

        # Fallback: return strategy at hero's first unfrozen decision node
        return self._get_strategy_at_player_node(player_idx, hand_idx)

    @staticmethod
    def _match_action_label(tree_label, frozen_label):
        """Fuzzy match action labels (e.g. "Check" matches "check")."""
        return tree_label.lower().strip() == frozen_label.lower().strip()

    def _get_strategy_at_node(self, node_idx, player_idx, hand_idx):
        """Extract avg strategy at a specific tree node for a specific hand."""
        max_h = self._max_hands
        for di in range(self._output.num_avg_nodes):
            nidx = self._output.avg_strategy_node_ids[di]
            if nidx != node_idx:
                continue
            node = self._tree.nodes[nidx]
            if node.player != player_idx or node.type != 0:
                continue
            na = node.num_children
            labels = _build_labels_at_node(self._tree, nidx)
            result = {}
            for a in range(min(na, len(labels))):
                freq = self._output.avg_strategies[
                    di * SS_MAX_ACTIONS * max_h + a * max_h + hand_idx]
                if freq > 0.001:
                    result[labels[a]] = float(freq)
            return result
        return {}

    def _get_strategy_at_player_node(self, player_idx, hand_idx):
        """Find first decision node for this player, extract avg strategy."""
        max_h = self._max_hands
        for di in range(self._output.num_avg_nodes):
            nidx = self._output.avg_strategy_node_ids[di]
            node = self._tree.nodes[nidx]
            if node.player != player_idx or node.type != 0:
                continue
            na = node.num_children
            labels = _build_labels_at_node(self._tree, nidx)
            result = {}
            for a in range(min(na, len(labels))):
                freq = self._output.avg_strategies[
                    di * SS_MAX_ACTIONS * max_h + a * max_h + hand_idx]
                if freq > 0.001:
                    result[labels[a]] = float(freq)
            return result
        return {}

    def get_all_strategies(self, player):
        """Get strategies for ALL hands at player's first decision node."""
        if not self._solved:
            raise RuntimeError("Must call solve() first")
        player_idx = self._resolve_player(player)
        result = {}
        for h in range(len(self.player_hands[player_idx])):
            strat = self.get_strategy(player_idx, hand_idx=h)
            if strat:
                result[h] = strat
        return result

    def get_avg_strategy(self, player):
        """Get weighted-average strategy at player's first decision.
        Returns {(card0,card1): {action: freq}}."""
        if not self._solved:
            raise RuntimeError("Must call solve() first")
        player_idx = self._resolve_player(player)
        hands = self.player_hands[player_idx]
        max_h = self._max_hands

        for di in range(self._output.num_avg_nodes):
            nidx = self._output.avg_strategy_node_ids[di]
            node = self._tree.nodes[nidx]
            if node.type != 0 or node.player != player_idx:
                continue
            na = node.num_children
            labels = _build_labels_at_node(self._tree, nidx)
            result = {}
            for h in range(len(hands)):
                c0, c1, w = hands[h]
                key = (min(c0, c1), max(c0, c1))
                strat = {}
                for a in range(min(na, len(labels))):
                    freq = self._output.avg_strategies[
                        di * SS_MAX_ACTIONS * max_h + a * max_h + h]
                    if freq > 0.001:
                        strat[labels[a]] = float(freq)
                result[key] = strat
            return result
        return {}

    def get_avg_strategy_probs(self, player, action_label):
        """Get P(action|hand) for all hands. For range narrowing."""
        avg = self.get_avg_strategy(player)
        return {k: v.get(action_label, 0.0) for k, v in avg.items()}

    def _resolve_player(self, player):
        if isinstance(player, int):
            return player
        if player == "oop":
            return 0
        if player == "ip":
            return 1
        return int(player)

    def _find_hand(self, player_idx, hand_str):
        c0 = card_to_int(hand_str[:2])
        c1 = card_to_int(hand_str[2:])
        key = (min(c0, c1), max(c0, c1))
        for i, (h0, h1, w) in enumerate(self.player_hands[player_idx]):
            if (h0, h1) == key or (h1, h0) == key:
                return i
        raise ValueError(f"Hand {hand_str} not in player {player_idx}'s range")

    def free(self):
        if self._freed:
            return
        self._freed = True
        if hasattr(self, '_tree') and self._tree.nodes:
            try:
                self.lib.ss_free_tree(ctypes.byref(self._tree))
            except Exception:
                pass
        if hasattr(self, '_output') and self._solved:
            try:
                self.lib.ss_free_output(ctypes.byref(self._output))
            except Exception:
                pass

    def __del__(self):
        self.free()
