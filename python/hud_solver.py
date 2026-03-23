"""HUD Solver — high-level interface for the ACR Poker HUD.

Wraps the solver, range narrowing, and blueprint I/O into a single
interface that the HUD can call per-decision.

Usage in cdp_ui.py:
    from hud_solver import HUDSolver

    hs = HUDSolver(blueprint_dir="path/to/flop_solutions")

    # Per hand:
    hs.new_hand(hero_pos="BB", villain_pos="CO", scenario_type="srp")

    # Preflop: hero defends BB vs CO open
    # (range narrowing happens based on scenario)

    # Flop dealt:
    result = hs.on_street("flop",
        board=["Qs", "As", "2d"],
        hero_cards=["5d", "4d"],
        villain_action="bet",      # villain bet
        hero_turn=True)             # it's hero's turn

    print(result)
    # {'action': 'Call', 'frequency': 0.65, 'ev': 1.2,
    #  'all_actions': [{'Check': 0.35}, {'Call': 0.65}]}
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Import solver components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from solver import card_to_int, int_to_card, parse_range_string, SCALE, MAX_ACTIONS
    from range_narrowing import RangeNarrower
    from blueprint_io import Blueprint
    from solver_pool import SolverPool
except ImportError:
    from python.solver import card_to_int, int_to_card, parse_range_string, SCALE, MAX_ACTIONS
    from python.range_narrowing import RangeNarrower
    from python.blueprint_io import Blueprint
    from python.solver_pool import SolverPool


class HUDSolver:
    """High-level solver for the poker HUD.

    Manages range tracking, blueprint lookups, and runtime solving
    for a single table. Create one instance per table.
    """

    def __init__(self, blueprint_dir=None, solver_pool=None):
        """
        Args:
            blueprint_dir: path to flop_solutions/ directory
            solver_pool: shared SolverPool instance (creates one if None)
        """
        self.blueprint = Blueprint(blueprint_dir) if blueprint_dir else None
        self.pool = solver_pool or SolverPool(max_workers=2)

        self.narrower = RangeNarrower()
        self.scenario_id = None
        self.hero_pos = None
        self.villain_pos = None
        self.hero_player = None  # "oop" or "ip"

        self._pending_solve = None  # request_id for async solve
        self._last_result = None
        self._current_street = None
        self._board = []

    def new_hand(self, hero_pos, villain_pos, scenario_type="srp",
                 ranges_json_path=None):
        """Initialize for a new hand.

        Args:
            hero_pos: e.g. "BB", "CO"
            villain_pos: e.g. "CO", "BTN"
            scenario_type: "srp" or "3bp"
            ranges_json_path: path to ranges.json for loading range strings
        """
        self.hero_pos = hero_pos
        self.villain_pos = villain_pos
        self.narrower = RangeNarrower()
        self._pending_solve = None
        self._last_result = None
        self._current_street = "preflop"
        self._board = []

        # Determine OOP/IP
        post_order = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
        h_idx = post_order.index(hero_pos) if hero_pos in post_order else 99
        v_idx = post_order.index(villain_pos) if villain_pos in post_order else 99

        if h_idx < v_idx:
            self.hero_player = "oop"
            oop_pos, ip_pos = hero_pos, villain_pos
        else:
            self.hero_player = "ip"
            oop_pos, ip_pos = villain_pos, hero_pos

        self.scenario_id = "{}_vs_{}_{}".format(oop_pos, ip_pos, scenario_type)

        # Load initial ranges from blueprint or ranges.json
        # For now, use a default range set
        if ranges_json_path and os.path.exists(ranges_json_path):
            import json
            with open(ranges_json_path) as f:
                ranges_data = json.load(f)
            # Look up ranges based on scenario
            hero_range_str = self._get_range_str(ranges_data, hero_pos,
                                                  villain_pos, scenario_type,
                                                  is_hero=True)
            villain_range_str = self._get_range_str(ranges_data, hero_pos,
                                                     villain_pos, scenario_type,
                                                     is_hero=False)
        else:
            # Fallback: wide ranges
            hero_range_str = "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,AKo,AQo,AJo,KQs,KJs,KTs,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s"
            villain_range_str = hero_range_str

        hero_hands = parse_range_string(hero_range_str)
        villain_hands = parse_range_string(villain_range_str)

        self.narrower.set_initial_range("hero", hero_hands)
        self.narrower.set_initial_range("villain", villain_hands)

    def _get_range_str(self, ranges_data, hero_pos, villain_pos,
                        scenario_type, is_hero):
        """Look up range string from ranges.json data."""
        if scenario_type == "srp":
            if is_hero:
                vs_key = "{}_vs_{}".format(hero_pos, villain_pos)
                vs = ranges_data.get("vs_rfi", {}).get(vs_key, {})
                return vs.get("call", "")
            else:
                return ranges_data.get("rfi", {}).get(villain_pos, "")
        elif scenario_type == "3bp":
            if is_hero:
                return ranges_data.get("vs_3bet", {}).get(hero_pos, {}).get("call", "")
            else:
                vs_key = "{}_vs_{}".format(villain_pos, hero_pos)
                return ranges_data.get("vs_rfi", {}).get(vs_key, {}).get("3bet", "")
        return ""

    def on_villain_action(self, street, action, board=None):
        """Process a villain action for range narrowing.

        Args:
            street: "flop", "turn", "river"
            action: "check", "bet", "bet33", "bet75", "call", "raise", "fold"
            board: current board cards (list of strings)
        """
        if board:
            self._board = board

        if self.blueprint and self.scenario_id:
            # Get P(action|hand) from blueprint
            bp_player = "oop" if self.hero_player == "ip" else "ip"
            probs = self.blueprint.get_action_probs(
                self.scenario_id, self._board[:3],
                bp_player, action)
            if probs:
                self.narrower.update("villain", action, probs)
                return

        # Fallback: uniform narrowing (no blueprint data)
        # Just reduce weight slightly for all hands
        pass

    def on_hero_action(self, street, action, board=None):
        """Process hero's action for range tracking."""
        if board:
            self._board = board

        if self.blueprint and self.scenario_id:
            probs = self.blueprint.get_action_probs(
                self.scenario_id, self._board[:3],
                self.hero_player, action)
            if probs:
                self.narrower.update("hero", action, probs)

    def get_strategy(self, board, hero_cards, street="river"):
        """Get the solver's recommended strategy for hero's hand.

        Runs a depth-limited re-solve with narrowed ranges.

        Args:
            board: current board cards (list of strings)
            hero_cards: hero's hole cards (list of 2 strings)
            street: "flop", "turn", "river"

        Returns:
            dict with:
                'actions': list of {action, frequency}
                'ev': expected value
                'solving': True if solve is still running
                'time_ms': solve time
        """
        self._board = board
        self._current_street = street

        # Get narrowed ranges
        hero_hands = self.narrower.get_weighted_hands("hero")
        villain_hands = self.narrower.get_weighted_hands("villain")

        if not hero_hands or not villain_hands:
            return {'actions': [], 'ev': 0, 'solving': False, 'error': 'empty range'}

        # Filter by board blockers
        board_ints = [card_to_int(c) for c in board]
        blocked = set(board_ints)
        hero_hands = [(c0, c1, w) for c0, c1, w in hero_hands
                      if c0 not in blocked and c1 not in blocked]
        villain_hands = [(c0, c1, w) for c0, c1, w in villain_hands
                         if c0 not in blocked and c1 not in blocked]

        if len(board) < 5:
            # Non-river: use blueprint lookup if available
            if self.blueprint and self.scenario_id and len(board) == 3:
                return self._blueprint_lookup(board, hero_cards)
            return {'actions': [], 'ev': 0, 'solving': False,
                    'error': 'need blueprint for non-river'}

        # River: runtime solve with narrowed ranges
        if self.hero_player == "oop":
            oop_hands, ip_hands = hero_hands, villain_hands
        else:
            oop_hands, ip_hands = villain_hands, hero_hands

        # Determine pot and stack from context (simplified)
        pot = 1000  # default 10 BB
        stack = 9000  # default 90 BB

        # Check for pending solve
        if self._pending_solve is not None:
            result = self.pool.get_result(self._pending_solve)
            if result:
                self._pending_solve = None
                self._last_result = result
                return self._format_result(result, hero_cards)
            else:
                return {'actions': [], 'ev': 0, 'solving': True, 'time_ms': 0}

        # Submit new solve
        self._pending_solve = self.pool.submit(
            board=board_ints,
            oop_hands=oop_hands,
            ip_hands=ip_hands,
            pot=pot,
            stack=stack,
            bet_sizes=[0.33, 0.75],
            iterations=500,
        )

        return {'actions': [], 'ev': 0, 'solving': True, 'time_ms': 0}

    def _blueprint_lookup(self, board, hero_cards):
        """Look up strategy from precomputed blueprint."""
        if not self.blueprint:
            return {'actions': [], 'ev': 0, 'solving': False}

        all_probs = self.blueprint.get_all_action_probs(
            self.scenario_id, board, self.hero_player)

        if not all_probs:
            return {'actions': [], 'ev': 0, 'solving': False,
                    'error': 'no blueprint data'}

        # Find hero's specific hand
        c0 = card_to_int(hero_cards[0])
        c1 = card_to_int(hero_cards[1])
        key = (min(c0, c1), max(c0, c1))

        hand_strat = all_probs.get(key)
        if not hand_strat:
            return {'actions': [], 'ev': 0, 'solving': False,
                    'error': 'hand not in blueprint'}

        actions = [{'action': name, 'frequency': freq}
                   for name, freq in sorted(hand_strat.items(),
                                            key=lambda x: -x[1])]

        # Get EV
        evs = self.blueprint.get_continuation_values(
            self.scenario_id, board, self.hero_player)
        ev = evs.get(key, 0)

        return {
            'actions': actions,
            'ev': ev,
            'solving': False,
            'source': 'blueprint',
        }

    def _format_result(self, result, hero_cards):
        """Format solver pool result for the HUD."""
        if 'error' in result:
            return {'actions': [], 'ev': 0, 'solving': False,
                    'error': result['error']}

        # Find hero's hand in the strategies
        player_idx = 0 if self.hero_player == "oop" else 1
        strategies = result.get('strategies', {}).get(player_idx, {})

        hand_str = hero_cards[0] + hero_cards[1]
        # Try both orderings
        hand_strat = strategies.get(hand_str)
        if not hand_strat:
            hand_strat = strategies.get(hero_cards[1] + hero_cards[0])
        if not hand_strat:
            # Try int_to_card ordering
            c0 = card_to_int(hero_cards[0])
            c1 = card_to_int(hero_cards[1])
            hand_strat = strategies.get(int_to_card(min(c0, c1)) + int_to_card(max(c0, c1)))

        if not hand_strat:
            return {'actions': [], 'ev': 0, 'solving': False,
                    'error': 'hand not in solver output'}

        actions = [{'action': name, 'frequency': freq}
                   for name, freq in sorted(hand_strat.items(),
                                            key=lambda x: -x[1])]

        return {
            'actions': actions,
            'ev': 0,  # TODO: get from solver
            'solving': False,
            'source': 'solver',
            'time_ms': result.get('time_ms', 0),
            'exploitability': result.get('exploit_pct', 0),
            'num_hands': result.get('num_hands', [0, 0]),
        }
