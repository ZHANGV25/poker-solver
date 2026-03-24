"""HUD Solver — high-level interface for the ACR Poker HUD.

Wraps solver_v2, range narrowing, blueprint I/O, and multiway adjustments
into a single interface that the HUD can call per-decision.

Aligned with Pluribus:
  - Flop: blueprint lookup for first action, re-solve for subsequent
  - Turn: re-solve with narrowed ranges + 4 continuation strategies
  - River: re-solve to showdown (no depth limit)
  - Range narrowing: uses WEIGHTED AVERAGE strategy (not final iteration)
  - Multiway: heuristic adjustments when 3+ players
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from solver import (card_to_int, int_to_card, parse_range_string,
                        SCALE, MAX_ACTIONS, StreetSolver)
    from range_narrowing import RangeNarrower
    from blueprint_io import Blueprint
    from solver_pool import SolverPool
    from multiway_adjust import (adjust_multiway_strategy, classify_hand_type,
                                  pick_primary_villain)
except ImportError:
    from python.solver import (card_to_int, int_to_card, parse_range_string,
                                SCALE, MAX_ACTIONS, StreetSolver)
    from python.range_narrowing import RangeNarrower
    from python.blueprint_io import Blueprint
    from python.solver_pool import SolverPool
    from python.multiway_adjust import (adjust_multiway_strategy,
                                         classify_hand_type,
                                         pick_primary_villain)


class HUDSolver:
    """High-level solver for the poker HUD.

    Manages range tracking, blueprint lookups, and runtime solving
    for a single table. Create one instance per table.
    """

    def __init__(self, blueprint_dir=None, solver_pool=None):
        self.blueprint = Blueprint(blueprint_dir) if blueprint_dir else None
        self.pool = solver_pool or SolverPool(max_workers=2)

        self.narrower = RangeNarrower()
        self.scenario_id = None
        self.hero_pos = None
        self.villain_pos = None
        self.hero_player = None  # "oop" or "ip"
        self.num_players = 2  # updated per hand

        self._pending_solve = None
        self._last_result = None
        self._current_street = None
        self._board = []
        self._pot_bb = 0
        self._stack_bb = 0

    def new_hand(self, hero_pos, villain_pos, scenario_type="srp",
                 ranges_json_path=None, num_players=2,
                 pot_bb=0, stack_bb=100):
        """Initialize for a new hand."""
        self.hero_pos = hero_pos
        self.villain_pos = villain_pos
        self.num_players = num_players
        self.narrower = RangeNarrower()
        self._pending_solve = None
        self._last_result = None
        self._current_street = "preflop"
        self._board = []
        self._pot_bb = pot_bb
        self._stack_bb = stack_bb

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

        # Load ranges
        if ranges_json_path and os.path.exists(ranges_json_path):
            import json
            with open(ranges_json_path) as f:
                ranges_data = json.load(f)
            hero_range_str = self._get_range_str(ranges_data, hero_pos,
                                                  villain_pos, scenario_type,
                                                  is_hero=True)
            villain_range_str = self._get_range_str(ranges_data, hero_pos,
                                                     villain_pos, scenario_type,
                                                     is_hero=False)
        else:
            hero_range_str = "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,AKo,AQo,AJo,KQs,KJs,KTs,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s"
            villain_range_str = hero_range_str

        hero_hands = parse_range_string(hero_range_str)
        villain_hands = parse_range_string(villain_range_str)
        self.narrower.set_initial_range("hero", hero_hands)
        self.narrower.set_initial_range("villain", villain_hands)

    def _get_range_str(self, ranges_data, hero_pos, villain_pos,
                        scenario_type, is_hero):
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

        Uses the WEIGHTED AVERAGE strategy from the most recent solve
        (or blueprint) for Bayesian updating, per Pluribus.
        """
        if board:
            self._board = board

        if self.blueprint and self.scenario_id:
            bp_player = "oop" if self.hero_player == "ip" else "ip"
            probs = self.blueprint.get_action_probs(
                self.scenario_id, self._board[:3],
                bp_player, action)
            if probs:
                self.narrower.update("villain", action, probs)
                return

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

    def get_strategy(self, board, hero_cards, street="river",
                     pot_bb=None, stack_bb=None):
        """Get solver's recommended strategy for hero's hand.

        For flop: uses blueprint lookup (first action) or re-solve (facing action)
        For turn: re-solves with narrowed ranges + continuation strategies
        For river: re-solves to showdown

        Args:
            board: current board cards (list of strings)
            hero_cards: hero's hole cards (list of 2 strings)
            street: "flop", "turn", "river"
            pot_bb: pot size in BB (uses stored value if None)
            stack_bb: effective stack in BB

        Returns:
            dict with actions, ev, solving status
        """
        self._board = board
        self._current_street = street
        if pot_bb is not None:
            self._pot_bb = pot_bb
        if stack_bb is not None:
            self._stack_bb = stack_bb

        hero_hands = self.narrower.get_weighted_hands("hero")
        villain_hands = self.narrower.get_weighted_hands("villain")

        if not hero_hands or not villain_hands:
            return {'actions': [], 'ev': 0, 'solving': False, 'error': 'empty range'}

        board_ints = [card_to_int(c) for c in board]
        blocked = set(board_ints)
        hero_hands = [(c0, c1, w) for c0, c1, w in hero_hands
                      if c0 not in blocked and c1 not in blocked]
        villain_hands = [(c0, c1, w) for c0, c1, w in villain_hands
                         if c0 not in blocked and c1 not in blocked]

        # Flop with blueprint available: use blueprint for first action
        if street == "flop" and len(board) == 3 and self.blueprint:
            return self._blueprint_or_resolve(board, hero_cards,
                                               hero_hands, villain_hands)

        # Turn or river: runtime solve
        return self._runtime_solve(board, hero_cards,
                                    hero_hands, villain_hands, street)

    def _blueprint_or_resolve(self, board, hero_cards,
                               hero_hands, villain_hands):
        """Flop: use blueprint for first action, re-solve if facing action."""
        result = self._blueprint_lookup(board, hero_cards)
        if result and result.get('actions'):
            # Apply multiway adjustments if needed
            if self.num_players > 2:
                result = self._apply_multiway(result, hero_cards, board)
            return result

        # No blueprint: fall back to re-solve
        return self._runtime_solve(board, hero_cards,
                                    hero_hands, villain_hands, "flop")

    def _runtime_solve(self, board, hero_cards,
                       hero_hands, villain_hands, street):
        """Re-solve from current position with narrowed ranges."""
        if self.hero_player == "oop":
            oop_hands, ip_hands = hero_hands, villain_hands
        else:
            oop_hands, ip_hands = villain_hands, hero_hands

        pot = self._pot_bb if self._pot_bb > 0 else 10.0
        stack = self._stack_bb if self._stack_bb > 0 else 90.0

        t0 = time.time()
        try:
            solver = StreetSolver(
                board=board,
                oop_range=oop_hands,
                ip_range=ip_hands,
                pot_bb=pot,
                stack_bb=stack,
                bet_sizes=[0.33, 0.75],
            )
            solver.solve(iterations=500)

            # Get strategy for hero's specific hand
            hand_str = hero_cards[0] + hero_cards[1]
            strat = solver.get_strategy(self.hero_player, hand_str)
            elapsed = (time.time() - t0) * 1000

            actions = [{'action': name, 'frequency': freq}
                       for name, freq in sorted(strat.items(),
                                                key=lambda x: -x[1])]

            result = {
                'actions': actions,
                'ev': 0,
                'solving': False,
                'source': 'solver_v2',
                'time_ms': elapsed,
                'street': street,
            }

            if self.num_players > 2:
                result = self._apply_multiway(result, hero_cards, board)

            return result

        except Exception as e:
            return {'actions': [], 'ev': 0, 'solving': False,
                    'error': str(e), 'time_ms': (time.time() - t0) * 1000}

    def _blueprint_lookup(self, board, hero_cards):
        """Look up strategy from precomputed blueprint."""
        if not self.blueprint:
            return None

        all_probs = self.blueprint.get_all_action_probs(
            self.scenario_id, board, self.hero_player)

        if not all_probs:
            return None

        c0 = card_to_int(hero_cards[0])
        c1 = card_to_int(hero_cards[1])
        key = (min(c0, c1), max(c0, c1))

        hand_strat = all_probs.get(key)
        if not hand_strat:
            return None

        actions = [{'action': name, 'frequency': freq}
                   for name, freq in sorted(hand_strat.items(),
                                            key=lambda x: -x[1])]

        evs = self.blueprint.get_continuation_values(
            self.scenario_id, board, self.hero_player)
        ev = evs.get(key, 0)

        return {
            'actions': actions,
            'ev': ev,
            'solving': False,
            'source': 'blueprint',
        }

    def _apply_multiway(self, result, hero_cards, board):
        """Apply multiway heuristic adjustments to solver output."""
        if self.num_players <= 2 or not result.get('actions'):
            return result

        # Estimate hand equity for classification
        # Simple heuristic based on action frequencies
        bet_freq = sum(a['frequency'] for a in result['actions']
                       if 'bet' in a['action'].lower() or 'raise' in a['action'].lower())
        if bet_freq > 0.6:
            hand_type = 'value'
        elif bet_freq > 0.2:
            hand_type = 'marginal'
        else:
            hand_type = 'bluff'

        strategy = {a['action']: a['frequency'] for a in result['actions']}
        adjusted = adjust_multiway_strategy(
            strategy, hand_type, self.num_players, self._current_street or 'flop')

        result['actions'] = [{'action': name, 'frequency': freq}
                              for name, freq in sorted(adjusted.items(),
                                                        key=lambda x: -x[1])]
        result['multiway_adjusted'] = True
        result['hand_type'] = hand_type
        return result
