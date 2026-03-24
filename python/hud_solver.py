"""HUD Solver — Pluribus-style street-by-street solver for the ACR Poker HUD.

Implements the full Pluribus hybrid architecture:
  - PREFLOP: lookup ranges from ranges.json
  - FLOP: blueprint lookup (first action) or GPU re-solve with turn leaf values
  - TURN: GPU re-solve with river leaf values + 4 continuation strategies
  - RIVER: GPU re-solve to showdown (no depth limit)
  - Range narrowing: WEIGHTED AVERAGE strategy (not final iteration)
  - Off-tree bets: pseudoharmonic interpolation
  - Multiway: heuristic adjustments when 3+ players

Decision pipeline:
  1. Load preflop ranges for this scenario
  2. On each villain/hero action: narrow ranges using weighted avg P(action|hand)
  3. On hero's turn to act: solve current street only (~50-100ms GPU)
  4. Leaf values from precomputed blueprint (flop/turn) or showdown (river)
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
    from blueprint_store import BlueprintStore
    from off_tree import pseudoharmonic_map, interpolate_narrowing
    from solver_pool import SolverPool
    from multiway_adjust import (adjust_multiway_strategy, classify_hand_type,
                                  pick_primary_villain)
except ImportError:
    from python.solver import (card_to_int, int_to_card, parse_range_string,
                                SCALE, MAX_ACTIONS, StreetSolver)
    from python.range_narrowing import RangeNarrower
    from python.blueprint_io import Blueprint
    from python.blueprint_store import BlueprintStore
    from python.off_tree import pseudoharmonic_map, interpolate_narrowing
    from python.solver_pool import SolverPool
    from python.multiway_adjust import (adjust_multiway_strategy,
                                         classify_hand_type,
                                         pick_primary_villain)

# Try GPU solver, fall back to CPU
_GPU_AVAILABLE = False
try:
    from street_solver_gpu import StreetSolverGPU
    _GPU_AVAILABLE = True
except Exception:
    try:
        from python.street_solver_gpu import StreetSolverGPU
        _GPU_AVAILABLE = True
    except Exception:
        pass


class HUDSolver:
    """Pluribus-style poker solver for the HUD.

    Manages range tracking, blueprint lookups, and street-by-street GPU solving
    for a single table. Create one instance per table.
    """

    # Default bet sizes per street (Pluribus uses 3-14 sizes depending on
    # the situation; we use 3 sizes + all-in which covers most strategic
    # value while keeping the tree small for single-street solving)
    DEFAULT_BET_SIZES = [0.33, 0.75, 1.5]

    def __init__(self, blueprint_dir=None, blueprint_store_dir=None,
                 solver_pool=None):
        """
        Args:
            blueprint_dir: path to old JSON flop_solutions/ (backward compat)
            blueprint_store_dir: path to new binary blueprints/ directory
            solver_pool: SolverPool for CPU fallback
        """
        # Old JSON blueprint (backward compat)
        self.blueprint = Blueprint(blueprint_dir) if blueprint_dir else None

        # New binary blueprint store
        self._bp_stores = {}  # scenario_id -> BlueprintStore
        self._bp_store_dir = blueprint_store_dir

        self.pool = solver_pool or SolverPool(max_workers=2)

        self.narrower = RangeNarrower()
        self.scenario_id = None
        self.hero_pos = None
        self.villain_pos = None
        self.hero_player = None  # "oop" or "ip"
        self.hero_player_idx = 0  # 0=OOP, 1=IP
        self.num_players = 2

        self._last_solve_avg_strategy = None  # weighted avg from most recent solve
        self._current_street = None
        self._board = []
        self._pot_bb = 0
        self._stack_bb = 0

        # A3: Strategy freezing (Pluribus)
        # Track hero's past actions on current street for re-solve safety.
        # When re-solving, hero's strategy at passed nodes is frozen for
        # hero's actual hand only. Other hands at those info sets are free.
        self._hero_actions_this_street = []  # [(action_idx, street), ...]

    def _get_bp_store(self, scenario_id):
        """Get or load BlueprintStore for a scenario."""
        if scenario_id in self._bp_stores:
            return self._bp_stores[scenario_id]

        if not self._bp_store_dir:
            return None

        store_dir = os.path.join(self._bp_store_dir, scenario_id)
        if not os.path.isdir(store_dir):
            return None
        if not os.path.exists(os.path.join(store_dir, 'index.bin')):
            return None

        store = BlueprintStore(store_dir, mode='r')
        self._bp_stores[scenario_id] = store
        return store

    def new_hand(self, hero_pos, villain_pos, scenario_type="srp",
                 ranges_json_path=None, num_players=2,
                 pot_bb=0, stack_bb=100):
        """Initialize for a new hand."""
        self.hero_pos = hero_pos
        self.villain_pos = villain_pos
        self.num_players = num_players
        self.narrower = RangeNarrower()
        self._last_solve_avg_strategy = None
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
            self.hero_player_idx = 0
            oop_pos, ip_pos = hero_pos, villain_pos
        else:
            self.hero_player = "ip"
            self.hero_player_idx = 1
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
            hero_range_str = ("AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,"
                              "AKs,AQs,AJs,ATs,AKo,AQo,AJo,"
                              "KQs,KJs,KTs,QJs,QTs,JTs,"
                              "T9s,98s,87s,76s,65s,54s")
            villain_range_str = hero_range_str

        hero_hands = parse_range_string(hero_range_str)
        villain_hands = parse_range_string(villain_range_str)
        self.narrower.set_initial_range("hero", hero_hands)
        self.narrower.set_initial_range("villain", villain_hands)

    def _get_range_str(self, ranges_data, hero_pos, villain_pos,
                        scenario_type, is_hero):
        """Extract range string from ranges.json.

        In SRP: one player is the opener (RFI range), the other is the
        defender (vs_rfi call range). We need to figure out who opened.
        Convention: the vs_rfi key is always "defender_vs_opener",
        e.g., "BB_vs_CO" means BB defended against CO's open.
        """
        if scenario_type == "srp":
            # Try both directions to find which is opener vs defender
            key_hero_defends = "{}_vs_{}".format(hero_pos, villain_pos)
            key_villain_defends = "{}_vs_{}".format(villain_pos, hero_pos)

            vs_rfi = ranges_data.get("vs_rfi", {})

            if is_hero:
                if key_hero_defends in vs_rfi:
                    # Hero is the defender (e.g., BB_vs_CO: hero=BB defends)
                    return vs_rfi[key_hero_defends].get("call", "")
                else:
                    # Hero is the opener
                    return ranges_data.get("rfi", {}).get(hero_pos, "")
            else:
                if key_villain_defends in vs_rfi:
                    # Villain is the defender
                    return vs_rfi[key_villain_defends].get("call", "")
                else:
                    # Villain is the opener
                    return ranges_data.get("rfi", {}).get(villain_pos, "")

        elif scenario_type == "3bp":
            # 3bet pot: opener raised, defender 3-bet, opener called
            key_hero_3bets = "{}_vs_{}".format(hero_pos, villain_pos)
            key_villain_3bets = "{}_vs_{}".format(villain_pos, hero_pos)
            vs_rfi = ranges_data.get("vs_rfi", {})

            if is_hero:
                if key_hero_3bets in vs_rfi and vs_rfi[key_hero_3bets].get("3bet"):
                    # Hero 3-bet → hero has the 3bet range
                    return vs_rfi[key_hero_3bets].get("3bet", "")
                else:
                    # Hero was the opener who called the 3bet
                    return ranges_data.get("vs_3bet", {}).get(
                        hero_pos, {}).get("call", "")
            else:
                if key_villain_3bets in vs_rfi and vs_rfi[key_villain_3bets].get("3bet"):
                    return vs_rfi[key_villain_3bets].get("3bet", "")
                else:
                    return ranges_data.get("vs_3bet", {}).get(
                        villain_pos, {}).get("call", "")
        return ""

    # ── Range narrowing ──────────────────────────────────────────────────

    def on_villain_action(self, street, action, board=None,
                          actual_bet_frac=None):
        """Process a villain action for range narrowing.

        Per Pluribus: uses WEIGHTED AVERAGE strategy from the most recent
        solve for Bayesian updating. Falls back to blueprint only if no
        solve has been done for this street.

        Args:
            street: "flop", "turn", "river"
            action: action string (e.g., "Check", "Bet 33%", "Call")
            board: current board cards
            actual_bet_frac: if off-tree, the actual bet as pot fraction
        """
        if board:
            self._board = board

        probs = self._get_narrowing_probs("villain", action, actual_bet_frac)
        if probs:
            self.narrower.update("villain", action, probs)

    def on_hero_action(self, street, action, board=None):
        """Process hero's action for range tracking.

        Also records the action for strategy freezing (A3):
        when re-solving later on this street, hero's strategy at
        previously-passed nodes is frozen for hero's actual hand.
        """
        if board:
            self._board = board

        # Track for strategy freezing
        self._hero_actions_this_street.append((action, street))

        probs = self._get_narrowing_probs("hero", action)
        if probs:
            self.narrower.update("hero", action, probs)

    def _get_narrowing_probs(self, player, action, actual_bet_frac=None):
        """Get P(action|hand) for range narrowing.

        Priority:
        1. Weighted average from most recent GPU solve
        2. Off-tree pseudoharmonic interpolation (if actual_bet_frac given)
        3. Blueprint (JSON) lookup

        Returns dict {(card0, card1): probability} or None.
        """
        # 1. Most recent solve's weighted average
        if self._last_solve_avg_strategy:
            probs = {}
            for hand_key, strat in self._last_solve_avg_strategy.items():
                probs[hand_key] = strat.get(action, 0.0)
            if any(v > 0 for v in probs.values()):
                return probs

        # 2. Off-tree bet handling
        if actual_bet_frac is not None:
            return self._compute_off_tree_probs(player, actual_bet_frac)

        # 3. Blueprint fallback
        if self.blueprint and self.scenario_id and self._board:
            if player == "villain":
                bp_player = "oop" if self.hero_player == "ip" else "ip"
            else:
                bp_player = self.hero_player
            probs = self.blueprint.get_action_probs(
                self.scenario_id, self._board[:3], bp_player, action)
            if probs:
                return probs

        return None

    def _compute_off_tree_probs(self, player, actual_bet_frac):
        """Compute narrowing probs for off-tree bet via pseudoharmonic."""
        tree_fracs = self.DEFAULT_BET_SIZES

        if player == "villain":
            bp_player = "oop" if self.hero_player == "ip" else "ip"
        else:
            bp_player = self.hero_player

        action_probs = []
        for frac in tree_fracs:
            action = "Bet {}%".format(int(frac * 100))
            if self.blueprint and self.scenario_id and self._board:
                probs = self.blueprint.get_action_probs(
                    self.scenario_id, self._board[:3], bp_player, action)
                action_probs.append(probs or {})
            else:
                action_probs.append({})

        if all(len(ap) == 0 for ap in action_probs):
            return None

        return interpolate_narrowing(actual_bet_frac, tree_fracs,
                                     action_probs)

    # ── Strategy computation ─────────────────────────────────────────────

    def get_strategy(self, board, hero_cards, street="river",
                     pot_bb=None, stack_bb=None):
        """Get solver's recommended strategy for hero's hand.

        Implements Pluribus street-by-street solving:
          - Flop: blueprint lookup (first action) or GPU re-solve
          - Turn: GPU re-solve with continuation leaf values
          - River: GPU re-solve to showdown

        Args:
            board: current board cards (list of strings)
            hero_cards: hero's hole cards (list of 2 strings)
            street: "flop", "turn", "river"
            pot_bb: pot size in BB
            stack_bb: effective stack in BB

        Returns:
            dict with actions, ev, solving status
        """
        self._board = board
        # Reset hero action tracking when street changes
        if street != self._current_street:
            self._hero_actions_this_street = []
        self._current_street = street
        if pot_bb is not None:
            self._pot_bb = pot_bb
        if stack_bb is not None:
            self._stack_bb = stack_bb

        hero_hands = self.narrower.get_weighted_hands("hero")
        villain_hands = self.narrower.get_weighted_hands("villain")

        if not hero_hands or not villain_hands:
            return {'actions': [], 'ev': 0, 'solving': False,
                    'error': 'empty range'}

        board_ints = [card_to_int(c) for c in board]
        blocked = set(board_ints)
        hero_hands = [(c0, c1, w) for c0, c1, w in hero_hands
                      if c0 not in blocked and c1 not in blocked]
        villain_hands = [(c0, c1, w) for c0, c1, w in villain_hands
                         if c0 not in blocked and c1 not in blocked]

        # Flop with blueprint: use blueprint for first action
        if street == "flop" and len(board) == 3:
            has_bp = self.blueprint or self._get_bp_store(self.scenario_id)
            if has_bp:
                return self._blueprint_or_resolve(board, hero_cards,
                                                   hero_hands, villain_hands)

        # GPU solve for current street
        return self._gpu_solve(board, hero_cards,
                               hero_hands, villain_hands, street)

    def _blueprint_or_resolve(self, board, hero_cards,
                               hero_hands, villain_hands):
        """Flop: blueprint for first action, re-solve if facing action.

        Tries binary BlueprintStore first (faster, denser data),
        then falls back to JSON Blueprint, then GPU re-solve.
        """
        # Try binary blueprint store first
        result = self._binary_blueprint_lookup(board, hero_cards)
        if result and result.get('actions'):
            if self.num_players > 2:
                result = self._apply_multiway(result, hero_cards, board)
            return result

        # Try old JSON blueprint
        result = self._blueprint_lookup(board, hero_cards)
        if result and result.get('actions'):
            if self.num_players > 2:
                result = self._apply_multiway(result, hero_cards, board)
            return result

        # No blueprint data — re-solve
        return self._gpu_solve(board, hero_cards,
                               hero_hands, villain_hands, "flop")

    def _binary_blueprint_lookup(self, board, hero_cards):
        """Look up flop strategy from binary BlueprintStore.

        Uses suit isomorphism to map hero's actual hand to the canonical
        texture, then looks up the hand index in the stored hand list.
        """
        bp_store = self._get_bp_store(self.scenario_id)
        if bp_store is None:
            return None

        from blueprint_io import texture_key as compute_texture_key
        tex_key, suit_map = compute_texture_key(board)
        data = bp_store.load_texture(tex_key)
        if data is None:
            return None

        player_idx = self.hero_player_idx
        strat_array = data['flop_strategies'].get(player_idx)
        if strat_array is None:
            return None

        hand_index = data.get('hand_index', {}).get(player_idx)
        if hand_index is None:
            return None

        # Map hero's hand through suit isomorphism (actual → canonical)
        canonical_c0 = card_to_int(
            hero_cards[0][0] + suit_map.get(hero_cards[0][1], hero_cards[0][1]))
        canonical_c1 = card_to_int(
            hero_cards[1][0] + suit_map.get(hero_cards[1][1], hero_cards[1][1]))

        key = (min(canonical_c0, canonical_c1), max(canonical_c0, canonical_c1))
        h_idx = hand_index.get(key)
        if h_idx is None:
            return None

        # Extract strategy for this hand
        num_actions = strat_array.shape[1]
        strat = strat_array[h_idx]

        # Build action labels (from precomputed tree structure)
        # For blueprint lookup, we use generic labels based on action count
        action_labels = ["Check"]
        for bs in self.DEFAULT_BET_SIZES:
            action_labels.append("Bet {}%".format(int(bs * 100)))
        action_labels.append("All-in")
        # Trim to actual action count
        action_labels = action_labels[:num_actions]

        actions = []
        for a in range(num_actions):
            freq = float(strat[a])
            if freq > 0.001:
                label = action_labels[a] if a < len(action_labels) else f"Action {a}"
                actions.append({'action': label, 'frequency': freq})
        actions.sort(key=lambda x: -x['frequency'])

        # Get EV if available
        ev = 0.0
        flop_evs = data.get('flop_evs', {}).get(player_idx)
        if flop_evs is not None and h_idx < len(flop_evs):
            ev = float(flop_evs[h_idx])

        return {
            'actions': actions,
            'ev': ev,
            'solving': False,
            'source': 'blueprint_binary',
        }

    def _gpu_solve(self, board, hero_cards,
                   hero_hands, villain_hands, street):
        """Solve the current street using GPU single-street solver.

        Implements Pluribus real-time search (Algorithm 2 in supplementary):
          1. Build single-street betting tree
          2. Compute leaf values from blueprint continuation strategies
          3. Run Linear CFR on GPU (200 iterations)
          4. Return final-iteration strategy for play
          5. Store weighted-average strategy for future narrowing
        """
        if self.hero_player == "oop":
            oop_hands, ip_hands = hero_hands, villain_hands
        else:
            oop_hands, ip_hands = villain_hands, hero_hands

        pot = self._pot_bb if self._pot_bb > 0 else 10.0
        stack = self._stack_bb if self._stack_bb > 0 else 90.0

        t0 = time.time()

        try:
            use_cont_strats = (street != "river")
            leaf_value_fn = None

            if street == "flop" and len(board) >= 3:
                bp_store = self._get_bp_store(self.scenario_id)
                if bp_store is not None:
                    # Build a closure that computes leaf values using
                    # the actual tree structure (solves chicken-and-egg).
                    board_strs = list(board[:3])

                    def _flop_leaf_fn(tree_data, oop_h, ip_h, max_h, pot_chips):
                        from leaf_values import (compute_flop_leaf_values,
                                                 extract_leaf_info_from_tree)
                        leaf_infos = extract_leaf_info_from_tree(tree_data)
                        if not leaf_infos:
                            return None
                        return compute_flop_leaf_values(
                            flop_board=[card_to_int(c) for c in board_strs],
                            oop_hands=oop_h,
                            ip_hands=ip_h,
                            blueprint_store=bp_store,
                            board_cards_str=board_strs,
                            leaf_infos=leaf_infos,
                            max_hands=max_h,
                            starting_pot=pot_chips,
                        )
                    leaf_value_fn = _flop_leaf_fn

            elif street == "turn" and len(board) >= 4:
                board_ints = [card_to_int(c) for c in board[:4]]

                def _turn_leaf_fn(tree_data, oop_h, ip_h, max_h, pot_chips):
                    from leaf_values import (compute_turn_leaf_values,
                                             extract_leaf_info_from_tree)
                    leaf_infos = extract_leaf_info_from_tree(tree_data)
                    if not leaf_infos:
                        return None
                    return compute_turn_leaf_values(
                        board_4=board_ints,
                        oop_hands=oop_h,
                        ip_hands=ip_h,
                        leaf_infos=leaf_infos,
                        max_hands=max_h,
                        starting_pot=pot_chips,
                    )
                leaf_value_fn = _turn_leaf_fn

            # Choose solver
            if _GPU_AVAILABLE:
                solver = StreetSolverGPU(
                    board=board,
                    oop_range=oop_hands,
                    ip_range=ip_hands,
                    pot_bb=pot,
                    stack_bb=stack,
                    bet_sizes=self.DEFAULT_BET_SIZES,
                    use_cont_strats=use_cont_strats,
                    leaf_value_fn=leaf_value_fn,
                )
                solver.solve(iterations=200)

                hand_str = hero_cards[0] + hero_cards[1]
                try:
                    strat = solver.get_strategy(self.hero_player, hand_str=hand_str)
                except ValueError:
                    # Hero's hand was trimmed from the range (>200 hands).
                    # This shouldn't happen in practice — hero's actual hand
                    # should always be in the range with weight 1.0.
                    # Return empty result with explanation.
                    elapsed = (time.time() - t0) * 1000
                    return {
                        'actions': [], 'ev': 0, 'solving': False,
                        'error': 'hero hand {} not in trimmed range ({}→{} hands)'.format(
                            hand_str, len(oop_hands) + len(ip_hands),
                            len(solver.oop_hands) + len(solver.ip_hands)),
                        'time_ms': elapsed, 'street': street,
                        'source': 'street_solve_gpu',
                    }

                # Store weighted average for future range narrowing
                self._last_solve_avg_strategy = solver.get_avg_strategy(
                    self.hero_player)

                elapsed = (time.time() - t0) * 1000

                actions = [{'action': name, 'frequency': freq}
                           for name, freq in sorted(strat.items(),
                                                    key=lambda x: -x[1])]

                result = {
                    'actions': actions,
                    'ev': 0,
                    'solving': False,
                    'source': 'street_solve_gpu',
                    'time_ms': elapsed,
                    'street': street,
                }
            else:
                # CPU fallback via solver_v2
                solver = StreetSolver(
                    board=board,
                    oop_range=oop_hands,
                    ip_range=ip_hands,
                    pot_bb=pot,
                    stack_bb=stack,
                    bet_sizes=self.DEFAULT_BET_SIZES,
                )
                solver.solve(iterations=500)

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
                    'source': 'solver_v2_cpu',
                    'time_ms': elapsed,
                    'street': street,
                }

            if self.num_players > 2:
                result = self._apply_multiway(result, hero_cards, board)

            return result

        except Exception as e:
            return {'actions': [], 'ev': 0, 'solving': False,
                    'error': str(e),
                    'time_ms': (time.time() - t0) * 1000}

    def _blueprint_lookup(self, board, hero_cards):
        """Look up strategy from precomputed blueprint (JSON format)."""
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

        bet_freq = sum(a['frequency'] for a in result['actions']
                       if 'bet' in a['action'].lower()
                       or 'raise' in a['action'].lower())
        if bet_freq > 0.6:
            hand_type = 'value'
        elif bet_freq > 0.2:
            hand_type = 'marginal'
        else:
            hand_type = 'bluff'

        strategy = {a['action']: a['frequency'] for a in result['actions']}
        adjusted = adjust_multiway_strategy(
            strategy, hand_type, self.num_players,
            self._current_street or 'flop')

        result['actions'] = [{'action': name, 'frequency': freq}
                              for name, freq in sorted(adjusted.items(),
                                                        key=lambda x: -x[1])]
        result['multiway_adjusted'] = True
        result['hand_type'] = hand_type
        return result
