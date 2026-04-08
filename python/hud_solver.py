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
except ImportError:
    from python.solver import (card_to_int, int_to_card, parse_range_string,
                                SCALE, MAX_ACTIONS, StreetSolver)
    from python.range_narrowing import RangeNarrower
    from python.blueprint_io import Blueprint
    from python.blueprint_store import BlueprintStore
    from python.off_tree import pseudoharmonic_map, interpolate_narrowing
    from python.solver_pool import SolverPool

# Multiway heuristic adjustments were removed in v3 — see docs/REALTIME_TODO.md
# T1.2 and docs/V3_PLAN.md Phase 1.2. The previous implementation was a relic
# from when the GPU solver was heads-up only. Now that street_solve.cu supports
# 2-6 players natively (SS_MAX_PLAYERS=6) and the canonical realtime path uses
# N-player CFR via player_ranges=[...], the post-hoc heuristic adjustments
# were double-correcting on top of CFR with arbitrary tuning constants.

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
                 blueprint_v2_dir=None, solver_pool=None):
        """
        Args:
            blueprint_dir: path to old JSON flop_solutions/ (backward compat)
            blueprint_store_dir: path to new binary blueprints/ directory
            blueprint_v2_dir: path to v2 .bps blueprint files (6-player MCCFR)
            solver_pool: SolverPool for CPU fallback
        """
        # Old JSON blueprint (backward compat)
        self.blueprint = Blueprint(blueprint_dir) if blueprint_dir else None

        # New binary blueprint store
        self._bp_stores = {}  # scenario_id -> BlueprintStore
        self._bp_store_dir = blueprint_store_dir

        # V2 blueprint (.bps files, supports scenario-filtered v2 layout)
        self.blueprint_v2 = None
        if blueprint_v2_dir:
            try:
                from blueprint_v2 import BlueprintV2
                self.blueprint_v2 = BlueprintV2(blueprint_v2_dir, streets_to_load=[1])
            except ImportError:
                try:
                    from python.blueprint_v2 import BlueprintV2
                    self.blueprint_v2 = BlueprintV2(blueprint_v2_dir, streets_to_load=[1])
                except ImportError:
                    pass

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
                 pot_bb=0, stack_bb=100,
                 extra_villain_positions=None,
                 uniform_beliefs=False):
        """Initialize for a new hand.

        Args:
            hero_pos: hero's position (e.g. "BB")
            villain_pos: primary villain's position (e.g. "BTN")
            scenario_type: "srp" or "3bp"
            ranges_json_path: path to ranges.json
            num_players: total players in the hand (2-6)
            pot_bb: pot size in BB
            stack_bb: effective stack in BB
            extra_villain_positions: list of additional villain positions for
                multiway pots (e.g. ["CO", "MP"]). If provided, ranges are
                loaded for each and tracked separately for N-player GPU search.
        """
        self.hero_pos = hero_pos
        self.villain_pos = villain_pos
        self.num_players = num_players
        self.narrower = RangeNarrower()
        self._last_solve_avg_strategy = None
        self._current_street = "preflop"
        self._board = []
        self._pot_bb = pot_bb
        self._stack_bb = stack_bb
        self._extra_villains = {}  # pos -> {"narrower": RangeNarrower, "player_idx": int}

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

        # Set scenario on v2 blueprint so it loads the correct .bps files
        if self.blueprint_v2:
            self.blueprint_v2.current_scenario = self.scenario_id

        # Load ranges
        ranges_data = None
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

        if uniform_beliefs:
            # Pluribus-style: start with all 1326 hands at uniform weight.
            # Narrow from the first action using blueprint P(action|hand).
            self.narrower.set_uniform_range("hero")
            self.narrower.set_uniform_range("villain")
        else:
            # Default: start from known preflop ranges (better for HUD use
            # case where we know the preflop ranges from population data).
            self.narrower.set_initial_range("hero", hero_hands)
            self.narrower.set_initial_range("villain", villain_hands)

        # Load extra villain ranges for multiway N-player GPU search
        if extra_villain_positions and ranges_data:
            for ev_pos in extra_villain_positions:
                ev_range_str = self._get_range_str(
                    ranges_data, hero_pos, ev_pos, scenario_type, is_hero=False)
                if ev_range_str:
                    ev_narrower = RangeNarrower()
                    ev_hands = parse_range_string(ev_range_str)
                    ev_narrower.set_initial_range("villain", ev_hands)
                    # Determine position index for acting order
                    ev_idx = post_order.index(ev_pos) if ev_pos in post_order else 99
                    self._extra_villains[ev_pos] = {
                        "narrower": ev_narrower,
                        "player_idx": ev_idx,
                    }

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

        # 3. V2 blueprint (6-player MCCFR)
        if self.blueprint_v2 and self._board and len(self._board) >= 3:
            board_ints = [card_to_int(c) if isinstance(c, str) else c
                          for c in self._board[:3]]
            canonical = self.blueprint_v2.get_canonical_board(board_ints)
            if canonical:
                all_strats = self.blueprint_v2.get_all_bucket_strategies(
                    canonical, [], 0, street=1)
                if all_strats is not None:
                    # Map action name to action index
                    action_idx = self._action_name_to_idx(action)
                    if action_idx is not None and action_idx < all_strats.shape[1]:
                        # Map each hand in the player's range to its bucket,
                        # then return P(action|bucket) keyed by (card0, card1).
                        player_key = "hero" if player == "hero" else "villain"
                        hands = self.narrower.get_weighted_hands(player_key)
                        if hands:
                            num_buckets = all_strats.shape[0]
                            probs = {}
                            for c0, c1, _w in hands:
                                bucket = self._hand_to_bucket(
                                    c0, c1, board_ints, num_buckets)
                                if bucket < all_strats.shape[0]:
                                    probs[(c0, c1)] = float(
                                        all_strats[bucket, action_idx])
                            if any(v > 0 for v in probs.values()):
                                return probs

        # 4. Old blueprint fallback
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

    @staticmethod
    def _action_name_to_idx(action):
        """Map action name string to index in blueprint action array.

        Matches the C solver action ordering:
          ACT_FOLD=0, ACT_CHECK=1, ACT_CALL=2, ACT_BET=3+
        """
        action_lower = action.lower().strip()
        if "fold" in action_lower:
            return 0
        elif "check" in action_lower:
            return 1
        elif "call" in action_lower:
            return 2
        elif "all" in action_lower or "shove" in action_lower:
            # All-in is the last bet size (highest index)
            return 5
        elif "50" in action_lower or "half" in action_lower:
            return 3  # first bet size
        elif "pot" in action_lower or "100" in action_lower:
            return 4  # second bet size
        elif "bet" in action_lower or "raise" in action_lower:
            return 3  # generic bet -> first bet action
        return None

    @staticmethod
    def _hand_to_bucket(c0, c1, flop_ints, num_buckets, n_samples=100):
        """Compute the EHS-based bucket for a single hand.

        Uses the same percentile-based bucketing as gpu_mccfr.compute_equity_buckets:
        compute EHS via Monte Carlo, then map to a bucket based on the EHS value.

        Since we don't have all hands sorted together for percentile ranking,
        we approximate by directly mapping EHS (which is in [0,1]) to a bucket
        index: bucket = floor(ehs * num_buckets), clamped to [0, num_buckets-1].
        """
        import numpy as np
        rng = np.random.RandomState(c0 * 52 + c1)  # deterministic per hand
        blocked = set(flop_ints) | {c0, c1}
        available = [c for c in range(52) if c not in blocked]

        if len(available) < 4:
            return 0

        try:
            from gpu_mccfr import _eval7_py
        except ImportError:
            from python.gpu_mccfr import _eval7_py

        available = np.array(available)
        wins = 0
        ties = 0
        total = 0
        for _ in range(n_samples):
            idx = rng.choice(len(available), 4, replace=False)
            oc0, oc1 = int(available[idx[0]]), int(available[idx[1]])
            bc0, bc1 = int(available[idx[2]]), int(available[idx[3]])

            board = list(flop_ints) + [bc0, bc1]
            hero_str = _eval7_py(board + [c0, c1])
            opp_str = _eval7_py(board + [oc0, oc1])

            if hero_str > opp_str:
                wins += 1
            elif hero_str == opp_str:
                ties += 1
            total += 1

        ehs = (wins + 0.5 * ties) / max(total, 1)
        # Map EHS [0,1] directly to bucket index
        bucket = int(ehs * num_buckets)
        return min(bucket, num_buckets - 1)

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

    # Default CFR iteration count for realtime re-solves.
    # Pluribus uses 1-33 seconds per subgame which is thousands of iterations.
    # We previously used 200 to fit a sub-100ms HUD latency budget. With the
    # pivot to a trainer use case the latency budget is gone, so we bump to
    # 2000 — much closer to Pluribus convergence and only ~500ms per re-solve
    # at typical subgame sizes. Override per-call via cfr_iterations= if you
    # need a different speed/quality trade-off.
    DEFAULT_CFR_ITERATIONS = 2000

    def get_strategy(self, board, hero_cards, street="river",
                     pot_bb=None, stack_bb=None, cfr_iterations=None):
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
            cfr_iterations: number of LCFR iterations for this re-solve.
                Defaults to DEFAULT_CFR_ITERATIONS (2000). Pluribus uses
                thousands; 200 was the old HUD-latency-budget value.

        Returns:
            dict with actions, ev, solving status
        """
        if cfr_iterations is None:
            cfr_iterations = self.DEFAULT_CFR_ITERATIONS
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
        # Try binary blueprint store first.
        # NOTE: blueprint lookups return strategies trained by the N-player
        # MCCFR solver, so they are correct for multiway pots without any
        # post-hoc adjustment. The previous code applied heuristic multiway
        # corrections here — those were removed in v3 (see V3_PLAN.md 1.2).
        result = self._binary_blueprint_lookup(board, hero_cards)
        if result and result.get('actions'):
            return result

        # Try old JSON blueprint
        result = self._blueprint_lookup(board, hero_cards)
        if result and result.get('actions'):
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
          1. Build single-street betting tree (2-6 players)
          2. Compute leaf values from blueprint continuation strategies
          3. Run Linear CFR on GPU (200 iterations)
          4. Return final-iteration strategy for play
          5. Store weighted-average strategy for future narrowing

        When extra villains are tracked (multiway), builds an N-player
        tree with all active players' ranges. The GPU solver handles
        2-6 players natively (SS_MAX_PLAYERS=6).
        """
        pot = self._pot_bb if self._pot_bb > 0 else 10.0
        stack = self._stack_bb if self._stack_bb > 0 else 90.0

        # Build player ranges in position order for N-player solve
        # post_order: SB(0), BB(1), UTG(2), MP(3), CO(4), BTN(5)
        post_order = ["SB", "BB", "UTG", "MP", "CO", "BTN"]

        if self._extra_villains:
            # N-player mode: gather all active players' ranges in position order
            all_players = {}  # pos -> (hands, is_hero)

            all_players[self.hero_pos] = (hero_hands, True)
            all_players[self.villain_pos] = (villain_hands, False)

            board_ints = [card_to_int(c) if isinstance(c, str) else c for c in board]
            blocked = set(board_ints)

            for ev_pos, ev_info in self._extra_villains.items():
                ev_hands = ev_info["narrower"].get_weighted_hands("villain")
                if ev_hands:
                    ev_hands = [(c0, c1, w) for c0, c1, w in ev_hands
                                if c0 not in blocked and c1 not in blocked]
                    all_players[ev_pos] = (ev_hands, False)

            # Sort by position order
            sorted_players = sorted(all_players.keys(),
                                    key=lambda p: post_order.index(p)
                                    if p in post_order else 99)

            player_ranges = []
            acting_order = []
            hero_player_idx_in_solve = 0
            for i, pos in enumerate(sorted_players):
                hands, is_hero = all_players[pos]
                player_ranges.append(hands)
                acting_order.append(i)
                if is_hero:
                    hero_player_idx_in_solve = i
        else:
            # Standard 2-player mode
            if self.hero_player == "oop":
                player_ranges = [hero_hands, villain_hands]
                hero_player_idx_in_solve = 0
            else:
                player_ranges = [villain_hands, hero_hands]
                hero_player_idx_in_solve = 1
            acting_order = None  # default [0, 1]

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
                elif self.blueprint_v2 is not None:
                    # Use v2 .bps blueprint for flop leaf values.
                    # Flop leaves are at the turn boundary. Pluribus computes
                    # continuation values by averaging equity over all 49 turn
                    # cards, weighted by the 4 biased continuation strategies.
                    #
                    # Without full per-action turn EVs from the v2 blueprint,
                    # we compute per-turn-card equity directly. This is the
                    # same method used for turn leaves (which are at the river
                    # boundary). The 4 continuation strategy pairs all collapse
                    # to the same equity value since we lack per-action EVs to
                    # differentiate them. This is conservative — the GPU search
                    # refines leaf values via CFR iterations regardless.
                    board_strs = list(board[:3])

                    def _flop_leaf_fn_v2(tree_data, player_hands,
                                         max_h, pot_chips):
                        from leaf_values import (compute_flop_leaf_equity,
                                                 extract_leaf_info_from_tree)
                        leaf_infos = extract_leaf_info_from_tree(tree_data)
                        if not leaf_infos:
                            return None
                        flop_ints = [card_to_int(c) for c in board_strs]
                        # player_hands is list of hand lists (N-player) or
                        # the oop_hands list (2-player callback from StreetSolverGPU)
                        if isinstance(player_hands, list) and \
                           len(player_hands) > 0 and \
                           isinstance(player_hands[0], list):
                            oop_h = player_hands[0]
                            ip_h = player_hands[1] if len(player_hands) > 1 else []
                        else:
                            oop_h = player_hands
                            ip_h = []
                        return compute_flop_leaf_equity(
                            flop_board=flop_ints,
                            oop_hands=oop_h,
                            ip_hands=ip_h,
                            leaf_infos=leaf_infos,
                            max_hands=max_h,
                            starting_pot=pot_chips,
                        )
                    leaf_value_fn = _flop_leaf_fn_v2

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
                # Use N-player interface when we have multiway ranges,
                # otherwise backward-compat 2-player interface
                if self._extra_villains and len(player_ranges) > 2:
                    solver = StreetSolverGPU(
                        board=board,
                        player_ranges=player_ranges,
                        acting_order=acting_order,
                        pot_bb=pot,
                        stack_bb=stack,
                        bet_sizes=self.DEFAULT_BET_SIZES,
                        use_cont_strats=use_cont_strats,
                        leaf_value_fn=leaf_value_fn,
                    )
                else:
                    oop_h = player_ranges[0]
                    ip_h = player_ranges[1] if len(player_ranges) > 1 else []
                    solver = StreetSolverGPU(
                        board=board,
                        oop_range=oop_h,
                        ip_range=ip_h,
                        pot_bb=pot,
                        stack_bb=stack,
                        bet_sizes=self.DEFAULT_BET_SIZES,
                        use_cont_strats=use_cont_strats,
                        leaf_value_fn=leaf_value_fn,
                    )
                solver.solve(iterations=cfr_iterations)

                hand_str = hero_cards[0] + hero_cards[1]
                # Pluribus A3: freeze hero's strategy at previously-passed nodes.
                # Pass the ordered list of action labels hero took this street.
                frozen = None
                if self._hero_actions_this_street:
                    frozen = [action for action, _ in self._hero_actions_this_street]
                try:
                    strat = solver.get_strategy(hero_player_idx_in_solve,
                                                hand_str=hand_str,
                                                frozen_actions=frozen)
                except ValueError:
                    elapsed = (time.time() - t0) * 1000
                    return {
                        'actions': [], 'ev': 0, 'solving': False,
                        'error': 'hero hand {} not in trimmed range'.format(hand_str),
                        'time_ms': elapsed, 'street': street,
                        'source': 'street_solve_gpu',
                    }

                # Store weighted average for future range narrowing
                self._last_solve_avg_strategy = solver.get_avg_strategy(
                    hero_player_idx_in_solve)

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
                    'num_players_in_solve': len(player_ranges),
                }
            else:
                # CPU fallback via solver_v2 (2-player only)
                oop_h = player_ranges[0]
                ip_h = player_ranges[1] if len(player_ranges) > 1 else []
                solver = StreetSolver(
                    board=board,
                    oop_range=oop_h,
                    ip_range=ip_h,
                    pot_bb=pot,
                    stack_bb=stack,
                    bet_sizes=self.DEFAULT_BET_SIZES,
                )
                # CPU fallback uses ~2.5x iterations vs GPU since CPU CFR
                # converges slightly slower per iteration. The ratio matches
                # what was here before (200 GPU vs 500 CPU).
                solver.solve(iterations=int(cfr_iterations * 2.5))

                hand_str = hero_cards[0] + hero_cards[1]
                strat = solver.get_strategy(hero_player_idx_in_solve, hand_str)
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

            # If we're in a multiway pot but the caller didn't pass extra
            # villain ranges (so the GPU/CPU solve ran heads-up), warn that
            # the result is heads-up-only and may not reflect multiway dynamics.
            # Previously this path applied heuristic multiway adjustments;
            # those were removed in v3 (see V3_PLAN.md 1.2). Pass the full
            # extra_villain_positions list to new_hand() to get a true
            # N-player solve.
            if self.num_players > 2 and not self._extra_villains:
                result['multiway_warning'] = (
                    'Solved heads-up but num_players > 2; pass '
                    'extra_villain_positions to new_hand() for N-player CFR.'
                )

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

    # _apply_multiway() was deleted in v3 — see V3_PLAN.md Phase 1.2.
    # The N-player GPU solver in src/cuda/street_solve.cu handles 2-6 players
    # natively, and the v2 blueprint was trained with 6-player MCCFR. The
    # post-hoc heuristic adjustments here were a relic from when the solver
    # was heads-up only and were double-correcting on top of CFR with
    # arbitrary tuning constants (e.g. value_bet *= max(0.3, 1.0 - 0.15 * extra)).
