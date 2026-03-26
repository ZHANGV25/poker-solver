"""Bayesian range narrowing for poker.

Tracks both hero and villain ranges across streets using
precomputed blueprint P(action|hand) for Bayesian updates.

IMPORTANT (Pluribus alignment):
  The P(action|hand) used for narrowing should come from the WEIGHTED AVERAGE
  strategy, NOT the final iteration strategy. When using solver_v2's output:
    - For play decisions: use sv2_get_strategy() (final iteration)
    - For range narrowing: use sv2_get_average_strategy() (iteration-weighted avg)

  If using Linear CFR, the weighted average naturally weights later iterations
  more heavily (iteration t's strategy has weight t in the sum).

Usage:
    narrower = RangeNarrower()

    # Preflop: start with known ranges
    narrower.set_initial_range("villain", parse_range("CO_rfi_range"))
    narrower.set_initial_range("hero", parse_range("BB_defend_range"))

    # Flop: villain bets 75%, use blueprint to narrow
    narrower.update("villain", action="bet75",
                    blueprint_probs=blueprint.get_action_probs(board, "villain"))
    narrower.update("hero", action="call",
                    blueprint_probs=blueprint.get_action_probs(board, "hero"))

    # Get narrowed ranges for solver
    villain_range = narrower.get_weighted_hands("villain")
    hero_range = narrower.get_weighted_hands("hero")
"""

from typing import Dict, List, Optional, Tuple


class RangeNarrower:
    """Bayesian range tracker for both players across streets.

    Maintains a weight for each hand combo (0.0 to 1.0).
    Weights are updated via Bayes' rule when an action is observed:
        P(hand | action) ∝ P(action | hand) × P(hand)
    """

    # Minimum weight floor — never zero out a hand completely
    # (opponent could always deviate from blueprint)
    WEIGHT_FLOOR = 0.005

    def __init__(self):
        self._ranges = {}  # player -> list of (card0, card1, weight)
        self._action_log = {}  # player -> list of (street, action)

    def set_initial_range(self, player, hands):
        """Set the initial range for a player.

        Args:
            player: "hero" or "villain"
            hands: list of (card0, card1, weight) tuples
        """
        self._ranges[player] = list(hands)
        self._action_log[player] = []

    def set_uniform_range(self, player):
        """Set all 1326 possible hands with uniform weight (Pluribus-style).

        Pluribus starts with P(hand) = 1/1326 for all hands and narrows via
        Bayesian updates from the first action onwards. This matches the exact
        Pluribus belief initialization.
        """
        hands = []
        for c0 in range(52):
            for c1 in range(c0 + 1, 52):
                hands.append((c0, c1, 1.0))
        self._ranges[player] = hands
        self._action_log[player] = []

    def update(self, player, action, blueprint_probs):
        """Narrow a player's range based on an observed action.

        Uses Bayes' rule: new_weight[h] = old_weight[h] * P(action | h)
        Then normalizes so max weight = 1.0 (preserves relative proportions).

        Args:
            player: "hero" or "villain"
            action: action string (e.g., "check", "bet33", "bet75", "call", "raise", "fold")
            blueprint_probs: dict mapping (card0, card1) -> P(action | hand)
                            from the precomputed blueprint.
                            If a hand is missing, assumes uniform probability.
        """
        if player not in self._ranges:
            return

        hands = self._ranges[player]
        n_actions_approx = max(len(set(blueprint_probs.values())), 2)

        updated = []
        for c0, c1, old_weight in hands:
            key = (c0, c1)
            # Get P(action | hand) from blueprint
            p_action = blueprint_probs.get(key)
            if p_action is None:
                # Hand not in blueprint — assume uniform
                p_action = 1.0 / n_actions_approx

            # Bayesian update
            new_weight = old_weight * p_action

            # Apply floor
            if new_weight < self.WEIGHT_FLOOR and old_weight > 0:
                new_weight = self.WEIGHT_FLOOR

            updated.append((c0, c1, new_weight))

        # Normalize: scale so max weight = 1.0
        max_w = max(w for _, _, w in updated) if updated else 1.0
        if max_w > 0:
            self._ranges[player] = [
                (c0, c1, w / max_w) for c0, c1, w in updated
            ]
        else:
            self._ranges[player] = updated

        self._action_log.setdefault(player, []).append(action)

    def remove_folded_hands(self, player, blueprint_probs):
        """Remove hands that would have folded (P(fold|hand) > threshold).

        More aggressive than update() — actually removes hands from the range
        rather than just reducing their weight.

        Args:
            player: "hero" or "villain"
            blueprint_probs: dict mapping (card0, card1) -> P(fold | hand)
        """
        if player not in self._ranges:
            return

        FOLD_THRESHOLD = 0.95  # Only remove hands that fold >95% of the time
        hands = self._ranges[player]
        kept = []
        for c0, c1, w in hands:
            p_fold = blueprint_probs.get((c0, c1), 0)
            if p_fold < FOLD_THRESHOLD:
                kept.append((c0, c1, w))
            # else: hand folds almost always, remove from range

        self._ranges[player] = kept

    def get_weighted_hands(self, player):
        """Get the current weighted range for a player.

        Returns:
            list of (card0, card1, weight) tuples, sorted by weight descending
        """
        hands = self._ranges.get(player, [])
        return sorted(hands, key=lambda x: -x[2])

    def get_hand_count(self, player):
        """Get number of hands in range (with weight > floor)."""
        hands = self._ranges.get(player, [])
        return sum(1 for _, _, w in hands if w > self.WEIGHT_FLOOR)

    def get_action_log(self, player):
        """Get the action history for a player."""
        return self._action_log.get(player, [])

    def copy(self):
        """Create a deep copy of this narrower."""
        new = RangeNarrower()
        for player, hands in self._ranges.items():
            new._ranges[player] = list(hands)
        for player, log in self._action_log.items():
            new._action_log[player] = list(log)
        return new


def generate_all_hands(board_cards=None):
    """Generate all 1326 possible 2-card hands, excluding board cards.

    Args:
        board_cards: list of card ints to exclude (board blockers)

    Returns:
        list of (card0, card1, weight=1.0) tuples
    """
    blocked = set(board_cards or [])
    hands = []
    for c0 in range(52):
        if c0 in blocked:
            continue
        for c1 in range(c0 + 1, 52):
            if c1 in blocked:
                continue
            hands.append((c0, c1, 1.0))
    return hands


def make_blueprint_probs(hand_strategies, action):
    """Extract P(action|hand) from a blueprint strategy dict.

    Args:
        hand_strategies: dict mapping hand_str -> {action_name: frequency, ...}
                        (as stored in precomputed flop solutions)
        action: action to extract probability for (e.g., "Check", "Bet 75%")

    Returns:
        dict mapping (card0, card1) -> float
    """
    from solver import card_to_int
    probs = {}
    for hand_str, strat in hand_strategies.items():
        # Parse hand string like "AhKs"
        c0 = card_to_int(hand_str[:2])
        c1 = card_to_int(hand_str[2:])
        key = (min(c0, c1), max(c0, c1))

        # Find matching action
        p = 0.0
        for act_info in strat.get("actions", []):
            act_name = act_info.get("action", "")
            if action.lower() in act_name.lower():
                p = act_info.get("frequency", 0.0)
                break
        probs[key] = p

    return probs
