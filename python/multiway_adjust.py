"""Multiway heuristic adjustments for 3+ player pots.

Since the solver is heads-up only, when multiple players see the flop
we apply heuristic adjustments to approximate multiway GTO behavior.

Key principles:
- Bluff less: more opponents = more likely someone has a strong hand
- Value bet thinner: need stronger hands to bet for value
- Call tighter: pot odds are similar but implied odds decrease
- Fold more: marginal hands lose value in multiway pots

Usage:
    from multiway_adjust import adjust_multiway_strategy

    # After getting HU solver strategy
    adjusted = adjust_multiway_strategy(
        strategy={'Check': 0.4, 'Bet 75%': 0.6},
        hand_type='bluff',    # or 'value', 'marginal'
        num_players=3,
        street='flop',
    )
"""

from typing import Dict, Optional


def classify_hand_type(equity: float) -> str:
    """Classify hand into value/marginal/bluff based on equity.

    Args:
        equity: hand's equity vs villain range (0-1)

    Returns:
        'value', 'marginal', or 'bluff'
    """
    if equity >= 0.60:
        return 'value'
    elif equity >= 0.35:
        return 'marginal'
    else:
        return 'bluff'


def get_multiway_multipliers(num_players: int, street: str) -> dict:
    """Get action frequency multipliers for multiway pots.

    Args:
        num_players: total players in pot (3, 4, 5, or 6)
        street: 'flop', 'turn', or 'river'

    Returns:
        dict with multipliers for different hand types:
        {
            'value_bet': float,      # multiply value bet frequency
            'bluff_bet': float,      # multiply bluff bet frequency
            'marginal_bet': float,   # multiply marginal hand bet frequency
            'call_threshold': float, # multiply calling threshold
            'fold_boost': float,     # add to fold frequency for marginals
        }
    """
    # Base adjustments scale with number of extra opponents
    extra_opponents = num_players - 2  # 1 for 3-way, 2 for 4-way, etc.

    if street == 'flop':
        return {
            'value_bet': max(0.3, 1.0 - 0.15 * extra_opponents),
            'bluff_bet': max(0.1, 1.0 - 0.50 * extra_opponents),
            'marginal_bet': max(0.1, 1.0 - 0.40 * extra_opponents),
            'call_threshold': 1.0 + 0.15 * extra_opponents,
            'fold_boost': 0.15 * extra_opponents,
        }
    elif street == 'turn':
        return {
            'value_bet': max(0.4, 1.0 - 0.10 * extra_opponents),
            'bluff_bet': max(0.05, 1.0 - 0.55 * extra_opponents),
            'marginal_bet': max(0.1, 1.0 - 0.45 * extra_opponents),
            'call_threshold': 1.0 + 0.20 * extra_opponents,
            'fold_boost': 0.20 * extra_opponents,
        }
    else:  # river
        return {
            'value_bet': max(0.5, 1.0 - 0.08 * extra_opponents),
            'bluff_bet': max(0.05, 1.0 - 0.60 * extra_opponents),
            'marginal_bet': max(0.05, 1.0 - 0.50 * extra_opponents),
            'call_threshold': 1.0 + 0.25 * extra_opponents,
            'fold_boost': 0.25 * extra_opponents,
        }


def adjust_multiway_strategy(strategy: Dict[str, float],
                             hand_type: str,
                             num_players: int,
                             street: str = 'flop') -> Dict[str, float]:
    """Adjust a heads-up solver strategy for multiway play.

    Args:
        strategy: dict mapping action names to frequencies (must sum to ~1.0)
                  e.g. {'Check': 0.4, 'Bet 75%': 0.6}
        hand_type: 'value', 'marginal', or 'bluff'
        num_players: total players in pot (3+)
        street: 'flop', 'turn', or 'river'

    Returns:
        Adjusted strategy dict (frequencies re-normalized to sum to 1.0)
    """
    if num_players <= 2:
        return strategy

    mults = get_multiway_multipliers(num_players, street)

    # Classify actions
    adjusted = {}
    for action, freq in strategy.items():
        action_lower = action.lower()

        if 'fold' in action_lower:
            # Folding: increase for marginal/bluff hands
            if hand_type in ('marginal', 'bluff'):
                adjusted[action] = freq + mults['fold_boost']
            else:
                adjusted[action] = freq
        elif 'check' in action_lower or 'call' in action_lower:
            # Passive actions: reduce for bluffs, keep for value
            if hand_type == 'bluff':
                adjusted[action] = freq * mults['bluff_bet']
            elif hand_type == 'marginal':
                adjusted[action] = freq
            else:
                adjusted[action] = freq
        elif 'bet' in action_lower or 'raise' in action_lower:
            # Aggressive actions: apply hand-type-specific multiplier
            if hand_type == 'value':
                adjusted[action] = freq * mults['value_bet']
            elif hand_type == 'bluff':
                adjusted[action] = freq * mults['bluff_bet']
            else:  # marginal
                adjusted[action] = freq * mults['marginal_bet']
        else:
            adjusted[action] = freq

    # Re-normalize
    total = sum(adjusted.values())
    if total > 0:
        for action in adjusted:
            adjusted[action] /= total
    else:
        # Degenerate: default to check/fold
        first = list(strategy.keys())[0]
        adjusted = {action: 0.0 for action in strategy}
        adjusted[first] = 1.0

    return adjusted


def pick_primary_villain(players, hero_pos, board=None):
    """Pick the most relevant villain for heads-up approximation.

    In multiway pots, we solve against the "most dangerous" opponent:
    the one with the most range overlap with hero.

    Args:
        players: list of dicts with 'position', 'stack', 'range_width' (optional)
        hero_pos: hero's position string
        board: board cards (unused for now)

    Returns:
        position string of the primary villain
    """
    if not players:
        return None

    # Heuristic: prefer in-position players, then by stack size
    post_order = ["SB", "BB", "UTG", "MP", "CO", "BTN"]

    def villain_score(p):
        pos = p.get('position', '')
        stack = p.get('stack', 100)
        range_width = p.get('range_width', 0.5)

        # Prefer:
        # 1. Players who are in position relative to hero
        # 2. Players with wider ranges (more dangerous)
        # 3. Players with larger stacks (more implied odds)
        pos_idx = post_order.index(pos) if pos in post_order else 0
        hero_idx = post_order.index(hero_pos) if hero_pos in post_order else 0

        ip_bonus = 1.0 if pos_idx > hero_idx else 0.0
        return ip_bonus + range_width * 0.5 + min(stack, 200) / 400.0

    non_hero = [p for p in players if p.get('position') != hero_pos]
    if not non_hero:
        return None

    best = max(non_hero, key=villain_score)
    return best.get('position')
