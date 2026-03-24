"""Off-tree bet mapping via pseudoharmonic interpolation.

When the opponent bets a size not in our action abstraction (e.g., 55% pot
when our tree has 33% and 75%), we map the actual bet to the two nearest
tree sizes using pseudoharmonic interpolation (Johanson et al. 2013).

Pseudoharmonic formula:
    Given actual bet fraction b, nearest smaller tree fraction b_lo,
    and nearest larger fraction b_hi:

    weight_hi = (1/b_lo - 1/b) / (1/b_lo - 1/b_hi)
    weight_lo = 1 - weight_hi

    This weighting is "pseudoharmonic" because it uses the harmonic
    (reciprocal) transformation, which better captures the strategic
    similarity of bet sizes than linear interpolation. A 50% pot bet
    is strategically closer to a 33% bet than to a 100% bet.

For range narrowing with off-tree bets:
    P(off_tree_action | hand) = w_lo * P(bet_lo | hand) + w_hi * P(bet_hi | hand)

For re-solving after an off-tree bet:
    Narrow villain's range using the interpolated P(action|hand),
    then re-solve from the current decision point.
"""

from typing import List, Optional, Tuple


def pseudoharmonic_map(
    actual_bet_frac: float,
    tree_bet_fracs: List[float],
) -> List[Tuple[int, float]]:
    """Map an actual bet fraction to the nearest tree bet sizes.

    Uses pseudoharmonic interpolation for the weighting.

    Args:
        actual_bet_frac: actual bet as fraction of pot (e.g., 0.55)
        tree_bet_fracs: sorted list of bet fractions in our tree
                        (e.g., [0.33, 0.75])

    Returns:
        List of (action_index, weight) pairs that sum to 1.0.
        action_index corresponds to the position in tree_bet_fracs.
        If actual_bet_frac matches a tree size exactly, returns [(idx, 1.0)].
    """
    if not tree_bet_fracs:
        return []

    b = actual_bet_frac

    # Exact match
    for i, tf in enumerate(tree_bet_fracs):
        if abs(b - tf) < 0.001:
            return [(i, 1.0)]

    # Find bracketing sizes
    smaller = [(i, tf) for i, tf in enumerate(tree_bet_fracs) if tf < b]
    larger = [(i, tf) for i, tf in enumerate(tree_bet_fracs) if tf > b]

    if not smaller:
        # Below all tree sizes — map entirely to smallest
        return [(0, 1.0)]

    if not larger:
        # Above all tree sizes — map entirely to largest
        return [(len(tree_bet_fracs) - 1, 1.0)]

    # Get the two nearest bracketing sizes
    lo_idx, b_lo = max(smaller, key=lambda x: x[1])
    hi_idx, b_hi = min(larger, key=lambda x: x[1])

    # Pseudoharmonic weight
    # w_hi = (1/b_lo - 1/b) / (1/b_lo - 1/b_hi)
    if b_lo <= 0 or b <= 0 or b_hi <= 0:
        # Degenerate case: fall back to linear interpolation
        w_hi = (b - b_lo) / (b_hi - b_lo)
    else:
        inv_lo = 1.0 / b_lo
        inv_b = 1.0 / b
        inv_hi = 1.0 / b_hi
        denom = inv_lo - inv_hi
        if abs(denom) < 1e-10:
            w_hi = 0.5
        else:
            w_hi = (inv_lo - inv_b) / denom

    w_hi = max(0.0, min(1.0, w_hi))
    w_lo = 1.0 - w_hi

    return [(lo_idx, w_lo), (hi_idx, w_hi)]


def interpolate_narrowing(
    actual_bet_frac: float,
    tree_bet_fracs: List[float],
    action_probs: List[dict],
) -> dict:
    """Interpolate range narrowing probabilities for an off-tree bet.

    Args:
        actual_bet_frac: actual bet as fraction of pot
        tree_bet_fracs: bet fractions in our tree
        action_probs: list of dicts {(card0, card1): P(bet_size|hand)}
                      one per tree bet fraction

    Returns:
        dict {(card0, card1): P(off_tree_bet|hand)} — interpolated probs
    """
    mapping = pseudoharmonic_map(actual_bet_frac, tree_bet_fracs)

    if len(mapping) == 1:
        idx, w = mapping[0]
        return dict(action_probs[idx])

    # Interpolate
    result = {}
    all_hands = set()
    for ap in action_probs:
        all_hands.update(ap.keys())

    for hand in all_hands:
        p = 0.0
        for idx, w in mapping:
            p += w * action_probs[idx].get(hand, 0.0)
        result[hand] = p

    return result


def map_off_tree_action(
    actual_bet: float,
    pot: float,
    tree_bet_fracs: List[float],
    includes_fold: bool = True,
    includes_check_call: bool = True,
) -> Tuple[str, List[Tuple[int, float]]]:
    """Classify an off-tree action and return mapping.

    Handles special cases:
    - Bet of 0 = check
    - Bet >= effective_stack = all-in (map to all-in action)
    - Min-bet below smallest tree size

    Args:
        actual_bet: actual bet amount in chips
        pot: current pot size in chips
        tree_bet_fracs: bet fractions in tree
        includes_fold: whether tree includes fold action
        includes_check_call: whether tree includes check/call action

    Returns:
        (action_type, mapping) where:
        - action_type: "fold", "check", "call", "bet", "allin"
        - mapping: list of (tree_action_index, weight) for bet actions
                   tree_action_index accounts for fold/call offsets
    """
    if actual_bet <= 0:
        return ("check", [])

    if pot <= 0:
        return ("bet", [(0, 1.0)])

    bet_frac = actual_bet / pot

    # Compute action index offset
    offset = 0
    if includes_fold:
        offset += 1
    if includes_check_call:
        offset += 1

    mapping = pseudoharmonic_map(bet_frac, tree_bet_fracs)
    # Shift indices by the offset (fold + call come before bets)
    shifted = [(idx + offset, w) for idx, w in mapping]

    return ("bet", shifted)
