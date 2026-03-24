"""Leaf value computation for Pluribus-style depth-limited solving.

Implements the exact depth-limited approach from:
  Brown & Sandholm, "Depth-Limited Solving for Imperfect-Information Games",
  NeurIPS 2018, Section 4.

At each non-river depth-limit leaf, both players simultaneously choose among
4 continuation strategies for the remainder of the game:
    0. Unmodified blueprint strategy
    1. Fold-biased:  P(fold)  *= 5, renormalize
    2. Call-biased:  P(call)  *= 5, renormalize
    3. Raise-biased: P(raise) *= 5, renormalize

This is modeled as two sequential (but unobserved) decision nodes:
P0 picks among {0,1,2,3}, then P1 picks among {0,1,2,3}, giving 16 pairs.
At each terminal, the value is the EV of playing out the game under the
two chosen biased strategies.

For the flop solver:
  - Each leaf leads to the turn. The continuation EV is computed by
    averaging over all 49 possible turn cards, using precomputed
    per-action EVs from the blueprint.
  - EV under bias = sum_a biased_P(a|hand) * action_EV(a, hand)
  - This is the key insight: we DON'T re-solve — we combine the
    precomputed per-action EVs with the biased action probabilities.

For the turn solver:
  - Each leaf leads to the river. Since river is solved at runtime
    (no depth limit), we use river equity as the continuation value.
  - Both players' biased strategies at the turn leaf affect the
    expected pot share via the equity computation.

The leaf value the GPU kernel expects:
  leaf_values[leaf_idx * 2 * max_hands + player * max_hands + hand]
  = EV per unit of opponent reach for this hand at this leaf.

The kernel multiplies this by the actual opponent reach sum:
  cfv[hand] = leaf_value[hand] * sum(opponent_reach[non_conflicting])
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import namedtuple


# ── Continuation strategy biasing (Pluribus, 5x multiplier) ──────────────

def classify_actions(num_actions: int, first_action_is_fold: bool):
    """Classify action indices into fold/call/raise categories.

    In our tree, actions at the start of a new street (turn root) are:
      OOP first to act: [Check, Bet33%, Bet75%, All-in]
      IP facing a bet:  [Fold, Call, Raise33%, Raise75%, All-in]

    Since the precomputed blueprint stores strategies at the TURN ROOT
    (start of turn betting), there is NO fold — P0 acts first with
    check/bet options.

    Returns:
        dict mapping action index to category ('fold', 'call', 'raise')
    """
    categories = {}
    if first_action_is_fold:
        categories[0] = 'fold'
        categories[1] = 'call'
        for a in range(2, num_actions):
            categories[a] = 'raise'
    else:
        categories[0] = 'call'   # check
        for a in range(1, num_actions):
            categories[a] = 'raise'  # bets
    return categories


def bias_strategy(strategy: np.ndarray, bias_type: int,
                  action_categories: dict,
                  multiplier: float = 5.0) -> np.ndarray:
    """Apply Pluribus-style bias to a strategy array.

    Args:
        strategy: [num_hands, num_actions] — weighted avg P(action|hand)
        bias_type: 0=unmodified, 1=fold-biased, 2=call-biased, 3=raise-biased
        action_categories: dict {action_idx: 'fold'|'call'|'raise'}
        multiplier: Pluribus uses 5x (paper Section 4)

    Returns:
        [num_hands, num_actions] — biased strategy (renormalized)
    """
    if bias_type == 0:
        return strategy.copy()

    target = {1: 'fold', 2: 'call', 3: 'raise'}[bias_type]
    biased = strategy.copy()

    for a, cat in action_categories.items():
        if a < biased.shape[1] and cat == target:
            biased[:, a] *= multiplier

    # Renormalize rows
    row_sums = biased.sum(axis=1, keepdims=True)
    valid = row_sums > 1e-10
    biased = np.where(valid, biased / np.maximum(row_sums, 1e-10),
                      1.0 / biased.shape[1])
    return biased


# ── Leaf info extraction from tree ───────────────────────────────────────

LeafInfo = namedtuple('LeafInfo', ['leaf_idx', 'pot', 'bets'])


def extract_leaf_info_from_tree(tree_data) -> List[LeafInfo]:
    """Extract pot size and bet info for each leaf in the tree.

    Each leaf has a different pot size depending on the betting path
    that led to it (check-check vs bet-call vs bet-raise-call etc).
    This is critical — Pluribus computes per-leaf continuation values.

    Args:
        tree_data: SSTreeData ctypes struct (after ss_build_tree)

    Returns:
        list of LeafInfo, indexed by the leaf's original index
        (before continuation strategy expansion)
    """
    leaves = []

    # Walk the tree to find the original leaves (before cont expansion).
    # After expansion, the original leaf nodes become decision nodes.
    # The first_child continuation decision nodes are at depth
    # (original_leaf_depth + 1).
    #
    # However, we can identify original leaves by looking at decision
    # nodes that have exactly 4 children where the children are also
    # decision nodes with 4 children each (the P0→P1 pattern).
    #
    # Simpler: track leaves by their pot sizes from the tree structure.
    # The tree stores pot in each node.

    # For now, we reconstruct leaf info from the node list.
    # Leaves in the unexpanded tree are at specific positions.
    # After expansion, they become the P0 decision nodes.
    # The P0 decision nodes have player=0, 4 children, and those
    # children have player=1 and 4 children each.

    # We identify expanded-leaf roots as decision nodes where:
    #   - player = 0 (P0 chooses)
    #   - num_children = 4
    #   - all children are decision nodes with player=1, num_children=4
    #   - all grandchildren are SS_NODE_LEAF (type=4)

    n_nodes = tree_data.num_nodes
    for i in range(n_nodes):
        node = tree_data.nodes[i]
        if node.type != 0:  # not decision
            continue
        if node.player != 0:
            continue
        if node.num_children != 4:
            continue

        # Check children pattern
        is_cont_root = True
        for c in range(4):
            child_idx = tree_data.children[node.first_child + c]
            child = tree_data.nodes[child_idx]
            if child.type != 0 or child.player != 1 or child.num_children != 4:
                is_cont_root = False
                break

        if is_cont_root:
            # This was an original leaf, now expanded.
            # Get the leaf_idx from its first grandchild.
            first_child = tree_data.children[node.first_child]
            first_grandchild_idx = tree_data.children[
                tree_data.nodes[first_child].first_child]
            base_leaf_idx = tree_data.nodes[first_grandchild_idx].leaf_idx
            # base_leaf_idx is the first of 16 terminal leaves for this original leaf.
            # The original leaf index is base_leaf_idx // 16.
            orig_idx = base_leaf_idx // 16

            leaves.append(LeafInfo(
                leaf_idx=orig_idx,
                pot=node.pot,
                bets=(node.bets[0], node.bets[1]),
            ))

    leaves.sort(key=lambda x: x.leaf_idx)
    return leaves


# ── Flop leaf values (turn continuation) ─────────────────────────────────

def compute_flop_leaf_values(
    flop_board: List[int],
    oop_hands: List[Tuple[int, int, float]],
    ip_hands: List[Tuple[int, int, float]],
    blueprint_store,
    board_cards_str: List[str],
    leaf_infos: List[LeafInfo],
    max_hands: int,
    starting_pot: int,
) -> np.ndarray:
    """Compute leaf values for a flop solve using precomputed turn blueprints.

    For each original betting leaf (before continuation expansion):
      For each pair of continuation strategies (s0, s1):
        For each turn card:
          Compute EV = sum_a biased_P(a|h) * action_EV(a, h)
          using the blueprint's per-action EVs
        Average over turn cards

    The key Pluribus insight: we DON'T re-solve the turn. We compute
    the expected value under each biased strategy using the precomputed
    per-action EVs. This makes leaf computation O(hands * actions * turns)
    instead of requiring a full CFR solve.

    The leaf value is normalized per-opponent-hand so the GPU kernel
    can multiply by actual opponent reach.

    Args:
        flop_board: [card0, card1, card2] ints
        oop_hands: [(c0, c1, weight), ...] — solver's hand list
        ip_hands: same
        blueprint_store: BlueprintStore instance
        board_cards_str: ["Qs", "As", "2d"] for blueprint lookup
        leaf_infos: from extract_leaf_info_from_tree()
        max_hands: max(len(oop_hands), len(ip_hands))
        starting_pot: pot at the start of this street (chips, scaled)

    Returns:
        np.array[num_total_leaves, 2, max_hands] where
        num_total_leaves = len(leaf_infos) * 16
    """
    try:
        from solver import int_to_card
    except ImportError:
        from python.solver import int_to_card

    num_orig_leaves = len(leaf_infos)
    total_leaves = num_orig_leaves * 16
    leaf_values = np.zeros((total_leaves, 2, max_hands), dtype=np.float32)

    board_set = set(flop_board)
    turn_cards = [c for c in range(52) if c not in board_set]

    hands_by_player = [oop_hands, ip_hands]

    for li_idx, li in enumerate(leaf_infos):
        # Each original leaf has a specific pot size
        leaf_pot = li.pot

        # Accumulate per (s0, s1, player, hand) across turn cards
        cont_values = np.zeros((4, 4, 2, max_hands), dtype=np.float64)
        valid_turns = 0

        for tc in turn_cards:
            tc_str = int_to_card(tc)
            tc_set = board_set | {tc}

            # Load turn strategy + per-action EVs for both players
            player_data = []
            for p in range(2):
                strat = blueprint_store.get_turn_strategy(
                    board_cards_str, tc_str, p)
                action_evs = blueprint_store.get_turn_action_evs(
                    board_cards_str, tc_str, p)

                if strat is None or action_evs is None:
                    player_data.append(None)
                    continue

                na = strat.shape[1]

                # Determine action categories from the strategy shape.
                # At the turn root, OOP acts first: [Check, Bet1, Bet2, ..., All-in]
                # No fold at root because no bet to face.
                first_is_fold = False
                cats = classify_actions(na, first_is_fold)

                # Build 4 biased strategies
                biased = [bias_strategy(strat, bt, cats, 5.0) for bt in range(4)]

                # Compute EV under each bias:
                # EV_bias[s][h] = sum_a biased_P(a|h) * action_EV(a, h)
                ev_per_bias = np.zeros((4, strat.shape[0]), dtype=np.float64)
                for s in range(4):
                    for a in range(min(na, action_evs.shape[0])):
                        nh_tc = min(biased[s].shape[0], action_evs.shape[1])
                        ev_per_bias[s, :nh_tc] += (
                            biased[s][:nh_tc, a] * action_evs[a, :nh_tc])

                player_data.append({
                    'strat': strat,
                    'action_evs': action_evs,
                    'biased': biased,
                    'ev_per_bias': ev_per_bias,
                    'na': na,
                })

            if player_data[0] is None and player_data[1] is None:
                continue

            valid_turns += 1

            # Map turn-card-specific hand indices back to solver hand indices.
            # Blueprint stores hands with turn-card blocking already applied.
            # Our solver hands are indexed WITHOUT turn-card blocking.
            # We need to map: solver_hand_idx → blueprint_hand_idx (skipping blocked).
            for p in range(2):
                if player_data[p] is None:
                    continue

                hands = hands_by_player[p]
                ev_bias = player_data[p]['ev_per_bias']
                nh_bp = ev_bias.shape[1]  # blueprint hand count for this turn card

                # Build mapping: which solver hands are valid for this turn card
                # The blueprint's hand ordering matches solver order but skips
                # hands blocked by the turn card.
                bp_idx = 0
                for h in range(len(hands)):
                    hc0, hc1 = hands[h][0], hands[h][1]
                    if hc0 in tc_set or hc1 in tc_set:
                        continue  # hand blocked by turn card
                    if bp_idx >= nh_bp:
                        break

                    # For each continuation strategy pair:
                    # The leaf value for player p when P0 picks s0, P1 picks s1
                    # is determined by whichever strategy P_p chose.
                    for s_self in range(4):
                        ev = ev_bias[s_self, bp_idx]
                        # Scale EV by leaf pot relative to starting pot.
                        # The EV from blueprint is in absolute chips.
                        # The GPU kernel multiplies by opponent reach, so
                        # leaf_value should be the per-opponent-hand payoff.
                        for s_other in range(4):
                            if p == 0:
                                cont_values[s_self, s_other, p, h] += ev
                            else:
                                cont_values[s_other, s_self, p, h] += ev

                    bp_idx += 1

        if valid_turns > 0:
            cont_values /= valid_turns

        # Write to flat leaf_values array
        for s0 in range(4):
            for s1 in range(4):
                flat_idx = li.leaf_idx * 16 + s0 * 4 + s1
                if flat_idx < total_leaves:
                    for p in range(2):
                        leaf_values[flat_idx, p, :max_hands] = cont_values[
                            s0, s1, p, :max_hands].astype(np.float32)

    return leaf_values


# ── Turn leaf values (river continuation) ────────────────────────────────

def compute_turn_leaf_values(
    board_4: List[int],
    oop_hands: List[Tuple[int, int, float]],
    ip_hands: List[Tuple[int, int, float]],
    leaf_infos: List[LeafInfo],
    max_hands: int,
    starting_pot: int,
) -> np.ndarray:
    """Compute leaf values for a turn solve.

    Turn leaves lead to the river. Pluribus solves the river in real-time
    (no depth limit at river). For turn depth-limit leaves, we compute the
    continuation value as the average showdown equity over all possible
    river cards, weighted by pot size.

    For performance with large ranges (200 hands), we sample a subset of
    river cards rather than computing all 46. The sampling error is small
    since equity averages converge quickly.

    Each leaf has a different pot (check-check vs bet-call), so the
    showdown payoff differs per leaf.

    Returns:
        np.array[num_total_leaves, 2, max_hands]
    """
    num_orig_leaves = len(leaf_infos)
    total_leaves = num_orig_leaves * 16
    leaf_values = np.zeros((total_leaves, 2, max_hands), dtype=np.float32)

    board_set = set(board_4)
    river_cards = [c for c in range(52) if c not in board_set]
    hands_by_player = [oop_hands, ip_hands]

    nh0 = len(oop_hands)
    nh1 = len(ip_hands)

    # For large ranges, sample river cards to keep computation feasible.
    # 200 hands × 200 opps × 46 rivers × eval7 = ~85M evals.
    # Sample 12 rivers → ~2.4M evals, still ~3 seconds on CPU.
    # With 80 hands: 80² × 12 = ~77K evals, very fast.
    MAX_RIVER_SAMPLES = 12
    if len(river_cards) > MAX_RIVER_SAMPLES and (nh0 > 50 or nh1 > 50):
        rng = np.random.RandomState(42)  # deterministic
        sampled = sorted(rng.choice(river_cards, MAX_RIVER_SAMPLES, replace=False))
    else:
        sampled = river_cards

    # Precompute all hand strengths for all sampled river cards.
    # strengths[rc_idx][p][h] = hand strength or None if blocked
    all_strengths = []
    for rc in sampled:
        board_5 = list(board_4) + [rc]
        board_5_set = set(board_5)
        rc_strengths = [{}, {}]
        for p in range(2):
            for h in range(len(hands_by_player[p])):
                hc0, hc1 = hands_by_player[p][h][0], hands_by_player[p][h][1]
                if hc0 in board_5_set or hc1 in board_5_set:
                    continue
                rc_strengths[p][h] = _eval7_py(board_5 + [hc0, hc1])
        all_strengths.append(rc_strengths)

    for li_idx, li in enumerate(leaf_infos):
        leaf_pot = li.pot

        equity = np.zeros((2, max_hands), dtype=np.float64)
        counts = np.zeros((2, max_hands), dtype=np.float64)

        for rc_idx, rc in enumerate(sampled):
            strengths = all_strengths[rc_idx]

            for p in range(2):
                opp = 1 - p
                for h in range(len(hands_by_player[p])):
                    if h not in strengths[p]:
                        continue
                    hs = strengths[p][h]
                    hc0, hc1 = hands_by_player[p][h][0], hands_by_player[p][h][1]

                    wins = 0.0
                    losses = 0.0
                    ties = 0.0
                    for o in range(len(hands_by_player[opp])):
                        if o not in strengths[opp]:
                            continue
                        oc0, oc1 = hands_by_player[opp][o][0], hands_by_player[opp][o][1]
                        if hc0 == oc0 or hc0 == oc1 or hc1 == oc0 or hc1 == oc1:
                            continue
                        os_val = strengths[opp][o]
                        if hs > os_val:
                            wins += 1.0
                        elif hs < os_val:
                            losses += 1.0
                        else:
                            ties += 1.0

                    total = wins + losses + ties
                    if total > 0:
                        eq = (wins + 0.5 * ties) / total
                        payoff = (eq - 0.5) * leaf_pot
                        equity[p, h] += payoff
                        counts[p, h] += 1.0

        valid = counts > 0
        equity[valid] /= counts[valid]

        # Without turn continuation strategy data, all 16 (s0, s1) pairs
        # collapse to the same equity value. This is correct for the turn
        # because the river will be solved exactly — the 4 cont strats
        # are primarily meaningful at the flop (where turn blueprint exists).
        for s0 in range(4):
            for s1 in range(4):
                flat_idx = li.leaf_idx * 16 + s0 * 4 + s1
                if flat_idx < total_leaves:
                    leaf_values[flat_idx, 0, :max_hands] = equity[0, :max_hands]
                    leaf_values[flat_idx, 1, :max_hands] = equity[1, :max_hands]

    return leaf_values


# ── Python hand evaluator ────────────────────────────────────────────────

def _eval5_py(cards):
    """Evaluate 5-card hand. Returns comparable tuple."""
    ranks = sorted([c >> 2 for c in cards], reverse=True)
    suits = [c & 3 for c in cards]
    is_flush = len(set(suits)) == 1
    is_straight = False
    high = ranks[0]
    if ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5:
        is_straight = True
    if ranks == [12, 3, 2, 1, 0]:
        is_straight = True
        high = 3

    from collections import Counter
    cnt = Counter(ranks)
    groups = sorted(cnt.items(), key=lambda x: (-x[1], -x[0]))

    if is_straight and is_flush:
        return (9, high)
    if groups[0][1] == 4:
        return (8, groups[0][0], groups[1][0])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (7, groups[0][0], groups[1][0])
    if is_flush:
        return (6,) + tuple(ranks)
    if is_straight:
        return (5, high)
    if groups[0][1] == 3:
        kickers = sorted([r for r, c in groups if c == 1], reverse=True)
        return (4, groups[0][0]) + tuple(kickers)
    if groups[0][1] == 2 and groups[1][1] == 2:
        pairs = sorted([r for r, c in groups if c == 2], reverse=True)
        kicker = [r for r, c in groups if c == 1][0]
        return (3, pairs[0], pairs[1], kicker)
    if groups[0][1] == 2:
        kickers = sorted([r for r, c in groups if c == 1], reverse=True)
        return (2, groups[0][0]) + tuple(kickers)
    return (1,) + tuple(ranks)


def _eval7_py(cards):
    """Evaluate 7-card hand by trying all 21 5-card combinations."""
    from itertools import combinations
    best = None
    for combo in combinations(range(7), 5):
        hand = [cards[i] for i in combo]
        val = _eval5_py(hand)
        if best is None or val > best:
            best = val
    return best
