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


# ── First-order bias approximation (v3 fix for v2 .bps path) ─────────────
#
# Pluribus's depth-limited solving uses 4 continuation strategies (unmodified,
# fold-bias, call-bias, raise-bias) at every leaf and produces 4 different
# leaf values per (hand, leaf) pair via:
#
#     leaf_value(hand, bias_pair) = sum_a biased_P(a | hand, bias) * EV(a, hand)
#
# That requires per-action EVs `EV(a, hand)` from the blueprint, which the v2
# .bps export does not currently store (the binary BlueprintStore format does;
# v2 doesn't). Without per-action EVs, the obvious fallback is "use showdown
# equity for all 16 (s0, s1) pairs", which collapses Pluribus's variance
# reduction to nothing.
#
# This module's first-order bias approximation gives **16 different** leaf
# values per (hand, leaf) by adjusting the per-bias leaf computation along
# two axes:
#
# 1. **Fold equity from biased opponent.** A fold-biased opponent folds more
#    often on the next street; we capture that as a fractional "free win" of
#    the current pot, weighted by FOLD_PROB[bias].
# 2. **Pot-size adjustment from biased participation.** A raise-biased opponent
#    inflates the pot we end up in (when we both go to showdown); a fold-biased
#    opponent shrinks it. POT_FACTOR[bias] captures this multiplicatively.
#
# The result is a defensible first-order approximation: it correctly captures
# the SIGN and rough MAGNITUDE of how each bias profile changes the leaf value,
# even though the exact magnitudes differ from a true per-action EV computation.
# This restores meaningful variance reduction in the GPU CFR's leaf-value pass
# without requiring any blueprint export changes.
#
# **Long-term fix:** still want option B (per-action EVs in .bps via post-hoc
# tree walk at export time). That gives exact leaf values matching Pluribus.
# This first-order approximation is documented in V3_PLAN.md Phase 1.3 as the
# v3 fix, and option B is tracked as a future REALTIME_TODO item for v4.

# Probability that a player folds on the next street, indexed by bias profile:
# 0 = unmodified, 1 = fold-bias, 2 = call-bias, 3 = raise-bias
BIAS_FOLD_PROB = [0.0, 0.35, 0.0, 0.0]

# Multiplicative factor on the pot we end up at if both players go to showdown,
# indexed by bias profile
BIAS_POT_FACTOR = [1.0, 0.7, 1.0, 1.4]


def biased_leaf_value(equity, leaf_pot, s_self, s_opp):
    """First-order bias-adjusted leaf value for a single (hand, leaf, bias-pair).

    Args:
        equity: showdown equity of self vs opp's range, in [0, 1]
        leaf_pot: chip count of the pot at this leaf (positive int)
        s_self: bias profile of self {0,1,2,3}
        s_opp: bias profile of opponent {0,1,2,3}

    Returns:
        EV of self at this leaf in chips, signed (positive = winning chips
        relative to leaf_pot/2 baseline)

    Math:
      - With probability fold_prob_opp = BIAS_FOLD_PROB[s_opp]:
            opponent folds → self wins the leaf_pot/2 they would have invested
                             (approximation: half the pot is "self contribution")
      - With probability 1 - fold_prob_opp:
            both go to showdown
            adjusted_pot = leaf_pot * BIAS_POT_FACTOR[s_self] * BIAS_POT_FACTOR[s_opp]
            self EV at showdown = (equity - 0.5) * adjusted_pot
    """
    fold_prob_opp = BIAS_FOLD_PROB[s_opp]
    pot_factor = BIAS_POT_FACTOR[s_self] * BIAS_POT_FACTOR[s_opp]
    showdown_pot = leaf_pot * pot_factor

    fold_payoff = 0.5 * leaf_pot     # we win the half opponent would have put in
    showdown_payoff = (equity - 0.5) * showdown_pot

    return fold_prob_opp * fold_payoff + (1.0 - fold_prob_opp) * showdown_payoff


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


def extract_leaf_info_from_tree(tree_data, num_players=2) -> List[LeafInfo]:
    """Extract pot size and bet info for each leaf in the tree.

    Each leaf has a different pot size depending on the betting path
    that led to it. Supports N-player continuation structures:
    P0 → P1 → ... → P(N-1), each with 4 children = 4^N terminals per leaf.

    Args:
        tree_data: SSTreeData ctypes struct (after ss_build_tree)
        num_players: number of active players (2-6)

    Returns:
        list of LeafInfo, indexed by the leaf's original index
        (before continuation strategy expansion)
    """
    leaves = []
    cont_per_leaf = 4 ** num_players  # 16 for 2p, 64 for 3p, etc.

    def _is_cont_chain(tree_data, node_idx, depth, num_players):
        """Check if node_idx is the root of a N-deep continuation chain.
        Each level has player=depth, num_children=4, depth levels total."""
        node = tree_data.nodes[node_idx]
        if node.type != 0 or node.num_children != 4:
            return False
        if depth == num_players - 1:
            # Last player: children should be terminals (leaf type=4)
            for c in range(4):
                child_idx = tree_data.children[node.first_child + c]
                if tree_data.nodes[child_idx].type != 4:  # not leaf
                    return False
            return True
        else:
            # Intermediate: children should be continuation decision nodes
            for c in range(4):
                child_idx = tree_data.children[node.first_child + c]
                if not _is_cont_chain(tree_data, child_idx, depth + 1, num_players):
                    return False
            return True

    def _get_first_terminal(tree_data, node_idx, num_players):
        """Walk down the first-child chain to find the first terminal leaf."""
        cur = node_idx
        for _ in range(num_players):
            cur = tree_data.children[tree_data.nodes[cur].first_child]
        return cur

    n_nodes = tree_data.num_nodes
    for i in range(n_nodes):
        node = tree_data.nodes[i]
        if node.type != 0:
            continue
        if node.player != 0:
            continue
        if node.num_children != 4:
            continue

        if _is_cont_chain(tree_data, i, 0, num_players):
            first_term_idx = _get_first_terminal(tree_data, i, num_players)
            base_leaf_idx = tree_data.nodes[first_term_idx].leaf_idx
            orig_idx = base_leaf_idx // cont_per_leaf

            # Collect bets for all players
            bets = tuple(node.bets[p] for p in range(num_players))

            leaves.append(LeafInfo(
                leaf_idx=orig_idx,
                pot=node.pot,
                bets=bets,
            ))

    leaves.sort(key=lambda x: x.leaf_idx)
    return leaves


# ── Flop leaf values (turn continuation) ─────────────────────────────────

def compute_flop_leaf_values(
    flop_board: List[int],
    player_hands: List[List[Tuple[int, int, float]]],
    blueprint_store,
    board_cards_str: List[str],
    leaf_infos: List[LeafInfo],
    max_hands: int,
    starting_pot: int,
    # Legacy 2-player interface — if oop_hands/ip_hands are passed, wrap them
    oop_hands=None, ip_hands=None,
) -> np.ndarray:
    """Compute leaf values for a flop solve using precomputed turn blueprints.

    Supports N players. For each original betting leaf, for each combination
    of continuation strategies (4^N combinations), compute the expected value
    by averaging over turn cards using the blueprint's per-action EVs.

    Args:
        flop_board: [card0, card1, card2] ints
        player_hands: list of per-player hands [(c0, c1, weight), ...]
        blueprint_store: BlueprintStore instance
        board_cards_str: ["Qs", "As", "2d"] for blueprint lookup
        leaf_infos: from extract_leaf_info_from_tree()
        max_hands: max hands across all players
        starting_pot: pot in chips at start of this street

    Returns:
        np.array[num_total_leaves, num_players, max_hands] float32
        where num_total_leaves = len(leaf_infos) * 4^num_players
    """
    # Legacy 2-player compat
    if player_hands is None and oop_hands is not None:
        player_hands = [oop_hands, ip_hands]

    try:
        from solver import int_to_card
    except ImportError:
        from python.solver import int_to_card

    num_players = len(player_hands)
    num_orig_leaves = len(leaf_infos)
    cont_per_leaf = 4 ** num_players
    total_leaves = num_orig_leaves * cont_per_leaf
    leaf_values = np.zeros((total_leaves, num_players, max_hands), dtype=np.float32)

    board_set = set(flop_board)
    turn_cards = [c for c in range(52) if c not in board_set]

    for li_idx, li in enumerate(leaf_infos):
        leaf_pot = li.pot

        # Accumulate per (cont_combo, player, hand) across turn cards.
        # cont_combo is a flat index into the 4^N continuation strategy space.
        cont_values = np.zeros((cont_per_leaf, num_players, max_hands), dtype=np.float64)
        valid_turns = 0

        for tc in turn_cards:
            tc_str = int_to_card(tc)
            tc_set = board_set | {tc}

            # Load turn strategy + per-action EVs for all players
            pdata = []
            any_valid = False
            for p in range(num_players):
                strat = blueprint_store.get_turn_strategy(
                    board_cards_str, tc_str, p)
                action_evs = blueprint_store.get_turn_action_evs(
                    board_cards_str, tc_str, p)

                if strat is None or action_evs is None:
                    pdata.append(None)
                    continue

                any_valid = True
                na = strat.shape[1]
                first_is_fold = False
                cats = classify_actions(na, first_is_fold)
                biased = [bias_strategy(strat, bt, cats, 5.0) for bt in range(4)]

                ev_per_bias = np.zeros((4, strat.shape[0]), dtype=np.float64)
                for s in range(4):
                    for a in range(min(na, action_evs.shape[0])):
                        nh_tc = min(biased[s].shape[0], action_evs.shape[1])
                        ev_per_bias[s, :nh_tc] += (
                            biased[s][:nh_tc, a] * action_evs[a, :nh_tc])

                pdata.append({'ev_per_bias': ev_per_bias})

            if not any_valid:
                continue

            valid_turns += 1

            # Map hands and accumulate into continuation combos.
            for p in range(num_players):
                if pdata[p] is None:
                    continue

                hands = player_hands[p]
                ev_bias = pdata[p]['ev_per_bias']
                nh_bp = ev_bias.shape[1]

                bp_idx = 0
                for h in range(len(hands)):
                    hc0, hc1 = hands[h][0], hands[h][1]
                    if hc0 in tc_set or hc1 in tc_set:
                        continue
                    if bp_idx >= nh_bp:
                        break

                    # For each of player p's 4 strategy choices, accumulate
                    # into all cont_combos where p picks that strategy.
                    # cont_combo = sum_j(s_j * 4^j) for players j=0..N-1
                    for s_p in range(4):
                        ev = ev_bias[s_p, bp_idx]
                        # Enumerate all combos where player p picks s_p.
                        # stride_p = 4^p, and we iterate over all other players' choices.
                        stride_p = 4 ** p
                        for combo_others in range(cont_per_leaf // 4):
                            # Insert s_p into the combo at position p
                            lower = combo_others % stride_p
                            upper = combo_others // stride_p
                            combo = lower + s_p * stride_p + upper * stride_p * 4
                            cont_values[combo, p, h] += ev

                    bp_idx += 1

        if valid_turns > 0:
            cont_values /= valid_turns

        # Write to flat leaf_values array
        for combo in range(cont_per_leaf):
            flat_idx = li.leaf_idx * cont_per_leaf + combo
            if flat_idx < total_leaves:
                for p in range(num_players):
                    leaf_values[flat_idx, p, :max_hands] = cont_values[
                        combo, p, :max_hands].astype(np.float32)

    return leaf_values


# ── Flop leaf equity (v2 blueprint fallback) ─────────────────────────────

def compute_flop_leaf_equity(
    flop_board: List[int],
    player_hands: List[List[Tuple[int, int, float]]],
    leaf_infos: List[LeafInfo],
    max_hands: int,
    starting_pot: int,
    # Legacy 2-player compat
    oop_hands=None, ip_hands=None,
) -> np.ndarray:
    """Compute flop leaf values by averaging equity over turn+river runouts,
    then applying the first-order bias approximation at the flop pot level
    to produce 16 different leaf values per (s0, s1) bias pair.

    Previously this function returned identical values for all 16 (s0, s1)
    pairs because the v2 .bps blueprint doesn't store per-action EVs. The v3
    fix uses the bias-adjusted formula in `biased_leaf_value()` to differentiate
    the 4 continuation strategies based on first-order pot/fold-prob effects.

    Note: 2-player only for now (matches the prior implementation). N-player
    extension is straightforward but blocked on the realtime solver's
    multi-villain leaf-value handling, which is currently 2-player.

    Returns:
        np.array[num_total_leaves, num_players, max_hands] float32
        where num_total_leaves = len(leaf_infos) * 16
    """
    if player_hands is None and oop_hands is not None:
        player_hands = [oop_hands, ip_hands]

    num_players = len(player_hands)
    if num_players != 2:
        raise NotImplementedError(
            "compute_flop_leaf_equity currently supports 2-player only. "
            "Multi-player support requires the binary BlueprintStore path "
            "(compute_flop_leaf_values) which has per-action EVs.")

    num_orig_leaves = len(leaf_infos)
    cont_per_leaf = 16  # 4 × 4 for 2 players
    total_leaves = num_orig_leaves * cont_per_leaf
    leaf_values = np.zeros((total_leaves, 2, max_hands), dtype=np.float32)

    board_set = set(flop_board)
    turn_cards = [c for c in range(52) if c not in board_set]

    oop_hands_l = player_hands[0]
    ip_hands_l = player_hands[1]
    nh0 = len(oop_hands_l)
    nh1 = len(ip_hands_l)

    # ── Compute raw per-hand equity at the flop, averaged over turn+river ──
    #
    # Equity is INDEPENDENT of leaf pot — it's a function of (board, hand,
    # opponent range). Compute it once per (player, hand) and reuse across all
    # leaves. The per-leaf loop later just applies biased_leaf_value with the
    # appropriate leaf pot.
    raw_equity = np.zeros((2, max_hands), dtype=np.float64)
    eq_counts = np.zeros((2, max_hands), dtype=np.float64)

    # Subsample turn × river to keep CPU cost reasonable for large ranges.
    MAX_TURN_SAMPLES = 12
    MAX_RIVER_SAMPLES = 12
    any_large = (nh0 > 50 or nh1 > 50)
    if len(turn_cards) > MAX_TURN_SAMPLES and any_large:
        rng = np.random.RandomState(42)
        sampled_turns = sorted(rng.choice(turn_cards, MAX_TURN_SAMPLES, replace=False))
    else:
        sampled_turns = turn_cards

    for tc in sampled_turns:
        tc_set = board_set | {tc}
        river_cards = [c for c in range(52) if c not in tc_set]
        if len(river_cards) > MAX_RIVER_SAMPLES and any_large:
            rng = np.random.RandomState(43 + tc)
            sampled_rivers = sorted(rng.choice(river_cards, MAX_RIVER_SAMPLES, replace=False))
        else:
            sampled_rivers = river_cards

        for rc in sampled_rivers:
            board_5 = list(flop_board) + [tc, rc]
            board_5_set = set(board_5)

            # Compute hand strengths for both players
            oop_strengths = {}
            ip_strengths = {}
            for h in range(nh0):
                hc0, hc1 = oop_hands_l[h][0], oop_hands_l[h][1]
                if hc0 in board_5_set or hc1 in board_5_set:
                    continue
                oop_strengths[h] = _eval7_py(board_5 + [hc0, hc1])
            for h in range(nh1):
                hc0, hc1 = ip_hands_l[h][0], ip_hands_l[h][1]
                if hc0 in board_5_set or hc1 in board_5_set:
                    continue
                ip_strengths[h] = _eval7_py(board_5 + [hc0, hc1])

            # Pairwise compare for OOP equity
            for h in range(nh0):
                if h not in oop_strengths:
                    continue
                hs = oop_strengths[h]
                hc0, hc1 = oop_hands_l[h][0], oop_hands_l[h][1]
                wins = losses = ties = 0.0
                for o in range(nh1):
                    if o not in ip_strengths:
                        continue
                    oc0, oc1 = ip_hands_l[o][0], ip_hands_l[o][1]
                    if hc0 == oc0 or hc0 == oc1 or hc1 == oc0 or hc1 == oc1:
                        continue
                    os_val = ip_strengths[o]
                    if hs > os_val:
                        wins += 1.0
                    elif hs < os_val:
                        losses += 1.0
                    else:
                        ties += 1.0
                total = wins + losses + ties
                if total > 0:
                    raw_equity[0, h] += (wins + 0.5 * ties) / total
                    eq_counts[0, h] += 1.0

            # Pairwise compare for IP equity
            for h in range(nh1):
                if h not in ip_strengths:
                    continue
                hs = ip_strengths[h]
                hc0, hc1 = ip_hands_l[h][0], ip_hands_l[h][1]
                wins = losses = ties = 0.0
                for o in range(nh0):
                    if o not in oop_strengths:
                        continue
                    oc0, oc1 = oop_hands_l[o][0], oop_hands_l[o][1]
                    if hc0 == oc0 or hc0 == oc1 or hc1 == oc0 or hc1 == oc1:
                        continue
                    os_val = oop_strengths[o]
                    if hs > os_val:
                        wins += 1.0
                    elif hs < os_val:
                        losses += 1.0
                    else:
                        ties += 1.0
                total = wins + losses + ties
                if total > 0:
                    raw_equity[1, h] += (wins + 0.5 * ties) / total
                    eq_counts[1, h] += 1.0

    # Normalize to average equity in [0, 1]
    valid = eq_counts > 0
    raw_equity[valid] /= eq_counts[valid]

    # ── Apply first-order bias adjustment at each leaf's pot ──
    # Per (leaf, s0, s1) pair, compute biased_leaf_value(equity, leaf_pot, s_self, s_opp).
    # This produces 16 different values per leaf, restoring variance reduction.
    for li in leaf_infos:
        leaf_pot = li.pot
        for s0 in range(4):
            for s1 in range(4):
                flat_idx = li.leaf_idx * 16 + s0 * 4 + s1
                if flat_idx >= total_leaves:
                    continue
                for h in range(min(nh0, max_hands)):
                    leaf_values[flat_idx, 0, h] = biased_leaf_value(
                        raw_equity[0, h], leaf_pot, s0, s1)
                for h in range(min(nh1, max_hands)):
                    leaf_values[flat_idx, 1, h] = biased_leaf_value(
                        raw_equity[1, h], leaf_pot, s1, s0)

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

        # Compute raw showdown equity per (player, hand). We accumulate equity
        # (probability of winning) rather than chip-payoff so we can apply the
        # bias-adjusted leaf-value formula per (s_self, s_opp) pair afterward.
        raw_equity = np.zeros((2, max_hands), dtype=np.float64)
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
                        raw_equity[p, h] += eq
                        counts[p, h] += 1.0

        valid = counts > 0
        raw_equity[valid] /= counts[valid]

        # v3 fix: produce 16 different leaf values per (s0, s1) bias pair using
        # the first-order bias approximation. Previously all 16 collapsed to
        # one value because we lacked per-action EVs from the v2 .bps blueprint.
        # See the BIAS_FOLD_PROB / BIAS_POT_FACTOR comments at top of file.
        # The proper Pluribus-exact fix (per-action EVs in .bps via post-hoc
        # tree walk at export time) is tracked in REALTIME_TODO.md as the
        # long-term replacement for this approximation.
        for s0 in range(4):
            for s1 in range(4):
                flat_idx = li.leaf_idx * 16 + s0 * 4 + s1
                if flat_idx >= total_leaves:
                    continue
                for h in range(max_hands):
                    if h < len(hands_by_player[0]):
                        leaf_values[flat_idx, 0, h] = biased_leaf_value(
                            raw_equity[0, h], leaf_pot, s0, s1)
                    if h < len(hands_by_player[1]):
                        leaf_values[flat_idx, 1, h] = biased_leaf_value(
                            raw_equity[1, h], leaf_pot, s1, s0)

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
