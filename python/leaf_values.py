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
    leaf_infos: List[LeafInfo],
    max_hands: int,
    starting_pot: int,
    # N-player input (preferred). When provided, num_players = len(player_hands).
    player_hands: Optional[List[List[Tuple[int, int, float]]]] = None,
    # Legacy 2-player input (still accepted for backwards compat with the
    # closure callbacks in hud_solver.py:_flop_leaf_fn_v2).
    oop_hands: Optional[List[Tuple[int, int, float]]] = None,
    ip_hands: Optional[List[Tuple[int, int, float]]] = None,
) -> np.ndarray:
    """Compute flop leaf values by averaging equity over turn+river runouts,
    then applying the first-order bias approximation at the flop pot level
    to produce 4^N leaf values per (s_0, ..., s_{N-1}) bias profile.

    Supports any player count from 2 to 6. The 2-player path is the heads-up
    case used by HU flop solving; the 3-6 player paths are used by Pluribus's
    multi-way flop subgame search where leaves sit at the start of the turn
    (or after the 2nd raise of the flop, whichever is earlier per the
    Pluribus supplement §4).

    Algorithm (per Brown & Sandholm 2018 NeurIPS depth-limited solving):
      1. For each player p, compute showdown equity averaged over all
         (turn × river) runouts: P(p's hand beats every other active
         player's hand at showdown).
      2. For each leaf, for each tuple (s_0, ..., s_{N-1}) ∈ {0,1,2,3}^N
         of bias choices, compute the leaf value via biased_leaf_value()
         which applies first-order pot-size and fold-probability effects.

    Equity at a flop leaf is INDEPENDENT of which betting line led to the
    leaf — only the leaf POT differs. We compute equity once per (player,
    hand) and apply the per-leaf pot adjustment later.

    Returns:
        np.array[num_total_leaves, num_players, max_hands] float32
        where num_total_leaves = len(leaf_infos) * 4^num_players
    """
    if player_hands is None:
        if oop_hands is None or ip_hands is None:
            raise ValueError(
                "compute_flop_leaf_equity requires either player_hands "
                "or both oop_hands and ip_hands"
            )
        player_hands = [oop_hands, ip_hands]

    num_players = len(player_hands)
    if num_players < 2 or num_players > 6:
        raise ValueError(
            f"compute_flop_leaf_equity supports 2-6 players, got {num_players}"
        )

    num_orig_leaves = len(leaf_infos)
    cont_per_leaf = 4 ** num_players
    total_leaves = num_orig_leaves * cont_per_leaf
    leaf_values = np.zeros((total_leaves, num_players, max_hands), dtype=np.float32)

    board_set = set(flop_board)
    turn_cards = [c for c in range(52) if c not in board_set]

    nh = [len(player_hands[p]) for p in range(num_players)]

    # ── Compute raw per-hand equity at the flop, averaged over turn+river ──
    #
    # equity[p, h] = P(player p's hand h beats EVERY other active player's
    #                 hand at showdown), averaged over all sampled (tc, rc).
    # For ties, we award fractional credit 1/k where k is the number of
    # tied-best hands (matching the standard pot-share rule).
    raw_equity = np.zeros((num_players, max_hands), dtype=np.float64)
    eq_counts = np.zeros((num_players, max_hands), dtype=np.float64)

    # Subsample turn × river to keep CPU cost reasonable. The "any_large"
    # gate triggers sampling only when at least one player's range is big
    # enough to make full enumeration prohibitive.
    MAX_TURN_SAMPLES = 12
    MAX_RIVER_SAMPLES = 12
    any_large = any(n > 50 for n in nh)
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

            # Per-player hand strengths for this runout, gated on board/hand
            # cards being disjoint. None for blocked combos.
            strengths_per_p = []
            for p in range(num_players):
                ps = {}
                for h in range(nh[p]):
                    hc0, hc1 = player_hands[p][h][0], player_hands[p][h][1]
                    if hc0 in board_5_set or hc1 in board_5_set:
                        continue
                    ps[h] = _eval7_py(board_5 + [hc0, hc1])
                strengths_per_p.append(ps)

            # For each (player, hand), compute equity vs the joint distribution
            # of all other players' hands on this runout. We iterate the cross
            # product of opponent hands for N-1 opponents which is O(prod_other_nh).
            # For N>=4 this gets expensive; subsample within when needed.
            for p in range(num_players):
                if not strengths_per_p[p]:
                    continue
                _accumulate_nplayer_equity(
                    p, num_players, player_hands, strengths_per_p,
                    raw_equity, eq_counts,
                )

    # Normalize raw_equity to mean equity in [0, 1]
    valid = eq_counts > 0
    raw_equity[valid] /= eq_counts[valid]

    # ── Apply first-order bias adjustment at each leaf's pot ──
    #
    # For an N-player leaf, the bias profile is a tuple s = (s_0, ..., s_{N-1})
    # in {0,1,2,3}^N. We flatten via the standard mixed-radix encoding:
    #   combo_idx = sum_i s_i * 4^i
    # which matches `extract_leaf_info_from_tree`'s leaf indexing convention.
    #
    # biased_leaf_value(equity, pot, s_self, s_opp) currently takes ONE
    # opponent bias. For N>2 we use the average of all opponents' biases as a
    # first-order proxy — that's the same approximation we use for 2-player,
    # extended naturally. The Layer 3 rollout-based path (replacing this
    # first-order approximation entirely) doesn't have this approximation
    # because it actually rolls out under each player's chosen strategy.
    for li in leaf_infos:
        leaf_pot = li.pot
        for combo in range(cont_per_leaf):
            flat_idx = li.leaf_idx * cont_per_leaf + combo
            if flat_idx >= total_leaves:
                continue
            # Decode combo into per-player bias choices
            biases = [(combo // (4 ** i)) % 4 for i in range(num_players)]
            for p in range(num_players):
                # First-order proxy for "opponent bias" when there are
                # multiple opponents: arithmetic mean of the others'
                # biases. With one opponent (N=2) this reduces to that
                # opponent's bias exactly, matching the original 2-player
                # behavior.
                if num_players > 2:
                    other_sum = sum(biases[q] for q in range(num_players) if q != p)
                    s_opp_proxy = int(round(other_sum / (num_players - 1)))
                    s_opp_proxy = max(0, min(3, s_opp_proxy))
                else:
                    s_opp_proxy = biases[1 - p]
                for h in range(min(nh[p], max_hands)):
                    leaf_values[flat_idx, p, h] = biased_leaf_value(
                        raw_equity[p, h], leaf_pot, biases[p], s_opp_proxy
                    )

    return leaf_values


def _accumulate_nplayer_equity(
    hero_p: int,
    num_players: int,
    player_hands,
    strengths_per_p,
    raw_equity: np.ndarray,
    eq_counts: np.ndarray,
) -> None:
    """For one runout (board_5 already encoded into strengths_per_p),
    accumulate hero player's per-hand equity vs the joint distribution of
    all other players' hands.

    For 2 players this is the original pairwise comparison. For 3+ players,
    we iterate the cross product of opponents' hands. Cardinality grows as
    O(prod_q!=p nh[q]); to keep this tractable for 6-max we subsample
    opponent combos when the cross product exceeds a threshold.

    Note: this is a quasi-Monte-Carlo expectation, not exact. The Layer 3
    rollout-based path provides exact (modulo rollout count) leaf values
    and replaces this approximation entirely.
    """
    import itertools
    import random as _random

    hero_strengths = strengths_per_p[hero_p]
    if not hero_strengths:
        return

    opponent_indices = [q for q in range(num_players) if q != hero_p]
    opp_hands_lists = [list(strengths_per_p[q].keys()) for q in opponent_indices]

    # Cross product cardinality
    total_combos = 1
    for opp_hands in opp_hands_lists:
        total_combos *= len(opp_hands)

    if total_combos == 0:
        return

    # Cap the cross product at OPP_COMBO_LIMIT samples per (hero hand, runout).
    # 2000 is empirically enough to bring the equity error below ~0.5% per
    # estimate, while keeping a 6-max evaluation under a few seconds total.
    OPP_COMBO_LIMIT = 2000

    if total_combos <= OPP_COMBO_LIMIT:
        opp_iter = itertools.product(*opp_hands_lists)
        do_subsample = False
    else:
        rng = _random.Random(42)
        def _sample_iter():
            for _ in range(OPP_COMBO_LIMIT):
                yield tuple(rng.choice(opp_hands) for opp_hands in opp_hands_lists)
        opp_iter = _sample_iter()
        do_subsample = True

    # Convert to a list so we can re-iterate per hero hand.
    opp_combos = list(opp_iter)

    for h, hs in hero_strengths.items():
        hc0, hc1 = player_hands[hero_p][h][0], player_hands[hero_p][h][1]
        wins_with_ties = 0.0  # accumulator: 1.0 for outright win, 1/k for k-way tie
        valid_combos = 0

        for combo in opp_combos:
            # Check that no opponent's cards conflict with hero's
            blocked = False
            for q_idx, opp_h in enumerate(combo):
                opp_p = opponent_indices[q_idx]
                oc0, oc1 = player_hands[opp_p][opp_h][0], player_hands[opp_p][opp_h][1]
                if hc0 == oc0 or hc0 == oc1 or hc1 == oc0 or hc1 == oc1:
                    blocked = True
                    break
            if blocked:
                continue
            # Also check pairwise conflicts BETWEEN opponents
            opp_cards = []
            ok = True
            for q_idx, opp_h in enumerate(combo):
                opp_p = opponent_indices[q_idx]
                oc0, oc1 = player_hands[opp_p][opp_h][0], player_hands[opp_p][opp_h][1]
                if oc0 in opp_cards or oc1 in opp_cards:
                    ok = False
                    break
                opp_cards.append(oc0)
                opp_cards.append(oc1)
            if not ok:
                continue

            # Compare hero strength to all opponents
            best_opp = None
            for q_idx, opp_h in enumerate(combo):
                opp_p = opponent_indices[q_idx]
                opp_strength = strengths_per_p[opp_p].get(opp_h)
                if opp_strength is None:
                    continue
                if best_opp is None or opp_strength > best_opp:
                    best_opp = opp_strength

            if best_opp is None:
                continue

            valid_combos += 1
            if hs > best_opp:
                wins_with_ties += 1.0
            elif hs == best_opp:
                # Count how many opponents tie hero
                ties = 1  # hero
                for q_idx, opp_h in enumerate(combo):
                    opp_p = opponent_indices[q_idx]
                    opp_strength = strengths_per_p[opp_p].get(opp_h)
                    if opp_strength == hs:
                        ties += 1
                wins_with_ties += 1.0 / ties

        if valid_combos > 0:
            raw_equity[hero_p, h] += wins_with_ties / valid_combos
            eq_counts[hero_p, h] += 1.0


# ── Turn leaf values (river continuation) ────────────────────────────────

def compute_turn_leaf_values(
    board_4: List[int],
    leaf_infos: List[LeafInfo],
    max_hands: int,
    starting_pot: int,
    # N-player input (preferred). When provided, num_players = len(player_hands).
    player_hands: Optional[List[List[Tuple[int, int, float]]]] = None,
    # Legacy 2-player compat (used by hud_solver.py:_turn_leaf_fn).
    oop_hands: Optional[List[Tuple[int, int, float]]] = None,
    ip_hands: Optional[List[Tuple[int, int, float]]] = None,
) -> np.ndarray:
    """Compute leaf values for a turn subgame, supporting 2-6 players.

    Turn leaves lead to the river. Pluribus solves the river in real-time
    (no depth limit at river — see supplement §4: "all other cases, the
    subgame extends to the end of the game"). For our depth-limited turn
    solver, we compute the continuation value as the average showdown
    equity over all possible river cards, then apply the first-order
    bias adjustment per (s_0, ..., s_{N-1}) bias profile.

    Per Pluribus, when our solver is the canonical end-of-game river
    solver, no leaf values are needed at all (showdown is the terminal).
    This function is the depth-limited fallback when the river solver
    isn't reached because the user's query stops at the turn boundary.

    Each leaf has a different pot (check-check vs bet-call), so the
    showdown payoff differs per leaf.

    Returns:
        np.array[num_total_leaves, num_players, max_hands]
        where num_total_leaves = len(leaf_infos) * 4^num_players
    """
    if player_hands is None:
        if oop_hands is None or ip_hands is None:
            raise ValueError(
                "compute_turn_leaf_values requires either player_hands "
                "or both oop_hands and ip_hands"
            )
        player_hands = [oop_hands, ip_hands]

    num_players = len(player_hands)
    if num_players < 2 or num_players > 6:
        raise ValueError(
            f"compute_turn_leaf_values supports 2-6 players, got {num_players}"
        )

    num_orig_leaves = len(leaf_infos)
    cont_per_leaf = 4 ** num_players
    total_leaves = num_orig_leaves * cont_per_leaf
    leaf_values = np.zeros((total_leaves, num_players, max_hands), dtype=np.float32)

    board_set = set(board_4)
    river_cards = [c for c in range(52) if c not in board_set]

    nh = [len(player_hands[p]) for p in range(num_players)]

    # For large ranges, sample river cards to keep computation feasible.
    # 200 hands × 200 opps × 46 rivers × eval7 = ~85M evals.
    # Sample 12 rivers → ~2.4M evals, still ~3 seconds on CPU.
    MAX_RIVER_SAMPLES = 12
    any_large = any(n > 50 for n in nh)
    if len(river_cards) > MAX_RIVER_SAMPLES and any_large:
        rng = np.random.RandomState(42)  # deterministic
        sampled = sorted(rng.choice(river_cards, MAX_RIVER_SAMPLES, replace=False))
    else:
        sampled = river_cards

    # Equity is independent of leaf pot — compute it once across all sampled
    # rivers, then apply the per-leaf pot adjustment. (The original 2-player
    # implementation accidentally re-computed equity per leaf in the inner
    # loop, which was both slower and gave numerically identical results.)
    raw_equity = np.zeros((num_players, max_hands), dtype=np.float64)
    eq_counts = np.zeros((num_players, max_hands), dtype=np.float64)

    for rc in sampled:
        board_5 = list(board_4) + [rc]
        board_5_set = set(board_5)
        strengths_per_p = []
        for p in range(num_players):
            ps = {}
            for h in range(nh[p]):
                hc0, hc1 = player_hands[p][h][0], player_hands[p][h][1]
                if hc0 in board_5_set or hc1 in board_5_set:
                    continue
                ps[h] = _eval7_py(board_5 + [hc0, hc1])
            strengths_per_p.append(ps)

        for p in range(num_players):
            if not strengths_per_p[p]:
                continue
            _accumulate_nplayer_equity(
                p, num_players, player_hands, strengths_per_p,
                raw_equity, eq_counts,
            )

    valid = eq_counts > 0
    raw_equity[valid] /= eq_counts[valid]

    # Apply first-order bias adjustment at each leaf's pot. Same logic as
    # compute_flop_leaf_equity above — see that function's docstring for
    # the bias-encoding convention.
    for li in leaf_infos:
        leaf_pot = li.pot
        for combo in range(cont_per_leaf):
            flat_idx = li.leaf_idx * cont_per_leaf + combo
            if flat_idx >= total_leaves:
                continue
            biases = [(combo // (4 ** i)) % 4 for i in range(num_players)]
            for p in range(num_players):
                if num_players > 2:
                    other_sum = sum(biases[q] for q in range(num_players) if q != p)
                    s_opp_proxy = int(round(other_sum / (num_players - 1)))
                    s_opp_proxy = max(0, min(3, s_opp_proxy))
                else:
                    s_opp_proxy = biases[1 - p]
                for h in range(min(nh[p], max_hands)):
                    leaf_values[flat_idx, p, h] = biased_leaf_value(
                        raw_equity[p, h], leaf_pot, biases[p], s_opp_proxy
                    )

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
