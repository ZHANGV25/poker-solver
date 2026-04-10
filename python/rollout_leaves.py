"""Rollout-based leaf value computation (Pluribus / DLS 2018 §4).

Replaces the first-order bias approximation in leaf_values.py with an
exact Monte-Carlo rollout under each player's chosen continuation
strategy. This is the canonical Brown-Sandholm approach:

  Brown & Sandholm 2018 NeurIPS "Depth-Limited Solving for Imperfect-
  Information Games" §6.2: "To estimate the values of a state when the
  depth limit is reached on the second round, we sample rollouts of
  each of the stored best-response strategies. In order to reduce
  variance and converge more quickly, we conduct multiple rollouts
  upon reaching a leaf node. We found the optimal number of rollouts
  to be three given our memory access speeds."

  Pluribus Science 2019 supplement §4 / Algorithm 2: "each player still
  in the hand simultaneously chooses one of four different continuation
  strategies to play for the remainder of the game. … This final choice
  of strategy for the remainder of the game is essentially just another
  action in the subgame and is selected via the search algorithm."

Mechanism:
  1. For each subgame leaf, for each of 4^N bias profiles (s_0,...,s_{N-1}):
  2. For each of NUM_ROLLOUTS rollouts (default 3):
       a. Starting from the leaf state (pot, active players, current board),
          deal the remaining chance cards one at a time.
       b. At each decision node reached, look up the blueprint σ̄ at the
          hand's bucket, apply the player's chosen bias transform, sample
          one action from the resulting distribution.
       c. Continue until showdown or everyone folds.
  3. Average chip payoffs across rollouts and across runouts.

Fallback: if the blueprint doesn't have postflop strategies loaded
(BlueprintV2 with streets_to_load=[1] only, or a legacy boost-mixer
file where lookups always fail), this module returns None and callers
fall back to leaf_values.compute_*_leaf_equity which uses the
first-order bias approximation.

Layer 3.3 (future) ports the hot rollout loop to CUDA for speed.
This Python implementation is the correctness reference.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Local imports (relative to the python/ package).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from leaf_values import _eval7_py, LeafInfo
    from solver import card_to_int, int_to_card
except ImportError:
    from python.leaf_values import _eval7_py, LeafInfo  # type: ignore[no-redef]
    from python.solver import card_to_int, int_to_card  # type: ignore[no-redef]


# ── Constants ───────────────────────────────────────────────────────────

NUM_ROLLOUTS_DEFAULT: int = 3

# Bias profile transforms. Must match the 4 continuation strategies defined
# in leaf_values.bias_strategy() (which in turn matches Pluribus §4).
#
# 0 = unmodified blueprint
# 1 = fold-biased:  P(fold)  *= 5, then renormalize
# 2 = call-biased:  P(call)  *= 5, then renormalize
# 3 = raise-biased: P(raise) *= 5, then renormalize
BIAS_MULTIPLIER: float = 5.0
NUM_BIAS_PROFILES: int = 4


def apply_bias(
    strategy: np.ndarray, bias_type: int, action_categories: List[str]
) -> np.ndarray:
    """Apply Pluribus 5x bias to a strategy vector.

    Args:
        strategy: [num_actions] probability distribution
        bias_type: 0=unmodified, 1=fold, 2=call, 3=raise
        action_categories: list of category strings per action,
            e.g. ['fold','call','raise','raise']

    Returns:
        [num_actions] biased + renormalized distribution.
    """
    if bias_type == 0:
        return strategy

    target = {1: "fold", 2: "call", 3: "raise"}.get(bias_type)
    if target is None:
        return strategy

    biased = strategy.copy()
    for a, cat in enumerate(action_categories):
        if a < len(biased) and cat == target:
            biased[a] *= BIAS_MULTIPLIER

    s = biased.sum()
    if s <= 1e-10:
        # Pathological: no mass on target after bias. Fall back to
        # uniform over the original distribution's support.
        nz = strategy > 0
        if nz.any():
            return strategy / strategy.sum()
        return np.ones_like(biased) / len(biased)
    return biased / s


def classify_actions(
    num_actions: int, first_is_fold: bool
) -> List[str]:
    """Build a per-action category vector for bias application.

    Matches leaf_values.classify_actions(). The category set is
    {'fold', 'call', 'raise'} — any bet action is categorized as 'raise'.

    Args:
        num_actions: number of actions at this node
        first_is_fold: True if the first action is a fold (facing a bet);
            False for a check-initial node (no bet to fold to).
    """
    categories: List[str] = []
    if first_is_fold:
        categories.append("fold")
        categories.append("call")
        for _ in range(2, num_actions):
            categories.append("raise")
    else:
        categories.append("call")  # check
        for _ in range(1, num_actions):
            categories.append("raise")
    return categories


# ── Rollout state ───────────────────────────────────────────────────────

class LeafState:
    """The state at a depth-limited subgame leaf, captured for rollout."""

    __slots__ = (
        "board",           # list of card ints (3, 4, or 5 cards)
        "pot",             # chips in pot
        "active",          # list of bool, one per player
        "stacks",          # list of remaining chip stacks per player
        "bets",            # list of total chips committed per player this hand
        "street",          # 'flop' | 'turn' | 'river'
        "player_hands",    # list[player] -> list of (c0, c1, weight)
    )

    def __init__(
        self,
        board: List[int],
        pot: int,
        active: List[bool],
        stacks: List[int],
        bets: List[int],
        street: str,
        player_hands: List[List[Tuple[int, int, float]]],
    ):
        self.board = list(board)
        self.pot = int(pot)
        self.active = list(active)
        self.stacks = list(stacks)
        self.bets = list(bets)
        self.street = street
        self.player_hands = player_hands


# ── Rollout simulator ───────────────────────────────────────────────────

def compute_leaf_value_via_rollout(
    leaf_state: LeafState,
    bias_profile: Sequence[int],
    blueprint_v2,
    num_rollouts: int = NUM_ROLLOUTS_DEFAULT,
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """Compute per-player per-hand EV at a depth-limited leaf via rollout.

    This is the canonical Pluribus / DLS 2018 leaf value computation.
    For each rollout:
      1. Deal the remaining chance cards (turn and/or river).
      2. Simulate the remaining betting rounds. At each decision, look
         up the blueprint σ̄ for the active player's bucket on the
         current board, apply the player's chosen bias, sample one
         action from the biased distribution.
      3. On terminal (showdown or everyone-but-one folded), compute
         each player's net chip payoff.

    Args:
        leaf_state: captured state at the subgame leaf
        bias_profile: per-player bias choice (0-3), length == num_players
        blueprint_v2: BlueprintV2 instance with postflop streets loaded
        num_rollouts: number of Monte-Carlo rollouts per (hero hand)
        rng: numpy Generator for reproducibility

    Returns:
        np.ndarray[num_players, max_hands] float32 — per-hand mean EV for
        each player at this leaf under the given bias profile. Returns
        None if blueprint_v2 lacks postflop strategies (caller should
        fall back to the first-order bias approximation).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if blueprint_v2 is None:
        return None

    # Verify blueprint has postflop streets loaded. We probe by asking
    # for a strategy at a canonical board; if the answer is None AND
    # has_action_evs is False, we assume no postflop coverage.
    if not _blueprint_has_postflop(blueprint_v2):
        return None

    num_players = len(leaf_state.player_hands)
    assert len(bias_profile) == num_players, "bias_profile length mismatch"

    max_hands = max(len(h) for h in leaf_state.player_hands)
    leaf_values = np.zeros((num_players, max_hands), dtype=np.float64)
    eq_counts = np.zeros((num_players, max_hands), dtype=np.float64)

    # For each (hero_player, hero_hand), we run num_rollouts independent
    # rollouts. The opponent's hand is sampled from their range each
    # rollout. The rollout follows all players' biased blueprint
    # strategies until terminal.
    for hero_p in range(num_players):
        if not leaf_state.active[hero_p]:
            continue
        for h_idx, (hc0, hc1, weight) in enumerate(leaf_state.player_hands[hero_p]):
            if weight <= 0.0:
                continue
            if _cards_conflict_with_board({hc0, hc1}, leaf_state.board):
                continue

            payoff_sum = 0.0
            payoff_n = 0
            for _ in range(num_rollouts):
                # Sample opponents' hands from their ranges, avoiding
                # conflicts with the board and hero's cards.
                used = set(leaf_state.board)
                used.add(hc0)
                used.add(hc1)
                sampled_hands: List[Optional[Tuple[int, int]]] = [None] * num_players
                sampled_hands[hero_p] = (hc0, hc1)
                ok = True
                for p in range(num_players):
                    if p == hero_p:
                        continue
                    if not leaf_state.active[p]:
                        continue
                    pair = _sample_hand_avoiding(
                        leaf_state.player_hands[p], used, rng
                    )
                    if pair is None:
                        ok = False
                        break
                    sampled_hands[p] = pair
                    used.add(pair[0])
                    used.add(pair[1])
                if not ok:
                    continue

                # Run the simulation
                payoff = _simulate_rollout(
                    leaf_state=leaf_state,
                    hero_player=hero_p,
                    bias_profile=bias_profile,
                    blueprint_v2=blueprint_v2,
                    sampled_hands=sampled_hands,
                    rng=rng,
                )
                payoff_sum += payoff
                payoff_n += 1

            if payoff_n > 0:
                leaf_values[hero_p, h_idx] = payoff_sum / payoff_n
                eq_counts[hero_p, h_idx] = 1.0

    # For hands we couldn't evaluate (blocked or bad sampling), set 0
    # and let the caller handle — the CFR kernel treats zero-weight hands
    # as not contributing anyway.
    return leaf_values.astype(np.float32)


def _blueprint_has_postflop(blueprint_v2) -> bool:
    """Heuristic check: does this BlueprintV2 have flop+ streets loaded?

    We look at streets_to_load to see if street 1, 2, or 3 are present,
    AND at whether the blueprint is non-empty.
    """
    streets = getattr(blueprint_v2, "streets_to_load", [])
    if 1 not in streets and 2 not in streets and 3 not in streets:
        return False
    textures = getattr(blueprint_v2, "_textures", {})
    return len(textures) > 0


def _cards_conflict_with_board(cards: set, board: List[int]) -> bool:
    return any(c in board for c in cards)


def _sample_hand_avoiding(
    hand_list: List[Tuple[int, int, float]],
    used: set,
    rng: np.random.Generator,
) -> Optional[Tuple[int, int]]:
    """Sample one (c0, c1) from hand_list weighted by the third field,
    skipping any pair that conflicts with `used`.
    """
    candidates = [(c0, c1, w) for c0, c1, w in hand_list
                  if w > 0.0 and c0 not in used and c1 not in used]
    if not candidates:
        return None
    weights = np.array([w for _, _, w in candidates], dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        return None
    idx = int(rng.choice(len(candidates), p=weights / total))
    c0, c1, _ = candidates[idx]
    return (c0, c1)


_ROLLOUT_BET_FRACTIONS: Tuple[float, ...] = (0.33, 0.75, 1.5)
_ROLLOUT_MAX_RAISES: int = 3
_NUM_BUCKETS: int = 200
_EHS_SAMPLES: int = 50


def _hand_to_bucket(
    c0: int, c1: int, board_ints: List[int], num_buckets: int = _NUM_BUCKETS,
) -> int:
    """Map a hand to its EHS bucket on the given board.

    Uses deterministic Monte-Carlo EHS estimation (same RNG seed per hand
    for reproducibility). Maps EHS ∈ [0,1] linearly to bucket index.
    """
    rng_seed = c0 * 52 + c1
    local_rng = np.random.RandomState(rng_seed)
    blocked = set(board_ints) | {c0, c1}
    available = [c for c in range(52) if c not in blocked]
    if len(available) < 4:
        return 0
    available_arr = np.array(available)
    wins, ties, total = 0, 0, 0
    for _ in range(_EHS_SAMPLES):
        idx = local_rng.choice(len(available_arr), 4, replace=False)
        oc0, oc1 = int(available_arr[idx[0]]), int(available_arr[idx[1]])
        bc0, bc1 = int(available_arr[idx[2]]), int(available_arr[idx[3]])
        board_5 = list(board_ints[:3]) + [bc0, bc1] if len(board_ints) <= 3 else list(board_ints)
        # For turn/river boards we already have enough cards
        if len(board_ints) == 4:
            board_5 = list(board_ints) + [bc0]
        elif len(board_ints) >= 5:
            board_5 = list(board_ints[:5])
        else:
            board_5 = list(board_ints) + [bc0, bc1]
        hero_str = _eval7_py(board_5 + [c0, c1])
        opp_str = _eval7_py(board_5 + [oc0, oc1])
        if hero_str > opp_str:
            wins += 1
        elif hero_str == opp_str:
            ties += 1
        total += 1
    ehs = (wins + 0.5 * ties) / max(total, 1)
    return min(int(ehs * num_buckets), num_buckets - 1)


def _get_blueprint_strategy(
    blueprint_v2,
    board_ints: List[int],
    action_history: List[int],
    player: int,
    bucket: int,
    street: int,
) -> Optional[np.ndarray]:
    """Look up the blueprint σ̄(bucket) at a postflop decision node.

    Returns a 1-D probability array over actions, or None if not found.
    """
    strat = blueprint_v2.get_strategy(
        board_ints, action_history, player, bucket, street=street,
    )
    if strat is None:
        return None
    arr = np.asarray(strat, dtype=np.float64)
    s = arr.sum()
    if s <= 1e-10:
        return None
    return arr / s


def _simulate_rollout(
    leaf_state: LeafState,
    hero_player: int,
    bias_profile: Sequence[int],
    blueprint_v2,
    sampled_hands: List[Optional[Tuple[int, int]]],
    rng: np.random.Generator,
) -> float:
    """Simulate the remainder of the hand from a leaf state and return
    hero's net chip payoff (positive = win, negative = loss).

    Pluribus-faithful path: at each decision node, look up the blueprint
    σ̄ for the player's hand bucket, apply bias, sample one action. Run
    through all remaining streets until showdown or fold.
    """
    num_players = len(leaf_state.active)
    active = list(leaf_state.active)
    pot = leaf_state.pot
    stacks = list(leaf_state.stacks)
    bets = list(leaf_state.bets)
    board = list(leaf_state.board)

    # Determine the streets remaining after the leaf
    streets_remaining = _streets_after(leaf_state.street)

    for street_name in streets_remaining:
        # Deal the next card
        street_num = {"flop": 1, "turn": 2, "river": 3}[street_name]
        if street_name == "turn" and len(board) < 4:
            card = _deal_one_card(board, sampled_hands, rng)
            if card is None:
                break
            board.append(card)
        elif street_name == "river" and len(board) < 5:
            card = _deal_one_card(board, sampled_hands, rng)
            if card is None:
                break
            board.append(card)

        # Reset street bets — each player starts a new betting round
        street_bets = [0] * num_players
        current_bet = 0
        num_raises = 0
        acted = [False] * num_players
        action_history: List[int] = []

        # Betting round: loop until all active players have acted and
        # all bets are equal, or only one player remains.
        max_iterations = num_players * (_ROLLOUT_MAX_RAISES + 2)
        iteration = 0
        # OOP acts first postflop. For simplicity with the 2-player
        # case (the only one supported in v1), player 0 = OOP.
        acting_player = 0

        while iteration < max_iterations:
            iteration += 1

            # Find next active player who needs to act
            found = False
            for _ in range(num_players):
                if active[acting_player] and (
                    not acted[acting_player]
                    or street_bets[acting_player] < current_bet
                ):
                    found = True
                    break
                acting_player = (acting_player + 1) % num_players
            if not found:
                break

            # Count active players
            n_active = sum(active)
            if n_active <= 1:
                break

            p = acting_player
            pair = sampled_hands[p]
            if pair is None:
                acting_player = (acting_player + 1) % num_players
                continue
            c0, c1 = pair

            # Look up blueprint strategy
            bucket = _hand_to_bucket(c0, c1, board)
            strat = _get_blueprint_strategy(
                blueprint_v2, board, action_history, p, bucket, street_num,
            )

            # Determine available actions and their categories
            facing_bet = current_bet > street_bets[p]
            if facing_bet:
                # Actions: fold, call, raise sizes..., all-in
                n_actions = 2 + len(_ROLLOUT_BET_FRACTIONS) + 1
                if num_raises >= _ROLLOUT_MAX_RAISES:
                    n_actions = 2  # fold, call only
                categories = classify_actions(n_actions, first_is_fold=True)
            else:
                # Actions: check, bet sizes..., all-in
                n_actions = 1 + len(_ROLLOUT_BET_FRACTIONS) + 1
                if num_raises >= _ROLLOUT_MAX_RAISES:
                    n_actions = 1  # check only
                categories = classify_actions(n_actions, first_is_fold=False)

            # Build probability distribution
            if strat is not None and len(strat) >= n_actions:
                probs = strat[:n_actions].copy()
            else:
                # Fallback: uniform over available actions
                probs = np.ones(n_actions, dtype=np.float64) / n_actions

            # Apply bias
            biased = apply_bias(probs, bias_profile[p], categories)

            # Sample action
            s = biased.sum()
            if s <= 1e-10:
                biased = np.ones(n_actions, dtype=np.float64) / n_actions
            else:
                biased = biased / s
            action_idx = int(rng.choice(n_actions, p=biased))

            # Execute action
            if facing_bet:
                if action_idx == 0:
                    # Fold
                    active[p] = False
                    acted[p] = True
                elif action_idx == 1:
                    # Call
                    call_amount = current_bet - street_bets[p]
                    call_amount = min(call_amount, stacks[p])
                    stacks[p] -= call_amount
                    street_bets[p] += call_amount
                    bets[p] += call_amount
                    pot += call_amount
                    acted[p] = True
                else:
                    # Raise or all-in
                    if action_idx == n_actions - 1 or num_raises >= _ROLLOUT_MAX_RAISES:
                        # All-in
                        allin = stacks[p]
                        stacks[p] = 0
                        street_bets[p] += allin
                        bets[p] += allin
                        pot += allin
                        current_bet = max(current_bet, street_bets[p])
                    else:
                        # Sized raise: pot fraction
                        frac_idx = action_idx - 2
                        frac = _ROLLOUT_BET_FRACTIONS[min(frac_idx, len(_ROLLOUT_BET_FRACTIONS) - 1)]
                        raise_to = int(current_bet + pot * frac)
                        raise_amount = raise_to - street_bets[p]
                        raise_amount = min(max(raise_amount, 1), stacks[p])
                        stacks[p] -= raise_amount
                        street_bets[p] += raise_amount
                        bets[p] += raise_amount
                        pot += raise_amount
                        current_bet = street_bets[p]
                    num_raises += 1
                    acted[p] = True
                    # Reopens action for others
                    for q in range(num_players):
                        if q != p and active[q]:
                            acted[q] = False
            else:
                if action_idx == 0:
                    # Check
                    acted[p] = True
                else:
                    # Bet or all-in
                    if action_idx == n_actions - 1 or num_raises >= _ROLLOUT_MAX_RAISES:
                        # All-in
                        allin = stacks[p]
                        stacks[p] = 0
                        street_bets[p] += allin
                        bets[p] += allin
                        pot += allin
                        current_bet = street_bets[p]
                    else:
                        # Sized bet: pot fraction
                        frac_idx = action_idx - 1
                        frac = _ROLLOUT_BET_FRACTIONS[min(frac_idx, len(_ROLLOUT_BET_FRACTIONS) - 1)]
                        bet_amount = int(pot * frac)
                        bet_amount = min(max(bet_amount, 1), stacks[p])
                        stacks[p] -= bet_amount
                        street_bets[p] += bet_amount
                        bets[p] += bet_amount
                        pot += bet_amount
                        current_bet = street_bets[p]
                    num_raises += 1
                    acted[p] = True
                    for q in range(num_players):
                        if q != p and active[q]:
                            acted[q] = False

            action_history.append(action_idx)
            acting_player = (acting_player + 1) % num_players

        # Check if hand ended (everyone folded to one)
        n_active = sum(active)
        if n_active <= 1:
            break

    # Terminal: showdown or last-player-standing
    n_active = sum(active)
    if n_active <= 1:
        # Last player standing wins the pot
        if active[hero_player]:
            return float(pot) - float(bets[hero_player])
        else:
            return -float(bets[hero_player])

    # Showdown — deal remaining board cards if needed
    while len(board) < 5:
        card = _deal_one_card(board, sampled_hands, rng)
        if card is None:
            break
        board.append(card)

    return _showdown_payoff(
        full_board=board,
        hero_player=hero_player,
        sampled_hands=sampled_hands,
        leaf_state=LeafState(
            board=board, pot=pot, active=active,
            stacks=stacks, bets=bets, street="river",
            player_hands=leaf_state.player_hands,
        ),
    )


def _streets_after(street: str) -> List[str]:
    """Return the list of streets remaining after the given street.

    A flop leaf needs turn + river betting. A turn leaf needs river.
    A river leaf needs nothing (immediate showdown).
    """
    if street == "flop":
        return ["turn", "river"]
    if street == "turn":
        return ["river"]
    return []


def _deal_one_card(
    board: List[int],
    sampled_hands: List[Optional[Tuple[int, int]]],
    rng: np.random.Generator,
) -> Optional[int]:
    """Deal one card that doesn't conflict with the board or any player's hand."""
    used = set(board)
    for pair in sampled_hands:
        if pair is not None:
            used.add(pair[0])
            used.add(pair[1])
    deck = [c for c in range(52) if c not in used]
    if not deck:
        return None
    return int(rng.choice(deck))


def _deal_to_river(
    current_board: List[int],
    active: List[bool],
    rng: np.random.Generator,
) -> Optional[List[int]]:
    """Deal remaining turn/river cards from a random distribution."""
    needed = 5 - len(current_board)
    if needed <= 0:
        return list(current_board)
    used = set(current_board)
    deck = [c for c in range(52) if c not in used]
    if len(deck) < needed:
        return None
    dealt = rng.choice(len(deck), size=needed, replace=False)
    return list(current_board) + [deck[i] for i in dealt]


def _showdown_payoff(
    full_board: List[int],
    hero_player: int,
    sampled_hands: List[Optional[Tuple[int, int]]],
    leaf_state: LeafState,
) -> float:
    """Compute hero's net chip payoff at showdown with the given hands."""
    # Gather strengths for all active players
    strengths = {}
    for p, pair in enumerate(sampled_hands):
        if pair is None or not leaf_state.active[p]:
            continue
        c0, c1 = pair
        strengths[p] = _eval7_py(full_board + [c0, c1])

    if hero_player not in strengths:
        return 0.0

    hero_strength = strengths[hero_player]
    # Count how many have strictly better / tie / worse
    ties = 0
    wins = True
    for p, strength in strengths.items():
        if p == hero_player:
            continue
        if strength > hero_strength:
            wins = False
            break
        if strength == hero_strength:
            ties += 1

    hero_committed = leaf_state.bets[hero_player]

    if not wins:
        # Hero loses — pays their committed amount.
        return -float(hero_committed)

    # Hero wins (outright or split with ties)
    pot = leaf_state.pot
    share = pot / (ties + 1)
    # Net = share won minus committed
    return float(share) - float(hero_committed)


# ── GPU-kernel-compatible leaf value array builder ─────────────────────


def compute_flop_leaf_values_rollout(
    flop_board: List[int],
    player_hands: List[List[Tuple[int, int, float]]],
    blueprint_v2,
    leaf_infos: List["LeafInfo"],
    max_hands: int,
    starting_pot: int,
    num_rollouts: int = NUM_ROLLOUTS_DEFAULT,
) -> Optional[np.ndarray]:
    """Compute flop leaf values via Monte-Carlo rollout.

    Drop-in replacement for leaf_values.compute_flop_leaf_equity with
    the same return shape: np.array[num_total_leaves, num_players, max_hands].

    For each leaf × each bias combo (4^N), runs rollouts under the
    biased blueprint strategies and returns the mean per-hand payoff.

    Returns None if the blueprint doesn't have postflop strategies,
    signaling the caller to fall back to equity-only.

    NOTE: O(leaves × 4^N × hands × rollouts) Python rollouts. Suitable
    for small trees or testing. Layer 3.3 ports this to CUDA.
    """
    if not _blueprint_has_postflop(blueprint_v2):
        return None

    num_players = len(player_hands)
    num_orig_leaves = len(leaf_infos)
    cont_per_leaf = 4 ** num_players
    total_leaves = num_orig_leaves * cont_per_leaf
    leaf_values = np.zeros((total_leaves, num_players, max_hands), dtype=np.float32)

    rng = np.random.default_rng(42)

    for li_idx, li in enumerate(leaf_infos):
        # Build a LeafState for this leaf
        active = [True] * num_players
        stacks = [max(0, int(starting_pot * 2 - li.pot) // num_players)] * num_players
        bets_list = list(li.bets) if hasattr(li, 'bets') else [0] * num_players

        leaf_state = LeafState(
            board=list(flop_board),
            pot=li.pot,
            active=active,
            stacks=stacks,
            bets=bets_list,
            street="flop",
            player_hands=player_hands,
        )

        for combo in range(cont_per_leaf):
            bias_profile = [(combo // (4 ** p)) % 4 for p in range(num_players)]

            result = compute_leaf_value_via_rollout(
                leaf_state=leaf_state,
                bias_profile=bias_profile,
                blueprint_v2=blueprint_v2,
                num_rollouts=num_rollouts,
                rng=rng,
            )

            flat_idx = li.leaf_idx * cont_per_leaf + combo
            if result is not None and flat_idx < total_leaves:
                for p in range(num_players):
                    nh = min(len(player_hands[p]), max_hands)
                    leaf_values[flat_idx, p, :nh] = result[p, :nh]

    return leaf_values
