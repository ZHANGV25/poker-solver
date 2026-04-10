"""Regression test for the N-player generalization of leaf_values.py.

Layer 2.1 lifted the 2-player guard in compute_flop_leaf_equity and
compute_turn_leaf_values. This test verifies:

1. The new player_hands API produces byte-identical output to the
   legacy oop_hands/ip_hands API for N=2 (no behavioral change for
   the existing 2-player callsites).
2. The functions accept 2, 3, 4, 5, and 6 players without raising.
3. Output shapes match the documented contract:
   [num_leaves * 4^N, num_players, max_hands]
4. All output values are finite (no NaN, no inf) for all player counts.
5. Bias profile decoding is consistent: per-leaf, the 4^N leaf values
   are NOT all identical (the first-order bias approximation produces
   meaningful variation).

Pluribus parity context: this is the equity-only fallback path used
when no per-action EVs are available. Layer 3 will replace this with
rollout-based leaf values per Brown & Sandholm 2018 NeurIPS.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from leaf_values import (  # noqa: E402
    LeafInfo,
    compute_flop_leaf_equity,
    compute_turn_leaf_values,
)


# Toy fixtures — small enough to run fast (~1 sec for the whole test)
FLOP = [0, 17, 35]  # 2c, 5d (4*4+1), 9c (8*4+3) — random board
TURN = [0, 17, 35, 47]
LEAF_INFOS = [
    LeafInfo(leaf_idx=0, pot=600, bets=(300, 300)),
    LeafInfo(leaf_idx=1, pot=200, bets=(100, 100)),
]
RANGES = {
    0: [(40, 41, 1.0), (44, 45, 1.0), (8, 9, 1.0)],   # 3 hands
    1: [(2, 3, 1.0), (12, 13, 1.0)],                    # 2 hands
    2: [(20, 21, 1.0), (24, 25, 1.0)],                  # 2 hands
    3: [(32, 33, 1.0)],                                  # 1 hand
    4: [(36, 37, 1.0)],                                  # 1 hand
    5: [(28, 29, 1.0)],                                  # 1 hand
}


def _player_ranges(num_players):
    return [RANGES[p] for p in range(num_players)]


def test_flop_2player_legacy_api_matches_new_api():
    """N-player API with N=2 must produce identical output to legacy oop/ip API."""
    legacy = compute_flop_leaf_equity(
        flop_board=FLOP,
        leaf_infos=LEAF_INFOS,
        max_hands=10,
        starting_pot=200,
        oop_hands=RANGES[0],
        ip_hands=RANGES[1],
    )
    new = compute_flop_leaf_equity(
        flop_board=FLOP,
        leaf_infos=LEAF_INFOS,
        max_hands=10,
        starting_pot=200,
        player_hands=[RANGES[0], RANGES[1]],
    )
    assert legacy.shape == new.shape == (2 * 16, 2, 10)
    assert np.array_equal(legacy, new), "N-player API drift for N=2"


@pytest.mark.parametrize("n_players", [2, 3, 4, 5, 6])
def test_flop_nplayer_shape(n_players):
    """compute_flop_leaf_equity must accept 2-6 players with the right shape."""
    out = compute_flop_leaf_equity(
        flop_board=FLOP,
        leaf_infos=LEAF_INFOS,
        max_hands=10,
        starting_pot=200,
        player_hands=_player_ranges(n_players),
    )
    expected_leaves = 2 * (4 ** n_players)
    assert out.shape == (expected_leaves, n_players, 10)
    assert np.all(np.isfinite(out)), "leaf values must be finite"


@pytest.mark.parametrize("n_players", [2, 3, 4, 5, 6])
def test_turn_nplayer_shape(n_players):
    """compute_turn_leaf_values must accept 2-6 players with the right shape."""
    out = compute_turn_leaf_values(
        board_4=TURN,
        leaf_infos=LEAF_INFOS,
        max_hands=10,
        starting_pot=300,
        player_hands=_player_ranges(n_players),
    )
    expected_leaves = 2 * (4 ** n_players)
    assert out.shape == (expected_leaves, n_players, 10)
    assert np.all(np.isfinite(out))


def test_flop_bias_profiles_produce_distinct_values():
    """The 4^N leaf-value tuple per leaf should NOT all be identical.

    The first-order bias approximation derives leaf values from a single
    raw equity per (player, hand) but adjusts via biased_leaf_value() to
    produce distinct values per (s_self, s_opp) pair. If all 16 (or 64,
    256, ...) values are identical we've lost the variance reduction.
    """
    out = compute_flop_leaf_equity(
        flop_board=FLOP,
        leaf_infos=LEAF_INFOS,
        max_hands=10,
        starting_pot=200,
        player_hands=[RANGES[0], RANGES[1]],
    )
    # Look at leaf 0, player 0, hand 0: 16 values across the bias profiles
    leaf0_p0_h0 = out[0:16, 0, 0]
    distinct = len(np.unique(np.round(leaf0_p0_h0, 4)))
    assert distinct >= 4, (
        f"expected at least 4 distinct leaf values across bias profiles, "
        f"got {distinct} (values: {leaf0_p0_h0.tolist()})"
    )


def test_flop_invalid_player_count_raises():
    with pytest.raises(ValueError, match="2-6 players"):
        compute_flop_leaf_equity(
            flop_board=FLOP,
            leaf_infos=LEAF_INFOS,
            max_hands=10,
            starting_pot=200,
            player_hands=[RANGES[0]],  # 1 player
        )


def test_flop_missing_input_raises():
    with pytest.raises(ValueError, match="player_hands or both"):
        compute_flop_leaf_equity(
            flop_board=FLOP,
            leaf_infos=LEAF_INFOS,
            max_hands=10,
            starting_pot=200,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
