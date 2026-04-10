"""Comprehensive tests for rollout_leaves.py Layer 3 betting simulation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from python.rollout_leaves import (
    LeafState, _simulate_rollout, _showdown_payoff,
    _streets_after, _deal_one_card, apply_bias, classify_actions,
    compute_leaf_value_via_rollout, compute_flop_leaf_values_rollout,
    _hand_to_bucket, _get_blueprint_strategy,
)
from python.solver import card_to_int
from python.leaf_values import LeafInfo


class MockBP:
    """Stub blueprint that returns None for all lookups (uniform fallback)."""
    streets_to_load = [1, 2, 3]
    _textures = {"__stub__": True}
    def get_strategy(self, *a, **kw):
        return None


bp = MockBP()
passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name} -- {detail}")


def ci(card_str):
    return card_to_int(card_str)


# ── 1. Unit tests ──────────────────────────────────────────────────────

def test_streets_after():
    check("flop -> turn,river", _streets_after("flop") == ["turn", "river"])
    check("turn -> river", _streets_after("turn") == ["river"])
    check("river -> empty", _streets_after("river") == [])


def test_classify_actions():
    c1 = classify_actions(5, first_is_fold=True)
    check("fold-first 5", c1 == ["fold", "call", "raise", "raise", "raise"], str(c1))
    c2 = classify_actions(4, first_is_fold=False)
    check("check-first 4", c2 == ["call", "raise", "raise", "raise"], str(c2))
    c3 = classify_actions(1, first_is_fold=False)
    check("single action", c3 == ["call"], str(c3))
    c4 = classify_actions(2, first_is_fold=True)
    check("fold+call only", c4 == ["fold", "call"], str(c4))


def test_apply_bias():
    strat = np.array([0.25, 0.25, 0.25, 0.25])
    cats = ["fold", "call", "raise", "raise"]

    b0 = apply_bias(strat, 0, cats)
    check("bias 0 unmodified", np.allclose(b0, strat))
    check("bias 0 sums 1", abs(b0.sum() - 1.0) < 1e-6)

    b1 = apply_bias(strat, 1, cats)
    check("fold-bias increases fold", b1[0] > strat[0])
    check("fold-bias sums 1", abs(b1.sum() - 1.0) < 1e-6)
    # fold gets 5x: 1.25 / (1.25 + 0.25 + 0.25 + 0.25) = 0.625
    check("fold-bias fold ~0.625", abs(b1[0] - 0.625) < 0.01, f"{b1[0]:.4f}")

    b2 = apply_bias(strat, 2, cats)
    check("call-bias increases call", b2[1] > strat[1])

    b3 = apply_bias(strat, 3, cats)
    check("raise-bias increases raise sum", b3[2] + b3[3] > strat[2] + strat[3])

    # Edge: zero strategy
    bz = apply_bias(np.array([0.0, 0.0, 0.0, 0.0]), 1, cats)
    check("zero strat sums 1", abs(bz.sum() - 1.0) < 1e-6)

    # Edge: no mass on target
    no_fold = np.array([0.0, 0.5, 0.25, 0.25])
    bf = apply_bias(no_fold, 1, cats)
    check("no fold mass still sums 1", abs(bf.sum() - 1.0) < 1e-6)

    # Edge: single action
    single = np.array([1.0])
    bs = apply_bias(single, 1, ["fold"])
    check("single action sums 1", abs(bs.sum() - 1.0) < 1e-6)


def test_hand_to_bucket():
    board = [ci("As"), ci("Kd"), ci("Qh")]
    b_aa = _hand_to_bucket(ci("Ah"), ci("Ad"), board)
    b_72 = _hand_to_bucket(ci("7c"), ci("2s"), board)
    check("AA bucket > 72o bucket", b_aa > b_72, f"AA={b_aa}, 72o={b_72}")
    check("buckets in [0,199]", 0 <= b_aa <= 199 and 0 <= b_72 <= 199)
    # Deterministic
    b_aa2 = _hand_to_bucket(ci("Ah"), ci("Ad"), board)
    check("bucket deterministic", b_aa == b_aa2)
    # JTs on AKQ board has a straight draw — higher EHS than overpair AA
    b_jt = _hand_to_bucket(ci("Jh"), ci("Tc"), board)
    check("JTs bucket > 72o bucket", b_jt > b_72,
          f"72o={b_72}, JTs={b_jt}")


def test_deal_one_card():
    board = [ci("Ah"), ci("Ks"), ci("Td")]
    hands = [(ci("Qh"), ci("Jh")), (ci("9s"), ci("8s"))]
    used = set(board) | {h for pair in hands for h in pair}
    for i in range(50):
        c = _deal_one_card(board, hands, np.random.default_rng(i))
        if c is None or c in used:
            check(f"deal_one_card run {i}", False, f"dealt {c}")
            return
    check("50 deals, none conflict", True)


# ── 2. Showdown payoff tests ──────────────────────────────────────────

def test_showdown_payoff():
    board5 = [ci(c) for c in ["Ks", "Qd", "8h", "5c", "3s"]]
    hero = (ci("Ah"), ci("Ad"))
    vill = (ci("7c"), ci("2s"))

    def mkleaf(pot, bets, active=None):
        n = 2
        if active is None:
            active = [True] * n
        return LeafState(board=board5, pot=pot, active=active,
                         stacks=[100 - b for b in bets], bets=bets,
                         street="river", player_hands=[[], []])

    # Hero wins
    p = _showdown_payoff(board5, 0, [hero, vill], mkleaf(100, [50, 50]))
    check("hero AA wins vs 72o", p == 50.0, str(p))

    # Villain loses
    p2 = _showdown_payoff(board5, 1, [hero, vill], mkleaf(100, [50, 50]))
    check("villain 72o loses", p2 == -50.0, str(p2))

    # Tie
    board_tie = [ci(c) for c in ["As", "Ks", "Qs", "Js", "Ts"]]
    h1 = (ci("2c"), ci("3c"))
    h2 = (ci("4c"), ci("5c"))
    lt = LeafState(board=board_tie, pot=100, active=[True, True],
                   stacks=[50, 50], bets=[50, 50], street="river",
                   player_hands=[[], []])
    pt = _showdown_payoff(board_tie, 0, [h1, h2], lt)
    check("tie splits pot: net 0", pt == 0.0, str(pt))

    # Asymmetric bets
    la = LeafState(board=board5, pot=150, active=[True, True],
                   stacks=[10, 40], bets=[90, 60], street="river",
                   player_hands=[[], []])
    pa = _showdown_payoff(board5, 0, [hero, vill], la)
    check("asym bets: hero net = pot - committed", pa == 150.0 - 90.0, str(pa))

    # Folded player
    lf = LeafState(board=board5, pot=80, active=[True, False],
                   stacks=[60, 60], bets=[40, 40], street="river",
                   player_hands=[[], []])
    pf = _showdown_payoff(board5, 0, [hero, vill], lf)
    check("folded villain: hero wins pot", pf == 80.0 - 40.0, str(pf))


# ── 3. Rollout simulation tests ──────────────────────────────────────

def test_rollout_river():
    """River leaf = pure showdown, no betting."""
    board = [ci(c) for c in ["Ks", "Qd", "8h", "5c", "3s"]]
    hero = (ci("Ah"), ci("Ad"))
    vill = (ci("7c"), ci("2s"))
    leaf = LeafState(board=board, pot=100, active=[True, True],
                     stacks=[50, 50], bets=[50, 50], street="river",
                     player_hands=[[(hero[0], hero[1], 1.0)],
                                   [(vill[0], vill[1], 1.0)]])
    payoffs = [_simulate_rollout(leaf, 0, [0, 0], bp, [hero, vill],
               np.random.default_rng(i)) for i in range(50)]
    check("river: hero always wins 50", all(p == 50.0 for p in payoffs),
          f"min={min(payoffs)}, max={max(payoffs)}")


def test_rollout_turn():
    """Turn leaf: one street of betting sim."""
    board = [ci(c) for c in ["Ks", "Qd", "8h", "5c"]]
    hero = (ci("Ah"), ci("Ad"))
    vill = (ci("7c"), ci("2s"))
    leaf = LeafState(board=board, pot=60, active=[True, True],
                     stacks=[70, 70], bets=[30, 30], street="turn",
                     player_hands=[[(hero[0], hero[1], 1.0)],
                                   [(vill[0], vill[1], 1.0)]])
    payoffs = [_simulate_rollout(leaf, 0, [0, 0], bp, [hero, vill],
               np.random.default_rng(i)) for i in range(200)]
    check("turn: all finite", all(np.isfinite(p) for p in payoffs))
    check("turn: hero AA positive EV", np.mean(payoffs) > 0,
          f"mean={np.mean(payoffs):.1f}")
    check("turn: variance exists", np.std(payoffs) > 1,
          f"std={np.std(payoffs):.1f}")


def test_rollout_flop():
    """Flop leaf: two streets of betting sim."""
    board = [ci(c) for c in ["Ks", "Qd", "8h"]]
    hero = (ci("Ah"), ci("Ad"))
    vill = (ci("7c"), ci("2s"))
    leaf = LeafState(board=board, pot=30, active=[True, True],
                     stacks=[85, 85], bets=[15, 15], street="flop",
                     player_hands=[[(hero[0], hero[1], 1.0)],
                                   [(vill[0], vill[1], 1.0)]])
    payoffs = [_simulate_rollout(leaf, 0, [0, 0], bp, [hero, vill],
               np.random.default_rng(i)) for i in range(200)]
    check("flop: all finite", all(np.isfinite(p) for p in payoffs))
    check("flop: hero AA positive EV", np.mean(payoffs) > 0,
          f"mean={np.mean(payoffs):.1f}")


def test_rollout_allin():
    """All-in: stacks=0, no decisions possible."""
    board = [ci(c) for c in ["Ks", "Qd", "8h", "5c"]]
    hero = (ci("Ah"), ci("Ad"))
    vill = (ci("7c"), ci("2s"))
    leaf = LeafState(board=board, pot=200, active=[True, True],
                     stacks=[0, 0], bets=[100, 100], street="turn",
                     player_hands=[[(hero[0], hero[1], 1.0)],
                                   [(vill[0], vill[1], 1.0)]])
    payoffs = [_simulate_rollout(leaf, 0, [0, 0], bp, [hero, vill],
               np.random.default_rng(i)) for i in range(50)]
    check("allin: all finite", all(np.isfinite(p) for p in payoffs))
    check("allin: bounded by committed", all(abs(p) <= 100.01 for p in payoffs),
          f"range=[{min(payoffs):.0f}, {max(payoffs):.0f}]")


def test_rollout_hero_perspective():
    """Hero=1 (IP) perspective should also work."""
    board = [ci(c) for c in ["Ks", "Qd", "8h"]]
    hero = (ci("Ah"), ci("Ad"))
    vill = (ci("7c"), ci("2s"))
    leaf = LeafState(board=board, pot=30, active=[True, True],
                     stacks=[85, 85], bets=[15, 15], street="flop",
                     player_hands=[[(vill[0], vill[1], 1.0)],
                                   [(hero[0], hero[1], 1.0)]])
    # Hero is player 1 now
    payoffs = [_simulate_rollout(leaf, 1, [0, 0], bp, [vill, hero],
               np.random.default_rng(i)) for i in range(100)]
    check("hero=IP: all finite", all(np.isfinite(p) for p in payoffs))
    check("hero=IP: AA positive EV", np.mean(payoffs) > 0,
          f"mean={np.mean(payoffs):.1f}")


# ── 4. Bias profile impact tests ─────────────────────────────────────

def test_bias_profiles():
    """Different bias combos should produce different leaf values."""
    board = [ci(c) for c in ["Ts", "7d", "3h"]]
    hero = (ci("Jh"), ci("Tc"))  # pair of tens
    vill = (ci("8s"), ci("9s"))  # OESD
    leaf = LeafState(board=board, pot=40, active=[True, True],
                     stacks=[80, 80], bets=[20, 20], street="flop",
                     player_hands=[[(hero[0], hero[1], 1.0)],
                                   [(vill[0], vill[1], 1.0)]])
    N = 200
    bias_means = {}
    for s_hero in range(4):
        for s_vill in range(4):
            payoffs = [_simulate_rollout(leaf, 0, [s_hero, s_vill], bp,
                       [hero, vill], np.random.default_rng(i))
                       for i in range(N)]
            bias_means[(s_hero, s_vill)] = np.mean(payoffs)

    all_means = list(bias_means.values())
    check("16 bias combos computed", len(all_means) == 16)
    unique = len(set(round(m, 0) for m in all_means))
    check("bias combos produce >=4 distinct values", unique >= 4,
          f"got {unique} unique")

    # Fold-biased villain != raise-biased villain
    diff = abs(bias_means[(0, 1)] - bias_means[(0, 3)])
    check("fold-vill != raise-vill", diff > 0.1, f"diff={diff:.1f}")


# ── 5. compute_leaf_value_via_rollout ─────────────────────────────────

def test_compute_leaf_value_api():
    board = [ci(c) for c in ["Ts", "7d", "3h"]]
    hero = (ci("Jh"), ci("Tc"))
    vill = (ci("8s"), ci("9s"))
    leaf = LeafState(board=board, pot=40, active=[True, True],
                     stacks=[80, 80], bets=[20, 20], street="flop",
                     player_hands=[[(hero[0], hero[1], 1.0)],
                                   [(vill[0], vill[1], 1.0)]])

    result = compute_leaf_value_via_rollout(leaf, [0, 0], bp, num_rollouts=5)
    check("returns ndarray", isinstance(result, np.ndarray))
    check("shape[0]=2 players", result.shape[0] == 2)
    check("hero value nonzero", result[0, 0] != 0.0, str(result[0, 0]))

    # All 4 bias profiles
    vals = []
    for b in range(4):
        r = compute_leaf_value_via_rollout(leaf, [b, 0], bp, num_rollouts=10)
        check(f"bias {b} returns value", r is not None)
        if r is not None:
            vals.append(r[0, 0])
    check("4 profiles all produce values", len(vals) == 4)


# ── 6. compute_flop_leaf_values_rollout array builder ─────────────────

def test_flop_leaf_values_array():
    board = [ci(c) for c in ["Ts", "7d", "3h"]]
    hero = (ci("Jh"), ci("Tc"))
    vill = (ci("8s"), ci("9s"))

    leaf_infos = [LeafInfo(leaf_idx=0, pot=40, bets=(20, 20))]
    result = compute_flop_leaf_values_rollout(
        flop_board=board,
        player_hands=[[(hero[0], hero[1], 1.0)],
                      [(vill[0], vill[1], 1.0)]],
        blueprint_v2=bp,
        leaf_infos=leaf_infos,
        max_hands=1,
        starting_pot=40,
        num_rollouts=3,
    )
    check("array builder returns ndarray", result is not None)
    if result is not None:
        check(f"shape = (16, 2, 1)", result.shape == (16, 2, 1),
              str(result.shape))
        check("not all zeros", np.any(result != 0))
        hero_vals = result[:, 0, 0]
        unique = len(set(round(float(v), 1) for v in hero_vals))
        check(f"hero vals: {unique} distinct of 16", unique >= 3)

    # Multiple leaves
    leaf_infos_2 = [
        LeafInfo(leaf_idx=0, pot=40, bets=(20, 20)),
        LeafInfo(leaf_idx=1, pot=80, bets=(40, 40)),
    ]
    result_2 = compute_flop_leaf_values_rollout(
        flop_board=board,
        player_hands=[[(hero[0], hero[1], 1.0)],
                      [(vill[0], vill[1], 1.0)]],
        blueprint_v2=bp,
        leaf_infos=leaf_infos_2,
        max_hands=1,
        starting_pot=40,
        num_rollouts=3,
    )
    if result_2 is not None:
        check(f"2 leaves: shape = (32, 2, 1)", result_2.shape == (32, 2, 1),
              str(result_2.shape))
        # Different pots should give different values
        pot40_mean = np.mean(result_2[:16, 0, 0])
        pot80_mean = np.mean(result_2[16:, 0, 0])
        check("different pots give different values",
              abs(pot40_mean - pot80_mean) > 0.01,
              f"pot40={pot40_mean:.1f} pot80={pot80_mean:.1f}")


# ── 7. None-blueprint fallback ────────────────────────────────────────

def test_none_blueprint():
    board = [ci(c) for c in ["Ts", "7d", "3h"]]
    hero = (ci("Jh"), ci("Tc"))
    vill = (ci("8s"), ci("9s"))
    leaf = LeafState(board=board, pot=40, active=[True, True],
                     stacks=[80, 80], bets=[20, 20], street="flop",
                     player_hands=[[(hero[0], hero[1], 1.0)],
                                   [(vill[0], vill[1], 1.0)]])

    # None blueprint should return None
    r = compute_leaf_value_via_rollout(leaf, [0, 0], None, num_rollouts=3)
    check("None blueprint returns None", r is None)

    # Blueprint without postflop should return None
    class NoBP:
        streets_to_load = [0]  # preflop only
        _textures = {}
        def get_strategy(self, *a, **kw):
            return None
    r2 = compute_leaf_value_via_rollout(leaf, [0, 0], NoBP(), num_rollouts=3)
    check("no-postflop blueprint returns None", r2 is None)


# ── Run all tests ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 1. Unit Tests ===")
    test_streets_after()
    test_classify_actions()
    test_apply_bias()
    test_hand_to_bucket()
    test_deal_one_card()

    print("\n=== 2. Showdown Payoff ===")
    test_showdown_payoff()

    print("\n=== 3. Rollout Simulation ===")
    test_rollout_river()
    test_rollout_turn()
    test_rollout_flop()
    test_rollout_allin()
    test_rollout_hero_perspective()

    print("\n=== 4. Bias Profiles ===")
    test_bias_profiles()

    print("\n=== 5. compute_leaf_value_via_rollout API ===")
    test_compute_leaf_value_api()

    print("\n=== 6. Array Builder ===")
    test_flop_leaf_values_array()

    print("\n=== 7. None Blueprint Fallback ===")
    test_none_blueprint()

    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("ALL TESTS PASSED")
