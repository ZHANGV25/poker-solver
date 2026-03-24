"""Python wrapper for batch outcome-sampling MCCFR on GPU.

Wraps gpu_mccfr.dll for novel GPU-parallel blueprint computation.

Usage:
    solver = GPUMCCFRSolver(
        flop=["Qs", "As", "2d"],
        player_ranges=[
            [(0, 1, 1.0), (2, 3, 1.0), ...],  # player 0's hands
            [(4, 5, 1.0), (6, 7, 1.0), ...],  # player 1's hands
        ],
        pot_bb=6.5, stack_bb=97.5,
    )
    solver.solve(iterations=100, batch_size=16384)
    strategy = solver.get_strategy(decision_idx=0, hand_idx=5)
"""

import ctypes
import os
import platform
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Constants matching gpu_mccfr.cuh
GM_MAX_PLAYERS = 6
GM_MAX_HANDS = 169
GM_MAX_ACTIONS = 6
GM_MAX_BOARD = 5
GM_MAX_RAISES = 3
GM_MAX_PATH_LEN = 64
GM_DEFAULT_BATCH_SIZE = 16384
SCALE = 100

try:
    from solver import card_to_int, int_to_card, parse_range_string
except ImportError:
    try:
        from python.solver import card_to_int, int_to_card, parse_range_string
    except ImportError:
        # Minimal card conversion if solver module not available
        _RANKS = "23456789TJQKA"
        _SUITS = "cdhs"
        def card_to_int(s):
            return _RANKS.index(s[0]) * 4 + _SUITS.index(s[1])
        def int_to_card(i):
            return _RANKS[i // 4] + _SUITS[i % 4]
        def parse_range_string(s):
            return []

# ── DLL loading ──────────────────────────────────────────────────────────

_LIB = None

def _get_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    solver_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_name = "gpu_mccfr.dll" if platform.system() == "Windows" else "gpu_mccfr.so"
    lib_path = os.path.join(solver_dir, "build", lib_name)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"gpu_mccfr not found at {lib_path}")
    _LIB = ctypes.CDLL(lib_path)

    # Set function signatures
    _LIB.gm_build_tree.restype = ctypes.c_int
    _LIB.gm_solve_gpu.restype = ctypes.c_int
    _LIB.gm_get_strategy.restype = ctypes.c_int
    _LIB.gm_free_tree.restype = None
    _LIB.gm_free_output.restype = None
    _LIB.gm_print_tree_stats.restype = None

    return _LIB

# ── ctypes structures (must match gpu_mccfr.cuh exactly) ─────────────────

class GMNode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("player", ctypes.c_int),
        ("num_children", ctypes.c_int),
        ("first_child", ctypes.c_int),
        ("street", ctypes.c_int),
        ("pot", ctypes.c_int),
        ("bets", ctypes.c_int * GM_MAX_PLAYERS),
        ("active", ctypes.c_int * GM_MAX_PLAYERS),
        ("num_active", ctypes.c_int),
        ("fold_player", ctypes.c_int),
        ("decision_idx", ctypes.c_int),
        ("num_actions", ctypes.c_int),
        ("parent", ctypes.c_int),
        ("parent_action", ctypes.c_int),
    ]

class GMTreeData(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.POINTER(GMNode)),
        ("children", ctypes.POINTER(ctypes.c_int)),
        ("num_nodes", ctypes.c_int),
        ("num_children_total", ctypes.c_int),
        ("decision_node_map", ctypes.POINTER(ctypes.c_int)),
        ("num_decision_nodes", ctypes.c_int),
        ("num_players", ctypes.c_int),
        ("hands", ((ctypes.c_int * 2) * GM_MAX_HANDS) * GM_MAX_PLAYERS),
        ("num_hands", ctypes.c_int * GM_MAX_PLAYERS),
        ("max_hands", ctypes.c_int),
        ("flop", ctypes.c_int * 3),
        ("bet_sizes", ctypes.c_float * GM_MAX_ACTIONS),
        ("num_bet_sizes", ctypes.c_int),
        ("starting_pot", ctypes.c_int),
        ("effective_stack", ctypes.c_int),
        # Card abstraction (equity buckets)
        ("use_buckets", ctypes.c_int),
        ("hand_to_bucket", ((ctypes.c_int * GM_MAX_HANDS) * GM_MAX_PLAYERS)),
        ("num_buckets", ctypes.c_int * GM_MAX_PLAYERS),
        ("max_buckets", ctypes.c_int),
    ]

class GMSolveConfig(ctypes.Structure):
    _fields_ = [
        ("max_iterations", ctypes.c_int),
        ("batch_size", ctypes.c_int),
        ("exploration_eps", ctypes.c_float),
        ("print_every", ctypes.c_int),
    ]

class GMOutput(ctypes.Structure):
    _fields_ = [
        ("avg_strategy", ctypes.POINTER(ctypes.c_float)),
        ("num_decision_nodes", ctypes.c_int),
        ("max_hands", ctypes.c_int),
        ("num_players", ctypes.c_int),
        ("use_buckets", ctypes.c_int),
        ("max_buckets", ctypes.c_int),
        ("decision_players", ctypes.POINTER(ctypes.c_int)),
        ("decision_num_actions", ctypes.POINTER(ctypes.c_int)),
        ("iterations_run", ctypes.c_int),
        ("total_trajectories", ctypes.c_int),
        ("solve_time_ms", ctypes.c_float),
    ]

# ── Card abstraction: EHS computation + percentile bucketing ──────────────

def _eval5_py(cards):
    """Minimal 5-card hand evaluator in Python. Returns a comparable strength value."""
    ranks = sorted([c // 4 for c in cards], reverse=True)
    suits = [c % 4 for c in cards]
    flush = len(set(suits)) == 1
    straight = False
    high = ranks[0]
    if ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5:
        straight = True
    if ranks == [12, 3, 2, 1, 0]:  # wheel
        straight = True
        high = 3

    from collections import Counter
    rc = Counter(ranks)
    groups = sorted(rc.items(), key=lambda x: (-x[1], -x[0]))

    if straight and flush:
        return (8, high)
    if groups[0][1] == 4:
        return (7, groups[0][0])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (6, groups[0][0], groups[1][0])
    if flush:
        return (5,) + tuple(ranks)
    if straight:
        return (4, high)
    if groups[0][1] == 3:
        return (3, groups[0][0])
    if groups[0][1] == 2 and groups[1][1] == 2:
        return (2, groups[0][0], groups[1][0], groups[2][0])
    if groups[0][1] == 2:
        kickers = sorted([r for r, c in groups if c == 1], reverse=True)
        return (1, groups[0][0]) + tuple(kickers)
    return (0,) + tuple(ranks)


def _eval7_py(cards):
    """Best 5 of 7 cards."""
    from itertools import combinations
    return max(_eval5_py(list(combo)) for combo in combinations(cards, 5))


def compute_equity_buckets(player_ranges, flop_ints, num_buckets, n_samples=100):
    """Compute Expected Hand Strength and assign percentile buckets.

    Uses Monte Carlo sampling with the Python 7-card evaluator.
    For faster computation with many hands, reduce n_samples.

    Args:
        player_ranges: list of hand lists per player [(c0,c1,w), ...]
        flop_ints: [3] flop card integers
        num_buckets: number of equity buckets per player
        n_samples: Monte Carlo samples for EHS estimation

    Returns:
        hand_to_bucket: list of lists, hand_to_bucket[p][h] = bucket index
        bucket_counts: list of lists, how many hands per bucket per player
    """
    rng = np.random.RandomState(42)
    blocked_flop = set(flop_ints)
    available = np.array([c for c in range(52) if c not in blocked_flop])

    hand_to_bucket = []
    bucket_counts = []

    for p, hands in enumerate(player_ranges):
        ehs_values = []
        for c0, c1, _w in hands:
            opp_avail = available[(available != c0) & (available != c1)]

            wins = 0
            ties = 0
            total = 0
            for _ in range(n_samples):
                if len(opp_avail) < 4:
                    break
                sample_idx = rng.choice(len(opp_avail), 4, replace=False)
                oc0, oc1 = int(opp_avail[sample_idx[0]]), int(opp_avail[sample_idx[1]])
                bc0, bc1 = int(opp_avail[sample_idx[2]]), int(opp_avail[sample_idx[3]])

                board = list(flop_ints) + [bc0, bc1]
                hero_str = _eval7_py(board + [c0, c1])
                opp_str = _eval7_py(board + [oc0, oc1])

                if hero_str > opp_str:
                    wins += 1
                elif hero_str == opp_str:
                    ties += 1
                total += 1

            ehs = (wins + 0.5 * ties) / max(total, 1)
            ehs_values.append(ehs)

        # Percentile bucketing
        n_hands = len(hands)
        sorted_indices = np.argsort(ehs_values)
        bucket_map = [0] * n_hands
        actual_buckets = min(num_buckets, n_hands)
        for rank, idx in enumerate(sorted_indices):
            bucket_map[idx] = min(rank * actual_buckets // n_hands, actual_buckets - 1)

        hand_to_bucket.append(bucket_map)

        # Count hands per bucket
        counts = [0] * actual_buckets
        for b in bucket_map:
            counts[b] += 1
        bucket_counts.append(counts)

    return hand_to_bucket, bucket_counts


# ── High-level wrapper ───────────────────────────────────────────────────

class GPUMCCFRSolver:
    """GPU-accelerated batch outcome-sampling MCCFR solver.

    This is a novel approach to blueprint computation using outcome sampling
    parallelized across thousands of GPU threads. Each thread follows one
    complete trajectory through a multi-street game tree (flop→turn→river),
    computes importance-weighted counterfactual regrets, and updates shared
    regret tables via atomic operations.
    """

    def __init__(
        self,
        flop: List[str],
        player_ranges: List[List[Tuple[int, int, float]]],
        pot_bb: float = 6.5,
        stack_bb: float = 97.5,
        bet_sizes: Optional[List[float]] = None,
        max_turn_cards: int = 0,
        max_river_cards: int = 0,
        num_buckets: int = 0,
    ):
        """
        Args:
            flop: 3 flop cards, e.g. ["Qs", "As", "2d"]
            player_ranges: list of hand lists per player.
                Each hand is (card0_int, card1_int, weight) or
                can be a list of (card0_str, card1_str) pairs.
            pot_bb: pot size in big blinds
            stack_bb: effective stack in big blinds
            bet_sizes: bet fractions (default: [0.33, 0.75, 1.5])
            max_turn_cards: max turn cards to enumerate (0=all 47)
            max_river_cards: max river cards to enumerate (0=all 46)
            num_buckets: equity buckets per player (0=exact hands, >0=bucketed)
        """
        self.lib = _get_lib()
        self.num_players = len(player_ranges)
        if bet_sizes is None:
            bet_sizes = [0.33, 0.75, 1.5]

        # Parse flop cards
        self.flop_cards = []
        for c in flop:
            if isinstance(c, str):
                self.flop_cards.append(card_to_int(c))
            else:
                self.flop_cards.append(int(c))

        # Parse player ranges
        self.player_hands = []
        self.max_hands = 0
        for p_range in player_ranges:
            hands = []
            for entry in p_range:
                if len(entry) >= 3:
                    c0, c1, w = int(entry[0]), int(entry[1]), float(entry[2])
                else:
                    c0, c1 = int(entry[0]), int(entry[1])
                    w = 1.0
                hands.append((c0, c1, w))
            self.player_hands.append(hands)
            if len(hands) > self.max_hands:
                self.max_hands = len(hands)

        # Build tree
        self.tree_data = GMTreeData()

        flop_arr = (ctypes.c_int * 3)(*self.flop_cards)
        acting_order = (ctypes.c_int * self.num_players)(*range(self.num_players))
        starting_pot = int(pot_bb * SCALE)
        effective_stack = int(stack_bb * SCALE)
        c_bet_sizes = (ctypes.c_float * len(bet_sizes))(*bet_sizes)

        ret = self.lib.gm_build_tree(
            flop_arr, self.num_players,
            acting_order,
            starting_pot, effective_stack,
            c_bet_sizes, len(bet_sizes),
            max_turn_cards, max_river_cards,
            ctypes.byref(self.tree_data)
        )
        if ret != 0:
            raise RuntimeError(f"gm_build_tree failed with code {ret}")

        # Set hands on tree_data
        for p in range(self.num_players):
            hands = self.player_hands[p]
            self.tree_data.num_hands[p] = len(hands)
            for h, (c0, c1, _w) in enumerate(hands):
                self.tree_data.hands[p][h][0] = c0
                self.tree_data.hands[p][h][1] = c1
        self.tree_data.max_hands = self.max_hands

        # Card abstraction (equity buckets)
        self.num_buckets = num_buckets
        self._hand_to_bucket = None
        if num_buckets > 0:
            print(f"[GM] Computing EHS for {sum(len(h) for h in self.player_hands)} hands...")
            htb, bcounts = compute_equity_buckets(
                self.player_hands, self.flop_cards, num_buckets
            )
            self._hand_to_bucket = htb
            self.tree_data.use_buckets = 1
            max_b = 0
            for p in range(self.num_players):
                actual_b = min(num_buckets, len(self.player_hands[p]))
                self.tree_data.num_buckets[p] = actual_b
                if actual_b > max_b:
                    max_b = actual_b
                for h in range(len(self.player_hands[p])):
                    self.tree_data.hand_to_bucket[p][h] = htb[p][h]
            self.tree_data.max_buckets = max_b
            print(f"[GM] Bucketed: {num_buckets} buckets, max_buckets={max_b}")
        else:
            self.tree_data.use_buckets = 0
            self.tree_data.max_buckets = 0

        self.output = None
        self._solved = False

    def print_tree_stats(self):
        """Print tree statistics."""
        self.lib.gm_print_tree_stats(ctypes.byref(self.tree_data))

    def solve(
        self,
        iterations: int = 100,
        batch_size: int = GM_DEFAULT_BATCH_SIZE,
        exploration_eps: float = 0.6,
        print_every: int = 10,
    ) -> dict:
        """Run batch outcome-sampling MCCFR on GPU.

        Args:
            iterations: number of outer iterations
            batch_size: trajectories per batch (per traverser per iteration)
            exploration_eps: epsilon for exploration policy
            print_every: print progress every N iterations

        Returns:
            dict with solve statistics
        """
        config = GMSolveConfig()
        config.max_iterations = iterations
        config.batch_size = batch_size
        config.exploration_eps = exploration_eps
        config.print_every = print_every

        self.output = GMOutput()

        t0 = time.time()
        ret = self.lib.gm_solve_gpu(
            ctypes.byref(self.tree_data),
            ctypes.byref(config),
            ctypes.byref(self.output)
        )
        t1 = time.time()

        if ret != 0:
            raise RuntimeError(f"gm_solve_gpu failed with code {ret}")

        self._solved = True

        return {
            "iterations": self.output.iterations_run,
            "total_trajectories": self.output.total_trajectories,
            "gpu_time_ms": self.output.solve_time_ms,
            "wall_time_s": t1 - t0,
            "num_decision_nodes": self.output.num_decision_nodes,
        }

    def get_strategy(self, decision_idx: int, slot_idx: int) -> np.ndarray:
        """Get the weighted-average strategy at a decision node.

        Args:
            decision_idx: index into the decision node array
            slot_idx: bucket index (if bucketed) or hand index (if exact)

        Returns:
            numpy array of action probabilities
        """
        if not self._solved:
            raise RuntimeError("Must call solve() first")

        strategy_out = (ctypes.c_float * GM_MAX_ACTIONS)()
        na = self.lib.gm_get_strategy(
            ctypes.byref(self.output),
            decision_idx, slot_idx,
            strategy_out
        )
        if na == 0:
            return np.array([])
        return np.array([strategy_out[a] for a in range(na)])

    def get_strategy_for_hand(self, decision_idx: int, player: int, hand_idx: int) -> np.ndarray:
        """Get strategy for a specific hand (maps through bucket if bucketed)."""
        if self._hand_to_bucket is not None:
            slot = self._hand_to_bucket[player][hand_idx]
        else:
            slot = hand_idx
        return self.get_strategy(decision_idx, slot)

    def get_all_root_strategies(self) -> List[np.ndarray]:
        """Get strategies at the root node (decision_idx=0) for all hands.

        Returns:
            list of numpy arrays, one per hand
        """
        if not self._solved:
            raise RuntimeError("Must call solve() first")

        strategies = []
        player = self.output.decision_players[0]
        nh = len(self.player_hands[player])
        for h in range(nh):
            strategies.append(self.get_strategy(0, h))
        return strategies

    def get_decision_info(self, decision_idx: int) -> dict:
        """Get info about a decision node."""
        if not self._solved:
            raise RuntimeError("Must call solve() first")
        if decision_idx < 0 or decision_idx >= self.output.num_decision_nodes:
            return {}
        return {
            "player": self.output.decision_players[decision_idx],
            "num_actions": self.output.decision_num_actions[decision_idx],
        }

    def __del__(self):
        if hasattr(self, 'tree_data') and self.tree_data.nodes:
            try:
                self.lib.gm_free_tree(ctypes.byref(self.tree_data))
            except Exception:
                pass
        if hasattr(self, 'output') and self.output is not None and self.output.avg_strategy:
            try:
                self.lib.gm_free_output(ctypes.byref(self.output))
            except Exception:
                pass


# ── Convenience: generate hand ranges ────────────────────────────────────

def make_top_n_range(n: int, exclude_cards: Optional[List[int]] = None) -> List[Tuple[int, int, float]]:
    """Generate top N hands (by card rank, no suit distinction).

    Generates pairs and suited/offsuit combos ordered roughly by strength.
    Returns list of (card0, card1, weight) tuples.
    """
    if exclude_cards is None:
        exclude_cards = []
    exclude_set = set(exclude_cards)

    hands = []
    # Generate all 1326 possible hands
    for c0 in range(52):
        if c0 in exclude_set:
            continue
        for c1 in range(c0 + 1, 52):
            if c1 in exclude_set:
                continue
            r0, r1 = c0 // 4, c1 // 4
            # Simple strength heuristic: sum of ranks + pair bonus
            strength = r0 + r1
            if r0 == r1:
                strength += 20  # pair bonus
            if c0 % 4 == c1 % 4:
                strength += 2   # suited bonus
            hands.append((c0, c1, 1.0, strength))

    # Sort by strength descending, take top N
    hands.sort(key=lambda x: -x[3])
    return [(c0, c1, w) for c0, c1, w, _ in hands[:n]]


def make_random_range(n: int, exclude_cards: Optional[List[int]] = None,
                      seed: int = 42) -> List[Tuple[int, int, float]]:
    """Generate N random hands, excluding specified cards."""
    rng = np.random.RandomState(seed)
    if exclude_cards is None:
        exclude_cards = []
    exclude_set = set(exclude_cards)

    all_cards = [c for c in range(52) if c not in exclude_set]
    hands = []
    seen = set()
    while len(hands) < n and len(all_cards) >= 2:
        idx = rng.choice(len(all_cards), 2, replace=False)
        c0, c1 = sorted([all_cards[idx[0]], all_cards[idx[1]]])
        key = (c0, c1)
        if key not in seen:
            seen.add(key)
            hands.append((c0, c1, 1.0))

    return hands


if __name__ == "__main__":
    # Quick smoke test
    print("=== GPU MCCFR Smoke Test ===")

    flop = ["Qs", "Ts", "2d"]
    flop_ints = [card_to_int(c) for c in flop]

    # 2-player, 20 hands each
    p0_range = make_top_n_range(20, exclude_cards=flop_ints)
    p1_range = make_top_n_range(20, exclude_cards=flop_ints)

    print(f"Flop: {flop}")
    print(f"Player 0: {len(p0_range)} hands")
    print(f"Player 1: {len(p1_range)} hands")

    solver = GPUMCCFRSolver(
        flop=flop,
        player_ranges=[p0_range, p1_range],
        pot_bb=6.5, stack_bb=97.5,
        max_turn_cards=5,   # limit for smoke test
        max_river_cards=5,
    )

    solver.print_tree_stats()

    stats = solver.solve(iterations=50, batch_size=4096, print_every=10)

    print(f"\n=== Results ===")
    print(f"Iterations: {stats['iterations']}")
    print(f"Trajectories: {stats['total_trajectories']:,}")
    print(f"GPU time: {stats['gpu_time_ms']:.1f} ms")
    print(f"Wall time: {stats['wall_time_s']:.2f} s")

    # Print root strategies
    print(f"\nRoot node strategies (player {solver.get_decision_info(0)['player']}):")
    for h in range(min(5, len(p0_range))):
        strat = solver.get_strategy(0, h)
        hand = p0_range[h]
        card_str = f"{int_to_card(hand[0])}{int_to_card(hand[1])}"
        print(f"  {card_str}: {strat}")
