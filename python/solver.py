"""Python bindings for the C DCFR poker solver.

Uses ctypes to call the compiled C solver library.
Provides a clean Python interface for range narrowing and strategy lookup.
"""

import ctypes
import os
import sys
import subprocess
import platform

# ── Constants ────────────────────────────────────────────────────────────────

MAX_ACTIONS = 8
MAX_HANDS = 1326
MAX_BOARD = 5
SCALE = 100  # 0.01 BB precision

RANK_MAP = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
            '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
SUIT_MAP = {'c': 0, 'd': 1, 'h': 2, 's': 3}
RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"


def card_to_int(card_str):
    """Convert card string like 'Ah' to integer 0-51."""
    rank = RANK_MAP.get(card_str[0], -1)
    suit = SUIT_MAP.get(card_str[1], -1)
    if rank < 0 or suit < 0:
        raise ValueError(f"Invalid card: {card_str}")
    return rank * 4 + suit


def int_to_card(card_int):
    """Convert integer 0-51 to card string like 'Ah'."""
    return RANK_CHARS[card_int >> 2] + SUIT_CHARS[card_int & 3]


def parse_range_string(range_str):
    """Parse PioSOLVER-format range string to list of (card0, card1, weight) tuples.

    Supports:
      - Specific combos: "AhKh" -> [(Ah, Kh, 1.0)]
      - Weighted: "AhKh:0.5" -> [(Ah, Kh, 0.5)]
      - Hand groups: "AKs" -> all 4 suited AK combos
      - Comma-separated: "AA,KK,AKs"
    """
    hands = []
    if not range_str:
        return hands

    for part in range_str.split(","):
        part = part.strip()
        if not part:
            continue

        weight = 1.0
        if ":" in part:
            part, w = part.rsplit(":", 1)
            try:
                weight = float(w)
            except ValueError:
                pass

        if len(part) == 4:
            # Specific combo: AhKs
            c0 = card_to_int(part[:2])
            c1 = card_to_int(part[2:])
            if c0 > c1:
                c0, c1 = c1, c0
            hands.append((c0, c1, weight))
        elif len(part) == 3:
            # Hand group: AKs, AKo, AA
            r0 = RANK_MAP.get(part[0], -1)
            r1 = RANK_MAP.get(part[1], -1)
            if r0 < 0 or r1 < 0:
                continue
            if part[2] == 's':
                for suit in range(4):
                    c0 = r0 * 4 + suit
                    c1 = r1 * 4 + suit
                    if c0 != c1:
                        hands.append((min(c0, c1), max(c0, c1), weight))
            elif part[2] == 'o':
                for s0 in range(4):
                    for s1 in range(4):
                        if s0 != s1:
                            c0 = r0 * 4 + s0
                            c1 = r1 * 4 + s1
                            hands.append((min(c0, c1), max(c0, c1), weight))
        elif len(part) == 2:
            # Pair: AA
            r0 = RANK_MAP.get(part[0], -1)
            r1 = RANK_MAP.get(part[1], -1)
            if r0 == r1 and r0 >= 0:
                for s0 in range(4):
                    for s1 in range(s0 + 1, 4):
                        c0 = r0 * 4 + s0
                        c1 = r1 * 4 + s1
                        hands.append((c0, c1, weight))

    # Deduplicate
    seen = set()
    unique = []
    for c0, c1, w in hands:
        key = (min(c0, c1), max(c0, c1))
        if key not in seen:
            seen.add(key)
            unique.append((key[0], key[1], w))
    return unique


# ── Solver wrapper ───────────────────────────────────────────────────────────

_LIB = None
_LIB_PATH = None


def _get_lib():
    """Load or compile the C solver library."""
    global _LIB, _LIB_PATH
    if _LIB is not None:
        return _LIB

    solver_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(solver_dir, "src")
    build_dir = os.path.join(solver_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    if platform.system() == "Windows":
        lib_name = "solver.dll"
    else:
        lib_name = "solver.so"

    lib_path = os.path.join(build_dir, lib_name)

    # Check if we need to compile
    src_file = os.path.join(src_dir, "solver.c")
    need_compile = (not os.path.exists(lib_path) or
                    os.path.getmtime(src_file) > os.path.getmtime(lib_path))

    if need_compile:
        print("[solver] Compiling C solver...", file=sys.stderr)
        cmd = [
            "gcc", "-O3", "-march=native", "-ffast-math", "-shared",
            "-o", lib_path,
            src_file,
            f"-I{src_dir}",
            "-lm",
        ]
        if platform.system() != "Windows":
            cmd.append("-fPIC")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile solver: {result.stderr}")
        print("[solver] Compiled OK", file=sys.stderr)

    _LIB = ctypes.CDLL(lib_path)
    _LIB_PATH = lib_path

    # Set up function signatures
    _LIB.solver_init.restype = ctypes.c_int
    _LIB.solver_solve.restype = ctypes.c_float
    _LIB.solver_exploitability.restype = ctypes.c_float
    _LIB.solver_get_strategy.restype = ctypes.c_float

    return _LIB


class RiverSolver:
    """Solve a river spot with weighted ranges using DCFR.

    Usage:
        solver = RiverSolver(
            board=["Qs", "As", "2d", "7h", "4c"],
            oop_range="AhKh,QhQc,JhTh,6h5h",
            ip_range="AcKc,3c3d,Tc9c,8c8d",
            pot_bb=10.0,
            stack_bb=50.0,
            bet_sizes=[0.33, 0.75],
        )
        solver.solve(iterations=500)
        strategy = solver.get_strategy("oop", "AhKh")
        print(strategy)  # {'Check': 0.0, 'Bet 75%': 1.0, ...}
    """

    def __init__(self, board, oop_range, ip_range, pot_bb, stack_bb,
                 bet_sizes=None):
        self.lib = _get_lib()

        # Parse board
        self.board_cards = [card_to_int(c) for c in board]
        if len(self.board_cards) != 5:
            raise ValueError("Board must have exactly 5 cards for river solving")

        # Parse ranges
        if isinstance(oop_range, str):
            oop_hands = parse_range_string(oop_range)
        else:
            oop_hands = oop_range  # list of (c0, c1, weight)

        if isinstance(ip_range, str):
            ip_hands = parse_range_string(ip_range)
        else:
            ip_hands = ip_range

        # Store for later reference
        self.oop_hands_raw = oop_hands
        self.ip_hands_raw = ip_hands

        # Bet sizes
        if bet_sizes is None:
            bet_sizes = [0.33, 0.75]
        self.bet_sizes = bet_sizes

        # Build C arrays
        board_arr = (ctypes.c_int * 5)(*self.board_cards)

        n0 = len(oop_hands)
        hands0_type = ctypes.c_int * (n0 * 2)
        hands0_flat = []
        weights0 = (ctypes.c_float * n0)()
        for i, (c0, c1, w) in enumerate(oop_hands):
            hands0_flat.extend([c0, c1])
            weights0[i] = w
        hands0_arr = hands0_type(*hands0_flat)

        n1 = len(ip_hands)
        hands1_type = ctypes.c_int * (n1 * 2)
        hands1_flat = []
        weights1 = (ctypes.c_float * n1)()
        for i, (c0, c1, w) in enumerate(ip_hands):
            hands1_flat.extend([c0, c1])
            weights1[i] = w
        hands1_arr = hands1_type(*hands1_flat)

        bet_arr = (ctypes.c_float * len(bet_sizes))(*bet_sizes)

        pot = int(pot_bb * SCALE)
        stack = int(stack_bb * SCALE)

        # Allocate solver struct (opaque, but we know the size from the header)
        # For simplicity, allocate a large buffer
        self._solver_buf = ctypes.create_string_buffer(1024 * 1024)  # 1MB
        self._solver = ctypes.cast(self._solver_buf, ctypes.c_void_p)

        # Initialize
        err = self.lib.solver_init(
            self._solver_buf,
            board_arr, 5,
            hands0_arr, weights0, n0,
            hands1_arr, weights1, n1,
            pot, stack,
            bet_arr, len(bet_sizes),
        )
        if err != 0:
            raise RuntimeError("solver_init failed")

        self._solved = False
        self._num_hands = [n0, n1]

    def solve(self, iterations=500, target_exploitability=0.01):
        """Run DCFR for the specified iterations."""
        self.lib.solver_solve(
            self._solver_buf,
            iterations,
            ctypes.c_float(target_exploitability),
        )
        self._solved = True

    def exploitability(self):
        """Compute and return exploitability in BB."""
        exploit = self.lib.solver_exploitability(self._solver_buf)
        return exploit / SCALE

    def get_strategy(self, player, hand_str):
        """Get converged strategy for a specific hand.

        Args:
            player: "oop" or "ip" (0 or 1)
            hand_str: e.g. "AhKh"

        Returns:
            dict mapping action labels to frequencies
        """
        player_idx = 0 if player in ("oop", 0) else 1
        hands = self.oop_hands_raw if player_idx == 0 else self.ip_hands_raw

        c0 = card_to_int(hand_str[:2])
        c1 = card_to_int(hand_str[2:])
        key = (min(c0, c1), max(c0, c1))

        hand_idx = -1
        for i, (h0, h1, w) in enumerate(hands):
            if (h0, h1) == key:
                hand_idx = i
                break

        if hand_idx < 0:
            raise ValueError(f"Hand {hand_str} not found in {player} range")

        strat = (ctypes.c_float * MAX_ACTIONS)()
        self.lib.solver_get_strategy(
            self._solver_buf, player_idx, hand_idx, strat)

        # TODO: map action indices to labels based on tree structure
        result = {}
        for i in range(MAX_ACTIONS):
            if strat[i] > 0.001:
                result[f"action_{i}"] = float(strat[i])
        return result

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, '_solver_buf'):
            try:
                self.lib.solver_free(self._solver_buf)
            except Exception:
                pass


# ── Solver V2 wrapper (Pluribus-aligned) ─────────────────────────────────────

_LIB_V2 = None


def _get_lib_v2():
    """Load solver_v2 library."""
    global _LIB_V2
    if _LIB_V2 is not None:
        return _LIB_V2

    solver_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(solver_dir, "src")
    build_dir = os.path.join(solver_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    if platform.system() == "Windows":
        lib_name = "solver_v2.dll"
    else:
        lib_name = "solver_v2.so"

    lib_path = os.path.join(build_dir, lib_name)

    src_file = os.path.join(src_dir, "solver_v2.c")
    need_compile = (not os.path.exists(lib_path) or
                    os.path.getmtime(src_file) > os.path.getmtime(lib_path))

    if need_compile:
        print("[solver_v2] Compiling...", file=sys.stderr)
        cmd = [
            "gcc", "-O2", "-shared",
            "-o", lib_path,
            src_file,
            f"-I{src_dir}",
            "-lm",
        ]
        if platform.system() != "Windows":
            cmd.append("-fPIC")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to compile solver_v2: {result.stderr}")
        print("[solver_v2] Compiled OK", file=sys.stderr)

    _LIB_V2 = ctypes.CDLL(lib_path)

    _LIB_V2.sv2_init.restype = ctypes.c_int
    _LIB_V2.sv2_solve.restype = ctypes.c_float
    _LIB_V2.sv2_exploitability.restype = ctypes.c_float

    return _LIB_V2


class StreetSolver:
    """Pluribus-aligned depth-limited solver for any street.

    Supports river (showdown), turn (with leaf continuation strategies),
    and flop (with turn continuation strategies).

    Usage:
        solver = StreetSolver(
            board=["Qs", "As", "2d", "7h"],  # 4 cards = turn
            oop_range="AhKh,QhQc,JhTh,6h5h",
            ip_range="AcKc,3c3d,Tc9c,8c8d",
            pot_bb=10.0,
            stack_bb=90.0,
            bet_sizes=[0.33, 0.75],
        )
        solver.solve(iterations=500)
        strat = solver.get_strategy("oop", "AhKh")
        avg = solver.get_average_strategy("oop", "AhKh")  # for range narrowing
    """

    def __init__(self, board, oop_range, ip_range, pot_bb, stack_bb,
                 bet_sizes=None):
        self.lib = _get_lib_v2()

        self.board_cards = [card_to_int(c) for c in board]
        self.num_board = len(self.board_cards)

        if isinstance(oop_range, str):
            oop_hands = parse_range_string(oop_range)
        else:
            oop_hands = oop_range

        if isinstance(ip_range, str):
            ip_hands = parse_range_string(ip_range)
        else:
            ip_hands = ip_range

        self.oop_hands_raw = oop_hands
        self.ip_hands_raw = ip_hands

        if bet_sizes is None:
            bet_sizes = [0.33, 0.75]
        self.bet_sizes = bet_sizes

        board_arr = (ctypes.c_int * len(self.board_cards))(*self.board_cards)

        n0 = len(oop_hands)
        hands0_flat = []
        weights0 = (ctypes.c_float * n0)()
        for i, (c0, c1, w) in enumerate(oop_hands):
            hands0_flat.extend([c0, c1])
            weights0[i] = w
        hands0_arr = (ctypes.c_int * (n0 * 2))(*hands0_flat)

        n1 = len(ip_hands)
        hands1_flat = []
        weights1 = (ctypes.c_float * n1)()
        for i, (c0, c1, w) in enumerate(ip_hands):
            hands1_flat.extend([c0, c1])
            weights1[i] = w
        hands1_arr = (ctypes.c_int * (n1 * 2))(*hands1_flat)

        bet_arr = (ctypes.c_float * len(bet_sizes))(*bet_sizes)

        pot = int(pot_bb * SCALE)
        stack = int(stack_bb * SCALE)

        self._solver_buf = ctypes.create_string_buffer(4 * 1024 * 1024)
        self._num_hands = [n0, n1]

        err = self.lib.sv2_init(
            self._solver_buf,
            board_arr, self.num_board,
            hands0_arr, weights0, n0,
            hands1_arr, weights1, n1,
            pot, stack,
            bet_arr, len(bet_sizes),
        )
        if err != 0:
            raise RuntimeError("sv2_init failed")

        # Multi-street tree is built automatically in sv2_init.
        # No separate precompute step needed — chance nodes handle
        # turn/river dealing internally during CFR traversal.
        self._solved = False

    def solve(self, iterations=500, target_exploitability=0.01):
        self.lib.sv2_solve(
            self._solver_buf,
            iterations,
            ctypes.c_float(target_exploitability),
        )
        self._solved = True

    def exploitability(self):
        exploit = self.lib.sv2_exploitability(self._solver_buf)
        return exploit / SCALE

    def get_strategy(self, player, hand_str):
        """Get FINAL ITERATION strategy (for play decisions)."""
        player_idx = 0 if player in ("oop", 0) else 1
        hands = self.oop_hands_raw if player_idx == 0 else self.ip_hands_raw
        hand_idx = self._find_hand(hands, hand_str)

        strat = (ctypes.c_float * MAX_ACTIONS)()
        self.lib.sv2_get_strategy(
            self._solver_buf, player_idx, hand_idx, strat)

        result = {}
        for i in range(MAX_ACTIONS):
            if strat[i] > 0.001:
                result[f"action_{i}"] = float(strat[i])
        return result

    def get_average_strategy(self, player, hand_str):
        """Get WEIGHTED AVERAGE strategy (for Bayesian range narrowing)."""
        player_idx = 0 if player in ("oop", 0) else 1
        hands = self.oop_hands_raw if player_idx == 0 else self.ip_hands_raw
        hand_idx = self._find_hand(hands, hand_str)

        strat = (ctypes.c_float * MAX_ACTIONS)()
        self.lib.sv2_get_average_strategy(
            self._solver_buf, player_idx, hand_idx, strat)

        result = {}
        for i in range(MAX_ACTIONS):
            if strat[i] > 0.001:
                result[f"action_{i}"] = float(strat[i])
        return result

    def get_strategy_at_node(self, action_seq, player, hand_str):
        """Get strategy at a specific node in the tree."""
        player_idx = 0 if player in ("oop", 0) else 1
        hands = self.oop_hands_raw if player_idx == 0 else self.ip_hands_raw
        hand_idx = self._find_hand(hands, hand_str)

        seq_arr = (ctypes.c_int * len(action_seq))(*action_seq)
        strat = (ctypes.c_float * MAX_ACTIONS)()
        na = ctypes.c_int(0)

        self.lib.sv2_get_strategy_at_node(
            self._solver_buf, seq_arr, len(action_seq),
            player_idx, hand_idx, strat, ctypes.byref(na))

        result = {}
        for i in range(na.value):
            if strat[i] > 0.001:
                result[f"action_{i}"] = float(strat[i])
        return result

    def _find_hand(self, hands, hand_str):
        c0 = card_to_int(hand_str[:2])
        c1 = card_to_int(hand_str[2:])
        key = (min(c0, c1), max(c0, c1))
        for i, (h0, h1, w) in enumerate(hands):
            if (h0, h1) == key:
                return i
        raise ValueError(f"Hand {hand_str} not found in range")

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, '_solver_buf'):
            try:
                self.lib.sv2_free(self._solver_buf)
            except Exception:
                pass
