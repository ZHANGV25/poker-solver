"""Solver pool — manage concurrent solver instances across tables.

Each table gets its own solver thread. Solves are non-blocking:
submit a request and poll for results.

Usage:
    pool = SolverPool(max_workers=8)

    # Submit a solve request
    request_id = pool.submit(
        board=["Qs", "As", "2d", "7h", "4c"],
        oop_hands=[(c0, c1, w), ...],
        ip_hands=[(c0, c1, w), ...],
        pot=1000, stack=9000,
        bet_sizes=[0.33, 0.75],
        iterations=500,
    )

    # Poll for result
    result = pool.get_result(request_id)  # None if still solving
    if result:
        print(result['strategies'])  # per-hand strategies
        print(result['exploitability'])  # exploitability in chips
"""

import ctypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Tuple

try:
    from solver import card_to_int, int_to_card, SCALE, MAX_ACTIONS
except ImportError:
    from python.solver import card_to_int, int_to_card, SCALE, MAX_ACTIONS


class SolverPool:
    """Thread pool for concurrent poker solver instances."""

    def __init__(self, max_workers=4, dll_path=None):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = {}  # request_id -> Future
        self._next_id = 0
        self._lock = threading.Lock()

        # Load DLL
        if dll_path is None:
            solver_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dll_path = os.path.join(solver_dir, "build", "solver.dll")
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"Solver DLL not found: {dll_path}")

        # Each thread needs its own DLL handle (thread-safe)
        self._dll_path = dll_path
        self._thread_local = threading.local()

    def _get_lib(self):
        """Get thread-local DLL instance."""
        if not hasattr(self._thread_local, 'lib'):
            self._thread_local.lib = ctypes.CDLL(self._dll_path)
            lib = self._thread_local.lib
            lib.solver_init.restype = ctypes.c_int
            lib.solver_solve.restype = ctypes.c_float
            lib.solver_exploitability.restype = ctypes.c_float
            lib.solver_get_strategy.restype = ctypes.c_float
        return self._thread_local.lib

    def submit(self, board, oop_hands, ip_hands, pot, stack,
               bet_sizes=None, iterations=500, target_exploit=0.01):
        """Submit a solve request (non-blocking).

        Args:
            board: list of card ints (5 for river)
            oop_hands: list of (card0, card1, weight) tuples
            ip_hands: list of (card0, card1, weight) tuples
            pot: pot size in chips (scale=100)
            stack: effective stack in chips (scale=100)
            bet_sizes: list of floats (pot fractions)
            iterations: max DCFR iterations
            target_exploit: stop if exploitability < this fraction of pot

        Returns:
            request_id (int)
        """
        with self._lock:
            request_id = self._next_id
            self._next_id += 1

        if bet_sizes is None:
            bet_sizes = [0.33, 0.75]

        future = self._executor.submit(
            self._solve_task, board, oop_hands, ip_hands,
            pot, stack, bet_sizes, iterations, target_exploit)

        with self._lock:
            self._futures[request_id] = future

        return request_id

    def get_result(self, request_id):
        """Poll for a solve result.

        Returns:
            dict with 'strategies', 'exploitability', 'time_ms' if done.
            None if still solving.
        """
        with self._lock:
            future = self._futures.get(request_id)
        if future is None:
            return None
        if not future.done():
            return None

        result = future.result()

        # Clean up
        with self._lock:
            self._futures.pop(request_id, None)

        return result

    def wait(self, request_id, timeout=None):
        """Block until a solve completes.

        Returns:
            Result dict, or None on timeout.
        """
        with self._lock:
            future = self._futures.get(request_id)
        if future is None:
            return None

        try:
            result = future.result(timeout=timeout)
            with self._lock:
                self._futures.pop(request_id, None)
            return result
        except Exception:
            return None

    def _solve_task(self, board, oop_hands, ip_hands,
                    pot, stack, bet_sizes, iterations, target_exploit):
        """Worker function that runs in a thread."""
        lib = self._get_lib()
        t0 = time.time()

        # Build C arrays
        board_arr = (ctypes.c_int * len(board))(*board)
        n0 = len(oop_hands)
        n1 = len(ip_hands)

        hands0 = (ctypes.c_int * (n0 * 2))(
            *[c for c0, c1, w in oop_hands for c in (c0, c1)])
        w0 = (ctypes.c_float * n0)(*[w for _, _, w in oop_hands])

        hands1 = (ctypes.c_int * (n1 * 2))(
            *[c for c0, c1, w in ip_hands for c in (c0, c1)])
        w1 = (ctypes.c_float * n1)(*[w for _, _, w in ip_hands])

        bet_arr = (ctypes.c_float * len(bet_sizes))(*bet_sizes)

        # Allocate solver
        buf = ctypes.create_string_buffer(4 * 1024 * 1024)

        err = lib.solver_init(buf, board_arr, len(board),
                              hands0, w0, n0, hands1, w1, n1,
                              pot, stack, bet_arr, len(bet_sizes))
        if err != 0:
            return {'error': 'solver_init failed', 'time_ms': 0}

        # Solve
        lib.solver_solve(buf, iterations, ctypes.c_float(target_exploit))

        # Get exploitability
        exploit = lib.solver_exploitability(buf)

        # Extract strategies for both players
        strategies = {0: {}, 1: {}}
        strat = (ctypes.c_float * MAX_ACTIONS)()

        for player in range(2):
            hands = oop_hands if player == 0 else ip_hands
            for h_idx, (c0, c1, w) in enumerate(hands):
                lib.solver_get_strategy(buf, player, h_idx, strat)
                hand_str = int_to_card(c0) + int_to_card(c1)
                actions = {}
                for a in range(MAX_ACTIONS):
                    if strat[a] > 0.001:
                        actions[f"action_{a}"] = float(strat[a])
                strategies[player][hand_str] = actions

        lib.solver_free(buf)
        elapsed = (time.time() - t0) * 1000

        return {
            'strategies': strategies,
            'exploitability': float(exploit) / SCALE,
            'exploit_pct': float(exploit) / pot * 100,
            'time_ms': elapsed,
            'iterations': iterations,
            'num_hands': [n0, n1],
        }

    def shutdown(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=True)
