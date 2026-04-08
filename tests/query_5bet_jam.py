#!/usr/bin/env python3
"""
Query the live solver to determine how often the 5-bet jam line is used.
This tells us whether dropping PREFLOP_MAX_RAISES from 4 → 3 (which removes
the 5-bet jam tier) would actually lose any strategy.

Method: walks specific preflop action sequences via bp_get_strategy and
prints the probabilities of (fold/call/raise) at each decision point along
4-bet → 5-bet lines for premium hands.

Run on the live solver instance:
    python3 query_5bet_jam.py
"""
import ctypes
import os
import sys

# Find the solver shared library
DLL = None
for path in [
    '/opt/poker-solver/build/mccfr_blueprint.so',
    '/home/ec2-user/poker-solver/build/mccfr_blueprint.so',
    './build/mccfr_blueprint.so',
    './mccfr_blueprint.so',
]:
    if os.path.exists(path):
        DLL = path
        break

if not DLL:
    print("ERROR: cannot find mccfr_blueprint.so")
    sys.exit(1)

print(f"Loading {DLL}")
bp = ctypes.CDLL(DLL)

# Bindings (subset of blueprint_worker_unified.py)
bp.bp_get_strategy.restype = ctypes.c_int
bp.bp_get_strategy.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
]
bp.bp_get_regrets.restype = ctypes.c_int
bp.bp_get_regrets.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
]


def attach_to_running_solver():
    """We can't attach to a running solver — it owns its memory.
    Instead, load from the latest checkpoint.
    """
    raise NotImplementedError("Run via launch_query_5bet.sh which loads checkpoint")


# Action indices for the deployed tiered preflop config:
#   tier 0 (open):  [F=0, C=1, raise_0.5=2, raise_0.7=3, raise_1.0=4]
#   tier 1 (3-bet): [F=0, C=1, raise_0.7=2, raise_1.0=3]
#   tier 2 (4-bet): [F=0, C=1, raise_1.0=2]
#   tier 3 (5-bet): [F=0, C=1, raise_8.0_=allin=2]

# Hand bucket indices (canonical AA=0 KK=25 ... per init_unified())
HAND_BUCKETS = {
    "AA": 0, "KK": 25, "QQ": 48, "JJ": 69, "TT": 88,
    "AKs": 1, "AKo": 2, "AQs": 3, "AQo": 4,
    "99": 105, "88": 120, "77": 133, "66": 144, "55": 153,
}

# Build action sequences leading to 4-bet and 5-bet decision points
# Preflop order: UTG(2), MP(3), CO(4), BTN(5), SB(0), BB(1)

LINES = [
    # UTG opens 1.0, all fold to BB, BB 3-bets 1.0, UTG 4-bets 1.0, BB 5-bet decision
    {
        'name': 'BB facing UTG 4-bet (5-bet decision)',
        'player': 1,  # BB
        'sequence': [4, 0, 0, 0, 0, 3, 2],  # UTG_R1.0, MP_F, CO_F, BTN_F, SB_F, BB_R1.0, UTG_R1.0
    },
    # UTG opens 1.0, all fold to BB, BB 3-bets 1.0, UTG 4-bet decision
    {
        'name': 'UTG facing BB 3-bet (4-bet decision)',
        'player': 2,  # UTG
        'sequence': [4, 0, 0, 0, 0, 3],  # UTG_R1.0, ..., BB_R1.0
    },
    # UTG opens, fold to BTN, BTN 3-bets, UTG 4-bet decision
    {
        'name': 'UTG facing BTN 3-bet (4-bet decision)',
        'player': 2,
        'sequence': [4, 0, 0, 3],  # UTG_R1.0, MP_F, CO_F, BTN_R1.0
    },
    # UTG opens, fold to BTN, BTN 3-bets, UTG 4-bets, BTN 5-bet decision
    {
        'name': 'BTN facing UTG 4-bet (5-bet decision)',
        'player': 5,
        'sequence': [4, 0, 0, 3, 2],  # UTG_R1.0, MP_F, CO_F, BTN_R1.0, UTG_R1.0
    },
]


def query_strategy(solver_handle, line, hand, bucket):
    """Query strategy for `hand` at the decision point of `line`."""
    seq = line['sequence']
    c_seq = (ctypes.c_int * len(seq))(*seq)
    empty_board = (ctypes.c_int * 1)(0)
    strat_buf = (ctypes.c_float * 16)()

    na = bp.bp_get_strategy(solver_handle, line['player'], empty_board, 0,
                             c_seq, len(seq), strat_buf, bucket)
    if na <= 0:
        return None
    strat = [strat_buf[i] for i in range(na)]
    return na, strat


def main():
    # This script needs to be run AFTER loading a checkpoint into a fresh
    # solver instance. The launch script handles that.
    print("This script must be invoked from launch_query_5bet.sh")
    print("It needs a solver handle from bp_init_unified + bp_load_regrets.")
    sys.exit(1)


if __name__ == '__main__':
    main()
