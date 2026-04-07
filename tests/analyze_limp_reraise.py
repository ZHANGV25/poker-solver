#!/usr/bin/env python3
"""Diagnose limp-reraise: what does UTG do after limping when someone raises behind?

Queries the info set for "UTG called, MP raised size X, everyone else folded, UTG acts again"
for AA, KK, and other key hands.
"""
import ctypes
import sys
import os

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "/opt/analysis/regrets.bin"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else None

for d in ["build", "/opt/poker-solver/build", "."]:
    p = os.path.join(d, "mccfr_blueprint.so")
    if os.path.exists(p):
        DLL_PATH = p
        break
else:
    print("ERROR: mccfr_blueprint.so not found")
    sys.exit(1)

bp = ctypes.CDLL(DLL_PATH)
bp.bp_default_config.restype = None
bp.bp_init_unified.restype = ctypes.c_int
bp.bp_set_preflop_tier.restype = ctypes.c_int
bp.bp_set_preflop_tier.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
]
bp.bp_load_regrets.restype = ctypes.c_int64
bp.bp_load_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
bp.bp_get_strategy.restype = ctypes.c_int
bp.bp_get_regrets.restype = ctypes.c_int


class BPConfig(ctypes.Structure):
    _fields_ = [
        ("discount_stop_iter", ctypes.c_int64),
        ("discount_interval", ctypes.c_int64),
        ("prune_start_iter", ctypes.c_int64),
        ("snapshot_start_iter", ctypes.c_int64),
        ("snapshot_interval", ctypes.c_int64),
        ("strategy_interval", ctypes.c_int64),
        ("num_threads", ctypes.c_int),
        ("hash_table_size", ctypes.c_int64),
        ("snapshot_dir", ctypes.c_char_p),
        ("include_preflop", ctypes.c_int),
        ("postflop_num_buckets", ctypes.c_int),
    ]


config = BPConfig()
bp.bp_default_config(ctypes.byref(config))
config.num_threads = 1
config.hash_table_size = 2000000000
config.include_preflop = 1

buf = (ctypes.c_char * 524288)()
solver = ctypes.cast(buf, ctypes.c_void_p)
postflop = (ctypes.c_float * 2)(0.5, 1.0)
preflop = (ctypes.c_float * 8)(0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0)

print("Initializing solver...", flush=True)
ret = bp.bp_init_unified(
    solver, 6, 50, 100, 10000, postflop, 2, preflop, 8, ctypes.byref(config)
)
if ret != 0:
    print("init failed: %d" % ret)
    sys.exit(1)

for level, sizes in [
    (0, [0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0]),
    (1, [0.7, 1.0, 2.5]),
    (2, [1.0, 4.0]),
    (3, [8.0]),
]:
    c = (ctypes.c_float * len(sizes))(*sizes)
    bp.bp_set_preflop_tier(solver, level, c, len(sizes), 4)

print("Loading checkpoint: %s" % CHECKPOINT, flush=True)
n = bp.bp_load_regrets(solver, CHECKPOINT.encode())
print("Loaded %d info sets" % n, flush=True)

out = open(OUTPUT, "w") if OUTPUT else sys.stdout

hands = [
    ("AA", 0), ("KK", 25), ("QQ", 48), ("JJ", 69), ("TT", 88),
    ("99", 105), ("AKs", 1), ("AKo", 2),
]

empty_board = (ctypes.c_int * 1)(0)
regret_buf = (ctypes.c_int * 16)()
strat_buf = (ctypes.c_float * 16)()

# Open raise sizes at level 0 (MP's raise after UTG limps)
# fold=0, call=1, r0.4x=2, r0.5x=3, r0.7x=4, r1.0x=5, r1.5x=6, r2.5x=7, r4.0x=8, r8.0x=9
OPEN_SIZES = {2: "r0.4x", 3: "r0.5x", 4: "r0.7x", 5: "r1.0x", 6: "r1.5x", 7: "r2.5x", 8: "r4.0x", 9: "r8.0x"}

# 3-bet sizes at level 1 (UTG's reraise options)
# fold=0, call=1, r0.7x=2, r1.0x=3, r2.5x=4
THREEB_ACTIONS = ["fold", "call", "3b_r0.7x", "3b_r1.0x", "3b_r2.5x"]

out.write("=== LIMP-RERAISE DIAGNOSTIC ===\n")
out.write("Scenario: UTG limps (call), MP raises size X, CO/BTN/SB/BB fold, UTG acts again\n")
out.write("UTG's options: fold, call the raise, or 3-bet (r0.7x, r1.0x, r2.5x)\n\n")

# Test each MP raise size
for mp_raise_action, mp_raise_name in sorted(OPEN_SIZES.items()):
    # Action sequence: UTG calls(1), MP raises(mp_raise_action), CO folds(0), BTN folds(0), SB folds(0), BB folds(0)
    seq = [1, mp_raise_action, 0, 0, 0, 0]
    c_seq = (ctypes.c_int * len(seq))(*seq)

    out.write("--- MP raises %s, folds to UTG ---\n" % mp_raise_name)
    found_any = False
    for name, bucket in hands:
        na = bp.bp_get_regrets(solver, 2, empty_board, 0, c_seq, len(seq), regret_buf, bucket)
        na2 = bp.bp_get_strategy(solver, 2, empty_board, 0, c_seq, len(seq), strat_buf, bucket)
        if na >= 2:
            found_any = True
            parts = []
            for a in range(na):
                label = THREEB_ACTIONS[a] if a < len(THREEB_ACTIONS) else ("a%d" % a)
                r = regret_buf[a]
                s = strat_buf[a] * 100
                parts.append("%s=%+d(%.0f%%)" % (label, r, s))
            out.write("  %s: %s\n" % (name, " ".join(parts)))
        else:
            out.write("  %s: not found (na=%d)\n" % (name, na))

    if not found_any:
        out.write("  (no info sets found for this sequence)\n")
    out.write("\n")
    out.flush()

# Also check: UTG limps, everyone limps/folds to BB, BB raises
out.write("\n=== BB RAISES AFTER UTG LIMP ===\n")
out.write("Scenario: UTG limps(1), MP folds(0), CO folds(0), BTN folds(0), SB folds(0), BB raises\n")
out.write("This is BB raising over a single limper\n\n")

for bb_raise_action, bb_raise_name in sorted(OPEN_SIZES.items()):
    # UTG calls(1), MP folds(0), CO folds(0), BTN folds(0), SB folds(0), BB raises(bb_raise_action)
    seq = [1, 0, 0, 0, 0, bb_raise_action]
    c_seq = (ctypes.c_int * len(seq))(*seq)

    out.write("--- BB raises %s, UTG acts ---\n" % bb_raise_name)
    found_any = False
    for name, bucket in hands:
        na = bp.bp_get_regrets(solver, 2, empty_board, 0, c_seq, len(seq), regret_buf, bucket)
        na2 = bp.bp_get_strategy(solver, 2, empty_board, 0, c_seq, len(seq), strat_buf, bucket)
        if na >= 2:
            found_any = True
            parts = []
            for a in range(na):
                label = THREEB_ACTIONS[a] if a < len(THREEB_ACTIONS) else ("a%d" % a)
                r = regret_buf[a]
                s = strat_buf[a] * 100
                parts.append("%s=%+d(%.0f%%)" % (label, r, s))
            out.write("  %s: %s\n" % (name, " ".join(parts)))
        else:
            out.write("  %s: not found (na=%d)\n" % (name, na))

    if not found_any:
        out.write("  (no info sets found for this sequence)\n")
    out.write("\n")
    out.flush()

# Also check: UTG limps, MP limps, CO folds, BTN folds, SB folds, BB checks
# Then: UTG limps, everyone folds/calls to BB, BB checks (limp pot goes to flop)
out.write("\n=== LIMP-THROUGH FREQUENCY ===\n")
out.write("How often does it get to a flop unraised after UTG limps?\n")
out.write("Check: MP's action when UTG limps\n\n")

# MP's decision after UTG limps: action_seq = [1], player = 3 (MP)
mp_seq = (ctypes.c_int * 1)(1)
for name, bucket in hands:
    na = bp.bp_get_strategy(solver, 3, empty_board, 0, mp_seq, 1, strat_buf, bucket)
    if na >= 2:
        parts = []
        action_labels = ["fold", "call"] + [OPEN_SIZES.get(i, "r?") for i in range(2, na)]
        for a in range(na):
            s = strat_buf[a] * 100
            label = action_labels[a] if a < len(action_labels) else ("a%d" % a)
            if s >= 0.5:
                parts.append("%s=%.0f%%" % (label, s))
        out.write("  MP w/%s after UTG limp: %s\n" % (name, " ".join(parts)))
    else:
        out.write("  MP w/%s: not found\n" % name)

out.write("\n")

if OUTPUT:
    out.close()
    print("Results written to %s" % OUTPUT)

bp.bp_free(solver)
print("Done.")
