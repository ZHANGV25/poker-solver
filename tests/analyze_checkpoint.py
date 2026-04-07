#!/usr/bin/env python3
"""Analyze a checkpoint: per-raise regrets for all key hands across positions.

Usage: python3 analyze_checkpoint.py /path/to/regrets.bin [output.txt]
"""
import ctypes
import sys
import os

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "/opt/analysis/regrets_200M.bin"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else None

# Find DLL
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
preflop = (ctypes.c_float * 3)(0.5, 0.7, 1.0)

print("Initializing solver...", flush=True)
ret = bp.bp_init_unified(
    solver, 6, 50, 100, 10000, postflop, 2, preflop, 3, ctypes.byref(config)
)
if ret != 0:
    print("init failed: %d" % ret)
    sys.exit(1)

for level, sizes in [
    (0, [0.5, 0.7, 1.0]),
    (1, [0.7, 1.0]),
    (2, [1.0]),
    (3, [8.0]),
]:
    c = (ctypes.c_float * len(sizes))(*sizes)
    bp.bp_set_preflop_tier(solver, level, c, len(sizes), 4)

print("Loading checkpoint: %s" % CHECKPOINT, flush=True)
n = bp.bp_load_regrets(solver, CHECKPOINT.encode())
print("Loaded %d info sets" % n, flush=True)

# ── Output ──
out = open(OUTPUT, "w") if OUTPUT else sys.stdout

SIZES = [
    "fold", "call",
    "r0.5x", "r0.7x", "r1.0x",
]

# BB equivalents for context
BB_EQUIV = {
    "r0.5x": "1.75BB", "r0.7x": "2.05BB", "r1.0x": "2.5BB",
}

hands = [
    ("AA", 0), ("KK", 25), ("QQ", 48), ("JJ", 69), ("TT", 88),
    ("99", 105), ("88", 120), ("77", 133), ("66", 144), ("55", 153),
    ("44", 160), ("33", 165), ("22", 168),
    ("AKs", 1), ("AKo", 2), ("AQs", 3), ("AQo", 4),
    ("AJs", 5), ("AJo", 6), ("ATs", 7), ("ATo", 8),
    ("A5s", 17), ("A4s", 19), ("A3s", 21), ("A2s", 23),
    ("KQs", 26), ("KQo", 27), ("KJs", 28), ("KTs", 30), ("K9s", 32),
    ("QJs", 49), ("QTs", 51), ("JTs", 70),
    ("T9s", 89), ("98s", 106), ("87s", 121), ("76s", 134),
    ("65s", 145), ("54s", 154),
]

empty_board = (ctypes.c_int * 1)(0)
utg_seq = (ctypes.c_int * 1)(0)
co_seq = (ctypes.c_int * 2)(0, 0)
btn_seq = (ctypes.c_int * 3)(0, 0, 0)
sb_seq = (ctypes.c_int * 4)(0, 0, 0, 0)

regret_buf = (ctypes.c_int * 16)()
strat_buf = (ctypes.c_float * 16)()

positions = [
    ("UTG", 2, utg_seq, 0),
    ("CO", 4, co_seq, 2),
    ("BTN", 5, btn_seq, 3),
    ("SB", 0, sb_seq, 4),
]

for pos_name, player, seq, seq_len in positions:
    out.write("\n=== %s (player %d) ===\n" % (pos_name, player))
    out.write("%-5s  %-12s %-12s " % ("Hand", "fold", "call"))
    for i in range(2, len(SIZES)):
        out.write("%-14s " % ("%s(%s)" % (SIZES[i], BB_EQUIV.get(SIZES[i], ""))))
    out.write("\n")
    out.write("-" * 140 + "\n")

    for name, bucket in hands:
        na = bp.bp_get_regrets(
            solver, player, empty_board, 0, seq, seq_len, regret_buf, bucket
        )
        na2 = bp.bp_get_strategy(
            solver, player, empty_board, 0, seq, seq_len, strat_buf, bucket
        )
        if na >= 3:
            out.write("%-5s  " % name)
            for a in range(na):
                r = regret_buf[a]
                s = strat_buf[a] * 100
                if s >= 1.0:
                    out.write("%+10d(%2.0f%%) " % (r, s))
                else:
                    out.write("%+10d( 0%%) " % r)
            out.write("\n")
        else:
            out.write("%-5s  not found\n" % name)

    out.write("\n")
    out.flush()

# Summary: which raise sizes have positive regret across premium hands
out.write("\n=== RAISE SIZE SUMMARY (UTG, positive regret count across top 13 pairs) ===\n")
utg_player = 2
for i in range(2, len(SIZES)):
    label = SIZES[i]
    bb = BB_EQUIV.get(label, "")
    pos_count = 0
    total_regret = 0
    for name, bucket in hands[:13]:  # 13 pocket pairs
        na = bp.bp_get_regrets(
            solver, utg_player, empty_board, 0, utg_seq, 0, regret_buf, bucket
        )
        if na > i:
            r = regret_buf[i]
            if r > 0:
                pos_count += 1
            total_regret += r
    out.write(
        "  %s (%s): %d/13 pairs positive, total regret = %+d\n"
        % (label, bb, pos_count, total_regret)
    )

out.write("\n")

if OUTPUT:
    out.close()
    print("Results written to %s" % OUTPUT)

bp.bp_free(solver)
print("Done.")
