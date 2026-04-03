#!/usr/bin/env python3
"""Generate a small BPR3 checkpoint file for local testing.

Creates a ~1 MB file with known entries across all streets,
including preflop roots for key hands (AA, KK, 72o, etc.).
"""

import struct
import random
import argparse

# Root action hash = 0xFEDCBA9876543210
ROOT_ACTION_HASH = 0xFEDCBA9876543210

# Preflop bucket indices for key hands
PAIR_BUCKETS = [0, 25, 48, 69, 88, 105, 120, 133, 144, 153, 160, 165, 168]
# 0=AA, 25=KK, 48=QQ, ..., 165=33, 168=22
# 167=72o (from the label ordering)

POS_NAMES = ["SB", "BB", "UTG", "MP", "CO", "BTN"]


def write_entry(f, player, street, bucket, board_hash, action_hash,
                regrets, strategy_sum=None):
    """Write a single info set entry in BPR3 format."""
    na = len(regrets)
    f.write(struct.pack('<i', player))
    f.write(struct.pack('<i', street))
    f.write(struct.pack('<i', bucket))
    f.write(struct.pack('<Q', board_hash))
    f.write(struct.pack('<Q', action_hash))
    f.write(struct.pack('<i', na))
    for r in regrets:
        f.write(struct.pack('<i', r))
    has_ss = 1 if strategy_sum else 0
    f.write(struct.pack('<i', has_ss))
    if strategy_sum:
        for s in strategy_sum:
            f.write(struct.pack('<f', s))


def generate_preflop_regrets(bucket, position):
    """Generate realistic-ish preflop regrets for a given hand class."""
    # AA (bucket 0): never fold, mostly raise
    if bucket == 0:
        return [-1000000, 50000, 800000, 2000000]  # fold, call, r0.5x, r1x
    # KK (bucket 25): similar to AA
    elif bucket == 25:
        return [-800000, 100000, 600000, 1500000]
    # QQ (bucket 48)
    elif bucket == 48:
        return [-500000, 200000, 800000, 1000000]
    # 72o (bucket 167): fold a lot, especially from early position
    elif bucket == 167:
        if position <= 3:  # UTG, MP, CO, SB
            return [3000000, -500000, -800000, -1000000]
        else:  # BTN, BB
            return [1000000, 200000, -200000, -500000]
    # 32o (bucket 168)
    elif bucket == 168:
        if position <= 3:
            return [3500000, -600000, -900000, -1100000]
        else:
            return [1200000, 100000, -300000, -600000]
    # Pocket pairs on BTN: moderate, don't fold much
    elif bucket in PAIR_BUCKETS:
        if position == 5:  # BTN
            return [-200000, 500000, 400000, 300000]
        else:
            return [100000, 400000, 300000, 200000]
    # Weak hands (bucket > 140) from UTG: fold a lot, don't raise much
    elif bucket > 140:
        if position == 2:  # UTG
            return [2000000, -300000, -500000, -700000]
        else:
            return [500000, 200000, 100000, -100000]
    # Medium hands
    else:
        return [200000, 300000, 400000, 100000]


def generate_checkpoint(path, num_iters=5000000000, num_extra_entries=5000):
    """Generate a test BPR3 checkpoint file."""
    entries = []

    # 1. Preflop root entries for all positions and key buckets
    key_buckets = [0, 1, 2, 3, 4, 5, 10, 25, 48, 69, 84, 88, 105,
                   120, 133, 144, 150, 153, 160, 165, 166, 167, 168]
    for pos in range(6):
        for bkt in key_buckets:
            regrets = generate_preflop_regrets(bkt, pos)
            entries.append((pos, 0, bkt, 0, ROOT_ACTION_HASH, regrets, None))

    # 2. Preflop non-root entries (action sequences after first action)
    random.seed(42)
    for _ in range(500):
        pos = random.randint(0, 5)
        bkt = random.randint(0, 168)
        ah = random.randint(1, 0xFFFFFFFFFFFFFFFF)
        na = random.randint(2, 5)
        regrets = [random.randint(-500000, 500000) for _ in range(na)]
        entries.append((pos, 0, bkt, 0, ah, regrets, None))

    # 3. Flop entries (street=1)
    for _ in range(num_extra_entries):
        pos = random.randint(0, 5)
        bkt = random.randint(0, 199)
        bh = random.randint(1, 0xFFFFFFFFFFFFFFFF)
        ah = random.randint(1, 0xFFFFFFFFFFFFFFFF)
        na = random.randint(3, 6)
        # Flop: more checking than betting
        regrets = [random.randint(-300000, 300000) for _ in range(na)]
        # Make check/call action slightly positive more often
        regrets[1] = abs(regrets[1]) + 50000
        entries.append((pos, 1, bkt, bh, ah, regrets, None))

    # 4. Turn entries (street=2)
    for _ in range(num_extra_entries):
        pos = random.randint(0, 5)
        bkt = random.randint(0, 199)
        bh = random.randint(1, 0xFFFFFFFFFFFFFFFF)
        ah = random.randint(1, 0xFFFFFFFFFFFFFFFF)
        na = random.randint(3, 5)
        regrets = [random.randint(-300000, 300000) for _ in range(na)]
        entries.append((pos, 2, bkt, bh, ah, regrets, None))

    # 5. River entries (street=3)
    for _ in range(num_extra_entries):
        pos = random.randint(0, 5)
        bkt = random.randint(0, 199)
        bh = random.randint(1, 0xFFFFFFFFFFFFFFFF)
        ah = random.randint(1, 0xFFFFFFFFFFFFFFFF)
        na = random.randint(2, 5)
        regrets = [random.randint(-300000, 300000) for _ in range(na)]
        # Near-nuts (bucket > 190) should not fold
        if bkt > 190:
            regrets[0] = -abs(regrets[0]) - 100000  # fold regret very negative
        entries.append((pos, 3, bkt, bh, ah, regrets, None))

    # 6. Some entries with strategy_sum (preflop only, as per Pluribus)
    for i in range(min(50, len(entries))):
        if entries[i][1] == 0:  # preflop
            na = len(entries[i][5])
            ss = [random.uniform(0, 1) for _ in range(na)]
            total = sum(ss)
            ss = [s / total for s in ss]
            entries[i] = entries[i][:6] + (ss,)

    # Write file
    num_entries = len(entries)
    table_size = 1 << 20  # 1M slots (doesn't matter for checker)

    with open(path, 'wb') as f:
        # Header: BPR3 magic + table_size + num_entries + iterations (int64)
        f.write(b'BPR3')
        f.write(struct.pack('<i', table_size))
        f.write(struct.pack('<i', num_entries))
        f.write(struct.pack('<q', num_iters))

        for entry in entries:
            player, street, bucket, bh, ah, regrets, ss = entry
            write_entry(f, player, street, bucket, bh, ah, regrets, ss)

    print(f"Generated {path}: {num_entries} entries, {num_iters} iterations")
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate test BPR3 checkpoint')
    parser.add_argument('-o', '--output', default='verification/test_checkpoint.bin',
                        help='Output file path')
    parser.add_argument('-n', '--num-extra', type=int, default=5000,
                        help='Extra entries per street (default 5000)')
    parser.add_argument('-i', '--iterations', type=int, default=5000000000,
                        help='Iterations count in header')
    args = parser.parse_args()
    generate_checkpoint(args.output, args.iterations, args.num_extra)
