#!/usr/bin/env python3
"""Spot-check the new export's UTG root strategies against expected GTO frequencies.

After re-extracting from the Bug B-fixed .bps, run this to verify the visible
hands have sane converged frequencies. Compare to my earlier byte-streamed
strategy_sum values (from regrets_1000M.bin direct read) — they should be
APPROXIMATELY similar shape but slightly different (different iter count: 1.0B
vs ~1.6B, more strategy_sum samples).

Sentinel expectations from the Bug A diagnostic on 1B strategy_sum:
- AA  : 0% fold, ~50% call (slow-play mix), ~50% raises split across sizes
- AKs : 0% fold, ~37% call, ~60% raises
- AKo : 0% fold, ~40% call, ~60% raises
- AQs : 0% fold, ~54% call, ~46% raises
- KK  : 0% fold, ~23% call, ~77% raises (premium pair raises hardest)

Junk hands should be near-100% fold:
- 72o, 32o, 22, 52s, J5s, Q5o
"""
import json
import base64
import sys

OUT_PATH = "../nexusgto/src/data/preflop-nodes.json"

with open(OUT_PATH) as f:
    data = json.load(f)

print(f"=== Spot-check on {OUT_PATH} ===")
print(f"Schema: {data['meta'].get('quantize')}, "
      f"max_depth: {data['meta'].get('max_depth')}, "
      f"nodes: {len(data['nodes'])}")
print()

root = data['nodes'].get('0xFEDCBA9876543210')
if root is None:
    print("FATAL: UTG root node missing")
    sys.exit(1)

labels = root['l']
buf = base64.b64decode(root['s'])
na = len(labels)
hand_order = data['meta']['hand_order']

print(f"UTG root labels: {labels}")
print()

probe = [
    # (hand, expected_max_fold, expected_min_play_pct, note)
    ('AA',  0.05, 95, 'premium pair'),
    ('KK',  0.05, 95, 'premium pair'),
    ('QQ',  0.05, 95, 'premium pair'),
    ('JJ',  0.10, 90, 'premium pair'),
    ('TT',  0.20, 80, 'mid pair'),
    ('AKs', 0.05, 95, 'premium suited'),
    ('AKo', 0.10, 90, 'premium offsuit'),
    ('AQs', 0.05, 95, 'premium suited'),
    ('AQo', 0.20, 80, 'AQo borderline UTG'),
    ('AJs', 0.10, 90, 'AJs plays UTG'),
    ('AJo', 0.50, 50, 'AJo borderline / fold UTG 6max'),
    ('KQs', 0.10, 90, 'KQs plays UTG'),
    ('KQo', 0.40, 60, 'KQo borderline'),
    ('JTs', 0.20, 80, 'JTs plays UTG'),
    ('T9s', 0.50, 50, 'T9s borderline'),
    ('22',  0.50, None, 'small pair, often fold UTG 6max'),
    ('72o', 0.99, None, 'trash, ALWAYS fold'),
    ('32o', 0.99, None, 'trash, ALWAYS fold'),
    ('52s', 0.50, None, 'weak suited, fold UTG'),
]

print(f"{'hand':5s} {'fold':>7s}  {'play':>6s}  expected             note")
print('-' * 75)
issues = 0
for hand, max_fold, min_play, note in probe:
    if hand not in hand_order:
        print(f"{hand:5s} NOT IN GRID")
        continue
    idx = hand_order.index(hand)
    row = buf[idx*na:(idx+1)*na]
    fold_freq = row[0] / 255
    play_pct = (1 - fold_freq) * 100

    if max_fold < 0.5:
        # Premium-ish: should NOT fold
        ok = fold_freq <= max_fold
    else:
        # Trash: should fold
        ok = fold_freq >= max_fold

    if not ok and max_fold >= 0.5 and play_pct > (100 - max_fold * 100 - 5):
        # close enough
        ok = True

    marker = "OK  " if ok else "FAIL"
    if not ok:
        issues += 1

    play_str = f"{play_pct:5.1f}%"
    fold_str = f"{fold_freq:.3f}"
    if max_fold < 0.5:
        exp = f"fold <= {max_fold:.2f}"
    else:
        exp = f"fold >= {max_fold:.2f}"
    print(f"{hand:5s} {fold_str:>7s}  {play_str:>6s}  {exp:20s} {marker} {note}")

print()
print(f"Total issues: {issues}")

# Also print the raw frequencies for AA, KK, AKs, AKo so I can compare to my
# earlier byte-streamed sample
print()
print("=== Raw frequencies (compare to byte-streamed strategy_sum from regrets_1000M.bin) ===")
print(f"{'hand':5s} " + ''.join(f"{l:>10s}" for l in labels))
for hand in ['AA', 'KK', 'AKs', 'AKo', 'AQs', 'AQo', 'KQs']:
    if hand not in hand_order:
        continue
    idx = hand_order.index(hand)
    row = buf[idx*na:(idx+1)*na]
    freqs = [round(b/255, 3) for b in row]
    print(f"{hand:5s} " + ''.join(f"{f:10.3f}" for f in freqs))
