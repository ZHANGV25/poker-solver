#!/usr/bin/env python3
"""Run one config from sweep_config_tree.py and print results immediately.
   Usage: run_one_config.py <CONFIG_NAME>"""
import sys
import time

sys.path.insert(0, '/Users/victor/Documents/Dev/poker-solver/tests')
from sweep_config_tree import enumerate_tree, CONFIGS

if len(sys.argv) < 2:
    print("Usage: run_one_config.py <CONFIG_NAME>")
    print(f"Available: {list(CONFIGS.keys())}")
    sys.exit(1)

name = sys.argv[1]
if name not in CONFIGS:
    print(f"Unknown config {name}")
    sys.exit(1)

cfg = CONFIGS[name]
print(f"[{name}] starting full-tree enumeration", flush=True)
t0 = time.time()
counts, walked, hit_limit = enumerate_tree(cfg, preflop_only=False, time_limit=1800)
elapsed = time.time() - t0

pre_is = counts[0] * 169
post_is = sum(counts[s] * 200 for s in (1, 2, 3))
total_is = pre_is + post_is

flag = " (TIMEOUT)" if hit_limit else ""
print(f"[{name}] DONE in {elapsed:.0f}s{flag}", flush=True)
print(f"[{name}] preflop={counts[0]:,}  flop={counts[1]:,}  turn={counts[2]:,}  river={counts[3]:,}", flush=True)
print(f"[{name}] total info sets={total_is:,}  ({total_is/664_845_654:.1f}x Pluribus)", flush=True)
