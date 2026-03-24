#!/usr/bin/env python3
"""Aggregate blueprint results from all EC2 workers.

Downloads from S3 (or reads local directory), combines per-texture JSONs
into a single blueprint file, and reports coverage + quality metrics.

Usage:
    # After download:
    python aggregate_blueprint.py --input-dir ./blueprint_output

    # Direct from S3:
    python aggregate_blueprint.py --s3-bucket poker-solver-blueprints
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Aggregate blueprint results")
    parser.add_argument("--input-dir", default="./blueprint_output")
    parser.add_argument("--s3-bucket", default="")
    parser.add_argument("--output", default="./blueprint_combined.json")
    args = parser.parse_args()

    # Download from S3 if specified
    if args.s3_bucket:
        print(f"Downloading from s3://{args.s3_bucket}/...")
        os.makedirs(args.input_dir, exist_ok=True)
        subprocess.run([
            "aws", "s3", "sync",
            f"s3://{args.s3_bucket}/", args.input_dir,
            "--exclude", "code/*", "--quiet"
        ], check=True)

    # Find all texture JSON files
    texture_files = []
    summaries = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            path = os.path.join(root, f)
            if f.startswith("summary_"):
                summaries.append(path)
            elif f.endswith(".json") and not f.startswith("summary"):
                texture_files.append(path)

    print(f"Found {len(texture_files)} texture files, {len(summaries)} summaries")

    # Load summaries
    total_time = 0
    total_solved = 0
    total_failed = 0
    for sp in summaries:
        with open(sp) as f:
            s = json.load(f)
        total_time += s.get("total_time_s", 0)
        total_solved += s.get("textures_solved", 0)
        total_failed += s.get("textures_failed", 0)
        if s.get("failures"):
            print(f"  Worker {s['worker_id']}: {s['textures_failed']} failures: {s['failures'][:5]}")

    print(f"Total solved: {total_solved}, failed: {total_failed}")
    print(f"Total compute time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print()

    # Load all textures
    blueprint = {}
    info_set_total = 0
    strategies_total = 0

    for tf in texture_files:
        try:
            with open(tf) as f:
                data = json.load(f)
            key = data["texture"]
            blueprint[key] = data
            info_set_total += data.get("num_info_sets", 0)
            strategies_total += len(data.get("root_strategies", {}))
        except Exception as e:
            print(f"  Error loading {tf}: {e}")

    print(f"Loaded {len(blueprint)} textures")
    print(f"Total info sets: {info_set_total:,}")
    print(f"Total root strategies: {strategies_total:,}")
    print()

    # Coverage check
    from precompute.solve_scenarios import generate_all_textures
    all_textures = {key for key, _ in generate_all_textures()}
    solved_textures = set(blueprint.keys())
    missing = all_textures - solved_textures
    coverage = len(solved_textures) / len(all_textures) * 100

    print(f"Coverage: {len(solved_textures)}/{len(all_textures)} ({coverage:.1f}%)")
    if missing:
        print(f"Missing textures: {len(missing)}")
        if len(missing) <= 20:
            for m in sorted(missing):
                print(f"  {m}")

    # Quality metrics
    uniform_count = 0
    aggressive_count = 0
    passive_count = 0
    for key, data in blueprint.items():
        strats = data.get("root_strategies", {})
        for b, s in strats.items():
            na = len(s)
            if na == 0:
                continue
            if all(abs(v - 1.0/na) < 0.02 for v in s):
                uniform_count += 1
            elif s[0] > 0.7:  # mostly check/fold
                passive_count += 1
            elif sum(s[1:]) > 0.7:  # mostly bet
                aggressive_count += 1

    print()
    print(f"Strategy quality:")
    print(f"  Uniform (unconverged): {uniform_count}/{strategies_total} ({100*uniform_count/max(strategies_total,1):.1f}%)")
    print(f"  Passive (check>70%): {passive_count}/{strategies_total}")
    print(f"  Aggressive (bet>70%): {aggressive_count}/{strategies_total}")

    # Save combined blueprint
    print(f"\nSaving combined blueprint to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(blueprint, f, separators=(',', ':'))
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Saved: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
