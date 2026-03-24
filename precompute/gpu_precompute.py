#!/usr/bin/env python3
"""GPU batch precompute: solve all flop textures using CUDA solver.

Uses the CUDA-accelerated Linear CFR solver to precompute strategies
for all 1,755 flop textures × 27 scenarios locally on the RTX 3060.

Extracts:
  - Flop root strategies (for blueprint lookup)
  - Turn root strategies per runout (for depth-limited leaf evaluation)

Usage:
    python gpu_precompute.py --scenario CO_vs_BB_srp --iterations 500
    python gpu_precompute.py --all --iterations 500

Output:
    flop_solutions/{scenario_id}/{texture_key}.json
"""

import argparse
import ctypes
import json
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.dirname(__file__))

from solve_scenarios import generate_all_textures, load_scenarios

# Try to load the GPU solver
def load_gpu_solver():
    """Load the CUDA solver DLL/SO."""
    solver_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(solver_dir, "build")

    candidates = [
        os.path.join(build_dir, "gpu_solver.dll"),
        os.path.join(build_dir, "gpu_solver.so"),
        os.path.join(build_dir, "bench_gpu.dll"),
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                lib = ctypes.CDLL(path)
                lib.gpu_solve_batch.restype = ctypes.c_int
                lib.gpu_get_info.restype = ctypes.c_int
                return lib
            except Exception as e:
                print(f"  Failed to load {path}: {e}", file=sys.stderr)

    return None


def check_gpu():
    """Check GPU availability and memory."""
    lib = load_gpu_solver()
    if lib is None:
        print("WARNING: GPU solver not available. Falling back to CPU.",
              file=sys.stderr)
        return None

    cores = ctypes.c_int(0)
    free_mem = ctypes.c_size_t(0)
    total_mem = ctypes.c_size_t(0)
    lib.gpu_get_info(ctypes.byref(cores), ctypes.byref(free_mem),
                     ctypes.byref(total_mem))
    return {
        'cores': cores.value,
        'free_mb': free_mem.value / 1e6,
        'total_mb': total_mem.value / 1e6,
        'lib': lib,
    }


def estimate_batch_size(gpu_info, hands_per_texture=80):
    """Estimate how many textures we can solve per batch on the GPU."""
    if gpu_info is None:
        return 1

    free_mb = gpu_info['free_mb']
    # Per-texture memory: nodes × actions × hands × 3 arrays (regrets, strategy, strategy_sum)
    # ~128 nodes × 6 actions × 80 hands × 4 bytes × 3 = ~7 MB
    # Plus CFV and reach: ~128 nodes × 80 hands × 4 bytes × 2 = ~0.08 MB
    per_texture_mb = 7.0 + 0.1  # conservative
    batch_size = int(free_mb * 0.7 / per_texture_mb)  # use 70% of free memory
    return max(1, min(batch_size, 2000))


def main():
    parser = argparse.ArgumentParser(description="GPU batch precompute")
    parser.add_argument("--scenario", help="Solve a specific scenario")
    parser.add_argument("--all", action="store_true", help="Solve all scenarios")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Linear CFR iterations per texture")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ranges", default=None)
    parser.add_argument("--check-gpu", action="store_true",
                        help="Check GPU and exit")

    args = parser.parse_args()

    # Check GPU
    gpu = check_gpu()
    if args.check_gpu:
        if gpu:
            batch = estimate_batch_size(gpu)
            print(f"GPU: {gpu['cores']} cores, {gpu['free_mb']:.0f} MB free")
            print(f"Estimated batch size: {batch} textures")
        else:
            print("No GPU solver available")
        return

    if gpu is None:
        print("GPU solver not available. Use solve_scenarios.py with --solver-bin instead.")
        print("To build the GPU solver: nvcc -shared -o build/gpu_solver.dll "
              "src/cuda/gpu_solver.cu")
        sys.exit(1)

    # Find ranges
    ranges_path = args.ranges
    if ranges_path is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "data", "ranges.json"),
        ]
        # Also check the ACR project
        acr_dir = os.environ.get("ACR_PROJECT_DIR", "")
        if acr_dir:
            candidates.append(os.path.join(acr_dir, "solver", "ranges.json"))

        for c in candidates:
            if os.path.isfile(c):
                ranges_path = c
                break

    if ranges_path is None:
        print("ERROR: ranges.json not found. Use --ranges.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "flop_solutions")

    scenarios = load_scenarios(ranges_path)
    if args.scenario:
        scenarios = {args.scenario: scenarios[args.scenario]}
    elif not args.all:
        print(f"Specify --scenario or --all")
        print(f"Available: {sorted(scenarios.keys())}")
        return

    textures = generate_all_textures()
    batch_size = estimate_batch_size(gpu)

    print(f"Loaded {len(scenarios)} scenarios, {len(textures)} textures")
    print(f"GPU batch size: {batch_size}")
    print(f"Iterations: {args.iterations}")

    total_start = time.time()
    for s_idx, (scenario_id, scenario) in enumerate(sorted(scenarios.items())):
        out_dir = os.path.join(output_dir, scenario_id)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{s_idx+1}/{len(scenarios)}] {scenario_id} "
              f"({len(textures)} textures)")

        # TODO: batch GPU solving
        # For now, fall back to CPU solve_scenarios.py behavior
        print(f"  GPU batch solve not yet integrated — use solve_scenarios.py")

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
