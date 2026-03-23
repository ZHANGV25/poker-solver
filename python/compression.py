"""LZMA compression for blueprint data.

Compresses and decompresses flop solution JSON files using LZMA,
with optional FP16 quantization for strategy frequencies.

Typical compression ratios:
    - JSON -> LZMA: 8-12x compression
    - JSON -> FP16+LZMA: 12-18x compression

Usage:
    # Compress a directory of solutions
    compress_directory("flop_solutions/CO_vs_BB_srp/",
                       "flop_solutions_lzma/CO_vs_BB_srp/")

    # Decompress a single file
    data = decompress_file("flop_solutions_lzma/CO_vs_BB_srp/AK5_fd12.json.lzma")

    # Compress entire flop_solutions directory
    compress_all("flop_solutions/", "flop_solutions_lzma/")
"""

import json
import lzma
import os
import struct
import sys
import time
from typing import Dict, Optional


def compress_solution(data, use_fp16=True):
    """Compress a solution dict to LZMA bytes.

    Args:
        data: solution dict (from JSON)
        use_fp16: if True, quantize strategy frequencies to FP16

    Returns:
        compressed bytes
    """
    if use_fp16:
        data = _quantize_strategies(data)

    json_bytes = json.dumps(data, separators=(',', ':')).encode('utf-8')
    return lzma.compress(json_bytes, preset=6)  # preset 6 = good ratio + speed


def decompress_solution(compressed_bytes):
    """Decompress LZMA bytes to a solution dict.

    Args:
        compressed_bytes: LZMA-compressed bytes

    Returns:
        solution dict
    """
    json_bytes = lzma.decompress(compressed_bytes)
    return json.loads(json_bytes)


def _quantize_strategies(data):
    """Quantize strategy frequencies to 4 decimal places (saves ~30% JSON size)."""
    if 'hands' not in data or data['hands'] is None:
        return data

    result = dict(data)
    result['hands'] = {}
    for node_key, hands in data['hands'].items():
        result['hands'][node_key] = {}
        for hand_str, strat in hands.items():
            q_strat = dict(strat)
            if 'actions' in q_strat:
                q_strat['actions'] = [
                    {
                        'action': a['action'],
                        'frequency': round(a['frequency'], 4),
                        **(({'ev': round(a['ev'], 2)} if a.get('ev', 0) != 0 else {})),
                    }
                    for a in q_strat['actions']
                ]
            if 'ev' in q_strat:
                q_strat['ev'] = round(q_strat['ev'], 3)
            if 'equity' in q_strat:
                q_strat['equity'] = round(q_strat['equity'], 4)
            result['hands'][node_key][hand_str] = q_strat

    return result


def compress_file(input_path, output_path=None):
    """Compress a single JSON solution file to LZMA.

    Args:
        input_path: path to .json file
        output_path: path for .json.lzma file (default: input + .lzma)

    Returns:
        (original_size, compressed_size, ratio)
    """
    if output_path is None:
        output_path = input_path + '.lzma'

    with open(input_path, 'r') as f:
        data = json.load(f)

    original_size = os.path.getsize(input_path)
    compressed = compress_solution(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(compressed)

    compressed_size = len(compressed)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    return original_size, compressed_size, ratio


def decompress_file(input_path):
    """Decompress a .json.lzma file to a dict.

    Args:
        input_path: path to .json.lzma file

    Returns:
        solution dict
    """
    with open(input_path, 'rb') as f:
        compressed = f.read()
    return decompress_solution(compressed)


def compress_directory(input_dir, output_dir, verbose=True):
    """Compress all JSON files in a directory to LZMA.

    Args:
        input_dir: source directory with .json files
        output_dir: destination directory for .json.lzma files
        verbose: print progress

    Returns:
        dict with stats (total_original, total_compressed, num_files, ratio)
    """
    os.makedirs(output_dir, exist_ok=True)
    total_orig = 0
    total_comp = 0
    num_files = 0

    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    t0 = time.time()

    for i, fname in enumerate(files):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname + '.lzma')

        orig, comp, ratio = compress_file(in_path, out_path)
        total_orig += orig
        total_comp += comp
        num_files += 1

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print("  {}/{} files ({:.0f}/s) ratio={:.1f}x".format(
                i + 1, len(files), rate, total_orig / total_comp))

    overall_ratio = total_orig / total_comp if total_comp > 0 else 0
    if verbose:
        print("  Done: {} files, {:.1f} MB -> {:.1f} MB ({:.1f}x)".format(
            num_files, total_orig / 1e6, total_comp / 1e6, overall_ratio))

    return {
        'total_original': total_orig,
        'total_compressed': total_comp,
        'num_files': num_files,
        'ratio': overall_ratio,
    }


def compress_all(solutions_dir, output_dir, verbose=True):
    """Compress all scenarios in the flop_solutions directory.

    Args:
        solutions_dir: path to flop_solutions/ with scenario subdirs
        output_dir: destination for compressed files
    """
    if verbose:
        print("Compressing all scenarios...")

    total_stats = {'total_original': 0, 'total_compressed': 0,
                   'num_files': 0}

    for scenario in sorted(os.listdir(solutions_dir)):
        scenario_dir = os.path.join(solutions_dir, scenario)
        if not os.path.isdir(scenario_dir):
            continue

        json_files = [f for f in os.listdir(scenario_dir) if f.endswith('.json')]
        if not json_files:
            continue

        if verbose:
            print("\n  Scenario: {} ({} files)".format(scenario, len(json_files)))

        out_dir = os.path.join(output_dir, scenario)
        stats = compress_directory(scenario_dir, out_dir, verbose=verbose)

        total_stats['total_original'] += stats['total_original']
        total_stats['total_compressed'] += stats['total_compressed']
        total_stats['num_files'] += stats['num_files']

    ratio = (total_stats['total_original'] / total_stats['total_compressed']
             if total_stats['total_compressed'] > 0 else 0)
    if verbose:
        print("\nTotal: {} files, {:.1f} MB -> {:.1f} MB ({:.1f}x)".format(
            total_stats['num_files'],
            total_stats['total_original'] / 1e6,
            total_stats['total_compressed'] / 1e6,
            ratio))

    return total_stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compress flop solutions with LZMA')
    parser.add_argument('input_dir', help='Input flop_solutions directory')
    parser.add_argument('output_dir', help='Output directory for compressed files')
    parser.add_argument('--scenario', help='Compress a single scenario')
    args = parser.parse_args()

    if args.scenario:
        in_dir = os.path.join(args.input_dir, args.scenario)
        out_dir = os.path.join(args.output_dir, args.scenario)
        compress_directory(in_dir, out_dir)
    else:
        compress_all(args.input_dir, args.output_dir)
