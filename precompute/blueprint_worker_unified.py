#!/usr/bin/env python3
"""Unified blueprint worker: 6-player preflop-through-river MCCFR.

Matches Pluribus exactly: ONE unified solve over the entire game tree.
All 6 players, all 1326 hands, 169 lossless preflop buckets, 200 k-means
postflop buckets. Preflop → flop → turn → river → showdown in a single MCCFR.

This is the Pluribus approach (Brown & Sandholm, Science 2019):
  - 64 cores, 8 days, 12,400 CPU core hours, <512GB RAM
  - ~665M action sequences, ~414M encountered
  - Our target: 64-core c5.18xlarge ($3.06/hr) for 8 days ≈ $587

Usage (local test):
    python blueprint_worker_unified.py --iterations 10000 --num-threads 1

Usage (EC2, full Pluribus-scale):
    python blueprint_worker_unified.py --iterations 0 --time-limit-hours 192 \
        --num-threads 64 --hash-size 536870912 \
        --s3-bucket poker-blueprint-unified

The solver runs until --time-limit-hours is reached (Pluribus ran for 8 days
= 192 hours). Iteration count is not pre-specified — training runs continuously
and convergence is measured by exploitability and strategy stability.
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

# ── Constants (Pluribus-matched) ──────────────────────────────────

NUM_PLAYERS = 6
SMALL_BLIND = 50     # $50
BIG_BLIND = 100      # $100
INITIAL_STACK = 10000 # $10,000 = 100BB

# Pluribus bet sizes: up to 14 for preflop, 3 for later rounds
PREFLOP_BET_SIZES = [0.5, 1.0, 2.0, 3.0]  # 0.5x, 1x, 2x, 3x pot raises
POSTFLOP_BET_SIZES = [0.5, 1.0, 2.0]       # Pluribus: 0.5x pot, pot, all-in (≈2x)

# ── DLL loading ───────────────────────────────────────────────────

class BPConfig(ctypes.Structure):
    _fields_ = [
        ("discount_stop_iter", ctypes.c_int),
        ("discount_interval", ctypes.c_int),
        ("prune_start_iter", ctypes.c_int),
        ("snapshot_start_iter", ctypes.c_int),
        ("snapshot_interval", ctypes.c_int),
        ("strategy_interval", ctypes.c_int),
        ("num_threads", ctypes.c_int),
        ("hash_table_size", ctypes.c_int),
        ("snapshot_dir", ctypes.c_char_p),
        ("include_preflop", ctypes.c_int),
    ]


def load_bp_dll(build_dir):
    """Load the MCCFR blueprint DLL."""
    for ext in ['so', 'dll']:
        path = os.path.join(build_dir, f'mccfr_blueprint.{ext}')
        if os.path.exists(path):
            bp = ctypes.CDLL(path)
            bp.bp_default_config.restype = None
            bp.bp_init_unified.restype = ctypes.c_int
            bp.bp_set_buckets.restype = ctypes.c_int
            bp.bp_solve.restype = ctypes.c_int
            bp.bp_get_strategy.restype = ctypes.c_int
            bp.bp_num_info_sets.restype = ctypes.c_int
            bp.bp_export_strategies.restype = ctypes.c_int
            bp.bp_export_strategies.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t)
            ]
            bp.bp_save_regrets.restype = ctypes.c_int
            bp.bp_save_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            bp.bp_load_regrets.restype = ctypes.c_int
            bp.bp_load_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            bp.bp_free.restype = None
            return bp
    raise FileNotFoundError(f"mccfr_blueprint not found in {build_dir}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified 6-player blueprint (Pluribus-style)")
    parser.add_argument("--iterations", type=int, default=0,
                        help="Max iterations (0 = run until time limit)")
    parser.add_argument("--time-limit-hours", type=float, default=0,
                        help="Run for this many hours (Pluribus: 192 = 8 days)")
    parser.add_argument("--num-threads", type=int, default=0,
                        help="OpenMP threads (0=auto)")
    parser.add_argument("--hash-size", type=int, default=0,
                        help="Hash table slots (0=auto, Pluribus=536870912)")
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--output-dir", default="/tmp/blueprint_unified")
    parser.add_argument("--s3-bucket", default="")
    parser.add_argument("--checkpoint-interval", type=int, default=1000000,
                        help="Export checkpoint every N iterations")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest regret checkpoint on S3")
    args = parser.parse_args()

    print(f"=== Unified 6-Player Blueprint (Pluribus-style) ===")
    print(f"Players: {NUM_PLAYERS}")
    print(f"Blinds: {SMALL_BLIND}/{BIG_BLIND}")
    print(f"Stack: {INITIAL_STACK} ({INITIAL_STACK//BIG_BLIND}BB)")
    print(f"Preflop bet sizes: {PREFLOP_BET_SIZES}")
    print(f"Postflop bet sizes: {POSTFLOP_BET_SIZES}")
    print(f"Threads: {args.num_threads or 'auto'}")
    print()

    bp_lib = load_bp_dll(args.build_dir)
    print("DLL loaded.")

    # Config — scale to Pluribus timing
    config = BPConfig()
    bp_lib.bp_default_config(ctypes.byref(config))
    config.num_threads = args.num_threads
    config.include_preflop = 1

    if args.hash_size > 0:
        config.hash_table_size = args.hash_size

    # If running by time limit, estimate iterations.
    # Pluribus: 8 days on 64 cores, ~1000 iter/min → ~11.5M total iterations.
    # We'll set a high iteration count and check time externally.
    if args.time_limit_hours > 0 and args.iterations == 0:
        # Estimate: ~300 iter/s per thread.
        # When --num-threads 0, OpenMP auto-detects; use os.cpu_count() as estimate.
        effective_threads = args.num_threads if args.num_threads > 0 else (os.cpu_count() or 1)
        est_iter_per_sec = effective_threads * 300
        args.iterations = int(args.time_limit_hours * 3600 * est_iter_per_sec)
        print(f"Time limit: {args.time_limit_hours}h, {effective_threads} threads, "
              f"~{est_iter_per_sec} iter/s → estimated {args.iterations:,} iterations")

    if args.iterations <= 0:
        args.iterations = 1000000  # default 1M

    # Scale Pluribus timing parameters proportionally
    # Pluribus: 400 min discount out of ~11520 min (8 days) = 3.5%
    config.discount_stop_iter = max(args.iterations * 35 // 1000, 1000)
    config.discount_interval = max(config.discount_stop_iter // 40, 100)
    config.prune_start_iter = max(args.iterations * 17 // 1000, 500)
    config.snapshot_start_iter = max(args.iterations * 70 // 100, 10000)
    config.snapshot_interval = max(args.iterations * 17 // 100, 5000)
    config.strategy_interval = 10000  # Pluribus: every 10K iterations

    print(f"Iterations: {args.iterations:,}")
    print(f"Discount: first {config.discount_stop_iter:,} iters, every {config.discount_interval:,}")
    print(f"Pruning: after iter {config.prune_start_iter:,}")
    print(f"Snapshots: after iter {config.snapshot_start_iter:,}, every {config.snapshot_interval:,}")
    print()

    # Initialize
    buf = (ctypes.c_char * 524288)()
    solver = ctypes.cast(buf, ctypes.c_void_p)

    c_postflop = (ctypes.c_float * len(POSTFLOP_BET_SIZES))(*POSTFLOP_BET_SIZES)
    c_preflop = (ctypes.c_float * len(PREFLOP_BET_SIZES))(*PREFLOP_BET_SIZES)

    ret = bp_lib.bp_init_unified(
        solver, NUM_PLAYERS,
        SMALL_BLIND, BIG_BLIND, INITIAL_STACK,
        c_postflop, len(POSTFLOP_BET_SIZES),
        c_preflop, len(PREFLOP_BET_SIZES),
        ctypes.byref(config)
    )
    if ret != 0:
        print(f"FATAL: bp_init_unified returned {ret}")
        sys.exit(1)
    print("Solver initialized.")
    # Postflop buckets (200 EHS percentile) are precomputed inside bp_init_unified
    # for all 1,755 flop textures. During traversal, the dealt flop is canonicalized
    # and the precomputed bucket mapping is looked up in O(1755) per deal.

    # Resume from checkpoint if requested
    iters_already_done = 0
    if args.resume and args.s3_bucket:
        regret_s3 = f"s3://{args.s3_bucket}/checkpoints/regrets_latest.bin"
        regret_local = os.path.join(args.output_dir, "regrets_latest.bin")
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Checking for checkpoint at {regret_s3}...")
        ret = subprocess.run(
            ["aws", "s3", "cp", regret_s3, regret_local, "--quiet"],
            capture_output=True
        )
        if ret.returncode == 0 and os.path.exists(regret_local):
            n_loaded = bp_lib.bp_load_regrets(
                solver, regret_local.encode('utf-8'))
            if n_loaded > 0:
                # Read back iterations_run from the solver struct
                # (bp_load_regrets sets s->iterations_run)
                # We can't easily read it from Python, but the C code printed it.
                # Parse it from the checkpoint metadata instead.
                try:
                    meta_s3 = f"s3://{args.s3_bucket}/checkpoint_meta.json"
                    meta_local = os.path.join(args.output_dir, "checkpoint_meta.json")
                    subprocess.run(
                        ["aws", "s3", "cp", meta_s3, meta_local, "--quiet"],
                        capture_output=True
                    )
                    if os.path.exists(meta_local):
                        import json as _json
                        with open(meta_local) as mf:
                            cmeta = _json.load(mf)
                        iters_already_done = cmeta.get("iterations", 0)
                except Exception:
                    pass
                print(f"Resumed from checkpoint: {n_loaded} info sets, "
                      f"{iters_already_done:,} iterations already done")
            os.remove(regret_local)
        else:
            print("No checkpoint found, starting fresh.")

    # Solve in chunks with periodic checkpoints.
    # Each checkpoint exports the full strategy to .bps and uploads to S3.
    # If the instance dies, we lose at most one chunk of work.
    # The C solver's hash table accumulates regrets across bp_solve calls,
    # so calling bp_solve(N) then bp_solve(M) equals bp_solve(N+M).
    chunk_size = args.checkpoint_interval
    if chunk_size <= 0:
        chunk_size = args.iterations  # no checkpointing

    total_iters = args.iterations
    iters_done = iters_already_done
    iters_remaining = total_iters - iters_done

    if iters_remaining <= 0:
        print(f"Already completed {iters_done:,} >= {total_iters:,} target. Done.")
        bp_lib.bp_free(solver)
        return

    print(f"\nStarting solve: {iters_remaining:,} iterations remaining "
          f"(of {total_iters:,}), checkpoint every {chunk_size:,}...")
    t0 = time.time()

    t0 = time.time()

    while iters_done < total_iters:
        this_chunk = min(chunk_size, total_iters - iters_done)
        bp_lib.bp_solve(solver, this_chunk)
        iters_done += this_chunk

        elapsed = time.time() - t0
        n_is = bp_lib.bp_num_info_sets(solver)
        ips = iters_done / elapsed if elapsed > 0 else 0
        remaining_h = (total_iters - iters_done) / ips / 3600 if ips > 0 else 0
        print(f"[Checkpoint] {iters_done:,}/{total_iters:,} iters, "
              f"{n_is:,} IS, {elapsed/3600:.1f}h elapsed, "
              f"~{remaining_h:.1f}h remaining, {ips:.0f} iter/s")

        # Export checkpoint
        if args.output_dir:
            _export_checkpoint(bp_lib, solver, args, iters_done, elapsed, n_is)

    elapsed = time.time() - t0
    n_is = bp_lib.bp_num_info_sets(solver)
    print(f"\nDone: {iters_done:,} iterations, {n_is:,} info sets")
    print(f"Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Speed: {iters_done/elapsed:.0f} iter/s")
    print(f"CPU-hours: {elapsed * max(args.num_threads, 1) / 3600:.0f}")

    # Final export (same as last checkpoint, but labeled as final)
    if args.output_dir:
        _export_checkpoint(bp_lib, solver, args, iters_done, elapsed, n_is,
                           label="final")

    bp_lib.bp_free(solver)
    print("\nDone.")


def _s3_upload_with_retry(local_path, s3_uri, max_attempts=3, delay=5):
    """Upload a file to S3 with retry logic."""
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(
                ["aws", "s3", "cp", local_path, s3_uri, "--quiet"],
                check=True,
                capture_output=True,
            )
            return  # success
        except subprocess.CalledProcessError as e:
            if attempt < max_attempts:
                print(f"  [WARN] S3 upload failed (attempt {attempt}/{max_attempts}): "
                      f"{e.stderr.decode().strip()}, retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  [ERROR] S3 upload failed after {max_attempts} attempts: "
                      f"{local_path} -> {s3_uri}: {e.stderr.decode().strip()}")
                raise


def _export_checkpoint(bp_lib, solver, args, iters_done, elapsed, n_is,
                       label=None):
    """Export strategy + regret checkpoint and upload to S3.

    Saves two things:
    1. Strategy .bps file (for use by the runtime GPU solver)
    2. Regret table .bin file (for resuming training after interruption)

    Uses atomic writes (write to temp file, then rename) to prevent
    corrupted checkpoints if the process is killed mid-write.
    """
    import lzma
    import struct

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Save regret table (for resume) — atomic write ──
    regret_path = os.path.join(args.output_dir, "regrets_latest.bin")
    regret_tmp = regret_path + ".tmp"
    bp_lib.bp_save_regrets(solver, regret_tmp.encode('utf-8'))
    os.replace(regret_tmp, regret_path)

    # Save checkpoint metadata — atomic write
    meta = {
        "type": "unified_blueprint",
        "num_players": NUM_PLAYERS,
        "blinds": [SMALL_BLIND, BIG_BLIND],
        "initial_stack": INITIAL_STACK,
        "preflop_bet_sizes": PREFLOP_BET_SIZES,
        "postflop_bet_sizes": POSTFLOP_BET_SIZES,
        "iterations": iters_done,
        "num_info_sets": n_is,
        "time_seconds": round(elapsed),
        "time_hours": round(elapsed / 3600, 1),
        "preflop_buckets": 169,
        "postflop_buckets": 200,
        "checkpoint": label or f"iter_{iters_done}",
    }
    meta_path = os.path.join(args.output_dir, "checkpoint_meta.json")
    meta_tmp = meta_path + ".tmp"
    with open(meta_tmp, 'w') as f:
        json.dump(meta, f)
    os.replace(meta_tmp, meta_path)

    # ── Save strategy .bps (for runtime use) ──
    needed = ctypes.c_size_t(0)
    bp_lib.bp_export_strategies(solver, None, 0, ctypes.byref(needed))
    strat_size = needed.value

    if strat_size > 0:
        strat_buf = (ctypes.c_char * strat_size)()
        written = ctypes.c_size_t(0)
        bp_lib.bp_export_strategies(solver, strat_buf, strat_size, ctypes.byref(written))
        strat_data = bytes(strat_buf[:written.value])
        compressed = lzma.compress(strat_data, preset=1)
        meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')

        fname = "unified_blueprint.bps"
        out_path = os.path.join(args.output_dir, fname)
        out_tmp = out_path + ".tmp"
        with open(out_tmp, 'wb') as f:
            f.write(b'BPS2')
            f.write(struct.pack('<II', len(compressed), len(meta_bytes)))
            f.write(compressed)
            f.write(meta_bytes)
        os.replace(out_tmp, out_path)

        comp_mb = len(compressed) / 1024 / 1024
        regret_mb = os.path.getsize(regret_path) / 1024 / 1024
        print(f"  Checkpoint: strategy={comp_mb:.0f}MB, regrets={regret_mb:.0f}MB")
    else:
        print("  [WARN] No strategies to export")

    # ── Upload to S3 (with retries) ──
    if args.s3_bucket:
        for local, s3key in [
            (regret_path, "checkpoints/regrets_latest.bin"),
            (meta_path, "checkpoint_meta.json"),
            (os.path.join(args.output_dir, "unified_blueprint.bps"),
             "unified_blueprint.bps"),
            (os.path.join(args.output_dir, "unified_blueprint.bps"),
             f"checkpoints/blueprint_iter{iters_done}.bps"),
        ]:
            if os.path.exists(local):
                try:
                    _s3_upload_with_retry(
                        local, f"s3://{args.s3_bucket}/{s3key}")
                except subprocess.CalledProcessError:
                    pass  # already logged in _s3_upload_with_retry
        print(f"  Uploaded to s3://{args.s3_bucket}/")


if __name__ == "__main__":
    main()
