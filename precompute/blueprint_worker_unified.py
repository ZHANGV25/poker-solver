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
import signal
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
# Flat preflop sizes (used as fallback if tiered not set)
PREFLOP_BET_SIZES = [0.5, 0.7, 1.0]
# Postflop first raise: Pluribus turn/river = {0.5x, 1x, all-in}
# (all-in added automatically by generate_actions; subsequent raises use {1x, all-in})
POSTFLOP_BET_SIZES = [0.5, 1.0]

# Tiered preflop sizing (Pluribus-style: fewer sizes at deeper raise levels).
# Open gets fine-grained sizes; 3-bet/4-bet/5-bet get progressively fewer.
# Enumerated tree: 2.28M preflop nodes × 169 = 386M preflop info sets (vs 7.4B flat).
PREFLOP_TIERS = {
    0: [0.5, 0.7, 1.0],    # open raise: 3 sizes (1.75BB, 2.05BB, 2.5BB)
    1: [0.7, 1.0],          # 3-bet: 2 sizes
    2: [1.0],               # 4-bet: 1 size
    3: [8.0],               # 5-bet: shove only
}
PREFLOP_MAX_RAISES = 4

# ── DLL loading ───────────────────────────────────────────────────

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
            bp.bp_solve.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            bp.bp_get_strategy.restype = ctypes.c_int
            bp.bp_num_info_sets.restype = ctypes.c_int64
            bp.bp_num_info_sets.argtypes = [ctypes.c_void_p]
            bp.bp_export_strategies.restype = ctypes.c_int
            bp.bp_export_strategies.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t)
            ]
            bp.bp_set_preflop_tier.restype = ctypes.c_int
            bp.bp_set_preflop_tier.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.c_int
            ]
            bp.bp_get_regrets.restype = ctypes.c_int
            bp.bp_save_regrets.restype = ctypes.c_int
            bp.bp_save_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            bp.bp_load_regrets.restype = ctypes.c_int64
            bp.bp_load_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            bp.bp_save_texture_cache.restype = ctypes.c_int
            bp.bp_save_texture_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            bp.bp_load_texture_cache.restype = ctypes.c_int
            bp.bp_load_texture_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            bp.bp_free.restype = None
            # Hash table health stats (added for v2)
            try:
                bp.bp_get_table_stats.restype = None
                bp.bp_get_table_stats.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_int64),
                    ctypes.POINTER(ctypes.c_int64),
                ]
            except AttributeError:
                # Older .so without the new symbol — silently disable.
                bp.bp_get_table_stats = None
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
    parser.add_argument("--output-dir", default="/opt/blueprint_unified")
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
    print(f"Preflop tiers: {PREFLOP_TIERS}")
    print(f"Preflop max raises: {PREFLOP_MAX_RAISES}")
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
    else:
        config.hash_table_size = 3000000000  # 3B slots (~180GB meta, fits c7a.metal 376GB)

    # If running by time limit, estimate iterations based on Pluribus core-hours.
    # Pluribus: 12,400 core-hours on 64 cores. We match total core-hours.
    # Burst throughput is ~280K iter/s (pruned) / ~140K (unpruned), average ~200K.
    # With large checkpoint intervals, overhead is negligible.
    if args.time_limit_hours > 0 and args.iterations == 0:
        effective_threads = args.num_threads if args.num_threads > 0 else (os.cpu_count() or 1)
        # Target: Pluribus's 12,400 core-hours scaled to our core count
        target_hours = 12400 / effective_threads  # wall-clock hours of solving
        # Use conservative average throughput (mix of pruned/unpruned phases)
        est_iter_per_sec = 200000  # ~200K iter/s on 96 threads
        args.iterations = int(target_hours * 3600 * est_iter_per_sec)
        print(f"Target: {12400} core-hours / {effective_threads} cores = {target_hours:.1f}h solving")
        print(f"Estimated throughput: ~{est_iter_per_sec:,} iter/s → {args.iterations:,} iterations")

    if args.iterations <= 0:
        args.iterations = 1000000  # default 1M

    # Scale Pluribus timing parameters proportionally (int64, no overflow concern).
    config.discount_stop_iter = max(args.iterations * 35 // 1000, 1000)
    config.discount_interval = max(config.discount_stop_iter // 40, 100)
    config.prune_start_iter = max(args.iterations * 17 // 1000, 500)
    config.snapshot_start_iter = max(args.iterations * 7 // 100, 10000)
    config.snapshot_interval = max(args.iterations * 17 // 1000, 5000)
    config.strategy_interval = 10000  # Pluribus: every 10K iterations

    print(f"Iterations: {args.iterations:,}")
    print(f"Discount: first {config.discount_stop_iter:,} iters, every {config.discount_interval:,}")
    print(f"Pruning: after iter {config.prune_start_iter:,}")
    print(f"Snapshots: after iter {config.snapshot_start_iter:,}, every {config.snapshot_interval:,}")
    print()

    # Try to load cached texture buckets from S3 (saves ~65 min precompute)
    texture_cache_local = os.path.join(args.output_dir, "texture_cache.bin")
    texture_loaded = False
    if args.s3_bucket:
        os.makedirs(args.output_dir, exist_ok=True)
        texture_s3 = f"s3://{args.s3_bucket}/texture_cache.bin"
        ret = subprocess.run(
            ["aws", "s3", "cp", texture_s3, texture_cache_local, "--quiet"],
            capture_output=True
        )
        if ret.returncode == 0 and os.path.exists(texture_cache_local):
            print(f"Found texture cache on S3, will load after init")
            texture_loaded = True
        else:
            print("No texture cache found, will precompute")

    # Initialize
    buf = (ctypes.c_char * 524288)()
    solver = ctypes.cast(buf, ctypes.c_void_p)

    c_postflop = (ctypes.c_float * len(POSTFLOP_BET_SIZES))(*POSTFLOP_BET_SIZES)
    c_preflop = (ctypes.c_float * len(PREFLOP_BET_SIZES))(*PREFLOP_BET_SIZES)

    # Load texture cache BEFORE init so bp_init_unified skips the 65-min precompute.
    # bp_load_texture_cache allocates the cache array and sets num_cached_textures > 0.
    # bp_init_unified checks this and skips if already loaded.
    if texture_loaded:
        n = bp_lib.bp_load_texture_cache(solver, texture_cache_local.encode('utf-8'))
        if n > 0:
            print(f"Loaded texture cache: {n} textures")
        else:
            print("Failed to load texture cache, will precompute")
            texture_loaded = False

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

    # Set tiered preflop sizing (overrides flat sizes)
    for level, sizes in sorted(PREFLOP_TIERS.items()):
        c_sizes = (ctypes.c_float * len(sizes))(*sizes)
        ret = bp_lib.bp_set_preflop_tier(
            solver, level, c_sizes, len(sizes), PREFLOP_MAX_RAISES)
        if ret != 0:
            print(f"FATAL: bp_set_preflop_tier level {level} returned {ret}")
            sys.exit(1)
    print(f"Tiered preflop: {len(PREFLOP_TIERS)} levels, max {PREFLOP_MAX_RAISES} raises")

    # Save texture cache for future runs (skips 65-min precompute next time)
    if not texture_loaded and args.s3_bucket:
        os.makedirs(args.output_dir, exist_ok=True)
        bp_lib.bp_save_texture_cache(solver, texture_cache_local.encode('utf-8'))
        try:
            subprocess.run(
                ["aws", "s3", "cp", texture_cache_local,
                 f"s3://{args.s3_bucket}/texture_cache.bin", "--quiet"],
                check=True, capture_output=True
            )
            print("Texture cache saved to S3")
        except subprocess.CalledProcessError:
            print("Warning: failed to upload texture cache to S3")
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
    total_iters = args.iterations
    iters_done = iters_already_done
    iters_remaining = total_iters - iters_done

    if iters_remaining <= 0:
        print(f"Already completed {iters_done:,} >= {total_iters:,} target. Done.")
        bp_lib.bp_free(solver)
        return

    print(f"\nStarting solve: {iters_remaining:,} iterations remaining "
          f"(of {total_iters:,}), adaptive checkpoints...")
    # Crash debugging: flush on every print and catch signals
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    def _signal_handler(signum, frame):
        print(f"\n[FATAL] Caught signal {signum} ({signal.Signals(signum).name})",
              flush=True)
        sys.exit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            pass

    t0 = time.time()

    # Checkpoint schedule: dense early, then every 300M for spot safety.
    # Target: 4B iterations. Every checkpoint uploaded to S3 so spot
    # reclamation loses at most ~300M iterations (~5.5h at 15K iter/s).
    checkpoint_milestones = [
        200_000_000, 400_000_000, 600_000_000,
        900_000_000, 1_200_000_000, 1_500_000_000, 1_800_000_000,
        2_100_000_000, 2_400_000_000, 2_700_000_000, 3_000_000_000,
        3_300_000_000, 3_600_000_000, 4_000_000_000,
    ]
    # After last explicit milestone, checkpoint every 300M
    checkpoint_ramp_interval = 300_000_000
    last_milestone = checkpoint_milestones[-1] if checkpoint_milestones else 0

    def _next_checkpoint(current):
        """Return the next checkpoint iteration after current."""
        for m in checkpoint_milestones:
            if m > current:
                return m
        # Past all milestones: use ramp interval
        base = last_milestone
        while base <= current:
            base += checkpoint_ramp_interval
        return base

    next_checkpoint_at = _next_checkpoint(iters_done)

    # bp_solve takes int32 max_iterations for the C call.
    INT32_MAX = 2_147_483_647

    # Solve in mini-chunks (50M) with lightweight strategy probes between each.
    # Full checkpoint (regret save + S3 upload) only at adaptive milestones.
    mini_chunk = 50_000_000  # strategy probe every 50M iterations

    while iters_done < total_iters:
        # Solve one mini-chunk
        call_size = min(mini_chunk, total_iters - iters_done, INT32_MAX)
        call_size_int = int(call_size)
        print(f"[Solve] {call_size_int:,} iters from {iters_done:,}...", flush=True)
        ret = bp_lib.bp_solve(solver, ctypes.c_int64(call_size_int))
        if ret != 0:
            print(f"[Solve] bp_solve returned {ret}", flush=True)
        iters_done += call_size

        elapsed = time.time() - t0
        n_is = bp_lib.bp_num_info_sets(solver)
        ips = iters_done / elapsed if elapsed > 0 else 0

        # Hash table health stats. New API as of v2; gated on availability so
        # the script still works against older .so builds.
        if getattr(bp_lib, 'bp_get_table_stats', None) is not None:
            ht_entries = ctypes.c_int64(0)
            ht_size = ctypes.c_int64(0)
            ht_fails = ctypes.c_int64(0)
            ht_max_probe = ctypes.c_int64(0)
            bp_lib.bp_get_table_stats(
                solver,
                ctypes.byref(ht_entries),
                ctypes.byref(ht_size),
                ctypes.byref(ht_fails),
                ctypes.byref(ht_max_probe),
            )
            if ht_size.value > 0:
                load_pct = 100.0 * ht_entries.value / ht_size.value
                print(
                    f"[Table] entries={ht_entries.value:,} "
                    f"({load_pct:.2f}% load), "
                    f"ins_fails={ht_fails.value}, "
                    f"max_probe={ht_max_probe.value}",
                    flush=True,
                )
                if ht_fails.value > 0:
                    print(
                        f"[Table] WARNING: {ht_fails.value} insertion failures — "
                        f"hash table too small or pathologically clustered. "
                        f"Bump --hash-size for next run.",
                        flush=True,
                    )

        # Lightweight strategy probe: extract strategies + raw regrets
        # directly from memory via bp_get_strategy/bp_get_regrets (no disk I/O).
        probe = _probe_preflop(bp_lib, solver)
        if probe:
            header = f"[Probe {iters_done:,}]"
            for line in probe.split("\n"):
                print(f"{header} {line}", flush=True)
            # Upload probe file to S3
            if args.s3_bucket and args.output_dir:
                iter_tag = f"{iters_done // 1_000_000}M"
                probe_path = os.path.join(args.output_dir, "probe_latest.txt")
                with open(probe_path, 'w') as pf:
                    pf.write(f"iteration: {iters_done}\n{probe}\n")
                try:
                    subprocess.run(
                        ["aws", "s3", "cp", probe_path,
                         f"s3://{args.s3_bucket}/probes/probe_{iter_tag}.txt",
                         "--quiet"], capture_output=True, timeout=30)
                    subprocess.run(
                        ["aws", "s3", "cp", probe_path,
                         f"s3://{args.s3_bucket}/probes/probe_latest.txt",
                         "--quiet"], capture_output=True, timeout=30)
                except Exception:
                    pass

        # Full checkpoint at adaptive milestones (200M, 400M, 600M, 2B, 4B, ...)
        if iters_done >= next_checkpoint_at or iters_done >= total_iters:
            remaining_h = (total_iters - iters_done) / ips / 3600 if ips > 0 else 0
            print(f"[Checkpoint] {iters_done:,}/{total_iters:,} iters, "
                  f"{n_is:,} IS, {elapsed/3600:.1f}h elapsed, "
                  f"~{remaining_h:.1f}h remaining, {ips:.0f} iter/s")
            if args.output_dir:
                _export_checkpoint(bp_lib, solver, args, iters_done, elapsed, n_is)
            next_checkpoint_at = _next_checkpoint(iters_done)

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


def _probe_preflop(bp_lib, solver):
    """Extract preflop strategies and raw regrets for key hands.

    Output format (multi-line):
      UTG: AA:F0/C2/R98 KK:F0/C5/R95 ... 22:F85/C10/R5
      BTN: AA:F0/C1/R99 KK:F0/C2/R98 ... 22:F60/C20/R20
      REGRETS UTG TT: fold=+1234 call=+56789 best_raise=+12345 pruned=3/8
      REGRETS UTG 99: fold=+5678 call=+45678 best_raise=-1234 pruned=5/8
      REGRETS UTG 44: fold=+9012 call=-3456 best_raise=-78901 pruned=6/8

    The REGRETS lines show raw int regrets for the 3 diagnostic hands.
    'pruned=3/8' means 3 of 8 raise actions are below -300M threshold.
    This directly measures whether the pruning fix is working."""
    try:
        PRUNE_THRESH = -300_000_000

        # Pocket pairs + key broadway/suited connectors that showed
        # convergence issues (AKo uniform stuck, A5s/K9s/87s/98s fold locked)
        pairs = [
            ("AA", 0), ("KK", 25), ("QQ", 48), ("JJ", 69), ("TT", 88),
            ("99", 105), ("88", 120), ("77", 133), ("66", 144), ("55", 153),
            ("44", 160), ("33", 165), ("22", 168),
            ("AKo", 2), ("AQs", 3), ("A5s", 17), ("K9s", 32),
            ("98s", 106), ("87s", 121),
        ]
        # Diagnostic hands: raw regrets for call trap / fold lock-in / uniform stuck
        diag_hands = [("TT", 88), ("99", 105), ("44", 160), ("AKo", 2), ("87s", 121)]

        # Positions with their "folds to me" action sequences.
        # Preflop order: UTG(2), MP(3), CO(4), BTN(5), SB(0), BB(1).
        # Action index 0 = fold. Each position's root = all prior players folded.
        positions = [
            ("UTG", 2, []),              # first to act
            ("BTN", 5, [0, 0, 0]),       # UTG, MP, CO folded
        ]

        strat_buf = (ctypes.c_float * 16)()
        regret_buf = (ctypes.c_int * 16)()
        empty_board = (ctypes.c_int * 1)(0)

        lines = []

        # Strategy lines for each position
        for pos_name, player, action_seq in positions:
            if action_seq:
                c_seq = (ctypes.c_int * len(action_seq))(*action_seq)
                seq_len = len(action_seq)
            else:
                c_seq = (ctypes.c_int * 1)(0)
                seq_len = 0
            parts = []
            for name, bucket in pairs:
                na = bp_lib.bp_get_strategy(solver, player, empty_board, 0,
                                             c_seq, seq_len, strat_buf, bucket)
                if na >= 3:
                    f_p = strat_buf[0] * 100
                    c_p = strat_buf[1] * 100
                    r_p = sum(strat_buf[a] for a in range(2, na)) * 100
                    parts.append(f"{name}:F{f_p:.0f}/C{c_p:.0f}/R{r_p:.0f}")
                else:
                    parts.append(f"{name}:?")
            lines.append(f"{pos_name}: {' '.join(parts)}")

        # Raw regret lines for diagnostic hands (UTG only)
        utg_regret_seq = (ctypes.c_int * 1)(0)
        for name, bucket in diag_hands:
            na = bp_lib.bp_get_regrets(solver, 2, empty_board, 0,
                                        utg_regret_seq, 0, regret_buf, bucket)
            if na >= 3:
                fold_r = regret_buf[0]
                call_r = regret_buf[1]
                raise_regrets = [regret_buf[a] for a in range(2, na)]
                best_raise = max(raise_regrets)
                n_pruned = sum(1 for r in raise_regrets if r < PRUNE_THRESH)
                n_raises = len(raise_regrets)
                lines.append(
                    f"REGRETS UTG {name}: fold={fold_r:+d} call={call_r:+d} "
                    f"best_raise={best_raise:+d} pruned={n_pruned}/{n_raises}"
                )
            else:
                lines.append(f"REGRETS UTG {name}: not_found")

        return "\n".join(lines)
    except Exception as e:
        return f"probe_error:{e}"


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
        "preflop_tiers": {str(k): v for k, v in PREFLOP_TIERS.items()},
        "preflop_max_raises": PREFLOP_MAX_RAISES,
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
    # Only export .bps on final checkpoint — intermediate exports are too
    # expensive (300M+ info sets × LZMA compress = minutes per checkpoint).
    # The regret checkpoint is sufficient for resume; .bps is only needed
    # for the runtime solver after training completes.
    bps_path = os.path.join(args.output_dir, "unified_blueprint.bps")
    regret_mb = os.path.getsize(regret_path) / 1024 / 1024

    if label == "final":
        needed = ctypes.c_size_t(0)
        bp_lib.bp_export_strategies(solver, None, 0, ctypes.byref(needed))
        strat_size = needed.value

        if strat_size > 0:
            strat_buf = (ctypes.c_char * strat_size)()
            written = ctypes.c_size_t(0)
            bp_lib.bp_export_strategies(solver, strat_buf, strat_size,
                                        ctypes.byref(written))
            strat_data = bytes(strat_buf[:written.value])
            compressed = lzma.compress(strat_data, preset=1)
            meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')

            out_tmp = bps_path + ".tmp"
            with open(out_tmp, 'wb') as f:
                f.write(b'BPS3')  # v3: uint64 size field for >4GB strategies
                f.write(struct.pack('<QI', len(compressed), len(meta_bytes)))
                f.write(compressed)
                f.write(meta_bytes)
            os.replace(out_tmp, bps_path)

            comp_mb = len(compressed) / 1024 / 1024
            print(f"  Final checkpoint: strategy={comp_mb:.0f}MB, regrets={regret_mb:.0f}MB")
        else:
            print("  [WARN] No strategies to export")
    else:
        print(f"  Checkpoint: regrets={regret_mb:.0f}MB (strategy export deferred to final)")

    # ── Upload to S3 (with retries) ──
    if args.s3_bucket:
        # Always upload latest (for resume)
        uploads = [
            (regret_path, "checkpoints/regrets_latest.bin"),
            (meta_path, "checkpoint_meta.json"),
        ]
        # Also save a timestamped copy of every checkpoint for later analysis.
        # Use "XB" for exact billions, "XM" otherwise (avoids collisions
        # where e.g. 1.5B and 1B both mapped to "1B" via integer division).
        if iters_done >= 1_000_000_000 and iters_done % 1_000_000_000 == 0:
            iter_tag = f"{iters_done // 1_000_000_000}B"
        else:
            iter_tag = f"{iters_done // 1_000_000}M"
        uploads.append((regret_path, f"checkpoints/regrets_{iter_tag}.bin"))
        uploads.append((meta_path, f"checkpoints/meta_{iter_tag}.json"))

        if label == "final" and os.path.exists(bps_path):
            uploads.append((bps_path, "unified_blueprint.bps"))

        # ── Run extract_roots for convergence monitoring ──
        extract_bin = os.path.join(args.build_dir, "extract_roots")
        summary_path = None
        if os.path.exists(extract_bin):
            summary_path = os.path.join(args.output_dir, f"strategy_{iter_tag}.txt")
            try:
                result = subprocess.run(
                    [extract_bin, regret_path],
                    capture_output=True, text=True, timeout=1800
                )
                with open(summary_path, 'w') as sf:
                    sf.write(result.stdout)
                # Print a compact UTG summary to the log
                for line in result.stdout.splitlines():
                    if "Summary:" in line or "SANITY" in line or "CHECK:" in line or "OK:" in line:
                        print(f"  {line.strip()}")
            except Exception as e:
                print(f"  [WARN] extract_roots failed: {e}")
                summary_path = None

        # Upload small files FIRST (summary + meta) so results are visible
        # immediately, then upload the large regret file once + server-side copy.
        small_uploads = [(meta_path, "checkpoint_meta.json"),
                         (meta_path, f"checkpoints/meta_{iter_tag}.json")]
        if summary_path and os.path.exists(summary_path):
            small_uploads.append((summary_path, f"summaries/strategy_{iter_tag}.txt"))
        if label == "final" and os.path.exists(bps_path):
            small_uploads.append((bps_path, "unified_blueprint.bps"))

        for local, s3key in small_uploads:
            if os.path.exists(local):
                try:
                    _s3_upload_with_retry(
                        local, f"s3://{args.s3_bucket}/{s3key}")
                except subprocess.CalledProcessError:
                    pass
        print(f"  Summary uploaded to s3://{args.s3_bucket}/summaries/ ({iter_tag})")

        # Upload regret file ONCE as latest, then server-side copy for timestamped.
        # Saves uploading 76GB twice (was ~15-20 min overhead per checkpoint).
        try:
            _s3_upload_with_retry(
                regret_path, f"s3://{args.s3_bucket}/checkpoints/regrets_latest.bin")
            subprocess.run(
                ["aws", "s3", "cp",
                 f"s3://{args.s3_bucket}/checkpoints/regrets_latest.bin",
                 f"s3://{args.s3_bucket}/checkpoints/regrets_{iter_tag}.bin",
                 "--quiet"],
                check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"  [WARN] regret upload/copy failed: {e}")
        print(f"  Regrets uploaded to s3://{args.s3_bucket}/ ({iter_tag})")


if __name__ == "__main__":
    main()
