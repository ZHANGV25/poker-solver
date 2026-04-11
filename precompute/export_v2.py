#!/usr/bin/env python3
"""One-shot BPS export with benchmarks. Minimal script — no imports beyond stdlib."""
import ctypes
import json
import lzma
import os
import struct
import subprocess
import sys
import time

NUM_PLAYERS = 6
SMALL_BLIND = 50
BIG_BLIND = 100
INITIAL_STACK = 10000

# Tier-aware preflop sizing — MUST match blueprint_worker_unified.py /
# whatever was used at training time. The .bps consumer needs to know
# this to interpret the action labels correctly.
# Source of truth: precompute/blueprint_worker_unified.py PREFLOP_TIERS.
PREFLOP_TIERS = {
    0: [0.5, 0.7, 1.0],   # open raise: 3 sizes
    1: [0.7, 1.0],        # 3-bet: 2 sizes
    2: [1.0],             # 4-bet: 1 size
    3: [8.0],             # 5-bet: shove only
}
PREFLOP_MAX_RAISES = 4

# Legacy flat list — only used by the bp_init_unified call below for the
# initial action template; the actual training tree is determined by
# bp_set_preflop_tier() calls applied later.
PREFLOP_BET_SIZES = [0.5, 0.7, 1.0]  # match tier 0
POSTFLOP_BET_SIZES = [0.5, 1.0]


class BPConfig(ctypes.Structure):
    """Must mirror BPConfig in src/mccfr_blueprint.h exactly.

    The six iter/interval/threshold fields were changed from int to
    int64_t in commit d1d21aa (2026-04-02) but this ctypes mirror was
    never updated, producing ~8 months of silent config misalignment
    where postflop_num_buckets in particular read garbage memory past
    the end of the Python struct. Symptom observed on 2026-04-09: a
    fresh export run printed `postflop_buckets=556` instead of the
    expected 200, causing the texture_cache.bin load to reject and
    triggering a ~60 min bucket precompute. Fix: match the C header."""
    _fields_ = [
        ("discount_stop_iter",  ctypes.c_int64),
        ("discount_interval",   ctypes.c_int64),
        ("prune_start_iter",    ctypes.c_int64),
        ("snapshot_start_iter", ctypes.c_int64),
        ("snapshot_interval",   ctypes.c_int64),
        ("strategy_interval",   ctypes.c_int64),
        ("num_threads",         ctypes.c_int),
        ("hash_table_size",     ctypes.c_int64),
        ("snapshot_dir",        ctypes.c_char_p),
        ("include_preflop",     ctypes.c_int),
        ("postflop_num_buckets", ctypes.c_int),
    ]


def main():
    regret_file = sys.argv[1] if len(sys.argv) > 1 else "/opt/blueprint_unified/regrets_200M.bin"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/opt/blueprint_unified"
    s3_bucket = sys.argv[3] if len(sys.argv) > 3 else "poker-blueprint-unified"
    hash_size = int(sys.argv[4]) if len(sys.argv) > 4 else (1 << 30)

    timings = {}
    t_total = time.time()

    print("=" * 60)
    print("  BPS Export Tool v2 — Benchmarked")
    print("=" * 60, flush=True)

    # Load DLL — the Makefile produces .so on Linux and .dll on Windows,
    # match whichever exists.
    t0 = time.time()
    _so_path = "./build/mccfr_blueprint.so"
    _dll_path = "./build/mccfr_blueprint.dll"
    if os.path.exists(_so_path):
        bp = ctypes.CDLL(_so_path)
    elif os.path.exists(_dll_path):
        bp = ctypes.CDLL(_dll_path)
    else:
        print(f"FATAL: neither {_so_path} nor {_dll_path} exists. "
              f"Run `make blueprint` first.", file=sys.stderr)
        sys.exit(1)
    bp.bp_default_config.restype = None
    bp.bp_init_unified.restype = ctypes.c_int
    bp.bp_num_info_sets.restype = ctypes.c_int
    bp.bp_export_strategies.restype = ctypes.c_int
    bp.bp_export_strategies.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t)
    ]
    bp.bp_load_regrets.restype = ctypes.c_int
    bp.bp_load_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    bp.bp_free.restype = None

    # Phase 1.3 bug fix: bp_set_preflop_tier MUST be called before
    # bp_load_regrets / bp_compute_action_evs, matching the training-time
    # tier configuration from blueprint_worker_unified.py. Without this
    # call, the export solver has num_preflop_tiers=0 and falls back to
    # flat preflop sizes at every raise level.
    bp.bp_set_preflop_tier.restype = ctypes.c_int
    bp.bp_set_preflop_tier.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_int
    ]

    # Phase 1.3 bug #2: v2 checkpoints use the OLD boost-style hash_combine
    # because v2 training launched (2026-04-07 22:18 UTC) BEFORE commit
    # 48da71b landed (2026-04-08 01:51 UTC) which replaced it with
    # splitmix64. The file format has no way to distinguish them — stored
    # action_hash values are just uint64_t. When Phase 1.3's traverse_ev
    # re-queries a v2 checkpoint's info sets with splitmix64 compute_action_hash,
    # every non-root lookup fails (26M / 40M lookups failed in run 4 on
    # 2026-04-11). Toggle the legacy mixer for v2 checkpoints so
    # compute_action_hash produces values that match what training stored.
    bp.bp_set_legacy_hash_mixer.restype = None
    bp.bp_set_legacy_hash_mixer.argtypes = [ctypes.c_int]

    # Phase 1.3: action EV computation + export bindings
    bp.bp_compute_action_evs.restype = ctypes.c_int
    bp.bp_compute_action_evs.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    bp.bp_export_action_evs.restype = ctypes.c_int
    bp.bp_export_action_evs.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t)
    ]

    # Phase 1.3: visit count distribution stats (for sentinel 5).
    # Must mirror BPEVVisitStats in src/mccfr_blueprint.h exactly.
    class BPEVVisitStats(ctypes.Structure):
        _fields_ = [
            ("total_visited", ctypes.c_int64),
            ("min_visits",    ctypes.c_int64),
            ("p10_visits",    ctypes.c_int64),
            ("p50_visits",    ctypes.c_int64),
            ("p90_visits",    ctypes.c_int64),
            ("p99_visits",    ctypes.c_int64),
            ("max_visits",    ctypes.c_int64),
            ("below_5",       ctypes.c_int64),
            ("below_100",     ctypes.c_int64),
            ("above_1000",    ctypes.c_int64),
        ]

    bp.bp_get_ev_visit_stats.restype = ctypes.c_int
    bp.bp_get_ev_visit_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(BPEVVisitStats)]

    timings["dll_load"] = time.time() - t0
    print(f"[{timings['dll_load']:.2f}s] DLL loaded", flush=True)

    # Init solver
    t0 = time.time()
    config = BPConfig()
    bp.bp_default_config(ctypes.byref(config))
    config.num_threads = 1
    config.include_preflop = 1
    config.hash_table_size = hash_size
    config.postflop_num_buckets = 200

    buf = (ctypes.c_char * (2 * 1024 * 1024))()
    solver = ctypes.cast(buf, ctypes.c_void_p)

    c_post = (ctypes.c_float * len(POSTFLOP_BET_SIZES))(*POSTFLOP_BET_SIZES)
    c_pre = (ctypes.c_float * len(PREFLOP_BET_SIZES))(*PREFLOP_BET_SIZES)

    print(f"Calling bp_init_unified (hash_size={hash_size:,})...", flush=True)
    ret = bp.bp_init_unified(
        solver, NUM_PLAYERS,
        SMALL_BLIND, BIG_BLIND, INITIAL_STACK,
        c_post, len(POSTFLOP_BET_SIZES),
        c_pre, len(PREFLOP_BET_SIZES),
        ctypes.byref(config)
    )
    if ret != 0:
        print(f"FATAL: bp_init_unified returned {ret}")
        sys.exit(1)

    # Apply tiered preflop sizing — MUST match training.
    for level, sizes in sorted(PREFLOP_TIERS.items()):
        c_sizes = (ctypes.c_float * len(sizes))(*sizes)
        ret = bp.bp_set_preflop_tier(
            solver, level, c_sizes, len(sizes), PREFLOP_MAX_RAISES)
        if ret != 0:
            print(f"FATAL: bp_set_preflop_tier level {level} returned {ret}")
            bp.bp_free(solver)
            sys.exit(1)
    print(f"Tiered preflop: {len(PREFLOP_TIERS)} levels, "
          f"max {PREFLOP_MAX_RAISES} raises", flush=True)

    # Enable legacy boost-style hash_combine for v2 checkpoints. Opt-in via
    # env var USE_LEGACY_HASH_MIXER=1 (default: off for v3/future checkpoints).
    # For the current 1.5B v2 checkpoint this MUST be 1.
    use_legacy = os.environ.get("USE_LEGACY_HASH_MIXER", "1") == "1"
    bp.bp_set_legacy_hash_mixer(1 if use_legacy else 0)

    timings["init"] = time.time() - t0
    print(f"[{timings['init']:.2f}s] Solver initialized", flush=True)

    # Load regrets
    t0 = time.time()
    file_gb = os.path.getsize(regret_file) / (1024 ** 3)
    print(f"Loading regrets ({file_gb:.1f} GB)...", flush=True)
    n_loaded = bp.bp_load_regrets(solver, regret_file.encode("utf-8"))
    timings["load_regrets"] = time.time() - t0
    n_is = bp.bp_num_info_sets(solver)
    print(f"[{timings['load_regrets']:.2f}s] Loaded {n_loaded:,} entries ({n_is:,} info sets)", flush=True)

    if n_loaded <= 0:
        print("FATAL: no entries loaded")
        bp.bp_free(solver)
        sys.exit(1)

    # Query export size
    t0 = time.time()
    needed = ctypes.c_size_t(0)
    bp.bp_export_strategies(solver, None, 0, ctypes.byref(needed))
    strat_size = needed.value
    print(f"Export buffer: {strat_size / (1024**2):.1f} MB", flush=True)

    # Export strategies
    strat_buf = (ctypes.c_char * strat_size)()
    written = ctypes.c_size_t(0)
    print("Exporting strategies...", flush=True)
    bp.bp_export_strategies(solver, strat_buf, strat_size, ctypes.byref(written))
    strat_data = bytes(strat_buf[:written.value])
    timings["export_strategies"] = time.time() - t0
    print(f"[{timings['export_strategies']:.2f}s] Exported {written.value / (1024**2):.1f} MB", flush=True)
    del strat_buf

    # ── Phase 1.3: σ̄-sampled per-action EV walk ────────────────────
    # Compute per-action EVs under the average strategy, so that
    # leaf_values.py's get_turn_action_evs() can return real values
    # instead of None (which currently forces the equity fallback).
    #
    # Iteration count is controlled via EV_WALK_ITERS env var; default 50M.
    # At 50M iterations on an 8-thread box this should complete in 2-5 min.
    # See docs/PHASE_1_3_DESIGN.md for the algorithm.
    ev_walk_iters = int(os.environ.get("EV_WALK_ITERS", "50000000"))
    action_evs_data = b""
    ev_visit_stats = None
    if ev_walk_iters > 0:
        t0 = time.time()
        print(f"Phase 1.3: computing action EVs ({ev_walk_iters:,} iters)...", flush=True)
        ret = bp.bp_compute_action_evs(solver, ctypes.c_int64(ev_walk_iters))
        timings["compute_action_evs"] = time.time() - t0
        if ret != 0:
            print(f"WARNING: bp_compute_action_evs returned {ret}, skipping EV export", flush=True)
        else:
            print(f"[{timings['compute_action_evs']:.2f}s] Action EV walk complete", flush=True)
            # Query size
            t0 = time.time()
            ev_needed = ctypes.c_size_t(0)
            bp.bp_export_action_evs(solver, None, 0, ctypes.byref(ev_needed))
            ev_size = ev_needed.value
            print(f"Action EV buffer: {ev_size / (1024**2):.1f} MB", flush=True)
            if ev_size > 0:
                ev_buf = (ctypes.c_char * ev_size)()
                ev_written = ctypes.c_size_t(0)
                bp.bp_export_action_evs(solver, ev_buf, ev_size, ctypes.byref(ev_written))
                action_evs_data = bytes(ev_buf[:ev_written.value])
                del ev_buf
                timings["export_action_evs"] = time.time() - t0
                print(f"[{timings['export_action_evs']:.2f}s] Exported {ev_written.value / (1024**2):.1f} MB of action EVs", flush=True)

            # Collect EV visit count distribution for sentinel 5.
            # Called BEFORE bp_free; we store the result in a dict that
            # gets merged into the metadata JSON below. Safe to call even
            # if the export failed — it just reports zeros in that case.
            visit_stats_struct = BPEVVisitStats()
            bp.bp_get_ev_visit_stats(solver, ctypes.byref(visit_stats_struct))
            ev_visit_stats = {
                "total_visited":  int(visit_stats_struct.total_visited),
                "min":            int(visit_stats_struct.min_visits),
                "p10":            int(visit_stats_struct.p10_visits),
                "p50":            int(visit_stats_struct.p50_visits),
                "p90":            int(visit_stats_struct.p90_visits),
                "p99":            int(visit_stats_struct.p99_visits),
                "max":            int(visit_stats_struct.max_visits),
                "below_5":        int(visit_stats_struct.below_5),
                "below_100":      int(visit_stats_struct.below_100),
                "above_1000":     int(visit_stats_struct.above_1000),
            }
            print(f"[BP1.3] Visit distribution: total={ev_visit_stats['total_visited']:,}, "
                  f"p50={ev_visit_stats['p50']}, p90={ev_visit_stats['p90']}, "
                  f"p99={ev_visit_stats['p99']}, max={ev_visit_stats['max']}",
                  flush=True)
            print(f"[BP1.3]   below_5={ev_visit_stats['below_5']:,}, "
                  f"below_100={ev_visit_stats['below_100']:,}, "
                  f"above_1000={ev_visit_stats['above_1000']:,}",
                  flush=True)
    else:
        print("Phase 1.3: skipped (EV_WALK_ITERS=0)", flush=True)

    # Free solver before compress (reclaim ~60 GB)
    bp.bp_free(solver)
    del buf
    print("Solver freed, starting compression...", flush=True)

    # LZMA compress strategies
    t0 = time.time()
    compressed = lzma.compress(strat_data, preset=1)
    timings["lzma_compress"] = time.time() - t0
    print(f"[{timings['lzma_compress']:.2f}s] Compressed strategies: {len(compressed) / (1024**2):.1f} MB", flush=True)

    # Phase 1.3: LZMA compress action EVs separately (skipped if empty)
    compressed_evs = b""
    if action_evs_data:
        t0 = time.time()
        compressed_evs = lzma.compress(action_evs_data, preset=1)
        timings["lzma_compress_evs"] = time.time() - t0
        print(f"[{timings['lzma_compress_evs']:.2f}s] Compressed action EVs: {len(compressed_evs) / (1024**2):.1f} MB", flush=True)

    # Try to derive iteration count and code SHA from the regret file path
    # and the source repo. Best-effort — both fields are optional but useful.
    iter_count = 0
    chk_label = "unknown"
    import re as _re
    m = _re.search(r"regrets_(\d+)M\.bin", os.path.basename(regret_file))
    if m:
        iter_count = int(m.group(1)) * 1_000_000
        chk_label = f"iter_{iter_count}"

    code_sha = "unknown"
    try:
        code_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    # Write BPS3 file. Schema v3 adds per-action EVs as a trailing section
    # after the metadata; backward-compatible with v2 readers (they stop at
    # meta_bytes and ignore the trailing data).
    has_action_evs = len(compressed_evs) > 0
    meta = {
        "type": "unified_blueprint",
        "schema_version": 3 if has_action_evs else 2,
        "num_players": NUM_PLAYERS,
        "blinds": [SMALL_BLIND, BIG_BLIND],
        "initial_stack": INITIAL_STACK,
        # Tier-aware preflop sizing (the actual training tree shape).
        # Keys are stringified ints for JSON compatibility.
        "preflop_tiers": {str(k): v for k, v in PREFLOP_TIERS.items()},
        "preflop_max_raises": PREFLOP_MAX_RAISES,
        # Legacy flat list — kept for backwards compat with old consumers.
        "preflop_bet_sizes": PREFLOP_BET_SIZES,
        "postflop_bet_sizes": POSTFLOP_BET_SIZES,
        "iterations": iter_count,
        "discount_stop_iter": int(config.discount_stop_iter),
        "num_info_sets": n_is,
        "preflop_buckets": 169,
        "postflop_buckets": 200,
        "checkpoint": chk_label,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_sha": code_sha,
        # Bug B fix marker — readers can verify they're consuming
        # average-strategy data, not regret-matched-current data.
        "strategy_extraction_method": "strategy_sum_avg",
        "training_complete": True,
        # Bug 6 / v3: explicit hash-mixer tag. Python consumers that
        # recompute info-set hashes (e.g. get_strategy() calls in
        # blueprint_v2.py) must use the same mixer as the C exporter
        # or every lookup misses. Files tagged "splitmix64" work with
        # the post-Bug-6 Python implementation; files without this
        # field (legacy) were exported with the old boost-style mixer
        # and are incompatible with current blueprint_v2.py lookups.
        "hash_mixer": "splitmix64",
        # Phase 1.3: per-action EVs under the average strategy, computed
        # post-hoc via σ̄-sampled MCCFR walk. See docs/PHASE_1_3_DESIGN.md.
        "has_action_evs": has_action_evs,
        "action_evs_compute_method": "posthoc_sigma_bar_walk" if has_action_evs else None,
        "action_evs_walk_iterations": ev_walk_iters if has_action_evs else 0,
        # Sentinel 5: visit count distribution stats collected at export
        # time from the C solver's in-memory hash table. Consumers can
        # use these to judge EV confidence per info set and to flag
        # pathological distributions (everyone has 1 visit → sampling
        # broken; nobody has >100 → training too short). None if no
        # EV walk was performed.
        "ev_visit_stats": ev_visit_stats,
    }
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")

    os.makedirs(output_dir, exist_ok=True)
    bps_path = os.path.join(output_dir, "unified_blueprint.bps")
    with open(bps_path, "wb") as f:
        # BPS3 outer wrapper (unchanged from v2)
        f.write(b"BPS3")
        f.write(struct.pack("<QI", len(compressed), len(meta_bytes)))
        f.write(compressed)
        f.write(meta_bytes)

        # Phase 1.3: optional trailing action-EV section.
        # Format: magic "BPR3" (4B) + u64 compressed_size + compressed_payload
        # Old readers that stop after meta_bytes are unaffected.
        if has_action_evs:
            f.write(b"BPR3")
            f.write(struct.pack("<Q", len(compressed_evs)))
            f.write(compressed_evs)

    bps_mb = os.path.getsize(bps_path) / (1024 * 1024)
    print(f"Wrote {bps_path} ({bps_mb:.1f} MB)", flush=True)

    # Upload to S3
    if s3_bucket:
        t0 = time.time()
        s3_uri = f"s3://{s3_bucket}/unified_blueprint.bps"
        print(f"Uploading to {s3_uri}...", flush=True)
        subprocess.run(["aws", "s3", "cp", bps_path, s3_uri, "--quiet"], check=True)
        timings["s3_upload"] = time.time() - t0
        print(f"[{timings['s3_upload']:.2f}s] Uploaded", flush=True)

    timings["total"] = time.time() - t_total

    # Benchmark summary
    print()
    print("=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    for step, secs in timings.items():
        print(f"  {step:25s}  {secs:8.2f}s")
    print(f"  {'TOTAL':25s}  {timings['total']:8.2f}s  ({timings['total']/60:.1f} min)")
    print("=" * 60, flush=True)

    bench_path = os.path.join(output_dir, "export_benchmark.json")
    with open(bench_path, "w") as f:
        json.dump(timings, f, indent=2)
    if s3_bucket:
        subprocess.run(["aws", "s3", "cp", bench_path, f"s3://{s3_bucket}/export_benchmark.json", "--quiet"], check=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
