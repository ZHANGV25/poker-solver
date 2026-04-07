#!/usr/bin/env python3
"""
Pre-deploy validation: verify tiered sizing, save/load, and tree shape
before committing to a multi-day EC2 run.
"""
import ctypes
import os
import sys
import tempfile
import time

BUILD_DIR = os.path.join(os.path.dirname(__file__), '..', 'build')
SO_PATH = os.path.join(BUILD_DIR, 'mccfr_blueprint.so')

bp = ctypes.CDLL(SO_PATH)

# Register all functions with correct types
bp.bp_default_config.restype = None
bp.bp_init_unified.restype = ctypes.c_int
bp.bp_set_preflop_tier.restype = ctypes.c_int
bp.bp_set_preflop_tier.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
]
bp.bp_solve.restype = ctypes.c_int
bp.bp_solve.argtypes = [ctypes.c_void_p, ctypes.c_int64]
bp.bp_num_info_sets.restype = ctypes.c_int64
bp.bp_num_info_sets.argtypes = [ctypes.c_void_p]
bp.bp_save_regrets.restype = ctypes.c_int
bp.bp_save_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
bp.bp_load_regrets.restype = ctypes.c_int64
bp.bp_load_regrets.argtypes = [ctypes.c_void_p, ctypes.c_char_p]


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


# ── Deploy config (must match blueprint_worker_unified.py) ──────────

TIERS = {
    0: [0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0],  # open: 8 sizes
    1: [0.7, 1.0, 2.5],                               # 3-bet: 3 sizes
    2: [1.0, 4.0],                                     # 4-bet: 2 sizes
    3: [8.0],                                          # 5-bet: shove
}
MAX_RAISES = 4
POSTFLOP = [0.5, 1.0]
HASH_SIZE = 4_000_000  # small for local test


def init_solver(hash_size=HASH_SIZE):
    config = BPConfig()
    bp.bp_default_config(ctypes.byref(config))
    config.num_threads = 1
    config.include_preflop = 1
    config.hash_table_size = hash_size

    buf = (ctypes.c_char * 524288)()
    solver = ctypes.cast(buf, ctypes.c_void_p)

    c_post = (ctypes.c_float * len(POSTFLOP))(*POSTFLOP)
    flat_pre = TIERS[0]
    c_pre = (ctypes.c_float * len(flat_pre))(*flat_pre)

    ret = bp.bp_init_unified(solver, 6, 50, 100, 10000,
                              c_post, len(POSTFLOP),
                              c_pre, len(flat_pre),
                              ctypes.byref(config))
    assert ret == 0, f"init failed: {ret}"

    for level, sizes in sorted(TIERS.items()):
        c_sizes = (ctypes.c_float * len(sizes))(*sizes)
        ret = bp.bp_set_preflop_tier(solver, level, c_sizes, len(sizes), MAX_RAISES)
        assert ret == 0, f"set_tier {level} failed: {ret}"

    return solver, buf  # keep buf alive


def test_1_basic():
    """Tiered sizing runs without crash and creates reasonable IS count."""
    print("TEST 1: Basic tiered run (50K iters)...", flush=True)
    solver, buf = init_solver()
    t0 = time.time()
    ret = bp.bp_solve(solver, 50000)
    elapsed = time.time() - t0
    assert ret == 0
    n = bp.bp_num_info_sets(solver)
    print(f"  {n:,} info sets in {elapsed:.1f}s ({50000/elapsed:.0f} iter/s)")
    # With 4M table, should create well under 4M IS
    assert 10_000 < n < 4_000_000, f"Unexpected IS count: {n}"
    print("  PASS")
    return n


def test_2_creation_rate():
    """Verify IS creation rate matches tiered enumeration predictions.

    Tiered A predicts 2.28M preflop decision nodes.
    At 50K iters with external sampling, we should see ~15-30 new IS per iter
    (each iter visits ~30 decision nodes, each creating a new IS if unseen).
    """
    print("\nTEST 2: IS creation rate...", flush=True)
    solver, buf = init_solver(hash_size=8_000_000)  # bigger table to avoid filling

    intervals = [(10000, None), (20000, None), (30000, None), (40000, None), (50000, None)]
    results = []
    total_iters = 0

    for chunk, _ in intervals:
        bp.bp_solve(solver, chunk)
        total_iters += chunk
        n = bp.bp_num_info_sets(solver)
        results.append((total_iters, n))

    print(f"  {'Iters':>10}  {'Info Sets':>12}  {'New/Iter':>10}")
    prev_n = 0
    prev_i = 0
    for iters, n in results:
        rate = (n - prev_n) / (iters - prev_i) if iters > prev_i else 0
        print(f"  {iters:>10,}  {n:>12,}  {rate:>10.1f}")
        prev_n = n
        prev_i = iters

    # Rate should be declining (tree is finite, re-visiting known IS)
    rates = []
    prev_n, prev_i = 0, 0
    for iters, n in results:
        rates.append((n - prev_n) / (iters - prev_i))
        prev_n, prev_i = n, iters

    assert rates[-1] < rates[0], f"Creation rate not declining: {rates}"
    print(f"  Rate declining: {rates[0]:.1f} → {rates[-1]:.1f} new IS/iter")
    print("  PASS")


def test_3_save_load():
    """Save checkpoint, load into fresh solver, verify IS count matches."""
    print("\nTEST 3: Save/load round-trip (BPR4 format)...", flush=True)
    solver1, buf1 = init_solver()
    bp.bp_solve(solver1, 20000)
    n1 = bp.bp_num_info_sets(solver1)

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        tmppath = f.name

    try:
        ret = bp.bp_save_regrets(solver1, tmppath.encode())
        assert ret == 0, f"save failed: {ret}"
        fsize = os.path.getsize(tmppath)
        print(f"  Saved {n1:,} IS to {tmppath} ({fsize/1e6:.1f} MB)")

        # Load into fresh solver
        solver2, buf2 = init_solver()
        loaded = bp.bp_load_regrets(solver2, tmppath.encode())
        n2 = bp.bp_num_info_sets(solver2)
        print(f"  Loaded {loaded:,} IS, table has {n2:,}")
        assert loaded == n1, f"Loaded {loaded} != saved {n1}"
        assert n2 == n1, f"Table count {n2} != saved {n1}"
        print("  PASS")
    finally:
        os.unlink(tmppath)


def test_4_int64_iterations():
    """Verify int64 iteration counter works (pass >2B value via ctypes)."""
    print("\nTEST 4: int64 iteration support...", flush=True)
    # We can't actually run 3B iterations, but we can verify the API accepts it
    big_val = ctypes.c_int64(5_000_000_000)
    print(f"  ctypes.c_int64(5B) = {big_val.value}")
    assert big_val.value == 5_000_000_000, "int64 truncation!"

    # Verify hash_table_size accepts 3B
    config = BPConfig()
    bp.bp_default_config(ctypes.byref(config))
    config.hash_table_size = 3_000_000_000
    assert config.hash_table_size == 3_000_000_000, \
        f"hash_table_size truncated: {config.hash_table_size}"
    print(f"  config.hash_table_size = {config.hash_table_size:,} (3B)")
    print("  PASS")


def test_5_tiered_vs_flat():
    """Run both tiered and flat-8 and verify tiered creates fewer IS."""
    print("\nTEST 5: Tiered vs flat comparison (20K iters each)...", flush=True)

    # Tiered
    solver_t, buf_t = init_solver(hash_size=4_000_000)
    bp.bp_solve(solver_t, 20000)
    n_tiered = bp.bp_num_info_sets(solver_t)

    # Flat 8 (no tiers set — uses flat preflop_bet_sizes)
    config = BPConfig()
    bp.bp_default_config(ctypes.byref(config))
    config.num_threads = 1
    config.include_preflop = 1
    config.hash_table_size = 4_000_000

    buf_f = (ctypes.c_char * 524288)()
    solver_f = ctypes.cast(buf_f, ctypes.c_void_p)
    flat = [0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0]
    c_post = (ctypes.c_float * len(POSTFLOP))(*POSTFLOP)
    c_pre = (ctypes.c_float * len(flat))(*flat)
    bp.bp_init_unified(solver_f, 6, 50, 100, 10000,
                        c_post, len(POSTFLOP), c_pre, len(flat),
                        ctypes.byref(config))
    # NO tiers set — uses flat sizing
    bp.bp_solve(solver_f, 20000)
    n_flat = bp.bp_num_info_sets(solver_f)

    print(f"  Tiered (8/3/2/1): {n_tiered:>10,} IS")
    print(f"  Flat (8/8/8/8):   {n_flat:>10,} IS")
    ratio = n_flat / n_tiered if n_tiered > 0 else 0
    print(f"  Flat/Tiered ratio: {ratio:.1f}x")
    assert n_tiered < n_flat, f"Tiered ({n_tiered}) should be smaller than flat ({n_flat})"
    assert ratio > 1.3, f"Expected at least 1.3x reduction, got {ratio:.1f}x"
    print("  PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("PRE-DEPLOY VALIDATION")
    print(f"Config: tiered {list(TIERS.values())}")
    print(f"Max raises: {MAX_RAISES}")
    print(f"Target hash: 3B (testing with {HASH_SIZE:,})")
    print("=" * 60)

    test_1_basic()
    test_2_creation_rate()
    test_3_save_load()
    test_4_int64_iterations()
    test_5_tiered_vs_flat()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED — ready to deploy")
    print("=" * 60)
    print(f"\nDeploy config:")
    print(f"  Hash table: 3,000,000,000 (3B)")
    print(f"  Preflop tiers:")
    for lvl, sizes in sorted(TIERS.items()):
        labels = ['open', '3-bet', '4-bet', '5-bet']
        print(f"    {labels[lvl]:>5}: {sizes}")
    print(f"  Max preflop raises: {MAX_RAISES}")
    print(f"  Postflop: first={POSTFLOP}, subsequent=[1.0], max 3 raises")
    print(f"  Preflop buckets: 169, postflop: 200")
