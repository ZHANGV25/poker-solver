#!/usr/bin/env python3
"""Extract ALL preflop info sets from unified_blueprint.bps into a flat action-hash map.

Output format:
{
  "meta": { preflop_bet_sizes, sb, bb, stack, pos_to_player, acting_order, root_hash, seed },
  "nodes": {
    "0xHEX_ACTION_HASH": {
      "player": <int>,                     # which player acts at this node (0-5)
      "actionLabels": ["fold","call","raise_X.X",...,"allin"],
      "strategies": [                       # 169 entries, in canonical hand-grid order
        { "hand": "AA", "actions": [{"action":"raise_3.5","frequency":0.85}, ...] },
        ...
      ]
    }
  }
}

The frontend walks its action path through the nodes by looking up each prefix's
hash, matching user action labels against the node's actionLabels to get an index,
and hashing that index into the next action_hash. This way the data layer needs no
tree structure — just the flat hash → node map.
"""

import base64
import json
import lzma
import os
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np

RANKS = "AKQJT98765432"
POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
POS_TO_PLAYER = {"UTG": 2, "MP": 3, "CO": 4, "BTN": 5, "SB": 0, "BB": 1}
PLAYER_TO_POS = {v: k for k, v in POS_TO_PLAYER.items()}

# Preflop tier system — must exactly match blueprint_worker_unified.py PREFLOP_TIERS.
# The training solver used tiered bet sizes: different raise options at each level
# of raise depth (open → 3-bet → 4-bet → 5-bet). `bp_set_preflop_tier()` in
# mccfr_blueprint.c picks from PREFLOP_TIERS[num_raises] at each node.
#
# Level n = num_raises at that node. Level 0 = unraised (UTG opens).
PREFLOP_TIERS = {
    0: [0.5, 0.7, 1.0],   # open raise: 3 sizes
    1: [0.7, 1.0],         # 3-bet: 2 sizes
    2: [1.0],              # 4-bet: 1 size
    3: [8.0],              # 5-bet: shove (8x pot ≫ stack → coerced to allin)
}
PREFLOP_MAX_RAISES = 4

# Integer chip representation — the C solver uses int chips, not floats.
# We replicate its exact arithmetic so action counts match byte-for-byte.
SMALL_BLIND = 50       # chips  (= 0.5 bb)
BIG_BLIND = 100        # chips  (= 1.0 bb)
INITIAL_STACK = 10000  # chips  (= 100 bb)
CHIPS_PER_BB = 100     # divider for display labels

SB_SIZE = 0.5
BB_SIZE = 1.0
INITIAL_STACK_BB = 100.0
ROOT_HASH_HEX = "0xFEDCBA9876543210"

# Hash seed constant — must match _compute_action_hash in blueprint_v2.py
# and mccfr_blueprint.c (compute_action_hash). Starts the hash chain.
HASH_SEED = 0xFEDCBA9876543210


def generate_hand_grid():
    """Generate the canonical 169-hand grid in the order used by the bucket mapping."""
    hands = []
    for row in range(13):
        for col in range(13):
            r1, r2 = RANKS[row], RANKS[col]
            if row == col:
                hands.append(f"{r1}{r2}")
            elif col < row:
                hands.append(f"{r2}{r1}s")
            else:
                hands.append(f"{r1}{r2}o")
    return hands


def _build_c_bucket_map():
    """Build the same hand-class → bucket index mapping that the C training code
    constructs in mccfr_blueprint.c init_unified() (lines ~1383-1394).

    The C code iterates rank pairs from high to low (Ace = rank 12 down to deuce = rank 0):

        for r0 in 12..0:
            for r1 in r0..0:
                if r0 == r1:        pair → 1 class
                else:               suited then offsuit → 2 classes

    So the bucket order is:
        AA(0), AKs(1), AKo(2), AQs(3), AQo(4), AJs(5), AJo(6), ..., A2s(23), A2o(24),
        KK(25), KQs(26), KQo(27), ..., K2s(47), K2o(48),
        QQ(49), ...
        ...
        22(168)

    The C uses rank values where 0=2 and 12=A (from card_int / 4). Our Python uses
    RANKS = "AKQJT98765432" where A is at string index 0, so we have to invert:
    rank_value = 12 - RANKS.index(char).

    Returns a dict mapping hand string ('AA', 'AKs', 'AKo', ...) → C bucket index.
    """
    bucket_map = {}
    n = 0
    for r0_val in range(12, -1, -1):
        for r1_val in range(r0_val, -1, -1):
            r0_char = RANKS[12 - r0_val]
            r1_char = RANKS[12 - r1_val]
            if r0_val == r1_val:
                bucket_map[r0_char + r1_char] = n
                n += 1
            else:
                # Suited first, then offsuit (per C class_map)
                bucket_map[r0_char + r1_char + "s"] = n
                n += 1
                bucket_map[r0_char + r1_char + "o"] = n
                n += 1
    assert n == 169, f"Expected 169 classes, got {n}"
    return bucket_map


_C_BUCKET_MAP = _build_c_bucket_map()


def hand_class_to_bucket(hand):
    """Map a hand string like 'AA', 'AKs', 'AKo' to its C bucket index (0-168).

    MUST match the bucket numbering in mccfr_blueprint.c init_unified() exactly,
    because the .bps stores strategies indexed by this convention. Using a
    different convention silently scrambles the per-hand strategy mapping
    (every cell in the range grid shows a different hand's strategy than its
    label) — see docs/EXTRACTOR_BUGS.md Bug A.
    """
    return _C_BUCKET_MAP[hand]


def enumerate_actions_chips(to_call_c, pot_c, stack_c, current_committed_c, num_raises, max_raises):
    """Reproduce mccfr_blueprint.c generate_actions() exactly, in INTEGER chips.

    All amounts are INT chips (not bb floats) so we match the C solver's
    truncation semantics byte-for-byte. Labels display in bb (chips / 100).

    Returns a list of (action_type, add_chips, label) tuples.
    - action_type: "fold" / "check" / "call" / "bet" / "allin"
    - add_chips:   exact integer chips added this action (matches C amount field)
    - label:       display string. For raises, encodes TOTAL committed (in bb).

    Picks bet sizes from PREFLOP_TIERS based on num_raises.
    """
    out = []
    if to_call_c > 0:
        out.append(("fold", 0, "fold"))

    if to_call_c > 0:
        out.append(("call", to_call_c, "call"))
    else:
        out.append(("check", 0, "check"))

    if num_raises < max_raises:
        # Pick bet sizes for this tier level
        tier_level = num_raises
        if tier_level >= len(PREFLOP_TIERS):
            tier_level = len(PREFLOP_TIERS) - 1
        bet_sizes = PREFLOP_TIERS[tier_level]

        added_allin = False
        for bs in bet_sizes:
            if to_call_c == 0:
                ba_c = int(bs * pot_c)
            else:
                ba_c = to_call_c + int(bs * (pot_c + to_call_c))
            if ba_c >= stack_c:
                ba_c = stack_c
            if ba_c <= to_call_c:
                continue
            if ba_c >= stack_c:
                if added_allin:
                    continue
                added_allin = True
                out.append(("allin", ba_c, "allin"))
            else:
                total_committed_c = current_committed_c + ba_c
                total_bb = total_committed_c / CHIPS_PER_BB
                out.append(("bet", ba_c, f"raise_{total_bb:.1f}"))
        if not added_allin and stack_c > to_call_c:
            out.append(("allin", stack_c, "allin"))
    return out


def mask64(x):
    return x & 0xFFFFFFFFFFFFFFFF


# ── Hash mixer ──────────────────────────────────────────────────────────
#
# Two mixers exist in the wild for our .bps files:
#
#   "splitmix64" — current C / blueprint_v2.py implementation, post-Bug-6 fix
#                  (commit 48da71b). All exports tagged hash_mixer:"splitmix64"
#                  in metadata. ~5,500 fewer silent slot collisions per 2B-slot
#                  table at saturation.
#
#   "boost"      — pre-Bug-6 boost-style hash_combine. Files exported before
#                  the splitmix64 fix and any tagged with no hash_mixer field.
#                  Still required for the legacy unified_blueprint.bps that
#                  blueprint_data/ holds and that the production preflop JSON
#                  was extracted from on 2026-04-07.
#
# The mixer is selected by `_HASH_MIXER` (set in main() after reading the .bps
# meta). compute_action_hash() must be called AFTER the .bps load so the right
# mixer is in place. Cache files are tagged with the mixer they were built
# with so we don't accidentally cross-pollinate.

_HASH_MIXER = "splitmix64"  # default; main() overrides from .bps meta


def _hash_combine_boost(a, b):
    """Pre-Bug-6 boost-style hash_combine. Used for legacy .bps files only."""
    a = mask64(a)
    b = mask64(b)
    return mask64(a ^ mask64(mask64(b + 0x9E3779B97F4A7C15) + mask64(a << 6) + (a >> 2)))


def _bp_mix64(x):
    """splitmix64 — must match bp_mix64 in mccfr_blueprint.c byte-for-byte.

    Verified by tests/test_hash_sync.c against the C implementation on a
    randomized test vector. Used by all current exports.
    """
    x = mask64(x)
    x ^= x >> 30; x = mask64(x * 0xbf58476d1ce4e5b9)
    x ^= x >> 27; x = mask64(x * 0x94d049bb133111eb)
    x ^= x >> 31
    return x


def _hash_combine_splitmix64(a, b):
    """Post-Bug-6 splitmix64-based hash_combine.

    C side:  return bp_mix64(a ^ bp_mix64(b));
    """
    return _bp_mix64(mask64(a) ^ _bp_mix64(b))


def hash_combine(a, b):
    """Dispatch to the active mixer (set by main() from .bps metadata)."""
    if _HASH_MIXER == "splitmix64":
        return _hash_combine_splitmix64(a, b)
    elif _HASH_MIXER == "boost":
        return _hash_combine_boost(a, b)
    else:
        raise ValueError(f"Unknown hash mixer: {_HASH_MIXER!r}")


def compute_action_hash(action_indices):
    h = HASH_SEED
    for a in action_indices:
        h = hash_combine(h, mask64(a * 17 + 3))
    return h


def _detect_mixer_from_meta(meta):
    """Read the hash_mixer field from .bps metadata.

    Files exported after commit 48da71b set this to "splitmix64". Files
    exported before that field was added (or with the legacy boost mixer)
    are missing the field; treat as "boost".
    """
    return meta.get("hash_mixer", "boost")


def load_bps3_preflop_direct(bps_path):
    """Fast loader: read BPS3, decompress, and extract ONLY preflop info sets.

    Returns: (by_node, by_node_evs, meta) where:
        by_node:     dict[(action_hash:int, player:int)] ->
                     dict[bucket:int -> np.ndarray[na] uint8]
        by_node_evs: dict[(action_hash:int, player:int)] ->
                     dict[bucket:int -> np.ndarray[na] float32]
                     (empty dict if the .bps has no BPR3 trailing section)
        meta:        the JSON metadata dict from the .bps

    This bypasses BlueprintV2._load_bps3 / _load_action_evs_section and skips:
    - All non-preflop (street != 0) entries during parsing
    - Per-bucket renormalization (we keep raw uint8 quantized strategies)
    - The expensive [num_buckets, na] zero-filled fallback array
    - Loading any non-preflop street's action EVs

    For an unknown but typical BPS3 (~6M entries), this is ~3-5x faster than
    the BlueprintV2 path because the inner loop is shorter and we never build
    fallback arrays for unused streets.
    """
    by_node_evs = {}
    with open(bps_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'BPS3':
            raise ValueError(f"Not a BPS3 file (magic={magic!r})")
        compressed_size, meta_size = struct.unpack('<QI', f.read(12))
        compressed = f.read(compressed_size)
        meta_bytes = f.read(meta_size)

        # Phase 1.3: optional trailing BPR3 section. Read it BEFORE we exit
        # the `with` block so the file handle stays open. The position is
        # exactly at the start of the trailing section after meta_bytes.
        bpr3_payload = None
        bpr3_magic = f.read(4)
        if bpr3_magic == b'BPR3':
            bpr3_compressed_size = struct.unpack('<Q', f.read(8))[0]
            bpr3_compressed = f.read(bpr3_compressed_size)
            bpr3_payload = lzma.decompress(bpr3_compressed)
        elif bpr3_magic:
            print(f"  WARN: trailing magic {bpr3_magic!r} != b'BPR3', "
                  f"treating as no-EV file", flush=True)

    meta = json.loads(meta_bytes.decode('utf-8'))
    print(f"  Meta: {meta.get('iterations', '?')} iters, "
          f"{meta.get('num_info_sets', '?')} info sets, "
          f"hash_mixer={meta.get('hash_mixer', 'boost')}, "
          f"has_action_evs={meta.get('has_action_evs', False)}", flush=True)

    print(f"  Decompressing {len(compressed) / 1e6:.1f} MB LZMA...", flush=True)
    t0 = time.time()
    strat_data = lzma.decompress(compressed)
    print(f"  Decompressed to {len(strat_data) / 1e6:.1f} MB in {time.time() - t0:.1f}s", flush=True)

    # Inner BPS3 header
    p = 0
    if strat_data[p:p+4] != b'BPS3':
        raise ValueError("Inner BPS3 magic missing")
    p += 4
    n_entries = struct.unpack_from('<I', strat_data, p)[0]; p += 4
    n_players = struct.unpack_from('<I', strat_data, p)[0]; p += 4
    print(f"  Entries: {n_entries:,}, players: {n_players}", flush=True)

    # Stream-parse, keep only street == 0 (preflop)
    print("  Scanning entries...", flush=True)
    t0 = time.time()
    by_node = {}  # (action_hash, player) -> {bucket: np.ndarray[na] uint8}
    kept = 0
    skipped = 0

    # Local references for speed
    sd = strat_data
    unpack_from = struct.unpack_from
    fmt_H = '<H'
    fmt_Q = '<Q'

    for _ in range(n_entries):
        player = sd[p]; p += 1
        street = sd[p]; p += 1
        bucket = unpack_from(fmt_H, sd, p)[0]; p += 2
        board_hash = unpack_from(fmt_Q, sd, p)[0]; p += 8
        action_hash = unpack_from(fmt_Q, sd, p)[0]; p += 8
        na = sd[p]; p += 1

        if street == 0:
            # Copy uint8 strategy bytes (zero-copy view, then materialize)
            strat = np.frombuffer(sd, dtype=np.uint8, count=na, offset=p).copy()
            key = (action_hash, player)
            d = by_node.get(key)
            if d is None:
                by_node[key] = {bucket: strat}
            else:
                d[bucket] = strat
            kept += 1
        else:
            skipped += 1

        p += na

    print(f"  Parsed {n_entries:,} entries in {time.time() - t0:.1f}s "
          f"(kept {kept:,} preflop, skipped {skipped:,})", flush=True)

    # Phase 1.3: parse the BPR3 payload if it was present.
    # Format per blueprint_v2._load_action_evs_section:
    #   'BPR3' (4B) + u32 num_entries
    #   per entry: player(1) + street(1) + bucket(2) + action_hash(8)
    #              + num_actions(1) + float32[num_actions] avg_ev
    if bpr3_payload is not None:
        print(f"  Scanning BPR3 ({len(bpr3_payload) / 1e6:.1f} MB)...", flush=True)
        t0 = time.time()
        ep = 0
        if bpr3_payload[ep:ep+4] != b'BPR3':
            print(f"  WARN: inner BPR3 magic missing, skipping EV section", flush=True)
        else:
            ep += 4
            ev_n_entries = struct.unpack_from('<I', bpr3_payload, ep)[0]; ep += 4
            ev_kept = 0
            ev_skipped = 0
            ep_local = bpr3_payload  # local ref for speed
            for _ in range(ev_n_entries):
                player = ep_local[ep]; ep += 1
                street = ep_local[ep]; ep += 1
                bucket = unpack_from(fmt_H, ep_local, ep)[0]; ep += 2
                action_hash = unpack_from(fmt_Q, ep_local, ep)[0]; ep += 8
                na = ep_local[ep]; ep += 1

                if street == 0:
                    evs = np.frombuffer(ep_local, dtype=np.float32,
                                        count=na, offset=ep).copy()
                    key = (action_hash, player)
                    d = by_node_evs.get(key)
                    if d is None:
                        by_node_evs[key] = {bucket: evs}
                    else:
                        d[bucket] = evs
                    ev_kept += 1
                else:
                    ev_skipped += 1

                ep += na * 4

            print(f"  Parsed {ev_n_entries:,} EV entries in {time.time() - t0:.1f}s "
                  f"(kept {ev_kept:,} preflop, skipped {ev_skipped:,})", flush=True)

    return by_node, by_node_evs, meta


def get_or_build_cached_table(bps_path):
    """Load (action_hash, player) -> packed strategies + per-action EVs.

    Cache file: <bps_path>.preflop_cache.npz
    Cache key: (file size, mtime, schema). On hit, load takes <1s instead of
    20+ min.

    Cache schema is bumped to v2 for the EV addition. Legacy v1 caches are
    rejected and rebuilt — there is no cross-compat. The schema field also
    encodes the hash mixer; mismatched mixer caches are rebuilt.

    Returns: (table, ev_table, meta) where:
        table:    dict[(action_hash, player)] -> np.ndarray[169, na] uint8
                  (dense per-bucket matrix, defaulting to 0 for unvisited)
        ev_table: dict[(action_hash, player)] -> np.ndarray[169, na] float32
                  (parallel structure, empty dict if no BPR3 section)
        meta:     the .bps metadata dict (used by main() to set _HASH_MIXER)
    """
    cache_path = bps_path + ".preflop_cache.npz"
    bps_stat = os.stat(bps_path)
    bps_size = bps_stat.st_size
    bps_mtime = int(bps_stat.st_mtime)

    CACHE_SCHEMA = 2  # bump on any layout change

    if os.path.exists(cache_path):
        try:
            print(f"Loading cache {os.path.basename(cache_path)}...", flush=True)
            t0 = time.time()
            cached = np.load(cache_path, allow_pickle=False)
            cached_size = int(cached['_bps_size'])
            cached_mtime = int(cached['_bps_mtime'])
            cached_schema = int(cached.get('_schema', np.array(1)))
            if (cached_size == bps_size and
                cached_mtime == bps_mtime and
                cached_schema == CACHE_SCHEMA):
                # Cache valid — reconstruct dicts
                ah_arr = cached['ah']          # [N] uint64
                player_arr = cached['player']  # [N] uint8
                offsets = cached['offsets']    # [N+1] int64 row offsets
                na_arr = cached['na']          # [N] uint8
                buf = cached['buf']            # [total_size] uint8 flattened
                meta_json = bytes(cached['meta_json']).decode('utf-8')
                meta = json.loads(meta_json)

                table = {}
                for i in range(len(ah_arr)):
                    na = int(na_arr[i])
                    start = int(offsets[i])
                    rows = buf[start:start + 169 * na].reshape(169, na)
                    table[(int(ah_arr[i]), int(player_arr[i]))] = rows

                ev_table = {}
                if 'ev_buf' in cached.files and cached['ev_buf'].size > 0:
                    ev_buf = cached['ev_buf']  # [total_floats] float32
                    ev_offsets = cached['ev_offsets']  # [N+1] int64
                    for i in range(len(ah_arr)):
                        na = int(na_arr[i])
                        start = int(ev_offsets[i])
                        evs = ev_buf[start:start + 169 * na].reshape(169, na)
                        ev_table[(int(ah_arr[i]), int(player_arr[i]))] = evs

                print(f"Cache hit: {len(table):,} nodes "
                      f"({len(ev_table):,} with EVs) in {time.time() - t0:.1f}s",
                      flush=True)
                return table, ev_table, meta
            else:
                if cached_schema != CACHE_SCHEMA:
                    print(f"Cache schema {cached_schema} != {CACHE_SCHEMA}, rebuilding...",
                          flush=True)
                else:
                    print(f"Cache stale (bps changed), rebuilding...", flush=True)
        except Exception as e:
            print(f"Cache load failed ({e}), rebuilding...", flush=True)

    # Build from scratch
    by_node, by_node_evs, meta = load_bps3_preflop_direct(bps_path)

    # Convert sparse {bucket: row} dicts into dense [169, na] uint8 matrices.
    # Unvisited buckets stay 0 (vs the old code's uniform default — we'll
    # treat 0-row as "no data" on the frontend and skip it during expansion).
    print(f"  Densifying {len(by_node):,} nodes...", flush=True)
    t0 = time.time()
    table = {}
    for key, bucket_map in by_node.items():
        sample = next(iter(bucket_map.values()))
        na = sample.shape[0]
        rows = np.zeros((169, na), dtype=np.uint8)
        for bucket, strat in bucket_map.items():
            if bucket < 169:
                rows[bucket] = strat
        table[key] = rows
    print(f"  Densified in {time.time() - t0:.1f}s", flush=True)

    ev_table = {}
    if by_node_evs:
        print(f"  Densifying {len(by_node_evs):,} EV nodes...", flush=True)
        t0 = time.time()
        for key, bucket_map in by_node_evs.items():
            sample = next(iter(bucket_map.values()))
            na = sample.shape[0]
            evs_arr = np.zeros((169, na), dtype=np.float32)
            for bucket, evs in bucket_map.items():
                if bucket < 169:
                    evs_arr[bucket, :na] = evs[:na]
            ev_table[key] = evs_arr
        print(f"  EV densified in {time.time() - t0:.1f}s", flush=True)

    # Save cache
    print(f"  Writing cache {os.path.basename(cache_path)}...", flush=True)
    t0 = time.time()
    keys_list = list(table.keys())
    ah_arr = np.array([k[0] for k in keys_list], dtype=np.uint64)
    player_arr = np.array([k[1] for k in keys_list], dtype=np.uint8)
    na_arr = np.array([table[k].shape[1] for k in keys_list], dtype=np.uint8)
    offsets = np.zeros(len(keys_list) + 1, dtype=np.int64)
    for i in range(len(keys_list)):
        offsets[i + 1] = offsets[i] + 169 * int(na_arr[i])
    total = int(offsets[-1])
    buf = np.zeros(total, dtype=np.uint8)
    for i, k in enumerate(keys_list):
        start = int(offsets[i])
        n = 169 * int(na_arr[i])
        buf[start:start + n] = table[k].reshape(-1)

    # Pack EVs in the same key order so they line up with the strategy buffer.
    # Nodes that have a strategy but no EV (BPR3 absent or EV walk skipped that
    # info set) get a zero-filled placeholder so the cache layout is dense.
    if ev_table:
        ev_offsets = np.zeros(len(keys_list) + 1, dtype=np.int64)
        for i in range(len(keys_list)):
            ev_offsets[i + 1] = ev_offsets[i] + 169 * int(na_arr[i])
        ev_total = int(ev_offsets[-1])
        ev_buf = np.zeros(ev_total, dtype=np.float32)
        for i, k in enumerate(keys_list):
            evs = ev_table.get(k)
            if evs is None:
                continue
            start = int(ev_offsets[i])
            n = 169 * int(na_arr[i])
            ev_buf[start:start + n] = evs.reshape(-1)
    else:
        ev_offsets = np.zeros(1, dtype=np.int64)
        ev_buf = np.zeros(0, dtype=np.float32)

    np.savez(
        cache_path,
        _bps_size=np.int64(bps_size),
        _bps_mtime=np.int64(bps_mtime),
        _schema=np.int32(CACHE_SCHEMA),
        ah=ah_arr,
        player=player_arr,
        na=na_arr,
        offsets=offsets,
        buf=buf,
        ev_offsets=ev_offsets,
        ev_buf=ev_buf,
        meta_json=np.frombuffer(json.dumps(meta).encode('utf-8'), dtype=np.uint8),
    )
    print(f"  Cache written in {time.time() - t0:.1f}s "
          f"({os.path.getsize(cache_path) / 1e6:.1f} MB)", flush=True)
    return table, ev_table, meta


def verify_utg_root_sanity(out_nodes, hand_grid):
    """Sanity-check the extracted UTG root strategies against known-good frequencies.

    Catches bucket-mapping bugs (Bug A in EXTRACTOR_BUGS.md), data corruption, and
    upstream training collapse. Refuses to write the JSON if too many sentinels fail.

    The thresholds are deliberately loose because the .bps currently contains
    regret-matched per-iteration strategies (not the converged strategy_sum average,
    see EXTRACTOR_BUGS Bug B). After Bug B is fixed upstream, we can tighten these.

    Sentinels (UTG = first to act in 6-max, the tightest position):
        Premium hands must NOT fold:
            AA, KK, AKs, AKo  -> fold <= 0.15
        Trash hands MUST fold:
            72o, 32o          -> fold >= 0.95
        Small pairs and weak suited fold UTG 6-max:
            22, 52s           -> fold >= 0.85

    Raises ValueError if 2 or more sentinels fail (1 is noise, 2 is structural).
    """
    import base64
    import numpy as np

    root = out_nodes.get(ROOT_HASH_HEX)
    if root is None:
        raise ValueError(
            f"UTG root node ({ROOT_HASH_HEX}) missing from extraction. "
            "Either the tree walker failed to enumerate it or the .bps doesn't contain it."
        )

    labels = root["l"]
    na = len(labels)
    buf = np.frombuffer(base64.b64decode(root["s"]), dtype=np.uint8).reshape(169, na)

    if "fold" not in labels:
        raise ValueError(f"UTG root has no 'fold' label: {labels}")
    fold_idx = labels.index("fold")

    sentinels = [
        # (hand, max_fold_freq or None, min_fold_freq or None, description)
        ("AA",  0.15, None, "premium pair never folds"),
        ("KK",  0.15, None, "premium pair never folds"),
        ("AKs", 0.15, None, "AKs never folds UTG"),
        ("AKo", 0.20, None, "AKo never folds UTG"),
        ("72o", None, 0.95, "trash always folds UTG"),
        ("32o", None, 0.95, "trash always folds UTG"),
        ("22",  None, 0.85, "small pair folds UTG 6-max"),
        ("52s", None, 0.85, "weak suited folds UTG 6-max"),
    ]

    failures = []
    print(f"  {'hand':6s} {'fold':>8s}  {'expected':12s}  status")
    print(f"  {'-'*6} {'-'*8}  {'-'*12}  {'-'*30}")
    for hand, max_fold, min_fold, desc in sentinels:
        if hand not in hand_grid:
            failures.append(f"{hand}: not in hand_grid")
            continue
        idx = hand_grid.index(hand)
        fold_freq = float(buf[idx, fold_idx]) / 255.0

        if max_fold is not None:
            ok = fold_freq <= max_fold
            expected = f"<= {max_fold:.2f}"
        else:
            ok = fold_freq >= min_fold
            expected = f">= {min_fold:.2f}"

        status = "OK" if ok else "FAIL"
        print(f"  {hand:6s} {fold_freq:8.3f}  {expected:12s}  {status}  ({desc})")
        if not ok:
            failures.append(f"{hand}: fold={fold_freq:.3f}, expected {expected}")

    if len(failures) >= 2:
        msg = (
            f"\n  EXTRACTION FAILED VERIFICATION ({len(failures)}/{len(sentinels)} sentinels failed):\n"
            + "\n".join(f"    - {f}" for f in failures)
            + "\n  This usually means the bucket mapping in hand_class_to_bucket() is wrong."
            + "\n  See poker-solver/docs/EXTRACTOR_BUGS.md Bug A for details."
            + "\n  Refusing to write preflop-nodes.json — fix the bug first."
        )
        raise ValueError(msg)

    if failures:
        print(f"  WARNING: 1 sentinel failed (allowed as noise tolerance):", flush=True)
        for f in failures:
            print(f"    - {f}", flush=True)
    else:
        print(f"  All {len(sentinels)} sentinels passed.", flush=True)


def compute_nodes_by_tree_walk(max_depth=8):
    """Enumerate reachable preflop tree nodes up to max_depth actions deep.

    Returns: dict mapping action_hash_int → {player, labels, to_call_c, pot_c, stack_c, num_raises}.

    Mirrors mccfr_blueprint.c exactly:
    - Uses INTEGER chip amounts (never floats)
    - Picks bet sizes from PREFLOP_TIERS[num_raises] (tier-aware)
    - Action history is a sequence of action INDICES into each node's labels list
    - Acting order rotates UTG→MP→CO→BTN→SB→BB, skipping folded players
    - Round ends when all active players have acted AND all bets match current max
    - max_raises = PREFLOP_MAX_RAISES
    """
    acting_order_players = [POS_TO_PLAYER[p] for p in POSITIONS]  # [2,3,4,5,0,1]
    nodes = {}
    visited = set()

    def recurse(bets_c, active, has_acted, stacks_c, pot_c, num_raises, history, next_order_idx):
        """Explore a preflop game state. All amounts in INT chips."""
        if len(history) > max_depth:
            return
        nactive = sum(1 for a in active if a)
        if nactive <= 1:
            return

        # Find next active player
        oi = next_order_idx
        for _ in range(6):
            ap = acting_order_players[oi % 6]
            if active[ap]:
                break
            oi += 1
        else:
            return

        mx_bet_c = max(bets_c[p] for p in range(6) if active[p])

        # Round complete?
        if all((not active[p]) or (has_acted[p] and bets_c[p] == mx_bet_c)
               for p in range(6)):
            return  # preflop closed

        to_call_c = max(0, mx_bet_c - bets_c[ap])
        stack_here_c = stacks_c[ap]
        current_committed_c = bets_c[ap]
        actions = enumerate_actions_chips(to_call_c, pot_c, stack_here_c,
                                          current_committed_c, num_raises,
                                          PREFLOP_MAX_RAISES)
        if not actions:
            return

        labels = [a[2] for a in actions]

        # Disambiguate duplicate labels (different raise sizes that round to the
        # same display string). Frontend relies on unique labels per node.
        seen_counts = {}
        unique_labels = []
        for lbl in labels:
            if lbl in seen_counts:
                seen_counts[lbl] += 1
                unique_labels.append(f"{lbl}_{seen_counts[lbl]}")
            else:
                seen_counts[lbl] = 0
                unique_labels.append(lbl)
        labels = unique_labels

        action_hash = compute_action_hash(history)

        if action_hash not in nodes:
            nodes[action_hash] = {
                "player": ap,
                "to_call_c": to_call_c,
                "pot_c": pot_c,
                "stack_c": stack_here_c,
                "num_raises": num_raises,
                "labels": labels,
            }

        # Memoize by full state to avoid redundant subtree exploration
        state_key = (tuple(bets_c), tuple(active), tuple(has_acted), num_raises, history)
        if state_key in visited:
            return
        visited.add(state_key)

        next_oi = (oi + 1) % 6

        for idx, (atype, add_c, _lbl) in enumerate(actions):
            new_bets_c = list(bets_c)
            new_active = list(active)
            new_has_acted = list(has_acted)
            new_stacks_c = list(stacks_c)
            new_pot_c = pot_c
            new_num_raises = num_raises

            if atype == "fold":
                new_active[ap] = False
            elif atype == "check":
                new_has_acted[ap] = True
            elif atype == "call":
                new_bets_c[ap] = mx_bet_c
                new_stacks_c[ap] -= add_c
                new_pot_c += add_c
                new_has_acted[ap] = True
            elif atype == "bet":
                # C solver: child.bets[ap] += amount (add_c is total added)
                # The C amount is already "to_call + raise increment" so we
                # add the whole thing to bets[ap].
                new_bets_c[ap] += add_c
                new_stacks_c[ap] -= add_c
                new_pot_c += add_c
                new_has_acted[ap] = True
                new_num_raises += 1
                for p in range(6):
                    if p != ap and new_active[p]:
                        new_has_acted[p] = False
            elif atype == "allin":
                actual_add = min(add_c, new_stacks_c[ap])
                new_bets_c[ap] += actual_add
                new_stacks_c[ap] -= actual_add
                new_pot_c += actual_add
                new_has_acted[ap] = True
                if new_bets_c[ap] > mx_bet_c:
                    new_num_raises += 1
                    for p in range(6):
                        if p != ap and new_active[p]:
                            new_has_acted[p] = False

            new_history = history + (idx,)
            recurse(new_bets_c, new_active, new_has_acted, new_stacks_c, new_pot_c,
                    new_num_raises, new_history, next_oi)

    # Initial state: SB=50, BB=100 chips, UTG first to act
    bets0 = [0] * 6
    bets0[0] = SMALL_BLIND  # player 0 = SB
    bets0[1] = BIG_BLIND    # player 1 = BB
    active0 = [True] * 6
    has_acted0 = [False] * 6
    stacks0 = [INITIAL_STACK - bets0[p] for p in range(6)]
    pot0 = SMALL_BLIND + BIG_BLIND

    recurse(bets0, active0, has_acted0, stacks0, pot0, 0, tuple(), 0)
    return nodes


def main():
    global _HASH_MIXER

    # Parse args: positional bps_path [output_path] [max_depth] [--mixer=...]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    bps_path = args[0] if len(args) > 0 else "blueprint_data/unified_blueprint.bps"
    output_path = args[1] if len(args) > 1 else None
    max_depth = int(args[2]) if len(args) > 2 else 8

    # --mixer=boost|splitmix64|auto overrides the metadata-based detection.
    # Useful when the .bps metadata says one mixer but the stored keys were
    # actually hashed with the other — e.g., v3 exports from v2 checkpoints
    # that were trained before the Bug 6 fix land with splitmix64 metadata
    # but boost-hashed keys (because the export reads key fields directly
    # from the raw-regrets file without recomputing hashes).
    mixer_override = None
    for f in flags:
        if f.startswith("--mixer="):
            mixer_override = f.split("=", 1)[1]

    t_total = time.time()

    print(f"=== Extracting preflop nodes from {bps_path} ===", flush=True)
    print(f"Step 1: Load .bps (cached if available)...", flush=True)
    t0 = time.time()
    by_ah_player, by_ah_player_evs, meta = get_or_build_cached_table(bps_path)
    print(f"Step 1 done in {time.time() - t0:.1f}s — {len(by_ah_player):,} preflop nodes "
          f"({len(by_ah_player_evs):,} with EVs)", flush=True)

    # Set hash mixer. Default = meta-based detection. Override via --mixer=X.
    if mixer_override:
        _HASH_MIXER = mixer_override
        print(f"  Using hash mixer: {_HASH_MIXER} (CLI override)", flush=True)
    else:
        _HASH_MIXER = _detect_mixer_from_meta(meta)
        print(f"  Using hash mixer: {_HASH_MIXER} (from metadata)", flush=True)

    # Detect stale metadata: if the meta says splitmix64 but the tree walker
    # with that mixer doesn't match any cache entries, auto-fallback to boost.
    # This catches the v2-checkpoint-reexported-with-new-code case where the
    # export process inherits old hash keys from regrets_*.bin but the
    # metadata tag is hard-coded fresh in export_v2.py.
    #
    # Probe: after action [0] (UTG folds), MP (player 3) acts next. Check
    # whether compute_action_hash([0]) matches ANY player in the cache under
    # the current mixer. If not, try boost and see if that matches.
    if mixer_override is None and _HASH_MIXER == "splitmix64":
        def _probe_mixer_matches():
            h0 = compute_action_hash([0])
            return any((h0, p) in by_ah_player for p in range(6))

        if not _probe_mixer_matches():
            saved = _HASH_MIXER
            _HASH_MIXER = "boost"
            if _probe_mixer_matches():
                print(f"  WARN: metadata says splitmix64 but cache entries are "
                      f"boost-hashed; auto-switched mixer to boost. This means the "
                      f".bps was exported from a pre-Bug-6 raw regrets file — the "
                      f"metadata tag in export_v2.py is hardcoded regardless of "
                      f"what the source checkpoint actually used.", flush=True)
            else:
                _HASH_MIXER = saved  # revert, give up on auto-detect
                print(f"  WARN: neither mixer matched the cache probe. Continuing "
                      f"with {saved} anyway — expect many 'missing' tree nodes.",
                      flush=True)

    print(f"\nStep 2: Walking preflop tree (max_depth={max_depth})...", flush=True)
    t0 = time.time()
    tree_nodes = compute_nodes_by_tree_walk(max_depth=max_depth)
    print(f"Step 2 done in {time.time() - t0:.1f}s — {len(tree_nodes):,} decision nodes",
          flush=True)

    hand_grid = generate_hand_grid()
    # Pre-compute hand_grid → bucket index map (one-time, then reused)
    hand_to_bucket = [hand_class_to_bucket(h) for h in hand_grid]

    print(f"\nStep 3: Match tree nodes to .bps entries and build output...", flush=True)
    t0 = time.time()
    out_nodes = {}
    missing = 0
    label_mismatch = 0
    nodes_with_evs = 0
    for action_hash, info in tree_nodes.items():
        player = info["player"]
        strats = by_ah_player.get((action_hash, player))
        if strats is None:
            missing += 1
            continue

        na_data = strats.shape[1]
        na_labels = len(info["labels"])
        if na_data != na_labels:
            label_mismatch += 1
            labels = info["labels"][:na_data]
            while len(labels) < na_data:
                labels.append(f"action_{len(labels)}")
        else:
            labels = info["labels"]

        # Compact encoding: reorder rows from bucket order to hand_grid order
        # then base64-encode the raw uint8 bytes. This is ~3x more compact
        # than JSON arrays and parses in O(1) time on the frontend.
        ordered = strats[hand_to_bucket]  # [169, na] uint8 in hand-grid order
        b64 = base64.b64encode(ordered.tobytes()).decode('ascii')

        node_obj = {
            "p": player,
            "l": labels,
            "s": b64,  # base64-encoded 169*na uint8 row-major buffer
        }

        # Phase 1.3: per-action EVs from BPR3, if present and non-zero.
        # Layout matches "s": 169 hand-grid rows × na actions, float32 in
        # chip space. Frontend converts to bb display via /CHIPS_PER_BB.
        #
        # Suppress zero-filled EV rows — the Phase 1.3 EV walker in the
        # C code has a known bug where it only visits the preflop root
        # and writes an empty BPR3 section (see STATUS.md). Until that's
        # fixed, every densified entry is zero-filled. Emitting an "e"
        # field full of zeros would mislead the frontend into showing
        # bogus 0.00 bb EVs for every action. Skip if entirely zero.
        evs = by_ah_player_evs.get((action_hash, player))
        if evs is not None and np.any(evs != 0):
            ordered_evs = evs[hand_to_bucket]
            # Match action label count: trim or pad zero columns
            if ordered_evs.shape[1] != na_data:
                if ordered_evs.shape[1] > na_data:
                    ordered_evs = ordered_evs[:, :na_data]
                else:
                    pad = np.zeros((169, na_data - ordered_evs.shape[1]),
                                   dtype=np.float32)
                    ordered_evs = np.concatenate([ordered_evs, pad], axis=1)
            ev_b64 = base64.b64encode(
                ordered_evs.astype(np.float32).tobytes()
            ).decode('ascii')
            node_obj["e"] = ev_b64
            nodes_with_evs += 1

        hash_hex = f"0x{action_hash:016X}"
        out_nodes[hash_hex] = node_obj

    print(f"Step 3 done in {time.time() - t0:.1f}s", flush=True)
    print(f"  Matched: {len(out_nodes):,} nodes", flush=True)
    print(f"  With EVs: {nodes_with_evs:,}", flush=True)
    print(f"  Missing from .bps: {missing:,} (unreachable / pruned during training)",
          flush=True)
    print(f"  Label mismatches: {label_mismatch:,}", flush=True)

    print(f"\nStep 3.5: Sanity-checking UTG root strategies...", flush=True)
    verify_utg_root_sanity(out_nodes, hand_grid)

    result = {
        "meta": {
            "preflop_tiers": {str(k): v for k, v in PREFLOP_TIERS.items()},
            "preflop_max_raises": PREFLOP_MAX_RAISES,
            "sb": SB_SIZE,
            "bb": BB_SIZE,
            "stack": INITIAL_STACK_BB,
            "pos_to_player": POS_TO_PLAYER,
            "player_to_pos": {str(v): k for k, v in POS_TO_PLAYER.items()},
            "acting_order": POSITIONS,
            "root_hash": ROOT_HASH_HEX,
            "hand_order": hand_grid,
            "max_depth": max_depth,
            "quantize": "uint8_b64",  # base64-encoded 169×na uint8 rows
            "freq_scale": 255,         # divide by this to get probability
            # ── Phase 1.3 / Bug 6 additions ──
            # Tag the file with the hash mixer used at extraction time so
            # the frontend's action-hash.ts can refuse to load on mismatch.
            # Required because pre-Bug-6 (boost) and post-Bug-6 (splitmix64)
            # JSONs have completely incompatible action_hash key spaces.
            "hash_mixer": _HASH_MIXER,
            # Per-action EVs in chips (NOT bb). Frontend divides by 100 for
            # display. Field "e" on each node is base64 of float32[169][na],
            # parallel to "s". Absent if the .bps had no BPR3 section.
            "has_action_evs": nodes_with_evs > 0,
            "ev_field_format": "float32_b64_chips" if nodes_with_evs > 0 else None,
            "ev_chips_per_bb": CHIPS_PER_BB,
            # Provenance from the source .bps so we can trace back when this
            # JSON gets stale relative to a newer training checkpoint.
            "source_bps_iterations": meta.get("iterations"),
            "source_bps_num_info_sets": meta.get("num_info_sets"),
            "source_bps_code_sha": meta.get("code_sha"),
            "source_bps_exported_at": meta.get("exported_at"),
        },
        "nodes": out_nodes,
    }

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'nexusgto',
            'src', 'data', 'preflop-nodes.json'
        )

    print(f"\nStep 4: Writing JSON to {output_path}...", flush=True)
    t0 = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, separators=(',', ':'))
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Step 4 done in {time.time() - t0:.1f}s — {size_mb:.1f} MB", flush=True)

    print(f"\n=== Total: {time.time() - t_total:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
