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


def hash_combine(a, b):
    a = mask64(a)
    b = mask64(b)
    return mask64(a ^ mask64(mask64(b + 0x9E3779B97F4A7C15) + mask64(a << 6) + (a >> 2)))


def compute_action_hash(action_indices):
    h = HASH_SEED
    for a in action_indices:
        h = hash_combine(h, mask64(a * 17 + 3))
    return h


def load_bps3_preflop_direct(bps_path):
    """Fast loader: read BPS3, decompress, and extract ONLY preflop info sets.

    Returns: dict[(action_hash:int, player:int)] -> dict[bucket:int -> np.ndarray[na] uint8]

    This bypasses BlueprintV2._load_bps3 and skips:
    - All non-preflop (street != 0) entries during parsing
    - Per-bucket renormalization (we keep raw uint8 quantized strategies)
    - The expensive [num_buckets, na] zero-filled fallback array

    For an unknown but typical BPS3 (~6M entries), this is ~3-5x faster than
    the BlueprintV2 path because the inner loop is shorter and we never build
    fallback arrays for unused streets.
    """
    with open(bps_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'BPS3':
            raise ValueError(f"Not a BPS3 file (magic={magic!r})")
        compressed_size, meta_size = struct.unpack('<QI', f.read(12))
        compressed = f.read(compressed_size)
        meta_bytes = f.read(meta_size)

    meta = json.loads(meta_bytes.decode('utf-8'))
    print(f"  Meta: {meta.get('iterations', '?')} iters, "
          f"{meta.get('num_info_sets', '?')} info sets", flush=True)

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
    return by_node, meta


def get_or_build_cached_table(bps_path):
    """Load (action_hash, player) -> packed strategies, using a disk cache.

    Cache file: <bps_path>.preflop_cache.npz
    Cache key: (file size, mtime). On hit, load takes <1s instead of 20+ min.

    Returns: dict[(action_hash, player)] -> np.ndarray[169, na] uint8
        (dense per-bucket matrix, defaulting to 0 for unvisited buckets)
    """
    cache_path = bps_path + ".preflop_cache.npz"
    bps_stat = os.stat(bps_path)
    bps_size = bps_stat.st_size
    bps_mtime = int(bps_stat.st_mtime)

    if os.path.exists(cache_path):
        try:
            print(f"Loading cache {os.path.basename(cache_path)}...", flush=True)
            t0 = time.time()
            cached = np.load(cache_path, allow_pickle=False)
            cached_size = int(cached['_bps_size'])
            cached_mtime = int(cached['_bps_mtime'])
            if cached_size == bps_size and cached_mtime == bps_mtime:
                # Cache valid — reconstruct dict
                keys_arr = cached['keys']      # [N, 2] int64 (action_hash_lo, player) packed
                ah_arr = cached['ah']          # [N] uint64
                player_arr = cached['player']  # [N] uint8
                offsets = cached['offsets']    # [N+1] int64 row offsets into matrix
                na_arr = cached['na']          # [N] uint8
                buf = cached['buf']            # [total_size] uint8 flattened
                table = {}
                for i in range(len(ah_arr)):
                    na = int(na_arr[i])
                    start = int(offsets[i])
                    rows = buf[start:start + 169 * na].reshape(169, na)
                    table[(int(ah_arr[i]), int(player_arr[i]))] = rows
                print(f"Cache hit: {len(table):,} nodes in {time.time() - t0:.1f}s",
                      flush=True)
                return table
            else:
                print(f"Cache stale (bps changed), rebuilding...", flush=True)
        except Exception as e:
            print(f"Cache load failed ({e}), rebuilding...", flush=True)

    # Build from scratch
    by_node, _meta = load_bps3_preflop_direct(bps_path)

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

    # Save cache
    print(f"  Writing cache {os.path.basename(cache_path)}...", flush=True)
    t0 = time.time()
    keys_list = list(table.keys())
    ah_arr = np.array([k[0] for k in keys_list], dtype=np.uint64)
    player_arr = np.array([k[1] for k in keys_list], dtype=np.uint8)
    na_arr = np.array([table[k].shape[1] for k in keys_list], dtype=np.uint8)
    offsets = np.zeros(len(keys_list) + 1, dtype=np.int64)
    for i, k in enumerate(keys_list):
        offsets[i + 1] = offsets[i] + 169 * int(na_arr[i])
    total = int(offsets[-1])
    buf = np.zeros(total, dtype=np.uint8)
    for i, k in enumerate(keys_list):
        start = int(offsets[i])
        n = 169 * int(na_arr[i])
        buf[start:start + n] = table[k].reshape(-1)
    np.savez(
        cache_path,
        _bps_size=np.int64(bps_size),
        _bps_mtime=np.int64(bps_mtime),
        ah=ah_arr,
        player=player_arr,
        na=na_arr,
        offsets=offsets,
        buf=buf,
        keys=np.zeros((1,), dtype=np.int64),  # unused but needed for old format
    )
    print(f"  Cache written in {time.time() - t0:.1f}s "
          f"({os.path.getsize(cache_path) / 1e6:.1f} MB)", flush=True)
    return table


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
    bps_path = sys.argv[1] if len(sys.argv) > 1 else "blueprint_data/unified_blueprint.bps"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    t_total = time.time()

    print(f"=== Extracting preflop nodes from {bps_path} ===", flush=True)
    print(f"Step 1: Load .bps (cached if available)...", flush=True)
    t0 = time.time()
    by_ah_player = get_or_build_cached_table(bps_path)
    print(f"Step 1 done in {time.time() - t0:.1f}s — {len(by_ah_player):,} preflop nodes",
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

        hash_hex = f"0x{action_hash:016X}"
        out_nodes[hash_hex] = {
            "p": player,
            "l": labels,
            "s": b64,  # base64-encoded 169*na uint8 row-major buffer
        }
    print(f"Step 3 done in {time.time() - t0:.1f}s", flush=True)
    print(f"  Matched: {len(out_nodes):,} nodes", flush=True)
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
