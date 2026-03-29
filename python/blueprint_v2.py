"""Blueprint v2 loader — reads .bps binary files from the MCCFR blueprint.

Provides fast lookup of P(action|bucket) at any game tree node via
(board_hash, action_hash, player, street) -> strategy[bucket].

Storage: .bps files per flop texture, LZMA compressed, with street tags.
Only loads what's needed: flop info sets for range narrowing, plus
turn/river roots for continuation leaf values.

Usage:
    bp = BlueprintV2("blueprint_data/")
    bp.load_texture("T72_r")  # loads from .bps file

    # Get strategy at a specific node
    strat = bp.get_strategy(
        board=[40, 32, 4],       # Qs Ts 2d
        action_history=[1, 0],   # bet, fold
        player=0,
        bucket=42
    )
    # strat = [0.3, 0.5, 0.1, 0.1]  (check, bet50, betPot, allin)

    # Get all strategies for range narrowing
    strats = bp.get_all_bucket_strategies(board, action_history, player)
    # strats[bucket] = [p_check, p_bet50, ...]
"""

import json
import lzma
import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_to_int(s):
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])


def int_to_card(i):
    return RANKS[i // 4] + SUITS[i % 4]


def _mask64(x):
    """Truncate to uint64."""
    return x & 0xFFFFFFFFFFFFFFFF


def _compute_board_hash(board, num_board):
    """Must match compute_board_hash in mccfr_blueprint.c exactly.
    Board cards are canonicalized first (suit-isomorphic mapping)."""
    if num_board >= 3:
        canon = _canonicalize_board(board, num_board)
    else:
        canon = list(board[:num_board])
    h = _mask64(0x123456789ABCDEF)
    for i in range(num_board):
        h = _hash_combine(h, _mask64(canon[i] * 31 + 7))
    return h


def _canonicalize_board(board, num_board):
    """Canonicalize board cards for info set hashing.
    Must match canonicalize_board() in mccfr_blueprint.c exactly."""
    # Sort flop by rank descending
    flop = sorted(board[:3], key=lambda c: -(c >> 2))
    r0, r1, r2 = flop[0] >> 2, flop[1] >> 2, flop[2] >> 2
    s0, s1, s2 = flop[0] & 3, flop[1] & 3, flop[2] & 3

    if r0 == r1 and r1 == r2:
        canon_flop = [r0*4+3, r1*4+2, r2*4+1]
    elif r0 == r1 or r1 == r2:
        if r0 == r1:
            if s2 == s0 or s2 == s1:
                canon_flop = [r0*4+3, r1*4+2, r2*4+3]
            else:
                canon_flop = [r0*4+3, r1*4+2, r2*4+1]
        else:
            if s0 == s1 or s0 == s2:
                canon_flop = [r0*4+3, r1*4+3, r2*4+2]
            else:
                canon_flop = [r0*4+3, r1*4+2, r2*4+1]
    else:
        if s0 == s1 and s1 == s2:
            canon_flop = [r0*4+3, r1*4+3, r2*4+3]
        elif s0 == s1:
            canon_flop = [r0*4+3, r1*4+3, r2*4+2]
        elif s0 == s2:
            canon_flop = [r0*4+3, r1*4+2, r2*4+3]
        elif s1 == s2:
            canon_flop = [r0*4+2, r1*4+3, r2*4+3]
        else:
            canon_flop = [r0*4+3, r1*4+2, r2*4+1]

    # Build suit mapping
    suit_map = [-1, -1, -1, -1]
    for i in range(3):
        actual_suit = flop[i] & 3
        canon_suit = canon_flop[i] & 3
        if suit_map[actual_suit] == -1:
            suit_map[actual_suit] = canon_suit
    # Fill unmapped suits
    nc = 0
    for i in range(4):
        if suit_map[i] == -1:
            while nc < 4 and any(suit_map[j] == nc for j in range(4)):
                nc += 1
            suit_map[i] = nc if nc < 4 else 0
            nc += 1

    # Build canonical board
    canon = list(canon_flop)
    for i in range(3, num_board):
        rank = board[i] >> 2
        suit = board[i] & 3
        canon.append(rank * 4 + suit_map[suit])
    return canon


def _compute_action_hash(actions):
    """Must match compute_action_hash in mccfr_blueprint.c exactly."""
    h = _mask64(0xFEDCBA9876543210)
    for a in actions:
        h = _hash_combine(h, _mask64(a * 17 + 3))
    return h


def _hash_combine(a, b):
    """Must match hash_combine in mccfr_blueprint.c exactly.
    a ^= b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2);
    All operations in uint64."""
    a = _mask64(a)
    b = _mask64(b)
    a = _mask64(a ^ _mask64(_mask64(b + 0x9e3779b97f4a7c15) + _mask64(a << 6) + (a >> 2)))
    return a


def board_to_texture_key(board_ints):
    """Map a 3-card flop to its suit-isomorphic texture key.

    Uses the same texture generation as solve_scenarios.py.
    Returns the canonical texture key (e.g., "AKQ_r", "T72_fd12").
    """
    # Sort by rank descending, keeping (rank, suit) pairs together
    pairs = sorted([(c // 4, c % 4) for c in board_ints], key=lambda x: -x[0])
    ranks = [r for r, s in pairs]
    suits = [s for r, s in pairs]

    # Determine suit pattern
    rank_str = "".join(RANKS[r] for r in ranks)

    if ranks[0] == ranks[1] == ranks[2]:
        return rank_str + "_r"  # trips: only rainbow
    elif ranks[0] == ranks[1] or ranks[1] == ranks[2]:
        # Paired board
        s0, s1, s2 = suits
        # Check which cards share the pair
        if ranks[0] == ranks[1]:
            if s0 == s1:
                return rank_str + "_fd"  # paired cards suited
            else:
                return rank_str + "_r"  # rainbow
        else:  # ranks[1] == ranks[2]
            if s1 == s2:
                return rank_str + "_fd"
            else:
                return rank_str + "_r"
    else:
        # Unpaired — check suit pattern
        s = suits
        if s[0] == s[1] == s[2]:
            return rank_str + "_m"  # monotone
        elif s[0] == s[1]:
            return rank_str + "_fd12"
        elif s[0] == s[2]:
            return rank_str + "_fd13"
        elif s[1] == s[2]:
            return rank_str + "_fd23"
        else:
            return rank_str + "_r"  # rainbow


class BlueprintV2:
    """Loader for .bps blueprint files.

    Supports two directory layouts:
      v1: blueprint_dir/worker-N/{texture_key}.bps  (flat, no scenarios)
      v2: blueprint_dir/worker-N/{scenario_id}/{texture_key}.bps  (scenario-filtered)

    For v2, use load_for_scenario() or set current_scenario to select which
    scenario's blueprints to serve. Falls back to v1 layout when no scenarios found.
    """

    def __init__(self, blueprint_dir: str, streets_to_load: Optional[List[int]] = None):
        """
        Args:
            blueprint_dir: directory containing worker-N/ subdirs with .bps files
            streets_to_load: which streets to load (default: [1] = flop only).
                Use [1, 2, 3] to load all streets.
        """
        self.blueprint_dir = blueprint_dir
        self.streets_to_load = streets_to_load or [1]  # flop by default

        # Loaded data: (scenario_id, texture_key) or texture_key -> info_set_table
        self._textures = {}
        self._metadata = {}

        # File index: supports both v1 and v2 layouts
        # v1: _file_index[texture_key] = path
        # v2: _scenario_file_index[(scenario_id, texture_key)] = path
        self._file_index = {}
        self._scenario_file_index = {}
        self._has_scenarios = False
        self.current_scenario = None  # set to filter by scenario
        self._build_file_index()

    def _build_file_index(self):
        """Scan blueprint_dir for .bps files (supports v1 flat and v2 scenario layouts)."""
        if not os.path.isdir(self.blueprint_dir):
            return
        for entry in os.listdir(self.blueprint_dir):
            subdir = os.path.join(self.blueprint_dir, entry)
            if not os.path.isdir(subdir):
                continue
            for item in os.listdir(subdir):
                item_path = os.path.join(subdir, item)
                if item.endswith('.bps') and os.path.isfile(item_path):
                    # v1 layout: worker-N/{texture}.bps
                    key = item[:-4]
                    self._file_index[key] = item_path
                elif os.path.isdir(item_path):
                    # v2 layout: worker-N/{scenario_id}/{texture}.bps
                    scenario_id = item
                    for fname in os.listdir(item_path):
                        if fname.endswith('.bps'):
                            tex_key = fname[:-4]
                            self._scenario_file_index[(scenario_id, tex_key)] = \
                                os.path.join(item_path, fname)
                            # Also index by texture_key alone for backward compat
                            # (first scenario found wins if no current_scenario set)
                            if tex_key not in self._file_index:
                                self._file_index[tex_key] = os.path.join(item_path, fname)
                            self._has_scenarios = True

    def available_textures(self) -> List[str]:
        return list(self._file_index.keys())

    def available_scenarios(self) -> List[str]:
        """Return list of scenario_ids found in v2 layout."""
        return sorted(set(sid for sid, _ in self._scenario_file_index.keys()))

    def is_loaded(self, texture_key: str, scenario_id: str = None) -> bool:
        if scenario_id:
            return (scenario_id, texture_key) in self._textures
        return texture_key in self._textures

    def load_texture(self, texture_key: str, scenario_id: str = None) -> bool:
        """Load a texture's .bps file into memory.

        Args:
            texture_key: texture identifier (e.g. "T72_r")
            scenario_id: optional scenario (e.g. "BB_vs_BTN_srp"). If None,
                         uses self.current_scenario, then falls back to any match.

        Only loads info sets matching self.streets_to_load.
        Returns True on success.
        """
        sid = scenario_id or self.current_scenario
        cache_key = (sid, texture_key) if sid else texture_key

        if cache_key in self._textures:
            return True

        # Find file: prefer scenario-specific, fall back to flat index
        fpath = None
        if sid and (sid, texture_key) in self._scenario_file_index:
            fpath = self._scenario_file_index[(sid, texture_key)]
        if not fpath:
            fpath = self._file_index.get(texture_key)
        if not fpath or not os.path.exists(fpath):
            return False

        with open(fpath, 'rb') as f:
            magic = f.read(4)
            if magic == b'BPS3':
                return self._load_bps3(f, cache_key, texture_key)
            elif magic == b'BPS2':
                return self._load_bps2(f, cache_key, texture_key)
            else:
                return False

    def _load_bps3(self, f, cache_key, texture_key) -> bool:
        """Load unified blueprint in BPS3 format (bucket-in-key).

        BPS3 outer format (Python wrapper):
            'BPS3' (4B) + uint64 compressed_size + uint32 meta_size
            + LZMA compressed data + JSON metadata

        BPS3 inner format (C binary):
            'BPS3' (4B) + uint32 num_entries + uint32 num_players
            Per entry: player(1) + street(1) + bucket(2) + board_hash(8)
                     + action_hash(8) + num_actions(1) + strategy[na] uint8
        """
        compressed_size, meta_size = struct.unpack('<QI', f.read(12))
        compressed = f.read(compressed_size)
        meta_bytes = f.read(meta_size)

        strat_data = lzma.decompress(compressed)
        meta = json.loads(meta_bytes.decode('utf-8'))

        # Parse BPS3 inner binary
        p = 0
        hdr = strat_data[p:p+4]; p += 4  # 'BPS3' magic
        n_entries = struct.unpack_from('<I', strat_data, p)[0]; p += 4
        n_players = struct.unpack_from('<I', strat_data, p)[0]; p += 4

        # BPS3 has one entry per (node, bucket). Group by node for the
        # same interface as BPS2: table[key] = [num_buckets, num_actions].
        # First pass: collect entries by node key.
        node_entries = {}  # (board_hash, action_hash, player, street) -> {bucket: strategy}
        loaded = 0

        for _ in range(n_entries):
            player = strat_data[p]; p += 1
            street = strat_data[p]; p += 1
            bucket = struct.unpack_from('<H', strat_data, p)[0]; p += 2
            board_hash = struct.unpack_from('<Q', strat_data, p)[0]; p += 8
            action_hash = struct.unpack_from('<Q', strat_data, p)[0]; p += 8
            na = strat_data[p]; p += 1

            if street in self.streets_to_load:
                raw = np.frombuffer(strat_data, dtype=np.uint8, count=na, offset=p)
                strat = raw.astype(np.float32) / 255.0
                s = strat.sum()
                if s > 0:
                    strat = strat / s

                key = (board_hash, action_hash, player, street)
                if key not in node_entries:
                    node_entries[key] = {}
                node_entries[key][bucket] = (na, strat)
                loaded += 1

            p += na

        # Build table: for each node, create [max_bucket+1, na] array
        num_buckets = meta.get('postflop_buckets', 200)
        if meta.get('preflop_buckets'):
            preflop_buckets = meta['preflop_buckets']
        else:
            preflop_buckets = 169

        table = {}
        for key, bucket_map in node_entries.items():
            street = key[3]
            nb = preflop_buckets if street == 0 else num_buckets
            # Determine na from any entry
            sample_na = next(iter(bucket_map.values()))[0]
            strategies = np.zeros((nb, sample_na), dtype=np.float32)
            # Uniform default for unvisited buckets
            strategies[:] = 1.0 / sample_na
            for bucket, (na, strat) in bucket_map.items():
                if bucket < nb:
                    strategies[bucket] = strat
            table[key] = strategies

        self._textures[cache_key] = table
        self._metadata[cache_key] = meta
        if cache_key != texture_key:
            self._textures[texture_key] = table
            self._metadata[texture_key] = meta
        return True

    def _load_bps2(self, f, cache_key, texture_key) -> bool:
        """Load per-scenario blueprint in BPS2 format (per-hand strategies)."""
        strat_size, meta_size = struct.unpack('<II', f.read(8))
        compressed = f.read(strat_size)
        meta_bytes = f.read(meta_size)

        strat_data = lzma.decompress(compressed)
        meta = json.loads(meta_bytes.decode('utf-8'))

        # Parse strategy entries into lookup table
        table = {}  # (board_hash, action_hash, player, street) -> strategies_per_bucket
        p = 0
        hdr = strat_data[p:p+4]; p += 4
        n_entries = struct.unpack_from('<I', strat_data, p)[0]; p += 4
        n_players = struct.unpack_from('<I', strat_data, p)[0]; p += 4

        loaded = 0
        skipped = 0
        for _ in range(n_entries):
            player = strat_data[p]; p += 1
            street = strat_data[p]; p += 1
            board_hash = struct.unpack_from('<Q', strat_data, p)[0]; p += 8
            action_hash = struct.unpack_from('<Q', strat_data, p)[0]; p += 8
            na = strat_data[p]; p += 1
            nh = struct.unpack_from('<H', strat_data, p)[0]; p += 2

            entry_size = na * nh

            if street in self.streets_to_load:
                # Parse strategies: [na * nh] uint8 values
                # Layout: for each hand/bucket h, na action probabilities
                strategies = np.frombuffer(strat_data, dtype=np.uint8,
                                            count=entry_size, offset=p)
                strategies = strategies.reshape(nh, na).astype(np.float32) / 255.0
                # Renormalize rows to sum to 1.0
                row_sums = strategies.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                strategies = strategies / row_sums

                key = (board_hash, action_hash, player, street)
                table[key] = strategies  # [nh, na] array
                loaded += 1
            else:
                skipped += 1

            p += entry_size

        self._textures[cache_key] = table
        self._metadata[cache_key] = meta
        # Also store under plain texture_key for backward compat
        if cache_key != texture_key:
            self._textures[texture_key] = table
            self._metadata[texture_key] = meta
        return True

    def load_unified(self, bps_path: str) -> bool:
        """Load a unified BPS3 blueprint file (covers all textures/streets).

        This is the primary loader for the Pluribus-style unified blueprint.
        The file contains strategies for all info sets across all streets.
        After loading, get_strategy() works for any board/action/player/bucket.

        Args:
            bps_path: path to unified_blueprint.bps

        Returns True on success.
        """
        if not os.path.exists(bps_path):
            return False

        with open(bps_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'BPS3':
                return False
            # Store under a special key; get_strategy will check it
            result = self._load_bps3(f, '__unified__', '__unified__')

        if result:
            self._unified_loaded = True
        return result

    def load_for_board(self, board_ints: List[int]) -> bool:
        """Load the texture for a given flop board."""
        if getattr(self, '_unified_loaded', False):
            return True  # unified blueprint covers all boards
        key = board_to_texture_key(board_ints)
        return self.load_texture(key)

    def get_strategy(self, board: List[int], action_history: List[int],
                     player: int, bucket: int,
                     street: int = 1) -> Optional[np.ndarray]:
        """Get P(action|bucket) at a specific game tree node.

        IMPORTANT: the board must use the canonical cards from the texture,
        not the actual game cards. Use get_canonical_board() to map.

        Args:
            board: CANONICAL board cards (from metadata flop_ints + dealt cards)
            action_history: sequence of action indices
            player: player index (0-5)
            bucket: hand bucket index (0-199)
            street: 1=flop, 2=turn, 3=river

        Returns:
            numpy array of action probabilities, or None if not found.
        """
        board_hash = _compute_board_hash(board, len(board))
        action_hash = _compute_action_hash(action_history)
        key = (board_hash, action_hash, player, street)

        # Check unified blueprint first
        if getattr(self, '_unified_loaded', False):
            table = self._textures.get('__unified__')
            if table is not None:
                strategies = table.get(key)
                if strategies is not None and bucket < len(strategies):
                    return strategies[bucket]

        # Fall back to per-texture lookup
        texture_key = board_to_texture_key(board[:3])
        if not self.load_texture(texture_key):
            return None

        table = self._textures.get(texture_key)
        if table is None:
            return None

        strategies = table.get(key)
        if strategies is None:
            return None

        if bucket >= len(strategies):
            return None

        return strategies[bucket]

    def get_canonical_board(self, board_ints: List[int]) -> Optional[List[int]]:
        """Map actual board cards to the canonical board used in the blueprint.

        The blueprint uses a canonical suit assignment per texture.
        This returns the flop_ints from the blueprint metadata.
        """
        tex = board_to_texture_key(board_ints[:3])
        meta = self._metadata.get(tex)
        if meta:
            return meta['flop_ints']
        # Try loading
        if self.load_texture(tex):
            meta = self._metadata.get(tex)
            if meta:
                return meta['flop_ints']
        return None

    def get_all_bucket_strategies(self, board: List[int],
                                   action_history: List[int],
                                   player: int,
                                   street: int = 1) -> Optional[np.ndarray]:
        """Get strategies for ALL buckets at a node.

        Returns [num_buckets, num_actions] array, or None.
        """
        board_hash = _compute_board_hash(board, len(board))
        action_hash = _compute_action_hash(action_history)
        key = (board_hash, action_hash, player, street)

        # Check unified blueprint first
        if getattr(self, '_unified_loaded', False):
            table = self._textures.get('__unified__')
            if table is not None:
                result = table.get(key)
                if result is not None:
                    return result

        # Fall back to per-texture lookup
        texture_key = board_to_texture_key(board[:3])
        if not self.load_texture(texture_key):
            return None

        table = self._textures.get(texture_key)
        if table is None:
            return None

        return table.get(key)

    def get_metadata(self, texture_key: str) -> Optional[dict]:
        """Get metadata for a loaded texture."""
        return self._metadata.get(texture_key)

    def get_root_strategy(self, board_ints: List[int], bucket: int) -> Optional[List[float]]:
        """Quick access to root strategy from metadata (no full load needed)."""
        texture_key = board_to_texture_key(board_ints[:3])
        meta = self._metadata.get(texture_key)
        if meta and 'root_strategies' in meta:
            strat = meta['root_strategies'].get(str(bucket))
            return strat
        return None

    def get_bucket_for_hand(self, texture_key: str, player: int,
                             hand_idx: int) -> Optional[int]:
        """Get bucket assignment for a hand from metadata."""
        meta = self._metadata.get(texture_key)
        if meta and 'bucket_assignments' in meta:
            p_str = str(player)
            assignments = meta['bucket_assignments'].get(p_str)
            if assignments and hand_idx < len(assignments):
                return assignments[hand_idx]
        return None

    def stats(self) -> dict:
        """Return loading statistics."""
        total_entries = sum(len(t) for t in self._textures.values())
        return {
            "indexed_textures": len(self._file_index),
            "loaded_textures": len(self._textures),
            "total_info_sets": total_entries,
            "streets_loaded": self.streets_to_load,
        }
