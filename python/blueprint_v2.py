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
    """Must match compute_board_hash in mccfr_blueprint.c exactly."""
    h = _mask64(0x123456789ABCDEF)
    for i in range(num_board):
        h = _hash_combine(h, _mask64(board[i] * 31 + 7))
    return h


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
    # Sort by rank descending
    ranks = sorted([c // 4 for c in board_ints], reverse=True)
    suits = [board_ints[i] % 4 for i in range(3)]

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
    """Loader for .bps blueprint files."""

    def __init__(self, blueprint_dir: str, streets_to_load: Optional[List[int]] = None):
        """
        Args:
            blueprint_dir: directory containing worker-N/ subdirs with .bps files
            streets_to_load: which streets to load (default: [1] = flop only).
                Use [1, 2, 3] to load all streets.
        """
        self.blueprint_dir = blueprint_dir
        self.streets_to_load = streets_to_load or [1]  # flop by default

        # Loaded data: texture_key -> {(board_hash, action_hash, player, street) -> strategy_array}
        self._textures = {}  # texture_key -> info_set_table
        self._metadata = {}  # texture_key -> metadata dict

        # File index: texture_key -> file path
        self._file_index = {}
        self._build_file_index()

    def _build_file_index(self):
        """Scan blueprint_dir for .bps files."""
        if not os.path.isdir(self.blueprint_dir):
            return
        for entry in os.listdir(self.blueprint_dir):
            subdir = os.path.join(self.blueprint_dir, entry)
            if not os.path.isdir(subdir):
                continue
            for fname in os.listdir(subdir):
                if fname.endswith('.bps'):
                    key = fname[:-4]
                    self._file_index[key] = os.path.join(subdir, fname)

    def available_textures(self) -> List[str]:
        return list(self._file_index.keys())

    def is_loaded(self, texture_key: str) -> bool:
        return texture_key in self._textures

    def load_texture(self, texture_key: str) -> bool:
        """Load a texture's .bps file into memory.

        Only loads info sets matching self.streets_to_load.
        Returns True on success.
        """
        if texture_key in self._textures:
            return True

        fpath = self._file_index.get(texture_key)
        if not fpath or not os.path.exists(fpath):
            return False

        with open(fpath, 'rb') as f:
            magic = f.read(4)
            if magic != b'BPS2':
                return False
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

        self._textures[texture_key] = table
        self._metadata[texture_key] = meta
        return True

    def load_for_board(self, board_ints: List[int]) -> bool:
        """Load the texture for a given flop board."""
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
        texture_key = board_to_texture_key(board[:3])
        if not self.load_texture(texture_key):
            return None

        table = self._textures.get(texture_key)
        if table is None:
            return None

        board_hash = _compute_board_hash(board, len(board))
        action_hash = _compute_action_hash(action_history)
        key = (board_hash, action_hash, player, street)

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
        texture_key = board_to_texture_key(board[:3])
        if not self.load_texture(texture_key):
            return None

        table = self._textures.get(texture_key)
        if table is None:
            return None

        board_hash = _compute_board_hash(board, len(board))
        action_hash = _compute_action_hash(action_history)
        key = (board_hash, action_hash, player, street)

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
