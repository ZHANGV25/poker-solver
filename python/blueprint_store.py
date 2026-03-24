"""Blueprint Store — binary format for precomputed Pluribus-style strategies.

Stores per-hand weighted-average strategies at flop roots and turn roots
for all textures within a scenario. Optimized for fast random access.

Format (per scenario directory):
    index.bin   — maps texture_key → (offset, size) in data.bin
    data.bin    — concatenated LZMA-compressed strategy blobs

Per texture blob (before compression):
    Header:
        uint16 num_hands_oop, num_hands_ip
        uint8  num_flop_actions
        uint8  num_turn_actions
        uint8  num_turn_cards  (typically 49)
    Flop root:
        For each player (oop, ip):
            float32[num_hands][num_flop_actions]  — weighted avg P(action|hand)
        float32[num_hands_oop]   — per-hand EV for OOP
        float32[num_hands_ip]    — per-hand EV for IP
        Per-action EV for each player:
            float32[num_flop_actions][num_hands]  — EV after taking each action
    Turn roots (one per turn card):
        uint8 turn_card
        uint16 num_hands_oop_tc, num_hands_ip_tc  (after board-blocking)
        For each player:
            float32[num_hands_tc][num_turn_actions] — weighted avg P(action|hand)
        float32[num_hands_oop_tc]   — per-hand EV for OOP
        float32[num_hands_ip_tc]    — per-hand EV for IP
        Per-action EV:
            float32[num_turn_actions][num_hands_tc]

The hand ordering matches the solver's internal ordering: hands are sorted
by (card0, card1) with card0 < card1, board-blocked hands removed.
"""

import json
import lzma
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from solver import card_to_int, int_to_card, parse_range_string
    from blueprint_io import texture_key, remap_hand
except ImportError:
    from python.solver import card_to_int, int_to_card, parse_range_string
    from python.blueprint_io import texture_key, remap_hand


# ── Index format ─────────────────────────────────────────────────────────

# Index entry: texture_key (null-padded to 16 bytes) + uint64 offset + uint32 size
INDEX_KEY_LEN = 16
INDEX_ENTRY_FMT = f'{INDEX_KEY_LEN}sQI'
INDEX_ENTRY_SIZE = struct.calcsize(INDEX_ENTRY_FMT)
INDEX_MAGIC = b'BPST0002'


def _pack_tex_key(key: str) -> bytes:
    """Pad texture key to INDEX_KEY_LEN bytes."""
    b = key.encode('ascii')
    return b[:INDEX_KEY_LEN].ljust(INDEX_KEY_LEN, b'\x00')


def _unpack_tex_key(raw: bytes) -> str:
    """Unpack null-padded texture key."""
    return raw.rstrip(b'\x00').decode('ascii')


# ── Blob packing/unpacking ───────────────────────────────────────────────

def pack_texture_blob(
    oop_hands: List[Tuple[int, int]],
    ip_hands: List[Tuple[int, int]],
    flop_strategies: Dict,   # {player_idx: np.array[num_hands, num_actions]}
    flop_evs: Dict,          # {player_idx: np.array[num_hands]}
    flop_action_evs: Dict,   # {player_idx: np.array[num_actions, num_hands]}
    turn_data: List[Dict],   # one per turn card, sorted by card id
    num_flop_actions: int,
    num_turn_actions: int,
) -> bytes:
    """Pack a texture's strategies into a binary blob."""
    # Get hand counts from strategy arrays (more reliable than hand lists
    # since the strategies are what actually gets stored)
    nh0 = flop_strategies[0].shape[0]
    nh1 = flop_strategies[1].shape[0]
    ntc = len(turn_data)

    parts = []

    # Header
    parts.append(struct.pack('<HHBBBx',
                             nh0, nh1,
                             num_flop_actions, num_turn_actions,
                             ntc))

    # Hand card pairs: [(card0, card1), ...] for each player
    # Stored as uint8 pairs so reader can map (card0, card1) → hand index
    for p, hands in enumerate([oop_hands, ip_hands]):
        nh = nh0 if p == 0 else nh1
        for i in range(nh):
            if i < len(hands):
                c0, c1 = hands[i][0], hands[i][1]
            else:
                c0, c1 = 255, 255  # sentinel for missing
            parts.append(struct.pack('BB', c0, c1))

    # Flop root strategies (both players)
    for p in range(2):
        strat = flop_strategies[p]  # [num_hands, num_actions]
        parts.append(strat.astype(np.float32).tobytes())

    # Flop root EVs
    for p in range(2):
        nh = nh0 if p == 0 else nh1
        ev = flop_evs.get(p, np.zeros(nh, dtype=np.float32))
        parts.append(ev.astype(np.float32).tobytes())

    # Flop per-action EVs
    for p in range(2):
        nh = nh0 if p == 0 else nh1
        na = num_flop_actions
        action_ev = flop_action_evs.get(p, np.zeros((na, nh), dtype=np.float32))
        parts.append(action_ev.astype(np.float32).tobytes())

    # Turn roots
    for td in turn_data:
        tc = td['turn_card']
        nh0_tc = td['num_hands_oop']
        nh1_tc = td['num_hands_ip']
        parts.append(struct.pack('<BHH', tc, nh0_tc, nh1_tc))

        for p in range(2):
            strat = td['strategies'][p]
            parts.append(strat.astype(np.float32).tobytes())

        for p in range(2):
            nh = nh0_tc if p == 0 else nh1_tc
            ev = td.get('evs', {}).get(p, np.zeros(nh, dtype=np.float32))
            parts.append(ev.astype(np.float32).tobytes())

        for p in range(2):
            nh = nh0_tc if p == 0 else nh1_tc
            na = num_turn_actions
            action_ev = td.get('action_evs', {}).get(p,
                              np.zeros((na, nh), dtype=np.float32))
            parts.append(action_ev.astype(np.float32).tobytes())

    raw = b''.join(parts)
    return lzma.compress(raw, preset=6)


def unpack_texture_blob(compressed: bytes) -> Dict:
    """Unpack a compressed texture blob to structured data."""
    raw = lzma.decompress(compressed)
    off = 0

    # Header
    nh0, nh1, nfa, nta, ntc = struct.unpack_from('<HHBBBx', raw, off)
    off += 8

    result = {
        'num_hands_oop': nh0,
        'num_hands_ip': nh1,
        'num_flop_actions': nfa,
        'num_turn_actions': nta,
        'num_turn_cards': ntc,
    }

    # Hand card pairs
    hand_cards = {}
    hand_index = {}  # {player: {(c0,c1): index}}
    for p in range(2):
        nh = nh0 if p == 0 else nh1
        cards = []
        idx_map = {}
        for i in range(nh):
            c0, c1 = struct.unpack_from('BB', raw, off)
            off += 2
            cards.append((c0, c1))
            key = (min(c0, c1), max(c0, c1))
            idx_map[key] = i
        hand_cards[p] = cards
        hand_index[p] = idx_map
    result['hand_cards'] = hand_cards
    result['hand_index'] = hand_index

    # Flop strategies
    flop_strats = {}
    for p in range(2):
        nh = nh0 if p == 0 else nh1
        sz = nh * nfa * 4
        flop_strats[p] = np.frombuffer(raw[off:off + sz], dtype=np.float32).reshape(nh, nfa).copy()
        off += sz
    result['flop_strategies'] = flop_strats

    # Flop EVs
    flop_evs = {}
    for p in range(2):
        nh = nh0 if p == 0 else nh1
        sz = nh * 4
        flop_evs[p] = np.frombuffer(raw[off:off + sz], dtype=np.float32).copy()
        off += sz
    result['flop_evs'] = flop_evs

    # Flop action EVs
    flop_action_evs = {}
    for p in range(2):
        nh = nh0 if p == 0 else nh1
        sz = nfa * nh * 4
        flop_action_evs[p] = np.frombuffer(raw[off:off + sz],
                                            dtype=np.float32).reshape(nfa, nh).copy()
        off += sz
    result['flop_action_evs'] = flop_action_evs

    # Turn roots
    turn_data = []
    for _ in range(ntc):
        tc, nh0_tc, nh1_tc = struct.unpack_from('<BHH', raw, off)
        off += 5

        strats = {}
        for p in range(2):
            nh = nh0_tc if p == 0 else nh1_tc
            sz = nh * nta * 4
            strats[p] = np.frombuffer(raw[off:off + sz],
                                       dtype=np.float32).reshape(nh, nta).copy()
            off += sz

        evs = {}
        for p in range(2):
            nh = nh0_tc if p == 0 else nh1_tc
            sz = nh * 4
            evs[p] = np.frombuffer(raw[off:off + sz], dtype=np.float32).copy()
            off += sz

        action_evs = {}
        for p in range(2):
            nh = nh0_tc if p == 0 else nh1_tc
            sz = nta * nh * 4
            action_evs[p] = np.frombuffer(raw[off:off + sz],
                                           dtype=np.float32).reshape(nta, nh).copy()
            off += sz

        turn_data.append({
            'turn_card': tc,
            'num_hands_oop': nh0_tc,
            'num_hands_ip': nh1_tc,
            'strategies': strats,
            'evs': evs,
            'action_evs': action_evs,
        })

    result['turn_data'] = turn_data
    return result


# ── BlueprintStore class ─────────────────────────────────────────────────

class BlueprintStore:
    """Read/write precomputed blueprint strategies in binary format.

    Usage (write):
        store = BlueprintStore("flop_blueprints/CO_vs_BB_srp", mode='w')
        store.write_texture("AKQ_r", blob_data)
        store.close()

    Usage (read):
        store = BlueprintStore("flop_blueprints/CO_vs_BB_srp")
        data = store.load_texture("AKQ_r")
        flop_strat = data['flop_strategies'][0]  # OOP's flop strategy
        turn_strat = data['turn_data'][3]['strategies'][1]  # IP's turn strategy for 4th card
    """

    def __init__(self, scenario_dir: str, mode: str = 'r'):
        self.scenario_dir = scenario_dir
        self.mode = mode
        self._index = {}       # tex_key -> (offset, size)
        self._cache = {}       # tex_key -> unpacked dict
        self._cache_order = [] # LRU order
        self._max_cache = 20

        self._data_fh = None   # file handle for data.bin (write mode)

        if mode == 'r':
            self._load_index()
        elif mode == 'w':
            os.makedirs(scenario_dir, exist_ok=True)
            self._data_fh = open(os.path.join(scenario_dir, 'data.bin'), 'wb')
            self._data_fh.write(INDEX_MAGIC)  # placeholder, will be overwritten

    def _load_index(self):
        """Load index.bin into memory."""
        idx_path = os.path.join(self.scenario_dir, 'index.bin')
        if not os.path.exists(idx_path):
            return

        with open(idx_path, 'rb') as f:
            magic = f.read(8)
            if magic != INDEX_MAGIC:
                raise ValueError(f"Bad index magic: {magic}")
            data = f.read()

        n_entries = len(data) // INDEX_ENTRY_SIZE
        for i in range(n_entries):
            off = i * INDEX_ENTRY_SIZE
            key_raw, offset, size = struct.unpack_from(INDEX_ENTRY_FMT, data, off)
            key = _unpack_tex_key(key_raw)
            self._index[key] = (offset, size)

    def write_texture(self, tex_key: str, compressed_blob: bytes):
        """Write a compressed texture blob to data.bin and record in index."""
        if self.mode != 'w':
            raise RuntimeError("Store not opened for writing")

        offset = self._data_fh.tell()
        self._data_fh.write(compressed_blob)
        self._index[tex_key] = (offset, len(compressed_blob))

    def close(self):
        """Finalize: write index.bin and close data file."""
        if self.mode == 'w' and self._data_fh:
            self._data_fh.close()
            self._data_fh = None

            # Write index
            idx_path = os.path.join(self.scenario_dir, 'index.bin')
            with open(idx_path, 'wb') as f:
                f.write(INDEX_MAGIC)
                for key in sorted(self._index.keys()):
                    offset, size = self._index[key]
                    f.write(struct.pack(INDEX_ENTRY_FMT,
                                        _pack_tex_key(key), offset, size))

    def load_texture(self, tex_key: str) -> Optional[Dict]:
        """Load and unpack a texture's data. Returns None if not found."""
        if tex_key in self._cache:
            return self._cache[tex_key]

        if tex_key not in self._index:
            return None

        offset, size = self._index[tex_key]
        data_path = os.path.join(self.scenario_dir, 'data.bin')
        with open(data_path, 'rb') as f:
            f.seek(offset)
            compressed = f.read(size)

        result = unpack_texture_blob(compressed)

        # LRU cache
        if len(self._cache) >= self._max_cache and self._cache_order:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        self._cache[tex_key] = result
        self._cache_order.append(tex_key)

        return result

    def has_texture(self, tex_key: str) -> bool:
        return tex_key in self._index

    @property
    def num_textures(self) -> int:
        return len(self._index)

    @property
    def available_textures(self) -> List[str]:
        return sorted(self._index.keys())

    def get_flop_strategy(self, board_cards: List[str], player: int) -> Optional[np.ndarray]:
        """Get flop root strategy for a player given actual board cards.

        Handles suit isomorphism: maps actual board to canonical texture.

        Args:
            board_cards: ["Qs", "As", "2d"] (3 cards)
            player: 0 (OOP) or 1 (IP)

        Returns:
            np.array[num_hands, num_actions] or None
        """
        tex_key_str, suit_map = texture_key(board_cards)
        data = self.load_texture(tex_key_str)
        if data is None:
            return None
        return data['flop_strategies'][player]

    def get_turn_strategy(self, board_cards: List[str], turn_card: str,
                          player: int) -> Optional[np.ndarray]:
        """Get turn root strategy for a player given flop + turn card.

        Args:
            board_cards: ["Qs", "As", "2d"] (3 flop cards)
            turn_card: "7h" (the turn card)
            player: 0 (OOP) or 1 (IP)

        Returns:
            np.array[num_hands_after_blocking, num_actions] or None
        """
        tex_key_str, suit_map = texture_key(board_cards)
        data = self.load_texture(tex_key_str)
        if data is None:
            return None

        # Map the turn card through suit isomorphism
        tc_rank = turn_card[0]
        tc_suit = turn_card[1]
        canonical_suit = suit_map.get(tc_suit, tc_suit)
        canonical_tc = card_to_int(tc_rank + canonical_suit)

        # Find this turn card in the stored data
        for td in data['turn_data']:
            if td['turn_card'] == canonical_tc:
                return td['strategies'][player]

        return None

    def get_flop_ev(self, board_cards: List[str], player: int) -> Optional[np.ndarray]:
        """Get per-hand EV at flop root."""
        tex_key_str, _ = texture_key(board_cards)
        data = self.load_texture(tex_key_str)
        if data is None:
            return None
        return data['flop_evs'].get(player)

    def get_turn_ev(self, board_cards: List[str], turn_card: str,
                    player: int) -> Optional[np.ndarray]:
        """Get per-hand EV at turn root."""
        tex_key_str, suit_map = texture_key(board_cards)
        data = self.load_texture(tex_key_str)
        if data is None:
            return None

        tc_rank = turn_card[0]
        tc_suit = turn_card[1]
        canonical_suit = suit_map.get(tc_suit, tc_suit)
        canonical_tc = card_to_int(tc_rank + canonical_suit)

        for td in data['turn_data']:
            if td['turn_card'] == canonical_tc:
                return td['evs'].get(player)
        return None

    def get_turn_action_evs(self, board_cards: List[str], turn_card: str,
                            player: int) -> Optional[np.ndarray]:
        """Get per-action per-hand EV at turn root."""
        tex_key_str, suit_map = texture_key(board_cards)
        data = self.load_texture(tex_key_str)
        if data is None:
            return None

        tc_rank = turn_card[0]
        tc_suit = turn_card[1]
        canonical_suit = suit_map.get(tc_suit, tc_suit)
        canonical_tc = card_to_int(tc_rank + canonical_suit)

        for td in data['turn_data']:
            if td['turn_card'] == canonical_tc:
                return td['action_evs'].get(player)
        return None

    def __del__(self):
        if self.mode == 'w' and self._data_fh:
            self.close()
