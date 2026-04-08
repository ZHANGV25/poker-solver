#!/usr/bin/env python3
"""
H2 — Hypothesis 2 measurement: action_hash inflation factor.

Walks the full betting tree with the DEPLOYED v2 config and records each
decision node twice:

  (a) ACTION_HASH key  — what the C solver currently uses
        Distinguishes paths that reach the same logical state via
        different action sequences. Mirrors compute_action_hash() in
        src/mccfr_blueprint.c:415.

  (b) LOGICAL key      — what a Pluribus-style state-based encoding
        would use. Collapses all paths reaching the same (street,
        position, num_raises_in_round, active_set, has_acted_set,
        pot_bucket, to_call_bucket).

Reports the ratio (a)/(b) per street. That ratio is the multiplicative
inflation factor we'd remove by switching from action-history hashing
to logical-state hashing — i.e. how much smaller our tree could be
without changing the strategy granularity.

The inflation manifests in the C solver as 1.7B encountered info sets vs
Pluribus's 413M, an unexplained ~3.5x gap. This script measures how much
of that gap is attributable to action-history vs logical-state encoding.

Run:
    python3 tests/count_actionhash_vs_logical.py [--preflop-only]
"""
import sys
import time
from dataclasses import dataclass


# ── Deployed v2 config (mirrors precompute/blueprint_worker_unified.py) ──
NUM_PLAYERS = 6
SMALL_BLIND = 50
BIG_BLIND = 100
INITIAL_STACK = 10000   # 100bb in chips
ORDER_PRE = [(i + 2) % NUM_PLAYERS for i in range(NUM_PLAYERS)]   # UTG..BB
ORDER_POST = list(range(NUM_PLAYERS))                              # SB..BTN

PREFLOP_TIERS = {
    0: (0.5, 0.7, 1.0),   # open: 3 sizes
    1: (0.7, 1.0),        # 3-bet: 2 sizes
    2: (1.0,),            # 4-bet: 1 size
    3: (8.0,),            # 5-bet: shove
}
PREFLOP_MAX_RAISES = 4
POSTFLOP_FIRST_BET_SIZES = (0.5, 1.0)        # bet/donk
POSTFLOP_SUBSEQUENT_BET_SIZES = (1.0,)       # raise after a bet
POSTFLOP_MAX_RAISES = 3

POT_BUCKETS = 12         # 12 logarithmic pot tiers (1bb..1000bb+)


def pot_bucket(pot_chips: int) -> int:
    """Map pot in chips to a coarse logarithmic bucket.
    BIG_BLIND=100, so pot/100 is pot in bb. log2 buckets cap at POT_BUCKETS-1."""
    if pot_chips <= 0:
        return 0
    bb = pot_chips / BIG_BLIND
    if bb < 1: return 0
    # 1-2bb=1, 2-4bb=2, 4-8bb=3, ..., >2048bb=11
    import math
    b = int(math.log2(bb)) + 1
    if b >= POT_BUCKETS: b = POT_BUCKETS - 1
    return b


def to_call_bucket(to_call: int) -> int:
    """Coarse to_call bucket (4 bins). Captures 'no bet / small / medium / large'."""
    if to_call == 0: return 0
    bb = to_call / BIG_BLIND
    if bb < 2: return 1
    if bb < 10: return 2
    return 3


# ── Action generation (mirrors src/mccfr_blueprint.c:688) ──

def generate_actions(pot, stack, to_call, num_raises, max_raises, sizes):
    """Returns list of action labels: 'fold', 'check', 'call', or ('raise', amt)."""
    out = []
    if to_call > 0:
        out.append('fold')
    out.append('call' if to_call > 0 else 'check')

    if num_raises < max_raises:
        added_allin = False
        for bs in sizes:
            if to_call == 0:
                ba = int(bs * pot)
            else:
                ba = to_call + int(bs * (pot + to_call))
            if ba >= stack:
                ba = stack
            if ba <= to_call:
                continue
            if ba >= stack:
                if added_allin:
                    continue
                added_allin = True
            out.append(('raise', ba))
        if not added_allin and stack > to_call:
            out.append(('raise', stack))
    return out


# ── Tree walker ──

def walk_tree(preflop_only=False):
    """Walk the full betting tree and return per-street counts of:
       distinct action_hash keys, distinct logical keys."""

    # Per-street sets. We don't include the bucket dimension here — that's a
    # constant multiplier (169 preflop, 200 postflop) we apply at the end.
    action_keys = [set(), set(), set(), set()]   # one set per street
    # Three logical encodings, from coarsest to finest:
    #   coarse: (street, ap, num_active, num_raises, pot_tier)        — Pluribus-like minimal
    #   medium: (street, ap, active, has_acted, num_raises, pot_b, to_call_b)
    #   fine:   (street, ap, active, has_acted, num_raises, pot_chips, to_call, invested_tuple)
    logical_coarse = [set(), set(), set(), set()]
    logical_medium = [set(), set(), set(), set()]
    logical_fine = [set(), set(), set(), set()]
    nodes_visited = [0]

    sys.setrecursionlimit(100000)

    def walk(street, active, has_acted, bets, stacks, invested, pot, num_raises,
             order, order_idx, action_path):
        """action_path is a tuple of action indices since the start of the hand.
        invested is per-player cumulative investment (across all streets)."""
        nodes_visited[0] += 1
        if nodes_visited[0] % 5_000_000 == 0:
            print(f"  ... {nodes_visited[0]/1e6:.1f}M nodes walked, "
                  f"street {street}, |action|={len(action_keys[street])}, "
                  f"|coarse|={len(logical_coarse[street])} "
                  f"|med|={len(logical_medium[street])} "
                  f"|fine|={len(logical_fine[street])}", flush=True)

        # Find next active player
        ap = None
        idx = order_idx
        for _ in range(NUM_PLAYERS):
            p = order[idx % NUM_PLAYERS]
            if active[p]:
                ap = p
                order_idx = idx % NUM_PLAYERS
                break
            idx += 1
        if ap is None:
            return

        # Round done?
        active_bets = [bets[i] for i in range(NUM_PLAYERS) if active[i]]
        if not active_bets:
            return
        mx = max(active_bets)
        if all((not active[i]) or (has_acted[i] and bets[i] == mx)
               for i in range(NUM_PLAYERS)):
            n_active = sum(active)
            if n_active <= 1:
                return  # fold-out
            if street == 3 or preflop_only:
                return  # showdown or skip
            # Transition to next street
            new_bets = [0] * NUM_PLAYERS
            new_has_acted = [False] * NUM_PLAYERS
            walk(street + 1, active, new_has_acted, new_bets, stacks, invested,
                 pot, 0, ORDER_POST, 0, action_path)
            return

        # Decision node — record both keys
        to_call = mx - bets[ap]
        if to_call < 0:
            to_call = 0

        # KEY (a): action_hash analog. We don't compute the actual 64-bit
        # hash; we use the full action_path tuple PLUS player+street+bucket
        # which are part of the C BPInfoKey. (Bucket is constant — omit.)
        action_key = (street, ap, action_path)
        action_keys[street].add(action_key)

        active_tuple = tuple(active)
        has_acted_tuple = tuple(has_acted)
        n_active = sum(active)

        # COARSE: only the things Pluribus is documented to track per round
        logical_coarse[street].add((
            street, ap, n_active, num_raises, pot_bucket(pot),
        ))

        # MEDIUM: include the active+has_acted bitmaps and coarse to_call
        logical_medium[street].add((
            street, ap, active_tuple, has_acted_tuple, num_raises,
            pot_bucket(pot), to_call_bucket(to_call),
        ))

        # FINE: include exact pot, exact to_call, and per-player invested
        # (everything that affects EV without including PATH order)
        logical_fine[street].add((
            street, ap, active_tuple, has_acted_tuple, num_raises,
            pot, to_call, tuple(invested),
        ))

        # Generate actions
        if street == 0:
            sizes = PREFLOP_TIERS[min(num_raises, max(PREFLOP_TIERS))]
            mr = PREFLOP_MAX_RAISES
        else:
            sizes = POSTFLOP_SUBSEQUENT_BET_SIZES if num_raises > 0 else POSTFLOP_FIRST_BET_SIZES
            mr = POSTFLOP_MAX_RAISES

        actions = generate_actions(pot, stacks[ap], to_call, num_raises, mr, sizes)

        for ai, act in enumerate(actions):
            new_active = list(active)
            new_bets = list(bets)
            new_stacks = list(stacks)
            new_has_acted = list(has_acted)
            new_invested = list(invested)
            new_pot = pot
            new_nr = num_raises

            if act == 'fold':
                new_active[ap] = False
                if sum(new_active) <= 1:
                    continue   # fold-out terminal, no decision node
            elif act in ('check', 'call'):
                if act == 'call':
                    new_bets[ap] = mx
                    new_stacks[ap] -= to_call
                    new_invested[ap] += to_call
                    new_pot += to_call
                new_has_acted[ap] = True
            else:
                _, amt = act
                if amt > new_stacks[ap]:
                    amt = new_stacks[ap]
                new_bets[ap] += amt
                new_stacks[ap] -= amt
                new_invested[ap] += amt
                new_pot += amt
                new_has_acted[ap] = True
                for p in range(NUM_PLAYERS):
                    if p != ap and new_active[p]:
                        new_has_acted[p] = False
                new_nr = num_raises + 1

            walk(street, new_active, new_has_acted, new_bets, new_stacks,
                 new_invested, new_pot, new_nr, order,
                 (order_idx + 1) % NUM_PLAYERS,
                 action_path + (ai,))

    # Initialize state and post blinds
    init_active = [True] * NUM_PLAYERS
    init_has_acted = [False] * NUM_PLAYERS
    init_bets = [0] * NUM_PLAYERS
    init_stacks = [INITIAL_STACK] * NUM_PLAYERS
    init_invested = [0] * NUM_PLAYERS
    init_bets[0] = SMALL_BLIND;  init_stacks[0] -= SMALL_BLIND;  init_invested[0] = SMALL_BLIND
    init_bets[1] = BIG_BLIND;    init_stacks[1] -= BIG_BLIND;    init_invested[1] = BIG_BLIND
    init_pot = SMALL_BLIND + BIG_BLIND

    walk(0, init_active, init_has_acted, init_bets, init_stacks, init_invested,
         init_pot, 0, ORDER_PRE, 0, ())

    return action_keys, logical_coarse, logical_medium, logical_fine, nodes_visited[0]


def main():
    preflop_only = '--preflop-only' in sys.argv

    print("=" * 80)
    print("H2 MEASUREMENT — action_hash vs logical state inflation")
    print(f"Config: deployed v2 (PREFLOP_TIERS, max raises {PREFLOP_MAX_RAISES})")
    print(f"        postflop {POSTFLOP_FIRST_BET_SIZES} first / "
          f"{POSTFLOP_SUBSEQUENT_BET_SIZES} subsequent, max {POSTFLOP_MAX_RAISES} raises")
    print(f"Mode: {'PREFLOP ONLY' if preflop_only else 'FULL TREE'}")
    print("=" * 80)

    t0 = time.time()
    action_keys, log_coarse, log_med, log_fine, nodes = walk_tree(preflop_only=preflop_only)
    elapsed = time.time() - t0

    streets_to_show = [0] if preflop_only else [0, 1, 2, 3]
    street_names = ['preflop', 'flop', 'turn', 'river']
    bucket_mult = {0: 169, 1: 200, 2: 200, 3: 200}

    print(f"\nWalked {nodes:,} tree nodes in {elapsed:.1f}s")
    print()
    print(f"{'Street':<8} {'action_hash':>14} {'fine':>12} {'medium':>12} {'coarse':>10} "
          f"{'fine→IS (×B)':>18}")
    print("-" * 90)

    totals = {'act': 0, 'fine': 0, 'med': 0, 'coarse': 0,
              'act_is': 0, 'fine_is': 0, 'med_is': 0, 'coarse_is': 0}
    for s in streets_to_show:
        a = len(action_keys[s])
        f = len(log_fine[s])
        m = len(log_med[s])
        c = len(log_coarse[s])
        b = bucket_mult[s]
        totals['act'] += a;        totals['act_is'] += a * b
        totals['fine'] += f;       totals['fine_is'] += f * b
        totals['med'] += m;        totals['med_is'] += m * b
        totals['coarse'] += c;     totals['coarse_is'] += c * b
        print(f"{street_names[s]:<8} {a:>14,} {f:>12,} {m:>12,} {c:>10,} {f*b:>18,}")

    print("-" * 90)
    print(f"{'TOTAL':<8} {totals['act']:>14,} {totals['fine']:>12,} "
          f"{totals['med']:>12,} {totals['coarse']:>10,} {totals['fine_is']:>18,}")
    print()

    def ratio(num, den):
        return f"{num/den:.2f}x" if den > 0 else "n/a"

    print(f"Inflation factors (action_hash / logical):")
    print(f"  vs FINE   (per-player invested + exact pot): {ratio(totals['act'], totals['fine'])}")
    print(f"  vs MEDIUM (active+has_acted bitmaps + tier): {ratio(totals['act'], totals['med'])}")
    print(f"  vs COARSE (n_active + raises + pot tier):    {ratio(totals['act'], totals['coarse'])}")
    print()
    print(f"Pluribus reported encountered info sets = 413,507,309")
    print(f"v2 projected info sets at 8B            ≈ 1,700,000,000  (action_hash encoding)")
    print()
    print(f"Predicted info sets at our deployed bucket counts (169/200):")
    print(f"  with action_hash:   {totals['act_is']:>15,}  (current)")
    print(f"  with FINE encoding: {totals['fine_is']:>15,}  → gap to Pluribus: {ratio(totals['fine_is'], 413e6)}")
    print(f"  with MEDIUM enc:    {totals['med_is']:>15,}  → gap to Pluribus: {ratio(totals['med_is'], 413e6)}")
    print(f"  with COARSE enc:    {totals['coarse_is']:>15,}  → gap to Pluribus: {ratio(totals['coarse_is'], 413e6)}")


if __name__ == '__main__':
    main()
