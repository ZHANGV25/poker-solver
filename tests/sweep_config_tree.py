#!/usr/bin/env python3
"""
Sweep alternative betting tree configurations and report total info set
counts. Tests the v3 question: which config knobs reduce the tree most?

Runs preflop-only enumeration for fast iteration on many configs, plus
full-tree enumeration for a small set of high-priority configs.

The DEPLOYED config is the v2 baseline. All others are reductions.
"""
import sys
import time
from copy import deepcopy

# Defaults match deployed v2
NUM_PLAYERS = 6
SMALL_BLIND = 50
BIG_BLIND = 100
INITIAL_STACK = 10000
ORDER_PRE = [(i + 2) % NUM_PLAYERS for i in range(NUM_PLAYERS)]
ORDER_POST = list(range(NUM_PLAYERS))


def generate_actions(pot, stack, to_call, num_raises, max_raises, sizes):
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


def enumerate_tree(cfg, preflop_only=False, time_limit=None):
    """Enumerate the betting tree, return (counts_per_street, walked, hit_limit)."""
    sys.setrecursionlimit(200000)
    counts = [0, 0, 0, 0]
    walked = [0]
    hit_limit = [False]
    t0 = time.time()

    PREFLOP_TIERS = cfg['preflop_tiers']
    PREFLOP_MAX_RAISES = cfg['preflop_max_raises']
    POSTFLOP_FIRST = cfg['postflop_first']
    POSTFLOP_SUBSEQUENT = cfg['postflop_subsequent']
    POSTFLOP_MAX_RAISES = cfg['postflop_max_raises']

    def walk(street, active, has_acted, bets, stacks, pot, num_raises,
             order, order_idx):
        if hit_limit[0]:
            return
        walked[0] += 1
        if walked[0] & 0xFFFFF == 0 and time_limit:
            if time.time() - t0 > time_limit:
                hit_limit[0] = True
                return

        ap = None
        for i in range(NUM_PLAYERS):
            p = order[(order_idx + i) % NUM_PLAYERS]
            if active[p]:
                ap = p
                order_idx = (order_idx + i) % NUM_PLAYERS
                break
        if ap is None:
            return

        active_bets = [bets[i] for i in range(NUM_PLAYERS) if active[i]]
        mx = max(active_bets) if active_bets else 0

        if all((not active[i]) or (has_acted[i] and bets[i] == mx)
               for i in range(NUM_PLAYERS)):
            n_active = sum(active)
            if n_active <= 1 or street == 3:
                return
            if preflop_only:
                return
            new_bets = [0] * NUM_PLAYERS
            new_has_acted = [False] * NUM_PLAYERS
            walk(street + 1, active, new_has_acted, new_bets, stacks, pot, 0,
                 ORDER_POST, 0)
            return

        counts[street] += 1
        to_call = mx - bets[ap]
        if to_call < 0:
            to_call = 0

        if street == 0:
            sizes = PREFLOP_TIERS[min(num_raises, max(PREFLOP_TIERS))]
            mr = PREFLOP_MAX_RAISES
        else:
            sizes = POSTFLOP_SUBSEQUENT if num_raises > 0 else POSTFLOP_FIRST
            mr = POSTFLOP_MAX_RAISES

        actions = generate_actions(pot, stacks[ap], to_call, num_raises, mr, sizes)

        for act in actions:
            new_active = list(active)
            new_bets = list(bets)
            new_stacks = list(stacks)
            new_has_acted = list(has_acted)
            new_pot = pot
            new_nr = num_raises

            if act == 'fold':
                new_active[ap] = False
                if sum(new_active) <= 1:
                    continue
            elif act in ('check', 'call'):
                if act == 'call':
                    new_bets[ap] = mx
                    new_stacks[ap] -= to_call
                    new_pot += to_call
                new_has_acted[ap] = True
            else:
                _, amt = act
                if amt > new_stacks[ap]:
                    amt = new_stacks[ap]
                new_bets[ap] += amt
                new_stacks[ap] -= amt
                new_pot += amt
                new_has_acted[ap] = True
                for p in range(NUM_PLAYERS):
                    if p != ap and new_active[p]:
                        new_has_acted[p] = False
                new_nr = num_raises + 1

            walk(street, new_active, new_has_acted, new_bets, new_stacks,
                 new_pot, new_nr, order, (order_idx + 1) % NUM_PLAYERS)

    init_active = [True] * NUM_PLAYERS
    init_has_acted = [False] * NUM_PLAYERS
    init_bets = [0] * NUM_PLAYERS
    init_stacks = [INITIAL_STACK] * NUM_PLAYERS
    init_bets[0] = SMALL_BLIND;  init_stacks[0] -= SMALL_BLIND
    init_bets[1] = BIG_BLIND;    init_stacks[1] -= BIG_BLIND
    init_pot = SMALL_BLIND + BIG_BLIND

    walk(0, init_active, init_has_acted, init_bets, init_stacks,
         init_pot, 0, ORDER_PRE, 0)
    return counts, walked[0], hit_limit[0]


# Configs to test (label → config dict)
DEPLOYED = {
    'preflop_tiers': {0: (0.5, 0.7, 1.0), 1: (0.7, 1.0), 2: (1.0,), 3: (8.0,)},
    'preflop_max_raises': 4,
    'postflop_first': (0.5, 1.0),
    'postflop_subsequent': (1.0,),
    'postflop_max_raises': 3,
}

CONFIGS = {
    'A_DEPLOYED': DEPLOYED,
    'B_premaxraises_3': {**DEPLOYED, 'preflop_max_raises': 3,
                          'preflop_tiers': {0: (0.5, 0.7, 1.0), 1: (0.7, 1.0), 2: (1.0,)}},
    'C_premax3_postmax2': {**DEPLOYED, 'preflop_max_raises': 3,
                            'preflop_tiers': {0: (0.5, 0.7, 1.0), 1: (0.7, 1.0), 2: (1.0,)},
                            'postflop_max_raises': 2},
    'D_postmax2_only': {**DEPLOYED, 'postflop_max_raises': 2},
    'E_postmax2_singlefirst': {**DEPLOYED, 'postflop_max_raises': 2,
                                'postflop_first': (1.0,)},
    'F_min_pluribus_likely': {  # very aggressive
        'preflop_tiers': {0: (0.5, 1.0), 1: (1.0,), 2: (1.0,)},
        'preflop_max_raises': 3,
        'postflop_first': (1.0,),
        'postflop_subsequent': (1.0,),
        'postflop_max_raises': 2,
    },
    'G_minimal_singlepre': {  # absolute floor
        'preflop_tiers': {0: (1.0,), 1: (1.0,)},
        'preflop_max_raises': 2,
        'postflop_first': (1.0,),
        'postflop_subsequent': (1.0,),
        'postflop_max_raises': 2,
    },
}


def main():
    full_tree = '--full' in sys.argv

    print("=" * 90)
    print(f"CONFIG SWEEP — {'FULL TREE' if full_tree else 'PREFLOP ONLY'}")
    print(f"All configs assume 6-max NLHE, 100bb stacks, 169 preflop / 200 postflop buckets")
    print("=" * 90)
    print()
    print(f"{'Config':<28} {'Pre nodes':>12} {'Flop':>10} {'Turn':>10} {'River':>10} "
          f"{'Total IS':>16} {'vs A':>8} {'time':>8}")
    print("-" * 110)

    baseline_is = None
    results = []
    for name, cfg in CONFIGS.items():
        t0 = time.time()
        counts, walked, hit_limit = enumerate_tree(cfg, preflop_only=not full_tree,
                                                    time_limit=900 if full_tree else None)
        elapsed = time.time() - t0

        # Compute info sets
        pre_is = counts[0] * 169
        post_is = sum(counts[s] * 200 for s in (1, 2, 3))
        total_is = pre_is + post_is

        if baseline_is is None:
            baseline_is = total_is
        ratio = total_is / baseline_is if baseline_is > 0 else 0

        flag = " ⚠TIMEOUT" if hit_limit else ""
        print(f"{name:<28} {counts[0]:>12,} {counts[1]:>10,} {counts[2]:>10,} "
              f"{counts[3]:>10,} {total_is:>16,} {ratio:>7.2%} {elapsed:>7.0f}s{flag}")
        results.append((name, counts, total_is, ratio, elapsed, hit_limit))

    print("-" * 110)
    print()
    print(f"Pluribus reference: 664,845,654 info sets")
    print()
    print("Reduction needed to match Pluribus:")
    for name, counts, total_is, ratio, _, _ in results:
        gap = total_is / 664_845_654
        print(f"  {name:<28} : {gap:>6.1f}x larger than Pluribus")


if __name__ == '__main__':
    main()
