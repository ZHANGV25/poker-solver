#!/usr/bin/env python3
"""
Enumerate the FULL betting tree (preflop + flop + turn + river) for the
deployed v2 config. Memory-conscious: only counts decision nodes per street,
no per-node storage. Reports the saturation upper bound for total info sets.

This is the apples-to-apples comparison to Pluribus's 664,845,654 total
action sequences in the blueprint action abstraction.
"""
import sys
import time

# Deployed v2 config (mirrors precompute/blueprint_worker_unified.py)
NUM_PLAYERS = 6
SMALL_BLIND = 50
BIG_BLIND = 100
INITIAL_STACK = 10000
ORDER_PRE = [(i + 2) % NUM_PLAYERS for i in range(NUM_PLAYERS)]
ORDER_POST = list(range(NUM_PLAYERS))

PREFLOP_TIERS = {
    0: (0.5, 0.7, 1.0),
    1: (0.7, 1.0),
    2: (1.0,),
    3: (8.0,),
}
PREFLOP_MAX_RAISES = 4
POSTFLOP_FIRST = (0.5, 1.0)
POSTFLOP_SUBSEQUENT = (1.0,)
POSTFLOP_MAX_RAISES = 3


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


def main():
    sys.setrecursionlimit(200000)
    counts = [0, 0, 0, 0]   # decision nodes per street
    nodes_walked = [0]
    last_print = [time.time()]
    t0 = time.time()

    def walk(street, active, has_acted, bets, stacks, pot, num_raises,
             order, order_idx):
        nodes_walked[0] += 1
        if nodes_walked[0] & 0xFFFFF == 0:
            now = time.time()
            if now - last_print[0] > 5:
                last_print[0] = now
                print(f"  ... {nodes_walked[0]/1e6:.1f}M walked, "
                      f"counts={counts}, elapsed={now-t0:.0f}s", flush=True)

        # Find next active player
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

        # Round done?
        if all((not active[i]) or (has_acted[i] and bets[i] == mx)
               for i in range(NUM_PLAYERS)):
            n_active = sum(active)
            if n_active <= 1 or street == 3:
                return
            # Transition to next street
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

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.0f}s ({nodes_walked[0]:,} nodes walked)")
    print()
    print(f"{'Street':<10} {'Decision nodes':>15} {'Buckets':>10} {'Info sets':>18}")
    print("-" * 60)
    bucket_mult = [169, 200, 200, 200]
    total_is = 0
    for s, name in enumerate(['preflop', 'flop', 'turn', 'river']):
        is_count = counts[s] * bucket_mult[s]
        total_is += is_count
        print(f"{name:<10} {counts[s]:>15,} {bucket_mult[s]:>10} {is_count:>18,}")
    print("-" * 60)
    print(f"{'TOTAL':<10} {sum(counts):>15,} {'':>10} {total_is:>18,}")
    print()
    print(f"Pluribus reference (664M info sets): {664_845_654:,}")
    print(f"Our enumerated tree: {total_is:,}")
    print(f"Ratio: {total_is / 664_845_654:.2f}x Pluribus")


if __name__ == '__main__':
    main()
