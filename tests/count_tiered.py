#!/usr/bin/env python3
"""
Count preflop action sequences with TIERED bet sizing.
Sizes vary by raise level (num_raises), matching Pluribus's approach.
"""
import time

NUM_PLAYERS = 6
SB = 1
BB = 2
STACK = 200  # 100bb
ORDER = [(i + 2) % NUM_PLAYERS for i in range(NUM_PLAYERS)]


def generate_actions(pot, stack, to_call, num_raises, max_raises, sizes_by_level):
    """Generate actions. sizes_by_level[num_raises] = tuple of bet fractions."""
    actions = []
    if to_call > 0:
        actions.append('fold')
    actions.append('checkcall')

    if num_raises < max_raises:
        sizes = sizes_by_level[min(num_raises, len(sizes_by_level) - 1)]
        amounts = []
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
            if ba not in [a for _, a in amounts]:
                amounts.append(('raise', ba))
        if not added_allin and stack > to_call:
            amounts.append(('raise', stack))
        actions.extend(amounts)
    return actions


def count_preflop(sizes_by_level, max_raises, label=""):
    decision_nodes = 0
    terminals_fold = 0
    terminals_flop = 0

    init_bets = [0] * NUM_PLAYERS
    init_stacks = [STACK] * NUM_PLAYERS
    init_bets[0] = SB;  init_stacks[0] -= SB
    init_bets[1] = BB;  init_stacks[1] -= BB

    # Stack-based DFS: (active, has_acted, bets, stacks, pot, num_raises, order_idx)
    stack = [(
        tuple([True]*NUM_PLAYERS),
        tuple([False]*NUM_PLAYERS),
        tuple(init_bets),
        tuple(init_stacks),
        SB + BB,
        0, 0
    )]

    t0 = time.time()
    explored = 0

    while stack:
        active, has_acted, bets, stacks, pot, num_raises, order_idx = stack.pop()
        explored += 1

        if explored % 2_000_000 == 0:
            print(f"  {explored/1e6:.0f}M explored, {decision_nodes/1e6:.1f}M decisions, "
                  f"{time.time()-t0:.0f}s", flush=True)

        # Find next active player
        ap = None
        for i in range(NUM_PLAYERS):
            p = ORDER[(order_idx + i) % NUM_PLAYERS]
            if active[p]:
                ap = p
                order_idx = (order_idx + i) % NUM_PLAYERS
                break
        if ap is None:
            continue

        # Round done?
        mx = max(bets[i] for i in range(NUM_PLAYERS) if active[i])
        if all((not active[i]) or (has_acted[i] and bets[i] == mx) for i in range(NUM_PLAYERS)):
            terminals_flop += 1
            continue

        decision_nodes += 1
        to_call = mx - bets[ap]
        remaining = stacks[ap]
        next_idx = (order_idx + 1) % NUM_PLAYERS

        actions = generate_actions(pot, remaining, to_call, num_raises, max_raises, sizes_by_level)

        for act in actions:
            if act == 'fold':
                na = list(active); na[ap] = False
                if sum(na) <= 1:
                    terminals_fold += 1
                else:
                    stack.append((tuple(na), has_acted, bets, stacks, pot, num_raises, next_idx))
            elif act == 'checkcall':
                nb = list(bets); ns = list(stacks); nh = list(has_acted)
                if to_call > 0:
                    nb[ap] = mx; ns[ap] -= to_call
                nh[ap] = True
                stack.append((active, tuple(nh), tuple(nb), tuple(ns), pot + (to_call if to_call > 0 else 0), num_raises, next_idx))
            else:
                _, amt = act
                nb = list(bets); ns = list(stacks)
                nh = [False]*NUM_PLAYERS
                if amt > ns[ap]: amt = ns[ap]
                nb[ap] += amt; ns[ap] -= amt
                nh[ap] = True
                for p in range(NUM_PLAYERS):
                    if not active[p]: nh[p] = has_acted[p]
                stack.append((active, tuple(nh), tuple(nb), tuple(ns), pot + amt, num_raises + 1, next_idx))

    elapsed = time.time() - t0
    return decision_nodes, terminals_fold, terminals_flop, elapsed


def main():
    configs = [
        # (label, sizes_by_level, max_raises)
        # sizes_by_level[i] = sizes to use when num_raises == i

        ("CURRENT: 8 flat, max 4",
         {0: (0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0),
          1: (0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0),
          2: (0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0),
          3: (0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0)}, 4),

        ("TIERED A: 8/3/2/allin, max 4",
         {0: (0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0),
          1: (0.7, 1.0, 2.5),
          2: (1.0, 4.0),
          3: (8.0,)}, 4),    # 5-bet = big shove

        ("TIERED B: 8/4/2/allin, max 4",
         {0: (0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0),
          1: (0.5, 0.7, 1.0, 2.5),
          2: (1.0, 4.0),
          3: (8.0,)}, 4),

        ("TIERED C: 8/3/1/allin, max 4",
         {0: (0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0),
          1: (0.7, 1.0, 2.5),
          2: (2.5,),          # 4-bet = one big sizing
          3: (8.0,)}, 4),

        ("TIERED D: 6/3/2/allin, max 4",
         {0: (0.4, 0.5, 0.7, 1.0, 2.5, 8.0),
          1: (0.7, 1.0, 2.5),
          2: (1.0, 4.0),
          3: (8.0,)}, 4),

        ("FLAT 4 sizes, max 3 (baseline)",
         {0: (0.5, 1.0, 2.5, 8.0),
          1: (0.5, 1.0, 2.5, 8.0),
          2: (0.5, 1.0, 2.5, 8.0)}, 3),

        ("TIERED A: 8/3/2/allin, max 3",
         {0: (0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0),
          1: (0.7, 1.0, 2.5),
          2: (1.0, 4.0)}, 3),

        ("TIERED B: 8/4/2, max 3",
         {0: (0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0),
          1: (0.5, 0.7, 1.0, 2.5),
          2: (1.0, 4.0)}, 3),
    ]

    print("="*80)
    print("TIERED PREFLOP SIZING ENUMERATION")
    print("6-player NLHE, 100bb deep")
    print("="*80)

    results = []
    for label, sizes, max_r in configs:
        print(f"\n--- {label} ---", flush=True)
        for lvl in sorted(sizes.keys()):
            if lvl < max_r:
                print(f"    Raise {lvl}: {sizes[lvl]}", flush=True)
        dn, tf, tfl, elapsed = count_preflop(sizes, max_r, label)
        pre_is = dn * 169
        print(f"    Decision nodes:      {dn:>15,}")
        print(f"    Lines to flop:       {tfl:>15,}")
        print(f"    Preflop info sets:   {pre_is:>15,} (× 169)")
        print(f"    Time: {elapsed:.1f}s", flush=True)
        results.append((label, dn, tfl, pre_is))

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<40} {'Pre Nodes':>12} {'Pre IS':>14} {'To Flop':>10}")
    print("-"*80)
    for label, dn, tfl, pis in results:
        print(f"{label:<40} {dn:>12,} {pis:>14,} {tfl:>10,}")

    # Memory estimation
    print(f"\n{'='*80}")
    print(f"MEMORY ESTIMATION (c7a.metal-48xl: 376 GB RAM)")
    print(f"{'='*80}")
    print(f"Per hash table slot: ~60 bytes (key=32 + set=24 + occupied=4)")
    print(f"Per occupied entry:  ~48 bytes additional (regrets + strategy_sum arena)")
    print(f"Total per occupied:  ~108 bytes")
    print()
    for table_label, table_slots in [
        ("500M", 500_000_000), ("750M", 750_000_000),
        ("1B", 1_000_000_000), ("1.5B", 1_500_000_000),
        ("2B", 2_000_000_000), ("3B", 3_000_000_000),
    ]:
        meta_gb = table_slots * 60 / 1e9
        # Assume 70% fill
        fill = int(table_slots * 0.7)
        arena_gb = fill * 48 / 1e9
        total_gb = meta_gb + arena_gb
        print(f"  {table_label:>4s} table: {meta_gb:.0f} GB meta + {arena_gb:.0f} GB arena "
              f"= {total_gb:.0f} GB total (70% fill = {fill:,} entries)")

    # Postflop multiplier reminder
    print(f"\n{'='*80}")
    print(f"FULL TREE ESTIMATION")
    print(f"{'='*80}")
    print(f"Postflop per line (2 players, our config): ~14,400 IS")
    print(f"Postflop per line (3 players):             ~88,200 IS")
    print()
    for label, dn, tfl, pis in results:
        # Conservative: assume avg 2.5 players reach flop
        # Weighted: 60% 2p, 25% 3p, 10% 4p, 5% 5p+
        avg_post_is = 0.60 * 14400 + 0.25 * 88200 + 0.10 * 261600 + 0.05 * 627000
        total = pis + tfl * avg_post_is
        print(f"  {label:<40} total IS ≈ {total:>18,.0f}")

    print(f"\n  Pluribus reference: 665,000,000 action sequences")
    print(f"  (If action seq = info set: target ≈ 500M-665M)")
    print(f"  (If action seq = tree node: target ≈ 665M nodes, IS = nodes × buckets)")


if __name__ == '__main__':
    main()
