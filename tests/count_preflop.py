#!/usr/bin/env python3
"""
Count preflop action sequences for 6-player NLHE.

Uses a compact state representation and iterative counting to handle
the combinatorial explosion. Key insight: we don't need to track exact
bet amounts for counting — just how many DISTINCT raise sizes are available
at each decision point.

For exact counting with pot/stack tracking, we model the pot and stacks
to determine how many of the N configured sizes produce distinct bet amounts
(i.e., how many are below all-in).

State = (active_mask, has_acted_mask, num_raises, pot, stacks_tuple, bet_level)
where bet_level is the current max bet (for computing to_call).
"""

import sys
import time
from functools import lru_cache

NUM_PLAYERS = 6
SB_AMOUNT = 1
BB_AMOUNT = 2
INITIAL_STACK = 200  # 100bb

# Acting order: UTG(2), MP(3), CO(4), BTN(5), SB(0), BB(1)
PREFLOP_ORDER = [(i + 2) % NUM_PLAYERS for i in range(NUM_PLAYERS)]


def count_distinct_raises(pot, stack, to_call, bet_sizes):
    """Count how many distinct raise amounts the bet_sizes produce.
    Returns (num_non_allin_raises, has_allin).
    Matches generate_actions() in mccfr_blueprint.c."""
    amounts = set()
    has_allin = False
    for bs in bet_sizes:
        if to_call == 0:
            ba = int(bs * pot)
        else:
            ba = to_call + int(bs * (pot + to_call))
        if ba >= stack:
            ba = stack
        if ba <= to_call:
            continue
        if ba >= stack:
            has_allin = True
        amounts.add(ba)
    # Add manual all-in if not yet included and stack > to_call
    if not has_allin and stack > to_call:
        amounts.add(stack)
        has_allin = True
    return len(amounts)


def count_preflop(bet_sizes, max_raises, verbose=True):
    """
    Count preflop decision nodes and terminal sequences.
    Uses memoized DFS over compact state.
    """
    decision_nodes = 0
    terminals_fold = 0
    terminals_round_done = 0
    nodes_explored = 0

    # Use explicit stack to avoid Python recursion limit
    # State: (active: tuple of bool, has_acted: tuple of bool,
    #         bets: tuple of int, stacks: tuple of int,
    #         pot: int, num_raises: int, next_order_idx: int)

    initial_stacks = [INITIAL_STACK] * NUM_PLAYERS
    initial_bets = [0] * NUM_PLAYERS
    initial_has_acted = [False] * NUM_PLAYERS

    # Post blinds
    initial_bets[0] = SB_AMOUNT
    initial_stacks[0] -= SB_AMOUNT
    initial_bets[1] = BB_AMOUNT
    initial_stacks[1] -= BB_AMOUNT
    initial_pot = SB_AMOUNT + BB_AMOUNT

    # (active, has_acted, bets, stacks, pot, num_raises, order_idx)
    stack = [(
        tuple([True] * NUM_PLAYERS),
        tuple(initial_has_acted),
        tuple(initial_bets),
        tuple(initial_stacks),
        initial_pot,
        0,
        0,  # start at PREFLOP_ORDER[0] = UTG
    )]

    t0 = time.time()

    while stack:
        active, has_acted, bets, stacks, pot, num_raises, order_idx = stack.pop()
        nodes_explored += 1

        if nodes_explored % 2_000_000 == 0 and verbose:
            elapsed = time.time() - t0
            print(f"  {nodes_explored/1e6:.0f}M nodes, "
                  f"{decision_nodes/1e6:.1f}M decisions, "
                  f"{len(stack)/1e6:.1f}M stack, "
                  f"{elapsed:.0f}s", flush=True)

        # Find next active player
        ap = None
        checked = 0
        idx = order_idx
        while checked < NUM_PLAYERS:
            p = PREFLOP_ORDER[idx % NUM_PLAYERS]
            if active[p]:
                ap = p
                order_idx = idx % NUM_PLAYERS
                break
            idx += 1
            checked += 1

        if ap is None:
            continue

        # Check round_done: all active players have acted and bets are equal
        mx_bet = max(bets[i] for i in range(NUM_PLAYERS) if active[i])
        round_done = True
        for i in range(NUM_PLAYERS):
            if not active[i]:
                continue
            if not has_acted[i] or bets[i] != mx_bet:
                round_done = False
                break

        if round_done:
            terminals_round_done += 1
            continue

        # This is a decision node
        decision_nodes += 1

        to_call = mx_bet - bets[ap]
        remaining = stacks[ap]

        # How many distinct raise actions?
        if num_raises < max_raises:
            num_raise_actions = count_distinct_raises(
                pot, remaining, to_call, bet_sizes)
        else:
            num_raise_actions = 0

        next_idx = (order_idx + 1) % NUM_PLAYERS

        # Action 1: Fold (if to_call > 0)
        if to_call > 0:
            new_active = list(active)
            new_active[ap] = False
            if sum(new_active) <= 1:
                terminals_fold += 1
            else:
                stack.append((
                    tuple(new_active), has_acted, bets, stacks,
                    pot, num_raises, next_idx))

        # Action 2: Check/Call
        new_bets = list(bets)
        new_stacks = list(stacks)
        new_has_acted = list(has_acted)
        new_pot = pot
        if to_call > 0:
            new_bets[ap] = mx_bet
            new_stacks[ap] -= to_call
            new_pot += to_call
        new_has_acted[ap] = True
        stack.append((
            active, tuple(new_has_acted), tuple(new_bets),
            tuple(new_stacks), new_pot, num_raises, next_idx))

        # Raise actions: enumerate each distinct raise amount
        if num_raise_actions > 0:
            # We need the actual amounts to compute child states
            amounts = []
            added_allin = False
            for bs in bet_sizes:
                if to_call == 0:
                    ba = int(bs * pot)
                else:
                    ba = to_call + int(bs * (pot + to_call))
                if ba >= remaining:
                    ba = remaining
                if ba <= to_call:
                    continue
                if ba >= remaining:
                    if added_allin:
                        continue
                    added_allin = True
                if ba not in amounts:
                    amounts.append(ba)
            if not added_allin and remaining > to_call:
                amounts.append(remaining)

            for amt in amounts:
                new_bets2 = list(bets)
                new_stacks2 = list(stacks)
                new_has_acted2 = [False] * NUM_PLAYERS
                new_bets2[ap] += amt
                new_stacks2[ap] -= amt
                new_pot2 = pot + amt
                new_has_acted2[ap] = True
                # Keep has_acted for folded players (doesn't matter, they're inactive)
                for p in range(NUM_PLAYERS):
                    if not active[p]:
                        new_has_acted2[p] = has_acted[p]

                stack.append((
                    active, tuple(new_has_acted2), tuple(new_bets2),
                    tuple(new_stacks2), new_pot2, num_raises + 1, next_idx))

    elapsed = time.time() - t0
    return decision_nodes, terminals_fold, terminals_round_done, nodes_explored, elapsed


def count_postflop_per_line(n_active, pot, stacks,
                            first_sizes, subseq_sizes, max_raises):
    """
    Count decision nodes for ONE postflop street with n_active players.
    Players act in order 0..n_active-1 (position doesn't matter for counting).
    Simplified: assume all players have same stack (avg).
    Returns decision_nodes for this street.
    """
    decision_nodes = 0
    avg_stack = sum(stacks[:n_active]) // n_active if n_active > 0 else 0

    # Use iterative DFS
    # State: (has_acted: tuple, bets: tuple, stacks: tuple, pot, num_raises, next_player)
    initial = (
        tuple([False] * n_active),
        tuple([0] * n_active),
        tuple([avg_stack] * n_active),
        pot,
        0,
        0,
    )
    stack = [initial]
    active = [True] * n_active  # all active (folded players already removed)

    while stack:
        has_acted, bets, p_stacks, pot, num_raises, next_p = stack.pop()

        # Find next active player
        ap = None
        for i in range(n_active):
            p = (next_p + i) % n_active
            if True:  # all active in simplified model
                ap = p
                break

        if ap is None:
            continue

        # Check round done
        mx_bet = max(bets)
        round_done = all(has_acted[i] and bets[i] == mx_bet for i in range(n_active))
        if round_done:
            continue

        decision_nodes += 1

        to_call = mx_bet - bets[ap]
        remaining = p_stacks[ap]

        if num_raises > 0:
            sizes = subseq_sizes
        else:
            sizes = first_sizes

        if num_raises < max_raises:
            num_raise_acts = count_distinct_raises(pot, remaining, to_call, sizes)
        else:
            num_raise_acts = 0

        next_player = (ap + 1) % n_active

        # Fold (only if to_call > 0 and more than 2 players... simplified)
        # For counting, fold reduces active count. To keep it simple,
        # we ignore fold in postflop street counting (fold terminates subtree,
        # so it doesn't add decision nodes to later streets)

        # Check/Call
        nb = list(bets); ns = list(p_stacks); nh = list(has_acted)
        if to_call > 0:
            nb[ap] = mx_bet; ns[ap] -= to_call
            new_pot = pot + to_call
        else:
            new_pot = pot
        nh[ap] = True
        stack.append((tuple(nh), tuple(nb), tuple(ns), new_pot, num_raises, next_player))

        # Raises
        if num_raise_acts > 0:
            amounts = []
            added_allin = False
            for bs in sizes:
                if to_call == 0:
                    ba = int(bs * pot)
                else:
                    ba = to_call + int(bs * (pot + to_call))
                if ba >= remaining: ba = remaining
                if ba <= to_call: continue
                if ba >= remaining:
                    if added_allin: continue
                    added_allin = True
                if ba not in amounts: amounts.append(ba)
            if not added_allin and remaining > to_call:
                amounts.append(remaining)

            for amt in amounts:
                nb2 = list(bets); ns2 = list(p_stacks); nh2 = [False]*n_active
                nb2[ap] += amt; ns2[ap] -= amt
                nh2[ap] = True
                stack.append((tuple(nh2), tuple(nb2), tuple(ns2),
                              pot + amt, num_raises + 1, next_player))

    return decision_nodes


def main():
    print("="*70)
    print("PREFLOP ACTION SEQUENCE ENUMERATION")
    print("6-player NLHE, 100bb deep")
    print("="*70)

    configs = [
        ("CURRENT: 8 flat sizes, max 4 raises",
         (0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0), 4),
        ("8 flat sizes, max 3 raises",
         (0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0), 3),
        ("5 flat sizes, max 4 raises",
         (0.5, 0.7, 1.0, 2.5, 8.0), 4),
        ("5 flat sizes, max 3 raises",
         (0.5, 0.7, 1.0, 2.5, 8.0), 3),
        ("4 flat sizes, max 4 raises",
         (0.5, 1.0, 2.5, 8.0), 4),
        ("4 flat sizes, max 3 raises",
         (0.5, 1.0, 2.5, 8.0), 3),
        ("3 flat sizes, max 4 raises",
         (0.7, 1.0, 4.0), 4),
        ("3 flat sizes, max 3 raises",
         (0.7, 1.0, 4.0), 3),
        ("2 flat sizes, max 4 raises (minimal)",
         (0.7, 2.5), 4),
    ]

    results = []

    for name, sizes, max_r in configs:
        print(f"\n--- {name} ---", flush=True)
        print(f"    Sizes: {sizes}, max raises: {max_r}", flush=True)

        dn, tf, trd, ne, elapsed = count_preflop(sizes, max_r, verbose=True)
        preflop_is = dn * 169

        print(f"    Decision nodes:      {dn:>15,}")
        print(f"    Terminals (fold):    {tf:>15,}")
        print(f"    Terminals (to flop): {trd:>15,}")
        print(f"    Nodes explored:      {ne:>15,}")
        print(f"    Time: {elapsed:.1f}s")
        print(f"    Preflop info sets:   {preflop_is:>15,} (× 169 buckets)")
        print(f"    Lines reaching flop: {trd:>15,}")
        results.append((name, dn, tf, trd, preflop_is))

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: Preflop Decision Nodes")
    print(f"{'='*70}")
    print(f"{'Config':<45} {'Nodes':>12} {'Info Sets':>14} {'To Flop':>10}")
    print("-" * 85)
    for name, dn, tf, trd, pis in results:
        print(f"{name:<45} {dn:>12,} {pis:>14,} {trd:>10,}")

    # Postflop estimation
    print(f"\n{'='*70}")
    print("POSTFLOP MULTIPLIER ESTIMATION")
    print(f"{'='*70}")
    print("For each preflop line reaching the flop, there's a postflop subtree.")
    print("Postflop config: first=[0.5,1.0]+allin, subsequent=[1.0]+allin, max 3 raises")
    print()

    for n_active in [2, 3, 4, 5, 6]:
        # Typical pot and stack for n_active players reaching flop
        # Rough: if 2 reach flop after open+call, pot ~ 8bb = 16 chips
        pot_estimate = n_active * 4  # rough
        avg_stack = INITIAL_STACK - pot_estimate // n_active
        stacks = [avg_stack] * n_active

        nodes = count_postflop_per_line(
            n_active, pot_estimate, stacks,
            (0.5, 1.0), (1.0,), 3)
        print(f"  {n_active} players on flop: ~{nodes:,} decision nodes per street")
        print(f"    × 3 streets × 200 buckets = ~{nodes * 3 * 200:,} info sets per line")

    # Full tree estimate
    print(f"\n{'='*70}")
    print("FULL TREE ESTIMATE")
    print(f"{'='*70}")
    if results:
        # Use first result (current config)
        _, dn, _, trd, pis = results[0]
        # Rough: assume average 2.5 players on flop, ~100 nodes per street per line
        # This is a very rough estimate — the actual postflop tree depends heavily
        # on how many players reach each street
        avg_postflop_nodes = 100  # per line per street (conservative for 2-3 players)
        postflop_nodes = trd * avg_postflop_nodes * 3  # 3 streets
        postflop_is = postflop_nodes * 200
        total = pis + postflop_is
        print(f"  Current config preflop IS:  {pis:>15,}")
        print(f"  Lines to flop:             {trd:>15,}")
        print(f"  Est. postflop nodes/line:  {avg_postflop_nodes:>15,} (per street)")
        print(f"  Est. postflop IS:          {postflop_is:>15,}")
        print(f"  Est. TOTAL info sets:      {total:>15,}")
        print(f"\n  Pluribus target:           {'~665,000,000':>15}")


if __name__ == '__main__':
    main()
