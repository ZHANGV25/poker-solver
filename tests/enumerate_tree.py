#!/usr/bin/env python3
"""
Analytically enumerate all decision nodes in the 6-player NLHE game tree.

Matches the logic in src/mccfr_blueprint.c:
- Preflop: UTG(2), MP(3), CO(4), BTN(5), SB(0), BB(1)
- SB=1, BB=2, starting stack=200 (100bb)
- Preflop sizes: pot-fractions, max 4 raises
- Postflop first raise: [0.5, 1.0] + all-in, max 3 raises
- Postflop subsequent: [1.0] + all-in, max 3 raises
- Postflop order: SB(0), BB(1), UTG(2), ..., BTN(5)

Counts decision nodes per street, then multiplies by buckets:
  preflop × 169,  flop/turn/river × 200

Also tests alternative configs (tiered preflop sizing, fewer sizes).
"""

import sys
from dataclasses import dataclass, field
from typing import List, Tuple
from functools import lru_cache
import time


# ── Game tree node counter ──────────────────────────────────────────

@dataclass
class Config:
    num_players: int = 6
    small_blind: int = 1
    big_blind: int = 2
    initial_stack: int = 200  # 100bb
    preflop_bet_sizes: Tuple[float, ...] = (0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0)
    preflop_max_raises: int = 4
    postflop_first_bet_sizes: Tuple[float, ...] = (0.5, 1.0)
    postflop_subsequent_bet_sizes: Tuple[float, ...] = (1.0,)
    postflop_max_raises: int = 3
    preflop_buckets: int = 169
    postflop_buckets: int = 200


def generate_actions(pot, stack, to_call, num_raises, max_raises, bet_sizes):
    """Return list of action types: 'fold', 'check', 'call', or raise amount (int)."""
    actions = []
    if to_call > 0:
        actions.append('fold')
    actions.append('call' if to_call > 0 else 'check')

    if num_raises < max_raises:
        added_allin = False
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
                if added_allin:
                    continue
                added_allin = True
            actions.append(('raise', ba))

        # Add all-in if not yet added
        if not added_allin and stack > to_call:
            actions.append(('raise', stack))

    return actions


def count_tree(cfg: Config, streets_to_count=(0, 1, 2, 3), verbose=True):
    """
    Count decision nodes per street in the full game tree.
    Returns dict: {street: num_decision_nodes}
    """
    NP = cfg.num_players
    counts = {0: 0, 1: 0, 2: 0, 3: 0}  # per street
    terminal_by_type = {'fold_out': 0, 'showdown': 0, 'allin': 0}

    # Track preflop non-terminal sequences (how many distinct preflop
    # endings lead to a flop)
    preflop_to_flop = [0]  # [count] — mutable for closure
    # Track active player counts reaching flop
    flop_player_counts = {}

    call_count = [0]
    max_depth = [0]

    def traverse_street(street, active, bets, stacks, pot, num_raises,
                        has_acted, acting_order, acting_idx, depth):
        """Recursively count decision nodes on a single street."""
        if depth > max_depth[0]:
            max_depth[0] = depth

        call_count[0] += 1
        if call_count[0] % 5_000_000 == 0 and verbose:
            print(f"  ... {call_count[0]/1e6:.0f}M nodes explored, "
                  f"street {street}, depth {depth}")

        # Find next active player
        ap = None
        checked = 0
        idx = acting_idx
        while checked < len(acting_order):
            p = acting_order[idx % len(acting_order)]
            if active[p]:
                ap = p
                acting_idx = idx % len(acting_order)
                break
            idx += 1
            checked += 1

        if ap is None:
            return

        # Check if round is done
        if all((not active[i]) or (has_acted[i] and bets[i] == max(
                bets[j] for j in range(NP) if active[j]))
               for i in range(NP)):
            # Round done — transition to next street or showdown
            n_active = sum(active)
            if n_active <= 1:
                terminal_by_type['fold_out'] += 1
                return

            if street == 0:
                # Preflop done → flop
                preflop_to_flop[0] += 1
                flop_player_counts[n_active] = flop_player_counts.get(n_active, 0) + 1
                if 1 not in streets_to_count and 2 not in streets_to_count and 3 not in streets_to_count:
                    return  # Skip postflop enumeration
                # Count postflop streets for this preflop line
                new_pot = pot
                new_stacks = list(stacks)
                postflop_order = list(range(NP))
                count_postflop(1, active[:], [0]*NP, new_stacks, new_pot, 0,
                               [False]*NP, postflop_order, 0, depth+1, counts,
                               terminal_by_type, cfg, call_count, max_depth, verbose,
                               streets_to_count)
            elif street < 3:
                # Flop→Turn or Turn→River
                new_pot = pot
                new_stacks = list(stacks)
                count_postflop(street+1, active[:], [0]*NP, new_stacks, new_pot, 0,
                               [False]*NP, acting_order, 0, depth+1, counts,
                               terminal_by_type, cfg, call_count, max_depth, verbose,
                               streets_to_count)
            else:
                # River done → showdown
                terminal_by_type['showdown'] += 1
            return

        # This is a decision node — count it
        if street in streets_to_count:
            counts[street] += 1

        # Generate actions
        mx = max(bets[i] for i in range(NP) if active[i])
        to_call = mx - bets[ap]
        if to_call < 0:
            to_call = 0

        if street == 0:
            bet_sizes = cfg.preflop_bet_sizes
            max_raises = cfg.preflop_max_raises
        else:
            if num_raises > 0:
                bet_sizes = cfg.postflop_subsequent_bet_sizes
            else:
                bet_sizes = cfg.postflop_first_bet_sizes
            max_raises = cfg.postflop_max_raises

        actions = generate_actions(pot, stacks[ap], to_call, num_raises,
                                   max_raises, bet_sizes)

        for act in actions:
            new_active = active[:]
            new_bets = bets[:]
            new_stacks = list(stacks)
            new_has_acted = has_acted[:]
            new_pot = pot
            new_num_raises = num_raises

            if act == 'fold':
                new_active[ap] = False
                # Check if only 1 player left
                if sum(new_active) <= 1:
                    terminal_by_type['fold_out'] += 1
                    continue
            elif act in ('check', 'call'):
                if act == 'call':
                    new_bets[ap] = mx
                    new_stacks[ap] -= to_call
                    new_pot += to_call
                new_has_acted[ap] = True
            else:
                # Raise
                _, amount = act
                if amount > new_stacks[ap]:
                    amount = new_stacks[ap]
                new_bets[ap] += amount
                new_stacks[ap] -= amount
                new_pot += amount
                new_has_acted[ap] = True
                # Reset has_acted for all other active players
                for p in range(NP):
                    if p != ap and new_active[p]:
                        new_has_acted[p] = False
                new_num_raises = num_raises + 1

            # Continue to next player
            next_idx = (acting_idx + 1) % len(acting_order)
            traverse_street(street, new_active, new_bets, new_stacks,
                            new_pot, new_num_raises, new_has_acted,
                            acting_order, next_idx, depth + 1)

    # ── Preflop ─────────────────────────────────────────────────
    active = [True] * NP
    bets = [0] * NP
    stacks = [cfg.initial_stack] * NP
    has_acted = [False] * NP

    # Post blinds
    bets[0] = cfg.small_blind
    stacks[0] -= cfg.small_blind
    bets[1] = cfg.big_blind
    stacks[1] -= cfg.big_blind
    pot = cfg.small_blind + cfg.big_blind

    # Preflop order: UTG(2), MP(3), CO(4), BTN(5), SB(0), BB(1)
    preflop_order = [(i + 2) % NP for i in range(NP)]

    if verbose:
        print(f"Enumerating tree: {len(cfg.preflop_bet_sizes)} preflop sizes, "
              f"max {cfg.preflop_max_raises} raises")
        print(f"  Postflop: {len(cfg.postflop_first_bet_sizes)} first-raise sizes, "
              f"{len(cfg.postflop_subsequent_bet_sizes)} subsequent, "
              f"max {cfg.postflop_max_raises} raises")

    traverse_street(0, active, bets, stacks, pot, 0, has_acted,
                    preflop_order, 0, 0)

    return counts, terminal_by_type, preflop_to_flop[0], flop_player_counts, max_depth[0]


def count_postflop(street, active, bets, stacks, pot, num_raises,
                   has_acted, acting_order, acting_idx, depth,
                   counts, terminal_by_type, cfg, call_count, max_depth,
                   verbose, streets_to_count):
    """Count decision nodes in a postflop street."""
    NP = cfg.num_players

    if depth > max_depth[0]:
        max_depth[0] = depth

    call_count[0] += 1
    if call_count[0] % 5_000_000 == 0 and verbose:
        print(f"  ... {call_count[0]/1e6:.0f}M nodes explored, "
              f"street {street}, depth {depth}")

    # Find next active player
    ap = None
    checked = 0
    idx = acting_idx
    while checked < len(acting_order):
        p = acting_order[idx % len(acting_order)]
        if p < NP and active[p]:
            ap = p
            acting_idx = idx % len(acting_order)
            break
        idx += 1
        checked += 1

    if ap is None:
        return

    # Check if round is done
    mx_bet = max(bets[j] for j in range(NP) if active[j])
    round_complete = True
    for i in range(NP):
        if not active[i]:
            continue
        if not has_acted[i]:
            round_complete = False
            break
        if bets[i] != mx_bet:
            round_complete = False
            break

    if round_complete:
        n_active = sum(active)
        if n_active <= 1:
            terminal_by_type['fold_out'] += 1
            return

        if street < 3:
            # Next street
            if street + 1 not in streets_to_count and all(
                    s not in streets_to_count for s in range(street+1, 4)):
                return  # Skip remaining streets
            count_postflop(street+1, active[:], [0]*NP, list(stacks), pot, 0,
                           [False]*NP, acting_order, 0, depth+1,
                           counts, terminal_by_type, cfg, call_count, max_depth,
                           verbose, streets_to_count)
        else:
            terminal_by_type['showdown'] += 1
        return

    # Decision node
    if street in streets_to_count:
        counts[street] += 1

    # Generate actions
    to_call = mx_bet - bets[ap]
    if to_call < 0:
        to_call = 0

    if num_raises > 0:
        bet_sizes = cfg.postflop_subsequent_bet_sizes
    else:
        bet_sizes = cfg.postflop_first_bet_sizes
    max_raises = cfg.postflop_max_raises

    actions = generate_actions(pot, stacks[ap], to_call, num_raises,
                               max_raises, bet_sizes)

    for act in actions:
        new_active = active[:]
        new_bets = bets[:]
        new_stacks = list(stacks)
        new_has_acted = has_acted[:]
        new_pot = pot
        new_num_raises = num_raises

        if act == 'fold':
            new_active[ap] = False
            if sum(new_active) <= 1:
                terminal_by_type['fold_out'] += 1
                continue
        elif act in ('check', 'call'):
            if act == 'call':
                new_bets[ap] = mx_bet
                new_stacks[ap] -= to_call
                new_pot += to_call
            new_has_acted[ap] = True
        else:
            _, amount = act
            if amount > new_stacks[ap]:
                amount = new_stacks[ap]
            new_bets[ap] += amount
            new_stacks[ap] -= amount
            new_pot += amount
            new_has_acted[ap] = True
            for p in range(NP):
                if p != ap and new_active[p]:
                    new_has_acted[p] = False
            new_num_raises = num_raises + 1

        next_idx = (acting_idx + 1) % len(acting_order)
        count_postflop(street, new_active, new_bets, new_stacks,
                       new_pot, new_num_raises, new_has_acted,
                       acting_order, next_idx, depth + 1,
                       counts, terminal_by_type, cfg, call_count, max_depth,
                       verbose, streets_to_count)


def run_config(name, cfg, preflop_only=False):
    """Run enumeration for a config and print results."""
    print(f"\n{'='*70}")
    print(f"Config: {name}")
    print(f"{'='*70}")

    streets = (0,) if preflop_only else (0, 1, 2, 3)
    t0 = time.time()
    counts, terminals, preflop_to_flop, flop_players, max_depth = \
        count_tree(cfg, streets_to_count=streets)
    elapsed = time.time() - t0

    print(f"\nResults ({elapsed:.1f}s, max depth {max_depth}):")
    print(f"  Preflop decision nodes:  {counts[0]:>15,}")
    if not preflop_only:
        print(f"  Flop decision nodes:     {counts[1]:>15,}")
        print(f"  Turn decision nodes:     {counts[2]:>15,}")
        print(f"  River decision nodes:    {counts[3]:>15,}")
    total_nodes = sum(counts.values())
    print(f"  Total decision nodes:    {total_nodes:>15,}")

    print(f"\n  Preflop lines to flop:   {preflop_to_flop:>15,}")
    if flop_players:
        print(f"  Flop player distribution: {dict(sorted(flop_players.items()))}")

    print(f"\n  Terminals (fold-out):    {terminals['fold_out']:>15,}")
    print(f"  Terminals (showdown):    {terminals['showdown']:>15,}")

    # Info set estimate
    pre_is = counts[0] * cfg.preflop_buckets
    post_is = sum(counts[s] * cfg.postflop_buckets for s in (1, 2, 3))
    total_is = pre_is + post_is
    print(f"\n  Estimated info sets:")
    print(f"    Preflop:  {counts[0]:>12,} × {cfg.preflop_buckets} = {pre_is:>15,}")
    if not preflop_only:
        for s, name_s in [(1, 'Flop'), (2, 'Turn'), (3, 'River')]:
            is_s = counts[s] * cfg.postflop_buckets
            print(f"    {name_s:8s}: {counts[s]:>12,} × {cfg.postflop_buckets} = {is_s:>15,}")
    print(f"    TOTAL:    {total_is:>15,}")
    print(f"    (Pluribus: ~665M action sequences)")

    return counts, total_is


# ── Configs to test ─────────────────────────────────────────────────

def main():
    sys.setrecursionlimit(50000)

    # Current config: 8 preflop sizes, flat
    current = Config()

    # First: preflop-only enumeration (fast) to understand the branching
    print("\n" + "#"*70)
    print("# PHASE 1: Preflop-only enumeration (fast)")
    print("#"*70)

    run_config("Current (8 flat preflop sizes, max 4 raises)", current, preflop_only=True)

    # Tiered preflop: 8 open, 4 3bet, 2 4bet, all-in only 5bet
    tiered = Config(
        preflop_bet_sizes=(0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0),  # open
        preflop_max_raises=4,
    )
    # We can't easily model tiered in the current framework since
    # generate_actions uses the same sizes for all raise levels.
    # Instead, model the effect by reducing to fewer sizes.

    # Reduced configs
    configs_preflop = [
        ("5 flat preflop sizes", Config(
            preflop_bet_sizes=(0.5, 0.7, 1.0, 2.5, 8.0),
            preflop_max_raises=4)),
        ("4 flat preflop sizes", Config(
            preflop_bet_sizes=(0.5, 1.0, 2.5, 8.0),
            preflop_max_raises=4)),
        ("3 flat preflop sizes", Config(
            preflop_bet_sizes=(0.7, 1.0, 4.0),
            preflop_max_raises=4)),
        ("8 sizes, max 3 raises", Config(
            preflop_bet_sizes=(0.4, 0.5, 0.7, 1.0, 1.5, 2.5, 4.0, 8.0),
            preflop_max_raises=3)),
        ("5 sizes, max 3 raises", Config(
            preflop_bet_sizes=(0.5, 0.7, 1.0, 2.5, 8.0),
            preflop_max_raises=3)),
    ]

    for name, cfg in configs_preflop:
        run_config(name, cfg, preflop_only=True)

    # Phase 2: Full tree for a small config to understand postflop multiplier
    print("\n" + "#"*70)
    print("# PHASE 2: Full tree enumeration (postflop included)")
    print("# Using smaller configs due to combinatorial explosion")
    print("#"*70)

    # Smallest reasonable config for full enumeration
    small = Config(
        preflop_bet_sizes=(0.7, 1.0, 2.5),
        preflop_max_raises=3,
        postflop_first_bet_sizes=(0.5, 1.0),
        postflop_subsequent_bet_sizes=(1.0,),
        postflop_max_raises=3,
    )
    run_config("Small full tree (3 pre sizes, max 3 raises)", small)

    # If that's tractable, try medium
    medium = Config(
        preflop_bet_sizes=(0.5, 0.7, 1.0, 2.5),
        preflop_max_raises=3,
        postflop_first_bet_sizes=(0.5, 1.0),
        postflop_subsequent_bet_sizes=(1.0,),
        postflop_max_raises=3,
    )
    run_config("Medium full tree (4 pre sizes, max 3 raises)", medium)

    # Pluribus-like postflop: 3 first raise sizes, 2 subsequent, max 3 raises
    # with smaller preflop for tractability
    pluribus_post = Config(
        preflop_bet_sizes=(0.5, 0.7, 1.0, 2.5),
        preflop_max_raises=3,
        postflop_first_bet_sizes=(0.5, 1.0),  # + all-in = 3 options
        postflop_subsequent_bet_sizes=(1.0,),  # + all-in = 2 options
        postflop_max_raises=3,
    )
    # This is identical to medium — our postflop already matches Pluribus

    print("\n" + "#"*70)
    print("# PHASE 3: Postflop multiplier estimation")
    print("# Ratio of total nodes to preflop nodes across configs")
    print("#"*70)

    # Run a few with just 2 players reaching flop to estimate multiplier
    small_2p = Config(
        num_players=2,
        preflop_bet_sizes=(0.7, 1.0, 2.5),
        preflop_max_raises=3,
        postflop_first_bet_sizes=(0.5, 1.0),
        postflop_subsequent_bet_sizes=(1.0,),
        postflop_max_raises=3,
    )
    run_config("2-player heads-up (3 pre sizes, max 3 raises)", small_2p)


if __name__ == '__main__':
    main()
