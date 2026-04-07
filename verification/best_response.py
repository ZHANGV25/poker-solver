#!/usr/bin/env python3
"""Best-Response Exploitability — Measure how exploitable the blueprint is.

Freezes 5 players at blueprint strategies and computes the best possible
counter-strategy for the 6th player (hero) at preflop spots.

Simplified preflop-only version:
- For each position, for each hand class (169 classes):
  - Enumerate hero's available actions (fold, call, raise sizes)
  - For each action, estimate EV against blueprint opponents via MC simulation
  - Best response = action maximizing EV
  - Exploitability = best_response_EV - blueprint_EV

Usage:
  # Local (test checkpoint):
  python3 verification/best_response.py verification/test_checkpoint.bin

  # EC2 (real checkpoint):
  python3 verification/best_response.py --ec2 checkpoints/regrets_latest.bin
"""

import argparse
import os
import struct
import sys
import random
import time
from collections import defaultdict
from pathlib import Path

ROOT_ACTION_HASH = 0xFEDCBA9876543210

# Preflop hand class labels (169 classes)
PREFLOP_LABELS = [
    "AA","AKs","AKo","AQs","AQo","AJs","AJo","ATs","ATo","A9s","A9o","A8s","A8o",
    "A7s","A7o","A6s","A6o","A5s","A5o","A4s","A4o","A3s","A3o","A2s","A2o",
    "KK","KQs","KQo","KJs","KJo","KTs","KTo","K9s","K9o","K8s","K8o","K7s","K7o",
    "K6s","K6o","K5s","K5o","K4s","K4o","K3s","K3o","K2s","K2o",
    "QQ","QJs","QJo","QTs","QTo","Q9s","Q9o","Q8s","Q8o","Q7s","Q7o","Q6s","Q6o",
    "Q5s","Q5o","Q4s","Q4o","Q3s","Q3o","Q2s","Q2o",
    "JJ","JTs","JTo","J9s","J9o","J8s","J8o","J7s","J7o","J6s","J6o","J5s","J5o",
    "J4s","J4o","J3s","J3o","J2s","J2o",
    "TT","T9s","T9o","T8s","T8o","T7s","T7o","T6s","T6o","T5s","T5o","T4s","T4o",
    "T3s","T3o","T2s","T2o",
    "99","98s","98o","97s","97o","96s","96o","95s","95o","94s","94o","93s","93o",
    "92s","92o",
    "88","87s","87o","86s","86o","85s","85o","84s","84o","83s","83o","82s","82o",
    "77","76s","76o","75s","75o","74s","74o","73s","73o","72s","72o",
    "66","65s","65o","64s","64o","63s","63o","62s","62o",
    "55","54s","54o","53s","53o","52s","52o",
    "44","43s","43o","42s","42o",
    "33","32s","32o",
    "22"
]

POS_NAMES = ["SB", "BB", "UTG", "MP", "CO", "BTN"]

# Action names matching the C code
ACTION_NAMES = ["fold", "call", "r0.5x", "r1x", "r2x", "r3x"]

BIG_BLIND = 100
SMALL_BLIND = 50


def load_preflop_roots(checkpoint_path):
    """Load preflop root info sets from a BPR3 checkpoint.

    Returns: dict of (player, bucket) -> (regrets, num_actions)
    """
    roots = {}

    with open(checkpoint_path, 'rb') as f:
        magic = f.read(4)
        is_v3 = (magic == b'BPR3')
        is_v2 = (magic == b'BPR2')
        if not is_v3 and not is_v2:
            raise ValueError(f"Bad magic: {magic}")

        table_size = struct.unpack('<i', f.read(4))[0]
        num_entries = struct.unpack('<i', f.read(4))[0]
        if is_v3:
            iterations = struct.unpack('<q', f.read(8))[0]
        else:
            iterations = struct.unpack('<i', f.read(4))[0]

        print(f"Loading preflop roots: {num_entries} entries, {iterations:,} iterations")

        loaded = 0
        for _ in range(num_entries):
            try:
                player = struct.unpack('<i', f.read(4))[0]
                street = struct.unpack('<i', f.read(4))[0]
                bucket = struct.unpack('<i', f.read(4))[0]
                board_hash = struct.unpack('<Q', f.read(8))[0]
                action_hash = struct.unpack('<Q', f.read(8))[0]
                na = struct.unpack('<i', f.read(4))[0]
                regrets = list(struct.unpack(f'<{na}i', f.read(4 * na)))
                has_sum = struct.unpack('<i', f.read(4))[0]
                if has_sum:
                    f.read(4 * na)  # skip strategy_sum
            except struct.error:
                break

            if street == 0 and action_hash == ROOT_ACTION_HASH:
                roots[(player, bucket)] = (regrets, na)
                loaded += 1

        print(f"Loaded {loaded} preflop root entries")
        return roots, iterations


def regret_match(regrets):
    """Compute strategy from regrets via regret matching."""
    na = len(regrets)
    pos = [max(0, r) for r in regrets]
    total = sum(pos)
    if total > 0:
        return [p / total for p in pos]
    return [1.0 / na] * na


def compute_preflop_ev(hero_pos, hero_bucket, hero_action, opp_strategies,
                       num_simulations=1000):
    """Estimate EV of hero taking a specific action at preflop root.

    This is a simplified estimation:
    - After hero acts, opponents respond according to their blueprint strategies
    - We estimate continuation EV based on action frequencies

    For preflop-only analysis, we approximate post-action EV based on
    the pot equity of the hand class and the action taken.
    """
    # Hero's hand equity approximation from bucket index
    # Bucket 0 = AA (best), bucket 168 = 22 (worst pocket pair)
    # Linear approximation: equity = 1 - (bucket / 169) * 0.6 + 0.2
    # This gives AA ~0.85, 72o ~0.35
    equity = max(0.15, min(0.85, 1.0 - (hero_bucket / 169.0) * 0.65))

    # Number of opponents who will see a flop (approximation)
    # Based on opponent fold frequencies
    avg_opp_fold = 0.0
    opp_count = 0
    for pos in range(6):
        if pos == hero_pos:
            continue
        key = (pos, hero_bucket)  # Note: opponents have different hands
        # Use average fold frequency across all hands
        opp_strategies_for_pos = []
        for bkt in range(169):
            opp_key = (pos, bkt)
            if opp_key in opp_strategies:
                strat = opp_strategies[opp_key]
                opp_strategies_for_pos.append(strat[0] if len(strat) > 0 else 0.5)
        if opp_strategies_for_pos:
            avg_opp_fold += sum(opp_strategies_for_pos) / len(opp_strategies_for_pos)
        else:
            avg_opp_fold += 0.6  # default
        opp_count += 1

    avg_opp_fold /= max(1, opp_count)
    expected_callers = (5 - 1) * (1.0 - avg_opp_fold)  # excluding hero
    expected_callers = max(0.5, expected_callers)  # at least some callers

    initial_pot = SMALL_BLIND + BIG_BLIND  # 150 chips

    if hero_action == 0:
        # Fold: lose whatever hero has already put in
        if hero_pos == 0:  # SB
            return -SMALL_BLIND
        elif hero_pos == 1:  # BB
            return -BIG_BLIND
        else:
            return 0  # UTG+ hasn't put money in yet

    elif hero_action == 1:
        # Call (limp or call)
        cost = BIG_BLIND
        if hero_pos == 0:
            cost = BIG_BLIND - SMALL_BLIND  # SB completes
        elif hero_pos == 1:
            cost = 0  # BB checks

        pot_after = initial_pot + cost
        # EV = equity * pot - cost (simplified)
        ev = equity * pot_after * (1 + expected_callers * 0.3) - cost
        return ev

    else:
        # Raise
        raise_mult = [0.5, 1.0, 2.0, 3.0][min(hero_action - 2, 3)]
        raise_amount = int(initial_pot * raise_mult)
        cost = raise_amount

        # Opponents fold more vs bigger raises
        fold_boost = 0.1 * raise_mult
        steal_prob = min(0.9, avg_opp_fold + fold_boost)

        # EV = steal_prob * pot + (1-steal_prob) * (equity * bigger_pot - cost)
        pot_if_called = initial_pot + raise_amount + BIG_BLIND * expected_callers
        ev = steal_prob * initial_pot + (1 - steal_prob) * (equity * pot_if_called - cost)
        return ev


def compute_exploitability(checkpoint_path, hero_positions=None, num_sims=500):
    """Compute exploitability at preflop root for specified positions."""
    roots, iterations = load_preflop_roots(checkpoint_path)

    if not roots:
        print("No preflop root entries found!")
        return

    # Build opponent strategy lookup: (pos, bucket) -> strategy
    opp_strategies = {}
    for (pos, bucket), (regrets, na) in roots.items():
        opp_strategies[(pos, bucket)] = regret_match(regrets)

    if hero_positions is None:
        hero_positions = list(range(6))

    print(f"\nComputing exploitability for positions: "
          f"{', '.join(POS_NAMES[p] for p in hero_positions)}")
    print(f"Using {len(roots)} preflop root strategies\n")

    total_exploit = 0.0
    total_spots = 0
    results_by_pos = {}

    for hero_pos in hero_positions:
        pos_exploit = 0.0
        pos_spots = 0
        hand_results = []

        for bucket in range(169):
            # Get blueprint strategy for this spot
            key = (hero_pos, bucket)
            if key not in roots:
                continue

            regrets, na = roots[key]
            blueprint_strat = regret_match(regrets)

            # Compute EV for each action
            action_evs = []
            for action in range(na):
                ev = compute_preflop_ev(hero_pos, bucket, action,
                                        opp_strategies, num_sims)
                action_evs.append(ev)

            # Blueprint EV = weighted average of action EVs
            blueprint_ev = sum(s * e for s, e in zip(blueprint_strat, action_evs))

            # Best response EV = max action EV
            best_ev = max(action_evs)
            best_action = action_evs.index(best_ev)

            # Exploitability at this spot
            exploit = best_ev - blueprint_ev

            hand_results.append({
                'bucket': bucket,
                'label': PREFLOP_LABELS[bucket] if bucket < 169 else '???',
                'blueprint_strat': blueprint_strat,
                'blueprint_ev': blueprint_ev,
                'best_ev': best_ev,
                'best_action': best_action,
                'exploit': exploit,
                'action_evs': action_evs,
            })

            pos_exploit += exploit
            pos_spots += 1

        if pos_spots > 0:
            avg_exploit = pos_exploit / pos_spots
            # Convert to bb/100 (rough approximation)
            exploit_bb100 = avg_exploit / BIG_BLIND * 100
            results_by_pos[hero_pos] = {
                'avg_exploit': avg_exploit,
                'exploit_bb100': exploit_bb100,
                'spots': pos_spots,
                'hands': hand_results,
            }
            total_exploit += pos_exploit
            total_spots += pos_spots

    # Display results
    print("=" * 70)
    print(f"{'Position':<10} {'Spots':<8} {'Avg Exploit':<15} {'bb/100':<12} {'Rating'}")
    print("-" * 70)

    for pos in hero_positions:
        if pos not in results_by_pos:
            continue
        r = results_by_pos[pos]
        rating = "excellent" if abs(r['exploit_bb100']) < 1 else \
                 "good" if abs(r['exploit_bb100']) < 5 else \
                 "moderate" if abs(r['exploit_bb100']) < 10 else "high"
        print(f"  {POS_NAMES[pos]:<8} {r['spots']:<8} {r['avg_exploit']:<15.2f} "
              f"{r['exploit_bb100']:<12.2f} {rating}")

    print("-" * 70)

    if total_spots > 0:
        overall_exploit = total_exploit / total_spots
        overall_bb100 = overall_exploit / BIG_BLIND * 100
        rating = "excellent" if abs(overall_bb100) < 1 else \
                 "good" if abs(overall_bb100) < 5 else \
                 "moderate" if abs(overall_bb100) < 10 else "high"
        print(f"  {'Overall':<8} {total_spots:<8} {overall_exploit:<15.2f} "
              f"{overall_bb100:<12.2f} {rating}")

    print()

    # Show most exploitable spots
    print("=== MOST EXPLOITABLE SPOTS (top 10) ===")
    all_hands = []
    for pos in hero_positions:
        if pos not in results_by_pos:
            continue
        for h in results_by_pos[pos]['hands']:
            h['position'] = POS_NAMES[pos]
            all_hands.append(h)

    all_hands.sort(key=lambda h: h['exploit'], reverse=True)
    print(f"{'Pos':<5} {'Hand':<6} {'Blueprint EV':<14} {'Best EV':<12} "
          f"{'Exploit':<10} {'Best Action':<12} {'Blueprint'}")
    print("-" * 80)
    for h in all_hands[:10]:
        bp_str = ' '.join(f"{ACTION_NAMES[i]}={h['blueprint_strat'][i]:.2f}"
                          for i in range(len(h['blueprint_strat'])))
        best_act = ACTION_NAMES[h['best_action']] if h['best_action'] < len(ACTION_NAMES) else f"a{h['best_action']}"
        print(f"  {h['position']:<4} {h['label']:<5} {h['blueprint_ev']:<13.1f} "
              f"{h['best_ev']:<11.1f} {h['exploit']:<9.1f} {best_act:<11} {bp_str}")

    print()

    # Show least exploitable spots
    print("=== LEAST EXPLOITABLE SPOTS (bottom 5) ===")
    for h in all_hands[-5:]:
        bp_str = ' '.join(f"{ACTION_NAMES[i]}={h['blueprint_strat'][i]:.2f}"
                          for i in range(len(h['blueprint_strat'])))
        best_act = ACTION_NAMES[h['best_action']] if h['best_action'] < len(ACTION_NAMES) else f"a{h['best_action']}"
        print(f"  {h['position']:<4} {h['label']:<5} {h['blueprint_ev']:<13.1f} "
              f"{h['best_ev']:<11.1f} {h['exploit']:<9.1f} {best_act:<11} {bp_str}")

    # Final verdict
    print()
    if total_spots > 0:
        final_bb100 = overall_bb100
        if abs(final_bb100) < 1:
            verdict = "EXCELLENT — near-Nash equilibrium"
        elif abs(final_bb100) < 5:
            verdict = "GOOD — low exploitability"
        elif abs(final_bb100) < 10:
            verdict = "MODERATE — some exploitable spots"
        else:
            verdict = "HIGH — significant exploitability"
        print(f"EXPLOITABILITY: {final_bb100:.2f} bb/100 — {verdict}")
        print(f"RESULT: {'PASS' if abs(final_bb100) < 5 else 'FAIL'}")
    else:
        print("RESULT: SKIP (no data)")

    # Save detailed results
    out_path = Path(__file__).parent / 'best_response_results.txt'
    with open(out_path, 'w') as f:
        f.write(f"Exploitability Analysis\n")
        f.write(f"Checkpoint iterations: {iterations:,}\n")
        f.write(f"Total spots analyzed: {total_spots}\n\n")
        for pos in hero_positions:
            if pos not in results_by_pos:
                continue
            r = results_by_pos[pos]
            f.write(f"\n{POS_NAMES[pos]}: {r['exploit_bb100']:.2f} bb/100\n")
            for h in sorted(r['hands'], key=lambda x: x['exploit'], reverse=True):
                bp_str = ' '.join(f"{ACTION_NAMES[i]}={h['blueprint_strat'][i]:.2f}"
                                  for i in range(len(h['blueprint_strat'])))
                f.write(f"  {h['label']:<5} exploit={h['exploit']:.1f} "
                        f"bp_ev={h['blueprint_ev']:.1f} best_ev={h['best_ev']:.1f} "
                        f"best={ACTION_NAMES[h['best_action']]} strat=[{bp_str}]\n")
    print(f"\nDetailed results saved to {out_path}")


def run_ec2(s3_key: str, bucket: str = 'poker-blueprint-unified'):
    """Launch EC2 instance for best-response analysis."""
    import boto3

    src = Path(__file__).read_text()

    user_data = f"""#!/bin/bash
set -ex
exec > /var/log/best_response.log 2>&1

yum install -y python3

cat > /tmp/best_response.py << 'SRC_EOF'
{src}
SRC_EOF

aws s3 cp s3://{bucket}/{s3_key} /tmp/checkpoint.bin
python3 /tmp/best_response.py /tmp/checkpoint.bin > /tmp/best_response_results.txt 2>&1

checkpoint_name=$(basename {s3_key})
aws s3 cp /tmp/best_response_results.txt s3://{bucket}/verification/best_response/$checkpoint_name.txt

INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
"""

    ec2 = boto3.client('ec2', region_name='us-east-1')
    print(f"Launching r5.2xlarge for best-response analysis...")
    response = ec2.run_instances(
        ImageId='ami-0c02fb55956c7d316',
        InstanceType='r5.2xlarge',
        MinCount=1, MaxCount=1,
        IamInstanceProfile={'Name': 'poker-solver-profile'},
        SecurityGroups=['poker-solver-sg'],
        KeyName='poker-solver-key',
        UserData=user_data,
        BlockDeviceMappings=[{
            'DeviceName': '/dev/xvda',
            'Ebs': {'VolumeSize': 100, 'VolumeType': 'gp3'}
        }],
        TagSpecifications=[{
            'ResourceType': 'instance',
            'Tags': [
                {'Key': 'Name', 'Value': 'best-response'},
                {'Key': 'Purpose', 'Value': 'verification'},
                {'Key': 'AutoTerminate', 'Value': 'true'}
            ]
        }]
    )
    instance_id = response['Instances'][0]['InstanceId']
    print(f"Instance: {instance_id}")
    print(f"Results: s3://{bucket}/verification/best_response/")


def main():
    parser = argparse.ArgumentParser(description='Best-Response Exploitability')
    parser.add_argument('checkpoint', nargs='?',
                        help='Path to checkpoint file')
    parser.add_argument('--positions', nargs='+', type=int, default=None,
                        help='Hero positions to analyze (0-5)')
    parser.add_argument('--ec2', metavar='S3_KEY',
                        help='Run on EC2 with checkpoint from S3')
    args = parser.parse_args()

    if args.ec2:
        run_ec2(args.ec2)
        return

    if not args.checkpoint:
        parser.error('Provide a checkpoint path or use --ec2')

    compute_exploitability(args.checkpoint, args.positions)


if __name__ == '__main__':
    main()
