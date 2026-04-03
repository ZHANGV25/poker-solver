#!/usr/bin/env python3
"""Convergence Trend Tool — Track blueprint solver convergence over time.

Downloads checkpoints from S3, runs the C checker (check_convergence.c),
parses output, and plots convergence curves.

Usage:
  # Local mode (small test checkpoint):
  python3 verification/convergence_trend.py --local verification/test_checkpoint.bin

  # EC2 mode (launches temporary instances to analyze real checkpoints):
  python3 verification/convergence_trend.py --ec2

  # Parse existing results from S3:
  python3 verification/convergence_trend.py --results-only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ConvergenceResult:
    """Parsed output from check_convergence."""
    iterations: int = 0
    num_entries: int = 0
    table_size: int = 0

    # Per-street stats
    street_counts: dict = field(default_factory=lambda: {
        'Preflop': 0, 'Flop': 0, 'Turn': 0, 'River': 0
    })
    street_pct: dict = field(default_factory=lambda: {
        'Preflop': 0.0, 'Flop': 0.0, 'Turn': 0.0, 'River': 0.0
    })
    street_avg_regret: dict = field(default_factory=lambda: {
        'Preflop': 0.0, 'Flop': 0.0, 'Turn': 0.0, 'River': 0.0
    })

    # Convergence metrics
    max_regret: int = 0
    min_regret: int = 0
    near_uniform_pct: float = 0.0
    dominant_pct: float = 0.0

    # Preflop strategies (bucket -> position -> action probs)
    preflop_strategies: dict = field(default_factory=dict)

    # Sanity checks
    sanity_checks: dict = field(default_factory=dict)

    # Metadata
    checkpoint_path: str = ""
    timestamp: str = ""


def parse_checker_output(text: str) -> ConvergenceResult:
    """Parse the text output of check_convergence into a ConvergenceResult."""
    result = ConvergenceResult()

    # Header
    m = re.search(r'Header: table=(\d+) entries_hdr=(\d+) iters=(\d+)', text)
    if m:
        result.table_size = int(m.group(1))
        result.num_entries = int(m.group(2))
        result.iterations = int(m.group(3))

    # Actual entries
    m = re.search(r'Actual entries: (\d+)', text)
    if m:
        result.num_entries = int(m.group(1))

    # Per-street distribution
    for street in ['Preflop', 'Flop', 'Turn', 'River']:
        pattern = rf'{street}:\s+(\d+)\s+\(\s*([\d.]+)%\)\s+avg\|regret\|=([\d.]+)'
        m = re.search(pattern, text)
        if m:
            result.street_counts[street] = int(m.group(1))
            result.street_pct[street] = float(m.group(2))
            result.street_avg_regret[street] = float(m.group(3))

    # Convergence stats
    m = re.search(r'Max regret: (-?\d+)', text)
    if m:
        result.max_regret = int(m.group(1))
    m = re.search(r'Min regret: (-?\d+)', text)
    if m:
        result.min_regret = int(m.group(1))
    m = re.search(r'Near-uniform:.*\(([\d.]+)%\)', text)
    if m:
        result.near_uniform_pct = float(m.group(1))
    m = re.search(r'Dominant >70%:.*\(([\d.]+)%\)', text)
    if m:
        result.dominant_pct = float(m.group(1))

    # Preflop strategies
    current_pos = None
    for line in text.split('\n'):
        pos_match = re.match(r'\s+(SB|BB|UTG|MP|CO|BTN):', line)
        if pos_match:
            current_pos = pos_match.group(1)
            continue
        strat_match = re.match(
            r'\s+(\d+)\s+(\S+)\s*:\s*(.*)', line
        )
        if strat_match and current_pos:
            bucket = int(strat_match.group(1))
            label = strat_match.group(2)
            actions_str = strat_match.group(3).strip()
            actions = {}
            for am in re.finditer(r'(\w[\w.]*?)=([\d.]+)', actions_str):
                actions[am.group(1)] = float(am.group(2))
            key = f"{bucket}_{label}"
            if key not in result.preflop_strategies:
                result.preflop_strategies[key] = {}
            result.preflop_strategies[key][current_pos] = actions

    # Sanity checks
    for line in text.split('\n'):
        check_match = re.match(r'\s+(\S+)\s+\((\w+)\):\s+(OK|FAIL).*\(fold=([\d.]+)\)', line)
        if check_match:
            hand = check_match.group(1)
            pos = check_match.group(2)
            status = check_match.group(3)
            fold_pct = float(check_match.group(4))
            result.sanity_checks[f"{hand}_{pos}"] = {
                'status': status, 'fold': fold_pct
            }

    return result


def run_checker_local(checkpoint_path: str, checker_binary: str = None) -> ConvergenceResult:
    """Run check_convergence locally on a checkpoint file."""
    if checker_binary is None:
        # Compile if needed
        src = Path(__file__).parent.parent / 'tests' / 'check_convergence.c'
        binary = Path(__file__).parent / 'check_convergence'
        if not binary.exists() or src.stat().st_mtime > binary.stat().st_mtime:
            print(f"Compiling {src}...")
            subprocess.run(
                ['gcc', '-O2', '-o', str(binary), str(src), '-lm'],
                check=True
            )
        checker_binary = str(binary)

    print(f"Running checker on {checkpoint_path}...")
    proc = subprocess.run(
        [checker_binary, checkpoint_path],
        capture_output=True, text=True, timeout=600
    )
    if proc.returncode != 0:
        print(f"Checker failed: {proc.stderr}", file=sys.stderr)
        sys.exit(1)

    result = parse_checker_output(proc.stdout)
    result.checkpoint_path = checkpoint_path
    result.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    print(proc.stdout)
    return result


def run_checker_ec2(s3_checkpoint_path: str,
                    instance_type: str = 't3.medium',
                    results_bucket: str = 'poker-blueprint-unified') -> Optional[ConvergenceResult]:
    """Launch an EC2 instance, run checker, upload results, terminate."""
    import boto3

    ec2 = boto3.client('ec2', region_name='us-east-1')
    s3 = boto3.client('s3', region_name='us-east-1')

    # Check if results already exist for this checkpoint
    checkpoint_name = s3_checkpoint_path.split('/')[-1]
    results_key = f"verification/convergence/{checkpoint_name}.json"
    try:
        obj = s3.get_object(Bucket=results_bucket, Key=results_key)
        data = json.loads(obj['Body'].read())
        print(f"Results already exist for {checkpoint_name}, skipping EC2 launch")
        return ConvergenceResult(**data)
    except s3.exceptions.NoSuchKey:
        pass
    except Exception:
        pass

    # User data script to run on instance
    checker_src = (Path(__file__).parent.parent / 'tests' / 'check_convergence.c').read_text()
    # Escape for heredoc
    checker_src_escaped = checker_src.replace('\\', '\\\\').replace('$', '\\$').replace('`', '\\`')

    user_data = f"""#!/bin/bash
set -ex
exec > /var/log/convergence_check.log 2>&1

# Install dependencies
yum install -y gcc

# Write checker source
cat > /tmp/check_convergence.c << 'CHECKER_EOF'
{checker_src}
CHECKER_EOF

# Compile
gcc -O2 -o /tmp/check_convergence /tmp/check_convergence.c -lm

# Download checkpoint
aws s3 cp s3://{results_bucket}/{s3_checkpoint_path} /tmp/checkpoint.bin

# Run checker
/tmp/check_convergence /tmp/checkpoint.bin > /tmp/results.txt 2>&1

# Upload results
aws s3 cp /tmp/results.txt s3://{results_bucket}/verification/convergence/{checkpoint_name}.txt

# Self-terminate
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
"""

    print(f"Launching {instance_type} for {checkpoint_name}...")
    try:
        response = ec2.run_instances(
            ImageId='ami-0c02fb55956c7d316',  # Amazon Linux 2
            InstanceType=instance_type,
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
                    {'Key': 'Name', 'Value': f'convergence-check-{checkpoint_name}'},
                    {'Key': 'Purpose', 'Value': 'verification'},
                    {'Key': 'AutoTerminate', 'Value': 'true'}
                ]
            }]
        )
        instance_id = response['Instances'][0]['InstanceId']
        print(f"  Instance: {instance_id}")
        return None  # Results will be available later
    except Exception as e:
        print(f"  EC2 launch failed: {e}", file=sys.stderr)
        return None


def list_checkpoints_s3(bucket: str = 'poker-blueprint-unified') -> list:
    """List available checkpoints in S3."""
    import boto3
    s3 = boto3.client('s3', region_name='us-east-1')
    checkpoints = []
    try:
        response = s3.list_objects_v2(
            Bucket=bucket, Prefix='checkpoints/', Delimiter='/'
        )
        for obj in response.get('Contents', []):
            if obj['Key'].endswith('.bin'):
                checkpoints.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'modified': obj['LastModified'].isoformat()
                })
    except Exception as e:
        print(f"Error listing S3: {e}", file=sys.stderr)
    return checkpoints


def collect_results_s3(bucket: str = 'poker-blueprint-unified') -> list:
    """Collect all convergence results from S3."""
    import boto3
    s3 = boto3.client('s3', region_name='us-east-1')
    results = []
    try:
        response = s3.list_objects_v2(
            Bucket=bucket, Prefix='verification/convergence/', Delimiter='/'
        )
        for obj in response.get('Contents', []):
            if obj['Key'].endswith('.txt'):
                body = s3.get_object(Bucket=bucket, Key=obj['Key'])['Body'].read().decode()
                result = parse_checker_output(body)
                result.checkpoint_path = obj['Key']
                result.timestamp = obj['LastModified'].isoformat()
                results.append(result)
    except Exception as e:
        print(f"Error collecting results: {e}", file=sys.stderr)
    return sorted(results, key=lambda r: r.iterations)


def plot_convergence(results: list, output_dir: str = 'verification'):
    """Plot convergence curves from a list of ConvergenceResult objects."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print("Skipping plot generation, printing data instead.")
        for r in results:
            print(f"  iters={r.iterations:,}  uniform={r.near_uniform_pct:.1f}%  "
                  f"dominant={r.dominant_pct:.1f}%  "
                  f"avg|r| preflop={r.street_avg_regret.get('Preflop', 0):.0f}  "
                  f"flop={r.street_avg_regret.get('Flop', 0):.0f}  "
                  f"turn={r.street_avg_regret.get('Turn', 0):.0f}  "
                  f"river={r.street_avg_regret.get('River', 0):.0f}")
        return

    iters = [r.iterations / 1e9 for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Blueprint Solver Convergence', fontsize=14, fontweight='bold')

    # 1. Near-uniform % and Dominant >70% over iterations
    ax = axes[0][0]
    ax.plot(iters, [r.near_uniform_pct for r in results], 'b-o', label='Near-uniform %', markersize=4)
    ax.plot(iters, [r.dominant_pct for r in results], 'r-s', label='Dominant >70% %', markersize=4)
    ax.set_xlabel('Iterations (billions)')
    ax.set_ylabel('Percentage')
    ax.set_title('Strategy Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Average |regret| per street
    ax = axes[0][1]
    for street, color in [('Preflop', 'blue'), ('Flop', 'green'),
                          ('Turn', 'orange'), ('River', 'red')]:
        vals = [r.street_avg_regret.get(street, 0) for r in results]
        if any(v > 0 for v in vals):
            ax.plot(iters, vals, '-o', color=color, label=street, markersize=4)
    ax.set_xlabel('Iterations (billions)')
    ax.set_ylabel('Average |regret|')
    ax.set_title('Average Max Regret by Street')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 3. Preflop AA fold % over iterations (should be ~0)
    ax = axes[1][0]
    aa_folds = []
    sevtwo_folds = []
    for r in results:
        aa_fold = 0.0
        s72_fold = 0.0
        aa_data = r.preflop_strategies.get('0_AA', {})
        for pos, acts in aa_data.items():
            aa_fold = max(aa_fold, acts.get('fold', 0.0))
        s72_data = r.preflop_strategies.get('167_32o', {})
        for pos, acts in s72_data.items():
            s72_fold = max(s72_fold, acts.get('fold', 0.0))
        aa_folds.append(aa_fold * 100)
        sevtwo_folds.append(s72_fold * 100)

    if aa_folds:
        ax.plot(iters, aa_folds, 'g-o', label='AA max fold %', markersize=4)
    if sevtwo_folds:
        ax.plot(iters, sevtwo_folds, 'r-s', label='72o max fold %', markersize=4)
    ax.set_xlabel('Iterations (billions)')
    ax.set_ylabel('Fold %')
    ax.set_title('Key Hand Fold Frequencies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    # 4. Info set distribution by street
    ax = axes[1][1]
    for street, color in [('Preflop', 'blue'), ('Flop', 'green'),
                          ('Turn', 'orange'), ('River', 'red')]:
        vals = [r.street_counts.get(street, 0) / max(1, r.num_entries) * 100
                for r in results]
        if any(v > 0 for v in vals):
            ax.plot(iters, vals, '-o', color=color, label=street, markersize=4)
    ax.set_xlabel('Iterations (billions)')
    ax.set_ylabel('% of total info sets')
    ax.set_title('Info Set Distribution by Street')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'convergence_trend.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved convergence plot to {out_path}")
    plt.close()


def save_results_json(results: list, output_dir: str = 'verification'):
    """Save results as JSON for later analysis."""
    out_path = os.path.join(output_dir, 'convergence_results.json')
    data = [asdict(r) for r in results]
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved results to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Blueprint Convergence Trend Tool')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--local', metavar='CHECKPOINT',
                       help='Run checker locally on a checkpoint file')
    group.add_argument('--ec2', action='store_true',
                       help='Launch EC2 instances for each S3 checkpoint')
    group.add_argument('--results-only', action='store_true',
                       help='Collect existing results from S3 and plot')
    parser.add_argument('--output-dir', default='verification',
                        help='Directory for output files')
    parser.add_argument('--bucket', default='poker-blueprint-unified',
                        help='S3 bucket name')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.local:
        result = run_checker_local(args.local)
        results = [result]
        save_results_json(results, args.output_dir)
        plot_convergence(results, args.output_dir)

    elif args.ec2:
        checkpoints = list_checkpoints_s3(args.bucket)
        if not checkpoints:
            print("No checkpoints found in S3")
            sys.exit(1)

        print(f"Found {len(checkpoints)} checkpoints:")
        for cp in checkpoints:
            print(f"  {cp['key']}  ({cp['size'] / 1e9:.1f} GB)  {cp['modified']}")

        for cp in checkpoints:
            run_checker_ec2(cp['key'], results_bucket=args.bucket)

        print("\nEC2 instances launched. Results will be uploaded to "
              f"s3://{args.bucket}/verification/convergence/")
        print("Run with --results-only to collect and plot after they complete.")

    elif args.results_only:
        results = collect_results_s3(args.bucket)
        if not results:
            print("No results found in S3")
            sys.exit(1)
        print(f"Collected {len(results)} results")
        save_results_json(results, args.output_dir)
        plot_convergence(results, args.output_dir)


if __name__ == '__main__':
    main()
