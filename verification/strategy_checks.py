#!/usr/bin/env python3
"""Strategy Consistency Checks — Verify poker-knowledge invariants.

Compiles and runs the C checker (strategy_checks.c), then parses and
displays results with PASS/FAIL for each check.

Usage:
  # Local mode:
  python3 verification/strategy_checks.py verification/test_checkpoint.bin

  # EC2 mode (launches instance, runs on real checkpoint):
  python3 verification/strategy_checks.py --ec2 checkpoints/regrets_latest.bin
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    status: str  # PASS, FAIL, SKIP
    details: list = field(default_factory=list)


def compile_checker():
    """Compile strategy_checks.c."""
    src = Path(__file__).parent / 'strategy_checks.c'
    binary = Path(__file__).parent / 'strategy_checks'
    if not binary.exists() or src.stat().st_mtime > binary.stat().st_mtime:
        print(f"Compiling {src}...")
        subprocess.run(
            ['gcc', '-O2', '-o', str(binary), str(src), '-lm'],
            check=True
        )
    return str(binary)


def run_checker(checkpoint_path: str) -> str:
    """Run the C checker and return its output."""
    binary = compile_checker()
    print(f"Running strategy checks on {checkpoint_path}...")
    proc = subprocess.run(
        [binary, checkpoint_path],
        capture_output=True, text=True, timeout=1800  # 30 min for 63 GB
    )
    return proc.stdout, proc.stderr, proc.returncode


def parse_results(output: str) -> list:
    """Parse checker output into CheckResult objects."""
    results = []
    current_check = None
    current_details = []

    for line in output.split('\n'):
        # New check section
        m = re.match(r'=== CHECK: (\S+) ===', line)
        if m:
            if current_check:
                results.append(current_check)
            current_check = CheckResult(name=m.group(1), status='UNKNOWN')
            current_details = []
            continue

        # Result line
        m = re.match(r'RESULT: (\S+)', line)
        if m and current_check:
            current_check.status = m.group(1)
            current_check.details = current_details[:]
            continue

        # Detail lines
        if current_check and line.strip() and not line.startswith('==='):
            current_details.append(line.strip())

    if current_check:
        results.append(current_check)

    return results


def display_results(output: str, results: list):
    """Display results in a formatted table."""
    # Extract header info
    m = re.search(r'STRATEGY_CHECKS iterations=(\d+) entries=(\d+)', output)
    if m:
        iters = int(m.group(1))
        entries = int(m.group(2))
        print(f"\nCheckpoint: {iters:,} iterations, {entries:,} entries\n")

    print(f"{'Check':<35} {'Result':<8} {'Details'}")
    print('-' * 80)

    passed = 0
    failed = 0
    skipped = 0

    for r in results:
        status_icon = {'PASS': '[OK]', 'FAIL': '[!!]', 'SKIP': '[--]'}.get(r.status, '[??]')
        # Get the most important detail line
        detail = ''
        for d in r.details:
            if 'fold=' in d or 'freq' in d or 'pct' in d or '%' in d:
                detail = d
                break
        if not detail and r.details:
            detail = r.details[0]

        print(f"  {r.name:<33} {status_icon:<8} {detail}")

        if r.status == 'PASS':
            passed += 1
        elif r.status == 'FAIL':
            failed += 1
        else:
            skipped += 1

    print('-' * 80)
    total = passed + failed
    print(f"  {passed}/{total} checks passed", end='')
    if skipped:
        print(f" ({skipped} skipped)", end='')
    if failed:
        print(f" -- {failed} FAILED", end='')
    print()

    return failed == 0


def run_ec2(s3_checkpoint_key: str, bucket: str = 'poker-blueprint-unified'):
    """Launch EC2 instance to run strategy checks on a real checkpoint."""
    import boto3

    checker_src = (Path(__file__).parent / 'strategy_checks.c').read_text()

    user_data = f"""#!/bin/bash
set -ex
exec > /var/log/strategy_checks.log 2>&1

yum install -y gcc

cat > /tmp/strategy_checks.c << 'CHECKER_EOF'
{checker_src}
CHECKER_EOF

gcc -O2 -o /tmp/strategy_checks /tmp/strategy_checks.c -lm

aws s3 cp s3://{bucket}/{s3_checkpoint_key} /tmp/checkpoint.bin
/tmp/strategy_checks /tmp/checkpoint.bin > /tmp/strategy_results.txt 2>&1

checkpoint_name=$(basename {s3_checkpoint_key})
aws s3 cp /tmp/strategy_results.txt s3://{bucket}/verification/strategy/$checkpoint_name.txt

INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
"""

    ec2 = boto3.client('ec2', region_name='us-east-1')
    print(f"Launching t3.medium for strategy checks on {s3_checkpoint_key}...")
    response = ec2.run_instances(
        ImageId='ami-0c02fb55956c7d316',
        InstanceType='t3.medium',
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
                {'Key': 'Name', 'Value': 'strategy-checks'},
                {'Key': 'Purpose', 'Value': 'verification'},
                {'Key': 'AutoTerminate', 'Value': 'true'}
            ]
        }]
    )
    instance_id = response['Instances'][0]['InstanceId']
    print(f"Instance: {instance_id}")
    print(f"Results will be at s3://{bucket}/verification/strategy/")


def main():
    parser = argparse.ArgumentParser(description='Strategy Consistency Checks')
    parser.add_argument('checkpoint', nargs='?',
                        help='Path to checkpoint file (local mode)')
    parser.add_argument('--ec2', metavar='S3_KEY',
                        help='Run on EC2 with checkpoint from S3')
    parser.add_argument('--parse', metavar='RESULTS_FILE',
                        help='Parse existing results file')
    args = parser.parse_args()

    if args.parse:
        with open(args.parse) as f:
            output = f.read()
        results = parse_results(output)
        ok = display_results(output, results)
        sys.exit(0 if ok else 1)

    if args.ec2:
        run_ec2(args.ec2)
        return

    if not args.checkpoint:
        parser.error('Provide a checkpoint path or use --ec2')

    output, stderr, rc = run_checker(args.checkpoint)
    if stderr:
        print(stderr, file=sys.stderr)

    results = parse_results(output)
    ok = display_results(output, results)

    # Save raw output
    out_path = Path(__file__).parent / 'strategy_results.txt'
    with open(out_path, 'w') as f:
        f.write(output)
    print(f"\nRaw output saved to {out_path}")

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
