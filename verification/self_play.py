#!/usr/bin/env python3
"""Self-Play Simulator — Simulate hands using blueprint strategies.

Compiles and runs the C simulator, parses and displays results.

Usage:
  # Local (test checkpoint, small number of hands):
  python3 verification/self_play.py verification/test_checkpoint.bin --hands 10000

  # EC2 (real checkpoint, 100K+ hands, needs ~80 GB RAM):
  python3 verification/self_play.py --ec2 checkpoints/regrets_latest.bin --hands 100000

  # Parse existing results:
  python3 verification/self_play.py --parse verification/self_play_results.txt
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def compile_simulator():
    """Compile self_play.c."""
    src = Path(__file__).parent / 'self_play.c'
    binary = Path(__file__).parent / 'self_play'
    if not binary.exists() or src.stat().st_mtime > binary.stat().st_mtime:
        print(f"Compiling {src}...")
        subprocess.run(
            ['gcc', '-O2', '-o', str(binary), str(src), '-lm'],
            check=True
        )
    return str(binary)


def run_simulator(checkpoint_path: str, num_hands: int = 10000,
                  seed: int = 0, ehs_samples: int = 100) -> str:
    """Run the C simulator and return its stdout."""
    binary = compile_simulator()
    print(f"Running self-play: {num_hands} hands on {checkpoint_path}...")
    args = [binary, checkpoint_path, str(num_hands), str(seed), str(ehs_samples)]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=7200)
    if proc.stderr:
        # stderr has progress info
        for line in proc.stderr.strip().split('\n'):
            if 'Loading' in line or 'Loaded' in line or 'Simulating' in line:
                print(f"  {line.strip()}")
    return proc.stdout, proc.returncode


def parse_and_display(output: str):
    """Parse and display self-play results."""
    # Extract header
    m = re.search(r'SELF_PLAY iterations=(\d+) hands=(\d+)', output)
    if m:
        print(f"\nBlueprint: {int(m.group(1)):,} iterations")
        print(f"Simulated: {int(m.group(2)):,} hands\n")

    # Print the structured output sections
    in_section = False
    results = {}
    for line in output.split('\n'):
        if line.startswith('===') or line.startswith('SELF_PLAY'):
            in_section = True
            if '===' in line:
                print(line)
            continue

        check_m = re.match(r'(\w+_CHECK): (PASS|FAIL)', line)
        if check_m:
            name = check_m.group(1)
            status = check_m.group(2)
            icon = '[OK]' if status == 'PASS' else '[!!]'
            results[name] = status
            print(f"  {icon} {line}")
            continue

        if line.strip():
            print(line)

    # Summary
    passed = sum(1 for v in results.values() if v == 'PASS')
    total = len(results)
    print(f"\n{'=' * 40}")
    print(f"  {passed}/{total} checks passed")
    if passed < total:
        failed = [k for k, v in results.items() if v == 'FAIL']
        print(f"  Failed: {', '.join(failed)}")

    return passed == total


def run_ec2(s3_key: str, num_hands: int = 100000,
            bucket: str = 'poker-blueprint-unified'):
    """Launch EC2 instance (r5.2xlarge for 64 GB RAM) to run self-play."""
    import boto3

    src = (Path(__file__).parent / 'self_play.c').read_text()

    user_data = f"""#!/bin/bash
set -ex
exec > /var/log/self_play.log 2>&1

yum install -y gcc

cat > /tmp/self_play.c << 'SRC_EOF'
{src}
SRC_EOF

gcc -O2 -o /tmp/self_play /tmp/self_play.c -lm

aws s3 cp s3://{bucket}/{s3_key} /tmp/checkpoint.bin
/tmp/self_play /tmp/checkpoint.bin {num_hands} 42 200 > /tmp/self_play_results.txt 2>/tmp/self_play_log.txt

checkpoint_name=$(basename {s3_key})
aws s3 cp /tmp/self_play_results.txt s3://{bucket}/verification/self_play/$checkpoint_name.txt
aws s3 cp /tmp/self_play_log.txt s3://{bucket}/verification/self_play/$checkpoint_name.log

INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
"""

    ec2 = boto3.client('ec2', region_name='us-east-1')
    print(f"Launching r5.2xlarge for self-play ({num_hands} hands)...")
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
                {'Key': 'Name', 'Value': 'self-play-sim'},
                {'Key': 'Purpose', 'Value': 'verification'},
                {'Key': 'AutoTerminate', 'Value': 'true'}
            ]
        }]
    )
    instance_id = response['Instances'][0]['InstanceId']
    print(f"Instance: {instance_id}")
    print(f"Results: s3://{bucket}/verification/self_play/")


def main():
    parser = argparse.ArgumentParser(description='Self-Play Simulator')
    parser.add_argument('checkpoint', nargs='?',
                        help='Path to checkpoint file (local mode)')
    parser.add_argument('--hands', type=int, default=10000,
                        help='Number of hands to simulate')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed')
    parser.add_argument('--ehs-samples', type=int, default=100,
                        help='Monte Carlo samples for EHS computation')
    parser.add_argument('--ec2', metavar='S3_KEY',
                        help='Run on EC2 with checkpoint from S3')
    parser.add_argument('--parse', metavar='RESULTS_FILE',
                        help='Parse existing results file')
    args = parser.parse_args()

    if args.parse:
        with open(args.parse) as f:
            output = f.read()
        ok = parse_and_display(output)
        sys.exit(0 if ok else 1)

    if args.ec2:
        run_ec2(args.ec2, args.hands)
        return

    if not args.checkpoint:
        parser.error('Provide a checkpoint path or use --ec2')

    output, rc = run_simulator(args.checkpoint, args.hands, args.seed, args.ehs_samples)

    # Save raw output
    out_path = Path(__file__).parent / 'self_play_results.txt'
    with open(out_path, 'w') as f:
        f.write(output)

    ok = parse_and_display(output)
    print(f"\nRaw output saved to {out_path}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
