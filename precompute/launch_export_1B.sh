#!/bin/bash
# Launch a one-shot r6a.8xlarge ON-DEMAND instance to export BPS strategies
# from the 1B regret checkpoint. Self-terminates when done.
#
# r6a.8xlarge: 32 vCPU, 256 GB RAM, ~$2.02/hr
# On-demand (not spot) because the 200M run got reclaimed mid-job.
# Hash table: 1B slots (matches training) — uses ~98 GB RAM.
#
# Expected: ~25-35 min total, ~$1-1.50 on-demand.

set -euo pipefail

REGION="us-east-1"
INSTANCE_TYPE="r6a.8xlarge"
KEY_NAME="poker-solver-key"
SECURITY_GROUP_ID="sg-07960382eb9d00a95"
S3_BUCKET="poker-blueprint-unified"
PROFILE_NAME="poker-solver-profile"
AMI_ID="ami-0446b021dec428a7b"  # al2023-ami-2023.10 x86_64
HASH_SIZE=1073741824  # 1 << 30 = 1B slots (matches training)

USERDATA=$(cat <<'USERDATA'
#!/bin/bash
set -euxo pipefail
exec > /var/log/bps-export.log 2>&1

BENCH_START=$(date +%s)
echo "=== BPS Export 1B starting at $(date) ==="

# Install build tools
T0=$(date +%s)
yum install -y gcc gcc-c++ python3 python3-pip libgomp
echo "BENCH yum_install: $(($(date +%s) - T0))s"

WORKDIR=/opt/poker-solver
mkdir -p $WORKDIR/build /opt/blueprint_unified && cd $WORKDIR

# Download code from S3
T0=$(date +%s)
aws s3 sync s3://poker-blueprint-unified/code/ $WORKDIR/ --quiet
echo "BENCH code_download: $(($(date +%s) - T0))s"

# Compile
T0=$(date +%s)
gcc -O2 -march=native -fno-strict-aliasing -fPIC -shared -fopenmp \
    -o build/mccfr_blueprint.so src/mccfr_blueprint.c src/card_abstraction.c \
    -I src -lm -lpthread
echo "BENCH compile: $(($(date +%s) - T0))s"

# Download cached texture + turn centroids
T0=$(date +%s)
aws s3 cp s3://poker-blueprint-unified/texture_cache.bin /tmp/texture_cache.bin --quiet 2>/dev/null || true
aws s3 cp s3://poker-blueprint-unified/turn_centroids.bin /tmp/turn_centroids.bin --quiet 2>/dev/null || true
echo "BENCH caches_download: $(($(date +%s) - T0))s"

# Download 1B regrets (42.6 GB)
T0=$(date +%s)
aws s3 cp s3://poker-blueprint-unified/checkpoints/regrets_1000M.bin \
    /opt/blueprint_unified/regrets_1000M.bin --quiet
echo "BENCH regrets_download: $(($(date +%s) - T0))s"
ls -lh /opt/blueprint_unified/regrets_1000M.bin

# Patch export_v2.py to write to unified_blueprint_1000M.bps
# so we preserve the 200M .bps in S3 for rollback.
sed -i 's|unified_blueprint\.bps|unified_blueprint_1000M.bps|g' precompute/export_v2.py
sed -i 's|"iterations": 200000000|"iterations": 1000000000|' precompute/export_v2.py
sed -i 's|"checkpoint": "iter_200000000"|"checkpoint": "iter_1000000000"|' precompute/export_v2.py
echo "Patched export_v2.py for 1B checkpoint"

# Run the export
export OMP_STACKSIZE=64m
python3 -u precompute/export_v2.py \
    /opt/blueprint_unified/regrets_1000M.bin \
    /opt/blueprint_unified \
    poker-blueprint-unified \
    1073741824

BENCH_END=$(date +%s)
echo "=== Export complete at $(date) ==="
echo "BENCH total_wall: $((BENCH_END - BENCH_START))s"

# Upload log BEFORE shutdown check (so we can debug on failure)
aws s3 cp /var/log/bps-export.log s3://poker-blueprint-unified/logs/bps-export-1B.log --quiet || true

# Verify upload before self-terminating
if aws s3 ls s3://poker-blueprint-unified/unified_blueprint_1000M.bps >/dev/null 2>&1; then
    SIZE=$(aws s3 ls s3://poker-blueprint-unified/unified_blueprint_1000M.bps --human-readable | awk '{print $3" "$4}')
    echo "Upload confirmed: $SIZE. Self-terminating."
    aws s3 cp /var/log/bps-export.log s3://poker-blueprint-unified/logs/bps-export-1B.log --quiet || true
    shutdown -h now
else
    echo "UPLOAD FAILED — NOT SHUTTING DOWN. Investigate manually."
    aws s3 cp /var/log/bps-export.log s3://poker-blueprint-unified/logs/bps-export-1B-FAILED.log --quiet || true
    sleep 3600
fi
USERDATA
)

echo "============================================"
echo "  BPS Export 1B — On-Demand r6a.8xlarge"
echo "============================================"
echo "Instance:    $INSTANCE_TYPE (256 GB RAM)"
echo "Hash table:  $HASH_SIZE slots (1B — matches training)"
echo "Checkpoint:  regrets_1000M.bin (42.6 GB)"
echo "Market:      on-demand (NOT spot)"
echo "Est. time:   25-35 min"
echo "Est. cost:   ~\$1-1.50"
echo ""

# Base64 encode userdata for --user-data
USERDATA_B64=$(echo "$USERDATA" | base64 -w 0)

# Launch on-demand instance
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP_ID" \
    --iam-instance-profile "Name=$PROFILE_NAME" \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":150,"VolumeType":"gp3","Iops":6000,"Throughput":400}}]' \
    --user-data "$USERDATA_B64" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=bps-export-1B},{Key=Project,Value=poker-solver-export}]" \
    --query "Instances[0].InstanceId" \
    --output text)

echo ""
echo "Instance: $INSTANCE_ID"
echo ""
echo "Monitor state:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text"
echo ""
echo "Tail log (SSH):"
echo "  ssh -i ~/poker-solver-key.pem ec2-user@\$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text) 'sudo tail -f /var/log/bps-export.log'"
echo ""
echo "Results:"
echo "  s3://poker-blueprint-unified/unified_blueprint_1000M.bps"
echo "  s3://poker-blueprint-unified/logs/bps-export-1B.log"
echo ""
echo "Kill if needed:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"

echo "$INSTANCE_ID" > /tmp/bps_export_1B_instance_id.txt
