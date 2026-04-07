#!/bin/bash
# Launch a one-shot r6a.4xlarge spot instance to export BPS strategies
# from the 200M regret checkpoint. Self-terminates when done.
#
# Expected timeline:
#   ~2 min   instance boot + yum install
#   ~5 min   download regrets_200M.bin (24.6 GB from S3, same region)
#   ~2-5 min init solver (1B-slot hash table allocation + texture precompute)
#   ~2-5 min load regrets into hash table
#   ~1-3 min export strategies (regret match + quantize)
#   ~1-5 min LZMA compress
#   ~1 min   upload .bps to S3
#   --------
#   ~15-25 min total, cost ~$0.10-0.15 spot

set -euo pipefail

REGION="us-east-1"
INSTANCE_TYPE="r6a.4xlarge"   # 16 vCPU, 128 GB RAM
KEY_NAME="poker-solver-key"
SECURITY_GROUP="poker-solver-sg"
S3_BUCKET="poker-blueprint-unified"
PROFILE_NAME="poker-solver-profile"
HASH_SIZE=1073741824  # 1 << 30 = 1B slots

AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023*-x86_64" \
              "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text 2>/dev/null)
echo "AMI: $AMI_ID"

USERDATA=$(cat <<'USERDATA'
#!/bin/bash
set -euxo pipefail
exec > /var/log/bps-export.log 2>&1

BENCH_START=$(date +%s)
echo "=== BPS Export starting at $(date) ==="

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

# Download texture cache (saves ~40 min of precomputation)
T0=$(date +%s)
aws s3 cp s3://poker-blueprint-unified/texture_cache.bin /tmp/texture_cache.bin --quiet 2>/dev/null || true
echo "BENCH texture_cache_download: $(($(date +%s) - T0))s"

# Download regrets checkpoint (24.6 GB)
T0=$(date +%s)
aws s3 cp s3://poker-blueprint-unified/checkpoints/regrets_200M.bin \
    /opt/blueprint_unified/regrets_200M.bin --quiet
echo "BENCH regrets_download: $(($(date +%s) - T0))s"
ls -lh /opt/blueprint_unified/regrets_200M.bin

# Run the export (benchmarked internally)
export OMP_STACKSIZE=64m
python3 -u precompute/export_bps_only.py \
    --regret-file /opt/blueprint_unified/regrets_200M.bin \
    --output-dir /opt/blueprint_unified \
    --s3-bucket poker-blueprint-unified \
    --build-dir build \
    --hash-size 1073741824

BENCH_END=$(date +%s)
echo "=== BPS Export complete at $(date) ==="
echo "BENCH total_wall: $((BENCH_END - BENCH_START))s"

# Upload log
aws s3 cp /var/log/bps-export.log s3://poker-blueprint-unified/logs/bps-export.log --quiet

# Self-terminate
shutdown -h now
USERDATA
)

USERDATA_B64=$(echo "$USERDATA" | base64 | tr -d '\n')

echo "============================================"
echo "  BPS Export — Spot Instance"
echo "============================================"
echo "Instance:    $INSTANCE_TYPE (128 GB RAM)"
echo "Hash table:  $HASH_SIZE slots (1B)"
echo "Checkpoint:  regrets_200M.bin (24.6 GB)"
echo "Est. time:   15-25 min"
echo "Est. cost:   ~\$0.10-0.15 (spot)"
echo ""

# Request spot instance
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-groups "$SECURITY_GROUP" \
    --iam-instance-profile "Name=$PROFILE_NAME" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3","Iops":6000,"Throughput":400}}]' \
    --user-data "$USERDATA_B64" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=bps-export},{Key=Project,Value=poker-solver-export}]" \
    --query "Instances[0].InstanceId" \
    --output text)

echo ""
echo "============================================"
echo "  Instance: $INSTANCE_ID"
echo "============================================"
echo ""
echo "Monitor:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name' --output text"
echo "  ssh -i ~/poker-solver-key.pem ec2-user@\$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text) 'tail -f /var/log/bps-export.log'"
echo ""
echo "Results will appear at:"
echo "  s3://poker-blueprint-unified/unified_blueprint.bps"
echo "  s3://poker-blueprint-unified/export_benchmark.json"
echo "  s3://poker-blueprint-unified/logs/bps-export.log"
echo ""
echo "Kill if needed:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"

echo "$INSTANCE_ID" > /tmp/bps_export_instance_id.txt
