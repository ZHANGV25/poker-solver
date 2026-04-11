#!/bin/bash
# Production Phase 1.3 run against the v2 1.5B checkpoint.
#
# Runs export_v2.py end-to-end:
#   1. Download 55.6 GB regrets checkpoint from S3
#   2. Load into 2.1B-slot hash table (parallel mmap loader)
#   3. Export strategies
#   4. Phase 1.3 EV walk (32-thread, 50M iters, σ̄-sampled)
#   5. Export action EVs as BPR3 section
#   6. LZMA-compress everything
#   7. Upload final .bps to S3
#
# The solver is configured with bp_set_legacy_hash_mixer(1) + tiered
# preflop sizing to match v2 training. Without those, every non-root
# info set lookup fails because the stored action_hash values don't
# match what the current splitmix64 code computes.
#
# Output: s3://poker-blueprint-unified-v2/unified_blueprint_v3_1.5B_FIXED.bps
# Expected wall-clock: ~65 min, cost ~$1.50 spot.

set -euo pipefail

REGION="us-east-1"
INSTANCE_TYPE="r7a.16xlarge"   # 64 vCPU, 512 GB RAM
KEY_NAME="poker-solver-key"
SECURITY_GROUP="sg-07960382eb9d00a95"
S3_BUCKET="poker-blueprint-unified-v2"
PROFILE_NAME="poker-solver-profile"

AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023*-x86_64" \
              "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text)
echo "AMI: $AMI_ID"

USERDATA=$(cat <<'USERDATA_EOF'
#!/bin/bash
set -euxo pipefail
exec > /var/log/phase13-prod.log 2>&1

echo "=== Phase 1.3 PROD run starting at $(date) ==="

S3_BUCKET=poker-blueprint-unified-v2

# Periodic log uploader
(
  while true; do
    aws s3 cp /var/log/phase13-prod.log s3://$S3_BUCKET/logs/phase13_prod_live.log --quiet 2>/dev/null || true
    sleep 30
  done
) &

yum install -y gcc gcc-c++ python3 python3-pip libgomp

WORKDIR=/opt/poker-solver
mkdir -p $WORKDIR/build /opt/blueprint_unified
cd $WORKDIR

echo "--- Syncing code from S3 ---"
aws s3 sync s3://$S3_BUCKET/code/src/ $WORKDIR/src/ --quiet
aws s3 sync s3://$S3_BUCKET/code/precompute/ $WORKDIR/precompute/ --quiet

echo "--- Sanity check: fix must be in source ---"
grep -q "bp_set_legacy_hash_mixer" src/mccfr_blueprint.c || {
  echo "FATAL: mccfr_blueprint.c is missing bp_set_legacy_hash_mixer"
  aws s3 cp /var/log/phase13-prod.log s3://$S3_BUCKET/logs/phase13_prod_FAILED.log || true
  shutdown -h now
  exit 1
}
grep -q "bp_set_preflop_tier" precompute/export_v2.py || {
  echo "FATAL: export_v2.py is missing bp_set_preflop_tier"
  aws s3 cp /var/log/phase13-prod.log s3://$S3_BUCKET/logs/phase13_prod_FAILED.log || true
  shutdown -h now
  exit 1
}
grep -q "bp_set_legacy_hash_mixer" precompute/export_v2.py || {
  echo "FATAL: export_v2.py is missing bp_set_legacy_hash_mixer call"
  aws s3 cp /var/log/phase13-prod.log s3://$S3_BUCKET/logs/phase13_prod_FAILED.log || true
  shutdown -h now
  exit 1
}

echo "--- Compiling mccfr_blueprint.so ---"
gcc -O2 -march=native -fno-strict-aliasing -fPIC -shared -fopenmp \
    -o build/mccfr_blueprint.so \
    src/mccfr_blueprint.c src/card_abstraction.c \
    -I src -lm -lpthread
echo "Compiled ($(ls -lh build/mccfr_blueprint.so | awk '{print $5}'))"

# Enable transparent hugepages for TLB efficiency
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true

echo "--- Downloading texture cache ---"
aws s3 cp s3://$S3_BUCKET/texture_cache.bin /tmp/texture_cache.bin --quiet
echo "Texture cache: $(ls -lh /tmp/texture_cache.bin)"

echo "--- Downloading 1.5B regrets checkpoint (55.6 GB) ---"
time aws s3 cp s3://$S3_BUCKET/checkpoints/regrets_1500M.bin /opt/blueprint_unified/regrets_1500M.bin
echo "Regrets file: $(ls -lh /opt/blueprint_unified/regrets_1500M.bin)"

echo "--- Memory state before export ---"
free -h

echo "--- Running export_v2.py (full pipeline with Phase 1.3) ---"
cd $WORKDIR
export EV_WALK_ITERS=50000000
export USE_LEGACY_HASH_MIXER=1
export NUM_THREADS=32
# Foreground run so the script waits for completion
python3 precompute/export_v2.py \
    /opt/blueprint_unified/regrets_1500M.bin \
    /opt/blueprint_unified \
    "$S3_BUCKET" \
    2147483648

echo "--- Memory state after export ---"
free -h

echo "--- Uploading fixed .bps to S3 ---"
OUTPUT=/opt/blueprint_unified/unified_blueprint.bps
if [ ! -f "$OUTPUT" ]; then
    echo "FATAL: output .bps not produced at $OUTPUT"
    ls -la /opt/blueprint_unified/
    aws s3 cp /var/log/phase13-prod.log s3://$S3_BUCKET/logs/phase13_prod_FAILED.log || true
    shutdown -h now
    exit 1
fi
FILESIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
FILEBYTES=$(stat -c%s "$OUTPUT")
echo "Output file: $OUTPUT ($FILESIZE, $FILEBYTES bytes)"
time aws s3 cp "$OUTPUT" s3://$S3_BUCKET/unified_blueprint_v3_1.5B_FIXED.bps
echo "Upload complete."

# Completion marker
cat > /tmp/phase13_prod_done.json <<DONE
{
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "instance_id": "$(curl -s http://169.254.169.254/latest/meta-data/instance-id)",
  "output_key": "unified_blueprint_v3_1.5B_FIXED.bps",
  "ev_walk_iters": "50000000",
  "num_threads": "32",
  "use_legacy_hash_mixer": "1",
  "source": "export_v2.py with bp_set_legacy_hash_mixer + bp_set_preflop_tier fix",
  "file_size_bytes": "$FILEBYTES"
}
DONE
aws s3 cp /tmp/phase13_prod_done.json s3://$S3_BUCKET/phase13_prod_done.json

echo "--- Final log upload ---"
aws s3 cp /var/log/phase13-prod.log s3://$S3_BUCKET/logs/phase13_prod_complete.log

echo "=== Done at $(date). Terminating instance. ==="
shutdown -h now
USERDATA_EOF
)

USERDATA_B64=$(echo "$USERDATA" | base64 -w0)

echo ""
echo "=== Launching r7a.16xlarge spot instance ==="
echo "Region: $REGION"
echo "AMI: $AMI_ID"
echo "Instance type: $INSTANCE_TYPE (64 vCPU, 512 GB RAM)"
echo ""

LAUNCH_JSON=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --iam-instance-profile "Name=$PROFILE_NAME" \
    --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --user-data "$USERDATA_B64" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=phase13-prod},{Key=Purpose,Value=phase_1_3_production_rerun}]')

INSTANCE_ID=$(echo "$LAUNCH_JSON" | grep -o '"InstanceId": "[^"]*"' | head -1 | cut -d'"' -f4)
echo "Instance launched: $INSTANCE_ID"
echo ""
echo "Monitor with:"
echo "  aws s3 cp s3://$S3_BUCKET/logs/phase13_prod_live.log - | tail -20"
echo ""
echo "Completion marker:"
echo "  aws s3 ls s3://$S3_BUCKET/phase13_prod_done.json"
echo ""
echo "Terminate manually if needed:"
echo "  aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"

echo "$INSTANCE_ID" > /tmp/phase13_prod_instance_id.txt
