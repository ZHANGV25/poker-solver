#!/bin/bash
# Launch a SINGLE large EC2 instance for unified Pluribus-style blueprint.
#
# Pluribus: 64 cores, 8 days, 12,400 CPU-hours, <512GB RAM, ~$144 spot
# Our target: c5.18xlarge (72 vCPU, 144GB) for 8 days ≈ $587 on-demand
#   or c5.metal (96 vCPU, 192GB) for faster convergence
#   or r5.24xlarge (96 vCPU, 768GB) if memory-constrained
#
# The unified solve runs ONE MCCFR instance with all 6 players, preflop
# through river, on a single shared-memory machine — exactly like Pluribus.
#
# Usage:
#   ./launch_blueprint_unified.sh                    # default: c5.18xlarge, 8 days
#   ./launch_blueprint_unified.sh --hours 24         # shorter test run
#   ./launch_blueprint_unified.sh --dry-run          # show plan
#
# Monitor:
#   ./launch_blueprint_unified.sh --status
#   ssh -i key.pem ec2-user@IP "tail -f /var/log/blueprint-unified.log"

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.metal}"  # 96 vCPU, 192 GB, $4.08/hr
KEY_NAME="${KEY_NAME:-poker-solver-key}"
SECURITY_GROUP="${SECURITY_GROUP:-poker-solver-sg}"
S3_BUCKET="${S3_BUCKET:-poker-blueprint-unified}"
PROFILE_NAME="poker-solver-profile"

HOURS=192           # 8 days (Pluribus)
HASH_SIZE=1342177280 # 1.25B slots (~80GB metadata) — hash table hit 100% at 1B with old size
DRY_RUN=0
STATUS_ONLY=0
DOWNLOAD_ONLY=0
AMI_ID=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

while [[ $# -gt 0 ]]; do
    case $1 in
        --hours)          HOURS="$2"; shift 2;;
        --instance-type)  INSTANCE_TYPE="$2"; shift 2;;
        --hash-size)      HASH_SIZE="$2"; shift 2;;
        --key)            KEY_NAME="$2"; shift 2;;
        --bucket)         S3_BUCKET="$2"; shift 2;;
        --dry-run)        DRY_RUN=1; shift;;
        --status)         STATUS_ONLY=1; shift;;
        --download)       DOWNLOAD_ONLY=1; shift;;
        *)                echo "Unknown: $1"; exit 1;;
    esac
done

if [ "$STATUS_ONLY" = "1" ]; then
    echo "=== Unified Blueprint Instance Status ==="
    aws ec2 describe-instances \
        --region "$REGION" \
        --filters "Name=tag:Project,Values=poker-solver-unified" "Name=instance-state-name,Values=running,pending" \
        --query "Reservations[].Instances[].{Id:InstanceId,State:State.Name,Type:InstanceType,Launch:LaunchTime,IP:PublicIpAddress}" \
        --output table
    echo ""
    echo "S3:"
    aws s3 ls "s3://$S3_BUCKET/" 2>/dev/null || echo "(empty)"
    exit 0
fi

if [ "$DOWNLOAD_ONLY" = "1" ]; then
    echo "=== Downloading Unified Blueprint ==="
    mkdir -p "$PROJECT_DIR/blueprint_unified"
    aws s3 sync "s3://$S3_BUCKET/" "$PROJECT_DIR/blueprint_unified/"
    echo "Downloaded to $PROJECT_DIR/blueprint_unified/"
    exit 0
fi

# Memory estimate: 512M slots × ~40 bytes/slot = ~20GB metadata
# Plus strategy data: ~400M sequences × ~16 bytes = ~6.4GB
# Total: ~30GB + overhead → 144GB instance is sufficient
MEM_GB=$(echo "$HASH_SIZE * 40 / 1073741824" | bc 2>/dev/null || echo "20")

echo "============================================"
echo "  Unified Pluribus Blueprint"
echo "============================================"
echo "Instance:    $INSTANCE_TYPE"
echo "Runtime:     ${HOURS}h ($(echo "$HOURS / 24" | bc 2>/dev/null || echo '?') days)"
echo "Hash table:  $HASH_SIZE slots (~${MEM_GB}GB)"
echo "S3 bucket:   $S3_BUCKET"
echo "Region:      $REGION"
echo ""

# Cost estimate
case "$INSTANCE_TYPE" in
    c5.18xlarge) OD_PRICE="3.06";;
    c5.metal)    OD_PRICE="4.08";;
    r5.24xlarge) OD_PRICE="6.05";;
    *)           OD_PRICE="3.00";;
esac
COST=$(echo "$OD_PRICE * $HOURS" | bc 2>/dev/null || echo "???")
echo "Estimated cost: ~\$$COST (on-demand)"
echo ""

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Exiting."
    exit 0
fi

if [ -z "$AMI_ID" ]; then
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=al2023-ami-2023*-x86_64" \
                  "Name=state,Values=available" \
        --query "sort_by(Images, &CreationDate)[-1].ImageId" \
        --output text 2>/dev/null)
    echo "AMI: $AMI_ID"
fi

echo "Uploading code..."
aws s3 sync "$PROJECT_DIR/src" "s3://$S3_BUCKET/code/src/" \
    --quiet --exclude "*.o" --exclude "*.obj" --exclude "*.dll" --exclude "*.exe"
aws s3 sync "$PROJECT_DIR/precompute" "s3://$S3_BUCKET/code/precompute/" --quiet
aws s3 sync "$PROJECT_DIR/python" "s3://$S3_BUCKET/code/python/" \
    --quiet --exclude "__pycache__/*" --exclude "*.pyc"
echo "Code uploaded."

USERDATA=$(cat <<USERDATA
#!/bin/bash
set -euxo pipefail
exec > /var/log/blueprint-unified.log 2>&1

echo "=== Unified Blueprint starting at \$(date) ==="

yum install -y gcc gcc-c++ python3 python3-pip libgomp

WORKDIR=/opt/poker-solver
mkdir -p \$WORKDIR/build /opt/blueprint_unified && cd \$WORKDIR
aws s3 sync s3://$S3_BUCKET/code/ \$WORKDIR/ --quiet

echo "Compiling with -O3 -march=native for maximum throughput..."
gcc -O3 -march=native -fPIC -shared -fopenmp -o build/mccfr_blueprint.so src/mccfr_blueprint.c src/card_abstraction.c -I src -lm -lpthread
echo "Compilation complete."

export OMP_STACKSIZE=64m
export OMP_NUM_THREADS=\$(nproc)

echo "Starting unified solve: \$(nproc) threads, ${HOURS}h..."
python3 precompute/blueprint_worker_unified.py \
    --time-limit-hours $HOURS \
    --num-threads \$(nproc) \
    --hash-size $HASH_SIZE \
    --output-dir /opt/blueprint_unified \
    --s3-bucket $S3_BUCKET \
    --checkpoint-interval 10000000000 \
    --build-dir build \
    --resume

echo "=== Complete at \$(date) ==="
aws s3 cp /var/log/blueprint-unified.log s3://$S3_BUCKET/logs/unified.log --quiet
shutdown -h now
USERDATA
)

USERDATA_B64=$(echo "$USERDATA" | base64 | tr -d '\n')

echo "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-groups "$SECURITY_GROUP" \
    --iam-instance-profile "Name=$PROFILE_NAME" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop"}}' \
    --user-data "$USERDATA_B64" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=bp-unified},{Key=Project,Value=poker-solver-unified}]" \
    --query "Instances[0].InstanceId" \
    --output text)

echo ""
echo "============================================"
echo "  Instance: $INSTANCE_ID"
echo "============================================"
echo ""
echo "Monitor:"
echo "  $0 --status"
echo "  ssh -i ~/poker-solver-key.pem ec2-user@\$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text) 'tail -f /var/log/blueprint-unified.log'"
echo ""
echo "Download: $0 --download"
echo "Kill: aws ec2 terminate-instances --instance-ids $INSTANCE_ID"

echo "$INSTANCE_ID" > /tmp/blueprint_unified_instance_id.txt
