#!/bin/bash
# Launch parallel EC2 spot instances for blueprint precomputation.
#
# Prerequisites:
#   1. Run aws_setup.sh once (creates S3 bucket, security group, IAM role)
#   2. AWS CLI configured (aws configure)
#
# Usage:
#   ./launch_blueprint.sh                           # defaults: 20 instances, 6 players, 1M iter
#   ./launch_blueprint.sh --instances 5 --players 2 # smaller run
#   ./launch_blueprint.sh --dry-run                 # show plan, don't launch
#
# Monitor:
#   ./launch_blueprint.sh --status                  # check instance states
#
# Download:
#   ./launch_blueprint.sh --download                # sync results from S3

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.4xlarge}"  # 16 vCPU, 32 GB, ~$0.28/hr spot
KEY_NAME="${KEY_NAME:-poker-solver-key}"
SECURITY_GROUP="${SECURITY_GROUP:-poker-solver-sg}"
S3_BUCKET="${S3_BUCKET:-poker-solver-blueprints}"
PROFILE_NAME="poker-solver-instance-profile"

NUM_INSTANCES=20
NUM_PLAYERS=6
ITERATIONS=1000000
NUM_BUCKETS=200
EHS_SAMPLES=500
DRY_RUN=0
STATUS_ONLY=0
DOWNLOAD_ONLY=0
AMI_ID=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Parse args ───────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --instances)    NUM_INSTANCES="$2"; shift 2;;
        --players)      NUM_PLAYERS="$2"; shift 2;;
        --iterations)   ITERATIONS="$2"; shift 2;;
        --buckets)      NUM_BUCKETS="$2"; shift 2;;
        --ehs-samples)  EHS_SAMPLES="$2"; shift 2;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2;;
        --key)          KEY_NAME="$2"; shift 2;;
        --bucket)       S3_BUCKET="$2"; shift 2;;
        --dry-run)      DRY_RUN=1; shift;;
        --status)       STATUS_ONLY=1; shift;;
        --download)     DOWNLOAD_ONLY=1; shift;;
        *)              echo "Unknown: $1"; exit 1;;
    esac
done

# ── Status check ─────────────────────────────────────────────────────────

if [ "$STATUS_ONLY" = "1" ]; then
    echo "=== Blueprint Instance Status ==="
    aws ec2 describe-instances \
        --region "$REGION" \
        --filters "Name=tag:Project,Values=poker-solver" "Name=instance-state-name,Values=running,pending" \
        --query "Reservations[].Instances[].{Id:InstanceId,State:State.Name,Type:InstanceType,Launch:LaunchTime}" \
        --output table
    echo ""
    echo "S3 results:"
    aws s3 ls "s3://$S3_BUCKET/" --recursive --summarize 2>/dev/null | tail -5
    exit 0
fi

# ── Download results ─────────────────────────────────────────────────────

if [ "$DOWNLOAD_ONLY" = "1" ]; then
    echo "=== Downloading Blueprint Results ==="
    DOWNLOAD_DIR="$PROJECT_DIR/blueprint_output"
    mkdir -p "$DOWNLOAD_DIR"
    aws s3 sync "s3://$S3_BUCKET/" "$DOWNLOAD_DIR/" --exclude "code/*"
    echo "Downloaded to $DOWNLOAD_DIR"
    # Count results
    NUM_FILES=$(find "$DOWNLOAD_DIR" -name "*.json" -not -name "summary_*" | wc -l)
    echo "Texture files: $NUM_FILES / 1755"
    exit 0
fi

# ── Main launch ──────────────────────────────────────────────────────────

echo "============================================"
echo "  Blueprint Precompute Launcher"
echo "============================================"
echo "Instances:   $NUM_INSTANCES x $INSTANCE_TYPE"
echo "Players:     $NUM_PLAYERS"
echo "Iterations:  $ITERATIONS per texture"
echo "Buckets:     $NUM_BUCKETS"
echo "EHS samples: $EHS_SAMPLES"
echo "S3 bucket:   $S3_BUCKET"
echo "Region:      $REGION"
echo ""

TEXTURES_PER_WORKER=$(( (1755 + NUM_INSTANCES - 1) / NUM_INSTANCES ))
echo "Textures per worker: ~$TEXTURES_PER_WORKER"
echo ""

# ── Spot price check ─────────────────────────────────────────────────────

SPOT_PRICE=$(aws ec2 describe-spot-price-history \
    --region "$REGION" \
    --instance-types "$INSTANCE_TYPE" \
    --product-descriptions "Linux/UNIX" \
    --max-items 1 \
    --query "SpotPriceHistory[0].SpotPrice" \
    --output text 2>/dev/null || echo "unknown")
echo "Current spot price: \$$SPOT_PRICE/hr"
echo "Estimated total cost: ~\$$(echo "$NUM_INSTANCES * $SPOT_PRICE * 2" | bc 2>/dev/null || echo '?') (assuming ~2hr)"
echo ""

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would launch $NUM_INSTANCES instances. Exiting."
    exit 0
fi

# ── AMI detection ────────────────────────────────────────────────────────

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

# ── Upload code to S3 ───────────────────────────────────────────────────

echo "Uploading code to S3..."
aws s3 sync "$PROJECT_DIR/src" "s3://$S3_BUCKET/code/src/" \
    --quiet --exclude "*.o" --exclude "*.obj" --exclude "*.dll" --exclude "*.exe"
aws s3 sync "$PROJECT_DIR/precompute" "s3://$S3_BUCKET/code/precompute/" --quiet
aws s3 sync "$PROJECT_DIR/python" "s3://$S3_BUCKET/code/python/" \
    --quiet --exclude "__pycache__/*" --exclude "*.pyc"
echo "Code uploaded."

# ── User-data generator ─────────────────────────────────────────────────

generate_userdata() {
    local worker_id=$1
    cat <<USERDATA
#!/bin/bash
set -euxo pipefail
exec > /var/log/blueprint-worker.log 2>&1

echo "=== Blueprint Worker $worker_id starting at \$(date) ==="

# Install build tools
yum install -y gcc gcc-c++ python3 python3-pip libgomp

# Work directory
WORKDIR=/tmp/poker-solver
mkdir -p \$WORKDIR/build && cd \$WORKDIR

# Download code
aws s3 sync s3://$S3_BUCKET/code/ \$WORKDIR/ --quiet

# Compile for Linux
echo "Compiling..."
gcc -O2 -fPIC -shared -fopenmp -o build/mccfr_blueprint.so src/mccfr_blueprint.c -I src -lm -lpthread
gcc -O2 -fPIC -shared -o build/card_abstraction.so src/card_abstraction.c -I src -lm
echo "Compilation complete."

# Run worker
echo "Starting solve..."
python3 precompute/blueprint_worker.py \
    --worker-id $worker_id \
    --total-workers $NUM_INSTANCES \
    --num-players $NUM_PLAYERS \
    --iterations $ITERATIONS \
    --num-buckets $NUM_BUCKETS \
    --ehs-samples $EHS_SAMPLES \
    --num-threads 0 \
    --output-dir /tmp/blueprint_output \
    --s3-bucket $S3_BUCKET \
    --build-dir build

echo "=== Worker $worker_id complete at \$(date) ==="

# Upload log
aws s3 cp /var/log/blueprint-worker.log s3://$S3_BUCKET/logs/worker-$worker_id.log --quiet

# Self-terminate
shutdown -h now
USERDATA
}

# ── Launch instances ─────────────────────────────────────────────────────

echo ""
echo "Launching $NUM_INSTANCES instances..."
INSTANCE_IDS=""
LAUNCHED=0

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    USERDATA_B64=$(generate_userdata $i | base64 -w 0)

    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-groups "$SECURITY_GROUP" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
        --iam-instance-profile "Name=$PROFILE_NAME" \
        --user-data "$USERDATA_B64" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=bp-worker-$i},{Key=Project,Value=poker-solver},{Key=WorkerId,Value=$i}]" \
        --query "Instances[0].InstanceId" \
        --output text 2>/dev/null) || {
        echo "  [WARN] Failed to launch worker $i"
        continue
    }

    INSTANCE_IDS="$INSTANCE_IDS $INSTANCE_ID"
    LAUNCHED=$((LAUNCHED + 1))
    echo "  Worker $i: $INSTANCE_ID"
done

echo ""
echo "============================================"
echo "  Launched $LAUNCHED / $NUM_INSTANCES instances"
echo "============================================"
echo ""
echo "Instance IDs:$INSTANCE_IDS"
echo ""
echo "Commands:"
echo "  Monitor:  $0 --status"
echo "  Logs:     aws s3 ls s3://$S3_BUCKET/logs/"
echo "  Results:  aws s3 ls s3://$S3_BUCKET/worker-0/ | head"
echo "  Download: $0 --download"
echo ""
echo "  Kill all: aws ec2 terminate-instances --instance-ids$INSTANCE_IDS"

# Save state
echo "$INSTANCE_IDS" > /tmp/blueprint_instance_ids.txt
