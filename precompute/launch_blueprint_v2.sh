#!/bin/bash
# Launch parallel EC2 instances for blueprint v2 precomputation.
#
# Blueprint v2: preflop-filtered 2-player solves
#   27 scenarios × 1,755 textures = 47,385 work items
#   20M iterations, ~45s/solve → ~5h on 30 × c5.4xlarge (~$100)
#
# Prerequisites:
#   1. Run aws_setup.sh once (creates S3 bucket, security group, IAM role)
#   2. AWS CLI configured (aws configure)
#   3. ranges.json available (auto-uploaded to S3)
#
# Usage:
#   ./launch_blueprint_v2.sh                              # defaults: 30 instances, 20M iter
#   ./launch_blueprint_v2.sh --instances 1 --iterations 100000  # local-equivalent test
#   ./launch_blueprint_v2.sh --dry-run                    # show plan, don't launch
#
# Monitor:
#   ./launch_blueprint_v2.sh --status
#
# Download:
#   ./launch_blueprint_v2.sh --download

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.4xlarge}"  # 16 vCPU, 32 GB, ~$0.68/hr
KEY_NAME="${KEY_NAME:-poker-solver-key}"
SECURITY_GROUP="${SECURITY_GROUP:-poker-solver-sg}"
S3_BUCKET="${S3_BUCKET:-poker-blueprint-v2}"
PROFILE_NAME="poker-solver-profile"

NUM_INSTANCES=30
ITERATIONS=20000000
EHS_SAMPLES=500
DRY_RUN=0
STATUS_ONLY=0
DOWNLOAD_ONLY=0
RESUME=0
AMI_ID=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ranges.json locations to search
RANGES_CANDIDATES=(
    "$PROJECT_DIR/../ACRPoker-Hud-PC/solver/ranges.json"
    "$PROJECT_DIR/data/ranges.json"
)

# ── Parse args ───────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --instances)     NUM_INSTANCES="$2"; shift 2;;
        --iterations)    ITERATIONS="$2"; shift 2;;
        --ehs-samples)   EHS_SAMPLES="$2"; shift 2;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2;;
        --key)           KEY_NAME="$2"; shift 2;;
        --bucket)        S3_BUCKET="$2"; shift 2;;
        --ranges)        RANGES_PATH="$2"; shift 2;;
        --resume)        RESUME=1; shift;;
        --dry-run)       DRY_RUN=1; shift;;
        --status)        STATUS_ONLY=1; shift;;
        --download)      DOWNLOAD_ONLY=1; shift;;
        *)               echo "Unknown: $1"; exit 1;;
    esac
done

# ── Find ranges.json ─────────────────────────────────────────────────────

if [ -z "${RANGES_PATH:-}" ]; then
    for c in "${RANGES_CANDIDATES[@]}"; do
        if [ -f "$c" ]; then
            RANGES_PATH="$c"
            break
        fi
    done
fi

if [ -z "${RANGES_PATH:-}" ] || [ ! -f "${RANGES_PATH}" ]; then
    echo "ERROR: ranges.json not found. Use --ranges /path/to/ranges.json"
    exit 1
fi

# ── Status check ─────────────────────────────────────────────────────────

if [ "$STATUS_ONLY" = "1" ]; then
    echo "=== Blueprint v2 Instance Status ==="
    aws ec2 describe-instances \
        --region "$REGION" \
        --filters "Name=tag:Project,Values=poker-solver-v2" "Name=instance-state-name,Values=running,pending" \
        --query "Reservations[].Instances[].{Id:InstanceId,State:State.Name,Type:InstanceType,Launch:LaunchTime}" \
        --output table
    echo ""
    echo "S3 results:"
    aws s3 ls "s3://$S3_BUCKET/" --recursive --summarize 2>/dev/null | tail -5
    echo ""

    # Count .bps files per scenario
    echo "Files per scenario:"
    for scenario in $(aws s3 ls "s3://$S3_BUCKET/worker-0/" 2>/dev/null | grep PRE | awk '{print $2}' | tr -d '/'); do
        count=$(aws s3 ls "s3://$S3_BUCKET/" --recursive 2>/dev/null | grep "/$scenario/" | grep -c ".bps" || true)
        echo "  $scenario: $count files"
    done
    exit 0
fi

# ── Download results ─────────────────────────────────────────────────────

if [ "$DOWNLOAD_ONLY" = "1" ]; then
    echo "=== Downloading Blueprint v2 Results ==="
    DOWNLOAD_DIR="$PROJECT_DIR/blueprint_v2_output"
    mkdir -p "$DOWNLOAD_DIR"
    aws s3 sync "s3://$S3_BUCKET/" "$DOWNLOAD_DIR/" --exclude "code/*" --exclude "logs/*"
    echo "Downloaded to $DOWNLOAD_DIR"
    NUM_FILES=$(find "$DOWNLOAD_DIR" -name "*.bps" | wc -l)
    echo "Blueprint files: $NUM_FILES / 47385 expected"
    exit 0
fi

# ── Main launch ──────────────────────────────────────────────────────────

TOTAL_ITEMS=47385
ITEMS_PER_WORKER=$(( (TOTAL_ITEMS + NUM_INSTANCES - 1) / NUM_INSTANCES ))

echo "============================================"
echo "  Blueprint v2 Precompute Launcher"
echo "============================================"
echo "Instances:      $NUM_INSTANCES x $INSTANCE_TYPE"
echo "Players:        2 (preflop-filtered)"
echo "Scenarios:      27 (12 SRP + 15 3BP)"
echo "Textures:       1,755"
echo "Total items:    $TOTAL_ITEMS"
echo "Items/worker:   ~$ITEMS_PER_WORKER"
echo "Iterations:     $ITERATIONS per solve"
echo "EHS samples:    $EHS_SAMPLES"
echo "S3 bucket:      $S3_BUCKET"
echo "Ranges:         $RANGES_PATH"
echo "Region:         $REGION"
echo "Resume:         $RESUME"
echo ""

# Cost estimate
OD_PRICE="0.68"
EST_HOURS=5
EST_COST=$(echo "$NUM_INSTANCES * $OD_PRICE * $EST_HOURS" | bc 2>/dev/null || echo "102")
echo "Estimated: ~${EST_HOURS}h runtime, ~\$$EST_COST total cost"
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

# ── Upload code + ranges to S3 ──────────────────────────────────────────

echo "Uploading code and ranges to S3..."
aws s3 sync "$PROJECT_DIR/src" "s3://$S3_BUCKET/code/src/" \
    --quiet --exclude "*.o" --exclude "*.obj" --exclude "*.dll" --exclude "*.exe"
aws s3 sync "$PROJECT_DIR/precompute" "s3://$S3_BUCKET/code/precompute/" --quiet
aws s3 sync "$PROJECT_DIR/python" "s3://$S3_BUCKET/code/python/" \
    --quiet --exclude "__pycache__/*" --exclude "*.pyc"
aws s3 cp "$RANGES_PATH" "s3://$S3_BUCKET/ranges.json" --quiet
echo "Code and ranges uploaded."

# ── User-data generator ─────────────────────────────────────────────────

generate_userdata() {
    local worker_id=$1
    local resume_flag=""
    if [ "$RESUME" = "1" ]; then
        resume_flag="--resume"
    fi
    cat <<USERDATA
#!/bin/bash
set -euxo pipefail
exec > /var/log/blueprint-v2-worker.log 2>&1

echo "=== Blueprint v2 Worker $worker_id starting at \$(date) ==="

# Install build tools
yum install -y gcc gcc-c++ python3 python3-pip libgomp

# Work directory
WORKDIR=/tmp/poker-solver
mkdir -p \$WORKDIR/build && cd \$WORKDIR

# Download code and ranges
aws s3 sync s3://$S3_BUCKET/code/ \$WORKDIR/ --quiet
aws s3 cp s3://$S3_BUCKET/ranges.json \$WORKDIR/ranges.json --quiet

# Compile for Linux
echo "Compiling..."
gcc -O2 -fPIC -shared -fopenmp -o build/mccfr_blueprint.so src/mccfr_blueprint.c -I src -lm -lpthread
gcc -O2 -fPIC -shared -o build/card_abstraction.so src/card_abstraction.c -I src -lm
echo "Compilation complete."

# Run v2 worker
echo "Starting v2 solve..."
export OMP_STACKSIZE=16m
python3 precompute/blueprint_worker_v2.py \
    --worker-id $worker_id \
    --total-workers $NUM_INSTANCES \
    --iterations $ITERATIONS \
    --ehs-samples $EHS_SAMPLES \
    --num-threads 0 \
    --output-dir /tmp/blueprint_v2_output \
    --s3-bucket $S3_BUCKET \
    --build-dir build \
    --ranges \$WORKDIR/ranges.json \
    $resume_flag

echo "=== Worker $worker_id complete at \$(date) ==="

# Upload log
aws s3 cp /var/log/blueprint-v2-worker.log s3://$S3_BUCKET/logs/worker-$worker_id.log --quiet

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
        --iam-instance-profile "Name=$PROFILE_NAME" \
        --user-data "$USERDATA_B64" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=bp-v2-worker-$i},{Key=Project,Value=poker-solver-v2},{Key=WorkerId,Value=$i}]" \
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
echo "  Results:  aws s3 ls s3://$S3_BUCKET/worker-0/ --recursive | head"
echo "  Download: $0 --download"
echo ""
echo "  Kill all: aws ec2 terminate-instances --instance-ids $INSTANCE_IDS"

# Save state
echo "$INSTANCE_IDS" > /tmp/blueprint_v2_instance_ids.txt
