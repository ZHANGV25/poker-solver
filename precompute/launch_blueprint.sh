#!/bin/bash
# Launch parallel EC2 instances for blueprint precomputation.
#
# Spins up N spot instances, each solving a partition of the 1,755 flop textures.
# Results uploaded to S3, then downloaded and combined locally.
#
# Prerequisites:
#   - AWS CLI configured with appropriate permissions
#   - SSH key pair registered in AWS
#   - S3 bucket created
#
# Usage:
#   ./launch_blueprint.sh                    # 20 instances, 100K iterations
#   ./launch_blueprint.sh --instances 10     # fewer instances
#   ./launch_blueprint.sh --dry-run          # print plan without launching

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.4xlarge}"    # 16 vCPU, 32 GB RAM, ~$0.28/hr spot
KEY_NAME="${KEY_NAME:-poker-solver-key}"
SECURITY_GROUP="${SECURITY_GROUP:-poker-solver-sg}"
S3_BUCKET="${S3_BUCKET:-poker-solver-blueprints}"
NUM_INSTANCES="${NUM_INSTANCES:-20}"
ITERATIONS="${ITERATIONS:-100000}"
NUM_BUCKETS="${NUM_BUCKETS:-200}"
DRY_RUN="${DRY_RUN:-0}"
AMI_ID="${AMI_ID:-}"  # empty = auto-detect Amazon Linux 2023

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --instances) NUM_INSTANCES="$2"; shift 2;;
        --iterations) ITERATIONS="$2"; shift 2;;
        --buckets) NUM_BUCKETS="$2"; shift 2;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2;;
        --key) KEY_NAME="$2"; shift 2;;
        --bucket) S3_BUCKET="$2"; shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

echo "=== Blueprint Precompute Launcher ==="
echo "Instances:     $NUM_INSTANCES × $INSTANCE_TYPE"
echo "Iterations:    $ITERATIONS per texture"
echo "Buckets:       $NUM_BUCKETS per street"
echo "S3 bucket:     $S3_BUCKET"
echo "Region:        $REGION"
echo ""

# ── Auto-detect AMI ──────────────────────────────────────────────────────

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

# ── Calculate work distribution ──────────────────────────────────────────

# 1,755 textures / N instances
TEXTURES_PER_WORKER=$(( (1755 + NUM_INSTANCES - 1) / NUM_INSTANCES ))
echo "Textures per worker: ~$TEXTURES_PER_WORKER"

# Estimate time: ~5-10 seconds per texture × 88 textures ≈ 10-15 min per worker
# With 100K iterations at 15K iter/s (16 cores) ≈ 7 seconds per texture
EST_SECS=$(( TEXTURES_PER_WORKER * 10 ))
EST_MINS=$(( EST_SECS / 60 ))
EST_COST_PER_INSTANCE="0.28"  # c5.4xlarge spot $/hr
EST_HOURS=$(( (EST_SECS + 3599) / 3600 ))
echo "Estimated time per worker: ~${EST_MINS} minutes"
echo "Estimated total cost: ~\$$(echo "$NUM_INSTANCES * $EST_COST_PER_INSTANCE * $EST_HOURS" | bc) "
echo ""

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would launch $NUM_INSTANCES instances. Exiting."
    exit 0
fi

# ── Upload project to S3 ────────────────────────────────────────────────

echo "Uploading project to S3..."
aws s3 sync "$PROJECT_DIR/src" "s3://$S3_BUCKET/code/src/" --quiet --exclude "*.o" --exclude "*.obj"
aws s3 sync "$PROJECT_DIR/precompute" "s3://$S3_BUCKET/code/precompute/" --quiet
aws s3 sync "$PROJECT_DIR/python" "s3://$S3_BUCKET/code/python/" --quiet --exclude "__pycache__/*"
echo "Code uploaded."

# ── Generate user-data script ────────────────────────────────────────────

generate_userdata() {
    local worker_id=$1
    cat <<'USERDATA_EOF'
#!/bin/bash
set -euxo pipefail

# Install dependencies
yum install -y gcc gcc-c++ python3 python3-pip git aws-cli

# Create work directory
WORKDIR=/tmp/poker-solver
mkdir -p $WORKDIR && cd $WORKDIR

# Download code from S3
USERDATA_EOF

    echo "aws s3 sync s3://$S3_BUCKET/code/ $WORKDIR/ --quiet"

    cat <<USERDATA_EOF2

# Compile C libraries for Linux
cd \$WORKDIR
gcc -O2 -shared -fopenmp -o build/mccfr_blueprint.so src/mccfr_blueprint.c -I src -lm -lpthread
gcc -O2 -shared -o build/card_abstraction.so src/card_abstraction.c -I src -lm

echo "Compilation complete."

# Run blueprint worker
python3 precompute/blueprint_worker.py \\
    --worker-id $worker_id \\
    --total-workers $NUM_INSTANCES \\
    --iterations $ITERATIONS \\
    --num-buckets $NUM_BUCKETS \\
    --num-threads 0 \\
    --output-dir /tmp/blueprint_output \\
    --s3-bucket $S3_BUCKET \\
    --build-dir build

echo "Worker $worker_id complete."

# Self-terminate
shutdown -h now
USERDATA_EOF2
}

# ── Launch instances ─────────────────────────────────────────────────────

echo "Launching $NUM_INSTANCES instances..."
INSTANCE_IDS=""

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    USERDATA=$(generate_userdata $i | base64 -w 0)

    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-groups "$SECURITY_GROUP" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
        --iam-instance-profile Name=poker-solver-instance-profile \
        --user-data "$USERDATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=blueprint-worker-$i},{Key=Project,Value=poker-solver}]" \
        --query "Instances[0].InstanceId" \
        --output text 2>/dev/null) || {
        echo "  [WARN] Failed to launch worker $i"
        continue
    }

    INSTANCE_IDS="$INSTANCE_IDS $INSTANCE_ID"
    echo "  Worker $i: $INSTANCE_ID"
done

echo ""
echo "=== All instances launched ==="
echo "Instance IDs: $INSTANCE_IDS"
echo ""
echo "Monitor with:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_IDS --query 'Reservations[].Instances[].{Id:InstanceId,State:State.Name}' --output table"
echo ""
echo "Check S3 for results:"
echo "  aws s3 ls s3://$S3_BUCKET/ --recursive | tail -20"
echo ""
echo "Download results:"
echo "  aws s3 sync s3://$S3_BUCKET/ ./blueprint_output/ --exclude 'code/*'"

# Save instance IDs for later cleanup
echo "$INSTANCE_IDS" > /tmp/blueprint_instance_ids.txt
echo "Instance IDs saved to /tmp/blueprint_instance_ids.txt"
