#!/bin/bash
# EC2 deployment script for flop solution precomputation.
#
# Launches g5.xlarge spot instances (A10G GPU) or c5.4xlarge (CPU-only)
# to solve all 27 scenarios × 1,755 flop textures.
#
# Prerequisites:
#   - AWS CLI configured
#   - SSH key pair registered in AWS
#   - The tbl-engine binary compiled for x86_64-linux
#
# Usage:
#   ./deploy.sh                    # launch with defaults
#   ./deploy.sh --instance-type c5.4xlarge --scenario CO_vs_BB_srp
#   ./deploy.sh --num-instances 4  # parallel instances

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.4xlarge}"  # 16 vCPU, 32 GB RAM
KEY_NAME="${KEY_NAME:-poker-solver-key}"
S3_BUCKET="${S3_BUCKET:-poker-solver-flop-solutions}"
NUM_INSTANCES="${NUM_INSTANCES:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-200}"
BET_SIZES="${BET_SIZES:-33%, 75%, a}"
SCENARIO="${SCENARIO:-}"  # empty = all scenarios
AMI_ID="${AMI_ID:-}"  # empty = auto-detect Amazon Linux 2023

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Auto-detect AMI ──────────────────────────────────────────────────────────

if [ -z "$AMI_ID" ]; then
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=al2023-ami-2023*-x86_64" \
                  "Name=state,Values=available" \
        --query "sort_by(Images, &CreationDate)[-1].ImageId" \
        --output text 2>/dev/null)
    echo "Auto-detected AMI: $AMI_ID"
fi

# ── User data script (runs on instance boot) ─────────────────────────────────

generate_userdata() {
    local scenario_flag=""
    if [ -n "$SCENARIO" ]; then
        scenario_flag="--scenario $SCENARIO"
    else
        scenario_flag="--all"
    fi

    cat <<'USERDATA'
#!/bin/bash
set -euxo pipefail

# Install dependencies
dnf install -y gcc rust cargo git python3 python3-pip

# Clone the solver repo (or download from S3)
cd /home/ec2-user
USERDATA

    echo "aws s3 sync s3://$S3_BUCKET/project/ ./poker-solver/ --quiet 2>/dev/null || true"
    echo "cd poker-solver"

    # Build the Rust solver
    cat <<'USERDATA'
if [ -d "solver-cli" ]; then
    cd solver-cli
    cargo build --release
    cd ..
    SOLVER_BIN="solver-cli/target/release/tbl-engine"
else
    echo "ERROR: solver-cli not found"
    exit 1
fi

# Sync existing solutions from S3 (resume mode)
USERDATA

    echo "mkdir -p flop_solutions"
    echo "aws s3 sync s3://$S3_BUCKET/flop_solutions/ flop_solutions/ --quiet 2>/dev/null || true"

    cat <<USERDATA

# Run precomputation
python3 precompute/solve_scenarios.py \\
    $scenario_flag \\
    --workers \$(nproc) \\
    --max-iterations $MAX_ITERATIONS \\
    --solver-bin \$SOLVER_BIN \\
    --output-dir flop_solutions \\
    --ranges data/ranges.json \\
    --bet-sizes "$BET_SIZES" \\
    2>&1 | tee /var/log/precompute.log

# Upload results to S3
aws s3 sync flop_solutions/ s3://$S3_BUCKET/flop_solutions/ --quiet

echo "Precomputation complete. Shutting down in 60 seconds."
sleep 60
shutdown -h now
USERDATA
}

# ── Launch instance ──────────────────────────────────────────────────────────

echo "=== Poker Solver EC2 Precomputation ==="
echo "Region: $REGION"
echo "Instance type: $INSTANCE_TYPE"
echo "Instances: $NUM_INSTANCES"
echo "Max iterations: $MAX_ITERATIONS"
echo "Bet sizes: $BET_SIZES"
echo "Scenario: ${SCENARIO:-all}"
echo ""

# Upload project to S3
echo "Uploading project to S3..."
aws s3 sync "$PROJECT_DIR" "s3://$S3_BUCKET/project/" \
    --exclude "build/*" --exclude "__pycache__/*" --exclude ".git/*" \
    --quiet

# Generate userdata
USERDATA=$(generate_userdata | base64 -w 0)

# Request spot instances
for i in $(seq 1 $NUM_INSTANCES); do
    echo "Launching instance $i/$NUM_INSTANCES..."

    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
        --iam-instance-profile Name=poker-solver-ec2-role \
        --user-data "$USERDATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=poker-solver-$i},{Key=Project,Value=poker-solver}]" \
        --query "Instances[0].InstanceId" \
        --output text 2>/dev/null)

    echo "  Instance: $INSTANCE_ID"
done

echo ""
echo "Instances launched. Monitor with:"
echo "  aws ec2 describe-instances --filters 'Name=tag:Project,Values=poker-solver' --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' --output table"
echo ""
echo "Results will be synced to: s3://$S3_BUCKET/flop_solutions/"
echo "Sync to local: aws s3 sync s3://$S3_BUCKET/flop_solutions/ flop_solutions/"
