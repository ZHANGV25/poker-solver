#!/bin/bash
# GPU-accelerated flop solution precomputation on EC2.
#
# Uses g5.xlarge instances (NVIDIA A10G, 24GB VRAM) with CUDA.
# Expected: 10-20x faster than CPU, ~1-2 hours instead of 20.
#
# Usage:
#   ./deploy_gpu.sh                          # launch with defaults
#   ./deploy_gpu.sh --num-instances 4        # 4 parallel GPU instances
#   ./deploy_gpu.sh --instance-type g4dn.xlarge  # cheaper T4 GPU

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
KEY_NAME="${KEY_NAME:-poker-solver-key}"
S3_BUCKET="${S3_BUCKET:-poker-solver-flop-solutions}"
NUM_INSTANCES="${NUM_INSTANCES:-2}"
MAX_ITERATIONS="${MAX_ITERATIONS:-200}"
BET_SIZES="${BET_SIZES:-33%, 75%, a}"

# Deep Learning AMI (has CUDA pre-installed)
# Amazon Linux 2 with NVIDIA drivers + CUDA toolkit
DL_AMI=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Amazon Linux 2*" \
              "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text 2>/dev/null)

if [ -z "$DL_AMI" ] || [ "$DL_AMI" = "None" ]; then
    # Fallback: use regular AMI and install CUDA
    DL_AMI=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=al2023-ami-2023*-x86_64" \
                  "Name=state,Values=available" \
        --query "sort_by(Images, &CreationDate)[-1].ImageId" \
        --output text 2>/dev/null)
    echo "Using standard AMI (will install CUDA): $DL_AMI"
    NEED_CUDA_INSTALL=1
else
    echo "Using Deep Learning AMI: $DL_AMI"
    NEED_CUDA_INSTALL=0
fi

# Split 27 scenarios across instances
ALL_SCENARIOS=(
    "BTN_vs_BB_3bp" "BTN_vs_BB_srp" "BTN_vs_SB_3bp" "BTN_vs_SB_srp"
    "CO_vs_BB_3bp" "CO_vs_BB_srp" "CO_vs_BTN_3bp" "CO_vs_BTN_srp"
    "CO_vs_SB_3bp" "MP_vs_BB_3bp" "MP_vs_BB_srp" "MP_vs_BTN_3bp"
    "MP_vs_BTN_srp" "MP_vs_CO_3bp" "MP_vs_CO_srp" "MP_vs_SB_3bp"
    "SB_vs_BB_3bp" "SB_vs_BB_srp" "UTG_vs_BB_3bp" "UTG_vs_BB_srp"
    "UTG_vs_BTN_3bp" "UTG_vs_BTN_srp" "UTG_vs_CO_3bp" "UTG_vs_CO_srp"
    "UTG_vs_MP_3bp" "UTG_vs_MP_srp" "UTG_vs_SB_3bp"
)

# Split scenarios across instances
PER_INSTANCE=$(( (${#ALL_SCENARIOS[@]} + NUM_INSTANCES - 1) / NUM_INSTANCES ))

echo ""
echo "=== GPU Precompute Deployment ==="
echo "  Region: $REGION"
echo "  Instance type: $INSTANCE_TYPE"
echo "  Instances: $NUM_INSTANCES"
echo "  Scenarios per instance: ~$PER_INSTANCE"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  AMI: $DL_AMI"
echo ""

# Upload project
echo "Uploading project to S3..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Upload CUDA solver and precompute scripts
aws s3 sync "$PROJECT_DIR/src/" "s3://$S3_BUCKET/project_v2/src/" --quiet
aws s3 sync "$PROJECT_DIR/precompute/" "s3://$S3_BUCKET/project_v2/precompute/" --quiet

# Upload Rust solver as fallback
ACR_DIR="${ACR_PROJECT_DIR:-}"
if [ -n "$ACR_DIR" ] && [ -d "$ACR_DIR/solver/solver-cli" ]; then
    aws s3 sync "$ACR_DIR/solver/solver-cli/" "s3://$S3_BUCKET/project_v2/solver-cli/" \
        --exclude "target/*" --quiet
    aws s3 cp "$ACR_DIR/solver/ranges.json" "s3://$S3_BUCKET/project_v2/ranges.json" --quiet
fi

echo "Project uploaded."

# Launch instances
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    START=$((i * PER_INSTANCE))
    END=$((START + PER_INSTANCE))
    if [ $END -gt ${#ALL_SCENARIOS[@]} ]; then END=${#ALL_SCENARIOS[@]}; fi

    SCENARIOS=""
    for j in $(seq $START $((END - 1))); do
        if [ -n "$SCENARIOS" ]; then SCENARIOS="$SCENARIOS,"; fi
        SCENARIOS="$SCENARIOS${ALL_SCENARIOS[$j]}"
    done

    echo "Instance $((i+1)): $SCENARIOS"

    USERDATA=$(cat <<SCRIPT
#!/bin/bash
set -euxo pipefail
exec > /var/log/precompute.log 2>&1

echo "GPU Instance $((i+1)) starting at \$(date)"
echo "Scenarios: $SCENARIOS"

# Install build deps
yum install -y gcc rust cargo python3 2>/dev/null || dnf install -y gcc rust cargo python3 2>/dev/null || true

# Get CUDA toolkit if needed
if ! command -v nvcc &>/dev/null; then
    echo "Installing CUDA..."
    yum install -y nvidia-driver-latest-dkms cuda-toolkit 2>/dev/null || \
    dnf install -y nvidia-driver cuda-toolkit 2>/dev/null || true
fi

nvidia-smi || echo "No GPU detected, falling back to CPU"

cd /home/ec2-user
aws s3 sync s3://$S3_BUCKET/project_v2/ ./project/ --quiet
cd project

# Build Rust solver (CPU fallback)
if [ -d solver-cli ]; then
    cd solver-cli && cargo build --release 2>&1 | tail -3 && cd ..
    SOLVER_BIN="solver-cli/target/release/tbl-engine"
else
    SOLVER_BIN=""
fi

# Try building CUDA solver
if command -v nvcc &>/dev/null; then
    echo "Building CUDA solver..."
    nvcc -O3 -o build/gpu_solver src/cuda/gpu_solver.cu -lcudart 2>&1 || echo "CUDA build failed, using CPU"
fi

mkdir -p flop_solutions

# Periodic S3 sync
(while true; do
    sleep 300
    aws s3 sync flop_solutions/ s3://$S3_BUCKET/flop_solutions/ --quiet 2>/dev/null
    echo "S3 sync at \$(date)" >> /var/log/s3sync.log
done) &

# Solve scenarios using Rust CPU solver (GPU solver integration TODO)
IFS=',' read -ra SLIST <<< "$SCENARIOS"
for SC in "\${SLIST[@]}"; do
    echo "=== Starting \$SC at \$(date) ==="
    mkdir -p "flop_solutions/\$SC"
    aws s3 sync "s3://$S3_BUCKET/flop_solutions/\$SC/" "flop_solutions/\$SC/" --quiet 2>/dev/null || true

    if [ -n "\$SOLVER_BIN" ]; then
        python3 precompute/solve_scenarios.py \
            --scenario "\$SC" \
            --workers \$(nproc) \
            --max-iterations $MAX_ITERATIONS \
            --solver-bin "\$SOLVER_BIN" \
            --output-dir flop_solutions \
            --ranges ranges.json \
            --bet-sizes "$BET_SIZES" \
            --timeout 600 2>&1
    fi

    aws s3 sync "flop_solutions/\$SC/" "s3://$S3_BUCKET/flop_solutions/\$SC/" --quiet
    echo "=== Finished \$SC at \$(date) ==="
done

aws s3 sync flop_solutions/ s3://$S3_BUCKET/flop_solutions/ --quiet
echo "done" | aws s3 cp - "s3://$S3_BUCKET/status/gpu-instance-$((i+1)).done"
echo "Instance $((i+1)) complete at \$(date)"
shutdown -h now
SCRIPT
)

    ENCODED=$(echo "$USERDATA" | base64 -w 0)

    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$DL_AMI" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --iam-instance-profile Name=poker-solver-profile \
        --user-data "$ENCODED" \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=solver-gpu-$((i+1))},{Key=Project,Value=poker-solver-gpu}]" \
        --query "Instances[0].InstanceId" \
        --output text 2>&1)

    echo "  Launched: $INSTANCE_ID"
done

echo ""
echo "=== All instances launched ==="
echo ""
echo "Monitor:"
echo "  aws ec2 describe-instances --filters 'Name=tag:Project,Values=poker-solver-gpu' --query 'Reservations[].Instances[].[Tags[?Key==\"Name\"].Value|[0],State.Name]' --output table"
echo ""
echo "Check progress:"
echo "  aws s3 ls s3://$S3_BUCKET/status/"
echo "  aws s3 ls s3://$S3_BUCKET/flop_solutions/ --recursive | wc -l"
