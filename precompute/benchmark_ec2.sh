#!/bin/bash
# Launch a SINGLE EC2 spot instance to benchmark blueprint generation.
#
# Usage:
#   ./benchmark_ec2.sh                    # launch benchmark
#   ./benchmark_ec2.sh --status           # check if running
#   ./benchmark_ec2.sh --results          # download and show results
#   ./benchmark_ec2.sh --ssh              # SSH into the instance
#   ./benchmark_ec2.sh --kill             # terminate the instance

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.4xlarge}"
KEY_NAME="${KEY_NAME:-poker-solver-key}"
SECURITY_GROUP="${SECURITY_GROUP:-poker-solver-sg}"
S3_BUCKET="${S3_BUCKET:-poker-solver-blueprints}"
PROFILE_NAME="poker-solver-instance-profile"
AMI_ID=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ACTION="launch"
while [[ $# -gt 0 ]]; do
    case $1 in
        --status)        ACTION="status"; shift;;
        --results)       ACTION="results"; shift;;
        --ssh)           ACTION="ssh"; shift;;
        --kill)          ACTION="kill"; shift;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

# ── Status ───────────────────────────────────────────────────────────────

if [ "$ACTION" = "status" ]; then
    aws ec2 describe-instances --region "$REGION" \
        --filters "Name=tag:Name,Values=bp-benchmark" "Name=instance-state-name,Values=running,pending" \
        --query "Reservations[].Instances[].{Id:InstanceId,State:State.Name,IP:PublicIpAddress,Launch:LaunchTime}" \
        --output table
    echo ""
    echo "Check log:"
    echo "  aws s3 cp s3://$S3_BUCKET/benchmark/benchmark.log - | tail -30"
    exit 0
fi

if [ "$ACTION" = "results" ]; then
    mkdir -p /tmp/bp_benchmark
    aws s3 sync "s3://$S3_BUCKET/benchmark/" /tmp/bp_benchmark/ --quiet 2>/dev/null
    if [ -f /tmp/bp_benchmark/benchmark_results.json ]; then
        echo "=== Benchmark Results ==="
        python3 -m json.tool /tmp/bp_benchmark/benchmark_results.json 2>/dev/null || cat /tmp/bp_benchmark/benchmark_results.json
    elif [ -f /tmp/bp_benchmark/benchmark.log ]; then
        echo "=== Log (results not ready yet) ==="
        tail -50 /tmp/bp_benchmark/benchmark.log
    else
        echo "No results yet."
    fi
    exit 0
fi

if [ "$ACTION" = "ssh" ]; then
    IP=$(aws ec2 describe-instances --region "$REGION" \
        --filters "Name=tag:Name,Values=bp-benchmark" "Name=instance-state-name,Values=running" \
        --query "Reservations[].Instances[0].PublicIpAddress" --output text)
    [ -z "$IP" ] || [ "$IP" = "None" ] && { echo "No running instance."; exit 1; }
    ssh -i "${KEY_NAME}.pem" -o StrictHostKeyChecking=no ec2-user@"$IP"
    exit 0
fi

if [ "$ACTION" = "kill" ]; then
    IDS=$(aws ec2 describe-instances --region "$REGION" \
        --filters "Name=tag:Name,Values=bp-benchmark" "Name=instance-state-name,Values=running,pending" \
        --query "Reservations[].Instances[].InstanceId" --output text)
    [ -z "$IDS" ] && { echo "No instances to kill."; exit 0; }
    echo "Terminating: $IDS"
    aws ec2 terminate-instances --instance-ids $IDS --region "$REGION" > /dev/null
    echo "Done."
    exit 0
fi

# ── Launch ───────────────────────────────────────────────────────────────

echo "=== Launching Benchmark Instance ==="
echo "Type: $INSTANCE_TYPE ($(echo $INSTANCE_TYPE | grep -oP '\d+' | head -1) vCPUs)"

if [ -z "$AMI_ID" ]; then
    AMI_ID=$(aws ec2 describe-images --region "$REGION" --owners amazon \
        --filters "Name=name,Values=al2023-ami-2023*-x86_64" "Name=state,Values=available" \
        --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text)
fi

# Upload code (including benchmark_run.py)
echo "Uploading code..."
aws s3 sync "$PROJECT_DIR/src" "s3://$S3_BUCKET/code/src/" \
    --quiet --exclude "*.o" --exclude "*.obj" --exclude "*.dll" --exclude "*.exe"
aws s3 sync "$PROJECT_DIR/precompute" "s3://$S3_BUCKET/code/precompute/" --quiet
aws s3 sync "$PROJECT_DIR/python" "s3://$S3_BUCKET/code/python/" \
    --quiet --exclude "__pycache__/*"
echo "Done."

# Generate user-data
USERDATA=$(cat <<SCRIPT
#!/bin/bash
set -euxo pipefail
exec > /var/log/benchmark.log 2>&1

echo "=== Benchmark starting at \$(date) ==="
echo "Instance: \$(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
echo "CPUs: \$(nproc)"
echo "RAM: \$(free -h | grep Mem | awk '{print \$2}')"

yum install -y gcc gcc-c++ python3 libgomp

WORKDIR=/tmp/poker-solver
mkdir -p \$WORKDIR/build && cd \$WORKDIR
aws s3 sync s3://$S3_BUCKET/code/ \$WORKDIR/ --quiet

echo "Compiling..."
gcc -O2 -fPIC -shared -fopenmp -o build/mccfr_blueprint.so src/mccfr_blueprint.c -I src -lm -lpthread
gcc -O2 -fPIC -shared -o build/card_abstraction.so src/card_abstraction.c -I src -lm
echo "Compiled."

python3 precompute/benchmark_run.py

aws s3 sync /tmp/benchmark_output/ s3://$S3_BUCKET/benchmark/ --quiet
aws s3 cp /var/log/benchmark.log s3://$S3_BUCKET/benchmark/benchmark.log --quiet

echo "=== Benchmark complete at \$(date) ==="
SCRIPT
)

USERDATA_B64=$(echo "$USERDATA" | base64 -w 0)

INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-groups "$SECURITY_GROUP" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
    --iam-instance-profile "Name=$PROFILE_NAME" \
    --user-data "$USERDATA_B64" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=bp-benchmark},{Key=Project,Value=poker-solver}]" \
    --query "Instances[0].InstanceId" \
    --output text)

echo ""
echo "Launched: $INSTANCE_ID"
echo ""
echo "Expected runtime: ~10-15 minutes"
echo ""
echo "Commands:"
echo "  Check progress: $0 --status"
echo "  View log:       aws s3 cp s3://$S3_BUCKET/benchmark/benchmark.log - | tail"
echo "  Get results:    $0 --results"
echo "  SSH in:         $0 --ssh"
echo "  Kill:           $0 --kill"
