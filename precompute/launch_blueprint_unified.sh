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

REGION="${AWS_REGION:-us-east-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c7a.metal-48xl}"  # 192 vCPU, 384 GB
KEY_NAME="${KEY_NAME:-poker-solver-key}"
SECURITY_GROUP="${SECURITY_GROUP:-poker-solver-sg}"
S3_BUCKET="${S3_BUCKET:-poker-blueprint-unified-v3}"  # v3: isolated from v2, new region us-east-2
PROFILE_NAME="poker-solver-profile"

HOURS=192           # 8 days (Pluribus)
ITER_TARGET=8000000000  # 8B iters (~66-74h on 192 threads at 30-35K iter/s, ~Pluribus core-hour eq + buffer)
HASH_SIZE=3500000000    # 3.5B slots (~51% load at projected 1.8B entries — v2 was 2B slots, hit 90% projected load and failed)
USE_SPOT=1              # 1 = spot one-time terminate-on-interruption, 0 = on-demand
DRY_RUN=0
STATUS_ONLY=0
DOWNLOAD_ONLY=0
AMI_ID=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

while [[ $# -gt 0 ]]; do
    case $1 in
        --hours)          HOURS="$2"; shift 2;;
        --iterations)     ITER_TARGET="$2"; shift 2;;
        --instance-type)  INSTANCE_TYPE="$2"; shift 2;;
        --hash-size)      HASH_SIZE="$2"; shift 2;;
        --key)            KEY_NAME="$2"; shift 2;;
        --bucket)         S3_BUCKET="$2"; shift 2;;
        --on-demand)      USE_SPOT=0; shift;;
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
echo "  Unified Pluribus Blueprint v3"
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
    c6a.metal)   OD_PRICE="4.90";;
    c7a.metal-48xl) OD_PRICE="5.20";;
    r5.24xlarge) OD_PRICE="6.05";;
    *)           OD_PRICE="5.00";;
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
aws s3 sync "$PROJECT_DIR/tests" "s3://$S3_BUCKET/code/tests/" \
    --quiet --exclude "*.o" --exclude "__pycache__/*"

# Upload the dashboard monitor script (lives in /tmp, not in the repo).
# The user-data script fetches this from S3 at boot and runs it in the
# background alongside the solver. See /tmp/hashprobe/monitor_v3.py.
if [ -f "/tmp/hashprobe/monitor_v3.py" ]; then
    aws s3 cp /tmp/hashprobe/monitor_v3.py "s3://$S3_BUCKET/code/monitor_v3.py" --quiet
    echo "Monitor script uploaded."
else
    echo "WARNING: /tmp/hashprobe/monitor_v3.py not found — dashboard monitor will not start"
fi

echo "Code uploaded."

USERDATA=$(cat <<USERDATA
#!/bin/bash
set -euxo pipefail
exec > /var/log/blueprint-unified.log 2>&1

echo "=== Unified Blueprint v3 starting at \$(date) ==="

# Bug α fix: install numactl for NUMA interleave (required by --interleave=all
# wrapper below). Without this the python solver runs on a single NUMA node and
# saturates one socket's memory bandwidth, costing ~30% throughput.
yum install -y gcc gcc-c++ python3 python3-pip libgomp numactl

WORKDIR=/opt/poker-solver
mkdir -p \$WORKDIR/build /opt/blueprint_unified && cd \$WORKDIR
aws s3 sync s3://$S3_BUCKET/code/ \$WORKDIR/ --quiet

echo "Compiling with -O2 -march=native..."
gcc -O2 -march=native -fno-strict-aliasing -fPIC -shared -fopenmp -o build/mccfr_blueprint.so src/mccfr_blueprint.c src/card_abstraction.c -I src -lm -lpthread
gcc -O2 -march=native -o build/extract_roots tests/extract_roots.c -lm
gcc -O2 -march=native -o build/dump_raw_regrets tests/dump_raw_regrets.c -lm
echo "Compilation complete."

# Bug α fix: enable transparent hugepages in madvise mode. The C code at
# mccfr_blueprint.c:1985 calls madvise(MADV_HUGEPAGE) on the hash table
# arrays, but this is a no-op unless the kernel allows it. Without THP, the
# hash table uses 4KB pages → ~40 million page table entries for a 2B slot
# table → catastrophic dTLB thrashing on every probe.
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true

# Download precomputed texture cache (saves ~40 min precomputation)
aws s3 cp s3://$S3_BUCKET/texture_cache.bin /tmp/texture_cache.bin --quiet 2>/dev/null || true
echo "Texture cache: \$(ls -lh /tmp/texture_cache.bin 2>/dev/null || echo 'not found, will precompute')"

# Show NUMA topology for diagnosis
numactl --hardware >> /var/log/blueprint-unified.log 2>&1 || true

# Start the dashboard status monitor in the background. It polls
# /var/log/blueprint-unified.log every 60s and writes status.json to the
# public poker-solver-dashboard S3 bucket. Monitor logs go to
# /var/log/monitor.log. Dies with the instance.
aws s3 cp s3://$S3_BUCKET/code/monitor_v3.py /opt/monitor_v3.py --quiet 2>/dev/null || true
if [ -f /opt/monitor_v3.py ]; then
    chmod +x /opt/monitor_v3.py
    nohup python3 /opt/monitor_v3.py > /var/log/monitor.log 2>&1 &
    echo "Dashboard monitor started (PID \$!)"
else
    echo "WARNING: monitor_v3.py not found on S3, dashboard will not update"
fi

# Bug α fix: pin threads to specific cores via OMP_PROC_BIND=spread +
# OMP_PLACES=cores. Without this, the OS bounces threads between cores at
# every barrier, losing L1/L2 cache state and costing ~15% throughput.
# OMP_WAIT_POLICY=ACTIVE makes threads busy-wait at barriers instead of
# sleeping (low-latency sync between batches, no futex syscalls).
export OMP_STACKSIZE=64m
export OMP_NUM_THREADS=\$(nproc)
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_WAIT_POLICY=ACTIVE

echo "Starting unified solve: \$(nproc) threads, $ITER_TARGET iterations..."
# Bug α fix: numactl --interleave=all distributes the hash table across all
# NUMA nodes so that all 4 sockets contribute memory bandwidth instead of
# bottlenecking one socket. This is the single largest perf win on
# c7a.metal-48xl.
numactl --interleave=all python3 -u precompute/blueprint_worker_unified.py \
    --iterations $ITER_TARGET \
    --num-threads \$(nproc) \
    --hash-size $HASH_SIZE \
    --output-dir /opt/blueprint_unified \
    --s3-bucket $S3_BUCKET \
    --build-dir build

echo "=== Complete at \$(date) ==="
aws s3 cp /var/log/blueprint-unified.log s3://$S3_BUCKET/logs/unified.log --quiet
shutdown -h now
USERDATA
)

USERDATA_B64=$(echo "$USERDATA" | base64 | tr -d '\n')

echo "Launching instance..."

# Bug ζ fix: spot one-time terminate. ~70% cost savings vs on-demand.
# Pass --on-demand to revert to the previous behavior.
SPOT_ARGS=""
if [ "$USE_SPOT" = "1" ]; then
    SPOT_ARGS='--instance-market-options {"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}'
fi

INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-groups "$SECURITY_GROUP" \
    --iam-instance-profile "Name=$PROFILE_NAME" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    $SPOT_ARGS \
    --user-data "$USERDATA_B64" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=bp-unified-v3},{Key=Project,Value=poker-solver-unified-v3}]" \
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
