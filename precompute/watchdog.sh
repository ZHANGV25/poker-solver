#!/bin/bash
# Watchdog: runs on a t3.micro instance, monitors the solver instance,
# and relaunches it from checkpoint if it dies (spot reclaim, crash, etc).
#
# Auto-restart with cron (recommended):
#   On the t3.micro watchdog instance, add a crontab entry so the watchdog
#   starts automatically on boot (e.g., after the watchdog instance reboots):
#
#     crontab -e
#     @reboot /home/ec2-user/watchdog.sh >> /var/log/watchdog.log 2>&1
#
#   This ensures the watchdog is always running, even if the t3.micro itself
#   is stopped/started. Combined with spot persistent + stop behavior on the
#   solver instance, the full pipeline is self-healing.
#
# Deploy:
#   1. Launch a t3.micro on-demand (< $0.01/hr, ~$2/month)
#   2. Copy this script + launch_blueprint_unified.sh to it
#   3. Run: nohup ./watchdog.sh &
#
# The watchdog checks every 5 minutes. If the solver instance is terminated
# or stopped, it launches a new one with --resume. Total overhead: ~$1.70
# for the entire 8-day run.
#
# Usage:
#   ./watchdog.sh                    # start monitoring
#   ./watchdog.sh --setup            # install on a new t3.micro
#
# Requires: aws cli configured, launch_blueprint_unified.sh in same directory

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-poker-blueprint-unified}"
INSTANCE_TYPE="${SOLVER_INSTANCE_TYPE:-c5.metal}"
CHECK_INTERVAL=300  # 5 minutes
MAX_RELAUNCHES=50   # safety limit (8 days / ~2hr avg spot lifetime = ~96 relaunches worst case)
LOG_FILE="/var/log/watchdog.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Setup mode ───────────────────────────────────────────────────────

if [[ "${1:-}" == "--setup" ]]; then
    echo "=== Watchdog Setup ==="
    echo "This should be run on a fresh t3.micro instance."
    echo "Installing dependencies..."
    sudo yum install -y aws-cli jq 2>/dev/null || sudo apt-get install -y awscli jq 2>/dev/null
    echo "Setup complete. Run: nohup ./watchdog.sh > /var/log/watchdog.log 2>&1 &"
    exit 0
fi

# ── Helper functions ─────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

get_solver_instance() {
    # Find the currently running solver instance
    aws ec2 describe-instances \
        --region "$REGION" \
        --filters "Name=tag:Project,Values=poker-solver-unified" \
                  "Name=instance-state-name,Values=running,pending" \
        --query "Reservations[].Instances[0].InstanceId" \
        --output text 2>/dev/null | head -1
}

get_solver_state() {
    local instance_id="$1"
    aws ec2 describe-instances \
        --region "$REGION" \
        --instance-ids "$instance_id" \
        --query "Reservations[0].Instances[0].State.Name" \
        --output text 2>/dev/null
}

check_progress() {
    # Check S3 for latest checkpoint metadata
    local meta_local="/tmp/watchdog_meta.json"
    aws s3 cp "s3://$S3_BUCKET/checkpoint_meta.json" "$meta_local" --quiet 2>/dev/null || return 1
    if [ -f "$meta_local" ]; then
        local iters=$(jq -r '.iterations // 0' "$meta_local" 2>/dev/null)
        local n_is=$(jq -r '.num_info_sets // 0' "$meta_local" 2>/dev/null)
        local hours=$(jq -r '.time_hours // 0' "$meta_local" 2>/dev/null)
        log "Progress: ${iters} iterations, ${n_is} info sets, ${hours}h compute"
        rm -f "$meta_local"
    fi
}

launch_solver() {
    # Launch a new solver instance with --resume flag
    local RESUME_FLAG="--resume"

    log "Launching new solver instance ($INSTANCE_TYPE, spot)..."

    # Get latest AMI
    local AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=al2023-ami-2023*-x86_64" \
                  "Name=state,Values=available" \
        --query "sort_by(Images, &CreationDate)[-1].ImageId" \
        --output text 2>/dev/null)

    if [ -z "$AMI_ID" ]; then
        log "ERROR: Could not find AMI"
        return 1
    fi

    # Upload latest code
    if [ -d "$SCRIPT_DIR/../src" ]; then
        aws s3 sync "$SCRIPT_DIR/../src" "s3://$S3_BUCKET/code/src/" \
            --quiet --exclude "*.o" --exclude "*.dll" 2>/dev/null
        aws s3 sync "$SCRIPT_DIR" "s3://$S3_BUCKET/code/precompute/" --quiet 2>/dev/null
        aws s3 sync "$SCRIPT_DIR/../python" "s3://$S3_BUCKET/code/python/" \
            --quiet --exclude "__pycache__/*" 2>/dev/null
    fi

    # Generate userdata with --resume
    local USERDATA=$(cat <<'INNEREOF'
#!/bin/bash
set -euxo pipefail
exec > /var/log/blueprint-unified.log 2>&1
echo "=== Solver starting (with resume) at $(date) ==="
yum install -y gcc gcc-c++ python3 python3-pip libgomp
WORKDIR=/tmp/poker-solver
mkdir -p $WORKDIR/build && cd $WORKDIR
aws s3 sync s3://BUCKET_PLACEHOLDER/code/ $WORKDIR/ --quiet
echo "Compiling..."
gcc -O3 -march=native -fPIC -shared -fopenmp -o build/mccfr_blueprint.so src/mccfr_blueprint.c -I src -lm -lpthread
gcc -O3 -march=native -fPIC -shared -o build/card_abstraction.so src/card_abstraction.c -I src -lm
echo "Compilation complete."
export OMP_STACKSIZE=64m
export OMP_NUM_THREADS=$(nproc)
python3 precompute/blueprint_worker_unified.py \
    --time-limit-hours 192 \
    --num-threads \$(nproc) \
    --hash-size 536870912 \
    --output-dir /tmp/blueprint_unified \
    --s3-bucket BUCKET_PLACEHOLDER \
    --checkpoint-interval 1000000 \
    --build-dir build \
    --resume
echo "=== Solver complete at $(date) ==="
aws s3 cp /var/log/blueprint-unified.log s3://BUCKET_PLACEHOLDER/logs/unified_$(date +%s).log --quiet
shutdown -h now
INNEREOF
)
    USERDATA="${USERDATA//BUCKET_PLACEHOLDER/$S3_BUCKET}"
    local USERDATA_B64=$(echo "$USERDATA" | base64 -w 0)

    # Try spot first
    local INSTANCE_ID=""
    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "${KEY_NAME:-poker-solver-key}" \
        --security-groups "${SECURITY_GROUP:-poker-solver-sg}" \
        --iam-instance-profile "Name=${PROFILE_NAME:-poker-solver-profile}" \
        --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
        --user-data "$USERDATA_B64" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=bp-unified-solver},{Key=Project,Value=poker-solver-unified}]" \
        --query "Instances[0].InstanceId" \
        --output text 2>/dev/null) || true

    if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
        log "Spot request failed, trying on-demand..."
        INSTANCE_ID=$(aws ec2 run-instances \
            --region "$REGION" \
            --image-id "$AMI_ID" \
            --instance-type "$INSTANCE_TYPE" \
            --key-name "${KEY_NAME:-poker-solver-key}" \
            --security-groups "${SECURITY_GROUP:-poker-solver-sg}" \
            --iam-instance-profile "Name=${PROFILE_NAME:-poker-solver-profile}" \
            --user-data "$USERDATA_B64" \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=bp-unified-solver},{Key=Project,Value=poker-solver-unified}]" \
            --query "Instances[0].InstanceId" \
            --output text 2>/dev/null) || true
    fi

    if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ]; then
        log "Launched: $INSTANCE_ID"
        echo "$INSTANCE_ID" > /tmp/watchdog_solver_id.txt
        return 0
    else
        log "ERROR: Failed to launch solver instance"
        return 1
    fi
}

# ── Main loop ────────────────────────────────────────────────────────

log "=== Watchdog started ==="
log "S3 bucket: $S3_BUCKET"
log "Solver instance type: $INSTANCE_TYPE"
log "Check interval: ${CHECK_INTERVAL}s"

relaunch_count=0

# Check if solver is already running
CURRENT_ID=$(get_solver_instance)
if [ -n "$CURRENT_ID" ] && [ "$CURRENT_ID" != "None" ]; then
    log "Found existing solver: $CURRENT_ID"
    echo "$CURRENT_ID" > /tmp/watchdog_solver_id.txt
else
    log "No solver running. Launching initial instance..."
    launch_solver
fi

while true; do
    sleep "$CHECK_INTERVAL"

    # Check if we've hit the safety limit
    if [ "$relaunch_count" -ge "$MAX_RELAUNCHES" ]; then
        log "Hit max relaunch limit ($MAX_RELAUNCHES). Stopping watchdog."
        break
    fi

    # Check solver status
    SOLVER_ID=$(get_solver_instance)

    if [ -z "$SOLVER_ID" ] || [ "$SOLVER_ID" = "None" ]; then
        # No running solver found
        log "Solver instance not running!"
        check_progress

        # Check if training is complete
        META_LOCAL="/tmp/watchdog_check_meta.json"
        aws s3 cp "s3://$S3_BUCKET/checkpoint_meta.json" "$META_LOCAL" --quiet 2>/dev/null || true
        if [ -f "$META_LOCAL" ]; then
            CHECKPOINT_LABEL=$(jq -r '.checkpoint // ""' "$META_LOCAL" 2>/dev/null)
            if [ "$CHECKPOINT_LABEL" = "final" ]; then
                log "Training complete (final checkpoint found). Stopping watchdog."
                rm -f "$META_LOCAL"
                break
            fi
            rm -f "$META_LOCAL"
        fi

        # Relaunch
        log "Relaunching solver (attempt $((relaunch_count + 1)))..."
        if launch_solver; then
            relaunch_count=$((relaunch_count + 1))
            log "Relaunch successful. Total relaunches: $relaunch_count"
        else
            log "Relaunch failed. Will retry in ${CHECK_INTERVAL}s."
        fi
    else
        # Solver is running — log progress
        check_progress
    fi
done

log "=== Watchdog stopped ==="
