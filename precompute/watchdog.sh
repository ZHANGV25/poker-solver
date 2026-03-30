#!/bin/bash
# Watchdog: runs on a t3.micro instance, monitors the solver instance,
# and relaunches it from checkpoint if it dies (spot reclaim, crash, OOM, etc).
#
# Auto-restart with cron (recommended):
#   On the t3.micro watchdog instance, add TWO crontab entries:
#
#     crontab -e
#     @reboot /home/ec2-user/watchdog.sh >> /var/log/watchdog4.log 2>&1
#     */5 * * * * pgrep -f 'watchdog.sh' > /dev/null || /home/ec2-user/watchdog.sh >> /var/log/watchdog4.log 2>&1
#
#   The @reboot entry starts watchdog on boot, and the */5 entry ensures
#   the watchdog itself gets restarted if it ever dies (self-healing).
#
# Deploy:
#   1. Launch a t3.micro on-demand (< $0.01/hr, ~$2/month)
#   2. Copy this script to it
#   3. Add cron entries above
#   4. Copy the SSH key: scp poker-solver-key.pem ec2-user@watchdog:/home/ec2-user/.ssh/poker-solver-key.pem
#   5. Run: nohup ./watchdog.sh >> /var/log/watchdog4.log 2>&1 &
#
# The watchdog checks every 5 minutes. If the solver instance is terminated
# or stopped, it launches a new one with --resume.
#
# Usage:
#   ./watchdog.sh                    # start monitoring
#   ./watchdog.sh --setup            # install on a new t3.micro

# NOTE: We do NOT use set -e. SSH failures and transient AWS CLI errors
# must not kill the watchdog. Each command handles its own errors.
set -uo pipefail

REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-poker-blueprint-unified}"
INSTANCE_TYPE="${SOLVER_INSTANCE_TYPE:-c5.metal}"
CHECK_INTERVAL=300  # 5 minutes
MAX_RELAUNCHES=50   # safety limit
LOG_FILE="/var/log/watchdog4.log"
SSH_KEY="/home/ec2-user/.ssh/poker-solver-key.pem"

# Staleness threshold: 24 checks × 5 min = 120 min.
# Must be longer than texture precompute (~93 min on first launch).
STALE_THRESHOLD=24

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Setup mode ───────────────────────────────────────────────────────

if [[ "${1:-}" == "--setup" ]]; then
    echo "=== Watchdog Setup ==="
    echo "This should be run on a fresh t3.micro instance."
    echo "Installing dependencies..."
    sudo yum install -y aws-cli jq 2>/dev/null || sudo apt-get install -y awscli jq 2>/dev/null
    echo ""
    echo "Setup complete. Next steps:"
    echo "  1. Copy SSH key:  scp poker-solver-key.pem ec2-user@this-host:/home/ec2-user/.ssh/poker-solver-key.pem"
    echo "  2. chmod 600 /home/ec2-user/.ssh/poker-solver-key.pem"
    echo "  3. Add cron entries:"
    echo "     crontab -e"
    echo "     @reboot /home/ec2-user/watchdog.sh >> /var/log/watchdog4.log 2>&1"
    echo "     */5 * * * * pgrep -f 'watchdog.sh' > /dev/null || /home/ec2-user/watchdog.sh >> /var/log/watchdog4.log 2>&1"
    echo "  4. Run: nohup ./watchdog.sh >> /var/log/watchdog4.log 2>&1 &"
    exit 0
fi

# ── Helper functions ─────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

get_solver_instance() {
    # Find the currently running solver instance (exclude the watchdog itself)
    local WATCHDOG_ID
    WATCHDOG_ID=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
    local ALL_IDS
    ALL_IDS=$(aws ec2 describe-instances \
        --region "$REGION" \
        --filters "Name=tag:Project,Values=poker-solver-unified" \
                  "Name=instance-state-name,Values=running,pending" \
        --query "Reservations[].Instances[].InstanceId" \
        --output text 2>/dev/null || echo "")
    # Return first ID that isn't the watchdog
    for id in $ALL_IDS; do
        if [ "$id" != "$WATCHDOG_ID" ] && [ -n "$id" ] && [ "$id" != "None" ]; then
            echo "$id"
            return
        fi
    done
}

check_progress() {
    # Check S3 for latest checkpoint metadata
    local meta_local="/tmp/watchdog_meta.json"
    if aws s3 cp "s3://$S3_BUCKET/checkpoint_meta.json" "$meta_local" --quiet 2>/dev/null; then
        if [ -f "$meta_local" ]; then
            local iters=$(jq -r '.iterations // 0' "$meta_local" 2>/dev/null || echo "?")
            local n_is=$(jq -r '.num_info_sets // 0' "$meta_local" 2>/dev/null || echo "?")
            local hours=$(jq -r '.time_hours // 0' "$meta_local" 2>/dev/null || echo "?")
            log "Progress: ${iters} iterations, ${n_is} info sets, ${hours}h compute"
            rm -f "$meta_local"
        fi
    else
        log "Could not fetch checkpoint metadata from S3"
    fi
}

launch_solver() {
    # Launch a new solver instance with --resume flag
    log "Launching new solver instance ($INSTANCE_TYPE, spot)..."

    # Get latest AMI
    local AMI_ID
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=al2023-ami-2023*-x86_64" \
                  "Name=state,Values=available" \
        --query "sort_by(Images, &CreationDate)[-1].ImageId" \
        --output text 2>/dev/null || echo "")

    if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
        log "ERROR: Could not find AMI"
        return 1
    fi

    # Upload latest code from S3 code/ prefix (already uploaded by launch script)
    # The watchdog doesn't have local source, so it relies on code already in S3.

    # Generate userdata with --resume
    local USERDATA
    USERDATA=$(cat <<'INNEREOF'
#!/bin/bash
set -euxo pipefail
exec > /var/log/blueprint-unified.log 2>&1
echo "=== Solver starting (with resume) at $(date) ==="
yum install -y gcc gcc-c++ python3 python3-pip libgomp
WORKDIR=/opt/poker-solver
mkdir -p $WORKDIR/build /opt/blueprint_unified && cd $WORKDIR
aws s3 sync s3://BUCKET_PLACEHOLDER/code/ $WORKDIR/ --quiet
echo "Compiling..."
gcc -O3 -march=native -fPIC -shared -fopenmp -o build/mccfr_blueprint.so src/mccfr_blueprint.c src/card_abstraction.c -I src -lm -lpthread
echo "Compiled at $(date)"
export OMP_STACKSIZE=64m
export OMP_NUM_THREADS=$(nproc)
python3 -u precompute/blueprint_worker_unified.py \
    --time-limit-hours 192 \
    --num-threads $(nproc) \
    --hash-size 1342177280 \
    --output-dir /opt/blueprint_unified \
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
    local USERDATA_B64
    USERDATA_B64=$(echo "$USERDATA" | base64 | tr -d '\n')

    # Try spot first
    local INSTANCE_ID=""
    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "${KEY_NAME:-poker-solver-key}" \
        --security-groups "${SECURITY_GROUP:-poker-solver-sg}" \
        --iam-instance-profile "Name=${PROFILE_NAME:-poker-solver-profile}" \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
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
            --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
            --user-data "$USERDATA_B64" \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=bp-unified-solver},{Key=Project,Value=poker-solver-unified}]" \
            --query "Instances[0].InstanceId" \
            --output text 2>/dev/null) || true
    fi

    if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ]; then
        log "Launched: $INSTANCE_ID ($INSTANCE_TYPE)"
        echo "$INSTANCE_ID" > /tmp/watchdog_solver_id.txt
        return 0
    else
        log "ERROR: Failed to launch solver instance"
        return 1
    fi
}

try_ssh_restart() {
    # Attempt to restart the solver process on a running instance via SSH.
    # Returns 0 on success, 1 on failure.
    local solver_id="$1"

    local SOLVER_IP
    SOLVER_IP=$(aws ec2 describe-instances \
        --region "$REGION" \
        --instance-ids "$solver_id" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text 2>/dev/null || echo "")

    if [ -z "$SOLVER_IP" ] || [ "$SOLVER_IP" = "None" ]; then
        log "Could not get solver IP for SSH restart"
        return 1
    fi

    if [ ! -f "$SSH_KEY" ]; then
        log "ERROR: SSH key not found at $SSH_KEY — cannot restart via SSH"
        log "Terminating stale instance and launching fresh one instead..."
        aws ec2 terminate-instances --instance-ids "$solver_id" --region "$REGION" 2>/dev/null || true
        return 1
    fi

    log "SSH restart: connecting to $SOLVER_IP..."

    # Use systemd-run for reliable process detachment (survives SSH disconnect)
    local SSH_CMD="sudo systemd-run --unit=blueprint-solver --remain-after-exit bash -c '
cd /opt/poker-solver 2>/dev/null || {
    mkdir -p /opt/poker-solver/build /opt/blueprint_unified
    cd /opt/poker-solver
    aws s3 sync s3://$S3_BUCKET/code/ /opt/poker-solver/ --quiet
    gcc -O3 -march=native -fPIC -shared -fopenmp -o build/mccfr_blueprint.so src/mccfr_blueprint.c src/card_abstraction.c -I src -lm -lpthread
}
export OMP_STACKSIZE=64m OMP_NUM_THREADS=\$(nproc)
python3 -u precompute/blueprint_worker_unified.py \
    --time-limit-hours 192 --num-threads \$(nproc) \
    --hash-size 1342177280 --output-dir /opt/blueprint_unified \
    --s3-bucket $S3_BUCKET --checkpoint-interval 1000000 \
    --build-dir build --resume > /var/log/blueprint-unified.log 2>&1
'"

    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
           -i "$SSH_KEY" "ec2-user@$SOLVER_IP" "$SSH_CMD" 2>&1 | \
       while IFS= read -r line; do log "SSH: $line"; done; then
        log "SSH restart command sent successfully"
    else
        log "SSH restart failed (exit code $?)"
        return 1
    fi

    # Verify: wait 30s then check if checkpoint advances
    log "Waiting 30s to verify solver started..."
    sleep 30
    local PID_CHECK
    PID_CHECK=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
                    -i "$SSH_KEY" "ec2-user@$SOLVER_IP" \
                    "pgrep -f blueprint_worker_unified" 2>/dev/null || echo "")
    if [ -n "$PID_CHECK" ]; then
        log "Verified: solver process running (PID $PID_CHECK)"
        return 0
    else
        log "WARNING: solver process not found after SSH restart"
        return 1
    fi
}

# ── Main loop ────────────────────────────────────────────────────────

log "=== Watchdog started ==="
log "S3 bucket: $S3_BUCKET"
log "Solver instance type: $INSTANCE_TYPE"
log "Check interval: ${CHECK_INTERVAL}s"
log "Stale threshold: ${STALE_THRESHOLD} checks ($((STALE_THRESHOLD * CHECK_INTERVAL / 60)) min)"

relaunch_count=0

# Check if solver is already running
CURRENT_ID=$(get_solver_instance)
if [ -n "$CURRENT_ID" ] && [ "$CURRENT_ID" != "None" ]; then
    log "Found existing solver: $CURRENT_ID"
    echo "$CURRENT_ID" > /tmp/watchdog_solver_id.txt
else
    log "No solver running. Launching initial instance..."
    if ! launch_solver; then
        log "Initial launch failed. Will retry in ${CHECK_INTERVAL}s."
    fi
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
        if aws s3 cp "s3://$S3_BUCKET/checkpoint_meta.json" "$META_LOCAL" --quiet 2>/dev/null; then
            if [ -f "$META_LOCAL" ]; then
                CHECKPOINT_LABEL=$(jq -r '.checkpoint // ""' "$META_LOCAL" 2>/dev/null || echo "")
                if [ "$CHECKPOINT_LABEL" = "final" ]; then
                    log "Training complete (final checkpoint found). Stopping watchdog."
                    rm -f "$META_LOCAL"
                    break
                fi
                rm -f "$META_LOCAL"
            fi
        fi

        # Relaunch
        log "Relaunching solver (attempt $((relaunch_count + 1)))..."
        if launch_solver; then
            relaunch_count=$((relaunch_count + 1))
            log "Relaunch successful. Total relaunches: $relaunch_count"
            # Reset stale counters for the new instance
            echo 0 > /tmp/watchdog_stale_count.txt
            rm -f /tmp/watchdog_last_iters.txt
        else
            log "Relaunch failed. Will retry in ${CHECK_INTERVAL}s."
        fi
    else
        # Solver is running — log progress and check for staleness
        check_progress

        # Detect stale solver: if checkpoint hasn't advanced in STALE_THRESHOLD
        # checks, the solver process likely crashed but the instance is still running.
        META_LOCAL="/tmp/watchdog_check_meta.json"
        CURRENT_ITERS=0
        if aws s3 cp "s3://$S3_BUCKET/checkpoint_meta.json" "$META_LOCAL" --quiet 2>/dev/null; then
            if [ -f "$META_LOCAL" ]; then
                CURRENT_ITERS=$(jq -r '.iterations // 0' "$META_LOCAL" 2>/dev/null || echo 0)
                rm -f "$META_LOCAL"
            fi
        fi

        LAST_ITERS=0
        LAST_CHECK=0
        if [ -f /tmp/watchdog_last_iters.txt ]; then
            LAST_ITERS=$(cat /tmp/watchdog_last_iters.txt 2>/dev/null || echo 0)
            LAST_CHECK=$(cat /tmp/watchdog_stale_count.txt 2>/dev/null || echo 0)
        fi

        if [ "$CURRENT_ITERS" = "$LAST_ITERS" ] && [ "$CURRENT_ITERS" != "0" ]; then
            LAST_CHECK=$((LAST_CHECK + 1))
            echo "$LAST_CHECK" > /tmp/watchdog_stale_count.txt
            if [ "$LAST_CHECK" -ge "$STALE_THRESHOLD" ]; then
                log "WARNING: Checkpoint stale for $((LAST_CHECK * CHECK_INTERVAL / 60))+ min (stuck at $CURRENT_ITERS iters)"
                log "Solver process likely crashed. Attempting SSH restart..."

                if try_ssh_restart "$SOLVER_ID"; then
                    echo 0 > /tmp/watchdog_stale_count.txt
                    log "SSH restart succeeded"
                else
                    log "SSH restart failed. Terminating instance and launching fresh..."
                    aws ec2 terminate-instances --instance-ids "$SOLVER_ID" --region "$REGION" 2>/dev/null || true
                    sleep 30  # Wait for termination to register
                    if launch_solver; then
                        relaunch_count=$((relaunch_count + 1))
                        log "Fresh relaunch successful. Total relaunches: $relaunch_count"
                    else
                        log "Fresh relaunch also failed. Will retry next cycle."
                    fi
                    echo 0 > /tmp/watchdog_stale_count.txt
                    rm -f /tmp/watchdog_last_iters.txt
                fi
            else
                log "Checkpoint unchanged ($LAST_CHECK/$STALE_THRESHOLD stale checks)"
            fi
        else
            echo 0 > /tmp/watchdog_stale_count.txt
        fi
        echo "$CURRENT_ITERS" > /tmp/watchdog_last_iters.txt
    fi
done

log "=== Watchdog stopped ==="
