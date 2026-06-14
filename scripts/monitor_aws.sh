#!/usr/bin/env bash
# Monitor AWS training VM every 15 minutes and update cloud-progress.md
# Run from repo root: bash scripts/monitor_aws.sh

set -euo pipefail

VM_IP="54.84.106.100"
PROGRESS_FILE="cloud-progress.md"
INTERVAL=900  # 15 minutes

echo "==> Starting AWS VM monitor (checks every 15 minutes)"
echo "==> VM: $VM_IP"
echo "==> Progress file: $PROGRESS_FILE"
echo "==> Press Ctrl+C to stop"

while true; do
    TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M UTC")

    # Check if VM is reachable
    if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$VM_IP "echo 'alive'" >/dev/null 2>&1; then
        echo "$TIMESTAMP: VM is not reachable - may have powered off"
        cat >> "$PROGRESS_FILE" << EOF

### $TIMESTAMP
- **Status**: VM UNREACHABLE
- Note: VM may have powered off or network issue

EOF
        break
    fi

    # Get completion status
    COMPLETE=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$VM_IP "cat /srv/M5/.train-complete 2>/dev/null || echo 'Running'" 2>/dev/null)

    # Get process info
    PROC_INFO=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$VM_IP "ps aux | grep -E 'm5|python|cv' | grep -v grep | grep -v unattended | grep -v cloud-init | grep -v networkd | head -3" 2>/dev/null)

    # Get artifacts
    ARTIFACTS=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$VM_IP "ls -la /srv/M5/artifacts/ 2>/dev/null || echo 'No artifacts'" 2>/dev/null)

    # Get latest log
    LOG_TAIL=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$VM_IP "tail -n 5 /var/log/m5-train.log 2>/dev/null || tail -n 5 /var/log/m5-hier.log 2>/dev/null || echo 'No log'" 2>/dev/null)

    # Get memory usage
    MEM=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$VM_IP "free -h | grep Mem" 2>/dev/null)

    # Update progress file
    cat >> "$PROGRESS_FILE" << EOF

### $TIMESTAMP
- **Status**: $COMPLETE
- **Processes**:
$PROC_INFO
- **Memory**:
$MEM
- **Artifacts**:
$ARTIFACTS
- **Log tail**:
$LOG_TAIL

EOF

    echo "$TIMESTAMP: Status: $COMPLETE"

    # Check if complete
    if [ "$COMPLETE" != "Running" ] && [ "$COMPLETE" != "Not complete" ]; then
        echo "$TIMESTAMP: Training completed!"
        break
    fi

    # Wait for next check
    echo "$TIMESTAMP: Waiting 15 minutes..."
    sleep $INTERVAL
done

echo "==> Monitor finished"
