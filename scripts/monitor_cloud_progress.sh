#!/usr/bin/env bash
# Monitor cloud training progress and update cloud-progress.md
# Run this locally: ./scripts/monitor_cloud_progress.sh

set -euo pipefail

VM_IP="167.233.52.167"
LOG_FILE="/var/log/m5-train.log"
PROGRESS_FILE="cloud-progress.md"

# Check if VM is reachable
if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@$VM_IP "echo 'alive'" 2>/dev/null; then
    echo "$(date -Is): VM is not reachable or powered off"
    echo "$(date -Is): VM is not reachable or powered off" >> "$PROGRESS_FILE"
    exit 0
fi

# Get current status
STATUS=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@$VM_IP "cat /srv/M5/.train-complete 2>/dev/null || echo 'IN_PROGRESS'" 2>/dev/null)

# Get last log entries
LOG_TAIL=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@$VM_IP "tail -n 5 $LOG_FILE" 2>/dev/null)

# Get process info
PROC_INFO=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@$VM_IP "ps aux | grep -E 'm5|python' | grep -v grep | grep -v unattended | head -3" 2>/dev/null)

# Update progress file
cat >> "$PROGRESS_FILE" << EOF

### $(date -Is)
- Status: $STATUS
- Processes:
$PROC_INFO
- Log tail:
$LOG_TAIL

EOF

# Check if training completed
if [ "$STATUS" != "IN_PROGRESS" ]; then
    echo "$(date -Is): Training completed!"
    echo "$(date -Is): Training completed!" >> "$PROGRESS_FILE"

    # Check if VM is still running (should power off)
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@$VM_IP "echo 'alive'" 2>/dev/null; then
        echo "$(date -Is): WARNING: VM is still running after completion!"
        echo "$(date -Is): WARNING: VM is still running after completion!" >> "$PROGRESS_FILE"
    else
        echo "$(date -Is): VM powered off successfully"
        echo "$(date -Is): VM powered off successfully" >> "$PROGRESS_FILE"
    fi
fi
