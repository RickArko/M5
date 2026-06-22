# Running Full Hierarchical CV on M5

Guide for running the complete hierarchical reconciliation pipeline on M5 data with 12 aggregation levels.

---

## Prerequisites

**Minimum RAM:** 64 GB (128 GB recommended for full 30,490 series)
**CPU:** 8+ cores (hierarchical reconciliation is CPU-bound)
**Time:** 2-4 hours depending on instance size

---

## Architecture

The M5 hierarchy has **12 levels**:

```
Level 0:  Total (1 series)
Level 1:  State (3 series: CA, TX, WI)
Level 2:  Category (3 series: FOODS, HOBBIES, HOUSEHOLD)
Level 3:  Store (10 series: CA_1, CA_2, ..., WI_3)
Level 4:  Dept (7 series: FOODS_1, FOODS_2, ..., HOUSEHOLD_2)
Level 5:  State × Category (9 series)
Level 6:  State × Dept (21 series)
Level 7:  Category × Store (30 series)
Level 8:  Dept × Store (70 series)
Level 9:  State × Category × Dept (63 series)
Level 10: Category × Dept × Store (210 series)
Level 11: Item × Store (30,490 series) [bottom level]
```

**Total series across all levels:** 42,840

---

## Instance Size Recommendations

| Provider | Instance | vCPU | RAM | Cost/Hour | CV Hier Time | Total Cost |
|----------|----------|------|-----|-----------|--------------|------------|
| AWS | r7i.4xlarge | 16 | 128 GB | $0.80 | ~2h | ~$1.60 |
| AWS | r7i.8xlarge | 32 | 256 GB | $1.60 | ~1h | ~$1.60 |
| Hetzner | ccx43 | 16 | 64 GB | €0.12 | ~3h | ~€0.36 |
| Hetzner | ccx53 | 32 | 128 GB | €0.24 | ~2h | ~€0.48 |

**Note:** The original ccx33 (32 GB) **will fail** on hierarchy. Use minimum 64 GB.

---

## Option 1: Run Hierarchy on Existing VM

If you already have a VM running with stats + lgbm complete:

```bash
# SSH into the VM
ssh ubuntu@<vm-ip>

# Switch to root
sudo su

# Navigate to project
cd /srv/M5

# Run hierarchy CV only
uv run m5 cv hier --horizon 28 --n-windows 3

# Check if it's running (in another terminal)
ps aux | grep hier

# If it fails silently, check for OOM
dmesg | grep -i "oom\|killed"
```

---

## Option 2: Run Full Pipeline with Hierarchy

### Step 1: Create VM with sufficient RAM

**AWS (recommended):**
```bash
export PROVIDER=aws
export AWS_PROFILE=default

cd ~/Git/GitHub/M5
make cloud-init PROVIDER=aws

make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='artifact_bucket_name=m5-artifacts-yourname' \
           -var='train_instance_type=r7i.4xlarge' \
           -var='last_n_days=-1' \
           -var='n_series=-1' \
           -var='cv_n_windows=3' \
           -var='run_stats_cv=true' \
           -var='run_lgbm_cv=true' \
           -var='run_hier_cv=true' \
           -var='score_models=stats lgbm hier' \
           -var='run_train=true' \
           -var='push_processed=true'"
```

**Hetzner:**
```bash
export HCLOUD_TOKEN='...'
export TF_VAR_hcloud_token="$HCLOUD_TOKEN"

cd ~/Git/GitHub/M5
make cloud-init PROVIDER=hetzner

# Edit terraform.tfvars first:
# train_server_type = "ccx43"  # 16 vCPU / 64 GB

make cloud-train-up PROVIDER=hetzner \
  TF_VARS="-var='last_n_days=-1' \
           -var='n_series=-1' \
           -var='cv_n_windows=3' \
           -var='run_stats_cv=true' \
           -var='run_lgbm_cv=true' \
           -var='run_hier_cv=true' \
           -var='score_models=stats lgbm hier' \
           -var='run_train=true' \
           -var='push_processed=true'"
```

### Step 2: Monitor with tmux (Critical)

```bash
# SSH into the VM
ssh ubuntu@<vm-ip>

# Install tmux
sudo apt-get update && sudo apt-get install -y tmux

# Start a persistent session
tmux new -s m5-hier

# The training will start automatically via cloud-init
# Or run manually:
cd /srv/M5
uv run m5 cv hier --horizon 28 --n-windows 3

# DETACH: Press Ctrl+B, then D
# The process continues running in background

# Reattach later:
# tmux attach -t m5-hier

# Check all sessions:
# tmux ls
```

### Step 3: Watch for completion

```bash
# Check if process is running
ssh ubuntu@<vm-ip> "ps aux | grep hier | grep -v grep"

# Check artifacts
ssh ubuntu@<vm-ip> "ls -la /srv/M5/artifacts/"

# Check for completion
ssh ubuntu@<vm-ip> "cat /srv/M5/.train-complete 2>/dev/null || echo 'Running'"

# Check logs
ssh ubuntu@<vm-ip> "tail -n 20 /var/log/m5-train.log"
```

### Step 4: Pull results

```bash
# After VM auto-shutdowns
make cloud-pull-run PROVIDER=aws LOCAL_DIR=artifacts/cloud/latest

# Or copy directly from VM
scp ubuntu@<vm-ip>:/srv/M5/artifacts/cv_hier.parquet artifacts/
```

---

## Option 3: Run Hierarchy Locally (If you have RAM)

```bash
# Ensure you have the processed data
ls data/processed/long.parquet

# Run hierarchy CV
uv run m5 cv hier --horizon 28 --n-windows 3

# Monitor RAM usage (in another terminal)
watch -n 5 free -h

# If OOM, try with fewer series
# Edit .env: M5_N_SERIES=10000
# Or use a smaller instance
```

---

## Understanding the Hierarchy Output

The `cv_hier.parquet` contains reconciled forecasts at the **bottom level** (item × store):

```python
import pandas as pd

cv_hier = pd.read_parquet("artifacts/cv_hier.parquet")
print(cv_hier.columns)
# Output: ['unique_id', 'ds', 'cutoff', 'y', 'Theta', 'BU', 'MinT_OLS', 'MinT_shrink']
```

**Models in hierarchy CV:**
- `Theta`: Base Theta forecast at each level
- `BU`: Bottom-Up reconciliation
- `MinT_OLS`: MinTrace with OLS
- `MinT_shrink`: MinTrace with shrinkage

**Reconciliation methods:**
- **Bottom-Up**: Sum bottom-level forecasts up
- **MinTrace**: Minimize trace of covariance matrix
- **OLS**: Ordinary least squares
- **Shrink**: Shrinkage estimator for covariance

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| Process dies silently | OOM (needs 64GB+) | Use r7i.4xlarge (128GB) or larger |
| `ValueError: number sections` | statsforecast n_jobs | Use `n_jobs=1` in `build_stats_forecaster` |
| `Killed process` | OOM killer | Check `dmesg`, upgrade instance |
| No `cv_hier.parquet` | Process crashed | Check logs, retry with tmux |
| Slow progress | 42,840 series | Normal — takes 2-4 hours |
| SSH disconnect | Network issue | Use `tmux` to persist session |

---

## Best Practices

1. **Always use tmux or nohup** — SSH disconnects kill processes
2. **Monitor RAM** — hierarchy peaks at 60-80% of 128GB
3. **Check dmesg** — for OOM kills: `dmesg | grep -i oom`
4. **Use larger instances** — hierarchy is the most expensive step
5. **Run hierarchy separately** — if you only need hierarchy, skip stats/lgbm
6. **Save artifacts** — S3 or local backup before destroying VM

---

## Cost-Optimized Strategy

```bash
# Phase 1: Run stats + lgbm on cheap instance (ccx33 / c7i.2xlarge)
# Phase 2: Destroy VM
# Phase 3: Pull artifacts
# Phase 4: Run hierarchy on larger instance (r7i.4xlarge)
# Phase 5: Merge all CVs and score

# This saves ~$5-10 by not running all models on the expensive instance
```

---

## Example: Full Run with Monitoring

```bash
#!/bin/bash
# run_hierarchy.sh

set -euo pipefail

VM_IP="54.84.106.100"
LOG="/var/log/m5-hier.log"

# Start hierarchy with monitoring
ssh ubuntu@$VM_IP "
    sudo su
    cd /srv/M5
    nohup uv run m5 cv hier --horizon 28 --n-windows 3 > $LOG 2>&1 &
    echo 'Hierarchy started, PID: '
    ps aux | grep hier | grep -v grep | awk '{print \$2}'
"

# Monitor every 15 minutes
while true; do
    sleep 900
    STATUS=$(ssh ubuntu@$VM_IP "ps aux | grep hier | grep -v grep | wc -l")
    if [ "$STATUS" -eq 0 ]; then
        echo "Hierarchy completed or crashed"
        ssh ubuntu@$VM_IP "tail -n 20 $LOG"
        break
    fi
    echo "$(date): Hierarchy still running..."
done
```

---

## Next Steps After Hierarchy

```bash
# Score all models
uv run m5 score --model stats --model lgbm --model hier

# Train final model
uv run m5 train

# Export dashboard
cd frontend
npm run export:data
npm run dev

# View results at http://localhost:5173
```

---

*For more details, see:*
- [`aws-instructions.md`](aws-instructions.md) — AWS setup guide
- [`cloud-train.md`](cloud-train.md) — General cloud training guide
- `src/m5/models/hierarchical.py` — Implementation details
