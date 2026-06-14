# AWS Cloud Training — M5 Forecasting

Complete guide for running M5 training on AWS EC2 with S3 artifact storage.

---

## Prerequisites

| Tool | Check | Install |
|------|-------|---------|
| `git` | `git --version` | `apt install git` / `brew install git` |
| `make` | `make --version` | `apt install make` / `brew install make` |
| `terraform` | `terraform version` | https://developer.hashicorp.com/terraform/install |
| `aws` CLI | `aws --version` | `aws --version` or `brew install awscli` |
| `ssh` key | `ls ~/.ssh/id_*.pub` | `ssh-keygen -t ed25519` |

**Verify AWS credentials:**
```bash
aws sts get-caller-identity
# Expected: {"Account": "123456789012", "Arn": "arn:aws:iam::..."}
```

---

## 1. One-Time Setup

### 1.1 Create S3 Bucket

```bash
# Use the helper script
bash scripts/create_s3_bucket.sh

# Or manually with a unique name
aws s3 mb s3://m5-artifacts-$(whoami)-$(date +%s) --region us-east-1
```

**Note:** The bucket name must be globally unique. The script generates `m5-artifacts-<username>-<timestamp>`.

### 1.2 Seed Terraform Config

```bash
cd cloud/terraform/aws
cp terraform.tfvars.example terraform.tfvars
```

**Edit `terraform.tfvars`:**
```hcl
region               = "us-east-1"
ssh_public_key       = "ssh-ed25519 AAAA... your-comment"
artifact_bucket_name = "m5-artifacts-ricka-1781390300"  # from step 1.1

# Instance sizing for full M5 + hierarchy
train_instance_type  = "r7i.4xlarge"   # 16 vCPU / 128 GB / ~$0.80/h
# serve_instance_type = "t3.medium"    # 2 vCPU / 4 GB / ~$30/mo
```

**Available instance types:**
| Type | vCPU | RAM | Cost/hour | Use Case |
|------|------|-----|-----------|----------|
| `c7i.2xlarge` | 8 | 16 GB | ~$0.36 | Quick tests |
| `r7i.4xlarge` | 16 | 128 GB | ~$0.80 | Full M5 + hierarchy |
| `r7i.8xlarge` | 32 | 256 GB | ~$1.60 | Multiple experiments |
| `r7i.16xlarge` | 64 | 512 GB | ~$3.20 | Maximum speed |

---

## 2. Launch Training

### 2.1 Full Pipeline (All Models)

```bash
export PROVIDER=aws
export AWS_PROFILE=default  # or your profile

cd ~/Git/GitHub/M5

# Initialize Terraform
make cloud-init PROVIDER=aws

# Launch training VM
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='artifact_bucket_name=m5-artifacts-ricka-1781390300' \
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

### 2.2 Fast Iteration (500 series, 200 days)

```bash
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='artifact_bucket_name=m5-artifacts-ricka-1781390300' \
           -var='train_instance_type=c7i.2xlarge' \
           -var='last_n_days=200' \
           -var='n_series=500' \
           -var='cv_n_windows=1' \
           -var='run_stats_cv=true' \
           -var='run_lgbm_cv=true' \
           -var='run_hier_cv=false' \
           -var='score_models=stats lgbm' \
           -var='run_train=false'"
```

### 2.3 Hierarchy Only (Large Instance)

```bash
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='artifact_bucket_name=m5-artifacts-ricka-1781390300' \
           -var='train_instance_type=r7i.8xlarge' \
           -var='run_stats_cv=false' \
           -var='run_lgbm_cv=false' \
           -var='run_hier_cv=true' \
           -var='score_models=hier' \
           -var='run_train=false'"
```

---

## 3. Monitor Progress

### 3.1 Get VM Info

```bash
make cloud-output PROVIDER=aws

# Example output:
# artifact_bucket = "m5-artifacts-ricka-1781390300"
# artifact_uri    = "s3://m5-artifacts-ricka-1781390300/m5/lgbm"
# ssh_train       = "ssh ubuntu@54.84.106.100"
# train_public_ip = "54.84.106.100"
```

### 3.2 Watch Training Log

```bash
# Real-time (best practice)
ssh ubuntu@<train_public_ip> "tail -f /var/log/m5-train.log"

# Or check periodically
ssh ubuntu@<train_public_ip> "tail -n 20 /var/log/m5-train.log"
```

### 3.3 Check Completion

```bash
ssh ubuntu@<train_public_ip> "cat /srv/M5/.train-complete 2>/dev/null || echo 'Running'"

# Check artifacts
ssh ubuntu@<train_public_ip> "ls -la /srv/M5/artifacts/"

# Check process
ssh ubuntu@<train_public_ip> "ps aux | grep -E 'm5|python' | grep -v grep | head -5"
```

### 3.4 Check for Errors

```bash
# Check for OOM or crashes
ssh ubuntu@<train_public_ip> "dmesg | grep -i 'oom\|killed' | tail -5"

# Check log for errors
ssh ubuntu@<train_public_ip> "grep -i 'error\|exception\|traceback' /var/log/m5-train.log | tail -10"
```

---

## 4. Best Practices

### 4.1 Use tmux for Long-Running Processes

```bash
# On the VM
ssh ubuntu@<train_public_ip>

# Install tmux
sudo apt-get update && sudo apt-get install -y tmux

# Start session
tmux new -s m5-train

# Run training
m5 cv hier --horizon 28 --n-windows 3

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t m5-train
# List: tmux ls
```

### 4.2 Use nohup for Background Processes

```bash
# Run without blocking SSH
nohup uv run m5 cv hier --horizon 28 --n-windows 3 > /var/log/m5-hier.log 2>&1 &

# Check later
tail -f /var/log/m5-hier.log
```

### 4.3 Monitor Every 15 Minutes

```bash
# Create a background monitor
while true; do
    ssh -o ConnectTimeout=5 ubuntu@<train_public_ip> \
        "cat /srv/M5/.train-complete 2>/dev/null || echo 'Running'" \
        >> cloud-progress.log 2>&1
    sleep 900  # 15 minutes
done &
```

### 4.4 Set CloudWatch Alarms (Optional)

```bash
# Create alarm for high CPU (training active)
aws cloudwatch put-metric-alarm \
    --alarm-name m5-training-active \
    --alarm-description "M5 training is running" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --evaluation-periods 2 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=InstanceId,Value=<instance-id>

# Create alarm for idle (training done)
aws cloudwatch put-metric-alarm \
    --alarm-name m5-training-idle \
    --alarm-description "M5 training is idle" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --evaluation-periods 2 \
    --threshold 10 \
    --comparison-operator LessThanThreshold \
    --dimensions Name=InstanceId,Value=<instance-id>
```

---

## 5. Pull Results

### 5.1 After VM Auto-Shuts Down

```bash
# Pull entire run bundle
make cloud-pull-run PROVIDER=aws LOCAL_DIR=artifacts/cloud/latest

# Stage for local analysis
cp artifacts/cloud/latest/artifacts/cv_*.parquet artifacts/
cp artifacts/cloud/latest/forecasts/forecast_*.parquet forecasts/ 2>/dev/null || true
cp artifacts/cloud/latest/data/long.parquet data/processed/ 2>/dev/null || true
```

### 5.2 While VM is Running

```bash
# Pull individual artifacts
scp ubuntu@<train_public_ip>:/srv/M5/artifacts/cv_stats.parquet artifacts/
scp ubuntu@<train_public_ip>:/srv/M5/artifacts/cv_lgbm.parquet artifacts/
scp ubuntu@<train_public_ip>:/srv/M5/artifacts/cv_hier.parquet artifacts/ 2>/dev/null || true
```

### 5.3 From S3 Directly

```bash
# Sync from S3
aws s3 sync s3://m5-artifacts-ricka-1781390300/m5/lgbm/runs/latest/ artifacts/cloud/latest/

# Or specific files
aws s3 ls s3://m5-artifacts-ricka-1781390300/m5/lgbm/runs/latest/artifacts/
aws s3 cp s3://m5-artifacts-ricka-1781390300/m5/lgbm/runs/latest/artifacts/cv_stats.parquet artifacts/
```

---

## 6. Vue.js Dashboard

```bash
cd frontend

# Export dashboard data
npm run export:data

# Start dev server
npm run dev

# Open the URL printed (e.g., http://localhost:5173/)
```

---

## 7. Tear Down

### 7.1 Destroy VM (Keep Bucket)

```bash
make cloud-down PROVIDER=aws

# Or manually
cd cloud/terraform/aws
terraform destroy -auto-approve
```

### 7.2 Clean Everything (Bucket + VM)

```bash
# Destroy VM
make cloud-down PROVIDER=aws

# Delete bucket (WARNING: irreversible)
aws s3 rm s3://m5-artifacts-ricka-1781390300 --recursive
aws s3api delete-bucket --bucket m5-artifacts-ricka-1781390300
```

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `terraform init` fails | No AWS credentials | `aws configure` or `export AWS_PROFILE=...` |
| `state lock` | Previous terraform crashed | `killall terraform` or `terraform force-unlock <id>` |
| `BucketAlreadyExists` | Bucket name not unique | Use script: `bash scripts/create_s3_bucket.sh` |
| `InvalidParameterValue` | Unicode in security group description | Fixed in main.tf — pull latest |
| `Server type not found` | Wrong region or instance type | Check `aws ec2 describe-instance-types` |
| `OOM killed` | Hierarchy needs 64GB+ RAM | Use `r7i.4xlarge` (128GB) or larger |
| `hier_cv silent death` | Memory spike during reconciliation | Use `tmux` or `nohup`, monitor with `dmesg` |
| `SSH connection refused` | VM still booting | Wait 2-3 minutes after `terraform apply` |
| `Port 5173 in use` | Vite auto-picks next port | Check output for actual port (e.g., 5174) |
| `No artifacts after pull` | VM still running | Check `tail -f /var/log/m5-train.log` |

---

## 9. Cost Management

### 9.1 Estimated Costs

| Phase | Instance | Duration | Cost |
|-------|----------|----------|------|
| Full run (all models) | r7i.4xlarge | ~4h | ~$3.20 |
| Full run (all models) | r7i.8xlarge | ~2h | ~$3.20 |
| Quick test | c7i.2xlarge | ~1h | ~$0.36 |
| Hierarchy only | r7i.4xlarge | ~1h | ~$0.80 |
| S3 storage | — | 1 month | ~$0.02/GB |

### 9.2 Stop Billing Immediately

```bash
# Destroy VM (stops EC2 billing)
make cloud-down PROVIDER=aws

# VM auto-shutdowns after training if configured
# But if it fails, you must manually destroy
```

### 9.3 Set Billing Alerts

```bash
# Create AWS budget alert
aws budgets create-budget \
    --account-id $(aws sts get-caller-identity --query Account --output text) \
    --budget file://budget.json \
    --notifications-with-subscribers file://notifications.json
```

---

## 10. Quick Reference

### 10.1 Common Commands

```bash
# Full pipeline
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='artifact_bucket_name=...' \
           -var='train_instance_type=r7i.4xlarge' \
           -var='run_stats_cv=true' \
           -var='run_lgbm_cv=true' \
           -var='run_hier_cv=true' \
           -var='run_train=true'"

# Monitor
ssh ubuntu@<ip> "tail -f /var/log/m5-train.log"

# Pull
make cloud-pull-run PROVIDER=aws LOCAL_DIR=artifacts/cloud/latest

# Destroy
make cloud-down PROVIDER=aws
```

### 10.2 Environment Variables

```bash
# Required
export AWS_PROFILE=default
export AWS_REGION=us-east-1

# Optional
export TF_VAR_artifact_bucket_name="m5-artifacts-..."
export TF_VAR_train_instance_type="r7i.4xlarge"
```

### 10.3 File Locations

| Path | Purpose |
|------|---------|
| `cloud/terraform/aws/terraform.tfvars` | AWS config |
| `cloud/terraform/aws/main.tf` | EC2 + S3 + IAM |
| `scripts/create_s3_bucket.sh` | Bucket creation helper |
| `cloud-progress.md` | Progress tracking |
| `aws-instructions.md` | This guide |

---

## 11. Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Local Dev      │     │   AWS EC2        │     │   AWS S3         │
│   (Your Laptop)  │◄────┤   (ephemeral)    │────►│   (persistent)   │
│                  │pull │   r7i.4xlarge    │push │   bucket         │
│  Vue dashboard   │     │   Ubuntu 24.04   │     │   cv_*.parquet   │
│  artifacts/      │     │   m5 train.sh    │     │   reports/       │
│  data/           │     │   cloud-init     │     │   forecasts/     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## 12. Next Steps

1. **Evaluate**: Open `frontend/public/data/accuracy-dashboard.json` in Vue app
2. **Compare**: Run multiple models, see leaderboard in `reports/report.html`
3. **Submit**: Use `forecasts/forecast_lgbm.parquet` for Kaggle submission
4. **Iterate**: Adjust features, re-run CV, compare WRMSSE
5. **Production**: Use `make cloud-serve-up` to deploy FastAPI serving VM

---

*For more details, see:*
- [`cloud/README.md`](cloud/README.md) — Full cloud provider setup
- [`cloud-train.md`](cloud-train.md) — General cloud training guide
- [`docs/developer/ARCHITECTURE.md`](docs/developer/ARCHITECTURE.md) — Module map
- [`docs/developer/AGENTS.md`](docs/developer/AGENTS.md) — AI agent workflows
