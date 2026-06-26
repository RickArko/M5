# Cloud Training Guide

Run the full M5 pipeline (all 30,490 series, 3 CV windows, hierarchical reconciliation) on Hetzner or AWS. The VM auto-shutdowns after completion — you only pay for runtime (~€0.24–$3.20 per full run).

## Prerequisites

| Tool | Install |
|---|---|
| `terraform` | [terraform.io/downloads](https://developer.hashicorp.com/terraform/install) |
| `aws` CLI | `brew install awscli` / `apt install awscli` |
| SSH key | `ssh-keygen -t ed25519` |

## One-time setup

### Hetzner (cheapest)

1. Create project at [console.hetzner.cloud](https://console.hetzner.cloud)
2. API Token: Project → Security → API Tokens → Generate (Read & Write)
3. Object Storage: Project → Object Storage → Create Bucket + Access Keys

```bash
export HCLOUD_TOKEN='your-token'
export TF_VAR_hcloud_token="$HCLOUD_TOKEN"

cd cloud/terraform/hetzner
cp terraform.tfvars.example terraform.tfvars
# Edit: artifact_uri, object_store_endpoint, aws keys
```

### AWS

```bash
aws configure
bash scripts/create_s3_bucket.sh   # creates unique bucket, prints name

cd cloud/terraform/aws
cp terraform.tfvars.example terraform.tfvars
# Edit: artifact_bucket_name, region, ssh_public_key
```

## Launch training

```bash
# Hetzner
make cloud-init PROVIDER=hetzner
make cloud-train-up PROVIDER=hetzner \
  TF_VARS="-var='last_n_days=-1' -var='n_series=-1' -var='cv_n_windows=3' \
           -var='run_stats_cv=true' -var='run_lgbm_cv=true' -var='run_hier_cv=true'"

# AWS
make cloud-init PROVIDER=aws
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='artifact_bucket_name=m5-artifacts-...' \
           -var='train_instance_type=r7i.4xlarge' \
           -var='last_n_days=-1' -var='n_series=-1' -var='cv_n_windows=3' \
           -var='run_stats_cv=true' -var='run_lgbm_cv=true' -var='run_hier_cv=true'"

# Fast iteration (capped)
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='artifact_bucket_name=m5-artifacts-...' \
           -var='train_instance_type=c7i.2xlarge' \
           -var='last_n_days=200' -var='n_series=500' -var='cv_n_windows=1' \
           -var='run_hier_cv=false'"
```

### What happens

1. Terraform provisions VM + networking + IAM/S3
2. VM downloads M5 data, runs `m5 prep`, `m5 cv stats|lgbm|hier`, `m5 score`, `m5 train`
3. Pushes artifacts to object store
4. **VM auto-shutdowns** — no extra billing

### Instance sizing

| Provider | Type | vCPU | RAM | $/h | Use case |
|---|---|---|---|---|---|
| Hetzner | ccx33 | 8 | 32 GB | €0.06 | Stats + lgbm only |
| Hetzner | ccx43 | 16 | 64 GB | €0.12 | Full + hierarchy (minimum) |
| AWS | r7i.4xlarge | 16 | 128 GB | $0.80 | Full data + hierarchy |
| AWS | r7i.8xlarge | 32 | 256 GB | $1.60 | Faster hierarchy |

**Hierarchy needs 64 GB+.** ccx33 (32 GB) will OOM. Use ccx43+ or r7i.4xlarge.

## Monitor

```bash
# Get VM IP
make cloud-output PROVIDER=hetzner

# Watch log
ssh root@<ip> "tail -f /var/log/m5-train.log"

# Check completion
ssh root@<ip> "cat /srv/M5/.train-complete 2>/dev/null || echo 'Running'"

# Check for OOM
ssh root@<ip> "dmesg | grep -i oom"

# Use tmux for long-running processes on VM
tmux new -s m5-hier          # start session
# Ctrl+B, D to detach
tmux attach -t m5-hier       # reattach
```

### Timeline

| Step | Time | RAM |
|---|---|---|
| Download | 1 min | 1 GB |
| Prep | 2 min | 2 GB |
| CV Stats | 90 min | 5 GB |
| CV LGBM | 45 min | 15 GB |
| CV Hier | 30 min | 60 GB |
| Score | 15 min | 30 GB |
| **Total** | **~4 hours** | **peak 60 GB** |

## Pull results

```bash
# After VM shuts down
make cloud-pull-run PROVIDER=hetzner LOCAL_DIR=artifacts/cloud/latest

# Stage for local analysis
cp artifacts/cloud/latest/artifacts/cv_*.parquet artifacts/
cp artifacts/cloud/latest/forecasts/forecast_*.parquet forecasts/ 2>/dev/null || true
```

## Vue dashboard

```bash
cd frontend
npm run export:data   # reads CV artifacts
npm run dev           # http://localhost:5173
```

## Teardown

```bash
make cloud-down PROVIDER=hetzner   # destroy VM, keeps bucket
```

## Hierarchy CV

The M5 hierarchy has **12 levels** (42,840 total series at bottom). `m5 cv hier` fits Theta at every level and reconciles with BU/MintT:

```python
# Output columns: unique_id, ds, cutoff, y, Theta, BU, MinT_OLS, MinT_shrink
import pandas as pd
cv = pd.read_parquet("artifacts/cv_hier.parquet")
```

Run hierarchy separately to save cost:

```bash
# Phase 1: stats + lgbm on cheap instance
# Phase 2: hierarchy on larger instance
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='artifact_bucket_name=m5-artifacts-...' \
           -var='train_instance_type=r7i.4xlarge' \
           -var='run_stats_cv=false' -var='run_lgbm_cv=false' \
           -var='run_hier_cv=true' -var='run_train=false'"
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `terraform: command not found` | Install Terraform |
| `state lock` | `killall terraform` or `terraform force-unlock <id>` |
| `BucketAlreadyExists` | Use globally unique name (add random suffix) |
| OOM (`cv hier` dies) | Use 64 GB+ instance |
| SSH connection refused | VM still booting — wait 2-3 min |
| No artifacts after pull | VM still running — check `tail -f /var/log/m5-train.log` |

## Cost summary

| Run type | Instance | Cost |
|---|---|---|
| Full (Hetzner ccx43) | €0.12/h × 4h | ~€0.48 |
| Full (AWS r7i.4xlarge) | $0.80/h × 4h | ~$3.20 |
| Hierarchy only (AWS) | $0.80/h × 1h | ~$0.80 |
| Fast test (AWS c7i.2xlarge) | $0.36/h × 1h | ~$0.36 |

The VM auto-shutdowns — you only pay for runtime. S3 storage is ~$0.02/GB/month.
