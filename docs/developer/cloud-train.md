# M5 Cloud Training Guide

Launch a full M5 forecasting pipeline on cloud infrastructure (Hetzner or AWS) and mirror results locally for the Vue.js dashboard.

---

## Prerequisites

| Tool | Check | Install |
|------|-------|---------|
| `git` | `git --version` | `apt install git` / `brew install git` |
| `make` | `make --version` | `apt install make` / `brew install make` |
| `terraform` | `terraform version` | https://developer.hashicorp.com/terraform/install |
| `aws` CLI | `aws --version` | https://aws.amazon.com/cli/ |
| `ssh` key | `ls ~/.ssh/id_*.pub` | `ssh-keygen -t ed25519` |

## 1. Cloud Provider Setup (One-Time)

### Option A: Hetzner (Recommended — cheapest)

1. Create project at [console.hetzner.cloud](https://console.hetzner.cloud)
2. **API Token**: Project → Security → API Tokens → Generate (Read & Write)
3. **Object Storage**: Project → Object Storage → Create Bucket (note location: `fsn1`/`nbg1`/`hel1`)
4. **S3 Keys**: Object Storage → Access Keys → Generate (Read & Write)

```bash
export HCLOUD_TOKEN='your-api-token-here'
export AWS_ACCESS_KEY_ID='HCO-...'
export AWS_SECRET_ACCESS_KEY='...'
export TF_VAR_hcloud_token="$HCLOUD_TOKEN"
```

Seed config:
```bash
cd cloud/terraform/hetzner
cp terraform.tfvars.example terraform.tfvars
# Edit: artifact_uri, object_store_endpoint, aws_access_key_id, aws_secret_access_key
```

### Option B: AWS (More power — 128 GB+ RAM)

1. `aws configure` — enter your AWS access key / secret
2. Create S3 bucket:
```bash
bash scripts/create_s3_bucket.sh
```
3. Seed config:
```bash
cd cloud/terraform/aws
cp terraform.tfvars.example terraform.tfvars
# Edit: artifact_bucket_name, region, ssh_public_key
```

---

## 2. Configure Training Run

Edit `cloud/terraform/<provider>/terraform.tfvars` to control the pipeline:

```hcl
# ---- Pipeline knobs ----
train_server_type   = "ccx33"   # Hetzner: ccx33 (8c/32GB), ccx43 (16c/64GB)
                                # AWS: r7i.4xlarge (16c/128GB), r7i.8xlarge (32c/256GB)
last_n_days         = -1        # -1 = full history (~1941 days)
n_series            = -1        # -1 = all 30,490 series
cv_n_windows        = 3         # rolling-origin CV windows
run_stats_cv        = true      # Theta + AutoETS + SeasonalNaive
run_lgbm_cv         = true      # LightGBM global model
run_hier_cv         = true      # Hierarchical Theta + BU/MinT (needs 64GB+ RAM)
score_models        = "stats lgbm hier"
run_train           = true      # final fit on full data
push_processed      = true      # upload long.parquet to bucket
```

---

## 3. Launch Training

```bash
# Hetzner
cd ~/Git/GitHub/M5
make cloud-init PROVIDER=hetzner
make cloud-train-up PROVIDER=hetzner \
  TF_VARS="-var='last_n_days=-1' -var='n_series=-1' -var='cv_n_windows=3' \
           -var='run_stats_cv=true' -var='run_lgbm_cv=true' -var='run_hier_cv=true' \
           -var='score_models=stats lgbm hier' -var='run_train=true' \
           -var='push_processed=true'"

# AWS
cd ~/Git/GitHub/M5
export PROVIDER=aws
make cloud-init PROVIDER=aws
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='artifact_bucket_name=m5-artifacts-ricka-1781390300' \
           -var='train_instance_type=r7i.4xlarge' \
           -var='last_n_days=-1' -var='n_series=-1' -var='cv_n_windows=3' \
           -var='run_stats_cv=true' -var='run_lgbm_cv=true' -var='run_hier_cv=true' \
           -var='score_models=stats lgbm hier' -var='run_train=true' \
           -var='push_processed=true'"
```

**What happens:**
1. Terraform provisions VM + networking + IAM/S3
2. VM downloads M5 data (~250 MB)
3. `m5 prep` — builds `data/processed/long.parquet` (46M rows)
4. `m5 cv stats` — 3 statistical baselines
5. `m5 cv lgbm` — LightGBM global model
6. `m5 cv hier` — Hierarchical reconciliation (needs 64GB+)
7. `m5 score` — WRMSSE leaderboard
8. `m5 train` — final fit on full data
9. Push artifacts to S3 bucket
10. **VM auto-shutdown** (no extra billing)

---

## 4. Monitor Progress

```bash
# Get VM IP
make cloud-output PROVIDER=hetzner

# Watch training log in real-time
ssh root@<train_ipv4> "tail -f /var/log/m5-train.log"

# Check if complete
ssh root@<train_ipv4> "cat /srv/M5/.train-complete 2>/dev/null || echo 'Running'"

# Check artifacts
ssh root@<train_ipv4> "ls -la /srv/M5/artifacts/"
```

**Typical timeline:**
| Step | Time | RAM |
|------|------|-----|
| Download | 1 min | 1 GB |
| Prep | 2 min | 2 GB |
| CV Stats | 90 min | 5 GB |
| CV LGBM | 45 min | 15 GB |
| CV Hier | 30 min | 60 GB |
| Score | 15 min | 30 GB |
| Train | 10 min | 15 GB |
| **Total** | **~4 hours** | **peak 60 GB** |

---

## 5. Pull Results to Local Machine

After VM shuts down (or run while it's running):

```bash
# Pull entire run bundle
make cloud-pull-run PROVIDER=hetzner LOCAL_DIR=artifacts/cloud/latest

# Stage artifacts for local analysis
cp artifacts/cloud/latest/artifacts/cv_*.parquet artifacts/
cp artifacts/cloud/latest/forecasts/forecast_*.parquet forecasts/ 2>/dev/null || true
cp artifacts/cloud/latest/data/long.parquet data/processed/ 2>/dev/null || true
```

**What you get:**
```
artifacts/cloud/latest/
├── artifacts/
│   ├── cv_stats.parquet      # 3 statistical models
│   ├── cv_lgbm.parquet       # LightGBM
│   ├── cv_hier.parquet       # Hierarchical reconcilers
│   └── models/lgbm/<ts>/     # Trained serving artifact
├── reports/
│   ├── figures/              # WRMSSE plots
│   ├── metrics/              # CSV scores
│   ├── report.md             # Markdown summary
│   └── report.html           # HTML report
├── forecasts/
│   └── forecast_lgbm.parquet # 28-day future forecast
└── metadata/
    └── run.json              # Full run config
```

---

## 6. Vue.js Dashboard

```bash
cd frontend

# Export dashboard data from CV artifacts
npm run export:data

# Start dev server
npm run dev
```

Open the URL printed (e.g., `http://localhost:5173/`).

The dashboard shows:
- **Headline**: WRMSSE, RMSE, MAE, SMAPE per model
- **Hierarchy**: 12 M5 aggregation levels
- **Segments**: Breakdown by state, store, category, department
- **Horizon**: Per-day accuracy (h=1..28)
- **FVA**: Forecast Value Add over SeasonalNaive baseline

---

## 7. Tear Down

```bash
# Destroy VM + networking (bucket persists)
make cloud-down PROVIDER=hetzner

# Or keep bucket, destroy VM only
make cloud-down PROVIDER=hetzner
# Bucket data stays for future pulls
```

---

## 8. Quick Reference

### Common Commands

```bash
# Full pipeline with defaults
make cloud-train-up PROVIDER=hetzner

# Custom: full data, all models, 3 CV windows
make cloud-train-up PROVIDER=hetzner \
  TF_VARS="-var='last_n_days=-1' -var='n_series=-1' -var='cv_n_windows=3' \
           -var='run_stats_cv=true' -var='run_lgbm_cv=true' -var='run_hier_cv=true' \
           -var='score_models=stats lgbm hier' -var='run_train=true'"

# Fast iteration: 500 series, 200 days, 1 window
make cloud-train-up PROVIDER=hetzner \
  TF_VARS="-var='last_n_days=200' -var='n_series=500' -var='cv_n_windows=1' \
           -var='run_stats_cv=true' -var='run_lgbm_cv=true' -var='run_hier_cv=false' \
           -var='score_models=stats lgbm' -var='run_train=false'"

# Hierarchy-only on large AWS instance
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='train_instance_type=r7i.8xlarge' \
           -var='run_stats_cv=false' -var='run_lgbm_cv=false' \
           -var='run_hier_cv=true' -var='run_train=false'"

# Pull and view
make cloud-pull-run PROVIDER=hetzner LOCAL_DIR=artifacts/cloud/latest
cp artifacts/cloud/latest/artifacts/cv_*.parquet artifacts/
cd frontend && npm run export:data && npm run dev
```

### Environment Variables

```bash
# Hetzner
export HCLOUD_TOKEN='...'
export TF_VAR_hcloud_token="$HCLOUD_TOKEN"
export M5_OBJECT_STORE_ENDPOINT='https://fsn1.your-objectstorage.com'
export AWS_ACCESS_KEY_ID='...'
export AWS_SECRET_ACCESS_KEY='...'

# AWS
export AWS_PROFILE='default'
export TF_VAR_artifact_bucket_name='m5-artifacts-...'
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `terraform: command not found` | Install Terraform: `brew install terraform` |
| `No value for required variable` | Export token: `export TF_VAR_hcloud_token='...'` |
| `state lock` | `killall terraform` or `terraform force-unlock <id>` |
| `Server type not found` | Check available types in Hetzner console; use `ccx33` as fallback |
| `BucketAlreadyExists` | Use a globally unique bucket name (add random suffix) |
| `OOM killed` | Use larger instance: `ccx43` (64GB) or AWS `r7i.4xlarge` (128GB) |
| `cv hier OOM` | Skip hierarchy on small VMs, run separately on AWS |
| `No artifacts after pull` | VM may still be running; check `tail -f /var/log/m5-train.log` |
| `Port 5173 in use` | Vite auto-picks next port: `http://localhost:5174/` |

---

## 10. Cost Summary

| Provider | VM / Hour | Full Run (~4h) | Monthly (serve VM) |
|----------|-----------|----------------|--------------------|
| Hetzner ccx33 | €0.06 | ~€0.24 | — |
| Hetzner ccx43 | €0.12 | ~€0.48 | — |
| AWS r7i.4xlarge | $0.80 | ~$3.20 | — |
| AWS r7i.8xlarge | $1.60 | ~$6.40 | — |
| S3 Storage | — | ~$0.01 | ~$0.02/GB |

**Pro tip:** The VM auto-shutdowns after training. You only pay for runtime. S3 storage is pennies.

---

## 11. Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Local Dev      │     │   Cloud VM       │     │   Object Store   │
│   (Your Laptop)  │◄────┤   (ephemeral)    │────►│   (persistent) │
│                  │pull │   ccx33 / r7i    │push │   S3 / Hetzner   │
│  Vue dashboard   │     │   Ubuntu 24.04   │     │   bucket         │
│  artifacts/      │     │   m5 train.sh    │     │   cv_*.parquet   │
│  data/           │     │   cloud-init     │     │   reports/       │
│                  │     │                  │     │   forecasts/     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## 12. Next Steps

- **Evaluate**: Open `frontend/public/data/accuracy-dashboard.json` in the Vue app
- **Compare**: Run multiple models, see leaderboard in `reports/report.html`
- **Submit**: Use `forecasts/forecast_lgbm.parquet` for Kaggle submission
- **Iterate**: Adjust features, re-run CV, compare WRMSSE
- **Production**: Use `make cloud-serve-up` to deploy FastAPI serving VM

---

*For more details, see:*
- [`cloud/README.md`](cloud/README.md) — Full cloud provider setup
- [`docs/developer/ARCHITECTURE.md`](docs/developer/ARCHITECTURE.md) — Module map
- [`docs/developer/AGENTS.md`](docs/developer/AGENTS.md) — AI agent workflows
