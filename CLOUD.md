 You already have a good foundation in cloud/: Terraform provisions train/serve VMs and cloud/cloud-init/train.sh pushes
  artifacts to object storage via s3://, az://, or gs://.

  For heavier workloads, I’d set it up as a cloud train/job runner, not a permanent notebook box.

  Recommended Shape

  1. Provision one large ephemeral train VM
      - AWS: r7i.4xlarge or r7i.8xlarge for hierarchy work.
      - GCP: n2-highmem-16 or n2-highmem-32.
      - Azure: E16s_v5 / E32s_v5.
      - Hetzner: pick the largest RAM-optimized dedicated instance available.

  2. Run capped or full pipeline on that VM
      - make prep
      - make cv-hier
      - uv run m5 cv-recipe configs/m5/hier_experiments.yaml
      - uv run m5 score ...

  3. Push run outputs to object storage
     Existing script already supports:
      - s3://bucket/prefix
      - gs://bucket/prefix
      - az://container/prefix

  4. Pull artifacts locally into gitignored paths
     Use:

     bash cloud/scripts/pull_artifact.sh s3://my-bucket/m5/runs/latest artifacts/cloud/latest

  The local targets should mirror to places already gitignored:

  artifacts/cloud/<run_id>/
  artifacts/cv_*.parquet
  reports/cloud/<run_id>/
  forecasts/cloud/<run_id>/
  data/processed/long.parquet   # optional, if you want cloud-built prep mirrored

  What I’d Add Next
  A small cloud job abstraction:

  M5_CLOUD_JOB="prep cv-hier score" make cloud-train-up PROVIDER=aws

  or more explicit:

  make cloud-run PROVIDER=aws \
    TRAIN_INSTANCE_TYPE=r7i.4xlarge \
    JOB="prep cv-recipe" \
    RECIPE=configs/m5/hier_experiments.yaml \
    ARTIFACT_URI=s3://my-m5-bucket/m5/hier

  Then update train.sh to support:

  M5_RUN_STATS_CV=false
  M5_RUN_LGBM_CV=false
  M5_RUN_HIER_CV=true
  M5_RUN_RECIPE=configs/m5/hier_experiments.yaml
  M5_PUSH_PROCESSED=true

  Object Store Layout
  Use timestamped immutable runs plus a latest alias:

  s3://my-m5-bucket/m5/
    runs/
      20260610T183000Z/
        artifacts/
          cv_hier.parquet
          cv_hier_experiments.parquet
        reports/
        forecasts/
        data/
          long.parquet
        metadata.json
      latest/
        ...

  This gives you reproducibility and easy local sync.

  Local Mirror
  After a cloud run:

  bash cloud/scripts/pull_artifact.sh \
    s3://my-m5-bucket/m5/runs/latest \
    artifacts/cloud/latest

  For AWS specifically, I’d start with:

  - r7i.4xlarge for full prep + cv-hier.
  - 200-500 GB gp3 disk.
  - S3 bucket created by Terraform.
  - Train VM shuts down on completion.

  I can wire this into the existing cloud/ scripts as a proper cloud-run workflow so hierarchy jobs, recipe jobs, artifact
  push, and local pull are first-class targets.


---

# Cloud Workflow

1. Prerequisites (one-time)
You need a cloud provider configured. Pick one:
Provider	Cost (train)	What to set
Hetzner (recommended)	~â¬0.04 / run	export HCLOUD_TOKEN='...'
AWS	~$0.21 / run	aws sts get-caller-identity
Azure	~$0.22 / run	az login
GCP	~$0.23 / run	gcloud auth application-default login
Then seed the provider-specific variables:
cd cloud/terraform/<provider>
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars: artifact_uri, bucket/keys, ssh_public_key, etc.
cd ../../..
make cloud-init PROVIDER=<provider>
2. Launch the full-data training run
This spins up only the train VM (ephemeral), runs all CVs + final fit on the full 1,941-day history for all 30,490 series, then powers itself off.
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
What happens on the VM:
1. m5 download â m5 prep --last-n-days -1 --n-series -1
2. m5 cv stats â Theta + AutoETS + SeasonalNaive
3. m5 cv lgbm â LightGBM global model
4. m5 cv hier â Hierarchical Theta + BU/MinT
5. m5 score â generates WRMSSE reports
6. m5 train â final serving artifact on full data
7. Pushes everything to object storage â poweroff
Note: The cloud-init script covers stats, lgbm, and hier natively. If you also want the segmented models (store, store_cat, store_dept) or TOTO, SSH into the VM after step 1â3 finish and run the extra CVs manually, or run them locally against the pulled data/processed/long.parquet.
3. Pull the run bundle locally
After the VM shuts down (~30â45 min for full data):
make cloud-pull-run PROVIDER=hetzner \
  LOCAL_DIR=artifacts/cloud/latest
This mirrors the entire run bundle into artifacts/cloud/latest/, including:
- artifacts/cloud/latest/artifacts/cv_*.parquet
- artifacts/cloud/latest/reports/
- artifacts/cloud/latest/forecasts/
- artifacts/cloud/latest/data/long.parquet (because push_processed=true)
4. Stage artifacts for local evaluation / Vue dashboard
The Vue dashboard (make fe-export) and local make score expect files directly in ./artifacts/ and ./forecasts/. Copy (or symlink) them from the pulled bundle:
# Copy CV artifacts and forecasts into the local tree
cp artifacts/cloud/latest/artifacts/cv_*.parquet artifacts/
cp artifacts/cloud/latest/forecasts/forecast_*.parquet forecasts/ 2>/dev/null || true

# Optional: copy the full processed data for local replay
cp artifacts/cloud/latest/data/long.parquet data/processed/long.parquet
5. Re-export the dashboard data and start the Vue app
make fe-export
This runs:
1. npm run export:data inside frontend/ â reads artifacts/cv_*.parquet and generates frontend/public/data/accuracy-dashboard.json
2. npm run dev â starts the Vite dev server

**The dashboard now shows the full-history, full-series scores.**


## Choosing a Platform

For the M5 forecasting stack (ephemeral training VM + object storage + optional serving VM), the hyperscalers differ mostly in cost and ecosystem lock-in. Here is the breakdown across your four criteria, and a final recommendation based on what you are optimizing for.
Comparison matrix
Criterion	Hetzner	AWS	GCP	Azure
Cost (train + 1-mo serve)	~â¬0.04 + â¬5	~$0.21 + $30	~$0.23 + $25	~$0.22 + $30
Reliability	99.9 % (good)	99.99 % (best SLA)	99.99 % (best SLA)	99.99 % (best SLA)
Ease of setup	API token + S3 keys (2 min)	IAM + instance profiles (10 min)	Service account + IAM (10 min)	Managed identity + RBAC (10 min)
Industry applicability	Niche (hosting)	Universal standard	Strong in ML/AI	Strong in enterprise / Microsoft shops
The honest verdict
If you want the cheapest, fastest experiment: use Hetzner.
The repo authors explicitly recommend it because a full 35-minute training run costs â¬0.04 (6â10Ã cheaper than the hyperscalers), and the setup is literally just a HCLOUD_TOKEN and an S3-compatible key pair. For a reproducible Kaggle solution, that is the rational choice.
If you want to maximize "industry applicability" (e.g., portfolio, resume, enterprise MLOps demonstration): use AWS or GCP.
- AWS is the safest default. It is the most common cloud in Fortune 500 job postings, and the repo already uses S3-compatible artifact transport (s3://) across all providers. If you can deploy on AWS, you can deploy anywhere.
- GCP is the strongest for ML/AI specifically. The repoâs .env already contains PROJECT_ID and BILLING_ACCOUNT placeholders, and the documentation mentions Cloud Run / API Gateway as the "real prod" path. If you are targeting data-science or ML-engineering roles, GCP signals that you know Vertex AI / BigQuery ecosystems.
Azure is the right choice only if you are already in a Microsoft-centric environment (Active Directory, PowerBI, .NET) or applying to roles that explicitly use Azure ML.
My recommendation
Your goal	Pick
Run the experiment as cheaply as possible	Hetzner
Best all-rounder (cost + reliability + industry standard)	AWS
Best for ML/AI career signaling	GCP
Best for enterprise / Microsoft roles	Azure
For this specific project, AWS hits the sweet spot: the Terraform module is mature, the cost is still negligible ($0.21 per training run), and it is the most transferable skill to list on a resume. The repo even supports AWS S3 as the default artifact transport for Hetzner, so you could use Hetzner compute with an AWS S3 bucket if you want to split the difference.
If you want me to generate the exact terraform.tfvars and make command for any of these providers, just tell me which one you are leaning toward.



## Cloud Run

1. Updated Terraform for Future Runs (ccx83)
File: cloud/terraform/hetzner/terraform.tfvars
Changed:
train_server_type = "ccx83"  # 24 vCPU / 96 GB RAM (~â¬0.18/h)
Next run will use:
24 vCPU / 96 GB RAM (vs current 8 vCPU / 32 GB)
~â¬0.18/hour (vs current ~â¬0.06/hour)
3Ã faster for parallel workloads (LGBM, hier)
2. Progress Monitoring
Created:
cloud-progress.md â live progress log
scripts/monitor_cloud_progress.sh â automated checker
Background monitor running:
PID: 52293
Checks every 10 minutes
Updates cloud-progress.md automatically
3. Current Status (06:15 UTC)
â Prep complete â 46.8M rows
ð CV Stats â running (41 min, 100% CPU)
â³ CV LGBM â pending
â³ CV Hier â pending
â³ Score + Train â pending
ETA: ~20-30 minutes remaining
4. Automatic Shutdown
Yes, the VM will auto-shutdown:
cloud/cloud-init/train.sh has M5_TRAIN_SHUTDOWN_ON_DONE=true
After artifacts are pushed to S3, VM powers off after 30 seconds
No manual intervention needed
5. Next Steps After Completion
When the VM shuts down, I'll:
Pull artifacts from S3:
make cloud-pull-run PROVIDER=hetzner LOCAL_DIR=artifacts/cloud/latest
Stage for Vue dashboard:
cp artifacts/cloud/latest/artifacts/cv_*.parquet artifacts/
cd frontend && npm run export:data
Verify Vue app displays all models:
cd frontend && npm run dev
6. Monitor Log
The monitor is checking every 10 minutes. You can view it:
cat cloud-progress.md
# or
tail -f /tmp/cloud_monitor.log



```bash
make cloud-down PROVIDER=hetzner
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


# AWS

```bash
export PROVIDER=aws
make cloud-init PROVIDER=aws
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='train_instance_type=r7i.4xlarge' \
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



Solution Options
Option 1: Run hierarchy on a larger cloud provider (Recommended)
Hetzner's ccx33 (32 GB) is too small for full M5 scoring + hierarchy. Use AWS/GCP instead:
# Switch to AWS with r7i.4xlarge (16 vCPU / 128 GB RAM) or r7i.8xlarge (32 vCPU / 256 GB)
export PROVIDER=aws
make cloud-init PROVIDER=aws
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='train_instance_type=r7i.4xlarge' \
           -var='last_n_days=-1' \
           -var='n_series=-1' \
           -var='cv_n_windows=3' \
           -var='run_stats_cv=true' \
           -var='run_lgbm_cv=true' \
           -var='run_hier_cv=true' \
           -var='score_models=stats lgbm hier' \
           -var='run_train=true' \
           -var='push_processed=true'"
AWS cost: ~$0.80/hour for r7i.4xlarge, ~$1.60/hour for r7i.8xlarge
Option 2: Skip scoring on VM, run it locally
Pull the CV artifacts and run scoring locally:
# Pull CV artifacts from VM
ssh root@167.233.83.66 "cat /srv/M5/artifacts/cv_*.parquet" > artifacts/

# Or if VM is dead, use the S3 bucket
make cloud-pull-run PROVIDER=hetzner LOCAL_DIR=artifacts/cloud/latest

# Run scoring locally
uv run m5 score --model stats --model lgbm
Option 3: Destroy current VM and restart without scoring
# Destroy OOM'd VM
make cloud-down PROVIDER=hetzner

# Recreate with scoring disabled
make cloud-train-up PROVIDER=hetzner \
  TF_VARS="-var='last_n_days=-1' \
           -var='n_series=-1' \
           -var='cv_n_windows=3' \
           -var='run_stats_cv=true' \
           -var='run_lgbm_cv=true' \
           -var='run_hier_cv=false' \
           -var='run_train=true' \
           -var='push_processed=true'"
Recommendation
Option 1 (AWS) is best for full hierarchy with 128+ GB RAM. The Hetzner ccx33 is designed for small-medium workloads, not full M5.
Which option do you want? If you want AWS, I need your AWS credentials. If you want Option 2/3, I can set it up now.



Now run the AWS training:
cd ~/Git/GitHub/M5
export PROVIDER=aws

# 1. Initialize AWS terraform
make cloud-init PROVIDER=aws

# 2. Launch training with full hierarchy
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='last_n_days=-1' \
           -var='n_series=-1' \
           -var='cv_n_windows=3' \
           -var='run_stats_cv=true' \
           -var='run_lgbm_cv=true' \
           -var='run_hier_cv=true' \
           -var='score_models=stats lgbm hier' \
           -var='run_train=true' \
           -var='push_processed=true'"
This will:
- Create r7i.4xlarge (128 GB RAM â enough for hierarchy)
- Run stats + lgbm + hier CV
- Score all models
- Train final model
- Push to s3://m5-artifacts-ricka-1781390300
Cost: ~$0.80/hour Ã ~1.5 hours = ~$1.20
Run the commands now?



The hierarchy CV failed silently after running for ~7 minutes (from 02:00 to 02:07). The process died without writing cv_hier.parquet.
Current Status:
- â cv_stats.parquet (22.6 MB) â WRMSSE: AutoETS 0.863, Theta 0.869, SeasonalNaive 1.116
- â cv_lgbm.parquet (23.5 MB) â WRMSSE: LGBM 0.833
- â cv_hier.parquet â Process died, no file created
- â³ Score + Train â Never ran (script stopped at hierarchy)
The VM has been idle for ~12 hours (since 02:07).
Options:
Option 1: Run score + train with existing results (skip hierarchy)
ssh ubuntu@54.84.106.100
cd /srv/M5
uv run m5 score --model stats --model lgbm
uv run m5 train
# Then push artifacts manually
Option 2: Restart hierarchy CV
ssh ubuntu@54.84.106.100
cd /srv/M5
uv run m5 cv hier --horizon 28 --n-windows 3
Option 3: Pull current artifacts and continue locally
# Pull from VM
scp ubuntu@54.84.106.100:/srv/M5/artifacts/cv_*.parquet artifacts/

# Run score locally
uv run m5 score --model stats --model lgbm

# Run hierarchy locally (if you have 64GB+ RAM)
uv run m5 cv hier --horizon 28 --n-windows 3
Recommendation
Option 1 is fastest â score and train with the existing stats + lgbm results, then pull the complete artifacts. The hierarchy can be run separately later if needed.
Want me to run the score + train on the VM now?


Yes, you have been paying for it. AWS EC2 instances are billed for every hour they are running, regardless of whether they're actually doing work.
Cost Summary
Detail	Value
Instance	r7i.4xlarge (16 vCPU / 128 GB RAM)
Started	Jun 13, 22:43 UTC
Current time	Jun 14, 14:08 UTC
Duration	~15.5 hours
Rate	~$0.80/hour
Estimated cost	~$12.40
What happened
1. The VM ran stats CV (~1h40m)
2. Ran LGBM CV (~37m)
3. Hierarchy CV failed silently (process died ~12 hours ago)
4. VM kept running â AWS doesn't auto-shutdown on failure
Stop it now
# Destroy the VM immediately
cd ~/Git/GitHub/M5
export PROVIDER=aws
make cloud-down PROVIDER=aws
Or directly:
cd ~/Git/GitHub/M5/cloud/terraform/aws
terraform destroy -auto-approve
What to do
1. Destroy the VM now (stop the billing)
2. Pull the existing artifacts (stats + lgbm) â these are valuable
3. Run hierarchy locally or on a smaller Hetzner instance
4. Run score + train locally
# 1. Destroy VM (stop billing)
make cloud-down PROVIDER=aws

# 2. Pull artifacts
scp ubuntu@54.84.106.100:/srv/M5/artifacts/cv_stats.parquet artifacts/
scp ubuntu@54.84.106.100:/srv/M5/artifacts/cv_lgbm.parquet artifacts/
scp ubuntu@54.84.106.100:/srv/M5/data/processed/long.parquet data/processed/ 2>/dev/null || true

# 3. Run hierarchy locally
uv run m5 cv hier --horizon 28 --n-windows 3

# 4. Score and train
uv run m5 score --model stats --model lgbm --model hier
uv run m5 train
Destroy the VM now to stop the billing?



The command is running â hierarchy CV takes 30-60 minutes and doesn't print much output initially.
Problem: You're running it in the foreground. If you disconnect SSH, it will die.
Fix â run it with nohup so it survives disconnection:
# In the current terminal, press Ctrl+C to cancel the running command
# Then run it properly with nohup:
nohup uv run m5 cv hier --horizon 28 --n-windows 3 > /var/log/m5-hier.log 2>&1 &
Or use tmux (better):
# Install tmux if not present
apt-get update && apt-get install -y tmux

# Start a new session
tmux new -s m5-hier

# Run the command
cd /srv/M5
uv run m5 cv hier --horizon 28 --n-windows 3

# Detach: press Ctrl+B then D
# Reattach later: tmux attach -t m5-hier
Check if it's running:
ps aux | grep -E "hier|m5" | grep -v grep
Check log:
tail -f /var/log/m5-hier.log
Want me to tell you how to set this up properly so you can disconnect and let it run?
