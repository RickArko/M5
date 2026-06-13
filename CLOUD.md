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
