# Cloud Pipeline for Heavy Forecasting Runs

## Goal

Run memory-heavy M5 workloads on ephemeral cloud VMs, persist every useful
output to object storage, and mirror selected outputs back into local
gitignored directories for analysis.

The workflow should support:

- full or capped `prep`
- `stats`, `lgbm`, and `hier` CV
- recipe-driven experiments such as `configs/m5/hier_experiments.yaml`
- optional final `m5 train` serving artifact
- immutable timestamped cloud runs plus a stable `latest` alias
- local pull-back into `artifacts/cloud/latest`

## Storage Layout

Given a base URI such as `s3://bucket/m5/lgbm`, `gs://bucket/m5/lgbm`, or
`az://container/m5/lgbm`, each cloud run writes:

```text
<base-uri>/
  latest/                         # serving artifact, if m5 train ran
  <model-timestamp>/              # serving artifact, if m5 train ran
  reports/latest/                 # latest score report
  reports/<model-timestamp>/      # legacy report location
  cv/latest/                      # latest raw cv_*.parquet files
  cv/<model-timestamp>/           # legacy CV location
  runs/
    <run-id>/
      artifacts/
        cv_*.parquet
        models/lgbm/...           # if m5 train ran
      reports/
      forecasts/
      data/
        long.parquet              # only when M5_PUSH_PROCESSED=true
      metadata/
        run.json
    latest/
      ...                         # stable mirror of the newest run bundle
```

The legacy `latest/` serving artifact path remains unchanged so existing serve
VMs continue to work.

## Runtime Knobs

Terraform writes these values into `/etc/m5-cloud.env` on the train VM:

| Env var | Default | Purpose |
|---|---:|---|
| `M5_RUN_ID` | UTC timestamp | Immutable run id. |
| `M5_RUN_STATS_CV` | `true` | Run `m5 cv stats`. |
| `M5_RUN_LGBM_CV` | `true` | Run `m5 cv lgbm`. |
| `M5_RUN_HIER_CV` | `false` | Run `m5 cv hier`. |
| `M5_CV_RECIPE` | empty | Optional YAML passed to `m5 cv-recipe`. |
| `M5_CV_N_WINDOWS` | `3` | Rolling-origin CV windows. |
| `M5_SCORE_MODELS` | `stats lgbm` | Names passed to `m5 score --model ...`. |
| `M5_RUN_TRAIN` | `true` | Run final `m5 train` serving fit. |
| `M5_PUSH_PROCESSED` | `false` | Include `data/processed/long.parquet` in the run bundle. |

## Suggested Instance Sizes

Start small, then scale:

| Workload | RAM target | Example AWS |
|---|---:|---|
| capped hierarchy, 500-5,000 series | 32 GB | `r7i.xlarge` / `r7i.2xlarge` |
| full `prep` + regular CV | 32-64 GB | `r7i.2xlarge` |
| full `cv-hier` | 64-128 GB | `r7i.4xlarge` / `r7i.8xlarge` |
| expanded MinTrace/ERM experiments | 128 GB preferred | `r7i.8xlarge` |

Use high-memory families for hierarchy work. CPU helps, but RAM is the main
constraint.

## Example Runs

Full default stats+lgbm train/report:

```bash
make cloud-train-up PROVIDER=aws
```

Hierarchy-only CV, no final LightGBM serving fit:

```bash
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='train_instance_type=r7i.4xlarge' \
           -var='run_stats_cv=false' \
           -var='run_lgbm_cv=false' \
           -var='run_hier_cv=true' \
           -var='score_models=hier' \
           -var='run_train=false'"
```

Expanded hierarchy recipe:

```bash
make cloud-train-up PROVIDER=aws \
  TF_VARS="-var='train_instance_type=r7i.8xlarge' \
           -var='run_stats_cv=false' \
           -var='run_lgbm_cv=false' \
           -var='run_hier_cv=false' \
           -var='cv_recipe=configs/m5/hier_experiments.yaml' \
           -var='score_models=hier_experiments' \
           -var='run_train=false' \
           -var='push_processed=true'"
```

Pull the latest cloud run back locally:

```bash
make cloud-pull-run PROVIDER=aws LOCAL_DIR=artifacts/cloud/latest
```

For non-Terraform-managed object stores or ad hoc pulls:

```bash
bash cloud/scripts/pull_artifact.sh \
  s3://my-bucket/m5/lgbm/runs/latest \
  artifacts/cloud/latest
```

## Guardrails

- Keep train VMs ephemeral and shut them down after upload.
- Keep object storage private; the Terraform modules create private buckets
  where the provider supports it.
- Do not commit `.tfvars`, Terraform state, pulled artifacts, or processed data.
- Run full hierarchy only on high-memory instances.
