# API load test — runbook

Production-scalability sweep of the `m5 serve` FastAPI endpoint across GCP machine
tiers, producing a `$/MRPS × p95-latency` comparison. Plan doc:
[`docs/plans/api_loadtest.md`](plans/api_loadtest.md).

## Prereqs

1. **Trained artifact in GCS.** A previous `make cloud-train-up PROVIDER=gcp` run
   has populated `gs://<bucket>/m5/lgbm/latest/`. Verify:
   ```bash
   gcloud storage ls gs://m5-rickarko-2026-artifacts/m5/lgbm/latest/
   ```
2. **Terraform tfvars** populated for the GCP module — `cloud/terraform/gcp/terraform.tfvars`
   (project_id, bucket, ssh_key, IP allow-list). Tear-down state from the
   prior train run is fine; the sweep only touches the serve VM.
3. **`GOOGLE_APPLICATION_CREDENTIALS`** exported to the terraform SA key.
4. **Locust + pyyaml** installed via the `loadtest` dep group:
   ```bash
   uv sync --group loadtest
   ```
5. **`make loadtest-payload`** has built `loadtest/payloads/unique_ids.txt`
   from `artifacts/cv_lgbm.parquet`.

## Quick start

```bash
# (one-time)
make loadtest-payload                  # builds the unique_id corpus

# (per sweep)
make loadtest-sweep-plan               # dry-run — shows the plan + spend ceiling
make loadtest-sweep-gcp                # full 4-tier sweep
make loadtest-aggregate TS=20260510T060000Z   # build summary.md + figures
```

Single-tier driver if you just want one number:

```bash
make loadtest-tier-gcp TIER=cheap      # cheap | default | dedicated | cpu-heavy
```

## What gets produced

```
reports/loadtest/<UTC-timestamp>/
  cheap_stats.csv                       # locust aggregate per endpoint
  cheap_stats_history.csv               # locust time-series
  cheap_failures.csv                    # locust failure breakdown
  cheap_meta.json                       # tier config + realised wall_s/usd
  cheap.html                            # locust HTML summary
  default_stats.csv … etc.              # same set per tier
  summary.md                            # comparison table + interpretation
  figures/
    01_latency_vs_rps.png               # p50/p95/p99 by tier
    02_cost_per_mrps.png                # $/MRPS bar chart
    03_users_vs_p99.png                 # p99 vs concurrent users (saturation knee)
```

## Hardening notes

- **Auto-teardown**: every tier runs inside `try/finally`; the serve VM is destroyed
  via `terraform destroy -target=google_compute_instance.serve` even on
  KeyboardInterrupt / SIGTERM / locust crash / readiness timeout.
- **Cost guardrail**: `max_total_spend_usd` in `loadtest/tiers.yaml` is checked
  before each tier; sweep aborts if cumulative + next-tier-max would breach.
- **Single-flight**: rerunning with the same `<UTC-timestamp>` overwrites; concurrent
  sweeps each get their own timestamp dir so they don't clobber.
- **Readiness gate**: orchestrator polls `/readyz` (not `/healthz`) before measurement
  starts — model artifact must be loaded.
- **Realistic payloads**: corpus drawn from cv_lgbm.parquet output, not synthesised.
- **Auth path exercised**: if `M5_SERVE_API_KEY` is set, locust sends `X-API-Key`
  on every request.

## Interpreting `$/MRPS`

`$/MRPS` = realised tier wall-cost (USD) / (total requests served / 1,000,000).
Lower is better. **Watch out for `fail %` > 5**: when a tier saturates, RPS plateaus
and `$/MRPS` looks great even though the box is dropping requests. Re-run that tier
with `max_users` halved to find the sustainable RPS.

## Expected outcome (hypothesis)

- `cheap` (e2-small): looks great on $/MRPS until CPU credits exhaust → p99 spikes
- `default` (e2-medium): best for low-traffic apps (≤ 50 RPS)
- `dedicated` (n2-standard-2): cleanest p99, ~80–120 RPS
- `cpu-heavy` (n2-highcpu-4): hypothesised winner on $/MRPS — LGBM is CPU-bound

If `cpu-heavy` loses to `default` on $/MRPS, the sweep is doing its job: the
hypothesis was wrong and you can pick the cheaper tier with confidence.

## See also

- `loadtest/locustfile.py` — Locust scenarios + weights
- `loadtest/tiers.yaml` — tier definitions
- `loadtest/sweep.py` — orchestrator
- `loadtest/aggregate.py` — report renderer
- `cloud/cloud-init/serve.sh` — serve-VM bootstrap (Docker + artifact pull + uvicorn)
