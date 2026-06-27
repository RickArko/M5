# API load-test sweep — production scalability by cost/performance tier

 _Status: plan, not implemented. Review before invoking the Agent Prompt at the bottom._

 ## Goal

 Quantify the M5 FastAPI service's serving cost and latency profile across a sweep of GCP machine
 tiers, producing one **`$/MRPS` × `p95 latency`** table + figure pack that lets us pick the
 right tier per traffic volume.

 Stack: existing `m5 serve` (uvicorn / FastAPI factory at `src/m5/serve/app.py`) + Locust against
 the trained `gs://m5-rickarko-2026-artifacts/m5/lgbm/latest/` artifact. No code changes to the
 serving path itself — the goal is observation, not optimisation.

 ## Approach (one cycle)

 ```
 for tier in tiers.yaml:
     1. terraform apply -var serve_machine_type=$tier  # ~60s
     2. wait for /healthz green (max 5min)
     3. warm: 30s of light traffic (hit /predict with sample unique_ids)
     4. measure: locust shape ramps users 1 → N over T seconds, holds, ramps down
     5. capture: locust stats CSV + p50/p95/p99 + RPS @ saturation
     6. terraform destroy -target=google_compute_instance.serve  # ~60s
 aggregate per-tier CSVs into reports/loadtest/<ts>/summary.md
 ```

 Each tier costs ≤ $0.05 wall-time (∼15min × $0.06–0.20/h). Full 4-tier sweep budget: $2.

 ## Tiers (proposed — final list lives in `loadtest/tiers.yaml`)

 | tier alias    | machine_type     | vCPU       | RAM | $/h    | $/mo (730h) | hypothesis                              |
 |---------------|------------------|------------|----:|-------:|------------:|------------------------------------------|
 | `cheap`       | e2-small         | 2 (shared) | 2 G | $0.018 | $13         | floor — under-provisioned, p99 spikes    |
 | `default`     | e2-medium        | 2 (shared) | 4 G | $0.034 | $25         | current cloud/README.md serve default    |
 | `dedicated`   | n2-standard-2    | 2          | 8 G | $0.097 | $71         | predictable latency, no noisy neighbour  |
 | `cpu-heavy`   | n2-highcpu-4     | 4          | 4 G | $0.143 | $104        | LGBM inference is CPU-bound; sweet spot? |

 Optional 5th tier `n2d-highcpu-4` (AMD) for a $/RPS sanity check vs `n2-highcpu-4`.

 ## Locust scenarios (in priority order)

 1. **`predict_single`** — 1 series × 28-day horizon. Highest weight (80%) — represents the
    common request shape. Random `unique_id` drawn from `artifacts/cv_lgbm.parquet`.
 2. **`predict_batch_10`** — 10 series in one POST. Weight 15%. Tests the batch path's
    amortisation.
 3. **`healthz`** — Weight 5%. Catches when the API is up but mlforecast is wedged on a
    bad request.

 No write paths exist yet, so all scenarios are read-only. No mutation safety needed.

 ## File-by-file specs

 ### `loadtest/locustfile.py` (new, ~120 lines)

 - `class M5User(HttpUser)` with `@task(8)` predict_single, `@task(2)` predict_batch_10,
   `@task(1)` healthz.
 - `host` defaults to `os.environ["M5_LOADTEST_HOST"]`; respects `M5_SERVE_API_KEY` →
   adds `X-API-Key` header per request.
 - Payload corpus loaded once at startup from `loadtest/payloads/unique_ids.txt`
   (precomputed in step 4 of the orchestrator from cv_lgbm.parquet).
 - Custom `LoadTestShape` ramps `users 1 → N` over `ramp_s`, holds for `hold_s`, ramps
   down `N → 0` over `cooldown_s`. Constants come from `tiers.yaml`'s per-tier section.
 - On_request hooks: log first-30 responses' latency to `cold_start.csv` (separates
   cold-start from steady-state).

 ### `loadtest/tiers.yaml` (new, ~50 lines)

 ```yaml
 sweep_id: 2026-05-loadtest-v1
 max_total_spend_usd: 2.00         # guardrail: orchestrator aborts if exceeded
 default_warm_s: 30
 default_hold_s: 180
 default_ramp_s: 60
 default_cooldown_s: 30

 tiers:
   - alias: cheap
     machine_type: e2-small
     hourly_usd: 0.018
     max_users: 50
   - alias: default
     machine_type: e2-medium
     hourly_usd: 0.034
     max_users: 100
   - alias: dedicated
     machine_type: n2-standard-2
     hourly_usd: 0.097
     max_users: 200
   - alias: cpu-heavy
     machine_type: n2-highcpu-4
     hourly_usd: 0.143
     max_users: 400
 ```

 ### `loadtest/run_tier_sweep.sh` (new, ~150 lines, `set -euo pipefail`)

 Orchestrator. For each tier:

 1. Read tier config from `tiers.yaml`.
 2. `terraform apply -auto-approve -var=create_train=false -var=create_serve=true
    -var=serve_machine_type=$MACHINE_TYPE`. Re-uses VPC/SA/firewall already in state.
 3. Poll `curl http://$IP:8000/healthz` until green (timeout 5 min).
 4. Warm: ./loadtest_warm.sh $IP $WARM_S
 5. Run locust headless: `locust -f loadtest/locustfile.py --headless -u $MAX_USERS
    -r $((MAX_USERS / RAMP_S)) -H http://$IP:8000 --csv reports/loadtest/$TS/$ALIAS
    --run-time ${HOLD_S}s --html reports/loadtest/$TS/$ALIAS.html`
 6. Capture wall-time start/end, compute realised cost = (end - start) × $/h.
 7. **`trap` ensures `terraform destroy -target=google_compute_instance.serve` runs
    even on failure** — the tier MUST be torn down.
 8. After all tiers: `python loadtest/aggregate.py reports/loadtest/$TS/`.

 Includes a max-spend guardrail: before each tier starts, sum realised spend so far
 vs `max_total_spend_usd`; abort if next tier would breach.

 ### `loadtest/aggregate.py` (new, ~200 lines)

 - Reads each `<alias>_stats.csv` + `<alias>_stats_history.csv` produced by locust.
 - Computes per-tier: median RPS at saturation, p50/p95/p99, error rate, cold-start
   latency (first 30 reqs).
 - Realised cost per tier = wall_seconds × $/s.
 - **`$/MRPS`** = realised_cost / (median_rps × duration_s × 1e-6).
 - Outputs:
   - `summary.md` — comparison table + interpretation paragraph
   - `figures/01_latency_vs_rps.png` — small-multiples per tier, p50/p95/p99 vs offered RPS
   - `figures/02_cost_per_mrps.png` — bar chart, lower-is-better
   - `figures/03_users_vs_p99.png` — finds the knee where each tier saturates

 ### `loadtest/loadtest_warm.sh` (new, ~30 lines)

 Hits `/healthz` and `/predict` (single-series) at 1 RPS for `$WARM_S` seconds.
 Discards results — purpose is to get JIT, page cache, and any model lazy-loads warm.

 ### `cloud/terraform/gcp/variables.tf` (modify, +6 lines)

 Already has `serve_machine_type`. Add a `serve_replicas` integer var (default 1) for
 future scale-out tiers; this plan doesn't use it but the var should exist so
 `tier_sweep.sh` doesn't have to be re-touched later.

 ### `Makefile.cloud` (modify, +12 lines)

 ```
 loadtest-tier:        ## one tier  (TIER=<alias>)
 	bash loadtest/run_tier_sweep.sh --tier=$(TIER)
 loadtest-sweep:       ## full sweep across loadtest/tiers.yaml
 	bash loadtest/run_tier_sweep.sh --all
 loadtest-report:      ## re-render summary from existing reports/loadtest/$(TS)/
 	uv run python loadtest/aggregate.py reports/loadtest/$(TS)
 ```

 ### `pyproject.toml` (modify, +3 lines)

 Add a `loadtest` dep group:

 ```toml
 [dependency-groups]
 loadtest = ["locust>=2.31", "pyyaml>=6.0", "matplotlib>=3.9"]
 ```

 ### `docs/api_loadtest.md` (new, ~100 lines)

 Runbook: prereqs (artifact in GCS, terraform tfvars set, M5_SERVE_API_KEY), how to run,
 how to read the comparison table, what each metric means, when to redo the sweep.

 ### `loadtest/payloads/unique_ids.txt` (new, generated, ~3000 lines)

 Generated step at orchestrator startup: `python -c "import pandas as pd;
 pd.read_parquet('artifacts/cv_lgbm.parquet')['unique_id'].drop_duplicates()
 .head(3000).to_csv('loadtest/payloads/unique_ids.txt', index=False, header=False)"`.
 Gitignored — it's a derived artifact.

 ### `.gitignore` (modify, +2 lines)

 `loadtest/payloads/`, `reports/loadtest/`.

 ## Hardening checklist

 - [ ] **Idempotent**: `tier_sweep.sh --tier=cheap` reruns cleanly; existing
   `reports/loadtest/<ts>/cheap_stats.csv` causes refusal unless `--force`.
 - [ ] **Auto-teardown on failure**: `trap` in `tier_sweep.sh` always destroys the
   serve VM, even on locust crash, ctrl-C, or terraform error.
 - [ ] **Cost guardrail**: `max_total_spend_usd` in tiers.yaml; orchestrator tracks
   realised spend and aborts before exceeding.
 - [ ] **Pre-flight checks**: artifact present in GCS, terraform state consistent,
   M5_SERVE_API_KEY set, gcloud auth valid. Fail fast with a clear message.
 - [ ] **Realistic payload**: unique_ids drawn from cv_lgbm output, not synthetic.
 - [ ] **Cold-start separated**: first-30 responses' latency in its own CSV; not
   averaged into steady-state numbers.
 - [ ] **Warm-up**: 30s warm before measurement.
 - [ ] **Reproducible**: locust `--seed` set, payload corpus checksum recorded in
   summary.md, terraform.tfvars committed (with secrets stripped).
 - [ ] **Observability**: per-tier locust HTML report saved alongside CSVs; uvicorn
   stdout streamed to `reports/loadtest/<ts>/<alias>_uvicorn.log` via `gcloud
   compute ssh ... -- tail -F` in background.
 - [ ] **No secret leakage**: M5_SERVE_API_KEY scrubbed from logs; SA key path
   referenced not embedded; `gitleaks` runs as a pre-commit hook for `loadtest/`.
 - [ ] **CI smoke**: `make loadtest-tier TIER=cheap DRY_RUN=1` runs in GH Actions
   without spinning a VM (uses a local `m5 serve` for shape validation).
 - [ ] **Single-flight**: lock file `reports/loadtest/.in_progress` prevents
   concurrent sweeps from clobbering each other's CSVs.
 - [ ] **Saturation detection**: locust shape extends users past saturation so the
   knee is visible in figures, not extrapolated.
 - [ ] **Graceful degradation captured**: when error rate > 5%, mark tier as
   saturated and stop ramping (records max sustained RPS instead of crashing).
 - [ ] **Report reproducibility**: `summary.md` header records git SHA, sweep_id,
   each tier's machine_type, locust version, uvicorn workers count.

 ## Open questions for review

 1. **Replica count**: do we test single-instance only (current default), or also
    `serve_replicas=2,4` per tier? The latter doubles the matrix but maps better to
    real production deploys. Recommend single-instance for v1; add a
    `loadtest-replicas` Makefile target later.
 2. **API key**: serve currently supports `M5_SERVE_API_KEY` (header `X-API-Key`).
    Run with auth on or off? Recommend on — exercises the auth hot path which is
    typically a 5–10% latency drag.
 3. **Workers per VM**: uvicorn workers default = `min(2 × vCPU, 4)`. Constant or
    swept? Recommend constant `workers=2` across tiers — keeps the variable count
    low so the per-vCPU comparison stays clean.
 4. **Tear-down between tiers vs reuse**: the plan tears down between tiers (clean
    isolation). Alternative: keep the VM up and just `gcloud compute instances
    stop/start --machine-type` (faster, but couples runs and risks state drift).
    Recommend tear-down for v1.

 ## File count summary

 | bucket          | new | modified | deleted |
 |-----------------|----:|---------:|--------:|
 | `loadtest/`     | 5   | 0        | 0       |
 | `cloud/`        | 0   | 1        | 0       |
 | `docs/`         | 1   | 0        | 0       |
 | `Makefile*`     | 0   | 1        | 0       |
 | `pyproject.toml`| 0   | 1        | 0       |
 | `.gitignore`    | 0   | 1        | 0       |
 | **total**       | **6** | **5**  | **0**   |

 Net addition: ~700 LOC; ~100 LOC of those generated (payload corpus, locust HTML).

 ## Agent prompt (paste-ready)

 ```
 Implement the M5 API load-test sweep per docs/plans/api_loadtest.md. The plan is
 the source of truth — if it conflicts with the existing repo, follow the plan.

 Constraints (do not violate):
 - Do NOT modify src/m5/serve/* (the service is already shipped — only observe).
 - Do NOT modify cloud/cloud-init/* (the train.sh is now in production use).
 - Reuse existing cloud/terraform/gcp/ module; add `serve_replicas` var only.
 - Locust must be in a new dep group `loadtest`, NOT in main deps.
 - Every tier must self-tear-down on success AND failure (trap EXIT in tier_sweep.sh).
 - Total spend per full sweep MUST stay < $2 (enforce via tiers.yaml guardrail).
 - All new files under loadtest/ MUST have ruff/mypy passing.
 - reports/loadtest/ and loadtest/payloads/ MUST be gitignored.

 Acceptance criteria:
 - `make loadtest-tier TIER=cheap` runs end-to-end on the existing GCP project,
   produces reports/loadtest/<ts>/cheap_stats.csv, and tears down cleanly.
 - `make loadtest-sweep` runs all 4 tiers, produces summary.md with the
   $/MRPS × p95 table, and the 3 figures.
 - `make loadtest-report TS=<ts>` re-renders summary from existing CSVs (no GCP calls).
 - CI runs `make loadtest-tier TIER=cheap DRY_RUN=1` (local m5 serve, no GCP).
 - All hardening checklist boxes verifiable from the implementation.

 Workflow:
 - Branch off main as feat/api-loadtest.
 - One PR; squash-mergeable.
 - PR description mirrors `docs/plans/api_loadtest.md` § Approach + § Hardening.

 Out of scope (defer to a v2):
 - HTTPS / Cloud Load Balancer in front of the VM.
 - Multi-replica horizontal scale tests.
 - Distributed locust workers (one master/local-worker is enough at this scale).
 - Cross-region tier comparison (us-central1 only).
 - Comparing alternate runtimes (gunicorn / hypercorn) — uvicorn only.
 ```

 ## Once implemented — interpretation note

 Expected shape of the result:

 - `cheap` (e2-small) saturates fast (15–30 RPS) with p99 spikes from CPU credit
   exhaustion. **$/MRPS will look great** until the credits run out — that's the
   classic e2 trap.
 - `default` (e2-medium) probably the best **$/MRPS for low-traffic** apps (≤ 50 RPS).
 - `dedicated` (n2-standard-2) gives the cleanest p99 numbers; expect ~80–120 RPS.
 - `cpu-heavy` (n2-highcpu-4) likely the **best $/MRPS overall** for this workload —
   LGBM inference is CPU-bound and 4 dedicated vCPUs at $0.143/h beat 2 dedicated
   at $0.097/h on a per-request basis. If this hypothesis fails, the sweep is
   doing its job.
