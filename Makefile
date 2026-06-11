# M5 Forecasting — canonical entrypoint.
# Linux/macOS/WSL only. `make help` to see everything.

.DEFAULT_GOAL := help
.PHONY: help bootstrap install activate lint fmt typecheck test test-smoke test-unit \
        test-integration test-fast cov check \
        download prep cv-stats cv-lgbm cv-hier cv-recipe cv-segmented cv-store cv-store-cat cv-store-dept cv-toto \
        forecast-stats forecast-lgbm forecast-hier forecast-segmented forecast-store forecast-store-cat forecast-store-dept forecast-toto \
        train serve serve-prod docker-build docker-up docker-down docker-logs \
        score score-all compare compare-existing eval viz notebook notebook-toto clean clean-all

UV       ?= uv
VENV     ?= .venv
HORIZON  ?= 28
WINDOWS  ?= 3
MODEL    ?= stats
MODELS   ?= stats lgbm
REPORT   ?= reports
RUN_ID   ?= latest

# Pin uv's project environment to the path the rest of the toolchain expects
# (.vscode/settings.json, the README activate snippet, scripts/bootstrap.sh).
export UV_PROJECT_ENVIRONMENT := $(VENV)

# Silence statsforecast's docstring SyntaxWarnings (cosmetic noise from raw-string
# escapes). They fire at compile time before any code-level filter can run, so
# this env-var-based filter is the only reliable knob.
export PYTHONWARNINGS := ignore::SyntaxWarning

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "; \
	             printf "\nM5 Forecasting — make targets\n\nUsage: make <target> [VAR=value]\n\n"} \
	     /^[a-zA-Z_.-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nVariables (override on CLI): HORIZON=%s WINDOWS=%s MODEL=%s\n\n" \
	        "$(HORIZON)" "$(WINDOWS)" "$(MODEL)"

# ---- Setup ---------------------------------------------------------

bootstrap: ## First-time setup (installs uv, syncs deps, downloads M5 data)
	@bash scripts/bootstrap.sh

install: ## Sync deps, register Jupyter kernel, install pre-commit hooks, print activation hint
	$(UV) sync --all-groups
	@echo "==> Registering Jupyter kernel 'm5'"
	@$(UV) run python -m ipykernel install --user --name m5 --display-name "Python (m5)" >/dev/null
	@if [ -d .git ]; then \
	    echo "==> Installing pre-commit hooks"; \
	    $(UV) run pre-commit install >/dev/null; \
	fi
	@if [ ! -f .env ] && [ -f .env.example ]; then \
	    echo "==> Seeding .env from .env.example"; \
	    cp .env.example .env; \
	fi
	@printf '\n\033[32m==> Install complete.\033[0m venv at \033[36m./%s\033[0m\n' "$(VENV)"
	@printf '   • Shell:    \033[36msource %s/bin/activate\033[0m\n' "$(VENV)"
	@printf '   • VSCode:   auto-activates via .vscode/settings.json (reload window if needed)\n'
	@printf '   • No-activate: \033[36muv run <cmd>\033[0m always works\n'
	@printf '   • Or run:   \033[36mmake activate\033[0m to re-print this hint\n\n'

activate: ## Print the command to activate the project venv (run with `eval $$(make activate)`)
	@if [ ! -d "$(VENV)" ]; then \
	    printf '\033[31m==> No venv at ./%s — run `make install` first.\033[0m\n' "$(VENV)" >&2; \
	    exit 1; \
	fi
	@echo "source $(VENV)/bin/activate"
	@printf '\n# Tip: shell-source it directly:\n#   \033[36msource %s/bin/activate\033[0m\n' "$(VENV)" >&2
	@printf '# Or eval the make output:\n#   \033[36meval "$$(make activate | head -1)"\033[0m\n' >&2

# ---- Quality -------------------------------------------------------

lint: ## Lint with ruff
	$(UV) run ruff check .

fmt: ## Format with ruff
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck: ## mypy on src/m5
	$(UV) run mypy

test: ## Run the full pytest suite (smoke + unit + integration)
	$(UV) run pytest

test-smoke: ## Smoke tier — imports, CLI help, package metadata (~1s)
	$(UV) run pytest -m smoke

test-unit: ## Unit tier — pure-function tests on config/data/features/eval
	$(UV) run pytest -m unit

test-integration: ## Integration tier — model fit/predict + CV on toy data
	$(UV) run pytest -m integration

test-fast: ## Smoke + unit (skip integration and `slow`)
	$(UV) run pytest -m "smoke or unit" --no-cov -q

cov: ## Run the suite with coverage (terminal + htmlcov/)
	$(UV) run pytest --cov=m5 --cov-report=term-missing --cov-report=html

check: lint typecheck test ## Lint + types + tests (CI entry point)

# ---- Pipeline ------------------------------------------------------

download: ## Download M5 raw data into data/m5
	$(UV) run m5 download

prep: ## Build the long-format training parquet
	$(UV) run m5 prep

cv-stats: ## Cross-validate Theta + AutoETS + SeasonalNaive
	$(UV) run m5 cv stats --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-lgbm: ## Cross-validate the LightGBM global model
	$(UV) run m5 cv lgbm --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-hier: ## Cross-validate the hierarchical pipeline (Theta + BU/MinT reconcilers)
	$(UV) run m5 cv hier --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-recipe: ## Cross-validate from a YAML recipe (RECIPE=configs/m5/lgbm.yaml)
	$(UV) run m5 cv-recipe $(RECIPE) --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-segmented: ## Cross-validate 10 store-level LightGBM models
	$(UV) run m5 cv segmented --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-store: ## Alias for cv-segmented (10 store-level models)
	$(UV) run m5 cv store --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-store-cat: ## Cross-validate 30 store-category LightGBM models
	$(UV) run m5 cv store_cat --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-store-dept: ## Cross-validate 70 store-department LightGBM models
	$(UV) run m5 cv store_dept --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-toto: ## Cross-validate the TOTO zero-shot foundation model (needs --group toto)
	$(UV) run --group toto m5 cv toto --horizon $(HORIZON) --n-windows $(WINDOWS)

forecast-stats: ## Train+predict statistical baselines
	$(UV) run m5 forecast stats --horizon $(HORIZON)

forecast-lgbm: ## Train+predict LightGBM global model
	$(UV) run m5 forecast lgbm --horizon $(HORIZON)

forecast-hier: ## Train+predict hierarchical (Theta base, 4 reconcilers)
	$(UV) run m5 forecast hier --horizon $(HORIZON)

forecast-segmented: ## Train+predict 10 store-level LightGBM models
	$(UV) run m5 forecast segmented --horizon $(HORIZON)

forecast-store: ## Alias for forecast-segmented
	$(UV) run m5 forecast store --horizon $(HORIZON)

forecast-store-cat: ## Train+predict 30 store-category LightGBM models
	$(UV) run m5 forecast store_cat --horizon $(HORIZON)

forecast-store-dept: ## Train+predict 70 store-department LightGBM models
	$(UV) run m5 forecast store_dept --horizon $(HORIZON)

forecast-toto: ## Zero-shot forecast with TOTO (needs --group toto)
	$(UV) run --group toto m5 forecast toto --horizon $(HORIZON)

# ---- Serving (FastAPI) --------------------------------------------
# `make train` produces a versioned artifact under artifacts/models/lgbm/<ts>/
# and updates the `latest` symlink. `make serve` boots uvicorn pointing at
# M5_SERVE_MODEL_DIR (defaults to artifacts/models/lgbm/latest).

train: ## Fit LightGBM and persist a serving artifact (model.joblib + metadata + history + statics)
	$(UV) run m5 train --horizon $(HORIZON)

serve: ## Run the FastAPI service in dev mode (uvicorn --reload)
	$(UV) run m5 serve --reload

serve-prod: ## Run the FastAPI service with prod-style settings (no reload, multi-worker via env)
	M5_SERVE_LOG_JSON=true $(UV) run m5 serve

docker-build: ## Build the production container image (m5-forecaster:local)
	docker build -t m5-forecaster:local .

docker-up: ## Bring up the service via docker compose (mounts the latest artifact)
	docker compose up -d --build
	@echo "==> Service: http://localhost:$${M5_SERVE_PORT:-8000} (docs at /docs, metrics at /metrics)"

docker-down: ## Tear down the docker compose stack
	docker compose down

docker-logs: ## Tail the container logs
	docker compose logs -f m5-forecaster

# ---- Scoring + report ----------------------------------------------
# `make score MODELS="stats lgbm"` reads artifacts/cv_<m>.parquet for each
# model, computes the full metric pack, and writes figures + report under
# $(REPORT)/. `make score-all` is the convenience batch.

score: ## Score CV artifacts → reports/{figures,metrics,report.md,report.html} (MODELS="stats lgbm")
	@set -e; CMD="$(UV) run m5 score --out $(REPORT) --run-id $(RUN_ID)"; \
	for m in $(MODELS); do CMD="$$CMD --model $$m"; done; \
	echo "$$CMD"; eval $$CMD

score-all: ## Score every CV artifact found in artifacts/ (cv_<name>.parquet)
	@set -e; \
	models=$$(ls artifacts/cv_*.parquet 2>/dev/null | sed -e 's|artifacts/cv_||' -e 's|\.parquet$$||'); \
	if [ -z "$$models" ]; then echo "No artifacts/cv_*.parquet found — run a cv-* target first." >&2; exit 1; fi; \
	CMD="$(UV) run m5 score --out $(REPORT) --run-id $(RUN_ID)"; \
	for m in $$models; do CMD="$$CMD --model $$m"; done; \
	echo "$$CMD"; eval $$CMD

compare-existing: ## Score existing artifacts and print comparison table
	@set -e; \
	models=$$(ls artifacts/cv_*.parquet 2>/dev/null | sed -e 's|artifacts/cv_||' -e 's|\.parquet$$||'); \
	if [ -z "$$models" ]; then echo "No artifacts/cv_*.parquet found — run a cv-* target first." >&2; exit 1; fi; \
	CMD="$(UV) run m5 score --out $(REPORT) --run-id $(RUN_ID)"; \
	for m in $$models; do CMD="$$CMD --model $$m"; done; \
	echo "$$CMD"; eval $$CMD; \
	$(UV) run python scripts/compare_scores.py $(REPORT) --baseline lgbm

eval: cv-stats cv-lgbm score ## End-to-end: stats + lgbm CV, then score the merged report

ensemble: ## Build ensemble from existing CV artifacts (lgbm + store + store_cat + stats)
	@echo "==> Building ensemble from existing CV artifacts"
	$(UV) run m5 ensemble --model lgbm --model store --model store_cat --model stats

compare: ## Run all CVs + build ensemble + score + print comparison table
	@echo "==> Running baseline + new model CVs (capped for dev)"
	M5_N_SERIES=5000 M5_N_WINDOWS=1 $(UV) run m5 cv stats --horizon $(HORIZON)
	M5_N_SERIES=5000 M5_N_WINDOWS=1 $(UV) run m5 cv lgbm --horizon $(HORIZON)
	M5_N_SERIES=5000 M5_N_WINDOWS=1 $(UV) run m5 cv store --horizon $(HORIZON)
	M5_N_SERIES=5000 M5_N_WINDOWS=1 $(UV) run m5 cv store_cat --horizon $(HORIZON)
	@echo "==> Building ensemble"
	$(UV) run m5 ensemble --model lgbm --model store --model store_cat --model stats
	@echo "==> Scoring all models (including ensemble)"
	$(UV) run m5 score --out $(REPORT) --run-id $(RUN_ID) \
		--model stats --model lgbm --model store --model store_cat --model ensemble
	@echo "==> Comparison table"
	$(UV) run python scripts/compare_scores.py $(REPORT) --baseline lgbm

# ---- Load test (phase 1: local; phases 2-3 add GCP tier sweep) -----
# Plan: docs/plans/api_loadtest.md. Generate the payload corpus once
# (reads unique_ids from artifacts/cv_lgbm.parquet), then run locust
# against a local `make serve` instance.

LOADTEST_HOST    ?= http://localhost:8000
LOADTEST_USERS   ?= 20
LOADTEST_SPAWN   ?= 5
LOADTEST_TIME    ?= 60s
LOADTEST_PAYLOAD ?= loadtest/payloads/unique_ids.txt
LOADTEST_CORPUS_N ?= 3000

loadtest-payload: ## Build loadtest/payloads/unique_ids.txt from artifacts/cv_lgbm.parquet
	@set -e; \
	src=artifacts/cv_lgbm.parquet; \
	[ -f "$$src" ] || { echo "$$src not found — run `make cv-lgbm` first." >&2; exit 1; }; \
	mkdir -p loadtest/payloads; \
	$(UV) run --group loadtest python -c "import pandas as pd; pd.read_parquet('$$src')['unique_id'].drop_duplicates().head($(LOADTEST_CORPUS_N)).to_csv('$(LOADTEST_PAYLOAD)', index=False, header=False)"; \
	echo "==> wrote $(LOADTEST_PAYLOAD) ($$(wc -l < $(LOADTEST_PAYLOAD)) ids)"

loadtest-local: ## Run locust against a local `make serve` (headless, csv → reports/loadtest/local_*)
	@[ -f $(LOADTEST_PAYLOAD) ] || { echo "$(LOADTEST_PAYLOAD) not found — run `make loadtest-payload` first." >&2; exit 1; }
	@mkdir -p reports/loadtest
	M5_LOADTEST_PAYLOAD=$(LOADTEST_PAYLOAD) \
	$(UV) run --group loadtest locust -f loadtest/locustfile.py --headless \
	    -u $(LOADTEST_USERS) -r $(LOADTEST_SPAWN) -H $(LOADTEST_HOST) \
	    --run-time $(LOADTEST_TIME) \
	    --csv reports/loadtest/local \
	    --html reports/loadtest/local.html

loadtest-tier-gcp: ## Run a single GCP tier end-to-end (TIER=cheap; needs GOOGLE_APPLICATION_CREDENTIALS)
	@[ -n "$(TIER)" ] || { echo "Usage: make loadtest-tier-gcp TIER=cheap" >&2; exit 1; }
	@[ -f $(LOADTEST_PAYLOAD) ] || { echo "$(LOADTEST_PAYLOAD) not found — run `make loadtest-payload` first." >&2; exit 1; }
	$(UV) run --group loadtest python -m loadtest.sweep tier --alias $(TIER)

loadtest-sweep-gcp: ## Run the full GCP tier sweep (cost-guarded by loadtest/tiers.yaml)
	@[ -f $(LOADTEST_PAYLOAD) ] || { echo "$(LOADTEST_PAYLOAD) not found — run `make loadtest-payload` first." >&2; exit 1; }
	$(UV) run --group loadtest python -m loadtest.sweep all

loadtest-sweep-plan: ## Print the sweep plan (no GCP calls; dry-run)
	$(UV) run --group loadtest python -m loadtest.sweep all --dry-run

loadtest-aggregate: ## Build summary.md + figures from a sweep dir (TS=<UTC-timestamp>)
	@[ -n "$(TS)" ] || { echo "Usage: make loadtest-aggregate TS=20260510T060000Z" >&2; exit 1; }
	$(UV) run --group loadtest python -m loadtest.aggregate reports/loadtest/$(TS)

# ---- Visualisation -------------------------------------------------
# `make viz` reads the latest fitted artifact + long.parquet and renders
# assets/pipeline.svg (auto-plays in README) + assets/pipeline.html
# (interactive D3 page, scrub through CV windows).

viz: ## Render assets/pipeline.{svg,html} from the latest fitted artifact
	$(UV) run m5 viz
# ---- Notebooks -----------------------------------------------------

notebook: ## Launch Jupyter Lab with the notebook dep group
	$(UV) run --group notebook jupyter lab

notebook-toto: ## Launch Jupyter Lab with notebook + toto groups (TOTO notebook)
	$(UV) run --group notebook --group toto jupyter lab

# ---- Cleanup -------------------------------------------------------

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean ## Also remove .venv, processed data, and forecasts
	rm -rf .venv data/processed forecasts artifacts

# ---- Cloud (multi-provider train + serve via Terraform) ------------

include cloud/Makefile.cloud
