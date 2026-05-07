# M5 Forecasting — canonical entrypoint.
# Linux/macOS/WSL only. `make help` to see everything.

.DEFAULT_GOAL := help
.PHONY: help bootstrap install activate lint fmt typecheck test test-smoke test-unit \
        test-integration test-fast cov check \
        download prep cv-stats cv-lgbm cv-hier cv-recipe forecast-stats forecast-lgbm forecast-hier \
        train serve serve-prod docker-build docker-up docker-down docker-logs \
        score score-all eval notebook clean clean-all

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

cv-hier: ## Cross-validate the hierarchical pipeline (Theta + BU/TD/MinT reconcilers)
	$(UV) run m5 cv hier --horizon $(HORIZON) --n-windows $(WINDOWS)

cv-recipe: ## Cross-validate from a YAML recipe (RECIPE=configs/m5/lgbm.yaml)
	$(UV) run m5 cv-recipe $(RECIPE) --horizon $(HORIZON) --n-windows $(WINDOWS)

forecast-stats: ## Train+predict statistical baselines
	$(UV) run m5 forecast stats --horizon $(HORIZON)

forecast-lgbm: ## Train+predict LightGBM global model
	$(UV) run m5 forecast lgbm --horizon $(HORIZON)

forecast-hier: ## Train+predict hierarchical (Theta base, 4 reconcilers)
	$(UV) run m5 forecast hier --horizon $(HORIZON)

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

eval: cv-stats cv-lgbm score ## End-to-end: stats + lgbm CV, then score the merged report

# ---- Notebooks -----------------------------------------------------

notebook: ## Launch Jupyter Lab with the notebook dep group
	$(UV) run --group notebook jupyter lab

# ---- Cleanup -------------------------------------------------------

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean ## Also remove .venv, processed data, and forecasts
	rm -rf .venv data/processed forecasts artifacts

# ---- Cloud (multi-provider train + serve via Terraform) ------------

include cloud/Makefile.cloud
