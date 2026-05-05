# M5 Forecasting — canonical entrypoint.
# Linux/macOS/WSL only. `make help` to see everything.

.DEFAULT_GOAL := help
.PHONY: help bootstrap install lint fmt typecheck test cov check \
        download prep cv-stats cv-lgbm forecast-stats forecast-lgbm \
        notebook clean clean-all

UV       ?= uv
HORIZON  ?= 28
WINDOWS  ?= 3
MODEL    ?= stats

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "; \
	             printf "\nM5 Forecasting — make targets\n\nUsage: make <target> [VAR=value]\n\n"} \
	     /^[a-zA-Z_.-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nVariables (override on CLI): HORIZON=%s WINDOWS=%s MODEL=%s\n\n" \
	        "$(HORIZON)" "$(WINDOWS)" "$(MODEL)"

# ---- Setup ---------------------------------------------------------

bootstrap: ## First-time setup (installs uv, syncs deps, downloads M5 data)
	@bash scripts/bootstrap.sh

install: ## Sync deps and register the `m5` Jupyter kernel
	$(UV) sync --all-groups
	@echo "==> Registering Jupyter kernel 'm5'"
	@$(UV) run python -m ipykernel install --user --name m5 --display-name "Python (m5)"

# ---- Quality -------------------------------------------------------

lint: ## Lint with ruff
	$(UV) run ruff check .

fmt: ## Format with ruff
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck: ## mypy on src/m5
	$(UV) run mypy

test: ## Run pytest
	$(UV) run pytest

cov: ## Run pytest with coverage
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

forecast-stats: ## Train+predict statistical baselines
	$(UV) run m5 forecast stats --horizon $(HORIZON)

forecast-lgbm: ## Train+predict LightGBM global model
	$(UV) run m5 forecast lgbm --horizon $(HORIZON)

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
