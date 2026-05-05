#!/usr/bin/env bash
# First-time setup for the M5 project. Idempotent.
# Usage: bash scripts/bootstrap.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# 1) Install uv if missing -------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "${HOME}/.local/bin/env" 2>/dev/null || true
fi

# 2) Sync deps -------------------------------------------------------
# Pin the venv path so `.vscode/settings.json` and `make activate` agree.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv}"
echo "==> Syncing dependencies (uv sync --all-groups → ./${UV_PROJECT_ENVIRONMENT})"
uv sync --all-groups

# 3) Seed .env -------------------------------------------------------
if [ ! -f .env ] && [ -f .env.example ]; then
    echo "==> Creating .env from .env.example"
    cp .env.example .env
fi

# 4) Register Jupyter kernel ----------------------------------------
echo "==> Registering Jupyter kernel 'm5'"
uv run python -m ipykernel install --user --name m5 --display-name "Python (m5)" >/dev/null

# 5) Download data ---------------------------------------------------
if [ ! -d data/m5/datasets ]; then
    echo "==> Downloading M5 raw data (one-time, ~250MB)"
    uv run m5 download
else
    echo "==> M5 raw data already present (skip download)"
fi

cat <<NEXT

==> Bootstrap complete. venv at ./${UV_PROJECT_ENVIRONMENT}

Activate the environment (any one):
    source ${UV_PROJECT_ENVIRONMENT}/bin/activate     # explicit shell activation
    code .                                             # VSCode auto-activates via .vscode/settings.json
    uv run <cmd>                                       # works without activating

Next steps:
    make prep            # build the long-format training parquet
    make cv-stats        # rolling-origin CV with Theta/ETS/SeasonalNaive
    make cv-lgbm         # rolling-origin CV with LightGBM
    make notebook        # launch Jupyter Lab

Run \`make help\` for everything else.
NEXT
