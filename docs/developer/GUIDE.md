# Developer Guide

## Quick start

```bash
git clone <repo-url> M5 && cd M5
make bootstrap   # install uv, sync deps, seed .env, download M5 raw data (~250 MB)
make check       # lint + types + tests — must pass
make prep        # build data/processed/long.parquet
make cv-stats    # CV: Theta + AutoETS + SeasonalNaive
make cv-lgbm     # CV: LightGBM global model
```

If `command not found: uv` after bootstrap: `source ~/.local/bin/env` or open a new terminal.

**Prerequisites:** `bash`, `make`, `git`, `curl`. No need to pre-install Python — `make bootstrap` provisions everything via `uv`.

**Windows:** use WSL2 (Ubuntu). No native PowerShell path.

### Activate the venv

```bash
source .venv/bin/activate     # explicit
code .                        # VSCode auto-activates
uv run m5 --help              # no activation needed
```

## Inner loop

```bash
make fmt          # ruff format + autofix
make lint         # ruff check
make typecheck    # mypy on src/m5
make test-fast    # smoke + unit (~5 s) — run after every edit
make check        # all of the above (CI entry point)
```

Aim for green `make check` before every commit.

## Running the pipeline

```bash
make download               # one-time raw data pull
make prep                   # → data/processed/long.parquet
make cv-stats               # → artifacts/cv_stats.parquet
make cv-lgbm                # → artifacts/cv_lgbm.parquet
make forecast-lgbm          # train on full data → forecasts/forecast_lgbm.parquet
```

Every flag is also an env var. Fast iteration:

```bash
M5_N_SERIES=500 M5_LAST_N_DAYS=200 M5_N_WINDOWS=1 make prep cv-lgbm
```

## Testing

Tests live in `tests/` with auto-marking by directory:

| Tier | Make target | What it covers |
|---|---|---|
| `tests/smoke/` | `make test-smoke` (~1 s) | imports, CLI help, metadata |
| `tests/unit/` | `make test-unit` | config, data, features, evaluation |
| `tests/integration/` | `make test-integration` | model fit/predict on toy data |
| all | `make test` | full suite |
| fast subset | `make test-fast` | smoke + unit only |

Use `uv run pytest tests/unit/test_evaluation.py`, `-k wrmsse`, or `-x --pdb`.

The shared fixture is `tests/conftest.py::toy_long` (3 series × 200 days). Don't synthesize data ad-hoc.

Add `@pytest.mark.slow` to any test > ~5 s so `make test-fast` skips it.

## Adding dependencies

```bash
uv add <package>                       # runtime dep
uv add --dev <package>                 # dev-only
uv add --group notebook <package>      # `make notebook` group
```

Commits both `pyproject.toml` and `uv.lock`.

## Notebooks

```bash
make notebook         # Jupyter Lab with the `notebook` dep group
```

Always select the **Python (m5)** kernel. Heavy lifting goes in `src/m5/`, not cells.

## Debugging

- **VSCode:** breakpoint → *"Debug Test"* from the Test Explorer
- **pdb:** `uv run pytest -x --pdb`
- **loguru:** `LOG_LEVEL=DEBUG make prep`
- **Crash trace:** `uv run python -X faulthandler -m m5 prep`

## Common fixes

| Symptom | Fix |
|---|---|
| `command not found: uv` | `source ~/.local/bin/env` or new terminal |
| No *Python (m5)* kernel | `make install` re-registers it |
| `ModuleNotFoundError: m5` in notebook | Wrong kernel — pick **Python (m5)** |
| `data/m5/datasets/` empty | `make download` (~250 MB) |
| RAM blowup during `make prep` | `M5_LAST_N_DAYS=200 M5_N_SERIES=5000 make prep` |
| Two CV runs disagree | Determinism bug — file an issue with the diff |
| VSCode "ruff-lsp deprecated" | Remove old `ruff.*` from User settings; workspace already uses native server |
| Tests pass locally, fail CI | Hardcoded path → use `SETTINGS.data_dir`; wall-clock seed → use `set_global_seed()` |
