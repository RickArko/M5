# Development workflow

The Makefile is the single entry point. `make help` prints every target.

## The inner loop

```bash
make fmt        # auto-format + auto-fix (ruff)
make lint       # ruff lint (no mutation)
make typecheck  # mypy on src/m5
make test       # pytest
make check      # all of the above (CI runs this)
```

Aim for a green `make check` before every commit.

## Editing the package

```text
src/m5/
├── config.py       # paths, seeds, env-driven Settings
├── data.py         # load + melt → Nixtla long frame
├── features.py     # minimal date / snap / event / price features
├── evaluation.py   # WRMSSE
├── cv.py           # reproducible rolling-origin CV
├── cli.py          # Typer CLI (`uv run m5 …`)
├── plots.py        # matplotlib helpers
└── models/
    ├── stats.py    # Theta + AutoETS + SeasonalNaive
    └── lgbm.py     # LightGBM via mlforecast
```

When you add a new module:

1. Put logic in `src/m5/<module>.py`.
2. Add a smoke test in `tests/test_<module>.py`.
3. If it has a CLI surface, add it to `src/m5/cli.py` and a Make target.

## Running the pipeline

```bash
make download           # pull raw data (one-time)
make prep               # build data/processed/long.parquet
make cv-stats           # Theta + AutoETS + SeasonalNaive cross-validation
make cv-lgbm            # LightGBM cross-validation
make forecast-lgbm      # forecast with LightGBM (full data, no holdout)
```

Iterate fast on a subsample:

```bash
M5_N_SERIES=500 make prep
M5_N_WINDOWS=1 make cv-lgbm
```

Every CLI flag is also an env var (see `.env.example`). Using env vars means you
don't have to retype them between Make targets.

## Adding dependencies

```bash
uv add <package>                         # runtime dep
uv add --dev <package>                   # dev-only
uv add --group notebook <package>        # only for `make notebook`
```

`uv` updates `pyproject.toml` and `uv.lock` atomically. Commit both.

## Running tests

The suite is split into three tiers — pick the one that matches what you're
doing.

```text
tests/
├── conftest.py            # shared fixtures, auto-marks tests by directory
├── smoke/                 # fast: imports, CLI --help, package metadata (~1s)
├── unit/                  # pure functions: config, data, features, evaluation
└── integration/           # end-to-end on toy data: models, CV, pipeline
```

```bash
make test               # everything (smoke + unit + integration)
make test-smoke         # ~1s sanity check — run on every save
make test-unit          # ~1s pure-function tests
make test-integration   # ~10s — fits real models on toy data
make test-fast          # smoke + unit, no slow integration

uv run pytest tests/unit/test_evaluation.py   # one file
uv run pytest -k wrmsse                       # by name
uv run pytest -m "not slow"                   # skip > ~5s tests
uv run pytest -x --pdb                        # stop on first failure, drop into pdb
make cov                                      # coverage (terminal + htmlcov/)
```

Markers (`smoke`, `unit`, `integration`, `slow`) are declared in
`[tool.pytest.ini_options]` and `--strict-markers` is on, so a typo will fail
collection rather than silently skip work. You don't add `pytestmark` to test
files — `conftest.py` auto-marks tests based on which subdirectory they live in.

### When to put a test in which tier

| Tier         | Criteria                                                  | Examples                            |
|--------------|-----------------------------------------------------------|-------------------------------------|
| `smoke`      | < 100ms, no I/O, no model fitting                         | imports, CLI `--help`, version      |
| `unit`       | Pure function on a fixture; no third-party model calls    | WRMSSE math, feature transforms     |
| `integration`| Touches a real model / CV / pipeline; toy data is OK      | LGBM fit, statsforecast CV          |

Add `@pytest.mark.slow` to any integration test that takes more than ~5s so
`make test-fast` can skip it.

## Notebooks

Notebooks live in `notebooks/`:

- `00_run_pipeline.ipynb` — minimal driver that calls `src/m5` end-to-end. Use
  this for sanity-checking a fresh install.
- `01_eda.ipynb` … `06_score.ipynb` — the original analytical notebooks.

```bash
make notebook                  # uv run --group notebook jupyter lab
```

Always select the **Python (m5)** kernel.

### Notebook hygiene rules

- Heavy lifting goes in `src/m5`, not in cells.
- Don't commit notebooks with secrets or huge HTML outputs.
- If you want to convert a notebook to a script for headless runs:
  `uv run jupyter nbconvert --to script notebooks/foo.ipynb`.

## Debugging

- **VSCode** — set a breakpoint, run *"Debug Test"* from the Test Explorer.
- **CLI** — `uv run python -X faulthandler -m m5 prep` for crash traces.
- **pdb** — `uv run pytest -x --pdb` drops into the debugger on first failure.
- **loguru level** — `LOG_LEVEL=DEBUG make prep` for verbose timings.

## Reproducibility

Everything that touches randomness goes through `m5.config.set_global_seed`,
which seeds `random`, NumPy, and `PYTHONHASHSEED`. LightGBM is set to
`deterministic=True` and `force_row_wise=True`. The seed is read from
`M5_SEED` (default 42).

If two runs of `make cv-lgbm` give different WRMSSE numbers, that's a bug —
file an issue with the diff.

## Committing

```bash
git switch -c <branch>
make check
git add -p
git commit
```

Conventional commit prefixes are nice but not enforced.

## Releasing

There's no release pipeline yet — this repo is a working analysis project, not
a published package. If that changes, we'll wire `uv build` and `gh release` here.
