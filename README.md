# M5 Forecasting Accuracy — reproducible Nixtla solution

Daily 28-day forecast for **30,490 Walmart product–store series** (Kaggle
[M5 Forecasting – Accuracy](https://kaggle.com/competitions/m5-forecasting-accuracy)).
Goal: get close to world-class with as few features as possible — three model
families (Theta, AutoETS, LightGBM) on a deliberately small feature menu, with
reproducible rolling-origin cross-validation.

## Stack

- **Python 3.12**, dependency-managed by [`uv`](https://docs.astral.sh/uv/)
- **[Nixtla](https://github.com/Nixtla)** — `statsforecast`, `mlforecast`,
  `utilsforecast`, `datasetsforecast`, `hierarchicalforecast`
- **LightGBM** for the global ML model (Tweedie objective, deterministic mode)
- **Polars + pandas + pyarrow** for I/O and shaping
- **ruff + mypy + pytest** for quality
- **Typer** CLI exposed as `uv run m5 …`

## Prerequisites

- A POSIX shell with `make` — Linux, macOS, or **Windows via WSL**
- Internet access for the first dependency sync and dataset download

Everything else (`uv`, the venv, deps, raw data) is installed by
`make bootstrap`. There is no PowerShell / `cmd.exe` path; that's intentional.

## Quick start

```bash
make bootstrap     # install uv, sync deps, seed .env, download M5 data
make prep          # build data/processed/long.parquet
make cv-stats      # rolling-origin CV: Theta + AutoETS + SeasonalNaive
make cv-lgbm       # rolling-origin CV: LightGBM global model
```

Run `make help` for everything.

| Target              | What it does                                            |
|---------------------|---------------------------------------------------------|
| `make bootstrap`    | First-time setup (idempotent)                           |
| `make install`      | `uv sync --all-groups` + register Jupyter kernel        |
| `make lint` / `fmt` | Ruff lint / format                                      |
| `make typecheck`    | mypy on `src/m5`                                        |
| `make test` / `cov` | pytest, optionally with coverage                        |
| `make check`        | `lint` + `typecheck` + `test` (CI entry)                |
| `make download`     | Pull M5 raw CSVs into `data/m5`                         |
| `make prep`         | Build the long-format training parquet                  |
| `make cv-stats`     | CV with Theta / AutoETS / SeasonalNaive                 |
| `make cv-lgbm`      | CV with LightGBM (`mlforecast`)                         |
| `make forecast-*`   | Train on full data and emit a 28-day future forecast    |
| `make notebook`     | Jupyter Lab with the `notebook` dep group               |
| `make clean[-all]`  | Remove caches (and `.venv`/data with `clean-all`)       |

## CLI

The Make targets are thin wrappers over the `m5` Typer CLI:

```bash
uv run m5 --help
uv run m5 download
uv run m5 prep --last-n-days 400 --n-series -1
uv run m5 cv stats --horizon 28 --n-windows 3
uv run m5 cv lgbm --horizon 28 --n-windows 3
uv run m5 forecast lgbm --horizon 28
```

## Project layout

```
M5/
├── Makefile                    # canonical entrypoint (Linux/macOS/WSL)
├── pyproject.toml              # uv-managed deps, ruff/mypy/pytest config
├── .env.example                # DATA_DIR, M5_SEED, M5_HORIZON, …
├── .python-version             # 3.12
├── .vscode/settings.json       # ruff on save, pytest, .venv interpreter
├── scripts/
│   ├── bootstrap.sh            # one-shot setup
│   └── download_data.sh        # cron-friendly data refresh
├── src/m5/                     # the package
│   ├── config.py               # paths, seeds, env-driven Settings
│   ├── data.py                 # load + melt → Nixtla long frame
│   ├── features.py             # minimal date / snap / event / price feats
│   ├── evaluation.py           # WRMSSE
│   ├── cv.py                   # reproducible rolling-origin CV
│   ├── cli.py                  # Typer CLI (`m5 …`)
│   ├── plots.py                # matplotlib helpers
│   └── models/
│       ├── stats.py            # Theta + AutoETS + SeasonalNaive
│       └── lgbm.py             # LightGBM via mlforecast
├── notebooks/                  # 00_run_pipeline + the original EDA suite
├── tests/                      # pytest unit + smoke tests
├── plots/                      # static images from the original analysis
└── data/                       # raw + processed (gitignored)
```

## Configuration

```bash
cp .env.example .env
```

Variables (all optional, sensible defaults):

```
DATA_DIR=data
M5_SEED=42
M5_HORIZON=28          # M5 evaluation window
M5_N_WINDOWS=3         # rolling-origin CV windows
M5_LAST_N_DAYS=400     # trailing window of training data
M5_N_SERIES=-1         # subsample (-1 = all 30,490)
LOG_LEVEL=INFO
```

## Documentation

- [`docs/developer/SETUP.md`](docs/developer/SETUP.md) — first-time install (WSL, uv, VSCode).
- [`docs/developer/DEVELOPMENT.md`](docs/developer/DEVELOPMENT.md) — daily workflow, testing, debugging.
- [`docs/developer/ARCHITECTURE.md`](docs/developer/ARCHITECTURE.md) — why this stack, package layout, data flow.
- [`docs/developer/TROUBLESHOOTING.md`](docs/developer/TROUBLESHOOTING.md) — common errors.

## Approach

See [WriteUp.md](WriteUp.md) for the full methodology and references.

The short version:

1. **EDA → minimal features.** Daily seasonality is dominant; weekly cycle,
   month-of-year, snap day, and a single binary "any event today" flag carry
   most of the calendar signal. We add normalised and week-over-week-changed
   prices on top.
2. **Theta + AutoETS** as univariate baselines (`statsforecast`).
3. **LightGBM global model** via `mlforecast` with lags 7/14/28, rolling means
   over 7/28 days, and the same minimal date/price feature set. Tweedie
   objective, `deterministic=True`, fixed seed.
4. **Reproducible CV.** `cross_validation(h=28, n_windows=3)` with a global
   seed set before every run. Every script writes its forecast to
   `artifacts/cv_<model>.parquet` so leaderboards are diff-friendly.
5. **WRMSSE** is implemented in `m5.evaluation` (bottom-level item × store
   weights from trailing dollar sales, scales from in-sample naive-1 MSE).

## License

MIT — see the original M5 competition for data terms.
