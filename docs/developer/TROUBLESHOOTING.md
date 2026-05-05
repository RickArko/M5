# Troubleshooting

A growing list of "I hit X — now what?" entries.

## `command not found: uv` after bootstrap

`uv` was installed into `~/.local/bin`, which isn't on your `PATH` yet.

```bash
source ~/.local/bin/env   # bash/zsh
# or open a new terminal, then:
make bootstrap
```

## VSCode warning: *"The legacy server (ruff-lsp) has been deprecated"*

The repo ships with `"ruff.nativeServer": "on"` in `.vscode/settings.json`, so
your **workspace** is fine. The warning means your **User** settings still have
the deprecated key, e.g.:

```jsonc
// In your User settings.json — REMOVE these:
"ruff.lint.run": "onSave"
```

Open VSCode → `Cmd/Ctrl+Shift+P` → *"Preferences: Open User Settings (JSON)"*
and delete any `ruff.*` keys you don't recognise. Reload the window.

## VSCode can't find the *Python (m5)* kernel in notebooks

The Jupyter kernel is registered globally by `make bootstrap` /
`make install`. If it's missing:

```bash
make install
```

That re-runs `python -m ipykernel install --user --name m5 ...`. Restart VSCode
and the kernel picker will show **Python (m5)**.

## `ModuleNotFoundError: No module named 'm5'` in a notebook

Two common causes:

1. The notebook is using a kernel that isn't `Python (m5)`. Check the
   top-right kernel picker.
2. The package wasn't installed editable. `make install` (or `uv sync`) does
   the right thing — `uv` handles editable installs automatically with the
   `uv_build` backend.

## `mlforecast`: `num_threads must be -1 or a positive integer`

You're on an old version of the codebase. Pull `main`. The fix is
[`src/m5/models/lgbm.py`](../../src/m5/models/lgbm.py) using `num_threads=n_jobs`
directly (not `n_jobs if n_jobs > 0 else 0`).

## Tests pass locally but fail in CI

Almost always one of:

- **Path-sensitive code** — use `m5.config.SETTINGS.data_dir`, never hardcode `data/`.
- **Wall-clock seeded** — use `set_global_seed()`, never `np.random.seed(time.time())`.
- **Order-sensitive sets** — sort before comparing.

## `data/m5/datasets/` is empty

Run `make download` (or `uv run m5 download`). That pulls the raw CSVs from
`datasetsforecast.M5.load(...)`. The download is ~250 MB, one-time.

## RAM blows up during `make prep`

The default keeps the last 400 days of all 30,490 series. Reduce either axis:

```bash
M5_LAST_N_DAYS=200 M5_N_SERIES=5000 make prep
```

## LightGBM produces non-deterministic results

Check that you didn't override `seed`, `deterministic`, or `force_row_wise` in
`src/m5/models/lgbm.py::lgbm_params`. Also confirm `set_global_seed()` is being
called — every CV/forecast entry point in `src/m5/cv.py` does this on entry.

## I want a feature added

The repo's design tenet is **fewer features, not more**. Before adding one,
audit whether the existing model is missing the *signal* (and demonstrate that
with a CV diff), not just the feature column. See
[`ARCHITECTURE.md`](ARCHITECTURE.md#feature-menu-deliberately-small).
