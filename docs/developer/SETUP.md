# Setup — from zero to a running pipeline

This guide takes you from a fresh machine to a working M5 environment. It is
written so that someone who has never used `uv` or VSCode for Python can follow
it; experienced devs can skip to [the TL;DR](#tldr).

## TL;DR

```bash
git clone <repo-url> M5 && cd M5
make bootstrap   # installs uv, deps, .env, downloads M5 raw data
make check       # lint + types + tests must pass before you start hacking
make notebook    # opens Jupyter Lab
```

## What you need on your machine

| Tool       | Why                                     | Check it works            |
|------------|------------------------------------------|---------------------------|
| `bash`     | Driver shell for `make` and scripts      | `bash --version`          |
| `make`     | Canonical task runner                    | `make --version`          |
| `git`      | Source control                           | `git --version`           |
| `curl`     | Used by `bootstrap.sh` to install `uv`   | `curl --version`          |
| Python 3.12 | Pulled automatically by `uv` if missing  | `uv python list`          |

You do **not** need to install `uv`, the venv, or any Python packages by hand —
`make bootstrap` does that.

### Windows users

Run everything inside **WSL2** (Ubuntu recommended). The native PowerShell /
`cmd.exe` flow is intentionally unsupported so there is exactly one blessed path.

```powershell
# In PowerShell (one time):
wsl --install -d Ubuntu
# Then open the Ubuntu shell and continue with `git clone ...`
```

### Linux / macOS

Install `make`, `git`, and `curl` via your package manager. Examples:

- Ubuntu / Debian: `sudo apt update && sudo apt install -y build-essential git curl`
- Fedora: `sudo dnf install -y make git curl`
- macOS: `xcode-select --install` (provides `make`, `git`, `curl`)

## First-time setup

```bash
git clone <repo-url> M5
cd M5
make bootstrap
```

`make bootstrap` is idempotent — safe to re-run any time. It will:

1. Install [`uv`](https://docs.astral.sh/uv/) into `~/.local/bin` (if missing).
2. Run `uv sync --all-groups` to create `.venv` with Python 3.12 and every
   dependency (Nixtla, LightGBM, etc.) pinned by `uv.lock`.
3. Copy `.env.example` → `.env` so you can override settings later.
4. Register a Jupyter kernel called **Python (m5)** that points into `.venv`.
5. Download the M5 raw CSVs into `data/m5/datasets/` (~250 MB, one-time).

If `uv` was just installed and your shell can't find it, open a new terminal or
run `source ~/.local/bin/env` and re-run `make bootstrap`.

## Verify the install

```bash
make check         # ruff lint + mypy + pytest, all must pass
uv run m5 --help   # should print the CLI usage
```

If both succeed you're done.

## VSCode — recommended setup

### Install VSCode

- **Linux / macOS**: download from <https://code.visualstudio.com/>.
- **Windows + WSL**: install VSCode on Windows, then add the
  [WSL extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl)
  and open the project from inside WSL with `code .` from the Ubuntu shell.

### Open the project

```bash
code .
```

The workspace ships with `.vscode/settings.json` (config) and
`.vscode/extensions.json` (extension picks). VSCode will prompt **"Install
recommended extensions?"** the first time — say yes. The list:

| Extension                            | Why                                              |
|--------------------------------------|--------------------------------------------------|
| **Python** (`ms-python.python`)      | Python language support                          |
| **Pylance** (`ms-python.vscode-pylance`) | Type checking + IntelliSense                  |
| **Debugpy** (`ms-python.debugpy`)    | Step-through debugger                            |
| **Ruff** (`charliermarsh.ruff`)      | Lint + format on save (native server)            |
| **Jupyter** (`ms-toolsai.jupyter`)   | Run notebooks in VSCode                          |
| **Even Better TOML**                 | `pyproject.toml` syntax + linting                |
| **Rainbow CSV**                      | Inspect M5 raw CSVs                              |
| **GitLens** + **Git Graph**          | History, blame, branch graph                     |
| **EditorConfig**                     | Honour `.editorconfig`                           |
| **Code Spell Checker**               | Catch typos in docstrings + markdown             |
| **Shell Format** + **ShellCheck**    | Format and lint `scripts/*.sh`                   |
| **Markdown All in One**              | Markdown editing helpers                         |
| **Makefile Tools**                   | Syntax + jump-to-target for the Makefile         |

### What's already configured for you

`.vscode/settings.json` does the following without any extra clicks:

- Python interpreter points to `.venv/bin/python`.
- Format-on-save uses **Ruff** (Python files and notebook cells).
- Imports auto-sort on save, lint fixes auto-applied.
- Pytest is the test runner; tests live in `tests/`.
- Files matching `__pycache__/`, caches, lockfiles, parquet, and `data/`
  are hidden from search and the file watcher to keep VSCode snappy.

> **Note on Ruff settings**: we use the modern **native server** (`ruff.nativeServer: "on"`).
> The legacy `ruff-lsp` is deprecated — if you see warnings about
> `ruff.lint.run`, your local user-settings have an old key. Remove it from
> your User settings (Cmd/Ctrl+Shift+P → *"Preferences: Open User Settings (JSON)"*).

### Run a notebook

1. Open `notebooks/00_run_pipeline.ipynb`.
2. In the top-right kernel picker, choose **Python (m5)** (registered by
   `make bootstrap`). If it isn't there, run `make install` to re-register it.
3. Run cells with `Shift+Enter`.

### Debug a test

1. Open a test file under `tests/`.
2. Click the green ▷ next to a test, or use the Test Explorer side panel.
3. To debug, hover over a test and click *"Debug Test"*.

## Optional: GPU / Apple Silicon

LightGBM ships CPU-only by default. For GPU builds, see the LightGBM docs and
override the dependency with `uv add lightgbm --extra gpu`. We don't pin GPU
extras here because most M5 users don't need them — the Tweedie model trains
quickly on CPU.

## Where to go next

- [`DEVELOPMENT.md`](DEVELOPMENT.md) — daily workflow.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — why this stack and how the package fits together.
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — common errors.
