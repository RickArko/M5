# Working on M5 with an agentic harness

> Audience: programmers driving an AI coding agent (Claude Code, OpenAI Codex
> CLI, Gemini CLI, Aider, or any agents.md-aware tool) against this repo.
> Goal: get the agent productive in minutes without breaking the
> reproducibility contract or melting the dev box.

If you only read three things first, read them in this order:

1. [`AGENTS.md`](../../AGENTS.md) at the repo root — the short cross-tool
   agent contract (rules, commands, hard limits).
2. [`AI-CONTEXT.md`](../../AI-CONTEXT.md) — token-optimized full repo
   summary. Single best file to drop into an agent's system prompt.
3. This file — extensive contributor guide for agents.

[`CLAUDE.md`](../../CLAUDE.md) is loaded automatically by Claude Code and
overlaps `AGENTS.md`; you don't need to read both unless you're editing
them.

---

## Why M5 is agent-friendly

| Property | What it gives an agent |
|---|---|
| Single CLI (`uv run m5 …`) | One surface to learn; every Make target is a thin wrapper |
| `Makefile` is the entrypoint | `make help` enumerates everything; no hidden scripts |
| Tiered tests (smoke / unit / integration) | Sub-second feedback for most edits |
| Toy synthetic fixture (`tests/conftest.py::toy_long`) | Reproduce and verify without ever touching raw M5 data |
| Reproducibility contract (seeded everywhere) | Same input → same WRMSSE; regression tests are stable |
| Frozen, condensed `AI-CONTEXT.md` | Drop one file and the agent has the whole repo's mental model |
| Nixtla long-frame schema | One canonical shape: `unique_id, ds, y` + statics + features |
| `uv` + locked deps | `uv sync` is hermetic; no "works on my machine" |

Tenet from `docs/developer/ARCHITECTURE.md`: **fewer features, not more.**
Original M5 winners had hundreds of features and deep ensembles. This
repo is a defensible minimum — three model families, ~10 features, full
reproducibility. An agent's first instinct should be to run an
experiment before adding a column.

---

## What every agent should read first (in order)

1. `AGENTS.md` (root) — rules, hard limits, build/test commands.
2. `AI-CONTEXT.md` — full condensed context (~40 KB, token-optimized).
3. `README.md` — happy path, CLI surface, serving section.
4. `docs/developer/ARCHITECTURE.md` — module map + data flow.
5. The specific source file the agent is about to change.
6. `WriteUp.md` — methodology + EDA highlights, only if working on
   forecasting models or evaluation.

`CLAUDE.md` overlaps `AGENTS.md` and is auto-loaded by Claude Code; read
only if editing it.

---

## Hard rules (don't violate)

These are in [`CLAUDE.md`](../../CLAUDE.md) and the root
[`AGENTS.md`](../../AGENTS.md) too. Repeated here so an agent reading
just this file isn't surprised.

1. **Reproducibility is a contract.** Every CV / forecast / train entry
   point calls `set_global_seed()` before fitting. LightGBM uses
   `deterministic=True, force_row_wise=True, seed=SETTINGS.seed`. Two
   runs of the same command must produce the same WRMSSE — if they
   don't, that's a bug.
2. **RAM constraint on dev hardware.** Never run `make prep` /
   `make cv-*` against the full 30,490 series on a laptop. Cap with
   `M5_N_SERIES=200 M5_LAST_N_DAYS=120 M5_N_WINDOWS=1`. Full runs are
   for cloud / remote nodes (`cloud/` has terraform + cloud-init).
3. **The Nixtla schema is fixed.** `unique_id` (`item_id + "_" +
   store_id`), `ds` (datetime64), `y` (float32). Don't reintroduce
   legacy `id` or `_evaluation` suffixes — they were intentionally
   stripped.
4. **The feature menu is deliberately small.** Adding a feature column
   requires a CV diff in the PR body. Prefer extending the lag /
   rolling config in `models/lgbm.py` over piling onto `features.py`.
5. **`data/processed/long.parquet` is the only handoff** between prep
   and modelling. Don't write parallel intermediate parquets.
6. **Drop leading zeros once.** `_drop_leading_zeros` runs in
   `build_long_frame`. Don't repeat it downstream.
7. **`AI-CONTEXT.md` is regenerated, not hand-edited.** Use the
   `/ai-condense` skill (Claude Code) when something material changes.
   Manual edits are silently overwritten.
8. **Don't edit `uv.lock` by hand.** Use `uv add <pkg>` /
   `uv sync --upgrade-package <pkg>`.
9. **Don't bypass pre-commit hooks** (`--no-verify`). If a hook fails,
   fix the underlying issue and create a new commit.
10. **Configuration via env / `.env` only.** Don't hardcode paths,
    seeds, or horizons inside modules.

---

## The standard agent loop

```
┌─ orient ──────────────────────────────────────────────────────┐
│ Read AI-CONTEXT.md + the file you'll change.                  │
│ grep for the symbol you're touching; read its tests.          │
└───────────────────────────────────────────────────────────────┘
            │
            ▼
┌─ reproduce ───────────────────────────────────────────────────┐
│ Use the toy fixture: pytest tests/unit/test_<x>.py -k <case>  │
│ Or capped CV: M5_N_SERIES=200 ... make cv-lgbm                │
└───────────────────────────────────────────────────────────────┘
            │
            ▼
┌─ edit small ──────────────────────────────────────────────────┐
│ One logical change. No drive-by refactors.                    │
└───────────────────────────────────────────────────────────────┘
            │
            ▼
┌─ inner loop (~5 s) ───────────────────────────────────────────┐
│ make test-fast    # smoke + unit                              │
│ make typecheck    # mypy on src/m5                            │
└───────────────────────────────────────────────────────────────┘
            │
            ▼
┌─ wider check (~30 s) ─────────────────────────────────────────┐
│ make test-integration                                         │
│ (model fit/predict + CV on toy data)                          │
└───────────────────────────────────────────────────────────────┘
            │
            ▼
┌─ CV diff (if model / features changed) ───────────────────────┐
│ M5_N_SERIES=500 M5_LAST_N_DAYS=200 M5_N_WINDOWS=1 make cv-lgbm│
│ Compare WRMSSE before/after; paste both into PR body.         │
└───────────────────────────────────────────────────────────────┘
            │
            ▼
┌─ ship ────────────────────────────────────────────────────────┐
│ make check        # lint + typecheck + test                   │
│ Open PR with CV diff + test plan.                             │
└───────────────────────────────────────────────────────────────┘
```

---

## Cheap commands — use these in tight loops

| Command | Time | What it covers |
|---|---|---|
| `make test-smoke` | ~1 s | imports + CLI help + package metadata |
| `make test-fast` | ~5 s | smoke + unit (toy fixture only, no model fits) |
| `make typecheck` | ~3 s | mypy on `src/m5` |
| `make lint` | ~1 s | ruff check |
| `pytest tests/unit/test_<x>.py::<test>` | < 1 s | single test |
| `pytest -m smoke` | ~1 s | every test marked `smoke` |
| `pytest -m "not slow"` | ~10 s | everything except `slow` |
| `uv run m5 --help` | < 1 s | CLI surface |

These cover ~80 % of the loop. Reach for the expensive ones below only
when you're verifying a model change.

---

## Expensive commands — only when asked / on remote hardware

| Command | Time | Why expensive |
|---|---|---|
| `make download` | ~3 min, ~250 MB | One-time fetch from `datasetsforecast` |
| `make prep` (no env caps) | minutes, ~8 GB RAM | Full 30,490-series melt |
| `make cv-stats` (full) | ~30+ min | StatsForecast over all series |
| `make cv-lgbm` (full) | ~10–20 min | LightGBM with deterministic mode |
| `make cv-hier` (full) | ~30+ min | Hierarchical forecast at 12 levels |
| `make eval` | minutes | `cv-stats + cv-lgbm + score` end-to-end |
| `make train` | minutes + disk | Persist serving artifact |

If a user asks the agent to run any of these full-throttle without
environment caps, **confirm first** — it can OOM the dev box. The
project memory under `~/.claude/projects/-home-ricka-Git-GitHub-M5/`
notes "RAM-constrained; use toy fixtures or hard-cap M5_N_SERIES /
M5_LAST_N_DAYS instead."

---

## The toy fixture

`tests/conftest.py::toy_long` is a 3-series × 200-day Nixtla-shaped
synthetic frame with weekly seasonality. Every unit and integration
test depends on it; new tests should too.

```python
def test_my_thing(toy_long: pd.DataFrame) -> None:
    # toy_long has: unique_id, ds, y (float32),
    # item_id, dept_id, cat_id, store_id, state_id, sell_price.
    # Three series: FOODS_1_001_CA_1, FOODS_1_002_CA_1, HOUSEHOLD_1_001_TX_1.
    out = my_function(toy_long)
    assert "expected_col" in out.columns
```

You can drive a full mini-pipeline against it:

```python
from m5.cv import lgbm_cv
df = build_feature_frame(toy_long)
cv_df = lgbm_cv(df, h=14, n_windows=1)   # ~5 s on a laptop
```

`tests/integration/test_pipeline.py` already does this end-to-end —
copy from there for new integration cases.

---

## Test tiers and pytest markers

`pyproject.toml` defines four markers:

| Marker | Meaning | Make target |
|---|---|---|
| `smoke` | Imports, CLI help, package metadata. Sub-second. | `make test-smoke` |
| `unit` | Pure-function tests on config / data / features / eval. | `make test-unit` |
| `integration` | Model fit/predict + CV on toy data. | `make test-integration` |
| `slow` | Wall-clock > ~5 s; skip in tight loops. | `pytest -m "not slow"` |

`make test-fast` = `smoke + unit` (no integration). Best inner-loop
target. CI runs `make check` = `lint + typecheck + test` (full suite).

---

## Per-harness setup

The repo follows the [agents.md](https://agents.md) spec — `AGENTS.md`
at the repo root is the canonical entry every modern agentic CLI
auto-discovers. Each harness below also has its own native conventions
documented here.

### Claude Code (Anthropic)

**Status:** Already configured.

- `CLAUDE.md` is auto-loaded by every Claude Code session in this repo.
- `.claude/settings.local.json` already permits `Bash(uv run *)`,
  `Bash(uv sync *)`, and a handful of git status reads.
- Useful skills enabled at the user level (`~/.claude/skills/`):
  - `/wiki` — maintain `.claude/wiki/` (this repo has one).
  - `/ai-condense` — regenerate `AI-CONTEXT.md` after material changes.
  - `/init` — bootstrap a new CLAUDE.md (don't re-run here).
  - `/review`, `/security-review`, `/simplify`, `/gh-fix-review`,
    `/gh-pr` — code-review and PR helpers.
- Recommended startup: `claude` from the repo root. The session picks
  up `CLAUDE.md`, the auto-memory under
  `~/.claude/projects/-home-ricka-Git-GitHub-M5/memory/`, and any
  per-project skills.

If you want to scope a session, point Claude at a specific module:

```bash
claude "read src/m5/models/lgbm.py and explain the lag transforms"
```

### OpenAI Codex CLI

**Status:** Works out of the box once `AGENTS.md` is committed.

```bash
# Install (Node 18+)
npm install -g @openai/codex
# or: brew install codex

# Run from the repo root
codex
```

Codex reads `AGENTS.md` hierarchically — the root file is the global
contract; you can add `tests/AGENTS.md` later if test-specific
overrides become useful. Codex respects `.gitignore` and won't crawl
`data/`.

Suggested first prompt to a fresh Codex session:

> Read `AGENTS.md` and `AI-CONTEXT.md` before doing anything. Then
> tell me which module owns WRMSSE.

Useful flags:

| Flag | Purpose |
|---|---|
| `--full-auto` | Approve tool calls automatically (sandboxed) |
| `--model gpt-5-codex` | Pick a specific model |
| `--quiet` | Trim conversational chrome |

### Gemini CLI (Google)

**Status:** Works out of the box once `GEMINI.md` is committed.

```bash
# Install (Node 20+)
npm install -g @google/gemini-cli

# Authenticate (Google account or API key)
gemini auth login

# Run from the repo root
gemini
```

`GEMINI.md` at the root is a thin pointer to `AGENTS.md` and
`AI-CONTEXT.md`. Gemini CLI supports `@path/to/file` inclusion at the
prompt — handy for surgical context loads:

```
> @AI-CONTEXT.md @src/m5/cv.py what does lgbm_cv guarantee about determinism?
```

The `--all-files` flag will fold every tracked markdown context file
into the system prompt at session start.

### Aider (FOSS, leading choice)

**Status:** Works once you point it at the agent docs.

```bash
# Install
pip install aider-chat
# or: pipx install aider-chat
# or: uv tool install aider-chat

# Run with the agent docs preloaded
aider \
  --read AGENTS.md \
  --read AI-CONTEXT.md \
  --read docs/developer/AGENTS.md \
  src/m5/<module>.py
```

Aider doesn't auto-discover `AGENTS.md`; pass it explicitly with
`--read`. To make this permanent, drop a `.aider.conf.yml` at the repo
root:

```yaml
# .aider.conf.yml
model: sonnet                        # or gpt-5, deepseek/deepseek-chat, etc.
read:
  - AGENTS.md
  - AI-CONTEXT.md
  - docs/developer/AGENTS.md
auto-test: true
test-cmd: make test-fast
lint-cmd:
  python: uv run ruff check --fix
```

Aider has first-class support for many models including local ones via
Ollama / LM Studio — see the project's docs. Inside the session,
`/test` runs `test-cmd`, `/lint` runs `lint-cmd`, `/diff` shows the
working diff before commit.

### Other agentic harnesses (brief)

| Tool | Reads | Notes |
|---|---|---|
| **opencode** (`opencode-ai/opencode`) | `AGENTS.md` (auto) | Fully agents.md-compliant. FOSS, TypeScript. |
| **continue.dev** | `.continue/config.json` | VSCode + JetBrains extension. Add a system prompt that includes `AI-CONTEXT.md` contents. |
| **Cursor** | `.cursorrules` or `.cursor/rules/*.mdc` | Copy the contents of `AGENTS.md` into `.cursorrules` if you want auto-load. |
| **factory.ai droids** | `AGENTS.md` (auto) | Cloud agent platform; agents.md-compliant. |
| **Cline / Roo Code** | VSCode settings | System prompt = paste `AGENTS.md` body. |
| **Anything with a system-prompt slot** | n/a | Paste `AGENTS.md`, attach `AI-CONTEXT.md`. |

For local/FOSS-first setups, **aider** is the most polished. **opencode**
is the closest equivalent in the agents.md ecosystem and is fully open
source.

---

## Contributing findings (PR flow)

The user's working branch is `ai`; PRs target `main`.

### Before opening

```bash
make check          # lint + typecheck + test (CI entry point)
```

For changes touching model code, features, evaluation, or CV:

```bash
# Capped CV diff — paste both numbers into the PR body
M5_N_SERIES=500 M5_LAST_N_DAYS=200 M5_N_WINDOWS=1 make cv-lgbm
uv run python -c "
import pandas as pd
from m5.evaluation import compute_components, wrmsse_for_models
cv = pd.read_parquet('artifacts/cv_lgbm.parquet')
df = pd.read_parquet('data/processed/long.parquet')
comps = compute_components(df[df['ds'] < cv['ds'].min()])
truth = cv[['unique_id','ds','y']]
print(wrmsse_for_models(truth, cv, comps))
"
```

### PR body conventions

```markdown
## Summary
- One-line statement of what changed.

## Why
- The motivation. If the change touches the feature menu, link the
  CV diff that justifies the column.

## CV diff (capped: N_SERIES=500, LAST_N_DAYS=200, N_WINDOWS=1)
| Model | WRMSSE before | WRMSSE after |
|---|---|---|
| LGBM  | 0.xxx         | 0.xxx        |

## Test plan
- [x] make test-fast
- [x] make test-integration
- [x] make typecheck
- [ ] make check (full)
```

### When extending features (one of the most common asks)

`CLAUDE.md` already lists the touch points; the agent must update
**all** of them or the feature won't reach the model:

1. `src/m5/features.py` — add the function.
2. `build_feature_frame` — chain it in.
3. `src/m5/models/lgbm.py::fit_predict_lgbm` — add to `keep_cols`.
4. `src/m5/cv.py::lgbm_cv` — add to `keep_cols`.
5. `tests/unit/test_features.py` — add a test using `toy_long`.
6. CV diff in the PR body.

If any of those steps are skipped, the feature silently no-ops at
training time.

### When adding a model

`CLAUDE.md` covers this:

1. `src/m5/models/<x>.py` — `build_<x>_forecaster` + `fit_predict_<x>`,
   mirroring `stats.py` / `lgbm.py`.
2. `src/m5/models/__init__.py` — re-export.
3. `src/m5/cv.py` — `<x>_cv` runner; **always call `set_global_seed()`
   first**.
4. `src/m5/cli.py` — wire into the `cv` and `forecast` `if/elif` chain.
5. `Makefile` — add `cv-<x>` and `forecast-<x>` targets.
6. `tests/integration/test_models.py` — smoke fit/predict on `toy_long`.

---

## Worked examples

### Example 1 — "Why does `cv-lgbm` give a different score on rerun?"

The agent's first move should be to confirm the determinism contract
holds. The expected behavior is **bit-identical** WRMSSE between runs.

```bash
# Run twice with the same caps
M5_N_SERIES=200 M5_LAST_N_DAYS=120 M5_N_WINDOWS=1 make cv-lgbm
mv artifacts/cv_lgbm.parquet artifacts/cv_lgbm_run1.parquet
M5_N_SERIES=200 M5_LAST_N_DAYS=120 M5_N_WINDOWS=1 make cv-lgbm
diff <(uv run python -c "import pandas as pd; print(pd.read_parquet('artifacts/cv_lgbm_run1.parquet').head(20))") \
     <(uv run python -c "import pandas as pd; print(pd.read_parquet('artifacts/cv_lgbm.parquet').head(20))")
```

If the diff is empty, determinism holds. If not, suspect:

- A new randomness source not threaded through `set_global_seed()`.
- A LightGBM param overriding `deterministic=True` (`feature_fraction`
  with a per-iteration random seed, multi-threading without
  `force_row_wise=True`).
- A dependency upgrade in `uv.lock` that loosened determinism.

### Example 2 — "Add a `is_holiday_eve` feature"

Repo tenet: **fewer features, not more.** The agent should:

1. Implement the feature behind a small function in `features.py`.
2. Run a CV diff (capped) to show it actually moves WRMSSE.
3. Only if the diff is favorable, wire it through `build_feature_frame`,
   the two `keep_cols` lists, and tests.
4. If the diff is flat or worse, the change does **not** ship.

A PR that adds a feature without a CV diff is incomplete by repo
convention.

### Example 3 — "Add a new model family (e.g. `models/torch_lstm.py`)"

Follow the recipe under "When adding a model" above. Determinism for
PyTorch needs:

```python
import torch
torch.manual_seed(SETTINGS.seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Add `torch` to the `[project]` deps in `pyproject.toml` and run
`uv sync`. Don't pin a CUDA build into the lockfile unless the user
explicitly wants GPU.

---

## Common pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: m5` in a notebook | Wrong kernel | Pick **Python (m5)**; `make install` re-registers |
| Tests pass locally, fail CI | Hardcoded path / wall-clock seed | Use `SETTINGS.data_dir` and `set_global_seed()` |
| `cv-lgbm` OOMs the box | Full data | Cap with `M5_N_SERIES=200 M5_LAST_N_DAYS=120 M5_N_WINDOWS=1` |
| Two CV runs disagree | Determinism bug | See Example 1 above |
| `mlforecast: num_threads must be -1 or a positive integer` | Old version | `uv sync --upgrade-package mlforecast` |
| New feature doesn't change WRMSSE | Not in `keep_cols` | Add to `models/lgbm.py` AND `cv.py` |
| Pre-commit fails on commit | Hook flagged something | Fix and create a **new** commit; never `--no-verify` |
| `data/m5/datasets/` empty | Data not downloaded | `make download` (~250 MB, one-time) |
| `command not found: uv` after bootstrap | Shell hasn't reloaded | `source ~/.local/bin/env` or new terminal |
| `AI-CONTEXT.md` and source disagree | File was hand-edited | Re-run `/ai-condense` to regenerate |

Full troubleshooting matrix: [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

---

## What NOT to touch without confirming with the user

| Path | Why |
|---|---|
| `AI-CONTEXT.md` | Regenerated by `/ai-condense`; manual edits get clobbered |
| `uv.lock` | Use `uv add` / `uv sync --upgrade-package`; never hand-edit |
| `data/m5/datasets/*` | Raw Kaggle data, immutable |
| `.pre-commit-config.yaml` | Mirrors CI; loosening it desynchronizes them |
| `.github/workflows/*` | CI definition |
| `WriteUp.md` | Canonical methodology — touched in dedicated docs PRs only |
| `CLAUDE.md` / `AGENTS.md` / this file | Conventions; touched in dedicated docs PRs |
| `cloud/terraform/*` | Infra; needs review for cost / blast radius |

When in doubt, the agent should ask before editing any of the above.

---

## Glossary

| Term | Meaning |
|---|---|
| **WRMSSE** | Weighted Root Mean Squared Scaled Error. M5 official metric. Lower is better. Bottom-level only in `m5.evaluation`; multi-level via `m5.hierarchy`. |
| **Nixtla long frame** | Schema convention: `unique_id, ds, y` + statics + time-varying covariates. Used by `statsforecast`, `mlforecast`, `hierarchicalforecast`. |
| **`unique_id`** | Series id = `f"{item_id}_{store_id}"`. 30,490 of them. |
| **bottom level** | Item × store granularity. M5 has 12 hierarchy levels; we score at the bottom by default. |
| **toy fixture** | `tests/conftest.py::toy_long` — 3 series × 200 days, deterministic, no data download required. |
| **Tweedie** | LightGBM objective for sparse count-like retail data. Set with `objective="tweedie", tweedie_variance_power=1.1`. |
| **rolling-origin CV** | Walk-forward CV: `n_windows` windows of length `h`, stepping `step_size` days each. Implemented by `StatsForecast.cross_validation` and `MLForecast.cross_validation`. |
| **`SETTINGS`** | The frozen dataclass in `m5.config` that owns paths/seed/horizon. Read once per process from `.env`. |
| **`set_global_seed()`** | Seeds `random`, NumPy, and `PYTHONHASHSEED`. Called at the top of every CV / forecast / train entry. |
| **`mlforecast`** | Nixtla's recursive ML wrapper. Owns lags / rolling features for the LightGBM model. |
| **`statsforecast`** | Nixtla's univariate stats wrapper. Owns Theta + AutoETS + SeasonalNaive. |
| **`hierarchicalforecast`** | Nixtla's hierarchical reconciler. Powers `models/hierarchical.py` (BU / TD / MinT). |

---

## See also

- [`AGENTS.md`](../../AGENTS.md) — root cross-tool agent contract.
- [`CLAUDE.md`](../../CLAUDE.md) — Claude Code's auto-loaded conventions.
- [`AI-CONTEXT.md`](../../AI-CONTEXT.md) — token-condensed full repo context.
- [`README.md`](../../README.md) — happy path.
- [`docs/developer/SETUP.md`](SETUP.md) — first-time install.
- [`docs/developer/DEVELOPMENT.md`](DEVELOPMENT.md) — daily workflow.
- [`docs/developer/ARCHITECTURE.md`](ARCHITECTURE.md) — module map + data flow.
- [`docs/developer/TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — error matrix.
- [`WriteUp.md`](../../WriteUp.md) — methodology.
- [agents.md spec](https://agents.md) — the cross-tool convention this repo follows.
