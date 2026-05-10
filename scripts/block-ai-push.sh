#!/usr/bin/env bash
# Local-only guard — refuses `git push` while the current branch is `ai`.
#
# Wired in via .pre-commit-config.yaml at the pre-push stage. Triggered by every
# `git push` invocation that runs the pre-commit pre-push hook (i.e. once
# `pre-commit install --hook-type pre-push` has run via `make install`).
#
# Bypass once:   M5_ALLOW_AI_PUSH=1 git push
# Remove guard:  drop the `block-ai-branch-push` hook from .pre-commit-config.yaml
#
# Rationale: the `ai` branch is intentionally local-only until the FastAPI
# scaffold work merges. A loud failure here is preferable to an accidental push
# that publishes work-in-progress.

set -euo pipefail

branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "${branch}" == "ai" && -z "${M5_ALLOW_AI_PUSH:-}" ]]; then
    cat <<'EOF' >&2

  ⛔  Pushes from the `ai` branch are blocked (local-only mode).

      Bypass once:   M5_ALLOW_AI_PUSH=1 git push
      Remove guard:  edit .pre-commit-config.yaml — delete the
                     `block-ai-branch-push` hook block

EOF
    exit 1
fi
