#!/usr/bin/env bash
# Thin wrapper that just calls the CLI. Useful for cron / CI jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run m5 download
