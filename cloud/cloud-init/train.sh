#!/usr/bin/env bash
# Bootstrap a one-shot M5 training VM.
#
# Lifecycle:
#   1. Install OS deps (git, curl, make, libgomp1) + uv.
#   2. Clone the M5 repo and checkout the requested ref.
#   3. uv-sync (no dev / no notebook groups — keep the box lean).
#   4. Run `m5 download && m5 prep && m5 train`.
#   5. Push the resulting artifact directory to object storage.
#   6. (Optional) `poweroff` so the VM doesn't keep billing.
#
# Inputs are read from /etc/m5-cloud.env which Terraform writes during boot.
# Logs to /var/log/m5-train.log so `journalctl` and `cloud-init analyze` both see it.

set -euo pipefail
exec > >(tee -a /var/log/m5-train.log) 2>&1
echo "==> $(date -Is) m5-train: starting"

# ---- env -----------------------------------------------------------------
# /etc/m5-cloud.env is rendered by Terraform with:
#   M5_GIT_REPO, M5_GIT_REF, M5_ARTIFACT_DEST, M5_LAST_N_DAYS, M5_N_SERIES,
#   M5_HORIZON, M5_TRAIN_SHUTDOWN_ON_DONE, M5_OBJECT_STORE_ENDPOINT
[ -f /etc/m5-cloud.env ] && set -a && source /etc/m5-cloud.env && set +a

: "${M5_GIT_REPO:=https://github.com/RickArko/M5.git}"
: "${M5_GIT_REF:=main}"
: "${M5_ARTIFACT_DEST:?M5_ARTIFACT_DEST required (e.g. s3://my-bucket/m5/lgbm or gs://... / az://...)}"
: "${M5_LAST_N_DAYS:=400}"
: "${M5_N_SERIES:=-1}"
: "${M5_HORIZON:=28}"
: "${M5_TRAIN_SHUTDOWN_ON_DONE:=true}"

REPO_DIR=/srv/M5
UV_BIN=/root/.local/bin/uv

# ---- OS deps + uv --------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    git curl ca-certificates make jq libgomp1 awscli unzip

if [ ! -x "$UV_BIN" ]; then
    echo "==> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:$PATH"

# ---- clone repo ----------------------------------------------------------
mkdir -p "$(dirname "$REPO_DIR")"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$M5_GIT_REPO" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch --all --tags --prune
git checkout "$M5_GIT_REF"
git pull --ff-only origin "$M5_GIT_REF" || true

# ---- python deps ---------------------------------------------------------
"$UV_BIN" sync --no-group dev --no-group notebook

# ---- pipeline ------------------------------------------------------------
"$UV_BIN" run m5 download
"$UV_BIN" run m5 prep \
    --last-n-days "$M5_LAST_N_DAYS" \
    --n-series "$M5_N_SERIES"
"$UV_BIN" run m5 train --horizon "$M5_HORIZON"

# ---- push artifact -------------------------------------------------------
ARTIFACT_DIR=$(readlink -f artifacts/models/lgbm/latest)
TIMESTAMP=$(basename "$ARTIFACT_DIR")
DEST="${M5_ARTIFACT_DEST%/}/$TIMESTAMP"
LATEST_DEST="${M5_ARTIFACT_DEST%/}/latest"

echo "==> pushing $ARTIFACT_DIR -> $DEST"
bash cloud/scripts/push_artifact.sh "$ARTIFACT_DIR" "$DEST"

# Mirror to a stable "latest" prefix so serve VMs don't need to know the timestamp.
echo "==> pushing $ARTIFACT_DIR -> $LATEST_DEST (stable alias)"
bash cloud/scripts/push_artifact.sh "$ARTIFACT_DIR" "$LATEST_DEST"

echo "==> $(date -Is) m5-train: complete (timestamp=$TIMESTAMP)"
echo "$TIMESTAMP" > /srv/M5/.train-complete

# ---- self-destruct -------------------------------------------------------
if [ "$M5_TRAIN_SHUTDOWN_ON_DONE" = "true" ]; then
    echo "==> shutting down (M5_TRAIN_SHUTDOWN_ON_DONE=true)"
    # Sleep so cloud-init has time to record completion before the box dies.
    sleep 30
    /sbin/poweroff
fi
