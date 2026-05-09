#!/usr/bin/env bash
# Bootstrap a one-shot M5 training VM.
#
# Lifecycle:
#   1. Install OS deps (git, curl, make, libgomp1) + uv + the per-cloud CLI
#      that matches M5_ARTIFACT_DEST (gcloud / aws / az).
#   2. Clone the M5 repo and checkout the requested ref.
#   3. uv-sync (no dev / no notebook groups — keep the box lean).
#   4. Run `m5 download → prep → cv stats → cv lgbm → score → train`.
#   5. Push the model artifact AND the score report to object storage.
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
: "${M5_RUN_CV:=true}"            # opt-out: set to false to skip cv + score
: "${M5_CV_N_WINDOWS:=3}"

REPO_DIR=/srv/M5
UV_BIN=/root/.local/bin/uv

# ---- OS deps + uv --------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    git curl ca-certificates make jq libgomp1 unzip apt-transport-https gnupg

# Install the object-storage CLI that matches the URI scheme of the artifact dest.
case "$M5_ARTIFACT_DEST" in
    gs://*)
        echo "==> installing google-cloud-cli (for gs:// push)"
        curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
            | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
            > /etc/apt/sources.list.d/google-cloud-sdk.list
        apt-get update -y
        apt-get install -y --no-install-recommends google-cloud-cli
        ;;
    s3://*)
        echo "==> installing awscli (for s3:// push)"
        apt-get install -y --no-install-recommends awscli
        ;;
    az://*)
        echo "==> installing azure-cli (for az:// push)"
        curl -sL https://aka.ms/InstallAzureCLIDeb | bash
        ;;
    *)
        echo "WARN: unknown M5_ARTIFACT_DEST scheme: $M5_ARTIFACT_DEST" >&2
        ;;
esac

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
echo "==> $(date -Is) m5-train: download"
"$UV_BIN" run m5 download
echo "==> $(date -Is) m5-train: prep (last_n_days=$M5_LAST_N_DAYS, n_series=$M5_N_SERIES)"
"$UV_BIN" run m5 prep \
    --last-n-days "$M5_LAST_N_DAYS" \
    --n-series "$M5_N_SERIES"

if [ "$M5_RUN_CV" = "true" ]; then
    echo "==> $(date -Is) m5-train: cv stats (h=$M5_HORIZON, n_windows=$M5_CV_N_WINDOWS)"
    "$UV_BIN" run m5 cv stats --horizon "$M5_HORIZON" --n-windows "$M5_CV_N_WINDOWS"
    echo "==> $(date -Is) m5-train: cv lgbm (h=$M5_HORIZON, n_windows=$M5_CV_N_WINDOWS)"
    "$UV_BIN" run m5 cv lgbm  --horizon "$M5_HORIZON" --n-windows "$M5_CV_N_WINDOWS"
    echo "==> $(date -Is) m5-train: score"
    "$UV_BIN" run m5 score -m cv_stats -m cv_lgbm
fi

echo "==> $(date -Is) m5-train: train (final fit on full data)"
"$UV_BIN" run m5 train --horizon "$M5_HORIZON"

# ---- push model artifact -------------------------------------------------
ARTIFACT_DIR=$(readlink -f artifacts/models/lgbm/latest)
TIMESTAMP=$(basename "$ARTIFACT_DIR")
DEST="${M5_ARTIFACT_DEST%/}/$TIMESTAMP"
LATEST_DEST="${M5_ARTIFACT_DEST%/}/latest"

echo "==> pushing $ARTIFACT_DIR -> $DEST"
bash cloud/scripts/push_artifact.sh "$ARTIFACT_DIR" "$DEST"

# Mirror to a stable "latest" prefix so serve VMs don't need to know the timestamp.
echo "==> pushing $ARTIFACT_DIR -> $LATEST_DEST (stable alias)"
bash cloud/scripts/push_artifact.sh "$ARTIFACT_DIR" "$LATEST_DEST"

# ---- push score report (if cv ran) ---------------------------------------
if [ "$M5_RUN_CV" = "true" ] && [ -d reports ]; then
    REPORT_DEST="${M5_ARTIFACT_DEST%/}/reports/$TIMESTAMP"
    REPORT_LATEST="${M5_ARTIFACT_DEST%/}/reports/latest"
    echo "==> pushing reports/ -> $REPORT_DEST"
    bash cloud/scripts/push_artifact.sh reports "$REPORT_DEST"
    echo "==> pushing reports/ -> $REPORT_LATEST (stable alias)"
    bash cloud/scripts/push_artifact.sh reports "$REPORT_LATEST"
fi

# Also push the raw cv parquets so we can re-score offline.
if [ "$M5_RUN_CV" = "true" ] && compgen -G "artifacts/cv_*.parquet" > /dev/null; then
    CV_DIR=$(mktemp -d)
    cp artifacts/cv_*.parquet "$CV_DIR/"
    CV_DEST="${M5_ARTIFACT_DEST%/}/cv/$TIMESTAMP"
    CV_LATEST="${M5_ARTIFACT_DEST%/}/cv/latest"
    echo "==> pushing artifacts/cv_*.parquet -> $CV_DEST"
    bash cloud/scripts/push_artifact.sh "$CV_DIR" "$CV_DEST"
    echo "==> pushing artifacts/cv_*.parquet -> $CV_LATEST (stable alias)"
    bash cloud/scripts/push_artifact.sh "$CV_DIR" "$CV_LATEST"
fi

echo "==> $(date -Is) m5-train: complete (timestamp=$TIMESTAMP)"
echo "$TIMESTAMP" > /srv/M5/.train-complete

# ---- self-destruct -------------------------------------------------------
if [ "$M5_TRAIN_SHUTDOWN_ON_DONE" = "true" ]; then
    echo "==> shutting down (M5_TRAIN_SHUTDOWN_ON_DONE=true)"
    # Sleep so cloud-init has time to record completion before the box dies.
    sleep 30
    /sbin/poweroff
fi
