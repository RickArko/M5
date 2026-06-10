#!/usr/bin/env bash
# Bootstrap a one-shot M5 training VM.
#
# Lifecycle:
#   1. Install OS deps (git, curl, make, libgomp1) + uv + the per-cloud CLI
#      that matches M5_ARTIFACT_DEST (gcloud / aws / az).
#   2. Clone the M5 repo and checkout the requested ref.
#   3. uv-sync (no dev / no notebook groups — keep the box lean).
#   4. Run `m5 download → prep`, selected CV/recipe jobs, optional score,
#      and optional final `m5 train`.
#   5. Push model/report/CV artifacts plus a timestamped run bundle to object storage.
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
#   M5_HORIZON, M5_RUN_*, M5_CV_*, M5_SCORE_MODELS,
#   M5_TRAIN_SHUTDOWN_ON_DONE, M5_OBJECT_STORE_ENDPOINT
[ -f /etc/m5-cloud.env ] && set -a && source /etc/m5-cloud.env && set +a

: "${M5_GIT_REPO:=https://github.com/RickArko/M5.git}"
: "${M5_GIT_REF:=main}"
: "${M5_ARTIFACT_DEST:?M5_ARTIFACT_DEST required (e.g. s3://my-bucket/m5/lgbm or gs://... / az://...)}"
: "${M5_LAST_N_DAYS:=400}"
: "${M5_N_SERIES:=-1}"
: "${M5_HORIZON:=28}"
: "${M5_TRAIN_SHUTDOWN_ON_DONE:=true}"
: "${M5_RUN_ID:=}"
: "${M5_RUN_CV:=true}"            # legacy opt-out: false skips stats/lgbm CV by default
: "${M5_RUN_STATS_CV:=$M5_RUN_CV}"
: "${M5_RUN_LGBM_CV:=$M5_RUN_CV}"
: "${M5_RUN_HIER_CV:=false}"
: "${M5_CV_RECIPE:=}"
: "${M5_CV_N_WINDOWS:=3}"
: "${M5_SCORE_MODELS:=stats lgbm}"
: "${M5_RUN_TRAIN:=true}"
: "${M5_PUSH_PROCESSED:=false}"

if [ -z "$M5_RUN_ID" ]; then
    M5_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi

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

if [ "$M5_RUN_STATS_CV" = "true" ]; then
    echo "==> $(date -Is) m5-train: cv stats (h=$M5_HORIZON, n_windows=$M5_CV_N_WINDOWS)"
    "$UV_BIN" run m5 cv stats --horizon "$M5_HORIZON" --n-windows "$M5_CV_N_WINDOWS"
fi

if [ "$M5_RUN_LGBM_CV" = "true" ]; then
    echo "==> $(date -Is) m5-train: cv lgbm (h=$M5_HORIZON, n_windows=$M5_CV_N_WINDOWS)"
    "$UV_BIN" run m5 cv lgbm  --horizon "$M5_HORIZON" --n-windows "$M5_CV_N_WINDOWS"
fi

if [ "$M5_RUN_HIER_CV" = "true" ]; then
    echo "==> $(date -Is) m5-train: cv hier (h=$M5_HORIZON, n_windows=$M5_CV_N_WINDOWS)"
    "$UV_BIN" run m5 cv hier --horizon "$M5_HORIZON" --n-windows "$M5_CV_N_WINDOWS"
fi

if [ -n "$M5_CV_RECIPE" ]; then
    echo "==> $(date -Is) m5-train: cv recipe $M5_CV_RECIPE (h=$M5_HORIZON, n_windows=$M5_CV_N_WINDOWS)"
    "$UV_BIN" run m5 cv-recipe "$M5_CV_RECIPE" --horizon "$M5_HORIZON" --n-windows "$M5_CV_N_WINDOWS"
fi

if compgen -G "artifacts/cv_*.parquet" > /dev/null && [ -n "$M5_SCORE_MODELS" ]; then
    echo "==> $(date -Is) m5-train: score models=[$M5_SCORE_MODELS]"
    score_cmd=("$UV_BIN" run m5 score --out reports --run-id "$M5_RUN_ID")
    for model in $M5_SCORE_MODELS; do
        score_cmd+=(--model "$model")
    done
    "${score_cmd[@]}"
else
    echo "==> $(date -Is) m5-train: score skipped"
fi

MODEL_TIMESTAMP="$M5_RUN_ID"
if [ "$M5_RUN_TRAIN" = "true" ]; then
    echo "==> $(date -Is) m5-train: train (final fit on full data)"
    "$UV_BIN" run m5 train --horizon "$M5_HORIZON"

    # ---- push model artifact ---------------------------------------------
    ARTIFACT_DIR=$(readlink -f artifacts/models/lgbm/latest)
    MODEL_TIMESTAMP=$(basename "$ARTIFACT_DIR")
    DEST="${M5_ARTIFACT_DEST%/}/$MODEL_TIMESTAMP"
    LATEST_DEST="${M5_ARTIFACT_DEST%/}/latest"

    echo "==> pushing $ARTIFACT_DIR -> $DEST"
    bash cloud/scripts/push_artifact.sh "$ARTIFACT_DIR" "$DEST"

    # Mirror to a stable "latest" prefix so serve VMs don't need to know the timestamp.
    echo "==> pushing $ARTIFACT_DIR -> $LATEST_DEST (stable alias)"
    bash cloud/scripts/push_artifact.sh "$ARTIFACT_DIR" "$LATEST_DEST"
else
    echo "==> $(date -Is) m5-train: final train skipped (M5_RUN_TRAIN=false)"
fi

# ---- push score report (if cv ran) ---------------------------------------
if [ -d reports ]; then
    REPORT_DEST="${M5_ARTIFACT_DEST%/}/reports/$MODEL_TIMESTAMP"
    REPORT_LATEST="${M5_ARTIFACT_DEST%/}/reports/latest"
    echo "==> pushing reports/ -> $REPORT_DEST"
    bash cloud/scripts/push_artifact.sh reports "$REPORT_DEST"
    echo "==> pushing reports/ -> $REPORT_LATEST (stable alias)"
    bash cloud/scripts/push_artifact.sh reports "$REPORT_LATEST"
fi

# Also push the raw cv parquets so we can re-score offline.
if compgen -G "artifacts/cv_*.parquet" > /dev/null; then
    CV_DIR=$(mktemp -d)
    cp artifacts/cv_*.parquet "$CV_DIR/"
    CV_DEST="${M5_ARTIFACT_DEST%/}/cv/$MODEL_TIMESTAMP"
    CV_LATEST="${M5_ARTIFACT_DEST%/}/cv/latest"
    echo "==> pushing artifacts/cv_*.parquet -> $CV_DEST"
    bash cloud/scripts/push_artifact.sh "$CV_DIR" "$CV_DEST"
    echo "==> pushing artifacts/cv_*.parquet -> $CV_LATEST (stable alias)"
    bash cloud/scripts/push_artifact.sh "$CV_DIR" "$CV_LATEST"
fi

# ---- push full run bundle ------------------------------------------------
RUN_BUNDLE=$(mktemp -d)
mkdir -p "$RUN_BUNDLE/metadata"
if [ -d artifacts ]; then
    cp -a artifacts "$RUN_BUNDLE/artifacts"
fi
if [ -d reports ]; then
    cp -a reports "$RUN_BUNDLE/reports"
fi
if [ -d forecasts ]; then
    cp -a forecasts "$RUN_BUNDLE/forecasts"
fi
if [ "$M5_PUSH_PROCESSED" = "true" ] && [ -f data/processed/long.parquet ]; then
    mkdir -p "$RUN_BUNDLE/data"
    cp data/processed/long.parquet "$RUN_BUNDLE/data/long.parquet"
fi

jq -n \
    --arg run_id "$M5_RUN_ID" \
    --arg model_timestamp "$MODEL_TIMESTAMP" \
    --arg git_repo "$M5_GIT_REPO" \
    --arg git_ref "$M5_GIT_REF" \
    --arg artifact_dest "$M5_ARTIFACT_DEST" \
    --arg horizon "$M5_HORIZON" \
    --arg last_n_days "$M5_LAST_N_DAYS" \
    --arg n_series "$M5_N_SERIES" \
    --arg cv_n_windows "$M5_CV_N_WINDOWS" \
    --arg run_stats_cv "$M5_RUN_STATS_CV" \
    --arg run_lgbm_cv "$M5_RUN_LGBM_CV" \
    --arg run_hier_cv "$M5_RUN_HIER_CV" \
    --arg cv_recipe "$M5_CV_RECIPE" \
    --arg score_models "$M5_SCORE_MODELS" \
    --arg run_train "$M5_RUN_TRAIN" \
    --arg push_processed "$M5_PUSH_PROCESSED" \
    '{
      run_id: $run_id,
      model_timestamp: $model_timestamp,
      git_repo: $git_repo,
      git_ref: $git_ref,
      artifact_dest: $artifact_dest,
      horizon: ($horizon | tonumber),
      last_n_days: ($last_n_days | tonumber),
      n_series: ($n_series | tonumber),
      cv_n_windows: ($cv_n_windows | tonumber),
      run_stats_cv: ($run_stats_cv == "true"),
      run_lgbm_cv: ($run_lgbm_cv == "true"),
      run_hier_cv: ($run_hier_cv == "true"),
      cv_recipe: $cv_recipe,
      score_models: $score_models,
      run_train: ($run_train == "true"),
      push_processed: ($push_processed == "true")
    }' > "$RUN_BUNDLE/metadata/run.json"

RUN_DEST="${M5_ARTIFACT_DEST%/}/runs/$M5_RUN_ID"
RUN_LATEST="${M5_ARTIFACT_DEST%/}/runs/latest"
echo "==> pushing run bundle -> $RUN_DEST"
bash cloud/scripts/push_artifact.sh "$RUN_BUNDLE" "$RUN_DEST"
echo "==> pushing run bundle -> $RUN_LATEST (stable alias)"
bash cloud/scripts/push_artifact.sh "$RUN_BUNDLE" "$RUN_LATEST"

echo "==> $(date -Is) m5-train: complete (run_id=$M5_RUN_ID, model_timestamp=$MODEL_TIMESTAMP)"
echo "$M5_RUN_ID" > /srv/M5/.train-complete

# ---- self-destruct -------------------------------------------------------
if [ "$M5_TRAIN_SHUTDOWN_ON_DONE" = "true" ]; then
    echo "==> shutting down (M5_TRAIN_SHUTDOWN_ON_DONE=true)"
    # Sleep so cloud-init has time to record completion before the box dies.
    sleep 30
    /sbin/poweroff
fi
