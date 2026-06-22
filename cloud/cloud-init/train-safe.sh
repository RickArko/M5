#!/usr/bin/env bash
# Bootstrap a one-shot M5 training VM with robust error handling and monitoring.
#
# Key improvements over original:
#   - Trap errors and always shutdown (prevents idle VMs on failure)
#   - Monitor process health (kill runaway processes)
#   - Disk space checks
#   - Detailed logging of every step
#   - Push artifacts even on partial failure
#   - Timeout protection for long-running steps
#
# Logs: /var/log/m5-train.log (stdout+stderr)
# Health: /var/log/m5-health.log (periodic checks)
# Errors: /var/log/m5-errors.log (captured errors)

set -euo pipefail

# Error handler: always shutdown, even on failure
error_handler() {
    local exit_code=$?
    local line=$1
    echo "==> $(date -Is) m5-train: ERROR at line $line (exit code $exit_code)" | tee -a /var/log/m5-errors.log

    # Push whatever artifacts we have
    echo "==> $(date -Is) m5-train: pushing partial artifacts before shutdown"
    bash "$REPO_DIR/cloud/scripts/push_artifact.sh" "$REPO_DIR/artifacts" "${M5_ARTIFACT_DEST%/}/partial-$(date -u +%Y%m%dT%H%M%SZ)" || true

    # Write failure marker
    echo "FAILED:$exit_code" > /srv/M5/.train-complete

    # Always shutdown to prevent idle billing
    if [ "${M5_TRAIN_SHUTDOWN_ON_DONE:-true}" = "true" ]; then
        echo "==> $(date -Is) m5-train: shutting down after error"
        sleep 30
        /sbin/poweroff
    fi

    exit $exit_code
}
trap 'error_handler ${LINENO}' ERR

# Start logging
exec > >(tee -a /var/log/m5-train.log) 2>&1
echo "==> $(date -Is) m5-train: starting with error protection"

# ---- env -----------------------------------------------------------------
[ -f /etc/m5-cloud.env ] && set -a && source /etc/m5-cloud.env && set +a

: "${M5_GIT_REPO:=https://github.com/RickArko/M5.git}"
: "${M5_GIT_REF:=main}"
: "${M5_ARTIFACT_DEST:?M5_ARTIFACT_DEST required}"
: "${M5_LAST_N_DAYS:=400}"
: "${M5_N_SERIES:=-1}"
: "${M5_HORIZON:=28}"
: "${M5_TRAIN_SHUTDOWN_ON_DONE:=true}"
: "${M5_RUN_ID:=}"
: "${M5_RUN_CV:=true}"
: "${M5_RUN_STATS_CV:=$M5_RUN_CV}"
: "${M5_RUN_LGBM_CV:=$M5_RUN_CV}"
: "${M5_RUN_HIER_CV:=false}"
: "${M5_CV_RECIPE:=}"
: "${M5_CV_N_WINDOWS:=3}"
: "${M5_SCORE_MODELS:=stats lgbm}"
: "${M5_RUN_TRAIN:=true}"
: "${M5_PUSH_PROCESSED:=false}"
: "${M5_MAX_IDLE_MINUTES:=60}"
: "${M5_DISK_MIN_GB:=10}"

if [ -z "$M5_RUN_ID" ]; then
    M5_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi

REPO_DIR=/srv/M5
UV_BIN=/root/.local/bin/uv
LOG_FILE=/var/log/m5-train.log
HEALTH_LOG=/var/log/m5-health.log

# ---- health monitoring background process --------------------------------
health_monitor() {
    while true; do
        sleep 300  # 5 minutes

        # Check disk space
        DISK_AVAIL=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
        if [ "$DISK_AVAIL" -lt "$M5_DISK_MIN_GB" ]; then
            echo "$(date -Is) HEALTH: DISK LOW ($DISK_AVAIL GB < $M5_DISK_MIN_GB GB)" >> "$HEALTH_LOG"
            echo "$(date -Is) HEALTH: DISK LOW - may cause failures" >> "$LOG_FILE"
        fi

        # Check memory
        MEM_PCT=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100)}')
        if [ "$MEM_PCT" -gt 95 ]; then
            echo "$(date -Is) HEALTH: MEMORY HIGH ($MEM_PCT%)" >> "$HEALTH_LOG"
        fi

        # Check if main process is still running
        if ! pgrep -f "m5 (cv|score|train|download|prep)" > /dev/null; then
            echo "$(date -Is) HEALTH: No m5 process found" >> "$HEALTH_LOG"
        fi

        # Check idle time (if no m5 process running for M5_MAX_IDLE_MINUTES)
        if [ -f /srv/M5/.last-activity ]; then
            LAST_ACTIVITY=$(cat /srv/M5/.last-activity)
            CURRENT_TIME=$(date +%s)
            IDLE_MINUTES=$(( (CURRENT_TIME - LAST_ACTIVITY) / 60 ))
            if [ "$IDLE_MINUTES" -gt "$M5_MAX_IDLE_MINUTES" ]; then
                echo "$(date -Is) HEALTH: IDLE FOR $IDLE_MINUTES MINUTES - SHUTTING DOWN" >> "$HEALTH_LOG"
                echo "$(date -Is) m5-train: idle timeout - shutting down" >> "$LOG_FILE"
                echo "IDLE:$IDLE_MINUTES" > /srv/M5/.train-complete
                sleep 30
                /sbin/poweroff
            fi
        fi

        echo "$(date -Is) HEALTH: disk=${DISK_AVAIL}GB mem=${MEM_PCT}% idle=${IDLE_MINUTES}m" >> "$HEALTH_LOG"
    done
}

# Start health monitor in background
health_monitor &
HEALTH_PID=$!

# Update activity timestamp function
update_activity() {
    date +%s > /srv/M5/.last-activity
}

# ---- OS deps + uv --------------------------------------------------------
update_activity
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    git curl ca-certificates make jq libgomp1 unzip apt-transport-https gnupg

# Install the object-storage CLI that matches the URI scheme.
case "$M5_ARTIFACT_DEST" in
    gs://*)
        echo "==> $(date -Is) installing google-cloud-cli"
        curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
            | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
        echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
            > /etc/apt/sources.list.d/google-cloud-sdk.list
        apt-get update -y
        apt-get install -y --no-install-recommends google-cloud-cli
        ;;
    s3://*)
        echo "==> $(date -Is) installing awscli v2"
        curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
        unzip -q /tmp/awscliv2.zip -d /tmp
        /tmp/aws/install --update
        rm -rf /tmp/awscliv2.zip /tmp/aws
        ;;
    az://*)
        echo "==> $(date -Is) installing azure-cli"
        curl -sL https://aka.ms/InstallAzureCLIDeb | bash
        ;;
    *)
        echo "WARN: unknown M5_ARTIFACT_DEST scheme: $M5_ARTIFACT_DEST" >&2
        ;;
esac

if [ ! -x "$UV_BIN" ]; then
    echo "==> $(date -Is) installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:$PATH"

# ---- clone repo ----------------------------------------------------------
update_activity
mkdir -p "$(dirname "$REPO_DIR")"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$M5_GIT_REPO" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch --all --tags --prune
git checkout "$M5_GIT_REF"
git pull --ff-only origin "$M5_GIT_REF" || true

# ---- python deps ---------------------------------------------------------
update_activity
"$UV_BIN" sync --no-group dev --no-group notebook

# ---- pipeline with timeouts ----------------------------------------------
# Each step has a 6-hour timeout to prevent indefinite hangs

run_with_timeout() {
    local step_name="$1"
    local timeout_seconds="${2:-21600}"  # 6 hours default
    shift 2

    echo "==> $(date -Is) m5-train: $step_name (timeout=${timeout_seconds}s)"
    update_activity

    if timeout "$timeout_seconds" "$@"; then
        echo "==> $(date -Is) m5-train: $step_name complete"
        update_activity
        return 0
    else
        echo "==> $(date -Is) m5-train: $step_name FAILED or TIMEOUT"
        return 1
    fi
}

# Step 1: Download
run_with_timeout "download" 3600 "$UV_BIN" run m5 download

# Step 2: Prep
run_with_timeout "prep" 3600 "$UV_BIN" run m5 prep \
    --last-n-days "$M5_LAST_N_DAYS" \
    --n-series "$M5_N_SERIES"

# Step 3: CV Stats
if [ "$M5_RUN_STATS_CV" = "true" ]; then
    run_with_timeout "cv stats" 21600 "$UV_BIN" run m5 cv stats \
        --horizon "$M5_HORIZON" \
        --n-windows "$M5_CV_N_WINDOWS" \
        || echo "==> $(date -Is) m5-train: cv stats failed but continuing"
fi

# Step 4: CV LGBM
if [ "$M5_RUN_LGBM_CV" = "true" ]; then
    run_with_timeout "cv lgbm" 21600 "$UV_BIN" run m5 cv lgbm \
        --horizon "$M5_HORIZON" \
        --n-windows "$M5_CV_N_WINDOWS" \
        || echo "==> $(date -Is) m5-train: cv lgbm failed but continuing"
fi

# Step 5: CV Hier (with extended timeout, most likely to fail)
if [ "$M5_RUN_HIER_CV" = "true" ]; then
    echo "==> $(date -Is) m5-train: cv hier starting - this may take 2-4 hours"
    run_with_timeout "cv hier" 28800 "$UV_BIN" run m5 cv hier \
        --horizon "$M5_HORIZON" \
        --n-windows "$M5_CV_N_WINDOWS" \
        || echo "==> $(date -Is) m5-train: cv hier failed but continuing"
fi

# Step 6: CV Recipe
if [ -n "$M5_CV_RECIPE" ]; then
    run_with_timeout "cv recipe" 21600 "$UV_BIN" run m5 cv-recipe \
        "$M5_CV_RECIPE" \
        --horizon "$M5_HORIZON" \
        --n-windows "$M5_CV_N_WINDOWS" \
        || echo "==> $(date -Is) m5-train: cv recipe failed but continuing"
fi

# Step 7: Score
if compgen -G "artifacts/cv_*.parquet" > /dev/null && [ -n "$M5_SCORE_MODELS" ]; then
    update_activity
    echo "==> $(date -Is) m5-train: score models=[$M5_SCORE_MODELS]"
    score_cmd=("$UV_BIN" run m5 score --out reports --run-id "$M5_RUN_ID")
    for model in $M5_SCORE_MODELS; do
        score_cmd+=(--model "$model")
    done
    run_with_timeout "score" 21600 "${score_cmd[@]}" \
        || echo "==> $(date -Is) m5-train: score failed but continuing"
else
    echo "==> $(date -Is) m5-train: score skipped (no CV artifacts or no models specified)"
fi

# Step 8: Train
MODEL_TIMESTAMP="$M5_RUN_ID"
if [ "$M5_RUN_TRAIN" = "true" ]; then
    update_activity
    echo "==> $(date -Is) m5-train: train (final fit on full data)"
    run_with_timeout "train" 21600 "$UV_BIN" run m5 train --horizon "$M5_HORIZON" \
        || echo "==> $(date -Is) m5-train: train failed but continuing"

    # Push model artifact
    if [ -d artifacts/models/lgbm/latest ]; then
        ARTIFACT_DIR=$(readlink -f artifacts/models/lgbm/latest)
        MODEL_TIMESTAMP=$(basename "$ARTIFACT_DIR")
        DEST="${M5_ARTIFACT_DEST%/}/$MODEL_TIMESTAMP"
        LATEST_DEST="${M5_ARTIFACT_DEST%/}/latest"

        echo "==> $(date -Is) pushing $ARTIFACT_DIR -> $DEST"
        bash cloud/scripts/push_artifact.sh "$ARTIFACT_DIR" "$DEST" || true

        echo "==> $(date -Is) pushing $ARTIFACT_DIR -> $LATEST_DEST"
        bash cloud/scripts/push_artifact.sh "$ARTIFACT_DIR" "$LATEST_DEST" || true
    fi
else
    echo "==> $(date -Is) m5-train: final train skipped"
fi

# ---- push reports --------------------------------------------------------
if [ -d reports ]; then
    REPORT_DEST="${M5_ARTIFACT_DEST%/}/reports/$MODEL_TIMESTAMP"
    REPORT_LATEST="${M5_ARTIFACT_DEST%/}/reports/latest"
    echo "==> $(date -Is) pushing reports/ -> $REPORT_DEST"
    bash cloud/scripts/push_artifact.sh reports "$REPORT_DEST" || true
    bash cloud/scripts/push_artifact.sh reports "$REPORT_LATEST" || true
fi

# ---- push CV artifacts ---------------------------------------------------
if compgen -G "artifacts/cv_*.parquet" > /dev/null; then
    CV_DIR=$(mktemp -d)
    cp artifacts/cv_*.parquet "$CV_DIR/"
    CV_DEST="${M5_ARTIFACT_DEST%/}/cv/$MODEL_TIMESTAMP"
    CV_LATEST="${M5_ARTIFACT_DEST%/}/cv/latest"
    echo "==> $(date -Is) pushing artifacts/cv_*.parquet -> $CV_DEST"
    bash cloud/scripts/push_artifact.sh "$CV_DIR" "$CV_DEST" || true
    bash cloud/scripts/push_artifact.sh "$CV_DIR" "$CV_LATEST" || true
fi

# ---- push run bundle -----------------------------------------------------
RUN_BUNDLE=$(mktemp -d)
mkdir -p "$RUN_BUNDLE/metadata"
[ -d artifacts ] && cp -a artifacts "$RUN_BUNDLE/"
[ -d reports ] && cp -a reports "$RUN_BUNDLE/"
[ -d forecasts ] && cp -a forecasts "$RUN_BUNDLE/"
if [ "$M5_PUSH_PROCESSED" = "true" ] && [ -f data/processed/long.parquet ]; then
    mkdir -p "$RUN_BUNDLE/data"
    cp data/processed/long.parquet "$RUN_BUNDLE/data/long.parquet"
fi

# Write metadata
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
    --arg status "complete" \
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
      push_processed: ($push_processed == "true"),
      status: $status
    }' > "$RUN_BUNDLE/metadata/run.json"

RUN_DEST="${M5_ARTIFACT_DEST%/}/runs/$M5_RUN_ID"
RUN_LATEST="${M5_ARTIFACT_DEST%/}/runs/latest"
echo "==> $(date -Is) pushing run bundle -> $RUN_DEST"
bash cloud/scripts/push_artifact.sh "$RUN_BUNDLE" "$RUN_DEST" || true
bash cloud/scripts/push_artifact.sh "$RUN_BUNDLE" "$RUN_LATEST" || true

# ---- cleanup -------------------------------------------------------------
echo "==> $(date -Is) m5-train: complete (run_id=$M5_RUN_ID, model_timestamp=$MODEL_TIMESTAMP)"
echo "$M5_RUN_ID" > /srv/M5/.train-complete

# Kill health monitor
kill "$HEALTH_PID" 2>/dev/null || true

# ---- self-destruct -------------------------------------------------------
if [ "$M5_TRAIN_SHUTDOWN_ON_DONE" = "true" ]; then
    echo "==> $(date -Is) shutting down (M5_TRAIN_SHUTDOWN_ON_DONE=true)"
    sleep 30
    /sbin/poweroff
fi
