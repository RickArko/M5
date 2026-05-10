#!/usr/bin/env bash
# Bootstrap a long-running M5 serving VM.
#
# Idempotent — safe to re-run. Cloud-init invokes this on first boot;
# the m5-bootstrap.service systemd unit re-invokes it on every subsequent
# boot so a reboot doesn't drop the model handle.
#
# Lifecycle:
#   1. Install Docker + Compose + AWS CLI (S3-compat puller).
#   2. Clone the M5 repo (so Dockerfile + compose file are local).
#   3. Pull the trained artifact from object storage to /srv/m5-artifact.
#   4. Build + start the FastAPI container via `docker compose`.
#   5. Wait for /healthz to pass.
#
# /etc/m5-cloud.env is rendered by Terraform with:
#   M5_GIT_REPO, M5_GIT_REF, M5_ARTIFACT_SOURCE, M5_SERVE_API_KEY,
#   M5_SERVE_PORT, M5_OBJECT_STORE_ENDPOINT

set -euo pipefail
exec > >(tee -a /var/log/m5-serve.log) 2>&1
echo "==> $(date -Is) m5-serve: starting"

# ---- env -----------------------------------------------------------------
[ -f /etc/m5-cloud.env ] && set -a && source /etc/m5-cloud.env && set +a

: "${M5_GIT_REPO:=https://github.com/RickArko/M5.git}"
: "${M5_GIT_REF:=main}"
: "${M5_ARTIFACT_SOURCE:?M5_ARTIFACT_SOURCE required (e.g. s3://my-bucket/m5/lgbm/latest)}"
: "${M5_SERVE_PORT:=8000}"
: "${M5_SERVE_API_KEY:=}"

REPO_DIR=/srv/M5
ARTIFACT_DIR=/srv/m5-artifact

# ---- OS deps -------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    git curl ca-certificates make jq unzip apt-transport-https gnupg

# Install the object-storage CLI that matches the artifact source scheme.
case "$M5_ARTIFACT_SOURCE" in
    gs://*)
        if ! command -v gcloud >/dev/null 2>&1; then
            echo "==> installing google-cloud-cli (for gs:// pull)"
            curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
                | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
            echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
                > /etc/apt/sources.list.d/google-cloud-sdk.list
            apt-get update -y
            apt-get install -y --no-install-recommends google-cloud-cli
        fi
        ;;
    s3://*)
        apt-get install -y --no-install-recommends awscli
        ;;
    az://*)
        curl -sL https://aka.ms/InstallAzureCLIDeb | bash
        ;;
    *)
        echo "WARN: unknown M5_ARTIFACT_SOURCE scheme: $M5_ARTIFACT_SOURCE" >&2
        ;;
esac

if ! command -v docker >/dev/null 2>&1; then
    echo "==> installing Docker"
    curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker

# ---- clone repo ----------------------------------------------------------
mkdir -p "$(dirname "$REPO_DIR")"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$M5_GIT_REPO" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch --all --tags --prune
git checkout "$M5_GIT_REF"
git pull --ff-only origin "$M5_GIT_REF" || true

# ---- artifact pull -------------------------------------------------------
mkdir -p "$ARTIFACT_DIR"
echo "==> pulling artifact from $M5_ARTIFACT_SOURCE -> $ARTIFACT_DIR"
bash cloud/scripts/pull_artifact.sh "$M5_ARTIFACT_SOURCE" "$ARTIFACT_DIR"

if [ ! -f "$ARTIFACT_DIR/metadata.json" ] || [ ! -f "$ARTIFACT_DIR/model.joblib" ]; then
    echo "FATAL: artifact pull did not produce metadata.json + model.joblib in $ARTIFACT_DIR" >&2
    ls -la "$ARTIFACT_DIR" || true
    exit 1
fi

# ---- compose override ----------------------------------------------------
# Bind the pulled artifact dir into the container; pass through the API key.
cat > docker-compose.override.yaml <<EOF
services:
  m5-forecaster:
    volumes:
      - $ARTIFACT_DIR:/srv/model:ro
    environment:
      M5_SERVE_API_KEY: "$M5_SERVE_API_KEY"
    ports:
      - "$M5_SERVE_PORT:8000"
EOF

# ---- start ---------------------------------------------------------------
echo "==> docker compose up -d --build"
docker compose up -d --build

# ---- health check --------------------------------------------------------
for attempt in $(seq 1 30); do
    if curl --fail --silent "http://127.0.0.1:$M5_SERVE_PORT/healthz" >/dev/null; then
        echo "==> healthz ok after ${attempt}s"
        echo "==> $(date -Is) m5-serve: ready"
        exit 0
    fi
    sleep 1
done

echo "FATAL: /healthz did not respond after 30s" >&2
docker compose logs --tail=200 || true
exit 1
