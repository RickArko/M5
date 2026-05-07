#!/usr/bin/env bash
# Upload a directory to object storage. URI scheme picks the backend:
#   s3://bucket/path     → aws s3 sync          (works for AWS S3, Hetzner Object Storage,
#                                                Cloudflare R2, MinIO — set M5_OBJECT_STORE_ENDPOINT)
#   az://container/path  → az storage blob upload-batch
#   gs://bucket/path     → gcloud storage rsync
#
# Usage:  push_artifact.sh <local_dir> <remote_uri>
# Env:    M5_OBJECT_STORE_ENDPOINT  (optional — for non-AWS S3-compatibles)
#         AZ_STORAGE_ACCOUNT        (required for az://)
#         AZ_STORAGE_KEY or AZURE_STORAGE_SAS_TOKEN

set -euo pipefail

LOCAL_DIR="${1:?usage: push_artifact.sh <local_dir> <remote_uri>}"
REMOTE_URI="${2:?usage: push_artifact.sh <local_dir> <remote_uri>}"

[ -d "$LOCAL_DIR" ] || { echo "FATAL: $LOCAL_DIR is not a directory" >&2; exit 2; }

case "$REMOTE_URI" in
    s3://*)
        endpoint_args=()
        if [ -n "${M5_OBJECT_STORE_ENDPOINT:-}" ]; then
            endpoint_args=(--endpoint-url "$M5_OBJECT_STORE_ENDPOINT")
        fi
        # --delete keeps the prefix in sync (drops files no longer in the source).
        # Skip it for the timestamped per-run prefix; keep it for "latest".
        sync_args=(sync "$LOCAL_DIR" "$REMOTE_URI" --no-progress)
        case "$REMOTE_URI" in
            */latest|*/latest/) sync_args+=(--delete) ;;
        esac
        aws "${endpoint_args[@]}" s3 "${sync_args[@]}"
        ;;
    az://*)
        # az://container/path → container=container, path=path
        rest="${REMOTE_URI#az://}"
        container="${rest%%/*}"
        prefix="${rest#"$container"}"
        prefix="${prefix#/}"
        : "${AZ_STORAGE_ACCOUNT:?AZ_STORAGE_ACCOUNT required for az:// URIs}"
        az storage blob upload-batch \
            --account-name "$AZ_STORAGE_ACCOUNT" \
            --destination "$container" \
            --destination-path "$prefix" \
            --source "$LOCAL_DIR" \
            --overwrite \
            --output none
        ;;
    gs://*)
        # `gcloud storage rsync` is the supported successor to `gsutil rsync`.
        gcloud storage rsync --recursive --delete-unmatched-destination-objects \
            "$LOCAL_DIR" "$REMOTE_URI"
        ;;
    *)
        echo "FATAL: unsupported URI scheme: $REMOTE_URI (expected s3://, az://, or gs://)" >&2
        exit 2
        ;;
esac
echo "==> push: $LOCAL_DIR -> $REMOTE_URI"
