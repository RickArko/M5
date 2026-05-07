#!/usr/bin/env bash
# Download a remote artifact directory to a local path. Mirror of push_artifact.sh.
#
# Usage:  pull_artifact.sh <remote_uri> <local_dir>
# Env:    M5_OBJECT_STORE_ENDPOINT  (optional — for S3-compatibles)
#         AZ_STORAGE_ACCOUNT        (required for az://)

set -euo pipefail

REMOTE_URI="${1:?usage: pull_artifact.sh <remote_uri> <local_dir>}"
LOCAL_DIR="${2:?usage: pull_artifact.sh <remote_uri> <local_dir>}"

mkdir -p "$LOCAL_DIR"

case "$REMOTE_URI" in
    s3://*)
        endpoint_args=()
        if [ -n "${M5_OBJECT_STORE_ENDPOINT:-}" ]; then
            endpoint_args=(--endpoint-url "$M5_OBJECT_STORE_ENDPOINT")
        fi
        aws "${endpoint_args[@]}" s3 sync "$REMOTE_URI" "$LOCAL_DIR" --no-progress
        ;;
    az://*)
        rest="${REMOTE_URI#az://}"
        container="${rest%%/*}"
        prefix="${rest#"$container"}"
        prefix="${prefix#/}"
        : "${AZ_STORAGE_ACCOUNT:?AZ_STORAGE_ACCOUNT required for az:// URIs}"
        az storage blob download-batch \
            --account-name "$AZ_STORAGE_ACCOUNT" \
            --source "$container" \
            --pattern "$prefix/*" \
            --destination "$LOCAL_DIR" \
            --output none
        # download-batch preserves the prefix as nested dirs; flatten it.
        if [ -n "$prefix" ] && [ -d "$LOCAL_DIR/$prefix" ]; then
            mv "$LOCAL_DIR/$prefix"/* "$LOCAL_DIR/" || true
            rmdir -p "$LOCAL_DIR/$prefix" 2>/dev/null || true
        fi
        ;;
    gs://*)
        gcloud storage rsync --recursive "$REMOTE_URI" "$LOCAL_DIR"
        ;;
    *)
        echo "FATAL: unsupported URI scheme: $REMOTE_URI (expected s3://, az://, or gs://)" >&2
        exit 2
        ;;
esac
echo "==> pull: $REMOTE_URI -> $LOCAL_DIR"
