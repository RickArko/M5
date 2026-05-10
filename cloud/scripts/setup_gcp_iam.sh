#!/usr/bin/env bash
# One-shot GCP project IAM bootstrap for the M5 train run.
# Idempotent: safe to re-run.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-m5-rickarko-2026}"
SA_EMAIL="m5-terraform@${PROJECT_ID}.iam.gserviceaccount.com"

echo "==> Enabling cloudresourcemanager API"
gcloud services enable cloudresourcemanager.googleapis.com --project="$PROJECT_ID"

for ROLE in roles/editor roles/iam.securityAdmin roles/resourcemanager.projectIamAdmin; do
    echo "==> Granting $ROLE to $SA_EMAIL"
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$ROLE" \
        --condition=None \
        --quiet \
        --no-user-output-enabled
done

echo "==> Verifying IAM bindings:"
gcloud projects get-iam-policy "$PROJECT_ID" \
    --flatten="bindings[].members" \
    --filter="bindings.members:$SA_EMAIL" \
    --format="value(bindings.role)"

echo "==> done."
