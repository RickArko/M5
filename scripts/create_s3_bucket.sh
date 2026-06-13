#!/usr/bin/env bash
# Create an S3 bucket for M5 artifacts on AWS.
# Usage: bash scripts/create_s3_bucket.sh [bucket_name]
#
# If no bucket_name is provided, generates one like: m5-artifacts-$(whoami)-$(date +%s)

set -euo pipefail

# Default bucket name if not provided
BUCKET_NAME="${1:-m5-artifacts-$(whoami)-$(date +%s)}"
REGION="${AWS_REGION:-us-east-1}"

echo "==> Creating S3 bucket: $BUCKET_NAME in $REGION"

# Check AWS credentials
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "ERROR: AWS credentials not configured. Run 'aws configure' or set AWS_PROFILE."
    exit 1
fi

# Create bucket (us-east-1 doesn't need LocationConstraint)
if [ "$REGION" = "us-east-1" ]; then
    aws s3api create-bucket \
        --bucket "$BUCKET_NAME" \
        --region "$REGION"
else
    aws s3api create-bucket \
        --bucket "$BUCKET_NAME" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION"
fi

echo "==> Bucket created: s3://$BUCKET_NAME"

# Enable versioning for artifact history
aws s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled

echo "==> Versioning enabled"

# Output for terraform.tfvars
echo ""
echo "==> Add to cloud/terraform/aws/terraform.tfvars:"
echo "artifact_bucket_name = \"$BUCKET_NAME\""
echo "region               = \"$REGION\""
