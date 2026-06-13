# AWS Terraform configuration — created automatically by scripts/create_s3_bucket.sh
# AWS auth comes from ~/.aws/credentials, AWS_PROFILE, or instance role.

region         = "us-east-1"
ssh_public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBJxSgBelNlIaB1N2Gqoqqu0pVsw/W/z/nwFWWmltCVR"

# S3 bucket — must be globally unique.
artifact_bucket_name = "m5-artifacts-ricka-1781390300"

# ---- Instance sizing for full M5 + hierarchy --------------------------
train_instance_type = "r7i.4xlarge"   # 16 vCPU / 128 GB RAM / ~$0.80/h
# serve_instance_type = "t3.medium"   # 2 vCPU / 4 GB  / ~$30/mo

# ---- Cloud job controls -----------------------------------------------
# run_stats_cv      = true
# run_lgbm_cv       = true
# run_hier_cv       = true
# cv_n_windows      = 3
# score_models      = "stats lgbm hier"
# run_train         = true
# push_processed    = true

# ---- Optional: tighten firewall ---------------------------------------
# allowed_ssh_cidrs   = ["203.0.113.42/32"]
# allowed_serve_cidrs = ["203.0.113.42/32"]

# ---- Optional: serve-side API key -------------------------------------
# serve_api_key = "long-random-string"
