# GCP auth comes from `gcloud auth application-default login` or GOOGLE_APPLICATION_CREDENTIALS.
variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "zone" {
  type    = string
  default = "us-central1-a"
}

# ----------------------------------------------------------------------------
# SSH
# ----------------------------------------------------------------------------
variable "ssh_public_key" {
  description = "SSH public key contents."
  type        = string
}

variable "admin_username" {
  description = "Linux username injected via metadata.ssh-keys."
  type        = string
  default     = "m5"
}

# ----------------------------------------------------------------------------
# Sizing — n2-standard-8: 8 vCPU / 32 GB / ~$0.39/h.  e2-medium: 2 vCPU / 4 GB / ~$25/mo.
# ----------------------------------------------------------------------------
variable "train_machine_type" {
  type    = string
  default = "n2-standard-8"
}

variable "serve_machine_type" {
  type    = string
  default = "e2-medium"
}

# ----------------------------------------------------------------------------
# Repo
# ----------------------------------------------------------------------------
variable "git_repo" {
  type    = string
  default = "https://github.com/RickArko/M5.git"
}

variable "git_ref" {
  type    = string
  default = "main"
}

# ----------------------------------------------------------------------------
# Pipeline knobs
# ----------------------------------------------------------------------------
variable "horizon" {
  type    = number
  default = 28
}

variable "last_n_days" {
  type    = number
  default = 400
}

variable "n_series" {
  type    = number
  default = -1
}

variable "run_id" {
  description = "Cloud run id. Empty means the train VM creates a UTC timestamp."
  type        = string
  default     = ""
}

variable "run_stats_cv" {
  type    = bool
  default = true
}

variable "run_lgbm_cv" {
  type    = bool
  default = true
}

variable "run_hier_cv" {
  type    = bool
  default = false
}

variable "cv_recipe" {
  description = "Optional recipe path to run with m5 cv-recipe."
  type        = string
  default     = ""
}

variable "cv_n_windows" {
  type    = number
  default = 3
}

variable "score_models" {
  description = "Space-separated model artifact names passed to m5 score."
  type        = string
  default     = "stats lgbm"
}

variable "run_train" {
  description = "Run final m5 train serving artifact fit."
  type        = bool
  default     = true
}

variable "push_processed" {
  description = "Upload data/processed/long.parquet into runs/<run-id>/data."
  type        = bool
  default     = false
}

# ----------------------------------------------------------------------------
# Object storage — GCS bucket created by this module.
# ----------------------------------------------------------------------------
variable "artifact_bucket_name" {
  description = "GCS bucket name. Must be globally unique."
  type        = string
}

variable "artifact_prefix" {
  type    = string
  default = "m5/lgbm"
}

# ----------------------------------------------------------------------------
# Serve config
# ----------------------------------------------------------------------------
variable "serve_port" {
  type    = number
  default = 8000
}

variable "serve_api_key" {
  type      = string
  default   = ""
  sensitive = true
}

# ----------------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------------
variable "create_train" {
  type    = bool
  default = true
}

variable "create_serve" {
  type    = bool
  default = true
}

variable "shutdown_train_on_done" {
  type    = bool
  default = true
}

# ----------------------------------------------------------------------------
# Firewall
# ----------------------------------------------------------------------------
variable "allowed_ssh_cidrs" {
  type    = list(string)
  default = ["0.0.0.0/0"]
}

variable "allowed_serve_cidrs" {
  type    = list(string)
  default = ["0.0.0.0/0"]
}
