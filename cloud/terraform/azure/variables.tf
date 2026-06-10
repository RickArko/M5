# Azure auth comes from `az login`, env vars (ARM_*), or managed identity.
variable "location" {
  description = "Azure region."
  type        = string
  default     = "eastus"
}

variable "resource_group_name" {
  description = "Resource group name. Created by this module."
  type        = string
  default     = "rg-m5"
}

# ----------------------------------------------------------------------------
# SSH
# ----------------------------------------------------------------------------
variable "ssh_public_key" {
  description = "SSH public key contents."
  type        = string
}

variable "admin_username" {
  description = "Linux admin username on the VM."
  type        = string
  default     = "azureuser"
}

# ----------------------------------------------------------------------------
# Sizing
# Azure VMs:  D8s_v5 = 8 vCPU / 32 GB / ~$0.38/h.   B2s = 2 vCPU / 4 GB / ~$30/mo.
# ----------------------------------------------------------------------------
variable "train_vm_size" {
  type    = string
  default = "Standard_D8s_v5"
}

variable "serve_vm_size" {
  type    = string
  default = "Standard_B2s"
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
# Object storage — Azure Blob.
# ----------------------------------------------------------------------------
variable "storage_account_name" {
  description = "Storage account name (3-24 chars, lowercase + digits, globally unique)."
  type        = string
}

variable "container_name" {
  description = "Blob container for the artifact."
  type        = string
  default     = "m5-artifacts"
}

variable "artifact_prefix" {
  description = "Prefix inside the container."
  type        = string
  default     = "lgbm"
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
