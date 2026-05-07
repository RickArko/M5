# AWS auth comes from standard SDK chain (env vars, ~/.aws/credentials, IAM role).
variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "ssh_public_key" {
  description = "SSH public key contents (e.g. file('~/.ssh/id_ed25519.pub'))."
  type        = string
}

variable "ssh_key_name" {
  description = "EC2 key-pair name (alphanumeric, no path)."
  type        = string
  default     = "m5-key"
}

# ----------------------------------------------------------------------------
# Sizing — c7i.2xlarge: 8 vCPU / 16 GB / ~$0.36/h. t3.medium: 2 vCPU / 4 GB / ~$30/mo.
# ----------------------------------------------------------------------------
variable "train_instance_type" {
  type    = string
  default = "c7i.2xlarge"
}

variable "serve_instance_type" {
  type    = string
  default = "t3.medium"
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

# ----------------------------------------------------------------------------
# Object storage — S3 bucket created by this module.
# ----------------------------------------------------------------------------
variable "artifact_bucket_name" {
  description = "S3 bucket name for the trained artifact. Must be globally unique."
  type        = string
}

variable "artifact_prefix" {
  description = "Prefix inside the bucket where the artifact is stored."
  type        = string
  default     = "m5/lgbm"
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
# Lifecycle toggles
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
# Firewall rules
# ----------------------------------------------------------------------------
variable "allowed_ssh_cidrs" {
  description = "CIDRs allowed to SSH. Tighten in production."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "allowed_serve_cidrs" {
  description = "CIDRs allowed to hit the serve port."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# ----------------------------------------------------------------------------
# VPC — by default uses the account's default VPC; set to false to make a new one.
# ----------------------------------------------------------------------------
variable "use_default_vpc" {
  type    = bool
  default = true
}
