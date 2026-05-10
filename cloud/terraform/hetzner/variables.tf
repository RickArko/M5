# ----------------------------------------------------------------------------
# Hetzner Cloud auth
# ----------------------------------------------------------------------------
variable "hcloud_token" {
  description = "Hetzner Cloud API token. Set via TF_VAR_hcloud_token or HCLOUD_TOKEN env."
  type        = string
  sensitive   = true
}

# ----------------------------------------------------------------------------
# SSH access
# ----------------------------------------------------------------------------
variable "ssh_public_key" {
  description = "SSH public key contents (e.g. file('~/.ssh/id_ed25519.pub'))."
  type        = string
}

variable "ssh_key_name" {
  description = "Name to register the SSH key under."
  type        = string
  default     = "m5-key"
}

# ----------------------------------------------------------------------------
# Geography & sizing
# ----------------------------------------------------------------------------
variable "location" {
  description = "Hetzner location. Object-storage-compatible: fsn1, nbg1, hel1."
  type        = string
  default     = "fsn1"
}

variable "image" {
  description = "OS image. Ubuntu 24.04 LTS is what cloud-init scripts target."
  type        = string
  default     = "ubuntu-24.04"
}

variable "train_server_type" {
  description = "Server type for training. ccx33 = 8 dedicated vCPU / 32 GB RAM (~€0.06/hour)."
  type        = string
  default     = "ccx33"
}

variable "serve_server_type" {
  description = "Server type for serve. cpx21 = 3 vCPU / 4 GB RAM (~€5/month always-on)."
  type        = string
  default     = "cpx21"
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
# Pipeline knobs (forwarded as M5_* env on the VM)
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
# Object storage (Hetzner Object Storage is S3-compatible; bucket pre-created
# in the console — Terraform support is not yet GA as of 2024-09).
# ----------------------------------------------------------------------------
variable "artifact_uri" {
  description = "S3-compat URI for the artifact (e.g. s3://my-bucket/m5/lgbm)."
  type        = string
}

variable "object_store_endpoint" {
  description = "S3 endpoint. Hetzner Object Storage: https://<location>.your-objectstorage.com."
  type        = string
  default     = "https://fsn1.your-objectstorage.com"
}

variable "aws_access_key_id" {
  description = "Access key for the object store (S3 access keys generated in Hetzner console)."
  type        = string
  sensitive   = true
}

variable "aws_secret_access_key" {
  description = "Secret key for the object store."
  type        = string
  sensitive   = true
}

# ----------------------------------------------------------------------------
# Serve config
# ----------------------------------------------------------------------------
variable "serve_port" {
  type    = number
  default = 8000
}

variable "serve_api_key" {
  description = "X-API-Key the serve VM enforces. Empty = auth disabled."
  type        = string
  default     = ""
  sensitive   = true
}

# ----------------------------------------------------------------------------
# Lifecycle toggles
# ----------------------------------------------------------------------------
variable "create_train" {
  description = "Spin up the train VM."
  type        = bool
  default     = true
}

variable "create_serve" {
  description = "Spin up the serve VM."
  type        = bool
  default     = true
}

variable "shutdown_train_on_done" {
  description = "Train VM powers off after pushing artifact (still billed until `terraform destroy`)."
  type        = bool
  default     = true
}

# ----------------------------------------------------------------------------
# Firewall rules
# ----------------------------------------------------------------------------
variable "allowed_ssh_cidrs" {
  description = "CIDRs allowed to SSH. Tighten in production (e.g. [\"<your-ip>/32\"])."
  type        = list(string)
  default     = ["0.0.0.0/0", "::/0"]
}

variable "allowed_serve_cidrs" {
  description = "CIDRs allowed to hit the serve port."
  type        = list(string)
  default     = ["0.0.0.0/0", "::/0"]
}
