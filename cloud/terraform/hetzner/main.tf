# ----------------------------------------------------------------------------
# Common helpers
# ----------------------------------------------------------------------------
locals {
  user_data_template = "${path.module}/../../cloud-init/_user_data.sh.tftpl"

  user_data_train = templatefile(local.user_data_template, {
    role                  = "train"
    git_repo              = var.git_repo
    git_ref               = var.git_ref
    artifact_dest         = var.artifact_uri
    artifact_source       = "" # serve-only field; train doesn't read it
    last_n_days           = var.last_n_days
    n_series              = var.n_series
    horizon               = var.horizon
    run_id                = var.run_id
    run_stats_cv          = var.run_stats_cv ? "true" : "false"
    run_lgbm_cv           = var.run_lgbm_cv ? "true" : "false"
    run_hier_cv           = var.run_hier_cv ? "true" : "false"
    cv_recipe             = var.cv_recipe
    cv_n_windows          = var.cv_n_windows
    score_models          = var.score_models
    run_train             = var.run_train ? "true" : "false"
    push_processed        = var.push_processed ? "true" : "false"
    shutdown_on_done      = var.shutdown_train_on_done ? "true" : "false"
    serve_port            = var.serve_port
    serve_api_key         = var.serve_api_key
    object_store_endpoint = var.object_store_endpoint
    aws_access_key_id     = var.aws_access_key_id
    aws_secret_access_key = var.aws_secret_access_key
    aws_region            = "auto"
  })

  user_data_serve = templatefile(local.user_data_template, {
    role                  = "serve"
    git_repo              = var.git_repo
    git_ref               = var.git_ref
    artifact_dest         = "" # train-only field
    artifact_source       = "${trimsuffix(var.artifact_uri, "/")}/latest"
    last_n_days           = var.last_n_days
    n_series              = var.n_series
    horizon               = var.horizon
    run_id                = var.run_id
    run_stats_cv          = var.run_stats_cv ? "true" : "false"
    run_lgbm_cv           = var.run_lgbm_cv ? "true" : "false"
    run_hier_cv           = var.run_hier_cv ? "true" : "false"
    cv_recipe             = var.cv_recipe
    cv_n_windows          = var.cv_n_windows
    score_models          = var.score_models
    run_train             = var.run_train ? "true" : "false"
    push_processed        = var.push_processed ? "true" : "false"
    shutdown_on_done      = "false"
    serve_port            = var.serve_port
    serve_api_key         = var.serve_api_key
    object_store_endpoint = var.object_store_endpoint
    aws_access_key_id     = var.aws_access_key_id
    aws_secret_access_key = var.aws_secret_access_key
    aws_region            = "auto"
  })
}

# ----------------------------------------------------------------------------
# SSH key
# ----------------------------------------------------------------------------
resource "hcloud_ssh_key" "m5" {
  name       = var.ssh_key_name
  public_key = var.ssh_public_key
  labels     = { project = "m5" }
}

# ----------------------------------------------------------------------------
# Firewalls — separate rules for train (SSH only) vs serve (SSH + serve_port)
# ----------------------------------------------------------------------------
resource "hcloud_firewall" "train" {
  name   = "m5-train"
  labels = { project = "m5", role = "train" }
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "22"
    source_ips = var.allowed_ssh_cidrs
  }
}

resource "hcloud_firewall" "serve" {
  name   = "m5-serve"
  labels = { project = "m5", role = "serve" }
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "22"
    source_ips = var.allowed_ssh_cidrs
  }
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = tostring(var.serve_port)
    source_ips = var.allowed_serve_cidrs
  }
}

# ----------------------------------------------------------------------------
# Train VM — ephemeral; powers off after pushing artifact if shutdown_train_on_done.
# ----------------------------------------------------------------------------
resource "hcloud_server" "train" {
  count        = var.create_train ? 1 : 0
  name         = "m5-train"
  server_type  = var.train_server_type
  image        = var.image
  location     = var.location
  ssh_keys     = [hcloud_ssh_key.m5.id]
  firewall_ids = [hcloud_firewall.train.id]
  user_data    = local.user_data_train
  labels       = { project = "m5", role = "train" }

  public_net {
    ipv4_enabled = true
    ipv6_enabled = true
  }
}

# ----------------------------------------------------------------------------
# Serve VM — long-lived; m5-bootstrap.service rehydrates on every boot.
# ----------------------------------------------------------------------------
resource "hcloud_server" "serve" {
  count        = var.create_serve ? 1 : 0
  name         = "m5-serve"
  server_type  = var.serve_server_type
  image        = var.image
  location     = var.location
  ssh_keys     = [hcloud_ssh_key.m5.id]
  firewall_ids = [hcloud_firewall.serve.id]
  user_data    = local.user_data_serve
  labels       = { project = "m5", role = "serve" }

  public_net {
    ipv4_enabled = true
    ipv6_enabled = true
  }
}
