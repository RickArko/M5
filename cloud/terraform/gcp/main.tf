# ----------------------------------------------------------------------------
# Networking — dedicated VPC keeps blast radius small.
# ----------------------------------------------------------------------------
resource "google_compute_network" "m5" {
  name                    = "vpc-m5"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "m5" {
  name          = "snet-m5"
  ip_cidr_range = "10.50.1.0/24"
  region        = var.region
  network       = google_compute_network.m5.id
}

resource "google_compute_firewall" "ssh" {
  name    = "m5-ssh"
  network = google_compute_network.m5.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  source_ranges = var.allowed_ssh_cidrs
  target_tags   = ["m5-ssh"]
}

resource "google_compute_firewall" "serve" {
  name    = "m5-serve"
  network = google_compute_network.m5.name

  allow {
    protocol = "tcp"
    ports    = [tostring(var.serve_port)]
  }
  source_ranges = var.allowed_serve_cidrs
  target_tags   = ["m5-serve"]
}

# ----------------------------------------------------------------------------
# Service account for the VMs (read/write the artifact bucket)
# ----------------------------------------------------------------------------
resource "google_service_account" "vm" {
  account_id   = "m5-vm-sa"
  display_name = "M5 VM service account"
}

# ----------------------------------------------------------------------------
# GCS bucket
# ----------------------------------------------------------------------------
resource "google_storage_bucket" "artifact" {
  name                        = var.artifact_bucket_name
  location                    = var.region
  force_destroy               = false
  uniform_bucket_level_access = true
  labels                      = { project = "m5" }
}

resource "google_storage_bucket_iam_member" "vm_admin" {
  bucket = google_storage_bucket.artifact.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.vm.email}"
}

# ----------------------------------------------------------------------------
# User-data templating
# ----------------------------------------------------------------------------
locals {
  user_data_template = "${path.module}/../../cloud-init/_user_data.sh.tftpl"
  artifact_uri       = "gs://${google_storage_bucket.artifact.name}/${var.artifact_prefix}"

  user_data_train = templatefile(local.user_data_template, {
    role                  = "train"
    git_repo              = var.git_repo
    git_ref               = var.git_ref
    artifact_dest         = local.artifact_uri
    artifact_source       = ""
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
    object_store_endpoint = ""
    aws_access_key_id     = ""
    aws_secret_access_key = ""
    aws_region            = ""
  })

  user_data_serve = templatefile(local.user_data_template, {
    role                  = "serve"
    git_repo              = var.git_repo
    git_ref               = var.git_ref
    artifact_dest         = ""
    artifact_source       = "${local.artifact_uri}/latest"
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
    object_store_endpoint = ""
    aws_access_key_id     = ""
    aws_secret_access_key = ""
    aws_region            = ""
  })

  ssh_metadata = "${var.admin_username}:${var.ssh_public_key}"
}

# ----------------------------------------------------------------------------
# Train VM
# ----------------------------------------------------------------------------
resource "google_compute_instance" "train" {
  count        = var.create_train ? 1 : 0
  name         = "m5-train"
  machine_type = var.train_machine_type
  zone         = var.zone
  tags         = ["m5-ssh"]
  labels       = { project = "m5", role = "train" }

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
      size  = 60
      type  = "pd-balanced"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.m5.id
    access_config {} # ephemeral public IP
  }

  metadata = {
    "ssh-keys" = local.ssh_metadata
    # The startup-script runs on every boot; cloud-init also runs the same script
    # via user-data — we use startup-script-url style for GCP-native behaviour.
    "user-data" = local.user_data_train
  }

  metadata_startup_script = local.user_data_train

  service_account {
    email  = google_service_account.vm.email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
  }
}

# ----------------------------------------------------------------------------
# Serve VM
# ----------------------------------------------------------------------------
resource "google_compute_instance" "serve" {
  count        = var.create_serve ? 1 : 0
  name         = "m5-serve"
  machine_type = var.serve_machine_type
  zone         = var.zone
  tags         = ["m5-ssh", "m5-serve"]
  labels       = { project = "m5", role = "serve" }

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
      size  = 30
      type  = "pd-balanced"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.m5.id
    access_config {}
  }

  metadata = {
    "ssh-keys"  = local.ssh_metadata
    "user-data" = local.user_data_serve
  }

  metadata_startup_script = local.user_data_serve

  service_account {
    email  = google_service_account.vm.email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
  }
}
