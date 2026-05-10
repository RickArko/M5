output "artifact_bucket" {
  value = google_storage_bucket.artifact.name
}

output "artifact_uri" {
  value = "gs://${google_storage_bucket.artifact.name}/${var.artifact_prefix}"
}

output "train_public_ip" {
  value = try(google_compute_instance.train[0].network_interface[0].access_config[0].nat_ip, null)
}

output "serve_public_ip" {
  value = try(google_compute_instance.serve[0].network_interface[0].access_config[0].nat_ip, null)
}

output "ssh_train" {
  value = try("ssh ${var.admin_username}@${google_compute_instance.train[0].network_interface[0].access_config[0].nat_ip}", null)
}

output "ssh_serve" {
  value = try("ssh ${var.admin_username}@${google_compute_instance.serve[0].network_interface[0].access_config[0].nat_ip}", null)
}

output "serve_url" {
  value = try("http://${google_compute_instance.serve[0].network_interface[0].access_config[0].nat_ip}:${var.serve_port}", null)
}
