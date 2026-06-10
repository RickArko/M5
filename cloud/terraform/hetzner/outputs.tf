output "train_ipv4" {
  description = "Public IPv4 of the train VM (null if create_train=false)."
  value       = try(hcloud_server.train[0].ipv4_address, null)
}

output "artifact_uri" {
  description = "S3-compatible URI for the artifact prefix."
  value       = var.artifact_uri
}

output "serve_ipv4" {
  description = "Public IPv4 of the serve VM (null if create_serve=false)."
  value       = try(hcloud_server.serve[0].ipv4_address, null)
}

output "ssh_train" {
  description = "SSH command to reach the train VM."
  value       = try("ssh root@${hcloud_server.train[0].ipv4_address}", null)
}

output "ssh_serve" {
  description = "SSH command to reach the serve VM."
  value       = try("ssh root@${hcloud_server.serve[0].ipv4_address}", null)
}

output "serve_url" {
  description = "URL to the serve API (after /healthz comes back ok)."
  value       = try("http://${hcloud_server.serve[0].ipv4_address}:${var.serve_port}", null)
}
