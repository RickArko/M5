output "resource_group" {
  value = azurerm_resource_group.m5.name
}

output "storage_account" {
  value = azurerm_storage_account.m5.name
}

output "artifact_uri" {
  description = "Custom az:// URI consumed by cloud/scripts/{push,pull}_artifact.sh."
  value       = "az://${azurerm_storage_container.artifact.name}/${var.artifact_prefix}"
}

output "train_public_ip" {
  value = try(azurerm_public_ip.train[0].ip_address, null)
}

output "serve_public_ip" {
  value = try(azurerm_public_ip.serve[0].ip_address, null)
}

output "ssh_train" {
  value = try("ssh ${var.admin_username}@${azurerm_public_ip.train[0].ip_address}", null)
}

output "ssh_serve" {
  value = try("ssh ${var.admin_username}@${azurerm_public_ip.serve[0].ip_address}", null)
}

output "serve_url" {
  value = try("http://${azurerm_public_ip.serve[0].ip_address}:${var.serve_port}", null)
}
