output "artifact_bucket" {
  description = "S3 bucket holding the trained artifact."
  value       = aws_s3_bucket.artifact.id
}

output "artifact_uri" {
  description = "Full s3:// URI for the artifact prefix."
  value       = "s3://${aws_s3_bucket.artifact.id}/${var.artifact_prefix}"
}

output "train_public_ip" {
  description = "Public IPv4 of the train EC2 instance."
  value       = try(aws_instance.train[0].public_ip, null)
}

output "serve_public_ip" {
  description = "Public IPv4 of the serve EC2 instance."
  value       = try(aws_instance.serve[0].public_ip, null)
}

output "ssh_train" {
  description = "SSH command to reach the train instance."
  value       = try("ssh ubuntu@${aws_instance.train[0].public_ip}", null)
}

output "ssh_serve" {
  description = "SSH command to reach the serve instance."
  value       = try("ssh ubuntu@${aws_instance.serve[0].public_ip}", null)
}

output "serve_url" {
  description = "URL to the serve API."
  value       = try("http://${aws_instance.serve[0].public_ip}:${var.serve_port}", null)
}
