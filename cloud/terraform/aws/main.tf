# ----------------------------------------------------------------------------
# Networking — default VPC by default; can be made explicit if needed.
# ----------------------------------------------------------------------------
data "aws_vpc" "default" {
  count   = var.use_default_vpc ? 1 : 0
  default = true
}

data "aws_subnets" "default" {
  count = var.use_default_vpc ? 1 : 0
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default[0].id]
  }
}

locals {
  vpc_id    = var.use_default_vpc ? data.aws_vpc.default[0].id : null
  subnet_id = var.use_default_vpc ? data.aws_subnets.default[0].ids[0] : null
}

# ----------------------------------------------------------------------------
# AMI — Ubuntu 24.04 LTS HVM EBS-SSD, owned by Canonical (099720109477).
# ----------------------------------------------------------------------------
data "aws_ami" "ubuntu_2404" {
  most_recent = true
  owners      = ["099720109477"]
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ----------------------------------------------------------------------------
# Key pair
# ----------------------------------------------------------------------------
resource "aws_key_pair" "m5" {
  key_name   = var.ssh_key_name
  public_key = var.ssh_public_key
  tags       = { Project = "m5" }
}

# ----------------------------------------------------------------------------
# Security groups — train (SSH only) vs serve (SSH + serve_port)
# ----------------------------------------------------------------------------
resource "aws_security_group" "train" {
  name        = "m5-train"
  description = "M5 training VM - SSH only"
  vpc_id      = local.vpc_id
  tags        = { Project = "m5", Role = "train" }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "serve" {
  name        = "m5-serve"
  description = "M5 serve VM - SSH + FastAPI"
  vpc_id      = local.vpc_id
  tags        = { Project = "m5", Role = "serve" }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }
  ingress {
    description = "FastAPI"
    from_port   = var.serve_port
    to_port     = var.serve_port
    protocol    = "tcp"
    cidr_blocks = var.allowed_serve_cidrs
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ----------------------------------------------------------------------------
# S3 bucket for artifact handoff (must be pre-created)
# ----------------------------------------------------------------------------
# The bucket is created by scripts/create_s3_bucket.sh before terraform apply.
# This data source looks up the existing bucket.
data "aws_s3_bucket" "artifact" {
  bucket = var.artifact_bucket_name
}

# ----------------------------------------------------------------------------
# IAM — instance profile that lets EC2 read/write the artifact bucket.
# ----------------------------------------------------------------------------
data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "vm" {
  name               = "m5-vm"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
  tags               = { Project = "m5" }
}

data "aws_iam_policy_document" "vm" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = [data.aws_s3_bucket.artifact.arn]
  }
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = ["${data.aws_s3_bucket.artifact.arn}/*"]
  }
}

resource "aws_iam_role_policy" "vm" {
  name   = "m5-vm-s3"
  role   = aws_iam_role.vm.id
  policy = data.aws_iam_policy_document.vm.json
}

resource "aws_iam_instance_profile" "vm" {
  name = "m5-vm"
  role = aws_iam_role.vm.name
}

# ----------------------------------------------------------------------------
# User-data templating
# ----------------------------------------------------------------------------
locals {
  user_data_template = "${path.module}/../../cloud-init/_user_data.sh.tftpl"
  artifact_uri       = "s3://${data.aws_s3_bucket.artifact.id}/${var.artifact_prefix}"

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
    object_store_endpoint = "" # native S3 — instance profile handles auth
    aws_access_key_id     = ""
    aws_secret_access_key = ""
    aws_region            = var.region
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
    aws_region            = var.region
  })
}

# ----------------------------------------------------------------------------
# Train EC2 instance
# ----------------------------------------------------------------------------
resource "aws_instance" "train" {
  count                       = var.create_train ? 1 : 0
  ami                         = data.aws_ami.ubuntu_2404.id
  instance_type               = var.train_instance_type
  key_name                    = aws_key_pair.m5.key_name
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.train.id]
  iam_instance_profile        = aws_iam_instance_profile.vm.name
  user_data                   = local.user_data_train
  user_data_replace_on_change = true

  root_block_device {
    volume_size = 60
    volume_type = "gp3"
  }

  # If shutdown-on-done, the box self-poweroffs; this lets EC2 release the instance state cleanly.
  instance_initiated_shutdown_behavior = var.shutdown_train_on_done ? "stop" : "terminate"

  tags = {
    Name    = "m5-train"
    Project = "m5"
    Role    = "train"
  }
}

# ----------------------------------------------------------------------------
# Serve EC2 instance
# ----------------------------------------------------------------------------
resource "aws_instance" "serve" {
  count                       = var.create_serve ? 1 : 0
  ami                         = data.aws_ami.ubuntu_2404.id
  instance_type               = var.serve_instance_type
  key_name                    = aws_key_pair.m5.key_name
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.serve.id]
  iam_instance_profile        = aws_iam_instance_profile.vm.name
  user_data                   = local.user_data_serve
  user_data_replace_on_change = true

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  tags = {
    Name    = "m5-serve"
    Project = "m5"
    Role    = "serve"
  }
}
