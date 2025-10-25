# Main Terraform Configuration
# This file orchestrates the overall infrastructure setup

# Generate random suffix for globally unique resource names
resource "random_id" "suffix" {
  byte_length = 4
}

# Data source to get latest Amazon Linux 2 AMI
data "aws_ami" "amazon_linux_2" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Data source to get current AWS account ID
data "aws_caller_identity" "current" {}

# Data source to get current region
data "aws_region" "current" {}

# Local values for consistent naming and tagging
locals {
  common_tags = {
    Project       = var.project_name
    Environment   = var.environment
    ManagedBy     = "Terraform"
    Owner         = var.owner
    CostCenter    = var.cost_center
    WorkloadType  = var.ml_workload_type
  }

  name_prefix = "${var.project_name}-${var.environment}"

  # S3 bucket names (must be globally unique)
  datasets_bucket_name = "${local.name_prefix}-datasets-${random_id.suffix.hex}"
  models_bucket_name   = "${local.name_prefix}-models-${random_id.suffix.hex}"
}
