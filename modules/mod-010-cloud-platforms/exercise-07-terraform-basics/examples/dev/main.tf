# Development Environment Configuration
# This is an example of how to use the ML infrastructure modules for dev environment

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Use the parent terraform configuration as a module
module "ml_infrastructure" {
  source = "../../terraform"

  # Project Configuration
  project_name = "ml-dev"
  environment  = "dev"
  owner        = "dev-team"
  aws_region   = var.aws_region

  # VPC Configuration (smaller for dev)
  vpc_cidr             = "10.0.0.0/16"
  availability_zones   = ["us-east-1a", "us-east-1b"]
  public_subnet_cidrs  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnet_cidrs = ["10.0.11.0/24", "10.0.12.0/24"]

  # EC2 Configuration (smaller instance for dev)
  instance_type     = "t3.small"
  root_volume_size  = 30
  enable_jupyter    = true
  allowed_ssh_cidr  = var.allowed_ssh_cidr

  # S3 Configuration
  enable_s3_versioning  = false # Disable versioning in dev to save costs
  enable_s3_encryption  = true
  s3_lifecycle_days     = 30

  # ML Configuration
  ml_workload_type      = "notebook"
  enable_spot_instances = false

  # Monitoring (reduced for dev)
  enable_cloudwatch_alarms = false
  cpu_alarm_threshold      = 90

  # Cost Management
  enable_cost_tags = true
  cost_center      = "development"
}

# Outputs from the module
output "instance_public_ip" {
  description = "Public IP of the dev ML instance"
  value       = module.ml_infrastructure.ml_instance_public_ip
}

output "jupyter_url" {
  description = "Jupyter Notebook URL"
  value       = module.ml_infrastructure.jupyter_url
}

output "datasets_bucket" {
  description = "S3 bucket for datasets"
  value       = module.ml_infrastructure.datasets_bucket_name
}

output "models_bucket" {
  description = "S3 bucket for models"
  value       = module.ml_infrastructure.models_bucket_name
}

output "ssh_command" {
  description = "SSH command to connect to instance"
  value       = module.ml_infrastructure.ssh_connection_command
}
