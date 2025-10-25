# Production Environment Configuration
# This is an example of how to use the ML infrastructure modules for production

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Production should use remote state
  # Uncomment and configure after creating the S3 bucket and DynamoDB table
  # backend "s3" {
  #   bucket         = "your-company-terraform-state"
  #   key            = "ml-infrastructure/prod/terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   dynamodb_table = "terraform-state-lock"
  # }
}

provider "aws" {
  region = var.aws_region

  # Production should have additional tags
  default_tags {
    tags = {
      ManagedBy   = "Terraform"
      Environment = "Production"
      Compliance  = "Required"
    }
  }
}

# Use the parent terraform configuration as a module
module "ml_infrastructure" {
  source = "../../terraform"

  # Project Configuration
  project_name = "ml-prod"
  environment  = "prod"
  owner        = "ml-team"
  aws_region   = var.aws_region

  # VPC Configuration (production-ready)
  vpc_cidr             = "10.100.0.0/16"
  availability_zones   = ["us-east-1a", "us-east-1b", "us-east-1c"]
  public_subnet_cidrs  = ["10.100.1.0/24", "10.100.2.0/24", "10.100.3.0/24"]
  private_subnet_cidrs = ["10.100.11.0/24", "10.100.12.0/24", "10.100.13.0/24"]

  # EC2 Configuration (production instance)
  instance_type     = "t3.large"
  root_volume_size  = 100
  enable_jupyter    = false # Disable in prod for security
  allowed_ssh_cidr  = var.allowed_ssh_cidr

  # S3 Configuration (full protection)
  enable_s3_versioning = true
  enable_s3_encryption = true
  s3_lifecycle_days    = 90

  # ML Configuration
  ml_workload_type      = "training"
  enable_spot_instances = false

  # Monitoring (full monitoring in prod)
  enable_cloudwatch_alarms = true
  cpu_alarm_threshold      = 75

  # Cost Management
  enable_cost_tags = true
  cost_center      = "ml-production"
}

# Outputs from the module
output "instance_id" {
  description = "Instance ID for production ML server"
  value       = module.ml_infrastructure.ml_instance_id
}

output "instance_private_ip" {
  description = "Private IP of production ML instance"
  value       = module.ml_infrastructure.ml_instance_private_ip
}

output "datasets_bucket" {
  description = "S3 bucket for production datasets"
  value       = module.ml_infrastructure.datasets_bucket_name
}

output "models_bucket" {
  description = "S3 bucket for production models"
  value       = module.ml_infrastructure.models_bucket_name
}

output "iam_role_arn" {
  description = "IAM role ARN for the ML instance"
  value       = module.ml_infrastructure.ml_instance_role_arn
}

output "vpc_id" {
  description = "VPC ID for production environment"
  value       = module.ml_infrastructure.vpc_id
}

output "infrastructure_summary" {
  description = "Production infrastructure summary"
  value       = module.ml_infrastructure.infrastructure_summary
  sensitive   = true
}
