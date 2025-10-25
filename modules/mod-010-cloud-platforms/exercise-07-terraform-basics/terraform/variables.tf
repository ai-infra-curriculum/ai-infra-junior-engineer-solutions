# Input Variables
# Defines all configurable parameters for the infrastructure

# AWS Configuration
variable "aws_region" {
  description = "AWS region where resources will be created"
  type        = string
  default     = "us-east-1"

  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-[0-9]{1}$", var.aws_region))
    error_message = "AWS region must be a valid region name (e.g., us-east-1)."
  }
}

# Project Configuration
variable "project_name" {
  description = "Project name used for resource naming and tagging"
  type        = string
  default     = "ml-infrastructure"

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "owner" {
  description = "Owner or team responsible for the infrastructure"
  type        = string
  default     = "ml-team"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "availability_zones" {
  description = "List of availability zones to use"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24"]
}

# EC2 Configuration
variable "instance_type" {
  description = "EC2 instance type for ML workloads"
  type        = string
  default     = "t3.medium"

  validation {
    condition     = can(regex("^[a-z][0-9][a-z]?\\.[a-z]+$", var.instance_type))
    error_message = "Instance type must be a valid EC2 instance type."
  }
}

variable "root_volume_size" {
  description = "Size of the root EBS volume in GB"
  type        = number
  default     = 50

  validation {
    condition     = var.root_volume_size >= 20 && var.root_volume_size <= 1000
    error_message = "Root volume size must be between 20 and 1000 GB."
  }
}

variable "enable_jupyter" {
  description = "Enable Jupyter Notebook installation via user data"
  type        = bool
  default     = true
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into EC2 instances (use your IP for security)"
  type        = string
  default     = "0.0.0.0/0" # WARNING: Open to internet - change in production!
}

# S3 Configuration
variable "enable_s3_versioning" {
  description = "Enable versioning for S3 buckets"
  type        = bool
  default     = true
}

variable "enable_s3_encryption" {
  description = "Enable server-side encryption for S3 buckets"
  type        = bool
  default     = true
}

variable "s3_lifecycle_days" {
  description = "Number of days before transitioning objects to Glacier"
  type        = number
  default     = 90

  validation {
    condition     = var.s3_lifecycle_days >= 30
    error_message = "Lifecycle days must be at least 30."
  }
}

# ML-Specific Configuration
variable "ml_workload_type" {
  description = "Type of ML workload (training, inference, notebook)"
  type        = string
  default     = "training"

  validation {
    condition     = contains(["training", "inference", "notebook"], var.ml_workload_type)
    error_message = "ML workload type must be training, inference, or notebook."
  }
}

variable "enable_spot_instances" {
  description = "Use spot instances for cost optimization (not recommended for production)"
  type        = bool
  default     = false
}

# Monitoring and Alerting
variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms for monitoring"
  type        = bool
  default     = true
}

variable "cpu_alarm_threshold" {
  description = "CPU utilization threshold for CloudWatch alarm (percentage)"
  type        = number
  default     = 80

  validation {
    condition     = var.cpu_alarm_threshold > 0 && var.cpu_alarm_threshold <= 100
    error_message = "CPU alarm threshold must be between 0 and 100."
  }
}

# Cost Management
variable "enable_cost_tags" {
  description = "Enable detailed cost allocation tags"
  type        = bool
  default     = true
}

variable "cost_center" {
  description = "Cost center for billing purposes"
  type        = string
  default     = "ml-research"
}
