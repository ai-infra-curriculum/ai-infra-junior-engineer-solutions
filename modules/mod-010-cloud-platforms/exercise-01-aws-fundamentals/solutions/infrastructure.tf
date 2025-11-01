# AWS ML Infrastructure - Terraform Configuration
#
# This configuration creates:
# - VPC with public and private subnets
# - S3 bucket for ML data and models
# - IAM roles and policies
# - Security groups
# - EC2 launch template for GPU instances
#
# Usage:
#   terraform init
#   terraform plan
#   terraform apply

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment to use S3 backend for state
  # backend "s3" {
  #   bucket = "my-terraform-state"
  #   key    = "ml-infrastructure/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "ML-Infrastructure"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "ml-infrastructure"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Change to your IP for security
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

# Public Subnet
resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public-subnet-${count.index + 1}"
  }
}

# Private Subnet
resource "aws_subnet" "private" {
  count = 2

  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "${var.project_name}-private-subnet-${count.index + 1}"
  }
}

# NAT Gateway (for private subnet internet access)
resource "aws_eip" "nat" {
  count  = 1
  domain = "vpc"

  tags = {
    Name = "${var.project_name}-nat-eip"
  }
}

resource "aws_nat_gateway" "main" {
  count         = 1
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "${var.project_name}-nat-gateway"
  }

  depends_on = [aws_internet_gateway.main]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[0].id
  }

  tags = {
    Name = "${var.project_name}-private-rt"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = length(aws_subnet.private)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# S3 Bucket for ML Data
resource "aws_s3_bucket" "ml_data" {
  bucket = "${var.project_name}-ml-data-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "ML Data Bucket"
    Purpose     = "Store training data and models"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "ml_data" {
  bucket = aws_s3_bucket.ml_data.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "ml_data" {
  bucket = aws_s3_bucket.ml_data.id

  rule {
    id     = "archive-old-models"
    status = "Enabled"

    filter {
      prefix = "models/"
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }

  rule {
    id     = "delete-temp-data"
    status = "Enabled"

    filter {
      prefix = "temp/"
    }

    expiration {
      days = 7
    }
  }
}

resource "aws_s3_bucket_public_access_block" "ml_data" {
  bucket = aws_s3_bucket.ml_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# VPC Endpoint for S3 (for private access)
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.main.id
  service_name = "com.amazonaws.${var.aws_region}.s3"

  route_table_ids = [
    aws_route_table.private.id,
    aws_route_table.public.id
  ]

  tags = {
    Name = "${var.project_name}-s3-endpoint"
  }
}

# Security Group for ML Instances
resource "aws_security_group" "ml_instance" {
  name        = "${var.project_name}-ml-instance-sg"
  description = "Security group for ML training instances"
  vpc_id      = aws_vpc.main.id

  # SSH access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # Jupyter
  ingress {
    description = "Jupyter"
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # TensorBoard
  ingress {
    description = "TensorBoard"
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # Outbound internet access
  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-ml-instance-sg"
  }
}

# IAM Role for EC2 Instances
resource "aws_iam_role" "ml_instance" {
  name = "${var.project_name}-ml-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = {
    Name = "${var.project_name}-ml-instance-role"
  }
}

# IAM Policy for S3 Access
resource "aws_iam_policy" "ml_s3_access" {
  name        = "${var.project_name}-ml-s3-access"
  description = "Allow ML instances to access S3 bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.ml_data.arn,
          "${aws_s3_bucket.ml_data.arn}/*"
        ]
      }
    ]
  })
}

# Attach policy to role
resource "aws_iam_role_policy_attachment" "ml_s3_access" {
  role       = aws_iam_role.ml_instance.name
  policy_arn = aws_iam_policy.ml_s3_access.arn
}

# Attach CloudWatch logs policy
resource "aws_iam_role_policy_attachment" "cloudwatch_logs" {
  role       = aws_iam_role.ml_instance.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

# Instance Profile
resource "aws_iam_instance_profile" "ml_instance" {
  name = "${var.project_name}-ml-instance-profile"
  role = aws_iam_role.ml_instance.name

  tags = {
    Name = "${var.project_name}-ml-instance-profile"
  }
}

# IAM Role for SageMaker
resource "aws_iam_role" "sagemaker" {
  name = "${var.project_name}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
    }]
  })

  tags = {
    Name = "${var.project_name}-sagemaker-role"
  }
}

# Attach SageMaker full access
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# SSH Key Pair
resource "aws_key_pair" "ml_key" {
  key_name   = "${var.project_name}-key"
  public_key = file("~/.ssh/id_rsa.pub")  # Update path to your public key

  tags = {
    Name = "${var.project_name}-key"
  }
}

# Launch Template for GPU Instances
resource "aws_launch_template" "ml_gpu" {
  name_prefix   = "${var.project_name}-gpu-"
  image_id      = data.aws_ami.deep_learning.id
  instance_type = "p3.2xlarge"
  key_name      = aws_key_pair.ml_key.key_name

  iam_instance_profile {
    name = aws_iam_instance_profile.ml_instance.name
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups             = [aws_security_group.ml_instance.id]
  }

  block_device_mappings {
    device_name = "/dev/sda1"

    ebs {
      volume_size           = 200
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = true
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  tag_specifications {
    resource_type = "instance"

    tags = {
      Name        = "${var.project_name}-gpu-instance"
      Type        = "ML-Training"
      Environment = var.environment
    }
  }

  user_data = base64encode(<<-EOF
              #!/bin/bash
              # Install CloudWatch agent
              wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
              dpkg -i -E ./amazon-cloudwatch-agent.deb

              # Configure GPU monitoring
              nvidia-smi -pm 1
              nvidia-smi -acp 0

              # Mount EFS (if using)
              # echo "${aws_efs_file_system.ml_storage.id}:/ /mnt/efs efs defaults,_netdev 0 0" >> /etc/fstab
              # mount -a

              # Set up Jupyter
              su - ubuntu -c "jupyter notebook --generate-config"
              su - ubuntu -c "echo \"c.NotebookApp.ip = '0.0.0.0'\" >> ~/.jupyter/jupyter_notebook_config.py"
              su - ubuntu -c "echo \"c.NotebookApp.open_browser = False\" >> ~/.jupyter/jupyter_notebook_config.py"
              EOF
  )

  tags = {
    Name = "${var.project_name}-gpu-launch-template"
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "ml_training" {
  name              = "/aws/ml-training/${var.project_name}"
  retention_in_days = 7

  tags = {
    Name = "${var.project_name}-training-logs"
  }
}

# Data Sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "s3_bucket_name" {
  description = "S3 bucket name for ML data"
  value       = aws_s3_bucket.ml_data.id
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.ml_data.arn
}

output "ml_instance_security_group_id" {
  description = "Security group ID for ML instances"
  value       = aws_security_group.ml_instance.id
}

output "ml_instance_role_arn" {
  description = "IAM role ARN for ML instances"
  value       = aws_iam_role.ml_instance.arn
}

output "sagemaker_role_arn" {
  description = "IAM role ARN for SageMaker"
  value       = aws_iam_role.sagemaker.arn
}

output "launch_template_id" {
  description = "Launch template ID for GPU instances"
  value       = aws_launch_template.ml_gpu.id
}

output "deep_learning_ami_id" {
  description = "Deep Learning AMI ID"
  value       = data.aws_ami.deep_learning.id
}
