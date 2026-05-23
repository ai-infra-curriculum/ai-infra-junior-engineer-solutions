# AWS Infrastructure Module
# Provisions EKS cluster, S3 buckets, RDS, and supporting infrastructure

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "ml-platform"
      Environment = var.environment
      ManagedBy   = "terraform"
      Cloud       = "aws"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC for EKS cluster
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-vpc-${var.environment}"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment == "dev" ? true : false
  enable_dns_hostnames = true
  enable_dns_support   = true

  public_subnet_tags = {
    "kubernetes.io/role/elb"                                        = "1"
    "kubernetes.io/cluster/${var.project_name}-${var.environment}" = "shared"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"                               = "1"
    "kubernetes.io/cluster/${var.project_name}-${var.environment}" = "shared"
  }

  tags = {
    Name = "${var.project_name}-vpc-${var.environment}"
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "${var.project_name}-${var.environment}"
  cluster_version = var.kubernetes_version

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # OIDC Provider for IRSA
  enable_irsa = true

  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # Managed Node Groups
  eks_managed_node_groups = {
    general = {
      name           = "general"
      instance_types = ["t3.large"]

      min_size     = var.min_nodes
      max_size     = var.max_nodes
      desired_size = var.desired_nodes

      capacity_type = "ON_DEMAND"

      labels = {
        role = "general"
      }

      tags = {
        NodeGroup = "general"
      }
    }

    ml_workloads = {
      name           = "ml-workloads"
      instance_types = ["c5.2xlarge"]

      min_size     = 1
      max_size     = 10
      desired_size = 2

      capacity_type = "SPOT"

      labels = {
        role      = "ml-workloads"
        workload  = "training"
      }

      taints = [{
        key    = "workload"
        value  = "ml"
        effect = "NO_SCHEDULE"
      }]

      tags = {
        NodeGroup = "ml-workloads"
      }
    }

    gpu = {
      name           = "gpu"
      instance_types = ["g4dn.xlarge"]

      min_size     = 0
      max_size     = 5
      desired_size = 0

      capacity_type = "SPOT"

      ami_type = "AL2_x86_64_GPU"

      labels = {
        role             = "gpu"
        "nvidia.com/gpu" = "true"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]

      tags = {
        NodeGroup = "gpu"
      }
    }
  }

  # Cluster security group rules
  cluster_security_group_additional_rules = {
    ingress_from_vpc = {
      description = "Allow all traffic from VPC"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      cidr_blocks = [var.vpc_cidr]
    }
  }

  # Node security group rules
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }

    ingress_cluster_all = {
      description                   = "Cluster to node all ports/protocols"
      protocol                      = "-1"
      from_port                     = 0
      to_port                       = 0
      type                          = "ingress"
      source_cluster_security_group = true
    }
  }

  tags = {
    Environment = var.environment
  }
}

# S3 Buckets for ML artifacts
resource "aws_s3_bucket" "ml_models" {
  bucket = "${var.project_name}-ml-models-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "ML Models"
    Purpose     = "model-storage"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  rule {
    id     = "archive-old-versions"
    status = "Enabled"

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}

resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.project_name}-data-lake-${var.environment}-${data.aws_caller_identity.current.account_id}"

  tags = {
    Name        = "Data Lake"
    Purpose     = "data-storage"
    Environment = var.environment
  }
}

# RDS PostgreSQL for metadata
resource "aws_db_subnet_group" "ml_platform" {
  name       = "${var.project_name}-db-subnet-${var.environment}"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name = "${var.project_name}-db-subnet-${var.environment}"
  }
}

resource "aws_security_group" "rds" {
  name        = "${var.project_name}-rds-sg-${var.environment}"
  description = "Security group for RDS instance"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "PostgreSQL from VPC"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = {
    Name = "${var.project_name}-rds-sg-${var.environment}"
  }
}

resource "aws_db_instance" "ml_platform" {
  identifier = "${var.project_name}-db-${var.environment}"

  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = var.db_instance_class
  allocated_storage    = var.db_allocated_storage
  storage_type         = "gp3"
  storage_encrypted    = true

  db_name  = var.db_name
  username = var.db_username
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.ml_platform.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"

  multi_az               = var.environment == "prod" ? true : false
  publicly_accessible    = false
  skip_final_snapshot    = var.environment == "dev" ? true : false
  final_snapshot_identifier = var.environment == "dev" ? null : "${var.project_name}-db-final-${var.environment}"

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  tags = {
    Name        = "${var.project_name}-db-${var.environment}"
    Environment = var.environment
  }
}

# ElastiCache Redis for caching
resource "aws_elasticache_subnet_group" "ml_platform" {
  name       = "${var.project_name}-redis-subnet-${var.environment}"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name = "${var.project_name}-redis-subnet-${var.environment}"
  }
}

resource "aws_security_group" "redis" {
  name        = "${var.project_name}-redis-sg-${var.environment}"
  description = "Security group for Redis cluster"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Redis from VPC"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  tags = {
    Name = "${var.project_name}-redis-sg-${var.environment}"
  }
}

resource "aws_elasticache_replication_group" "ml_platform" {
  replication_group_id       = "${var.project_name}-redis-${var.environment}"
  replication_group_description = "Redis cluster for ML platform"

  engine               = "redis"
  engine_version       = "7.0"
  node_type            = var.redis_node_type
  num_cache_clusters   = var.environment == "prod" ? 2 : 1
  port                 = 6379

  subnet_group_name    = aws_elasticache_subnet_group.ml_platform.name
  security_group_ids   = [aws_security_group.redis.id]

  automatic_failover_enabled = var.environment == "prod" ? true : false
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  snapshot_retention_limit = 5
  snapshot_window         = "03:00-05:00"

  tags = {
    Name        = "${var.project_name}-redis-${var.environment}"
    Environment = var.environment
  }
}

# IAM Role for Lambda functions
resource "aws_iam_role" "lambda_execution" {
  name = "${var.project_name}-lambda-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = {
    Name = "${var.project_name}-lambda-role-${var.environment}"
  }
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_custom" {
  name = "lambda-custom-policy"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.ml_models.arn,
          "${aws_s3_bucket.ml_models.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "rds:DescribeDBInstances",
          "rds:Connect"
        ]
        Resource = "*"
      }
    ]
  })
}

# ECR Repositories for container images
resource "aws_ecr_repository" "api_gateway" {
  name                 = "${var.project_name}/api-gateway"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name = "api-gateway"
  }
}

resource "aws_ecr_repository" "model_serving" {
  name                 = "${var.project_name}/model-serving"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name = "model-serving"
  }
}

resource "aws_ecr_lifecycle_policy" "cleanup_old_images" {
  repository = aws_ecr_repository.api_gateway.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus     = "any"
        countType     = "imageCountMoreThan"
        countNumber   = 10
      }
      action = {
        type = "expire"
      }
    }]
  })
}
