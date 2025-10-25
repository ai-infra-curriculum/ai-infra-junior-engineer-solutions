# AWS Module Outputs

output "eks_cluster_id" {
  description = "EKS cluster ID"
  value       = module.eks.cluster_id
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_certificate_authority_data" {
  description = "EKS cluster certificate authority data"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "eks_oidc_provider_arn" {
  description = "ARN of the OIDC provider for IRSA"
  value       = module.eks.oidc_provider_arn
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = module.vpc.public_subnets
}

output "s3_ml_models_bucket" {
  description = "S3 bucket name for ML models"
  value       = aws_s3_bucket.ml_models.id
}

output "s3_ml_models_bucket_arn" {
  description = "S3 bucket ARN for ML models"
  value       = aws_s3_bucket.ml_models.arn
}

output "s3_data_lake_bucket" {
  description = "S3 bucket name for data lake"
  value       = aws_s3_bucket.data_lake.id
}

output "s3_data_lake_bucket_arn" {
  description = "S3 bucket ARN for data lake"
  value       = aws_s3_bucket.data_lake.arn
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.ml_platform.endpoint
}

output "rds_address" {
  description = "RDS instance address"
  value       = aws_db_instance.ml_platform.address
}

output "rds_port" {
  description = "RDS instance port"
  value       = aws_db_instance.ml_platform.port
}

output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = aws_elasticache_replication_group.ml_platform.primary_endpoint_address
}

output "redis_port" {
  description = "Redis port"
  value       = aws_elasticache_replication_group.ml_platform.port
}

output "ecr_api_gateway_repository_url" {
  description = "ECR repository URL for API gateway"
  value       = aws_ecr_repository.api_gateway.repository_url
}

output "ecr_model_serving_repository_url" {
  description = "ECR repository URL for model serving"
  value       = aws_ecr_repository.model_serving.repository_url
}

output "lambda_execution_role_arn" {
  description = "IAM role ARN for Lambda execution"
  value       = aws_iam_role.lambda_execution.arn
}

output "region" {
  description = "AWS region"
  value       = var.aws_region
}
