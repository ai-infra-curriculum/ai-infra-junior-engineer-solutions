# Output Values
# Exports important resource information for use by other modules or for reference

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = aws_subnet.private[*].id
}

# EC2 Outputs
output "ml_instance_id" {
  description = "ID of the ML training EC2 instance"
  value       = aws_instance.ml_training.id
}

output "ml_instance_public_ip" {
  description = "Public IP address of the ML training instance"
  value       = aws_instance.ml_training.public_ip
}

output "ml_instance_private_ip" {
  description = "Private IP address of the ML training instance"
  value       = aws_instance.ml_training.private_ip
}

output "ml_instance_arn" {
  description = "ARN of the ML training instance"
  value       = aws_instance.ml_training.arn
}

output "jupyter_url" {
  description = "URL to access Jupyter Notebook (if enabled)"
  value       = var.enable_jupyter ? "http://${aws_instance.ml_training.public_ip}:8888" : "Jupyter not enabled"
}

# S3 Outputs
output "datasets_bucket_name" {
  description = "Name of the S3 bucket for ML datasets"
  value       = aws_s3_bucket.ml_datasets.id
}

output "datasets_bucket_arn" {
  description = "ARN of the datasets S3 bucket"
  value       = aws_s3_bucket.ml_datasets.arn
}

output "models_bucket_name" {
  description = "Name of the S3 bucket for ML models"
  value       = aws_s3_bucket.ml_models.id
}

output "models_bucket_arn" {
  description = "ARN of the models S3 bucket"
  value       = aws_s3_bucket.ml_models.arn
}

output "datasets_bucket_region" {
  description = "Region where datasets bucket is created"
  value       = aws_s3_bucket.ml_datasets.region
}

# IAM Outputs
output "ml_instance_role_name" {
  description = "Name of the IAM role attached to ML instance"
  value       = aws_iam_role.ml_instance_role.name
}

output "ml_instance_role_arn" {
  description = "ARN of the IAM role attached to ML instance"
  value       = aws_iam_role.ml_instance_role.arn
}

output "ml_instance_profile_name" {
  description = "Name of the instance profile"
  value       = aws_iam_instance_profile.ml_instance.name
}

# Security Group Outputs
output "ml_instance_security_group_id" {
  description = "ID of the security group for ML instance"
  value       = aws_security_group.ml_instance.id
}

# Connection Information
output "ssh_connection_command" {
  description = "SSH command to connect to the ML instance"
  value       = "ssh -i ~/.ssh/your-key.pem ec2-user@${aws_instance.ml_training.public_ip}"
}

output "aws_console_instance_url" {
  description = "AWS Console URL for the EC2 instance"
  value       = "https://console.aws.amazon.com/ec2/v2/home?region=${var.aws_region}#Instances:instanceId=${aws_instance.ml_training.id}"
}

# Summary Output
output "infrastructure_summary" {
  description = "Summary of deployed infrastructure"
  value = {
    environment      = var.environment
    region           = var.aws_region
    vpc_id           = aws_vpc.main.id
    instance_id      = aws_instance.ml_training.id
    public_ip        = aws_instance.ml_training.public_ip
    datasets_bucket  = aws_s3_bucket.ml_datasets.id
    models_bucket    = aws_s3_bucket.ml_models.id
    instance_profile = aws_iam_instance_profile.ml_instance.name
  }
}

# Cost Estimation (informational)
output "estimated_monthly_cost" {
  description = "Estimated monthly cost in USD (approximate)"
  value = {
    ec2_instance = "~$30 (t3.medium running 24/7)"
    ebs_volume   = "~$5 (50GB gp3)"
    s3_storage   = "~$0.023 per GB per month"
    data_transfer = "Variable based on usage"
    total_minimum = "~$35/month + S3 storage and transfer"
  }
}
