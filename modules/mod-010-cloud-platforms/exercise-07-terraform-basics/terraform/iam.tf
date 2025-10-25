# IAM Configuration
# Creates IAM roles and policies with least privilege for ML instances

# IAM Role for EC2 Instance
resource "aws_iam_role" "ml_instance_role" {
  name        = "${local.name_prefix}-ml-instance-role"
  description = "IAM role for ML training EC2 instances"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "EC2AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-ml-instance-role"
    }
  )
}

# IAM Policy for S3 Access (Least Privilege)
resource "aws_iam_policy" "s3_ml_access" {
  name        = "${local.name_prefix}-s3-ml-access"
  description = "Policy for ML instance to access S3 buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListSpecificBuckets"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.ml_datasets.arn,
          aws_s3_bucket.ml_models.arn
        ]
      },
      {
        Sid    = "ReadWriteObjects"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:GetObjectVersion"
        ]
        Resource = [
          "${aws_s3_bucket.ml_datasets.arn}/*",
          "${aws_s3_bucket.ml_models.arn}/*"
        ]
      },
      {
        Sid    = "ListAllBuckets"
        Effect = "Allow"
        Action = [
          "s3:ListAllMyBuckets"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}

# IAM Policy for CloudWatch Logs
resource "aws_iam_policy" "cloudwatch_logs" {
  name        = "${local.name_prefix}-cloudwatch-logs"
  description = "Policy for ML instance to write to CloudWatch Logs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CreateLogGroup"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
      },
      {
        Sid    = "CreateLogStreamAndPutLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/ec2/${local.name_prefix}-ml-training:*"
      }
    ]
  })

  tags = local.common_tags
}

# IAM Policy for CloudWatch Metrics
resource "aws_iam_policy" "cloudwatch_metrics" {
  name        = "${local.name_prefix}-cloudwatch-metrics"
  description = "Policy for ML instance to publish custom metrics"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "PutMetrics"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = "${var.project_name}/${var.environment}"
          }
        }
      },
      {
        Sid    = "GetMetrics"
        Effect = "Allow"
        Action = [
          "cloudwatch:GetMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}

# IAM Policy for EC2 Instance Metadata
resource "aws_iam_policy" "ec2_describe" {
  name        = "${local.name_prefix}-ec2-describe"
  description = "Policy for ML instance to describe itself"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "DescribeInstance"
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeTags",
          "ec2:DescribeVolumes"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}

# Attach S3 Access Policy to Role
resource "aws_iam_role_policy_attachment" "s3_access" {
  role       = aws_iam_role.ml_instance_role.name
  policy_arn = aws_iam_policy.s3_ml_access.arn
}

# Attach CloudWatch Logs Policy to Role
resource "aws_iam_role_policy_attachment" "cloudwatch_logs" {
  role       = aws_iam_role.ml_instance_role.name
  policy_arn = aws_iam_policy.cloudwatch_logs.arn
}

# Attach CloudWatch Metrics Policy to Role
resource "aws_iam_role_policy_attachment" "cloudwatch_metrics" {
  role       = aws_iam_role.ml_instance_role.name
  policy_arn = aws_iam_policy.cloudwatch_metrics.arn
}

# Attach EC2 Describe Policy to Role
resource "aws_iam_role_policy_attachment" "ec2_describe" {
  role       = aws_iam_role.ml_instance_role.name
  policy_arn = aws_iam_policy.ec2_describe.arn
}

# Attach SSM Managed Policy (for Systems Manager access)
resource "aws_iam_role_policy_attachment" "ssm_managed_instance" {
  role       = aws_iam_role.ml_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# IAM Instance Profile
resource "aws_iam_instance_profile" "ml_instance" {
  name = "${local.name_prefix}-ml-instance-profile"
  role = aws_iam_role.ml_instance_role.name

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-ml-instance-profile"
    }
  )
}

# Optional: IAM User for CI/CD or External Access
resource "aws_iam_user" "ml_ci_user" {
  count = var.environment == "prod" ? 1 : 0
  name  = "${local.name_prefix}-ci-user"

  tags = merge(
    local.common_tags,
    {
      Name    = "${local.name_prefix}-ci-user"
      Purpose = "CI/CD Pipeline Access"
    }
  )
}

# Attach read-only S3 access to CI user
resource "aws_iam_user_policy_attachment" "ci_s3_readonly" {
  count      = var.environment == "prod" ? 1 : 0
  user       = aws_iam_user.ml_ci_user[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

# Output IAM policy for reference
output "iam_policy_document_s3" {
  description = "IAM policy document for S3 access (for reference)"
  value       = aws_iam_policy.s3_ml_access.policy
}
