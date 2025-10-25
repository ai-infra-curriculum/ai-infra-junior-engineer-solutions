# S3 Configuration
# Creates S3 buckets for ML datasets and models with security and lifecycle policies

# S3 Bucket for ML Datasets
resource "aws_s3_bucket" "ml_datasets" {
  bucket = local.datasets_bucket_name

  tags = merge(
    local.common_tags,
    {
      Name    = "${local.name_prefix}-datasets"
      Purpose = "ML Datasets Storage"
    }
  )
}

# S3 Bucket for ML Models
resource "aws_s3_bucket" "ml_models" {
  bucket = local.models_bucket_name

  tags = merge(
    local.common_tags,
    {
      Name    = "${local.name_prefix}-models"
      Purpose = "ML Models Storage"
    }
  )
}

# Block Public Access for Datasets Bucket
resource "aws_s3_bucket_public_access_block" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Block Public Access for Models Bucket
resource "aws_s3_bucket_public_access_block" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable Versioning for Datasets Bucket
resource "aws_s3_bucket_versioning" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  versioning_configuration {
    status = var.enable_s3_versioning ? "Enabled" : "Suspended"
  }
}

# Enable Versioning for Models Bucket
resource "aws_s3_bucket_versioning" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  versioning_configuration {
    status = var.enable_s3_versioning ? "Enabled" : "Suspended"
  }
}

# Server-Side Encryption for Datasets Bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "ml_datasets" {
  count  = var.enable_s3_encryption ? 1 : 0
  bucket = aws_s3_bucket.ml_datasets.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# Server-Side Encryption for Models Bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "ml_models" {
  count  = var.enable_s3_encryption ? 1 : 0
  bucket = aws_s3_bucket.ml_models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# Lifecycle Policy for Datasets Bucket
resource "aws_s3_bucket_lifecycle_configuration" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  rule {
    id     = "transition-to-glacier"
    status = "Enabled"

    transition {
      days          = var.s3_lifecycle_days
      storage_class = "GLACIER"
    }

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }

  rule {
    id     = "delete-incomplete-multipart-uploads"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Lifecycle Policy for Models Bucket
resource "aws_s3_bucket_lifecycle_configuration" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  rule {
    id     = "transition-old-versions"
    status = "Enabled"

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }

  rule {
    id     = "delete-incomplete-multipart-uploads"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# S3 Bucket Policy for Datasets Bucket (restrict to instance role)
resource "aws_s3_bucket_policy" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowMLInstanceAccess"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.ml_instance_role.arn
        }
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.ml_datasets.arn,
          "${aws_s3_bucket.ml_datasets.arn}/*"
        ]
      },
      {
        Sid    = "DenyInsecureTransport"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.ml_datasets.arn,
          "${aws_s3_bucket.ml_datasets.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })
}

# S3 Bucket Policy for Models Bucket
resource "aws_s3_bucket_policy" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowMLInstanceAccess"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.ml_instance_role.arn
        }
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
        Sid    = "DenyInsecureTransport"
        Effect = "Deny"
        Principal = "*"
        Action = "s3:*"
        Resource = [
          aws_s3_bucket.ml_models.arn,
          "${aws_s3_bucket.ml_models.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })
}

# Create folder structure in datasets bucket
resource "aws_s3_object" "datasets_folders" {
  for_each = toset([
    "raw/",
    "processed/",
    "interim/",
    "external/"
  ])

  bucket       = aws_s3_bucket.ml_datasets.id
  key          = each.value
  content_type = "application/x-directory"
}

# Create folder structure in models bucket
resource "aws_s3_object" "models_folders" {
  for_each = toset([
    "trained/",
    "artifacts/",
    "experiments/",
    "production/"
  ])

  bucket       = aws_s3_bucket.ml_models.id
  key          = each.value
  content_type = "application/x-directory"
}

# S3 Bucket notification (optional - for triggering ML pipelines)
# Uncomment if you want to enable S3 event notifications
# resource "aws_s3_bucket_notification" "ml_datasets" {
#   bucket = aws_s3_bucket.ml_datasets.id
#
#   lambda_function {
#     lambda_function_arn = aws_lambda_function.process_dataset.arn
#     events              = ["s3:ObjectCreated:*"]
#     filter_prefix       = "raw/"
#   }
# }
