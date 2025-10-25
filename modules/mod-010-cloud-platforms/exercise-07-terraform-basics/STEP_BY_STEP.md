# Step-by-Step Implementation Guide

This guide walks you through implementing the ML infrastructure from scratch, explaining every decision and teaching Terraform best practices along the way.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Phase 1: Project Setup](#phase-1-project-setup)
3. [Phase 2: Provider Configuration](#phase-2-provider-configuration)
4. [Phase 3: VPC and Networking](#phase-3-vpc-and-networking)
5. [Phase 4: S3 Storage](#phase-4-s3-storage)
6. [Phase 5: IAM Roles](#phase-5-iam-roles)
7. [Phase 6: EC2 Instances](#phase-6-ec2-instances)
8. [Phase 7: Monitoring](#phase-7-monitoring)
9. [Phase 8: Testing](#phase-8-testing)
10. [Phase 9: Production Readiness](#phase-9-production-readiness)

---

## Prerequisites

Before starting, ensure you have:

1. **AWS Account**: With appropriate permissions
2. **AWS CLI**: Installed and configured
3. **Terraform**: Version 1.0 or later
4. **Git**: For version control
5. **Text Editor**: VS Code, Vim, or your preferred editor

### Verify Prerequisites

```bash
# Check AWS CLI
aws --version
aws sts get-caller-identity

# Check Terraform
terraform version

# Check Git
git --version
```

---

## Phase 1: Project Setup

### Step 1.1: Create Project Structure

```bash
# Create project directory
mkdir -p exercise-07-terraform-basics
cd exercise-07-terraform-basics

# Create directory structure
mkdir -p terraform examples/{dev,prod} tests scripts docs

# Initialize Git
git init
```

### Step 1.2: Create .gitignore

Create `.gitignore` to prevent committing sensitive files:

```bash
cat > .gitignore << 'EOF'
# Terraform files
**/.terraform/*
*.tfstate
*.tfstate.*
*.tfvars
.terraform.lock.hcl

# OS files
.DS_Store
Thumbs.db

# IDE files
.vscode/
.idea/
*.swp
EOF
```

**Why?**
- `*.tfstate`: Contains sensitive data and resource IDs
- `*.tfvars`: May contain credentials or secrets
- `.terraform/`: Provider plugins (downloaded per-project)

### Step 1.3: Commit Initial Structure

```bash
git add .gitignore
git commit -m "Initial project structure"
```

---

## Phase 2: Provider Configuration

### Step 2.1: Create providers.tf

Navigate to the terraform directory and create `providers.tf`:

```bash
cd terraform
```

```hcl
# terraform/providers.tf
terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

provider "random" {}
```

**Key Concepts**:

1. **required_version**: Ensures team uses compatible Terraform version
2. **required_providers**: Pins provider versions for consistency
3. **default_tags**: Automatically applies tags to all resources
4. **~> 5.0**: Allows minor version updates (5.0, 5.1, ...) but not major (6.0)

### Step 2.2: Create variables.tf

Define all input variables:

```hcl
# terraform/variables.tf
variable "aws_region" {
  description = "AWS region where resources will be created"
  type        = string
  default     = "us-east-1"

  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-[0-9]{1}$", var.aws_region))
    error_message = "AWS region must be a valid region name."
  }
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "ml-infrastructure"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

# Add more variables as needed...
```

**Best Practices**:

1. **Descriptions**: Explain purpose of each variable
2. **Types**: Enforce correct data types
3. **Defaults**: Provide sensible defaults for optional variables
4. **Validation**: Catch errors early with validation rules

### Step 2.3: Create outputs.tf

Define output values:

```hcl
# terraform/outputs.tf
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "infrastructure_summary" {
  description = "Summary of deployed infrastructure"
  value = {
    environment = var.environment
    region      = var.aws_region
    vpc_id      = aws_vpc.main.id
  }
}
```

**Why Outputs?**
- Share information between modules
- Display important IDs after apply
- Use in scripts or CI/CD pipelines

### Step 2.4: Create main.tf

Main orchestration file:

```hcl
# terraform/main.tf
# Generate random suffix for unique names
resource "random_id" "suffix" {
  byte_length = 4
}

# Local values for consistency
locals {
  common_tags = {
    Project      = var.project_name
    Environment  = var.environment
    ManagedBy    = "Terraform"
  }

  name_prefix = "${var.project_name}-${var.environment}"
}
```

**Locals vs Variables**:
- **Variables**: User inputs
- **Locals**: Computed/derived values

### Step 2.5: Initialize Terraform

```bash
terraform init
```

**What happens?**
1. Downloads AWS provider plugin
2. Downloads random provider plugin
3. Creates `.terraform` directory
4. Creates `.terraform.lock.hcl` (dependency lock file)

### Step 2.6: Validate Configuration

```bash
terraform validate
terraform fmt
```

**Commands**:
- `validate`: Checks syntax and configuration
- `fmt`: Formats code consistently

---

## Phase 3: VPC and Networking

### Step 3.1: Create vpc.tf

```hcl
# terraform/vpc.tf
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-vpc"
    }
  )
}
```

**Why VPC?**
- Isolated network environment
- Control over IP addressing
- Security at network level
- Required for EC2 instances

### Step 3.2: Add Internet Gateway

```hcl
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-igw"
    }
  )
}
```

**Purpose**: Allows resources in public subnets to access internet

### Step 3.3: Create Public Subnets

```hcl
resource "aws_subnet" "public" {
  count                   = length(var.public_subnet_cidrs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-public-subnet-${count.index + 1}"
      Type = "Public"
    }
  )
}
```

**Key Features**:
- `count`: Creates multiple subnets dynamically
- `map_public_ip_on_launch`: Auto-assign public IPs
- Multiple AZs: High availability

### Step 3.4: Create Private Subnets

```hcl
resource "aws_subnet" "private" {
  count             = length(var.private_subnet_cidrs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-private-subnet-${count.index + 1}"
      Type = "Private"
    }
  )
}
```

**Private Subnets**:
- No direct internet access
- For databases, backend services
- Access internet via NAT Gateway

### Step 3.5: Add NAT Gateway

```hcl
resource "aws_eip" "nat" {
  domain = "vpc"

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-nat-eip"
    }
  )

  depends_on = [aws_internet_gateway.main]
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-nat-gateway"
    }
  )

  depends_on = [aws_internet_gateway.main]
}
```

**NAT Gateway**:
- Allows private subnet resources to reach internet
- One-way: Outbound only
- Cost: ~$32/month + data transfer

### Step 3.6: Configure Route Tables

```hcl
# Public route table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-public-rt"
    }
  )
}

# Private route table
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-private-rt"
    }
  )
}

# Associate subnets with route tables
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
```

**Route Tables**:
- Define how traffic flows
- Public: Routes to Internet Gateway
- Private: Routes to NAT Gateway

### Step 3.7: Test VPC Configuration

```bash
terraform plan
```

Review the plan to see what will be created:
- 1 VPC
- 1 Internet Gateway
- 1 NAT Gateway
- 1 Elastic IP
- 2 Public Subnets
- 2 Private Subnets
- 2 Route Tables
- 4 Route Table Associations

```bash
terraform apply
```

Type `yes` to confirm.

### Step 3.8: Verify VPC Creation

```bash
# Get VPC ID
terraform output vpc_id

# Verify in AWS
aws ec2 describe-vpcs --vpc-ids $(terraform output -raw vpc_id)
```

---

## Phase 4: S3 Storage

### Step 4.1: Create s3.tf

```hcl
# terraform/s3.tf
resource "aws_s3_bucket" "ml_datasets" {
  bucket = "${local.name_prefix}-datasets-${random_id.suffix.hex}"

  tags = merge(
    local.common_tags,
    {
      Name    = "${local.name_prefix}-datasets"
      Purpose = "ML Datasets Storage"
    }
  )
}

resource "aws_s3_bucket" "ml_models" {
  bucket = "${local.name_prefix}-models-${random_id.suffix.hex}"

  tags = merge(
    local.common_tags,
    {
      Name    = "${local.name_prefix}-models"
      Purpose = "ML Models Storage"
    }
  )
}
```

**Why Random Suffix?**
- S3 bucket names must be globally unique
- Random suffix prevents naming conflicts

### Step 4.2: Block Public Access

```hcl
resource "aws_s3_bucket_public_access_block" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

**Security**: Prevents accidental public exposure

### Step 4.3: Enable Versioning

```hcl
resource "aws_s3_bucket_versioning" "ml_datasets" {
  bucket = aws_s3_bucket.ml_datasets.id

  versioning_configuration {
    status = var.enable_s3_versioning ? "Enabled" : "Suspended"
  }
}
```

**Versioning Benefits**:
- Recover from accidental deletion
- Track changes over time
- Rollback to previous versions

### Step 4.4: Enable Encryption

```hcl
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
```

**Encryption**:
- AES256: AWS-managed keys (free)
- Alternative: KMS for custom keys (additional cost)

### Step 4.5: Configure Lifecycle Policies

```hcl
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
}
```

**Lifecycle Policies**:
- Reduce costs by archiving old data
- Glacier: ~80% cheaper than standard S3
- Automatic: No manual intervention needed

### Step 4.6: Create Folder Structure

```hcl
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
```

**Organization**:
- Consistent structure across environments
- Clear data flow: raw → interim → processed
- Easy to navigate and manage

### Step 4.7: Apply S3 Configuration

```bash
terraform plan
terraform apply
```

### Step 4.8: Verify S3 Buckets

```bash
# List buckets
aws s3 ls | grep ml-infrastructure

# Check bucket contents
aws s3 ls s3://$(terraform output -raw datasets_bucket_name)/
```

---

## Phase 5: IAM Roles

### Step 5.1: Create iam.tf

```hcl
# terraform/iam.tf
resource "aws_iam_role" "ml_instance_role" {
  name        = "${local.name_prefix}-ml-instance-role"
  description = "IAM role for ML training EC2 instances"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}
```

**Assume Role Policy**: Allows EC2 to use this role

### Step 5.2: Create S3 Access Policy

```hcl
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
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.ml_datasets.arn}/*",
          "${aws_s3_bucket.ml_models.arn}/*"
        ]
      }
    ]
  })
}
```

**Least Privilege**:
- Only access to specific buckets
- Separate permissions for buckets vs objects
- No unnecessary permissions

### Step 5.3: Create CloudWatch Policy

```hcl
resource "aws_iam_policy" "cloudwatch_logs" {
  name        = "${local.name_prefix}-cloudwatch-logs"
  description = "Policy for ML instance to write logs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/aws/ec2/${local.name_prefix}-ml-training:*"
      }
    ]
  })
}
```

**Monitoring**: Allows instance to send logs to CloudWatch

### Step 5.4: Attach Policies to Role

```hcl
resource "aws_iam_role_policy_attachment" "s3_access" {
  role       = aws_iam_role.ml_instance_role.name
  policy_arn = aws_iam_policy.s3_ml_access.arn
}

resource "aws_iam_role_policy_attachment" "cloudwatch_logs" {
  role       = aws_iam_role.ml_instance_role.name
  policy_arn = aws_iam_policy.cloudwatch_logs.arn
}
```

### Step 5.5: Create Instance Profile

```hcl
resource "aws_iam_instance_profile" "ml_instance" {
  name = "${local.name_prefix}-ml-instance-profile"
  role = aws_iam_role.ml_instance_role.name

  tags = local.common_tags
}
```

**Instance Profile**: Connects IAM role to EC2 instance

### Step 5.6: Apply IAM Configuration

```bash
terraform plan
terraform apply
```

---

## Phase 6: EC2 Instances

### Step 6.1: Create ec2.tf

Start with security group:

```hcl
# terraform/ec2.tf
resource "aws_security_group" "ml_instance" {
  name        = "${local.name_prefix}-ml-instance-sg"
  description = "Security group for ML training instance"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  dynamic "ingress" {
    for_each = var.enable_jupyter ? [1] : []
    content {
      description = "Jupyter Notebook"
      from_port   = 8888
      to_port     = 8888
      protocol    = "tcp"
      cidr_blocks = [var.allowed_ssh_cidr]
    }
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}
```

**Dynamic Blocks**: Conditionally add rules based on variables

### Step 6.2: Get Latest AMI

```hcl
data "aws_ami" "amazon_linux_2" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
}
```

**Data Sources**: Query existing AWS resources

### Step 6.3: Create EC2 Instance

```hcl
resource "aws_instance" "ml_training" {
  ami           = data.aws_ami.amazon_linux_2.id
  instance_type = var.instance_type

  subnet_id                   = aws_subnet.public[0].id
  vpc_security_group_ids      = [aws_security_group.ml_instance.id]
  associate_public_ip_address = true

  iam_instance_profile = aws_iam_instance_profile.ml_instance.name

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  user_data = file("${path.module}/user_data.sh")

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  monitoring = var.enable_cloudwatch_alarms

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-ml-training"
    }
  )
}
```

**Key Configurations**:
- `iam_instance_profile`: Grants S3/CloudWatch access
- `encrypted`: Encrypts root volume
- `http_tokens = "required"`: IMDSv2 (security best practice)
- `user_data`: Bootstrap script

### Step 6.4: Create user_data.sh

```bash
#!/bin/bash
# terraform/user_data.sh
set -e

# Update system
yum update -y

# Install Python and ML tools
yum install -y python3 python3-pip git

# Install ML libraries
pip3 install \
    numpy \
    pandas \
    scikit-learn \
    jupyter \
    torch \
    tensorflow \
    boto3

# Configure Jupyter (if enabled)
if [ "${enable_jupyter}" = "true" ]; then
    mkdir -p /home/ec2-user/.jupyter
    cat > /home/ec2-user/.jupyter/jupyter_notebook_config.py << 'EOF'
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.token = ''
EOF
    chown -R ec2-user:ec2-user /home/ec2-user/.jupyter

    # Create systemd service
    cat > /etc/systemd/system/jupyter.service << 'EOF'
[Unit]
Description=Jupyter Notebook

[Service]
Type=simple
User=ec2-user
ExecStart=/usr/local/bin/jupyter notebook
Restart=always

[Install]
WantedBy=multi-user.target
EOF

    systemctl enable jupyter
    systemctl start jupyter
fi

echo "Setup complete!"
```

**User Data**:
- Runs on first boot
- Installs dependencies
- Configures services
- Templated with Terraform variables

### Step 6.5: Apply EC2 Configuration

```bash
terraform plan
terraform apply
```

This may take 5-10 minutes as AWS creates the instance.

### Step 6.6: Test SSH Access

```bash
# Get public IP
terraform output ml_instance_public_ip

# SSH (requires key pair - add separately)
ssh ec2-user@$(terraform output -raw ml_instance_public_ip)
```

### Step 6.7: Test S3 Access

From the instance:

```bash
# List buckets
aws s3 ls

# Test write
echo "test" > test.txt
aws s3 cp test.txt s3://DATASETS_BUCKET_NAME/test.txt

# Test read
aws s3 cp s3://DATASETS_BUCKET_NAME/test.txt downloaded.txt
```

---

## Phase 7: Monitoring

### Step 7.1: Add CloudWatch Alarms

In `ec2.tf`, add:

```hcl
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${local.name_prefix}-ml-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = var.cpu_alarm_threshold

  dimensions = {
    InstanceId = aws_instance.ml_training.id
  }

  tags = local.common_tags
}
```

### Step 7.2: Add VPC Flow Logs

In `vpc.tf`, add:

```hcl
resource "aws_cloudwatch_log_group" "vpc_flow_log" {
  name              = "/aws/vpc/${local.name_prefix}"
  retention_in_days = 7

  tags = local.common_tags
}

resource "aws_flow_log" "main" {
  iam_role_arn    = aws_iam_role.vpc_flow_log.arn
  log_destination = aws_cloudwatch_log_group.vpc_flow_log.arn
  traffic_type    = "ALL"
  vpc_id          = aws_vpc.main.id

  tags = local.common_tags
}
```

### Step 7.3: Apply Monitoring

```bash
terraform apply
```

### Step 7.4: View Metrics

```bash
# CloudWatch console
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=$(terraform output -raw ml_instance_id) \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average
```

---

## Phase 8: Testing

### Step 8.1: Manual Testing

```bash
# Validate configuration
terraform validate

# Check formatting
terraform fmt -check -recursive

# Security scan (optional)
tfsec .

# Cost estimation (optional)
infracost breakdown --path .
```

### Step 8.2: Automated Tests

See `tests/terraform_test.go` for Terratest examples.

```bash
cd tests/
go test -v -timeout 30m
```

---

## Phase 9: Production Readiness

### Step 9.1: Remote State Backend

Create S3 bucket for state:

```bash
aws s3api create-bucket \
  --bucket my-terraform-state \
  --region us-east-1

aws s3api put-bucket-versioning \
  --bucket my-terraform-state \
  --versioning-configuration Status=Enabled
```

Create DynamoDB table for locking:

```bash
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
```

Update `providers.tf`:

```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "ml-infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}
```

Migrate state:

```bash
terraform init -migrate-state
```

### Step 9.2: Security Hardening

1. **Restrict SSH**:
   ```hcl
   allowed_ssh_cidr = "YOUR_IP/32"
   ```

2. **Enable CloudTrail**: Track all API calls

3. **Enable GuardDuty**: Threat detection

4. **Enable Config**: Compliance monitoring

5. **Rotate Keys**: Implement key rotation

### Step 9.3: Production Checklist

- [ ] Remote state configured
- [ ] State encryption enabled
- [ ] SSH access restricted
- [ ] S3 encryption enabled
- [ ] S3 versioning enabled
- [ ] IAM least privilege
- [ ] CloudWatch alarms configured
- [ ] Backup strategy defined
- [ ] Disaster recovery plan
- [ ] Documentation complete
- [ ] Team trained on Terraform
- [ ] CI/CD pipeline configured

---

## Conclusion

You've now built a complete, production-ready ML infrastructure using Terraform! The key takeaways:

1. **Infrastructure as Code**: Version controlled, reproducible
2. **Modularity**: Organized, reusable components
3. **Security**: Defense in depth, least privilege
4. **Monitoring**: Comprehensive logging and alerting
5. **Cost Optimization**: Right-sized resources
6. **Best Practices**: Industry-standard patterns

## Next Steps

1. Customize for your specific needs
2. Add more ML-specific features (GPU instances, etc.)
3. Integrate with CI/CD pipelines
4. Explore Terraform modules for reusability
5. Learn advanced Terraform features

Happy Infrastructure as Code!
