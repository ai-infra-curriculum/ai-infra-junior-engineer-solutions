# Exercise 07: Terraform Basics - Complete Solution

A production-ready Terraform configuration for ML infrastructure on AWS, featuring VPC networking, EC2 instances, S3 storage, IAM roles, and comprehensive monitoring.

## Overview

This solution demonstrates Infrastructure as Code (IaC) best practices for deploying ML workloads on AWS using Terraform. It creates a secure, scalable, and cost-optimized infrastructure suitable for machine learning training and experimentation.

## What's Included

- **Complete Terraform Configuration**: Production-ready IaC for AWS
- **VPC with Public/Private Subnets**: Secure network architecture
- **EC2 Instances**: Configured for ML workloads with Jupyter
- **S3 Buckets**: Organized storage for datasets and models
- **IAM Roles**: Least privilege access control
- **Monitoring**: CloudWatch logs and alarms
- **Helper Scripts**: Automation for common tasks
- **Tests**: Terratest-based infrastructure tests
- **Documentation**: Comprehensive guides and architecture docs

## Quick Start

### Prerequisites

- AWS account with CLI configured
- Terraform 1.0+ installed
- Basic understanding of AWS and Terraform

### 1. Setup

```bash
# Install Terraform (if not already installed)
./scripts/setup.sh

# Configure AWS credentials
aws configure
```

### 2. Deploy Infrastructure

```bash
# Initialize Terraform
cd terraform/
terraform init

# Review the plan
terraform plan

# Deploy infrastructure
terraform apply
```

### 3. Access Your Infrastructure

```bash
# Get outputs
terraform output

# SSH to instance
ssh ec2-user@$(terraform output -raw ml_instance_public_ip)

# Access Jupyter Notebook
open $(terraform output -raw jupyter_url)
```

### 4. Clean Up

```bash
# Destroy all resources
terraform destroy
```

## Project Structure

```
exercise-07-terraform-basics/
├── README.md                    # This file
├── STEP_BY_STEP.md             # Detailed implementation guide
├── .gitignore                  # Git ignore patterns
│
├── terraform/                  # Main Terraform configuration
│   ├── providers.tf           # Provider and backend configuration
│   ├── variables.tf           # Input variable definitions
│   ├── outputs.tf             # Output value definitions
│   ├── main.tf                # Main orchestration
│   ├── vpc.tf                 # VPC and networking resources
│   ├── ec2.tf                 # EC2 instance configuration
│   ├── s3.tf                  # S3 bucket configuration
│   ├── iam.tf                 # IAM roles and policies
│   └── user_data.sh           # EC2 initialization script
│
├── examples/                   # Example configurations
│   ├── dev/                   # Development environment
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── terraform.tfvars.example
│   │   └── README.md
│   └── prod/                  # Production environment
│       ├── main.tf
│       ├── variables.tf
│       ├── terraform.tfvars.example
│       └── README.md
│
├── scripts/                    # Helper scripts
│   ├── setup.sh               # Install and configure Terraform
│   ├── init.sh                # Initialize Terraform
│   ├── plan.sh                # Show execution plan
│   ├── apply.sh               # Apply configuration
│   ├── destroy.sh             # Destroy infrastructure
│   └── validate.sh            # Validate configuration
│
├── tests/                      # Infrastructure tests
│   ├── terraform_test.go      # Terratest test suite
│   └── README.md              # Test documentation
│
└── docs/                       # Documentation
    └── ARCHITECTURE.md         # Architecture documentation
```

## Key Features

### 1. Secure Networking

- **VPC**: Isolated network environment (10.0.0.0/16)
- **Public Subnets**: For instances with internet access
- **Private Subnets**: For backend services
- **NAT Gateway**: Secure outbound internet for private subnets
- **VPC Flow Logs**: Network traffic monitoring

### 2. ML-Ready Compute

- **Amazon Linux 2**: Latest stable AMI
- **Pre-installed Libraries**: PyTorch, TensorFlow, Scikit-learn
- **Jupyter Notebook**: Web-based development environment
- **Encrypted Storage**: 50GB gp3 volume with encryption
- **IAM Instance Profile**: Secure access to S3 and CloudWatch

### 3. Organized Storage

- **Datasets Bucket**: Structured storage for training data
  - `/raw/` - Raw unprocessed data
  - `/processed/` - Cleaned data
  - `/interim/` - Intermediate steps
  - `/external/` - External sources
- **Models Bucket**: Versioned storage for trained models
  - `/trained/` - Final models
  - `/artifacts/` - Model artifacts
  - `/experiments/` - Experimental models
  - `/production/` - Production models

### 4. Security Best Practices

- **Encryption**: All data encrypted at rest and in transit
- **IAM Roles**: No hardcoded credentials
- **Least Privilege**: Minimal required permissions
- **Public Access Blocking**: S3 buckets locked down
- **Security Groups**: Restricted network access
- **VPC Isolation**: Resources in private network

### 5. Cost Optimization

- **Right-sized Instances**: t3.medium for development
- **gp3 Volumes**: Cost-effective storage
- **S3 Lifecycle Policies**: Automatic data archival
- **Resource Tagging**: Cost allocation tracking
- **Estimated Cost**: ~$67/month (with optimizations)

### 6. Monitoring and Alerting

- **CloudWatch Alarms**: CPU and status check monitoring
- **VPC Flow Logs**: Network traffic analysis
- **EC2 Logs**: Application logging
- **Custom Metrics**: Extensible monitoring

## Usage Examples

### Example 1: Development Environment

Deploy a cost-optimized development environment:

```bash
cd examples/dev/
terraform init
terraform apply
```

Features:
- t3.small instance (~$15/month)
- 30GB storage
- Reduced monitoring
- Jupyter enabled

### Example 2: Production Environment

Deploy a production-ready environment:

```bash
cd examples/prod/
terraform init
terraform apply
```

Features:
- t3.large instance (~$60/month)
- 100GB storage
- Full monitoring
- Multi-AZ deployment

### Example 3: Custom Configuration

Create your own configuration:

```hcl
module "ml_infrastructure" {
  source = "./terraform"

  project_name     = "my-ml-project"
  environment      = "staging"
  instance_type    = "c5.xlarge"
  root_volume_size = 200
  enable_jupyter   = true
  # ... more variables
}
```

## Configuration Variables

### Essential Variables

```hcl
variable "aws_region" {
  default = "us-east-1"
}

variable "project_name" {
  default = "ml-infrastructure"
}

variable "environment" {
  default = "dev"
}

variable "instance_type" {
  default = "t3.medium"
}
```

See `terraform/variables.tf` for all available variables.

## Outputs

After applying, Terraform provides useful outputs:

```bash
# View all outputs
terraform output

# Specific outputs
terraform output ml_instance_public_ip
terraform output jupyter_url
terraform output datasets_bucket_name
terraform output ssh_connection_command
```

## Common Tasks

### Start/Stop Instance

```bash
# Stop instance (save costs)
aws ec2 stop-instances --instance-ids $(terraform output -raw ml_instance_id)

# Start instance
aws ec2 start-instances --instance-ids $(terraform output -raw ml_instance_id)
```

### Upload Data to S3

```bash
# Upload dataset
aws s3 cp local-data.csv s3://$(terraform output -raw datasets_bucket_name)/raw/

# Sync directory
aws s3 sync ./data/ s3://$(terraform output -raw datasets_bucket_name)/raw/
```

### View Logs

```bash
# CloudWatch logs
aws logs tail /aws/ec2/ml-infrastructure-dev-ml-training --follow

# SSH to instance and check logs
ssh ec2-user@$(terraform output -raw ml_instance_public_ip)
tail -f /var/log/user-data.log
```

### Modify Infrastructure

```bash
# Change instance type
vim terraform.tfvars
# instance_type = "t3.large"

# Apply changes
terraform apply
```

## Security Notes

### Default Configuration

The default configuration uses `0.0.0.0/0` for SSH access (open to internet). This is for demonstration purposes only.

### Production Recommendations

1. **Restrict SSH Access**:
   ```hcl
   allowed_ssh_cidr = "YOUR_IP/32"
   ```

2. **Use VPN or Bastion**:
   ```hcl
   allowed_ssh_cidr = "10.0.0.0/8"  # Corporate network
   ```

3. **Enable MFA**: For AWS console access

4. **Rotate Credentials**: Regularly rotate SSH keys

5. **Use Session Manager**: Instead of SSH
   ```bash
   aws ssm start-session --target $(terraform output -raw ml_instance_id)
   ```

## Testing

Run infrastructure tests with Terratest:

```bash
cd tests/
go test -v -timeout 30m
```

See `tests/README.md` for detailed testing documentation.

## Troubleshooting

### Terraform Init Fails

```bash
# Clear cache and retry
rm -rf .terraform .terraform.lock.hcl
terraform init
```

### Apply Fails

```bash
# Check AWS credentials
aws sts get-caller-identity

# Validate configuration
terraform validate

# Check detailed logs
TF_LOG=DEBUG terraform apply
```

### Can't Connect to Instance

```bash
# Check security group
terraform output ml_instance_security_group_id

# Verify instance is running
aws ec2 describe-instances --instance-ids $(terraform output -raw ml_instance_id)

# Check system logs
aws ec2 get-console-output --instance-id $(terraform output -raw ml_instance_id)
```

### S3 Access Denied

```bash
# Verify IAM role
terraform output ml_instance_role_arn

# Test from instance
ssh ec2-user@$(terraform output -raw ml_instance_public_ip)
python3 /home/ec2-user/test_s3_access.py
```

## Cost Estimation

### Base Infrastructure

| Resource | Cost/Month |
|----------|-----------|
| EC2 t3.medium | $30 |
| EBS 50GB gp3 | $5 |
| NAT Gateway | $32 |
| S3 Storage | $0.023/GB |
| Data Transfer | $0.09/GB |
| **Total** | **~$67 + storage/transfer** |

### Cost Saving Tips

- Stop instances when not in use: Save $30/month
- Use spot instances: Save 50-70%
- Use S3 lifecycle policies: Save 80% on old data
- Right-size instances: Start with t3.small
- Delete unused resources: `terraform destroy`

## Best Practices

1. **Version Control**: Always use Git for Terraform code
2. **State Management**: Use remote state for teams
3. **Module Structure**: Organize code into reusable modules
4. **Variable Validation**: Validate inputs to prevent errors
5. **Resource Tagging**: Tag everything for cost tracking
6. **Documentation**: Keep docs updated with changes
7. **Testing**: Test infrastructure changes before production
8. **Security**: Follow least privilege principle

## Learning Resources

- [Terraform Documentation](https://www.terraform.io/docs)
- [AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [HashiCorp Learn](https://learn.hashicorp.com/terraform)

## Contributing

This is a learning solution. Feel free to:
- Add enhancements
- Improve documentation
- Report issues
- Share feedback

## Next Steps

1. **Complete the Learning Exercise**: Work through the original exercise
2. **Explore the Code**: Understand each Terraform resource
3. **Read STEP_BY_STEP.md**: Detailed implementation guide
4. **Review ARCHITECTURE.md**: Understand the architecture
5. **Run Tests**: Validate your understanding
6. **Customize**: Adapt to your specific needs
7. **Deploy to Production**: Follow production checklist

## License

This solution is provided as-is for educational purposes.

## Support

For questions or issues:
1. Review the documentation
2. Check troubleshooting section
3. Consult Terraform/AWS documentation
4. Ask in team channels or forums

---

**Happy Learning!** Master Infrastructure as Code with Terraform and build reliable, scalable ML infrastructure.
