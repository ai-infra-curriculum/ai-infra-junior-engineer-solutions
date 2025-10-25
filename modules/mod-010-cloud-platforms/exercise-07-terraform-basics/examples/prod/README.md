# Production Environment Example

This example demonstrates how to deploy the ML infrastructure in a production environment with enhanced security and monitoring.

## Production Differences

Compared to dev, production has:
- **Instance Type**: t3.large (more powerful)
- **Storage**: 100GB root volume
- **S3 Versioning**: Enabled (data protection)
- **Monitoring**: Full CloudWatch alarms enabled
- **Security**: Jupyter disabled, restricted SSH access
- **Lifecycle**: 90 days before transitioning to Glacier
- **High Availability**: 3 availability zones

## Prerequisites

Before deploying to production:

1. Set up remote state backend:
```bash
# Create S3 bucket for state
aws s3api create-bucket --bucket your-company-terraform-state --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning --bucket your-company-terraform-state \
  --versioning-configuration Status=Enabled

# Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
```

2. Uncomment the backend configuration in `main.tf`

3. Configure your VPN or bastion host CIDR in variables

## Deployment

1. Copy the example variables file:
```bash
cp terraform.tfvars.example terraform.tfvars
```

2. Edit `terraform.tfvars` with production values:
```bash
vim terraform.tfvars
```

3. Initialize Terraform with backend:
```bash
terraform init
```

4. Review the plan carefully:
```bash
terraform plan -out=tfplan
```

5. Get approval from team lead

6. Apply the configuration:
```bash
terraform apply tfplan
```

## Security Considerations

- SSH access is restricted to corporate network
- Jupyter is disabled for security
- All S3 buckets have encryption enabled
- Versioning is enabled for data recovery
- VPC flow logs capture all network traffic
- CloudWatch alarms monitor instance health

## Monitoring

Access CloudWatch to monitor:
- CPU utilization alerts
- Status check failures
- VPC flow logs
- S3 access patterns

## Backup and Recovery

- S3 versioning protects against accidental deletion
- Regular backups to Glacier after 90 days
- Terraform state is in remote backend with versioning

## Estimated Costs

- EC2 t3.large: ~$60/month
- EBS 100GB: ~$10/month
- NAT Gateway: ~$32/month
- S3 storage: Variable
- Data transfer: Variable
- Total: ~$100/month + S3 and transfer costs

## Cleanup

IMPORTANT: Get approval before destroying production resources!

```bash
terraform destroy
```
