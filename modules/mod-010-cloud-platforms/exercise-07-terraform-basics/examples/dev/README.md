# Development Environment Example

This example demonstrates how to deploy the ML infrastructure in a development environment.

## Configuration

The development environment uses:
- **Instance Type**: t3.small (cost-optimized)
- **Storage**: 30GB root volume
- **S3 Versioning**: Disabled (to save costs)
- **Monitoring**: Reduced CloudWatch alarms
- **Lifecycle**: 30 days before transitioning to Glacier

## Quick Start

1. Copy the example variables file:
```bash
cp terraform.tfvars.example terraform.tfvars
```

2. Edit `terraform.tfvars` with your values:
```bash
vim terraform.tfvars
```

3. Initialize Terraform:
```bash
terraform init
```

4. Review the plan:
```bash
terraform plan
```

5. Apply the configuration:
```bash
terraform apply
```

6. Get the outputs:
```bash
terraform output
```

## Cleanup

To destroy all resources:
```bash
terraform destroy
```

## Estimated Costs

- EC2 t3.small: ~$15/month
- EBS 30GB: ~$3/month
- S3 storage: Variable
- Total: ~$20/month + S3 costs
