# Quick Start Guide

Get your ML infrastructure running in 5 minutes!

## Prerequisites

- AWS account with credentials configured
- Terraform 1.0+ installed
- AWS CLI installed

## Steps

### 1. Install Terraform (if needed)

```bash
./scripts/setup.sh
```

### 2. Configure AWS

```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region
```

### 3. Deploy Infrastructure

```bash
cd terraform/
terraform init
terraform plan
terraform apply
```

Type `yes` when prompted.

### 4. Get Access Information

```bash
# View all outputs
terraform output

# Get specific values
terraform output ml_instance_public_ip
terraform output jupyter_url
terraform output datasets_bucket_name
```

### 5. Connect to Your Instance

```bash
# SSH (requires key pair setup)
ssh ec2-user@$(terraform output -raw ml_instance_public_ip)

# Or access Jupyter in browser
open $(terraform output -raw jupyter_url)
```

### 6. Test S3 Access

From the EC2 instance:

```bash
# Test S3 access
python3 /home/ec2-user/test_s3_access.py

# Upload a file
aws s3 cp myfile.csv s3://$(terraform output -raw datasets_bucket_name)/raw/
```

### 7. Clean Up (Important!)

```bash
# Destroy all resources to avoid charges
terraform destroy
```

Type `yes` when prompted.

## Common Commands

```bash
# View infrastructure state
terraform show

# Format code
terraform fmt

# Validate configuration
terraform validate

# View specific output
terraform output <output_name>

# Refresh state
terraform refresh

# List resources
terraform state list
```

## Troubleshooting

### Can't connect to instance?

1. Check security group allows your IP:
   ```bash
   aws ec2 describe-security-groups \
     --group-ids $(terraform output -raw ml_instance_security_group_id)
   ```

2. Verify instance is running:
   ```bash
   aws ec2 describe-instances \
     --instance-ids $(terraform output -raw ml_instance_id)
   ```

### S3 access denied?

1. Verify IAM role is attached:
   ```bash
   aws ec2 describe-instances \
     --instance-ids $(terraform output -raw ml_instance_id) \
     --query 'Reservations[0].Instances[0].IamInstanceProfile'
   ```

2. Test from instance:
   ```bash
   ssh ec2-user@$(terraform output -raw ml_instance_public_ip)
   aws s3 ls
   ```

### Terraform errors?

1. Check AWS credentials:
   ```bash
   aws sts get-caller-identity
   ```

2. Validate configuration:
   ```bash
   terraform validate
   ```

3. Enable debug logging:
   ```bash
   TF_LOG=DEBUG terraform apply
   ```

## Cost Warning

Running this infrastructure 24/7 costs approximately $67/month:
- EC2 t3.medium: ~$30/month
- EBS 50GB: ~$5/month
- NAT Gateway: ~$32/month
- S3 storage: Variable

**Remember to `terraform destroy` when not in use!**

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Follow [STEP_BY_STEP.md](STEP_BY_STEP.md) to understand implementation
- Review [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for architecture details
- Try [examples/dev/](examples/dev/) for a minimal setup
- Customize variables in `terraform/variables.tf`

## Support

For issues:
1. Check this guide
2. Review [README.md](README.md) troubleshooting section
3. Consult Terraform/AWS documentation
4. Ask in team channels

Happy building!
