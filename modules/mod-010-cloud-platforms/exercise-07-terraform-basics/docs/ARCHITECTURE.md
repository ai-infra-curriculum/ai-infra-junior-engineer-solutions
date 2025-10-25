# ML Infrastructure Architecture

This document describes the architecture of the ML infrastructure deployed by Terraform.

## Overview

This Terraform configuration creates a production-ready ML infrastructure on AWS with:
- Secure networking (VPC with public/private subnets)
- Compute resources (EC2 instances for ML workloads)
- Storage (S3 buckets for datasets and models)
- Access control (IAM roles with least privilege)
- Monitoring (CloudWatch logs and alarms)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           AWS Account                                │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                        VPC (10.0.0.0/16)                     │   │
│  │                                                               │   │
│  │  ┌──────────────────┐              ┌──────────────────┐     │   │
│  │  │  Public Subnet   │              │  Public Subnet   │     │   │
│  │  │   10.0.1.0/24    │              │   10.0.2.0/24    │     │   │
│  │  │                  │              │                  │     │   │
│  │  │  ┌────────────┐  │              │                  │     │   │
│  │  │  │ EC2 ML     │  │              │  ┌────────────┐  │     │   │
│  │  │  │ Instance   │  │              │  │ NAT Gateway│  │     │   │
│  │  │  │            │  │              │  └────────────┘  │     │   │
│  │  │  │ - Jupyter  │  │              │                  │     │   │
│  │  │  │ - PyTorch  │  │              │                  │     │   │
│  │  │  │ - TF       │  │              │                  │     │   │
│  │  │  └────────────┘  │              │                  │     │   │
│  │  │        │         │              │         │        │     │   │
│  │  └────────│─────────┘              └─────────│────────┘     │   │
│  │           │                                   │              │   │
│  │           │  ┌───────────────────────────────┘              │   │
│  │           │  │                                               │   │
│  │  ┌────────┼──┼─────────┐              ┌──────────────────┐  │   │
│  │  │ Private│Subnet      │              │  Private Subnet  │  │   │
│  │  │   10.0.11.0/24     │              │   10.0.12.0/24   │  │   │
│  │  │                    │              │                  │  │   │
│  │  │  (Future: DBs,     │              │  (Future: DBs,   │  │   │
│  │  │   Batch Jobs)      │              │   Batch Jobs)    │  │   │
│  │  └────────────────────┘              └──────────────────┘  │   │
│  │                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      S3 Buckets                               │  │
│  │                                                               │  │
│  │  ┌──────────────────┐         ┌──────────────────┐          │  │
│  │  │  ML Datasets     │         │   ML Models      │          │  │
│  │  │                  │         │                  │          │  │
│  │  │  - Encrypted     │         │  - Encrypted     │          │  │
│  │  │  - Versioned     │         │  - Versioned     │          │  │
│  │  │  - Lifecycle     │         │  - Lifecycle     │          │  │
│  │  └──────────────────┘         └──────────────────┘          │  │
│  │           ▲                             ▲                    │  │
│  └───────────┼─────────────────────────────┼────────────────────┘  │
│              │                             │                       │
│              └─────────────┬───────────────┘                       │
│                            │                                       │
│  ┌─────────────────────────┼────────────────────────────────────┐ │
│  │                    IAM Role                                   │ │
│  │                                                               │ │
│  │  - S3 Read/Write Access                                      │ │
│  │  - CloudWatch Logs/Metrics                                   │ │
│  │  - EC2 Describe                                              │ │
│  │  - SSM Session Manager                                       │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    CloudWatch                                 │ │
│  │                                                               │ │
│  │  - VPC Flow Logs                                             │ │
│  │  - EC2 Logs                                                  │ │
│  │  - CPU Alarms                                                │ │
│  │  - Status Check Alarms                                       │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. VPC (Virtual Private Cloud)

**Purpose**: Isolated network environment for ML infrastructure

**Configuration**:
- CIDR: 10.0.0.0/16
- 2 Availability Zones (for high availability)
- DNS hostnames enabled
- DNS support enabled

**Resources**:
- 1 VPC
- 1 Internet Gateway
- 1 NAT Gateway
- 1 Elastic IP (for NAT)
- 2 Public Subnets
- 2 Private Subnets
- 2 Route Tables
- VPC Flow Logs

### 2. Public Subnets

**Purpose**: Host resources that need internet access (EC2, NAT Gateway)

**Configuration**:
- Subnet 1: 10.0.1.0/24 (AZ-1)
- Subnet 2: 10.0.2.0/24 (AZ-2)
- Auto-assign public IP: Enabled
- Route to Internet Gateway

**Use Cases**:
- ML training instances
- Jupyter notebooks
- Bastion hosts (future)

### 3. Private Subnets

**Purpose**: Host resources that don't need direct internet access

**Configuration**:
- Subnet 1: 10.0.11.0/24 (AZ-1)
- Subnet 2: 10.0.12.0/24 (AZ-2)
- Auto-assign public IP: Disabled
- Route to NAT Gateway

**Use Cases**:
- Databases (future)
- Batch processing (future)
- Internal services (future)

### 4. EC2 ML Instance

**Purpose**: Compute for ML training and experimentation

**Specifications**:
- AMI: Amazon Linux 2 (latest)
- Instance Type: t3.medium (configurable)
- Storage: 50GB gp3 (encrypted)
- Security Group: Restricted access

**Software Installed**:
- Python 3
- AWS CLI v2
- Machine Learning Libraries:
  - NumPy, Pandas, Scikit-learn
  - PyTorch, TensorFlow
  - Jupyter, JupyterLab
- Development Tools:
  - Git, Vim, Tmux

**Access Methods**:
- SSH (port 22)
- Jupyter Notebook (port 8888)
- TensorBoard (port 6006)

### 5. S3 Buckets

#### Datasets Bucket

**Purpose**: Store ML training datasets

**Configuration**:
- Encryption: AES256
- Versioning: Enabled
- Public Access: Blocked
- Lifecycle: Transition to Glacier after 90 days

**Folder Structure**:
```
/raw/         - Raw, unprocessed data
/processed/   - Cleaned, processed data
/interim/     - Intermediate processing steps
/external/    - External data sources
```

#### Models Bucket

**Purpose**: Store trained ML models and artifacts

**Configuration**:
- Encryption: AES256
- Versioning: Enabled
- Public Access: Blocked
- Lifecycle: Archive old versions

**Folder Structure**:
```
/trained/     - Final trained models
/artifacts/   - Model artifacts (weights, configs)
/experiments/ - Experimental models
/production/  - Production-ready models
```

### 6. IAM Configuration

#### EC2 Instance Role

**Purpose**: Grant EC2 instance access to AWS services

**Permissions**:
- **S3 Access**: Read/Write to ML buckets
- **CloudWatch Logs**: Create log streams and write logs
- **CloudWatch Metrics**: Publish custom metrics
- **EC2 Describe**: Query own metadata
- **SSM**: Session Manager access (secure SSH alternative)

**Policies**:
- Custom S3 policy (least privilege)
- CloudWatch logs policy
- CloudWatch metrics policy
- EC2 describe policy
- AWS managed: AmazonSSMManagedInstanceCore

### 7. Security Groups

#### ML Instance Security Group

**Inbound Rules**:
- SSH (22): Restricted CIDR (configurable)
- Jupyter (8888): Restricted CIDR (configurable)
- TensorBoard (6006): Restricted CIDR (configurable)

**Outbound Rules**:
- All traffic: 0.0.0.0/0 (for package downloads, AWS API)

**Security Notes**:
- Default: 0.0.0.0/0 (demo only!)
- Production: Use VPN or bastion CIDR
- Consider: AWS Session Manager instead of SSH

### 8. Monitoring

#### CloudWatch Logs

**Log Groups**:
- `/aws/vpc/{project}-{env}`: VPC flow logs
- `/aws/ec2/{project}-{env}-ml-training`: EC2 instance logs

**Retention**: 7 days (configurable)

#### CloudWatch Alarms

**CPU Utilization Alarm**:
- Metric: CPUUtilization
- Threshold: 80% (configurable)
- Period: 5 minutes
- Evaluation: 2 consecutive periods

**Status Check Alarm**:
- Metric: StatusCheckFailed
- Threshold: > 0
- Period: 5 minutes
- Action: Alert (expandable to auto-recovery)

## Data Flow

### Training Workflow

1. **Data Upload**:
   ```
   Local → S3 Datasets Bucket (/raw/)
   ```

2. **Data Processing**:
   ```
   EC2 reads from S3 (/raw/)
   → Process data
   → Write to S3 (/processed/)
   ```

3. **Model Training**:
   ```
   EC2 reads from S3 (/processed/)
   → Train model
   → Save checkpoints to S3 (/experiments/)
   ```

4. **Model Deployment**:
   ```
   Validate model
   → Copy to S3 (/production/)
   → Deploy to inference service
   ```

### Access Flow

1. **User Access**:
   ```
   Developer
   → VPN/Bastion
   → SSH to EC2
   → Jupyter Notebook
   ```

2. **EC2 to S3**:
   ```
   EC2 Instance
   → IAM Instance Profile
   → S3 API (HTTPS)
   → S3 Bucket
   ```

3. **Logging**:
   ```
   EC2/VPC
   → CloudWatch Logs
   → Log Group
   → (Optional) S3 archival
   ```

## Security Considerations

### Network Security

1. **VPC Isolation**: Resources in dedicated VPC
2. **Subnet Segmentation**: Public/private subnet separation
3. **Security Groups**: Least privilege access
4. **NACLs**: Additional network layer (default allow)
5. **VPC Flow Logs**: Network traffic monitoring

### Data Security

1. **Encryption at Rest**: S3 server-side encryption (AES256)
2. **Encryption in Transit**: HTTPS/TLS for all data transfer
3. **EBS Encryption**: All volumes encrypted
4. **Versioning**: S3 versioning for data recovery
5. **Backup**: Lifecycle policies for data archival

### Access Security

1. **IAM Roles**: No hardcoded credentials
2. **Least Privilege**: Minimal required permissions
3. **MFA**: Recommended for console access
4. **Session Manager**: Alternative to SSH keys
5. **Audit Logs**: CloudTrail for API calls

### Application Security

1. **OS Updates**: Automated via user data
2. **Security Patches**: Regular updates recommended
3. **Jupyter Security**: Token-based authentication (configurable)
4. **SSH Keys**: Rotate regularly
5. **Network Restrictions**: Limit source IPs

## Cost Optimization

### Cost Breakdown (Monthly)

| Resource | Configuration | Estimated Cost |
|----------|--------------|----------------|
| EC2 (t3.medium) | 24/7 running | $30 |
| EBS (50GB gp3) | Root volume | $5 |
| NAT Gateway | Data transfer | $32 + data |
| S3 Storage | Per GB | $0.023/GB |
| Data Transfer | Out to internet | $0.09/GB |
| **Total Base** | | **~$67/month** |

### Cost Saving Tips

1. **Stop instances when not in use**:
   ```bash
   aws ec2 stop-instances --instance-ids <id>
   ```
   Saves: ~$30/month

2. **Use spot instances** (for non-critical workloads):
   ```hcl
   spot_price = "0.05"
   ```
   Saves: ~50-70%

3. **S3 Lifecycle policies**:
   - Transition to Glacier: Save 80%
   - Delete old data: Save 100%

4. **Right-size instances**:
   - Start small (t3.small)
   - Scale up only when needed

5. **Delete unused resources**:
   ```bash
   terraform destroy
   ```

6. **Use NAT instances** instead of NAT Gateway:
   Saves: ~$32/month (but less reliable)

## Scalability

### Horizontal Scaling

Add more EC2 instances:
```hcl
count = var.instance_count
```

### Vertical Scaling

Increase instance size:
```hcl
instance_type = "t3.large"  # or c5.2xlarge for compute
```

### Auto Scaling

Add Auto Scaling Group (future):
```hcl
resource "aws_autoscaling_group" "ml_workers" {
  min_size = 1
  max_size = 10
  # ...
}
```

### Load Balancing

Add Application Load Balancer (future):
```hcl
resource "aws_lb" "ml_alb" {
  # ...
}
```

## High Availability

### Current HA Features

1. **Multi-AZ VPC**: Subnets in 2 AZs
2. **S3 Redundancy**: 99.999999999% durability
3. **CloudWatch Monitoring**: Detect failures

### Future HA Improvements

1. **Auto Scaling Group**: Replace failed instances
2. **EFS**: Shared filesystem across instances
3. **RDS Multi-AZ**: Database high availability
4. **Application Load Balancer**: Distribute traffic

## Disaster Recovery

### Backup Strategy

1. **S3 Versioning**: Recover deleted/modified files
2. **S3 Cross-Region Replication**: Geographic redundancy
3. **EBS Snapshots**: Backup volumes (manual/automated)
4. **Terraform State**: Version controlled in Git

### Recovery Procedures

1. **Instance Failure**:
   ```bash
   terraform apply  # Recreate instance
   ```

2. **Data Loss**:
   ```bash
   aws s3 cp s3://bucket/file s3://bucket/file?versionId=<version>
   ```

3. **Region Failure**:
   - Enable cross-region replication
   - Deploy to new region with Terraform

## Monitoring and Alerting

### Key Metrics

1. **CPU Utilization**: Training progress/resource usage
2. **Disk I/O**: Data loading bottlenecks
3. **Network Traffic**: Data transfer costs
4. **Memory Usage**: Out-of-memory risks
5. **GPU Utilization**: For GPU instances

### Alert Thresholds

- CPU > 80%: Check if training is efficient
- Disk > 85%: Clean up or expand volume
- Status Check Fail: Instance issue
- Network Anomaly: Security concern

### Dashboard

Create CloudWatch Dashboard:
```hcl
resource "aws_cloudwatch_dashboard" "ml_dashboard" {
  # Visualize all metrics
}
```

## Best Practices

1. **Infrastructure as Code**: Version all changes
2. **Least Privilege**: Minimal IAM permissions
3. **Defense in Depth**: Multiple security layers
4. **Monitoring**: Comprehensive logging and alerting
5. **Cost Management**: Regular cost reviews
6. **Documentation**: Keep architecture docs updated
7. **Testing**: Test infrastructure changes
8. **Automation**: Automate repetitive tasks

## Future Enhancements

1. **GPU Instances**: Add p3/g4 instances for deep learning
2. **Kubernetes**: EKS for container orchestration
3. **SageMaker**: Managed ML service integration
4. **CI/CD**: Automated training pipelines
5. **MLflow**: Experiment tracking
6. **Data Pipeline**: Airflow or Step Functions
7. **API Gateway**: Model serving endpoints
8. **Elastic Cache**: Redis for model caching

## References

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)
- [AWS ML Best Practices](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/)
