# Step-by-Step Implementation Guide: AWS Fundamentals for ML

## Overview

Master AWS core services for ML infrastructure! Learn EC2, S3, IAM, VPC, EKS, SageMaker basics, and production deployment patterns on AWS.

**Time**: 3-4 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Configure AWS CLI and credentials
✅ Create and manage EC2 instances for ML workloads
✅ Use S3 for model and data storage
✅ Implement IAM roles and policies
✅ Deploy ML applications on EKS
✅ Use SageMaker for model training
✅ Set up VPC for secure networking
✅ Monitor costs and optimize spending

---

## Phase 1: AWS Setup

### Install AWS CLI

```bash
# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify
aws --version
```

### Configure Credentials

```bash
# Configure AWS credentials
aws configure

# Inputs:
# AWS Access Key ID: YOUR_ACCESS_KEY
# AWS Secret Access Key: YOUR_SECRET_KEY
# Default region: us-west-2
# Default output format: json

# Verify
aws sts get-caller-identity
```

### Create IAM User

```bash
# Create ML admin user
aws iam create-user --user-name ml-admin

# Create access key
aws iam create-access-key --user-name ml-admin

# Attach policies
aws iam attach-user-policy \
  --user-name ml-admin \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess

aws iam attach-user-policy \
  --user-name ml-admin \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

---

## Phase 2: EC2 for ML Training

### Launch GPU Instance

```bash
# Find latest Deep Learning AMI
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning AMI*" \
  --query 'Images | sort_by(@, &CreationDate) | [-1]'

# Launch p3.2xlarge instance (1 V100 GPU)
aws ec2 run-instances \
  --image-id ami-0123456789abcdef0 \
  --instance-type p3.2xlarge \
  --key-name my-key-pair \
  --security-group-ids sg-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --iam-instance-profile Name=ML-Training-Role \
  --block-device-mappings '[
    {
      "DeviceName": "/dev/sda1",
      "Ebs": {
        "VolumeSize": 100,
        "VolumeType": "gp3",
        "DeleteOnTermination": true
      }
    }
  ]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ML-Training}]'

# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=ML-Training" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

# Wait for running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress'
```

### Connect and Setup

```bash
# SSH to instance
ssh -i my-key-pair.pem ubuntu@<PUBLIC_IP>

# Verify GPU
nvidia-smi

# Setup training environment
conda create -n ml python=3.11
conda activate ml
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Phase 3: S3 for Storage

### Create Buckets

```bash
# Create buckets for different purposes
aws s3 mb s3://my-ml-datasets --region us-west-2
aws s3 mb s3://my-ml-models --region us-west-2
aws s3 mb s3://my-ml-artifacts --region us-west-2

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket my-ml-models \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket my-ml-models \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'
```

### Upload/Download Data

```bash
# Upload dataset
aws s3 cp dataset.tar.gz s3://my-ml-datasets/imagenet/

# Sync directory
aws s3 sync ./models/ s3://my-ml-models/resnet50/v1.0/

# Download model
aws s3 cp s3://my-ml-models/resnet50/v1.0/model.pth ./

# List objects
aws s3 ls s3://my-ml-models/resnet50/ --recursive
```

### S3 Lifecycle Policy

```json
{
  "Rules": [{
    "Id": "DeleteOldCheckpoints",
    "Status": "Enabled",
    "Prefix": "checkpoints/",
    "Expiration": {
      "Days": 30
    }
  }, {
    "Id": "ArchiveOldModels",
    "Status": "Enabled",
    "Prefix": "models/archive/",
    "Transitions": [{
      "Days": 90,
      "StorageClass": "GLACIER"
    }]
  }]
}
```

```bash
# Apply lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-ml-models \
  --lifecycle-configuration file://lifecycle.json
```

---

## Phase 4: EKS for Production Deployment

### Create EKS Cluster

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster
eksctl create cluster \
  --name ml-production \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Add GPU node group
eksctl create nodegroup \
  --cluster ml-production \
  --name gpu-workers \
  --node-type p3.2xlarge \
  --nodes 0 \
  --nodes-min 0 \
  --nodes-max 5 \
  --node-labels workload=gpu

# Configure kubectl
aws eks update-kubeconfig --name ml-production --region us-west-2
```

### Deploy Application

```yaml
# ml-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-inference
  template:
    metadata:
      labels:
        app: ml-inference
    spec:
      serviceAccountName: ml-inference-sa
      containers:
      - name: api
        image: <AWS_ACCOUNT_ID>.dkr.ecr.us-west-2.amazonaws.com/ml-api:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_BUCKET
          value: my-ml-models
        - name: AWS_DEFAULT_REGION
          value: us-west-2
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

```bash
# Deploy
kubectl apply -f ml-deployment.yaml
```

---

## Phase 5: SageMaker Training

### Training Job

```python
import sagemaker
from sagemaker.pytorch import PyTorch

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::ACCOUNT_ID:role/SageMakerRole'

# Define estimator
estimator = PyTorch(
    entry_point='train.py',
    role=role,
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'epochs': 100,
        'batch-size': 32,
        'learning-rate': 0.001
    }
)

# Start training
estimator.fit({
    'training': 's3://my-ml-datasets/imagenet/train',
    'validation': 's3://my-ml-datasets/imagenet/val'
})
```

### Deploy Model

```python
# Deploy trained model
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name='resnet50-endpoint'
)

# Make prediction
result = predictor.predict(data)
```

---

## Phase 6: Cost Optimization

### Right-Sizing Instances

```bash
# Use AWS Compute Optimizer
aws compute-optimizer get-ec2-instance-recommendations \
  --instance-arns arn:aws:ec2:us-west-2:ACCOUNT_ID:instance/i-1234567890abcdef0
```

### Spot Instances

```bash
# Launch spot instance for training
aws ec2 run-instances \
  --instance-type p3.2xlarge \
  --instance-market-options '{
    "MarketType": "spot",
    "SpotOptions": {
      "MaxPrice": "1.50",
      "SpotInstanceType": "one-time"
    }
  }'
```

### S3 Intelligent-Tiering

```bash
# Enable intelligent tiering
aws s3api put-bucket-intelligent-tiering-configuration \
  --bucket my-ml-models \
  --id EntireDataset \
  --intelligent-tiering-configuration '{
    "Id": "EntireDataset",
    "Status": "Enabled",
    "Tierings": [{
      "Days": 90,
      "AccessTier": "ARCHIVE_ACCESS"
    }]
  }'
```

---

## Best Practices

✅ Use IAM roles instead of access keys
✅ Enable MFA for root account
✅ Use VPC for network isolation
✅ Enable CloudTrail for auditing
✅ Tag all resources for cost tracking
✅ Use spot instances for training
✅ Implement auto-scaling
✅ Monitor costs with Cost Explorer
✅ Use S3 lifecycle policies
✅ Enable encryption at rest

---

**AWS Fundamentals mastered!** ☁️
