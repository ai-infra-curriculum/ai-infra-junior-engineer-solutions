# Exercise 01: AWS Fundamentals for ML

Learn the core AWS services for machine learning infrastructure including EC2, S3, IAM, and introduction to SageMaker.

## Learning Objectives

- Launch and manage EC2 GPU instances for ML training
- Store and manage ML data in S3
- Configure IAM roles and policies for ML workflows
- Deploy a simple ML model using SageMaker
- Understand AWS billing and cost management
- Use AWS CLI and boto3 for automation

## Prerequisites

- AWS account (free tier available)
- AWS CLI installed and configured
- Python 3.8+ with boto3
- Basic understanding of cloud computing

## AWS Account Setup

### 1. Create AWS Account

1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click "Create an AWS Account"
3. Follow the signup process
4. Add payment method (required even for free tier)

### 2. Set Up AWS CLI

```bash
# Install AWS CLI
# macOS/Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify installation
aws --version

# Configure AWS CLI
aws configure
# AWS Access Key ID: [Your Access Key]
# AWS Secret Access Key: [Your Secret Key]
# Default region name: us-east-1
# Default output format: json
```

### 3. Create IAM User

```bash
# Create IAM user for ML operations
aws iam create-user --user-name ml-engineer

# Attach AdministratorAccess policy (for learning only)
aws iam attach-user-policy \
  --user-name ml-engineer \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess

# Create access key
aws iam create-access-key --user-name ml-engineer
```

**Best Practice**: In production, use specific policies, not AdministratorAccess.

## Part 1: EC2 for ML Training

### Launch GPU Instance

```bash
# List available AMIs (Deep Learning AMI)
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning AMI (Ubuntu*)*" \
  --query 'Images[*].[ImageId,Name,CreationDate]' \
  --output table

# Create security group
aws ec2 create-security-group \
  --group-name ml-sg \
  --description "Security group for ML instances"

# Allow SSH access
aws ec2 authorize-security-group-ingress \
  --group-name ml-sg \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0

# Create key pair
aws ec2 create-key-pair \
  --key-name ml-key \
  --query 'KeyMaterial' \
  --output text > ml-key.pem
chmod 400 ml-key.pem

# Launch GPU instance (p3.2xlarge with V100)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type p3.2xlarge \
  --key-name ml-key \
  --security-groups ml-sg \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ml-training}]' \
  --count 1

# Get instance public IP
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=ml-training" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text

# SSH into instance
ssh -i ml-key.pem ubuntu@<PUBLIC_IP>
```

### Using Spot Instances

Spot instances can save up to 90% of costs:

```bash
# Create spot instance request
aws ec2 request-spot-instances \
  --spot-price "1.00" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-specification.json
```

**spot-specification.json**:
```json
{
  "ImageId": "ami-0c55b159cbfafe1f0",
  "InstanceType": "p3.2xlarge",
  "KeyName": "ml-key",
  "SecurityGroups": ["ml-sg"],
  "UserData": "IyEvYmluL2Jhc2gKYXB0LWdldCB1cGRhdGUK",
  "IamInstanceProfile": {
    "Name": "ml-instance-profile"
  },
  "BlockDeviceMappings": [{
    "DeviceName": "/dev/sda1",
    "Ebs": {
      "VolumeSize": 200,
      "VolumeType": "gp3",
      "DeleteOnTermination": true
    }
  }]
}
```

### Training Script on EC2

Once connected to EC2:

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import boto3
import os

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}')

# Save model locally
torch.save(model.state_dict(), 'model.pth')
print("Model saved locally")

# Upload to S3
s3 = boto3.client('s3')
bucket_name = os.environ.get('S3_BUCKET', 'my-ml-bucket')

try:
    s3.upload_file('model.pth', bucket_name, 'models/mnist_model.pth')
    print(f"Model uploaded to s3://{bucket_name}/models/mnist_model.pth")
except Exception as e:
    print(f"Error uploading to S3: {e}")
```

Run training:
```bash
# Set environment variable
export S3_BUCKET=your-ml-bucket

# Install dependencies
pip install torch torchvision boto3

# Run training
python train.py
```

## Part 2: S3 for ML Data Management

### Create S3 Bucket

```bash
# Create bucket
aws s3 mb s3://my-ml-bucket-$(date +%s) --region us-east-1

# Set bucket name variable
export BUCKET_NAME=my-ml-bucket-1234567890

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket $BUCKET_NAME \
  --versioning-configuration Status=Enabled

# Add lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
  --bucket $BUCKET_NAME \
  --lifecycle-configuration file://lifecycle.json
```

**lifecycle.json**:
```json
{
  "Rules": [{
    "Id": "archive-old-models",
    "Status": "Enabled",
    "Prefix": "models/",
    "Transitions": [{
      "Days": 30,
      "StorageClass": "STANDARD_IA"
    }, {
      "Days": 90,
      "StorageClass": "GLACIER"
    }],
    "NoncurrentVersionExpiration": {
      "NoncurrentDays": 90
    }
  }, {
    "Id": "delete-temp-data",
    "Status": "Enabled",
    "Prefix": "temp/",
    "Expiration": {
      "Days": 7
    }
  }]
}
```

### S3 Operations with boto3

```python
# s3_operations.py
import boto3
import os
from pathlib import Path
import json

class S3Manager:
    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
        self.bucket_name = bucket_name
        self.bucket = self.s3_resource.Bucket(bucket_name)

    def upload_file(self, local_path, s3_key, metadata=None):
        """Upload file to S3 with optional metadata"""
        extra_args = {}
        if metadata:
            extra_args['Metadata'] = metadata

        try:
            self.s3.upload_file(local_path, self.bucket_name, s3_key, ExtraArgs=extra_args)
            print(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            print(f"Error uploading {local_path}: {e}")
            return False

    def download_file(self, s3_key, local_path):
        """Download file from S3"""
        try:
            self.s3.download_file(self.bucket_name, s3_key, local_path)
            print(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            print(f"Error downloading {s3_key}: {e}")
            return False

    def upload_directory(self, local_dir, s3_prefix):
        """Upload entire directory to S3"""
        local_path = Path(local_dir)
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{relative_path}"
                self.upload_file(str(file_path), s3_key)

    def list_objects(self, prefix='', max_keys=1000):
        """List objects in bucket"""
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )

            if 'Contents' in response:
                objects = response['Contents']
                for obj in objects:
                    size_mb = obj['Size'] / (1024 * 1024)
                    print(f"{obj['Key']}: {size_mb:.2f} MB (Modified: {obj['LastModified']})")
                return objects
            else:
                print(f"No objects found with prefix: {prefix}")
                return []
        except Exception as e:
            print(f"Error listing objects: {e}")
            return []

    def get_object_versions(self, key):
        """List all versions of an object"""
        try:
            response = self.s3.list_object_versions(
                Bucket=self.bucket_name,
                Prefix=key
            )

            if 'Versions' in response:
                versions = response['Versions']
                for v in versions:
                    print(f"Version {v['VersionId']}: {v['Size']} bytes, {v['LastModified']}")
                return versions
            else:
                print(f"No versions found for: {key}")
                return []
        except Exception as e:
            print(f"Error listing versions: {e}")
            return []

    def create_presigned_url(self, key, expiration=3600):
        """Generate presigned URL for temporary access"""
        try:
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expiration
            )
            print(f"Presigned URL (expires in {expiration}s):")
            print(url)
            return url
        except Exception as e:
            print(f"Error generating presigned URL: {e}")
            return None

    def sync_to_local(self, s3_prefix, local_dir):
        """Sync S3 prefix to local directory"""
        objects = self.list_objects(prefix=s3_prefix)

        for obj in objects:
            s3_key = obj['Key']
            local_path = os.path.join(local_dir, s3_key.replace(s3_prefix, '').lstrip('/'))

            # Create directories if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            self.download_file(s3_key, local_path)

# Example usage
if __name__ == '__main__':
    # Initialize manager
    bucket_name = os.environ.get('BUCKET_NAME', 'my-ml-bucket')
    s3_mgr = S3Manager(bucket_name)

    # Upload dataset
    print("\n=== Uploading dataset ===")
    s3_mgr.upload_file(
        'data/train.csv',
        'datasets/mnist/train.csv',
        metadata={'version': '1.0', 'type': 'training-data'}
    )

    # Upload model
    print("\n=== Uploading model ===")
    s3_mgr.upload_file('models/model.pth', 'models/mnist_v1.pth')

    # List models
    print("\n=== Listing models ===")
    s3_mgr.list_objects(prefix='models/')

    # Create presigned URL
    print("\n=== Creating presigned URL ===")
    s3_mgr.create_presigned_url('models/mnist_v1.pth', expiration=3600)

    # Get object versions
    print("\n=== Object versions ===")
    s3_mgr.get_object_versions('models/mnist_v1.pth')
```

### S3 Best Practices for ML

1. **Organize by project/version**:
   ```
   s3://my-ml-bucket/
   ├── datasets/
   │   ├── mnist/
   │   │   ├── train.csv
   │   │   └── test.csv
   │   └── cifar10/
   ├── models/
   │   ├── mnist/
   │   │   ├── v1.pth
   │   │   └── v2.pth
   │   └── cifar10/
   ├── experiments/
   │   └── exp-001/
   │       ├── config.json
   │       ├── metrics.json
   │       └── checkpoints/
   └── artifacts/
       └── plots/
   ```

2. **Use versioning** for models and datasets

3. **Implement lifecycle policies** to reduce costs

4. **Use S3 Select** for querying large datasets without downloading

## Part 3: IAM for ML Workflows

### IAM Policies for ML

**S3 Access Policy**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-ml-bucket",
        "arn:aws:s3:::my-ml-bucket/*"
      ]
    }
  ]
}
```

**EC2 Instance Role Policy**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-ml-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData"
      ],
      "Resource": "*"
    }
  ]
}
```

### Create IAM Role for EC2

```bash
# Create trust policy
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Service": "ec2.amazonaws.com"
    },
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Create role
aws iam create-role \
  --role-name MLInstanceRole \
  --assume-role-policy-document file://trust-policy.json

# Create policy
cat > ml-instance-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-ml-bucket",
        "arn:aws:s3:::my-ml-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
EOF

aws iam put-role-policy \
  --role-name MLInstanceRole \
  --policy-name MLInstancePolicy \
  --policy-document file://ml-instance-policy.json

# Create instance profile
aws iam create-instance-profile --instance-profile-name MLInstanceProfile

# Add role to instance profile
aws iam add-role-to-instance-profile \
  --instance-profile-name MLInstanceProfile \
  --role-name MLInstanceRole

# Attach instance profile to EC2 instance
aws ec2 associate-iam-instance-profile \
  --instance-id i-1234567890abcdef0 \
  --iam-instance-profile Name=MLInstanceProfile
```

## Part 4: Introduction to SageMaker

### SageMaker Training Job

```python
# sagemaker_training.py
import sagemaker
from sagemaker.pytorch import PyTorch
import boto3

# Setup
role = 'arn:aws:iam::123456789012:role/SageMakerRole'
session = sagemaker.Session()
bucket = session.default_bucket()

# Upload training data
s3_input_train = session.upload_data(
    path='data/train',
    bucket=bucket,
    key_prefix='mnist/train'
)

# Define PyTorch estimator
estimator = PyTorch(
    entry_point='train_sagemaker.py',
    source_dir='src',
    role=role,
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    framework_version='2.0',
    py_version='py310',
    hyperparameters={
        'epochs': 10,
        'batch-size': 64,
        'learning-rate': 0.001
    },
    output_path=f's3://{bucket}/models',
    code_location=f's3://{bucket}/code'
)

# Start training
estimator.fit({'training': s3_input_train})

# Get model artifacts
model_data = estimator.model_data
print(f"Model artifacts: {model_data}")
```

**train_sagemaker.py** (training script):
```python
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

def model_fn(model_dir):
    """Load model for inference"""
    model = Net()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_dataset = datasets.MNIST(
        args.data_dir,
        train=True,
        download=False,
        transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Initialize model
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

    # Save model
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    args = parser.parse_args()
    train(args)
```

### Deploy SageMaker Endpoint

```python
# Deploy model
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='mnist-endpoint'
)

# Make predictions
import numpy as np

test_data = np.random.randn(1, 1, 28, 28).astype('float32')
prediction = predictor.predict(test_data)
print(f"Prediction: {prediction}")

# Delete endpoint when done
predictor.delete_endpoint()
```

## Part 5: Cost Management

### AWS Cost Explorer

```python
# cost_explorer.py
import boto3
from datetime import datetime, timedelta

ce = boto3.client('ce', region_name='us-east-1')

# Get cost for last 30 days
end_date = datetime.now().date()
start_date = end_date - timedelta(days=30)

response = ce.get_cost_and_usage(
    TimePeriod={
        'Start': str(start_date),
        'End': str(end_date)
    },
    Granularity='DAILY',
    Metrics=['UnblendedCost'],
    GroupBy=[
        {'Type': 'DIMENSION', 'Key': 'SERVICE'}
    ]
)

# Print costs by service
for result in response['ResultsByTime']:
    print(f"\n{result['TimePeriod']['Start']}:")
    for group in result['Groups']:
        service = group['Keys'][0]
        cost = group['Metrics']['UnblendedCost']['Amount']
        print(f"  {service}: ${float(cost):.2f}")
```

### Cost Optimization Tips

1. **Use Spot Instances** for training (70-90% savings)
2. **Stop instances** when not in use
3. **Right-size instances** based on workload
4. **Use S3 lifecycle policies** to move old data to cheaper storage
5. **Set up billing alerts**:

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name monthly-cost-alert \
  --alarm-description "Alert when monthly cost exceeds $100" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 21600 \
  --evaluation-periods 1 \
  --threshold 100 \
  --comparison-operator GreaterThanThreshold
```

## Solutions

See `solutions/` directory for complete implementations:
- `ec2_manager.py` - EC2 instance management automation
- `s3_operations.py` - Complete S3 operations
- `sagemaker_pipeline.py` - End-to-end SageMaker pipeline
- `cost_monitor.py` - Cost tracking and alerts
- `infrastructure.tf` - Terraform configuration for AWS resources

## Exercises

1. Launch a GPU EC2 instance and train a PyTorch model
2. Create an S3 bucket and implement data versioning
3. Configure IAM roles for secure access
4. Run a SageMaker training job
5. Set up cost monitoring and alerts

## Next Steps

- **Exercise 02**: GCP ML Infrastructure - Compute Engine, GCS, Vertex AI

---

*Estimated completion time: 8-10 hours*
