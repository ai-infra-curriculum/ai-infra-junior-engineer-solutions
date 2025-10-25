# Exercise 02: GCP ML Infrastructure

## Overview

This exercise demonstrates how to build production-ready ML infrastructure on Google Cloud Platform (GCP) using Terraform and Python automation. It covers Vertex AI, GKE, Cloud Storage, BigQuery, and best practices for ML workloads.

## Comprehensive Implementation

This solution includes:
- Complete Terraform infrastructure as code
- Python automation scripts for GCP services
- Production-ready GKE cluster with GPU support
- Cloud Storage buckets with lifecycle policies
- BigQuery dataset configuration
- Vertex AI integration
- Comprehensive tests (15+ test cases)

---

## Original Exercise: Learn Google Cloud Platform services for machine learning infrastructure including Compute Engine, Cloud Storage, and Vertex AI.

## Learning Objectives

- Launch and manage GPU/TPU instances on Compute Engine
- Store and manage ML data in Google Cloud Storage
- Configure IAM and service accounts for ML workflows
- Train and deploy models using Vertex AI
- Use Cloud Run for serverless inference
- Understand GCP billing and cost optimization

## Prerequisites

- GCP account (free tier available with $300 credit)
- gcloud CLI installed and configured
- Python 3.8+ with google-cloud libraries
- Basic understanding of cloud computing

## GCP Account Setup

### 1. Create GCP Account

1. Go to [cloud.google.com](https://cloud.google.com)
2. Click "Get started for free"
3. Follow the signup process
4. Get $300 in free credits

### 2. Set Up gcloud CLI

```bash
# Install gcloud CLI
# macOS
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Verify installation
gcloud --version

# Initialize gcloud
gcloud init

# Set default project
gcloud config set project YOUR_PROJECT_ID

# Set default region and zone
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a

# Authenticate
gcloud auth login
gcloud auth application-default login
```

### 3. Enable Required APIs

```bash
# Enable Compute Engine API
gcloud services enable compute.googleapis.com

# Enable Cloud Storage API
gcloud services enable storage.googleapis.com

# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable Container Registry
gcloud services enable containerregistry.googleapis.com

# Enable Cloud Build
gcloud services enable cloudbuild.googleapis.com
```

## Part 1: Compute Engine for ML Training

### Launch GPU Instance

```bash
# List available GPU types
gcloud compute accelerator-types list

# Create GPU instance
gcloud compute instances create ml-gpu-instance \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform

# SSH into instance
gcloud compute ssh ml-gpu-instance --zone=us-central1-a

# Verify GPU
nvidia-smi
```

### Launch TPU Instance

TPUs offer better price/performance for large models:

```bash
# Create TPU VM
gcloud compute tpus tpu-vm create ml-tpu \
  --zone=us-central1-a \
  --accelerator-type=v3-8 \
  --version=tpu-vm-tf-2.13.0

# SSH into TPU VM
gcloud compute tpus tpu-vm ssh ml-tpu --zone=us-central1-a

# Verify TPU
python3 -c "import tensorflow as tf; print(tf.config.list_logical_devices('TPU'))"
```

### Using Preemptible Instances

Preemptible instances cost 70-80% less:

```bash
# Create preemptible GPU instance
gcloud compute instances create ml-gpu-preemptible \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --preemptible \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform
```

### Training Script on Compute Engine

```python
# train_gce.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from google.cloud import storage
import os

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
        return torch.log_softmax(self.fc2(x), dim=1)

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
            print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

    print(f'Epoch {epoch+1} completed, Average Loss: {total_loss/len(train_loader):.4f}')

# Save model
torch.save(model.state_dict(), 'model.pth')

# Upload to GCS
storage_client = storage.Client()
bucket_name = os.environ.get('GCS_BUCKET', 'my-ml-bucket')
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob('models/mnist_model.pth')
blob.upload_from_filename('model.pth')
print(f"Model uploaded to gs://{bucket_name}/models/mnist_model.pth")
```

## Part 2: Google Cloud Storage

### Create GCS Bucket

```bash
# Create bucket
gsutil mb -l us-central1 gs://my-ml-bucket-$(date +%s)

# Set bucket name variable
export BUCKET_NAME=my-ml-bucket-1234567890

# Enable versioning
gsutil versioning set on gs://$BUCKET_NAME

# Set lifecycle policy
cat > lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {
          "age": 30,
          "matchesPrefix": ["models/"]
        }
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
        "condition": {
          "age": 90,
          "matchesPrefix": ["models/"]
        }
      },
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 7,
          "matchesPrefix": ["temp/"]
        }
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://$BUCKET_NAME
```

### GCS Operations with Python

See `solutions/gcs_operations.py` for complete implementation.

### GCS Best Practices

1. **Organize by project/version**:
   ```
   gs://my-ml-bucket/
   ├── datasets/
   │   ├── mnist/
   │   └── cifar10/
   ├── models/
   │   ├── mnist/v1/
   │   └── cifar10/v1/
   ├── experiments/
   └── artifacts/
   ```

2. **Use object versioning** for models and datasets
3. **Implement lifecycle policies** to optimize costs
4. **Use signed URLs** for temporary access

## Part 3: IAM and Service Accounts

### Create Service Account

```bash
# Create service account
gcloud iam service-accounts create ml-training-sa \
  --display-name="ML Training Service Account"

# Get service account email
SA_EMAIL=$(gcloud iam service-accounts list \
  --filter="displayName:ML Training Service Account" \
  --format='value(email)')

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/aiplatform.user"

# Create key
gcloud iam service-accounts keys create key.json \
  --iam-account=$SA_EMAIL

# Use service account
export GOOGLE_APPLICATION_CREDENTIALS="key.json"
```

### Attach Service Account to Instance

```bash
# Create instance with service account
gcloud compute instances create ml-instance \
  --service-account=$SA_EMAIL \
  --scopes=cloud-platform \
  --zone=us-central1-a
```

## Part 4: Vertex AI

### Vertex AI Training Job

```python
# vertex_ai_training.py
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project='my-project', location='us-central1')

# Create custom training job
job = aiplatform.CustomTrainingJob(
    display_name='pytorch-mnist-training',
    container_uri='gcr.io/my-project/trainer:latest',
    requirements=['torch==2.0.0', 'torchvision'],
    model_serving_container_image_uri='gcr.io/my-project/predictor:latest'
)

# Run training
model = job.run(
    dataset=dataset,
    replica_count=1,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=1,
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    model_display_name='mnist-classifier'
)

# Deploy model
endpoint = model.deploy(
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=10,
    traffic_split={'0': 100}
)

# Make predictions
prediction = endpoint.predict(instances=[data])
print(prediction)
```

### Vertex AI Hyperparameter Tuning

```python
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# Define hyperparameter spec
hparam_spec = {
    'learning_rate': hpt.DoubleParameterSpec(min=0.0001, max=0.1, scale='log'),
    'batch_size': hpt.DiscreteParameterSpec(values=[32, 64, 128]),
    'num_layers': hpt.IntegerParameterSpec(min=2, max=5, scale='linear')
}

# Create tuning job
tuning_job = aiplatform.HyperparameterTuningJob(
    display_name='hp-tuning-job',
    custom_job=job,
    metric_spec={'accuracy': 'maximize'},
    parameter_spec=hparam_spec,
    max_trial_count=20,
    parallel_trial_count=5
)

# Run tuning
tuning_job.run()

# Get best trial
best_trial = tuning_job.trials[0]
print(f"Best hyperparameters: {best_trial.parameters}")
print(f"Best accuracy: {best_trial.final_measurement.metrics[0].value}")
```

## Part 5: Cloud Run for Inference

### Deploy Inference API

```bash
# Build container
gcloud builds submit --tag gcr.io/$PROJECT_ID/ml-api

# Deploy to Cloud Run
gcloud run deploy ml-api \
  --image gcr.io/$PROJECT_ID/ml-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --set-env-vars MODEL_PATH=gs://my-bucket/models/model.pth

# Get service URL
SERVICE_URL=$(gcloud run services describe ml-api \
  --platform managed \
  --region us-central1 \
  --format='value(status.url)')

# Test endpoint
curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[1, 2, 3, 4]]}'
```

## Part 6: Cost Management

### View Billing

```bash
# Enable Billing API
gcloud services enable cloudbilling.googleapis.com

# List billing accounts
gcloud billing accounts list

# Export billing data to BigQuery
gcloud billing accounts export create \
  --billing-account=BILLING_ACCOUNT_ID \
  --destination-dataset=billing_export \
  --destination-table=gcp_billing_export
```

### Cost Optimization Tips

1. **Use preemptible instances** (70-80% savings)
2. **Use committed use discounts** (30-50% savings for 1-3 year commitments)
3. **Right-size instances** based on actual usage
4. **Use lifecycle policies** for GCS
5. **Set up budget alerts**:

```bash
# Create budget
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Monthly ML Budget" \
  --budget-amount=1000 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100
```

## Solutions

See `solutions/` directory for complete implementations:
- `gce_manager.py` - Compute Engine management automation
- `gcs_operations.py` - Complete GCS operations
- `vertex_ai_pipeline.py` - End-to-end Vertex AI pipeline
- `cloud_run_deploy.py` - Cloud Run deployment automation
- `cost_monitor.py` - GCP cost tracking
- `infrastructure.tf` - Terraform configuration for GCP

## Exercises

1. Launch a GPU Compute Engine instance and train a model
2. Create a GCS bucket and implement data versioning
3. Configure service accounts for secure access
4. Run a Vertex AI training job
5. Deploy inference API to Cloud Run
6. Set up cost monitoring and alerts

## Next Steps

- **Exercise 03**: Azure ML Services - Azure VMs, Blob Storage, Azure ML

---

*Estimated completion time: 8-10 hours*
