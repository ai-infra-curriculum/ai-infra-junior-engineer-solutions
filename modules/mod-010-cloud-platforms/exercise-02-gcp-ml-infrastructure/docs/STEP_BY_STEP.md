# Step-by-Step Implementation Guide: GCP ML Infrastructure

## Overview

Deploy ML infrastructure on Google Cloud Platform! Learn Compute Engine, Cloud Storage, GKE, Vertex AI, and production ML patterns on GCP.

**Time**: 3 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Configure gcloud CLI
✅ Use Compute Engine for ML workloads
✅ Store data in Cloud Storage
✅ Deploy on GKE
✅ Use Vertex AI for training
✅ Implement CI/CD with Cloud Build
✅ Monitor with Cloud Monitoring

---

## Setup

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize
gcloud init

# Set project
gcloud config set project PROJECT_ID

# Enable APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

---

## Compute Engine

```bash
# Create GPU VM
gcloud compute instances create ml-training \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB

# SSH to instance
gcloud compute ssh ml-training --zone=us-central1-a
```

---

## Cloud Storage

```bash
# Create buckets
gsutil mb -l us-central1 gs://my-ml-datasets
gsutil mb -l us-central1 gs://my-ml-models

# Upload data
gsutil -m cp -r dataset/ gs://my-ml-datasets/

# Download model
gsutil cp gs://my-ml-models/model.pth ./

# Set lifecycle
cat > lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [{
      "action": {"type": "Delete"},
      "condition": {"age": 30}
    }]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://my-ml-models
```

---

## GKE Deployment

```bash
# Create cluster
gcloud container clusters create ml-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster ml-cluster \
  --zone us-central1-a \
  --num-nodes 0 \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 5

# Get credentials
gcloud container clusters get-credentials ml-cluster --zone us-central1-a

# Deploy
kubectl apply -f ml-deployment.yaml
```

---

## Vertex AI

```python
from google.cloud import aiplatform

aiplatform.init(project='PROJECT_ID', location='us-central1')

# Custom training
job = aiplatform.CustomTrainingJob(
    display_name='resnet-training',
    script_path='train.py',
    container_uri='gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest',
    requirements=['torch', 'torchvision'],
    model_serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-13:latest'
)

model = job.run(
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=1,
    replica_count=1
)

# Deploy endpoint
endpoint = model.deploy(
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=10
)
```

---

## Best Practices

✅ Use service accounts for authentication
✅ Enable Cloud Armor for DDoS protection
✅ Use preemptible VMs for training
✅ Implement auto-scaling
✅ Monitor with Cloud Monitoring
✅ Use Cloud CDN for model serving
✅ Enable VPC Service Controls

---

**GCP ML Infrastructure mastered!** 🔵
