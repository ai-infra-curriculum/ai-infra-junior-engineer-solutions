# Step-by-Step Implementation Guide: Multi-Cloud Deployment

## Overview

Deploy ML applications across multiple cloud providers! Learn multi-cloud strategies, Terraform for infrastructure-as-code, cross-cloud data sync, and disaster recovery.

**Time**: 3-4 hours | **Difficulty**: Advanced

---

## Learning Objectives

✅ Design multi-cloud architectures
✅ Use Terraform for cross-cloud provisioning
✅ Implement cross-cloud data replication
✅ Set up multi-cloud load balancing
✅ Implement disaster recovery
✅ Manage costs across clouds

---

## Multi-Cloud Strategy

### Benefits
- Avoid vendor lock-in
- Geographic redundancy
- Cost optimization
- Best-of-breed services

### Challenges
- Complexity
- Data transfer costs
- Compliance
- Tooling differences

---

## Terraform Setup

```hcl
# providers.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = "us-west-2"
}

provider "google" {
  project = "my-project"
  region  = "us-central1"
}

provider "azurerm" {
  features {}
}
```

### Infrastructure Definition

```hcl
# main.tf
# AWS Kubernetes Cluster
module "aws_eks" {
  source = "./modules/aws-eks"
  cluster_name = "ml-aws"
  region = "us-west-2"
}

# GCP Kubernetes Cluster
module "gcp_gke" {
  source = "./modules/gcp-gke"
  cluster_name = "ml-gcp"
  region = "us-central1"
}

# Azure Kubernetes Cluster
module "azure_aks" {
  source = "./modules/azure-aks"
  cluster_name = "ml-azure"
  location = "eastus"
}
```

---

## Cross-Cloud Data Replication

### S3 to GCS Sync

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure rclone
rclone config

# Sync S3 to GCS
rclone sync s3:my-ml-models gs:my-ml-models \
  --progress \
  --transfers=32

# Continuous sync (cron)
*/30 * * * * rclone sync s3:my-ml-models gs:my-ml-models
```

### Cross-Cloud Transfer Service

```python
# AWS to GCP transfer
from google.cloud import storage_transfer

transfer_client = storage_transfer.StorageTransferServiceClient()

transfer_job = {
    'description': 'S3 to GCS sync',
    'status': 'ENABLED',
    'schedule': {
        'schedule_start_date': {'year': 2024, 'month': 1, 'day': 1}
    },
    'transfer_spec': {
        'aws_s3_data_source': {
            'bucket_name': 'my-ml-models',
            'aws_access_key': {'access_key_id': 'KEY', 'secret_access_key': 'SECRET'}
        },
        'gcs_data_sink': {
            'bucket_name': 'my-ml-models'
        }
    }
}

response = transfer_client.create_transfer_job(transfer_job=transfer_job)
```

---

## Multi-Cloud Load Balancing

### Global Load Balancer with Cloudflare

```bash
# Configure Cloudflare
curl -X POST "https://api.cloudflare.com/client/v4/zones/ZONE_ID/load_balancers" \
  -H "Authorization: Bearer TOKEN" \
  -d '{
    "name": "ml-api.example.com",
    "default_pools": ["aws-pool", "gcp-pool", "azure-pool"],
    "fallback_pool": "aws-pool",
    "region_pools": {
      "WNAM": ["aws-pool"],
      "ENAM": ["azure-pool"],
      "WEU": ["gcp-pool"]
    }
  }'
```

---

## Disaster Recovery

### Active-Active Setup

```yaml
# Deploy to all clouds
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
        cloud: ${CLOUD_PROVIDER}
    spec:
      containers:
      - name: api
        image: ml-api:latest
        env:
        - name: CLOUD_PROVIDER
          value: ${CLOUD_PROVIDER}
```

### Failover Strategy

```python
# Health check and failover
import requests

def check_cloud_health(endpoints):
    for cloud, endpoint in endpoints.items():
        try:
            response = requests.get(f"{endpoint}/health", timeout=5)
            if response.status_code == 200:
                return cloud, endpoint
        except:
            continue
    return None, None

endpoints = {
    'aws': 'https://api-aws.example.com',
    'gcp': 'https://api-gcp.example.com',
    'azure': 'https://api-azure.example.com'
}

active_cloud, active_endpoint = check_cloud_health(endpoints)
```

---

## Cost Optimization

### Cost Comparison Script

```python
import boto3
import google.cloud.billing
from azure.mgmt.costmanagement import CostManagementClient

def get_aws_costs():
    ce = boto3.client('ce')
    response = ce.get_cost_and_usage(
        TimePeriod={'Start': '2024-01-01', 'End': '2024-01-31'},
        Granularity='MONTHLY',
        Metrics=['UnblendedCost']
    )
    return float(response['ResultsByTime'][0]['Total']['UnblendedCost']['Amount'])

def get_gcp_costs():
    # Implementation for GCP billing API
    pass

def get_azure_costs():
    # Implementation for Azure cost management
    pass

total_aws = get_aws_costs()
total_gcp = get_gcp_costs()
total_azure = get_azure_costs()

print(f"Total monthly costs:")
print(f"AWS: ${total_aws:.2f}")
print(f"GCP: ${total_gcp:.2f}")
print(f"Azure: ${total_azure:.2f}")
print(f"Total: ${total_aws + total_gcp + total_azure:.2f}")
```

---

## Best Practices

✅ Use infrastructure-as-code (Terraform)
✅ Implement unified monitoring
✅ Automate cross-cloud data sync
✅ Use cloud-agnostic tools (Kubernetes, Docker)
✅ Implement disaster recovery plans
✅ Monitor costs across all clouds
✅ Use managed services where possible
✅ Implement proper tagging
✅ Test failover procedures regularly

---

**Multi-Cloud Deployment mastered!** 🌐
