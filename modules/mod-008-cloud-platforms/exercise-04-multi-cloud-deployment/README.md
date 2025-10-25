# Exercise 04: Multi-Cloud ML Deployment

Learn to deploy and manage ML infrastructure across multiple cloud providers (AWS, GCP, Azure) using unified tools and strategies.

## Learning Objectives

- Deploy ML infrastructure across multiple clouds
- Use Terraform for multi-cloud infrastructure as code
- Implement Kubernetes clusters on EKS, GKE, and AKS
- Deploy models with vendor-agnostic tools
- Manage multi-cloud networking and data transfer
- Implement cross-cloud failover and high availability
- Monitor and observe multi-cloud deployments

## Prerequisites

- Completion of Exercises 01-03
- AWS, GCP, and Azure accounts
- Terraform installed
- kubectl and Helm installed
- Understanding of Kubernetes

## Multi-Cloud Strategies

### 1. Vendor-Agnostic Approach

Use portable technologies that work across all clouds:
- **Kubernetes** (EKS, GKE, AKS)
- **Docker** containers
- **Terraform** for infrastructure
- **Helm** for application deployment
- **Prometheus/Grafana** for monitoring

### 2. Best-of-Breed Approach

Use the best services from each provider:
- **AWS**: Cost-effective spot instances
- **GCP**: TPUs for large model training, superior ML tooling
- **Azure**: Enterprise integration, hybrid cloud

### 3. High Availability Approach

Deploy across multiple clouds for redundancy:
- Primary region: AWS us-east-1
- Secondary region: GCP us-central1
- Tertiary region: Azure eastus

### 4. Data Residency Approach

Use specific clouds for regulatory compliance:
- EU data: GCP europe-west1
- US data: AWS us-east-1
- Asia data: Azure southeastasia

## Part 1: Multi-Cloud Infrastructure with Terraform

### Terraform Multi-Cloud Configuration

```hcl
# main.tf
terraform {
  required_version = ">= 1.0"

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
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

# AWS Provider
provider "aws" {
  region = var.aws_region
}

# GCP Provider
provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

# Azure Provider
provider "azurerm" {
  features {}
  subscription_id = var.azure_subscription_id
}

# Variables
variable "aws_region" {
  default = "us-east-1"
}

variable "gcp_project" {}
variable "gcp_region" {
  default = "us-central1"
}

variable "azure_subscription_id" {}
variable "azure_location" {
  default = "eastus"
}

# AWS EKS Cluster
module "eks" {
  source = "./modules/eks"

  cluster_name    = "ml-cluster-aws"
  region          = var.aws_region
  node_groups = {
    gpu = {
      instance_types = ["p3.2xlarge"]
      min_size       = 1
      max_size       = 5
      desired_size   = 2
    }
  }
}

# GCP GKE Cluster
module "gke" {
  source = "./modules/gke"

  cluster_name = "ml-cluster-gcp"
  project      = var.gcp_project
  region       = var.gcp_region
  node_pools = {
    gpu = {
      machine_type  = "n1-standard-8"
      accelerator   = "nvidia-tesla-v100"
      min_nodes     = 1
      max_nodes     = 5
    }
  }
}

# Azure AKS Cluster
module "aks" {
  source = "./modules/aks"

  cluster_name        = "ml-cluster-azure"
  resource_group_name = azurerm_resource_group.ml.name
  location            = var.azure_location
  node_pools = {
    gpu = {
      vm_size    = "Standard_NC6s_v3"
      min_count  = 1
      max_count  = 5
      node_count = 2
    }
  }
}

# Azure Resource Group
resource "azurerm_resource_group" "ml" {
  name     = "ml-multi-cloud"
  location = var.azure_location
}
```

### EKS Module

```hcl
# modules/eks/main.tf
resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  role_arn = aws_iam_role.cluster.arn
  version  = "1.28"

  vpc_config {
    subnet_ids              = var.subnet_ids
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  depends_on = [
    aws_iam_role_policy_attachment.cluster_policy
  ]
}

resource "aws_eks_node_group" "gpu" {
  for_each = var.node_groups

  cluster_name    = aws_eks_cluster.main.name
  node_group_name = each.key
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.subnet_ids

  instance_types = each.value.instance_types
  capacity_type  = "SPOT"

  scaling_config {
    desired_size = each.value.desired_size
    max_size     = each.value.max_size
    min_size     = each.value.min_size
  }

  labels = {
    workload = "ml"
  }

  taints {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  depends_on = [
    aws_iam_role_policy_attachment.node_policy
  ]
}

# IAM Roles (simplified)
resource "aws_iam_role" "cluster" {
  name = "${var.cluster_name}-cluster-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "eks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "cluster_policy" {
  role       = aws_iam_role.cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}
```

## Part 2: Kubernetes Multi-Cloud Deployment

### Common Kubernetes Manifests

Deploy the same application across all three clouds:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
  labels:
    app: ml-inference
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
      containers:
      - name: api
        image: ghcr.io/your-org/ml-inference:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        env:
        - name: CLOUD_PROVIDER
          value: "REPLACE_WITH_CLOUD"  # AWS, GCP, or Azure
        - name: MODEL_PATH
          value: "REPLACE_WITH_CLOUD_STORAGE"  # s3://, gs://, or wasb://
---
apiVersion: v1
kind: Service
metadata:
  name: ml-inference
spec:
  type: LoadBalancer
  selector:
    app: ml-inference
  ports:
  - port: 80
    targetPort: 8000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Deploy to All Clouds

```bash
# Configure kubectl contexts
aws eks update-kubeconfig --name ml-cluster-aws --region us-east-1 --alias aws
gcloud container clusters get-credentials ml-cluster-gcp --region us-central1 --alias gcp
az aks get-credentials --resource-group ml-multi-cloud --name ml-cluster-azure --alias azure

# Deploy to AWS
kubectl --context aws apply -f deployment.yaml

# Deploy to GCP
kubectl --context gcp apply -f deployment.yaml

# Deploy to Azure
kubectl --context azure apply -f deployment.yaml

# Check deployments
for ctx in aws gcp azure; do
  echo "=== $ctx ==="
  kubectl --context $ctx get deployments,pods,svc
done
```

## Part 3: Multi-Cloud Data Management

### Data Synchronization

```python
# multi_cloud_sync.py
import boto3
from google.cloud import storage as gcs_storage
from azure.storage.blob import BlobServiceClient

class MultiCloudStorage:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.gcs = gcs_storage.Client()
        self.azure_blob = BlobServiceClient.from_connection_string(
            os.environ['AZURE_STORAGE_CONNECTION_STRING']
        )

    def sync_dataset(self, local_path: str):
        """Sync dataset to all clouds"""
        # Upload to S3
        self.s3.upload_file(
            local_path,
            'ml-data-aws',
            'datasets/train.csv'
        )
        print("✓ Synced to AWS S3")

        # Upload to GCS
        bucket = self.gcs.bucket('ml-data-gcp')
        blob = bucket.blob('datasets/train.csv')
        blob.upload_from_filename(local_path)
        print("✓ Synced to GCP GCS")

        # Upload to Azure Blob
        blob_client = self.azure_blob.get_blob_client(
            container='ml-data',
            blob='datasets/train.csv'
        )
        with open(local_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
        print("✓ Synced to Azure Blob")

    def cross_cloud_transfer(self, source_cloud: str, dest_cloud: str, key: str):
        """Transfer data between clouds"""
        # Download from source
        if source_cloud == 'aws':
            temp_file = '/tmp/data'
            self.s3.download_file('ml-data-aws', key, temp_file)
        elif source_cloud == 'gcp':
            bucket = self.gcs.bucket('ml-data-gcp')
            blob = bucket.blob(key)
            temp_file = '/tmp/data'
            blob.download_to_filename(temp_file)

        # Upload to destination
        if dest_cloud == 'aws':
            self.s3.upload_file(temp_file, 'ml-data-aws', key)
        elif dest_cloud == 'gcp':
            bucket = self.gcs.bucket('ml-data-gcp')
            blob = bucket.blob(key)
            blob.upload_from_filename(temp_file)
        elif dest_cloud == 'azure':
            blob_client = self.azure_blob.get_blob_client('ml-data', key)
            with open(temp_file, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)

        print(f"✓ Transferred {key} from {source_cloud} to {dest_cloud}")
```

## Part 4: Multi-Cloud Monitoring

### Unified Monitoring with Prometheus

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
  # AWS EKS
  - job_name: 'aws-ml-inference'
    kubernetes_sd_configs:
    - api_server: 'https://eks-api-server'
      role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      regex: ml-inference
      action: keep
    - source_labels: [__meta_kubernetes_pod_annotation_cloud]
      target_label: cloud
      replacement: aws

  # GCP GKE
  - job_name: 'gcp-ml-inference'
    kubernetes_sd_configs:
    - api_server: 'https://gke-api-server'
      role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      regex: ml-inference
      action: keep
    - source_labels: [__meta_kubernetes_pod_annotation_cloud]
      target_label: cloud
      replacement: gcp

  # Azure AKS
  - job_name: 'azure-ml-inference'
    kubernetes_sd_configs:
    - api_server: 'https://aks-api-server'
      role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      regex: ml-inference
      action: keep
    - source_labels: [__meta_kubernetes_pod_annotation_cloud]
      target_label: cloud
      replacement: azure
```

## Part 5: Multi-Cloud Failover

### DNS-Based Failover

```bash
# Route 53 health checks and failover
aws route53 create-health-check \
  --health-check-config IPAddress=<AWS_LB_IP>,Port=80,Type=HTTP,ResourcePath=/health

aws route53 create-health-check \
  --health-check-config IPAddress=<GCP_LB_IP>,Port=80,Type=HTTP,ResourcePath=/health

# Create failover routing
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch file://failover-config.json
```

**failover-config.json**:
```json
{
  "Changes": [{
    "Action": "CREATE",
    "ResourceRecordSet": {
      "Name": "ml-api.example.com",
      "Type": "A",
      "SetIdentifier": "Primary-AWS",
      "Failover": "PRIMARY",
      "AliasTarget": {
        "HostedZoneId": "Z1234567890ABC",
        "DNSName": "aws-lb.elb.amazonaws.com",
        "EvaluateTargetHealth": true
      }
    }
  }, {
    "Action": "CREATE",
    "ResourceRecordSet": {
      "Name": "ml-api.example.com",
      "Type": "A",
      "SetIdentifier": "Secondary-GCP",
      "Failover": "SECONDARY",
      "TTL": 60,
      "ResourceRecords": [{"Value": "<GCP_LB_IP>"}]
    }
  }]
}
```

## Part 6: Cost Optimization

### Multi-Cloud Cost Comparison

```python
# multi_cloud_cost.py
from aws_cost_monitor import AWSCostMonitor
from gcp_cost_monitor import GCPCostMonitor
from azure_cost_monitor import AzureCostMonitor

class MultiCloudCostMonitor:
    def __init__(self):
        self.aws = AWSCostMonitor()
        self.gcp = GCPCostMonitor()
        self.azure = AzureCostMonitor()

    def compare_costs(self, start_date: str, end_date: str):
        """Compare costs across clouds"""
        aws_cost = self.aws.get_total_cost(start_date, end_date)
        gcp_cost = self.gcp.get_total_cost(start_date, end_date)
        azure_cost = self.azure.get_total_cost(start_date, end_date)

        print(f"AWS:   ${aws_cost:.2f}")
        print(f"GCP:   ${gcp_cost:.2f}")
        print(f"Azure: ${azure_cost:.2f}")
        print(f"Total: ${aws_cost + gcp_cost + azure_cost:.2f}")

        # Find cheapest
        costs = {'AWS': aws_cost, 'GCP': gcp_cost, 'Azure': azure_cost}
        cheapest = min(costs, key=costs.get)
        print(f"\nCheapest: {cheapest} (${costs[cheapest]:.2f})")
```

## Solutions

See `solutions/` directory for complete implementations:
- `multi_cloud_terraform/` - Complete Terraform configurations
- `multi_cloud_deploy.py` - Automated multi-cloud deployment
- `data_sync.py` - Cross-cloud data synchronization
- `cost_optimizer.py` - Multi-cloud cost optimization
- `failover_manager.py` - Automated failover management

## Exercises

1. Deploy Terraform infrastructure to all three clouds
2. Deploy identical Kubernetes workloads to EKS, GKE, and AKS
3. Implement data synchronization across clouds
4. Set up unified monitoring with Prometheus
5. Configure DNS failover between clouds
6. Compare costs and optimize across clouds

## Next Steps

- **Exercise 05**: Cost Optimization & Management - Advanced cost optimization strategies

---

*Estimated completion time: 12-15 hours*
