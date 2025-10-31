# Step-by-Step Implementation Guide: Azure ML Services

## Overview

Build ML infrastructure on Microsoft Azure! Learn Azure ML, AKS, Blob Storage, and enterprise ML deployment patterns.

**Time**: 2-3 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Configure Azure CLI
✅ Use Azure ML for training
✅ Deploy models on AKS
✅ Store data in Blob Storage
✅ Implement MLOps with Azure DevOps
✅ Monitor with Application Insights

---

## Setup

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Set subscription
az account set --subscription "SUBSCRIPTION_ID"

# Create resource group
az group create --name ml-resources --location eastus
```

---

## Azure ML Workspace

```bash
# Create workspace
az ml workspace create \
  --name ml-workspace \
  --resource-group ml-resources \
  --location eastus

# Create compute
az ml compute create \
  --name gpu-cluster \
  --type amlcompute \
  --size Standard_NC6 \
  --min-instances 0 \
  --max-instances 4 \
  --resource-group ml-resources \
  --workspace-name ml-workspace
```

### Training Job

```python
from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential

# Connect to workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="SUBSCRIPTION_ID",
    resource_group_name="ml-resources",
    workspace_name="ml-workspace"
)

# Define job
job = command(
    code="./src",
    command="python train.py --epochs 100",
    environment="azureml:pytorch-1.13-gpu:latest",
    compute="gpu-cluster",
    inputs={
        "data": Input(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/data"
        )
    }
)

# Submit
ml_client.jobs.create_or_update(job)
```

---

## AKS Deployment

```bash
# Create AKS cluster
az aks create \
  --resource-group ml-resources \
  --name ml-cluster \
  --node-count 3 \
  --enable-managed-identity \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10

# Get credentials
az aks get-credentials --resource-group ml-resources --name ml-cluster

# Deploy model
kubectl apply -f ml-deployment.yaml
```

---

## Blob Storage

```bash
# Create storage account
az storage account create \
  --name mlstorage \
  --resource-group ml-resources \
  --sku Standard_LRS

# Create container
az storage container create \
  --name models \
  --account-name mlstorage

# Upload
az storage blob upload \
  --account-name mlstorage \
  --container-name models \
  --name model.pth \
  --file ./model.pth
```

---

## Best Practices

✅ Use managed identities
✅ Enable Azure Active Directory integration
✅ Implement Azure Key Vault for secrets
✅ Use Azure Monitor for observability
✅ Enable auto-scaling
✅ Implement Azure Policy for governance

---

**Azure ML Services mastered!** ⚡
