# Exercise 03: Azure ML Services

Learn Microsoft Azure services for machine learning infrastructure including Azure VMs, Blob Storage, and Azure Machine Learning.

## Learning Objectives

- Launch and manage GPU VMs on Azure
- Store and manage ML data in Azure Blob Storage
- Configure Azure AD and managed identities
- Train and deploy models using Azure Machine Learning
- Use Azure Functions for serverless inference
- Understand Azure billing and cost management

## Prerequisites

- Azure account (free tier available with $200 credit)
- Azure CLI installed and configured
- Python 3.8+ with azure libraries
- Basic understanding of cloud computing

## Azure Account Setup

### 1. Create Azure Account

1. Go to [azure.microsoft.com](https://azure.microsoft.com)
2. Click "Start free"
3. Follow the signup process
4. Get $200 in free credits

### 2. Set Up Azure CLI

```bash
# Install Azure CLI
# macOS
brew install azure-cli

# Linux
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Windows
# Download from https://aka.ms/installazurecliwindows

# Verify installation
az --version

# Login
az login

# Set default subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Set default location
az configure --defaults location=eastus

# Create resource group
az group create --name ml-resources --location eastus
```

### 3. Enable Required Services

```bash
# Register resource providers
az provider register --namespace Microsoft.Compute
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.ContainerRegistry
```

## Part 1: Azure Virtual Machines for ML

### Launch GPU VM

```bash
# List available VM sizes with GPUs
az vm list-sizes --location eastus --output table | grep NC

# Create GPU VM
az vm create \
  --resource-group ml-resources \
  --name ml-gpu-vm \
  --image UbuntuLTS \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard

# Install NVIDIA drivers (after SSH)
ssh azureuser@<PUBLIC_IP>
sudo apt-get update
sudo apt-get install -y nvidia-driver-515
sudo reboot

# Verify GPU
nvidia-smi
```

### Create VM Scale Set

```bash
# Create scale set for distributed training
az vmss create \
  --resource-group ml-resources \
  --name ml-scaleset \
  --image UbuntuLTS \
  --vm-sku Standard_NC6s_v3 \
  --instance-count 2 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --upgrade-policy-mode automatic

# Scale up/down
az vmss scale \
  --resource-group ml-resources \
  --name ml-scaleset \
  --new-capacity 4
```

### Using Spot VMs

```bash
# Create spot VM (up to 90% savings)
az vm create \
  --resource-group ml-resources \
  --name ml-spot-vm \
  --image UbuntuLTS \
  --size Standard_NC6s_v3 \
  --priority Spot \
  --max-price -1 \
  --eviction-policy Deallocate \
  --admin-username azureuser \
  --generate-ssh-keys
```

## Part 2: Azure Blob Storage

### Create Storage Account

```bash
# Create storage account
az storage account create \
  --name mlstorage$(date +%s) \
  --resource-group ml-resources \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2

# Get connection string
CONNECTION_STRING=$(az storage account show-connection-string \
  --name mlstorage1234567890 \
  --resource-group ml-resources \
  --output tsv)

# Create container
az storage container create \
  --name ml-data \
  --connection-string $CONNECTION_STRING

# Set lifecycle policy
cat > lifecycle-policy.json <<EOF
{
  "rules": [
    {
      "enabled": true,
      "name": "archive-old-models",
      "type": "Lifecycle",
      "definition": {
        "actions": {
          "baseBlob": {
            "tierToCool": {"daysAfterModificationGreaterThan": 30},
            "tierToArchive": {"daysAfterModificationGreaterThan": 90}
          }
        },
        "filters": {
          "prefixMatch": ["models/"],
          "blobTypes": ["blockBlob"]
        }
      }
    },
    {
      "enabled": true,
      "name": "delete-temp-data",
      "type": "Lifecycle",
      "definition": {
        "actions": {
          "baseBlob": {
            "delete": {"daysAfterModificationGreaterThan": 7}
          }
        },
        "filters": {
          "prefixMatch": ["temp/"],
          "blobTypes": ["blockBlob"]
        }
      }
    }
  ]
}
EOF

az storage account management-policy create \
  --account-name mlstorage1234567890 \
  --resource-group ml-resources \
  --policy @lifecycle-policy.json
```

### Blob Operations with Python

```python
# blob_operations.py
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob import BlobSasPermissions, generate_blob_sas
from datetime import datetime, timedelta
import os

class AzureBlobManager:
    def __init__(self, connection_string: str):
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)

    def upload_file(self, container_name: str, local_path: str, blob_name: str):
        """Upload file to blob storage"""
        blob_client = self.blob_service.get_blob_client(
            container=container_name,
            blob=blob_name
        )

        with open(local_path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)

        print(f"Uploaded {local_path} to {container_name}/{blob_name}")

    def download_file(self, container_name: str, blob_name: str, local_path: str):
        """Download file from blob storage"""
        blob_client = self.blob_service.get_blob_client(
            container=container_name,
            blob=blob_name
        )

        with open(local_path, 'wb') as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())

        print(f"Downloaded {container_name}/{blob_name} to {local_path}")

    def list_blobs(self, container_name: str, prefix: str = ''):
        """List blobs in container"""
        container_client = self.blob_service.get_container_client(container_name)

        blobs = container_client.list_blobs(name_starts_with=prefix)

        for blob in blobs:
            size_mb = blob.size / (1024 * 1024)
            print(f"{blob.name}: {size_mb:.2f} MB (Modified: {blob.last_modified})")

    def generate_sas_url(self, container_name: str, blob_name: str, expiry_hours: int = 1):
        """Generate SAS URL for temporary access"""
        account_name = self.blob_service.account_name
        account_key = self.blob_service.credential.account_key

        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
        )

        url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
        print(f"SAS URL (expires in {expiry_hours}h): {url}")
        return url

# Usage
if __name__ == '__main__':
    connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    manager = AzureBlobManager(connection_string)

    # Upload dataset
    manager.upload_file('ml-data', 'data/train.csv', 'datasets/mnist/train.csv')

    # List models
    manager.list_blobs('ml-data', prefix='models/')

    # Generate SAS URL
    manager.generate_sas_url('ml-data', 'models/model.pth', expiry_hours=24)
```

## Part 3: Azure AD and Managed Identities

### Create Service Principal

```bash
# Create service principal
az ad sp create-for-rbac \
  --name ml-training-sp \
  --role Contributor \
  --scopes /subscriptions/SUBSCRIPTION_ID/resourceGroups/ml-resources

# Assign specific permissions
az role assignment create \
  --assignee APP_ID \
  --role "Storage Blob Data Contributor" \
  --scope /subscriptions/SUBSCRIPTION_ID/resourceGroups/ml-resources
```

### Use Managed Identity

```bash
# Enable system-assigned managed identity on VM
az vm identity assign \
  --resource-group ml-resources \
  --name ml-gpu-vm

# Get managed identity
MANAGED_IDENTITY=$(az vm show \
  --resource-group ml-resources \
  --name ml-gpu-vm \
  --query identity.principalId \
  --output tsv)

# Grant permissions
az role assignment create \
  --assignee $MANAGED_IDENTITY \
  --role "Storage Blob Data Contributor" \
  --scope /subscriptions/SUBSCRIPTION_ID/resourceGroups/ml-resources
```

## Part 4: Azure Machine Learning

### Create Azure ML Workspace

```bash
# Create workspace
az ml workspace create \
  --name ml-workspace \
  --resource-group ml-resources \
  --location eastus

# Get workspace details
az ml workspace show \
  --name ml-workspace \
  --resource-group ml-resources
```

### Azure ML Training with Python

```python
# azure_ml_training.py
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Connect to workspace
ws = Workspace.from_config()

# Create compute cluster
compute_name = 'gpu-cluster'
try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
    print(f'Found existing compute target: {compute_name}')
except ComputeTargetException:
    print(f'Creating compute target: {compute_name}')

    compute_config = AmlCompute.provisioning_configuration(
        vm_size='Standard_NC6s_v3',
        max_nodes=4,
        idle_seconds_before_scaledown=300
    )

    compute_target = ComputeTarget.create(ws, compute_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# Define environment
env = Environment.from_conda_specification(
    'pytorch-env',
    'environment.yml'
)

# Create training configuration
config = ScriptRunConfig(
    source_directory='./src',
    script='train.py',
    compute_target=compute_target,
    environment=env,
    arguments=[
        '--epochs', 10,
        '--batch-size', 32,
        '--learning-rate', 0.001
    ]
)

# Submit experiment
experiment = Experiment(ws, 'mnist-training')
run = experiment.submit(config)
run.wait_for_completion(show_output=True)

# Register model
model = run.register_model(
    model_name='mnist-classifier',
    model_path='outputs/model.pt',
    tags={'framework': 'pytorch', 'task': 'classification'}
)

print(f"Model registered: {model.name}, version: {model.version}")
```

### Deploy Model

```python
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig, Model

# Create inference configuration
inference_config = InferenceConfig(
    entry_script='score.py',
    environment=env
)

# Create deployment configuration
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=4,
    auth_enabled=True,
    enable_app_insights=True
)

# Deploy model
service = Model.deploy(
    workspace=ws,
    name='mnist-service',
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)

service.wait_for_deployment(show_output=True)

# Get scoring URI
print(f"Scoring URI: {service.scoring_uri}")

# Test endpoint
import requests
import json

headers = {'Content-Type': 'application/json'}
if service.auth_enabled:
    headers['Authorization'] = f'Bearer {service.get_keys()[0]}'

data = {'data': [[1, 2, 3, 4]]}
response = requests.post(service.scoring_uri, json=data, headers=headers)
print(response.json())
```

## Part 5: Azure Functions for Inference

```python
# Azure Function (function_app.py)
import azure.functions as func
import torch
import json
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp()

# Load model at startup
model = None
def load_model():
    global model
    if model is None:
        # Download model from blob storage
        blob_service = BlobServiceClient.from_connection_string(
            os.environ['STORAGE_CONNECTION_STRING']
        )
        blob_client = blob_service.get_blob_client('models', 'model.pth')

        with open('/tmp/model.pth', 'wb') as f:
            f.write(blob_client.download_blob().readall())

        model = torch.load('/tmp/model.pth')
        model.eval()

@app.route(route="predict", methods=['POST'])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP triggered inference function"""
    load_model()

    try:
        data = req.get_json()
        input_data = torch.tensor(data['input'])

        with torch.no_grad():
            output = model(input_data)

        prediction = output.argmax(dim=1).tolist()

        return func.HttpResponse(
            json.dumps({'prediction': prediction}),
            mimetype='application/json'
        )
    except Exception as e:
        return func.HttpResponse(
            f"Error: {str(e)}",
            status_code=500
        )
```

## Part 6: Cost Management

```bash
# View cost analysis
az consumption usage list \
  --start-date 2024-01-01 \
  --end-date 2024-01-31

# Create budget
az consumption budget create \
  --resource-group ml-resources \
  --budget-name monthly-ml-budget \
  --amount 1000 \
  --time-grain Monthly \
  --time-period start-date=2024-01-01 \
  --notification enabled=true threshold=80 contact-emails=admin@example.com
```

## Solutions

See `solutions/` directory for complete implementations:
- `azure_vm_manager.py` - Azure VM management automation
- `blob_operations.py` - Complete blob storage operations
- `azure_ml_pipeline.py` - End-to-end Azure ML pipeline
- `cost_monitor.py` - Azure cost tracking
- `infrastructure.tf` - Terraform configuration for Azure

## Exercises

1. Launch a GPU Azure VM and train a model
2. Create a storage account and implement blob operations
3. Configure managed identities for secure access
4. Run an Azure ML training job
5. Deploy inference function to Azure Functions
6. Set up cost monitoring and budgets

## Next Steps

- **Exercise 04**: Multi-Cloud ML Deployment - Deploy across AWS, GCP, and Azure
- **Exercise 05**: Cost Optimization & Management - Optimize costs across clouds

---

*Estimated completion time: 8-10 hours*
