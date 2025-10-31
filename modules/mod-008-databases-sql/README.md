# Module 008: Cloud Platforms for ML (AWS/GCP/Azure)

Learn to deploy and manage ML infrastructure across major cloud platforms including AWS, GCP, and Azure.

## Learning Objectives

By the end of this module, you will be able to:

- Deploy ML workloads on AWS, GCP, and Azure
- Use cloud-native ML services (SageMaker, Vertex AI, Azure ML)
- Manage compute resources (EC2, GCE, Azure VMs, spot instances)
- Configure cloud storage for ML data (S3, GCS, Azure Blob)
- Set up cloud networking and security for ML systems
- Implement IAM and access controls
- Use managed Kubernetes services (EKS, GKE, AKS)
- Optimize cloud costs for ML workloads
- Implement multi-cloud and hybrid cloud strategies
- Monitor and observe cloud ML infrastructure

## Prerequisites

- Module 005: Docker & Containerization
- Module 006: Kubernetes & Orchestration
- Module 007: CI/CD for ML
- Basic understanding of cloud computing concepts
- AWS, GCP, or Azure account (free tier available)

## Cloud Platform Overview

### Major Cloud Providers

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Cloud Platform Comparison                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐             │
│  │   AWS    │        │   GCP    │        │  Azure   │             │
│  │          │        │          │        │          │             │
│  │ Market   │        │ ML/AI    │        │Enterprise│             │
│  │ Leader   │        │ Strong   │        │ Focus    │             │
│  └──────────┘        └──────────┘        └──────────┘             │
│      35%                 10%                 23%                    │
│   Market Share        Market Share        Market Share             │
│                                                                      │
│  Compute:             Compute:             Compute:                │
│  - EC2                - GCE                - VMs                   │
│  - Lambda             - Cloud Run          - Functions             │
│  - Fargate            - Cloud Run Jobs     - Container Instances   │
│                                                                      │
│  Storage:             Storage:             Storage:                │
│  - S3                 - GCS                - Blob Storage          │
│  - EBS                - Persistent Disk    - Managed Disks         │
│  - EFS                - Filestore          - Files                 │
│                                                                      │
│  ML Services:         ML Services:         ML Services:            │
│  - SageMaker          - Vertex AI          - Azure ML              │
│  - Bedrock            - AutoML             - Cognitive Services    │
│  - Rekognition        - Vision AI          - Computer Vision       │
│                                                                      │
│  Kubernetes:          Kubernetes:          Kubernetes:             │
│  - EKS                - GKE                - AKS                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Cloud Service Models

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Cloud Service Models for ML                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  IaaS (Infrastructure as a Service)                                 │
│  ├─ EC2 / GCE / Azure VMs                                          │
│  ├─ Storage (S3, GCS, Blob)                                        │
│  ├─ Networking (VPC, Cloud Network)                                │
│  └─ You manage: OS, runtime, ML frameworks, applications           │
│                                                                      │
│  PaaS (Platform as a Service)                                       │
│  ├─ App Engine / Cloud Run / App Service                           │
│  ├─ Managed Kubernetes (EKS, GKE, AKS)                            │
│  ├─ Managed Databases                                               │
│  └─ You manage: Applications, data                                  │
│                                                                      │
│  MLaaS (ML as a Service)                                            │
│  ├─ SageMaker / Vertex AI / Azure ML                              │
│  ├─ Managed training and inference                                  │
│  ├─ AutoML capabilities                                             │
│  └─ You manage: Models, data, configurations                        │
│                                                                      │
│  FaaS (Function as a Service)                                       │
│  ├─ Lambda / Cloud Functions / Azure Functions                     │
│  ├─ Event-driven inference                                          │
│  └─ You manage: Function code only                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## AWS for ML Infrastructure

### Core AWS Services for ML

#### Compute Services

**EC2 (Elastic Compute Cloud)**
- Virtual machines with GPU support
- Instance types for ML: P4, P3, G5, G4dn, Inf1, Inf2
- Spot instances for cost savings (up to 90% off)
- Auto Scaling Groups for dynamic scaling

```bash
# Launch GPU instance for ML training
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type p3.2xlarge \
  --key-name my-key-pair \
  --security-group-ids sg-903004f8 \
  --subnet-id subnet-6e7f829e \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ml-training}]'

# Use spot instances
aws ec2 request-spot-instances \
  --spot-price "1.00" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://specification.json
```

**Lambda**
- Serverless inference for lightweight models
- 15-minute timeout, 10GB memory limit
- Cost-effective for infrequent predictions

```python
# Lambda function for model inference
import json
import boto3
import pickle

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Load model from S3
    model_obj = s3.get_object(Bucket='my-models', Key='model.pkl')
    model = pickle.loads(model_obj['Body'].read())

    # Get input data
    input_data = json.loads(event['body'])

    # Predict
    prediction = model.predict([input_data['features']])

    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': prediction.tolist()})
    }
```

**ECS/Fargate**
- Container orchestration without managing servers
- Good for inference APIs
- Integrates with ALB for load balancing

#### Storage Services

**S3 (Simple Storage Service)**
- Object storage for datasets, models, artifacts
- Versioning and lifecycle policies
- S3 Select for querying data without downloading

```python
# S3 operations for ML
import boto3

s3 = boto3.client('s3')

# Upload training data
s3.upload_file('train.csv', 'my-ml-bucket', 'data/train.csv')

# Download model
s3.download_file('my-ml-bucket', 'models/model.pt', '/tmp/model.pt')

# List model versions
response = s3.list_object_versions(Bucket='my-ml-bucket', Prefix='models/')

# Create lifecycle policy
lifecycle_policy = {
    'Rules': [{
        'Id': 'archive-old-models',
        'Status': 'Enabled',
        'Transitions': [{
            'Days': 30,
            'StorageClass': 'GLACIER'
        }],
        'NoncurrentVersionExpiration': {'NoncurrentDays': 90}
    }]
}
s3.put_bucket_lifecycle_configuration(
    Bucket='my-ml-bucket',
    LifecycleConfiguration=lifecycle_policy
)
```

**EBS (Elastic Block Store)**
- Block storage for EC2 instances
- High-performance storage for training jobs
- Snapshots for backups

**EFS (Elastic File System)**
- Shared file system for distributed training
- NFS protocol
- Scales automatically

#### ML-Specific Services

**SageMaker**
- End-to-end ML platform
- Managed training, tuning, and deployment
- Built-in algorithms and frameworks
- SageMaker Studio for development

```python
# SageMaker training job
import sagemaker
from sagemaker.pytorch import PyTorch

role = 'arn:aws:iam::123456789012:role/SageMakerRole'
session = sagemaker.Session()

# Define training job
estimator = PyTorch(
    entry_point='train.py',
    role=role,
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    framework_version='2.0',
    py_version='py310',
    hyperparameters={
        'epochs': 10,
        'batch-size': 32
    }
)

# Start training
estimator.fit({'training': 's3://my-bucket/data/train'})

# Deploy model
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

# Make predictions
result = predictor.predict(data)
```

**SageMaker Features**:
- **Training**: Managed training with automatic resource scaling
- **Hyperparameter Tuning**: Automated hyperparameter optimization
- **Model Registry**: Version control for models
- **Endpoints**: Managed inference with auto-scaling
- **Pipelines**: ML workflow orchestration
- **Feature Store**: Centralized feature repository
- **Model Monitor**: Detect data drift and model degradation

#### Networking & Security

**VPC (Virtual Private Cloud)**
- Isolated network for ML infrastructure
- Subnets, security groups, NACLs
- VPC endpoints for private access to S3, SageMaker

```bash
# Create VPC for ML infrastructure
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=ml-vpc}]'

# Create private subnet for training
aws ec2 create-subnet --vpc-id vpc-12345678 --cidr-block 10.0.1.0/24 --availability-zone us-east-1a

# Create S3 VPC endpoint
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-12345678 \
  --service-name com.amazonaws.us-east-1.s3 \
  --route-table-ids rtb-12345678
```

**IAM (Identity and Access Management)**
- Fine-grained access control
- Roles for EC2, Lambda, SageMaker
- Policies for S3, ECR access

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::my-ml-bucket/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob"
      ],
      "Resource": "*"
    }
  ]
}
```

#### Managed Kubernetes

**EKS (Elastic Kubernetes Service)**
- Managed Kubernetes control plane
- Integrates with AWS services (IAM, VPC, ELB)
- Supports GPU nodes and spot instances

```bash
# Create EKS cluster
eksctl create cluster \
  --name ml-cluster \
  --region us-east-1 \
  --nodegroup-name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --name ml-cluster --region us-east-1

# Deploy workload
kubectl apply -f ml-deployment.yaml
```

## GCP for ML Infrastructure

### Core GCP Services for ML

#### Compute Services

**Compute Engine (GCE)**
- Virtual machines with GPU/TPU support
- GPU types: V100, A100, T4, P100, P4
- TPU Pods for large-scale training
- Preemptible instances for cost savings

```bash
# Create GPU instance
gcloud compute instances create ml-gpu-instance \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --metadata="install-nvidia-driver=True"

# Create TPU VM
gcloud compute tpus tpu-vm create ml-tpu \
  --zone=us-central1-a \
  --accelerator-type=v3-8 \
  --version=tpu-vm-tf-2.13.0
```

**Cloud Run**
- Serverless container platform
- Automatic scaling from 0 to N
- Good for inference APIs

```yaml
# Cloud Run service for inference
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ml-inference
spec:
  template:
    spec:
      containers:
      - image: gcr.io/my-project/ml-api:latest
        resources:
          limits:
            memory: 4Gi
            cpu: 2
        env:
        - name: MODEL_PATH
          value: gs://my-models/model.pt
```

#### Storage Services

**Cloud Storage (GCS)**
- Object storage for ML data
- Nearline/Coldline for archival
- Integration with ML services

```python
# GCS operations
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('my-ml-bucket')

# Upload dataset
blob = bucket.blob('data/train.csv')
blob.upload_from_filename('train.csv')

# Download model
model_blob = bucket.blob('models/model.pt')
model_blob.download_to_filename('/tmp/model.pt')

# Set lifecycle policy
bucket.add_lifecycle_delete_rule(age=90)
bucket.patch()
```

**Persistent Disk**
- Block storage for Compute Engine
- SSD and HDD options
- Snapshots for backups

**Filestore**
- Managed NFS file system
- Shared storage for distributed training

#### ML-Specific Services

**Vertex AI**
- Unified ML platform (successor to AI Platform)
- Managed training and prediction
- AutoML capabilities
- Model monitoring and explainability

```python
# Vertex AI training
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

# Create custom training job
job = aiplatform.CustomTrainingJob(
    display_name='pytorch-training',
    container_uri='gcr.io/my-project/trainer:latest',
    requirements=['torch==2.0.0', 'torchvision']
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
    test_fraction_split=0.1
)

# Deploy model
endpoint = model.deploy(
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=10,
    traffic_split={'0': 100}
)

# Predict
prediction = endpoint.predict(instances=[data])
```

**Vertex AI Features**:
- **Workbench**: Managed Jupyter notebooks
- **Training**: Custom and AutoML training
- **Pipelines**: Kubeflow Pipelines integration
- **Feature Store**: Centralized feature management
- **Model Registry**: Versioned model storage
- **Endpoints**: Managed inference with autoscaling
- **Monitoring**: Model and data monitoring
- **Explainable AI**: Model interpretability

**TPU (Tensor Processing Unit)**
- Google's custom AI accelerators
- Optimized for TensorFlow and JAX
- TPU Pods for distributed training
- Cost-effective for large models

```python
# TPU training with TensorFlow
import tensorflow as tf

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(train_dataset, epochs=10)
```

#### Networking & Security

**VPC (Virtual Private Cloud)**
- Private network for GCP resources
- Shared VPC for multi-project setups
- Private Google Access for accessing GCS without internet

**IAM (Identity and Access Management)**
- Fine-grained permissions
- Service accounts for GCE, Cloud Run
- Workload Identity for GKE

```json
{
  "bindings": [
    {
      "role": "roles/storage.objectViewer",
      "members": [
        "serviceAccount:ml-training@my-project.iam.gserviceaccount.com"
      ]
    },
    {
      "role": "roles/aiplatform.user",
      "members": [
        "serviceAccount:ml-training@my-project.iam.gserviceaccount.com"
      ]
    }
  ]
}
```

#### Managed Kubernetes

**GKE (Google Kubernetes Engine)**
- Managed Kubernetes with autopilot mode
- Node pools with GPUs/TPUs
- Workload Identity for secure access
- GKE Autopilot for serverless K8s

```bash
# Create GKE cluster with GPU
gcloud container clusters create ml-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-8 \
  --accelerator type=nvidia-tesla-v100,count=1 \
  --num-nodes 2 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --addons GcePersistentDiskCsiDriver \
  --workload-pool=my-project.svc.id.goog

# Install NVIDIA drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Create TPU node pool
gcloud container node-pools create tpu-pool \
  --cluster ml-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-8 \
  --num-nodes 2 \
  --enable-autoscaling \
  --tpu-accelerator-type v3-8
```

## Azure for ML Infrastructure

### Core Azure Services for ML

#### Compute Services

**Azure Virtual Machines**
- GPU VMs: NC, ND, NV series
- Spot instances for cost savings
- VM Scale Sets for auto-scaling

```bash
# Create GPU VM
az vm create \
  --resource-group ml-resources \
  --name ml-gpu-vm \
  --image UbuntuLTS \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# Create scale set
az vmss create \
  --resource-group ml-resources \
  --name ml-scaleset \
  --image UbuntuLTS \
  --vm-sku Standard_NC6s_v3 \
  --instance-count 2 \
  --admin-username azureuser \
  --generate-ssh-keys
```

**Azure Functions**
- Serverless compute for inference
- Event-driven execution
- Multiple language support

**Azure Container Instances**
- Fast container deployment
- Billed by second
- Good for batch inference

#### Storage Services

**Azure Blob Storage**
- Object storage for ML data
- Hot, Cool, Archive tiers
- Integration with ML services

```python
# Azure Blob operations
from azure.storage.blob import BlobServiceClient

connection_string = "DefaultEndpointsProtocol=https;..."
blob_service = BlobServiceClient.from_connection_string(connection_string)

# Upload data
blob_client = blob_service.get_blob_client(container='ml-data', blob='train.csv')
with open('train.csv', 'rb') as data:
    blob_client.upload_blob(data)

# Download model
model_blob = blob_service.get_blob_client(container='models', blob='model.pt')
with open('/tmp/model.pt', 'wb') as f:
    f.write(model_blob.download_blob().readall())
```

**Azure Files**
- SMB file shares
- Shared storage for distributed training

**Azure Managed Disks**
- Block storage for VMs
- SSD and HDD options

#### ML-Specific Services

**Azure Machine Learning**
- End-to-end ML platform
- Managed compute clusters
- Automated ML (AutoML)
- MLOps capabilities

```python
# Azure ML training
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute

# Connect to workspace
ws = Workspace.from_config()

# Create compute cluster
compute_config = AmlCompute.provisioning_configuration(
    vm_size='Standard_NC6s_v3',
    max_nodes=4,
    idle_seconds_before_scaledown=300
)
compute_target = ComputeTarget.create(ws, 'gpu-cluster', compute_config)

# Define environment
env = Environment.from_conda_specification('pytorch-env', 'environment.yml')

# Create training configuration
config = ScriptRunConfig(
    source_directory='./src',
    script='train.py',
    compute_target=compute_target,
    environment=env,
    arguments=['--epochs', 10, '--batch-size', 32]
)

# Submit experiment
experiment = Experiment(ws, 'mnist-training')
run = experiment.submit(config)
run.wait_for_completion(show_output=True)

# Register model
model = run.register_model(
    model_name='mnist-classifier',
    model_path='outputs/model.pt'
)

# Deploy model
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(
    entry_script='score.py',
    environment=env
)

deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=4,
    auth_enabled=True
)

service = Model.deploy(
    workspace=ws,
    name='mnist-service',
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)
service.wait_for_deployment(show_output=True)
```

**Azure ML Features**:
- **Designer**: Visual ML pipeline designer
- **Automated ML**: AutoML for model selection
- **Compute**: Managed compute clusters
- **Datasets**: Version-controlled datasets
- **Experiments**: Track training runs
- **Models**: Model registry
- **Endpoints**: Managed inference
- **Pipelines**: ML workflow orchestration
- **Monitoring**: Model and data monitoring

#### Networking & Security

**Azure Virtual Network (VNet)**
- Private network for Azure resources
- Network Security Groups (NSGs)
- Private endpoints for services

**Azure Active Directory (AAD)**
- Identity management
- Role-Based Access Control (RBAC)
- Managed identities for services

**Azure Key Vault**
- Secrets management
- Encryption keys
- Certificate storage

#### Managed Kubernetes

**AKS (Azure Kubernetes Service)**
- Managed Kubernetes service
- GPU node pools
- Azure CNI networking
- Azure AD integration

```bash
# Create AKS cluster with GPU
az aks create \
  --resource-group ml-resources \
  --name ml-cluster \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 5 \
  --generate-ssh-keys

# Add GPU node pool
az aks nodepool add \
  --resource-group ml-resources \
  --cluster-name ml-cluster \
  --name gpupool \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10

# Get credentials
az aks get-credentials --resource-group ml-resources --name ml-cluster
```

## Multi-Cloud Architecture

### Multi-Cloud Strategies

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Cloud Architecture                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Strategy 1: Vendor-Agnostic                                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Kubernetes (EKS/GKE/AKS)                                   │    │
│  │  ├─ Common manifests                                         │    │
│  │  ├─ Helm charts                                              │    │
│  │  └─ Portable across clouds                                   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Strategy 2: Best-of-Breed                                          │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  AWS: Cost-effective compute (spot instances)               │    │
│  │  GCP: TPUs for large model training                         │    │
│  │  Azure: Enterprise integration                               │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Strategy 3: High Availability                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Primary: AWS us-east-1                                     │    │
│  │  Secondary: GCP us-central1                                 │    │
│  │  Failover: Cross-cloud replication                          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Strategy 4: Data Residency                                         │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  EU data: GCP europe-west1                                  │    │
│  │  US data: AWS us-east-1                                     │    │
│  │  Asia data: Azure southeastasia                             │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Multi-Cloud Tools

**Terraform**
- Infrastructure as Code for all clouds
- Provider support for AWS, GCP, Azure
- State management and drift detection

```hcl
# Multi-cloud Terraform example
# AWS Provider
provider "aws" {
  region = "us-east-1"
}

# GCP Provider
provider "google" {
  project = "my-project"
  region  = "us-central1"
}

# Azure Provider
provider "azurerm" {
  features {}
}

# AWS Kubernetes cluster
resource "aws_eks_cluster" "main" {
  name     = "ml-cluster-aws"
  role_arn = aws_iam_role.cluster.arn
  vpc_config {
    subnet_ids = aws_subnet.private[*].id
  }
}

# GCP Kubernetes cluster
resource "google_container_cluster" "main" {
  name     = "ml-cluster-gcp"
  location = "us-central1"
  initial_node_count = 3
}

# Azure Kubernetes cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = "ml-cluster-azure"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "ml-cluster"
  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_D2_v2"
  }
}
```

**Crossplane**
- Kubernetes-native infrastructure management
- Control plane for multi-cloud resources
- Composition and abstraction layers

**Anthos (Google)**
- Multi-cloud and hybrid application platform
- Runs on GCP, AWS, Azure, on-premises
- Centralized management

**Azure Arc**
- Extends Azure management to other clouds
- Kubernetes management across clouds
- Unified governance and compliance

## Cost Optimization

### Cost Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ML Compute Cost Comparison                        │
│                    (Approximate monthly costs)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  GPU Instance Comparison:                                           │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ AWS p3.2xlarge (V100)       $3.06/hr  ~$2,200/month      │      │
│  │ GCP n1-standard-8 + V100    $2.48/hr  ~$1,800/month      │      │
│  │ Azure NC6s_v3 (V100)        $3.06/hr  ~$2,200/month      │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│  Spot/Preemptible Pricing (70-90% discount):                       │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ AWS Spot           $0.90/hr (~70% savings)               │      │
│  │ GCP Preemptible    $0.74/hr (~70% savings)               │      │
│  │ Azure Spot         $0.61/hr (~80% savings)               │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│  TPU Pricing (GCP):                                                 │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ TPU v3-8           $8.00/hr  ~$5,800/month               │      │
│  │ TPU v3-8 Preempt   $2.40/hr  ~$1,700/month (70% off)    │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
│  Storage Costs (per TB/month):                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ AWS S3 Standard             $23                           │      │
│  │ GCP Storage Standard        $20                           │      │
│  │ Azure Blob Hot              $18                           │      │
│  │                                                            │      │
│  │ AWS S3 Glacier              $4                            │      │
│  │ GCP Coldline                $4                            │      │
│  │ Azure Archive               $2                            │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Cost Optimization Strategies

1. **Use Spot/Preemptible Instances**
   - 70-90% cost savings
   - Good for training jobs that can handle interruptions
   - Implement checkpointing

2. **Right-Size Instances**
   - Monitor resource utilization
   - Scale down over-provisioned instances
   - Use smaller instances for inference

3. **Storage Lifecycle Policies**
   - Move old data to cold storage
   - Delete unnecessary artifacts
   - Use compression

4. **Reserved Instances/Committed Use**
   - 1-year or 3-year commitments
   - 30-70% savings for predictable workloads

5. **Auto-Scaling**
   - Scale down during off-hours
   - Use horizontal pod autoscaling
   - Scale to zero for dev/test environments

6. **Data Transfer Optimization**
   - Use same region for compute and storage
   - Minimize cross-region transfers
   - Use CDN for inference APIs

## Monitoring & Observability

### Cloud-Native Monitoring

**AWS CloudWatch**
- Metrics, logs, alarms
- Custom metrics for ML models
- Dashboards and insights

**GCP Cloud Monitoring (Stackdriver)**
- Metrics, logs, traces
- Uptime checks
- Alerting policies

**Azure Monitor**
- Application Insights
- Log Analytics
- Metrics and alerts

### Common Monitoring Stack

```yaml
# Prometheus + Grafana on Kubernetes
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
      - job_name: 'ml-api'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            regex: ml-api
            action: keep
```

## Best Practices

### Security Best Practices

1. **Use IAM Roles/Service Accounts**
   - Never hardcode credentials
   - Principle of least privilege
   - Rotate credentials regularly

2. **Encrypt Data**
   - At rest: S3/GCS/Blob encryption
   - In transit: TLS/SSL
   - Encryption keys in KMS

3. **Network Security**
   - Private subnets for compute
   - Security groups/firewall rules
   - VPC peering for multi-account

4. **Secrets Management**
   - AWS Secrets Manager
   - GCP Secret Manager
   - Azure Key Vault

### Reliability Best Practices

1. **Multi-AZ Deployments**
   - Spread across availability zones
   - Regional redundancy for critical services

2. **Health Checks**
   - Liveness and readiness probes
   - Circuit breakers for failures

3. **Backup and Disaster Recovery**
   - Regular snapshots
   - Cross-region replication
   - Test restore procedures

4. **Chaos Engineering**
   - Test failure scenarios
   - Use AWS Fault Injection Simulator
   - GCP Chaos Engineering tools

### Performance Best Practices

1. **Use Appropriate Instance Types**
   - GPU for training
   - CPU for inference (if appropriate)
   - Memory-optimized for large models

2. **Optimize Data Loading**
   - Use fast storage (SSD)
   - Prefetch and cache data
   - Parallel data loading

3. **Model Optimization**
   - Quantization
   - Pruning
   - Knowledge distillation

## Module Exercises

- **Exercise 01**: AWS Fundamentals for ML - EC2, S3, SageMaker basics
- **Exercise 02**: GCP ML Infrastructure - Compute Engine, GCS, Vertex AI
- **Exercise 03**: Azure ML Services - Azure VMs, Blob Storage, Azure ML
- **Exercise 04**: Multi-Cloud ML Deployment - Terraform, Kubernetes across clouds
- **Exercise 05**: Cost Optimization & Management - Monitoring, right-sizing, cost allocation

## Additional Resources

### Documentation
- [AWS ML Documentation](https://docs.aws.amazon.com/machine-learning/)
- [GCP AI/ML Documentation](https://cloud.google.com/products/ai)
- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)

### Training
- AWS Certified Machine Learning - Specialty
- Google Cloud Professional ML Engineer
- Microsoft Certified: Azure AI Engineer Associate

### Tools
- [Terraform](https://www.terraform.io/)
- [AWS CLI](https://aws.amazon.com/cli/)
- [gcloud CLI](https://cloud.google.com/sdk/gcloud)
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/)

## Summary

In this module, you've learned how to:
- Deploy ML workloads on AWS, GCP, and Azure
- Use cloud-native ML services
- Manage compute, storage, and networking
- Implement multi-cloud strategies
- Optimize costs for ML infrastructure
- Monitor and secure cloud ML systems

## Next Module

**Module 009: Advanced MLOps** - Production ML systems, A/B testing, feature stores, model monitoring

---

*Estimated completion time: 40-50 hours*
