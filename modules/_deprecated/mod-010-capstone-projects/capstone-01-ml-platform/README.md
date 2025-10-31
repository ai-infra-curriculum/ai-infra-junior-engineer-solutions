# Capstone 01: End-to-End ML Platform

Build a complete, self-service ML platform that allows data scientists to train, deploy, and monitor models without infrastructure expertise.

## Project Overview

**Duration**: 40-50 hours
**Difficulty**: Advanced
**Type**: Greenfield platform development

### Business Context

You're building an ML platform for a mid-sized company where data scientists currently struggle with:
- Manual model deployment taking 2+ weeks
- No standardized way to track experiments
- Inconsistent feature engineering across teams
- Limited model monitoring in production
- Complex infrastructure requiring DevOps support for every deployment

### Success Metrics

After implementation, the platform should enable:
- **Self-service deployment**: Data scientists deploy models in <2 hours without DevOps
- **Experiment tracking**: 100% of experiments tracked and reproducible
- **Feature reuse**: 50%+ reduction in duplicate feature engineering
- **Model monitoring**: Automated drift detection within 1 hour of occurrence
- **Reliability**: 99.5% uptime for model serving infrastructure

## Learning Objectives

By completing this project, you will:
- Design and implement a production ML platform architecture
- Integrate multiple MLOps tools into a cohesive system
- Build self-service workflows for data scientists
- Implement automated model lifecycle management
- Create comprehensive monitoring and alerting systems
- Develop platform documentation and runbooks
- Demonstrate job-ready ML infrastructure engineering skills

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ML Platform Architecture                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Data Scientists UI Layer                    │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │  │
│  │  │ MLflow   │  │ Airflow  │  │ Grafana  │  │ Jupyter  │     │  │
│  │  │ UI       │  │ UI       │  │ Dashboard│  │ Hub      │     │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                ↓                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                       Platform Services                        │  │
│  │                                                                 │  │
│  │  ┌──────────────────┐        ┌──────────────────┐            │  │
│  │  │  Feature Store   │        │  Model Registry  │            │  │
│  │  │     (Feast)      │        │    (MLflow)      │            │  │
│  │  └──────────────────┘        └──────────────────┘            │  │
│  │                                                                 │  │
│  │  ┌──────────────────┐        ┌──────────────────┐            │  │
│  │  │   Orchestration  │        │  Model Serving   │            │  │
│  │  │    (Airflow)     │        │   (BentoML)      │            │  │
│  │  └──────────────────┘        └──────────────────┘            │  │
│  │                                                                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                ↓                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  Infrastructure Layer (K8s)                    │  │
│  │                                                                 │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │  │
│  │  │Training │  │ Serving │  │Pipeline │  │Storage  │          │  │
│  │  │ Pods    │  │  Pods   │  │  Pods   │  │(MinIO)  │          │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │  │
│  │                                                                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                ↓                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              Observability & Monitoring                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │  │
│  │  │ Prometheus   │  │  Evidently   │  │     ELK      │        │  │
│  │  │  (Metrics)   │  │  (ML Mon.)   │  │   (Logs)     │        │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Feature Store (Feast)
- **Purpose**: Centralized feature management and serving
- **Features**:
  - Online feature serving (<10ms latency)
  - Offline feature retrieval for training
  - Point-in-time correct joins
  - Feature versioning and lineage

#### 2. Experiment Tracking & Model Registry (MLflow)
- **Purpose**: Track experiments and manage model lifecycle
- **Features**:
  - Experiment logging (parameters, metrics, artifacts)
  - Model versioning and registry
  - Model stage management (Staging → Production)
  - A/B testing support

#### 3. Pipeline Orchestration (Apache Airflow)
- **Purpose**: Automate ML workflows
- **Features**:
  - Scheduled training pipelines
  - Data validation workflows
  - Model deployment automation
  - Continuous training triggers

#### 4. Model Serving (BentoML)
- **Purpose**: Deploy and serve ML models
- **Features**:
  - RESTful API endpoints
  - Auto-scaling based on load
  - Multi-model serving
  - Request batching

#### 5. Model Monitoring (Evidently)
- **Purpose**: Monitor model performance and data drift
- **Features**:
  - Data drift detection
  - Prediction drift monitoring
  - Performance metrics tracking
  - Automated alerting

#### 6. Observability Stack
- **Prometheus + Grafana**: System and application metrics
- **ELK Stack**: Centralized logging
- **Alertmanager**: Alert routing and notifications

## Prerequisites

### Knowledge Requirements
- Strong Python programming (3.8+)
- Docker and containerization
- Kubernetes fundamentals
- Basic understanding of:
  - Machine learning concepts
  - CI/CD pipelines
  - SQL databases
  - RESTful APIs

### Infrastructure Requirements
- Kubernetes cluster (minikube/kind for local, or cloud provider)
- Minimum resources:
  - 16GB RAM
  - 8 CPU cores
  - 100GB storage
- Tools installed:
  - `kubectl`
  - `helm`
  - `docker`
  - `terraform` (for cloud deployment)

### Accounts Needed
- Cloud provider account (AWS/GCP/Azure) - optional for production deployment
- GitHub account
- Docker Hub account

## Project Phases

### Phase 1: Infrastructure Setup (8-10 hours)

#### 1.1 Kubernetes Cluster Setup

**Local Development (Minikube)**:
```bash
# Install minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start cluster with sufficient resources
minikube start --cpus=4 --memory=8192 --disk-size=50g

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server
```

**Production (AWS EKS)**:
```hcl
# terraform/eks-cluster.tf
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "ml-platform"
  cluster_version = "1.27"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    general = {
      desired_size = 2
      min_size     = 2
      max_size     = 10

      instance_types = ["t3.large"]
      capacity_type  = "ON_DEMAND"
    }

    gpu = {
      desired_size = 0
      min_size     = 0
      max_size     = 5

      instance_types = ["g4dn.xlarge"]
      capacity_type  = "SPOT"

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}
```

#### 1.2 Storage Setup

**MinIO for Object Storage**:
```yaml
# kubernetes/minio/deployment.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: minio-pvc
  namespace: ml-platform
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio
  namespace: ml-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      containers:
      - name: minio
        image: minio/minio:latest
        args:
        - server
        - /data
        - --console-address
        - ":9001"
        env:
        - name: MINIO_ROOT_USER
          valueFrom:
            secretKeyRef:
              name: minio-secret
              key: root-user
        - name: MINIO_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: minio-secret
              key: root-password
        ports:
        - containerPort: 9000
          name: api
        - containerPort: 9001
          name: console
        volumeMounts:
        - name: storage
          mountPath: /data
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: minio-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: minio
  namespace: ml-platform
spec:
  ports:
  - port: 9000
    name: api
  - port: 9001
    name: console
  selector:
    app: minio
```

**PostgreSQL for Metadata**:
```yaml
# kubernetes/postgres/deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: ml-platform
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:14
        env:
        - name: POSTGRES_DB
          value: mlplatform
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
          name: postgres
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
```

#### 1.3 Namespace and RBAC Setup

```yaml
# kubernetes/namespaces.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-platform
---
apiVersion: v1
kind: Namespace
metadata:
  name: ml-training
---
apiVersion: v1
kind: Namespace
metadata:
  name: ml-serving
---
# kubernetes/rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ml-platform-sa
  namespace: ml-platform
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ml-platform-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "secrets", "configmaps"]
  verbs: ["get", "list", "watch", "create", "update", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "statefulsets"]
  verbs: ["get", "list", "watch", "create", "update", "delete"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["get", "list", "watch", "create", "update", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: ml-platform-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: ml-platform-role
subjects:
- kind: ServiceAccount
  name: ml-platform-sa
  namespace: ml-platform
```

### Phase 2: Core Platform Components (12-15 hours)

#### 2.1 MLflow Deployment

**Helm Chart Installation**:
```bash
# Add Bitnami repo
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Create values file
cat > mlflow-values.yaml <<EOF
tracking:
  enabled: true

postgresql:
  enabled: false

externalDatabase:
  host: postgres.ml-platform.svc.cluster.local
  port: 5432
  database: mlflow
  user: mlflow
  password: <password>

s3:
  enabled: true
  endpoint: http://minio.ml-platform.svc.cluster.local:9000
  accessKeyID: <access-key>
  secretAccessKey: <secret-key>
  bucket: mlflow

service:
  type: ClusterIP
  port: 5000

ingress:
  enabled: true
  hostname: mlflow.ml-platform.local

resources:
  limits:
    cpu: 2
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi
EOF

# Install MLflow
helm install mlflow bitnami/mlflow \
  --namespace ml-platform \
  --values mlflow-values.yaml
```

**Custom MLflow Server (Alternative)**:
```python
# src/mlflow-server/server.py
import os
from mlflow.server import app

def create_mlflow_server():
    """Create MLflow tracking server with custom configuration"""

    # Backend store (PostgreSQL)
    backend_store_uri = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    # Artifact store (MinIO/S3)
    artifact_root = f"s3://{os.getenv('S3_BUCKET')}/mlflow/artifacts"

    os.environ['MLFLOW_BACKEND_STORE_URI'] = backend_store_uri
    os.environ['MLFLOW_DEFAULT_ARTIFACT_ROOT'] = artifact_root
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('S3_ENDPOINT')
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')

    return app

if __name__ == '__main__':
    app = create_mlflow_server()
    app.run(host='0.0.0.0', port=5000)
```

```dockerfile
# src/mlflow-server/Dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN pip install mlflow psycopg2-binary boto3

COPY server.py .

EXPOSE 5000

CMD ["python", "server.py"]
```

#### 2.2 Feast Feature Store

**Feature Repository Structure**:
```python
# feature_repo/feature_definitions.py
from datetime import timedelta
from feast import Entity, Feature, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64, String

# Define entity
user = Entity(
    name="user_id",
    value_type=ValueType.STRING,
    description="User ID"
)

# Define data source
user_features_source = FileSource(
    path="/data/user_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define feature view
user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=90),
    schema=[
        Feature(name="total_transactions", dtype=Int64),
        Feature(name="avg_transaction_amount", dtype=Float32),
        Feature(name="days_since_last_transaction", dtype=Int64),
        Feature(name="user_tenure_days", dtype=Int64),
        Feature(name="is_premium_user", dtype=Int64),
    ],
    online=True,
    source=user_features_source,
    tags={"team": "ml-platform", "version": "v1"},
)

# Transaction-level features
transaction_features_source = FileSource(
    path="/data/transaction_features.parquet",
    timestamp_field="event_timestamp",
)

transaction = Entity(
    name="transaction_id",
    value_type=ValueType.STRING,
    description="Transaction ID"
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=30),
    schema=[
        Feature(name="amount", dtype=Float32),
        Feature(name="merchant_category", dtype=String),
        Feature(name="is_international", dtype=Int64),
        Feature(name="hour_of_day", dtype=Int64),
        Feature(name="day_of_week", dtype=Int64),
    ],
    online=True,
    source=transaction_features_source,
)
```

**Feast Deployment**:
```yaml
# kubernetes/feast/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: feast-online-serving
  namespace: ml-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: feast-serving
  template:
    metadata:
      labels:
        app: feast-serving
    spec:
      containers:
      - name: feast-serving
        image: feastdev/feature-server:latest
        env:
        - name: FEAST_REDIS_HOST
          value: redis.ml-platform.svc.cluster.local
        - name: FEAST_REDIS_PORT
          value: "6379"
        ports:
        - containerPort: 6566
          name: http
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 6566
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 6566
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: feast-serving
  namespace: ml-platform
spec:
  selector:
    app: feast-serving
  ports:
  - port: 6566
    targetPort: 6566
  type: ClusterIP
```

**Redis for Online Store**:
```yaml
# kubernetes/redis/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: ml-platform
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            cpu: 250m
            memory: 512Mi
          limits:
            cpu: 1
            memory: 2Gi
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: ml-platform
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

#### 2.3 Apache Airflow

**Helm Installation**:
```bash
# Add Apache Airflow repo
helm repo add apache-airflow https://airflow.apache.org
helm repo update

# Create Airflow values
cat > airflow-values.yaml <<EOF
executor: KubernetesExecutor

postgresql:
  enabled: false

data:
  metadataConnection:
    user: airflow
    pass: <password>
    host: postgres.ml-platform.svc.cluster.local
    port: 5432
    db: airflow

webserver:
  replicas: 2
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2
      memory: 4Gi

scheduler:
  replicas: 2
  resources:
    requests:
      cpu: 500m
      memory: 1Gi

workers:
  persistence:
    enabled: true
    size: 10Gi

ingress:
  enabled: true
  web:
    host: airflow.ml-platform.local

dags:
  gitSync:
    enabled: true
    repo: https://github.com/your-org/ml-platform-dags.git
    branch: main
    subPath: dags
    wait: 60

env:
  - name: MLFLOW_TRACKING_URI
    value: http://mlflow.ml-platform.svc.cluster.local:5000
  - name: FEAST_SERVING_URL
    value: feast-serving.ml-platform.svc.cluster.local:6566
EOF

# Install Airflow
helm install airflow apache-airflow/airflow \
  --namespace ml-platform \
  --values airflow-values.yaml
```

### Phase 3: Model Serving Infrastructure (8-10 hours)

#### 3.1 BentoML Model Serving

**BentoML Service Definition**:
```python
# src/model-serving/service.py
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray, JSON
from pydantic import BaseModel
from typing import List, Dict

class PredictionRequest(BaseModel):
    """Request schema for predictions"""
    user_id: str
    transaction_amount: float
    merchant_category: str
    is_international: bool

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    prediction: int
    probability: float
    model_version: str
    features_used: Dict[str, float]

# Load model from MLflow
mlflow_model_uri = "models:/fraud_detection/production"
fraud_model = bentoml.mlflow.import_model(
    "fraud_detection",
    model_uri=mlflow_model_uri
)

# Create BentoML service
svc = bentoml.Service("fraud_detection_service")

@svc.api(
    input=JSON(pydantic_model=PredictionRequest),
    output=JSON(pydantic_model=PredictionResponse)
)
async def predict(input_data: PredictionRequest) -> PredictionResponse:
    """Make fraud predictions with feature retrieval"""
    from feast import FeatureStore

    # Initialize Feast client
    store = FeatureStore(repo_path=".")

    # Retrieve online features
    features = store.get_online_features(
        features=[
            "user_features:total_transactions",
            "user_features:avg_transaction_amount",
            "user_features:days_since_last_transaction",
        ],
        entity_rows=[{"user_id": input_data.user_id}]
    ).to_dict()

    # Combine with request features
    feature_vector = np.array([[
        features["total_transactions"][0],
        features["avg_transaction_amount"][0],
        input_data.transaction_amount,
        1 if input_data.is_international else 0,
    ]])

    # Get prediction
    runner = bentoml.mlflow.get(fraud_model).to_runner()
    prediction = await runner.predict.async_run(feature_vector)
    probability = await runner.predict_proba.async_run(feature_vector)

    return PredictionResponse(
        prediction=int(prediction[0]),
        probability=float(probability[0][1]),
        model_version=fraud_model.tag.version,
        features_used={
            "total_transactions": features["total_transactions"][0],
            "avg_transaction_amount": features["avg_transaction_amount"][0],
            "transaction_amount": input_data.transaction_amount,
        }
    )

@svc.api(input=JSON(), output=JSON())
def health() -> Dict:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": fraud_model.tag.name,
        "version": fraud_model.tag.version
    }
```

**BentoML Deployment**:
```yaml
# kubernetes/bentoml/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection-service
  namespace: ml-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detection
      version: v1
  template:
    metadata:
      labels:
        app: fraud-detection
        version: v1
    spec:
      containers:
      - name: bentoml
        image: your-registry/fraud-detection:v1.0.0
        ports:
        - containerPort: 3000
          name: http
        env:
        - name: MLFLOW_TRACKING_URI
          value: http://mlflow.ml-platform.svc.cluster.local:5000
        - name: FEAST_SERVING_URL
          value: feast-serving.ml-platform.svc.cluster.local:6566
        resources:
          requests:
            cpu: 1
            memory: 2Gi
          limits:
            cpu: 4
            memory: 8Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-detection
  namespace: ml-serving
spec:
  selector:
    app: fraud-detection
  ports:
  - port: 80
    targetPort: 3000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-detection-hpa
  namespace: ml-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-detection-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
```

### Phase 4: Automated ML Pipelines (10-12 hours)

#### 4.1 Training Pipeline DAG

```python
# dags/model_training_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from datetime import datetime, timedelta
import mlflow
from feast import FeatureStore

default_args = {
    'owner': 'ml-platform',
    'depends_on_past': False,
    'email': ['ml-alerts@company.com'],
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'fraud_detection_training',
    default_args=default_args,
    description='Train fraud detection model',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['training', 'fraud-detection']
)

def extract_training_data(**context):
    """Extract data for training"""
    from sqlalchemy import create_engine
    import pandas as pd

    engine = create_engine('postgresql://user:pass@postgres/mldb')

    # Extract last 90 days of data
    query = """
    SELECT
        user_id,
        transaction_id,
        amount,
        merchant_category,
        is_international,
        is_fraud,
        event_timestamp
    FROM transactions
    WHERE event_timestamp >= NOW() - INTERVAL '90 days'
    """

    df = pd.read_sql(query, engine)

    # Save to shared storage
    data_path = '/mnt/data/training/raw_data.parquet'
    df.to_parquet(data_path)

    context['task_instance'].xcom_push(key='data_path', value=data_path)
    context['task_instance'].xcom_push(key='data_size', value=len(df))

    print(f"Extracted {len(df)} rows to {data_path}")

def generate_features(**context):
    """Generate features using Feast"""
    import pandas as pd
    from feast import FeatureStore

    data_path = context['task_instance'].xcom_pull(key='data_path')
    df = pd.read_parquet(data_path)

    # Initialize Feast
    store = FeatureStore(repo_path="/feast/feature_repo")

    # Get historical features
    entity_df = df[['user_id', 'transaction_id', 'event_timestamp']]

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "user_features:total_transactions",
            "user_features:avg_transaction_amount",
            "user_features:days_since_last_transaction",
            "transaction_features:amount",
            "transaction_features:merchant_category",
            "transaction_features:is_international",
        ],
    ).to_df()

    # Merge with labels
    final_df = training_df.merge(
        df[['transaction_id', 'is_fraud']],
        on='transaction_id'
    )

    # Save feature set
    features_path = '/mnt/data/training/features.parquet'
    final_df.to_parquet(features_path)

    context['task_instance'].xcom_push(key='features_path', value=features_path)
    print(f"Generated {len(final_df.columns)} features for {len(final_df)} samples")

def validate_features(**context):
    """Validate feature quality"""
    import pandas as pd
    import great_expectations as ge

    features_path = context['task_instance'].xcom_pull(key='features_path')
    df = pd.read_parquet(features_path)

    # Create Great Expectations dataset
    ge_df = ge.from_pandas(df)

    # Define expectations
    expectations = [
        ge_df.expect_column_values_to_not_be_null('user_id'),
        ge_df.expect_column_values_to_be_between('amount', min_value=0, max_value=100000),
        ge_df.expect_column_proportion_of_unique_values_to_be_between('is_fraud', min_value=0.01, max_value=0.99),
    ]

    # Validate
    results = ge_df.validate()

    if not results['success']:
        raise ValueError(f"Feature validation failed: {results}")

    print("✓ Feature validation passed")

# Training task runs in Kubernetes pod with GPU
train_model_task = KubernetesPodOperator(
    task_id='train_model',
    name='fraud-detection-training',
    namespace='ml-training',
    image='your-registry/fraud-detection-trainer:latest',
    cmds=['python', 'train.py'],
    arguments=[
        '--features-path', '{{ task_instance.xcom_pull(key="features_path") }}',
        '--mlflow-tracking-uri', 'http://mlflow.ml-platform.svc.cluster.local:5000',
        '--experiment-name', 'fraud_detection',
    ],
    env_vars={
        'MLFLOW_TRACKING_URI': 'http://mlflow.ml-platform.svc.cluster.local:5000',
    },
    resources={
        'request_memory': '8Gi',
        'request_cpu': '4',
        'limit_memory': '16Gi',
        'limit_cpu': '8',
        'limit_nvidia_gpu': '1',
    },
    node_selector={'node-type': 'gpu'},
    tolerations=[{
        'key': 'nvidia.com/gpu',
        'operator': 'Equal',
        'value': 'true',
        'effect': 'NoSchedule'
    }],
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag
)

def evaluate_and_register_model(**context):
    """Evaluate model and register if performance meets threshold"""
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    # Get latest run from experiment
    experiment = client.get_experiment_by_name('fraud_detection')
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=['start_time DESC'],
        max_results=1
    )

    if not runs:
        raise ValueError("No training run found")

    run = runs[0]
    run_id = run.info.run_id

    # Check performance metrics
    accuracy = run.data.metrics.get('accuracy', 0)
    precision = run.data.metrics.get('precision', 0)
    recall = run.data.metrics.get('recall', 0)

    # Validation thresholds
    if accuracy < 0.90 or precision < 0.85 or recall < 0.85:
        raise ValueError(
            f"Model performance below threshold: "
            f"accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}"
        )

    # Register model
    model_uri = f"runs:/{run_id}/model"
    model_name = "fraud_detection"

    model_version = mlflow.register_model(model_uri, model_name)

    # Transition to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )

    context['task_instance'].xcom_push(key='model_version', value=model_version.version)
    context['task_instance'].xcom_push(key='run_id', value=run_id)

    print(f"✓ Model registered: {model_name} v{model_version.version}")
    print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

def deploy_to_staging(**context):
    """Deploy model to staging environment"""
    from kubernetes import client, config
    import base64

    model_version = context['task_instance'].xcom_pull(key='model_version')

    # Load k8s config
    config.load_incluster_config()
    apps_v1 = client.AppsV1Api()

    # Update deployment with new model version
    deployment = apps_v1.read_namespaced_deployment(
        name='fraud-detection-staging',
        namespace='ml-serving'
    )

    deployment.spec.template.spec.containers[0].env.append(
        client.V1EnvVar(
            name='MODEL_VERSION',
            value=str(model_version)
        )
    )

    # Update deployment
    apps_v1.patch_namespaced_deployment(
        name='fraud-detection-staging',
        namespace='ml-serving',
        body=deployment
    )

    print(f"✓ Deployed model version {model_version} to staging")

# Define task dependencies
extract_task = PythonOperator(
    task_id='extract_training_data',
    python_callable=extract_training_data,
    dag=dag
)

generate_features_task = PythonOperator(
    task_id='generate_features',
    python_callable=generate_features,
    dag=dag
)

validate_features_task = PythonOperator(
    task_id='validate_features',
    python_callable=validate_features,
    dag=dag
)

evaluate_task = PythonOperator(
    task_id='evaluate_and_register',
    python_callable=evaluate_and_register_model,
    dag=dag
)

deploy_staging_task = PythonOperator(
    task_id='deploy_to_staging',
    python_callable=deploy_to_staging,
    dag=dag
)

# Pipeline flow
extract_task >> generate_features_task >> validate_features_task >> train_model_task
train_model_task >> evaluate_task >> deploy_staging_task
```

#### 4.2 Training Script

```python
# src/training/train.py
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import numpy as np

def train_fraud_detection_model(features_path: str, experiment_name: str):
    """Train fraud detection model with MLflow tracking"""

    # Load features
    print(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)

    # Prepare data
    feature_cols = [col for col in df.columns if col not in ['is_fraud', 'user_id', 'transaction_id', 'event_timestamp']]
    X = df[feature_cols]
    y = df['is_fraud']

    # Train-validation-test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Fraud rate: {y.mean():.2%}")

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': 42
        }
        mlflow.log_params(params)
        mlflow.log_param('training_samples', len(X_train))
        mlflow.log_param('feature_count', len(feature_cols))
        mlflow.log_param('fraud_rate', y.mean())

        # Train model
        print("Training Random Forest model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]

        val_metrics = {
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_precision': precision_score(y_val, y_val_pred),
            'val_recall': recall_score(y_val, y_val_pred),
            'val_f1': f1_score(y_val, y_val_pred),
            'val_roc_auc': roc_auc_score(y_val, y_val_proba)
        }

        print("\nValidation Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
            mlflow.log_metric(metric, value)

        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        }

        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
            mlflow.log_metric(metric, value)

        # Log confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        mlflow.log_dict({
            'confusion_matrix': {
                'true_negatives': int(cm[0, 0]),
                'false_positives': int(cm[0, 1]),
                'false_negatives': int(cm[1, 0]),
                'true_positives': int(cm[1, 1])
            }
        }, 'confusion_matrix.json')

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        mlflow.log_dict(
            feature_importance.to_dict('records'),
            'feature_importance.json'
        )

        print("\nTop 10 Features:")
        print(feature_importance.head(10))

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=None,  # Will register later in pipeline
            signature=mlflow.models.infer_signature(X_train, y_train)
        )

        # Log feature names
        mlflow.log_dict({'features': feature_cols}, 'feature_names.json')

        print(f"\n✓ Training complete. Run ID: {mlflow.active_run().info.run_id}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-path', required=True)
    parser.add_argument('--mlflow-tracking-uri', required=True)
    parser.add_argument('--experiment-name', default='fraud_detection')

    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    train_fraud_detection_model(
        features_path=args.features_path,
        experiment_name=args.experiment_name
    )
```

### Phase 5: Monitoring & Observability (8-10 hours)

#### 5.1 Prometheus & Grafana Setup

**Prometheus Installation**:
```bash
# Add Prometheus community Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack (includes Grafana)
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
  --set grafana.adminPassword=admin123
```

**Custom Metrics for Model Serving**:
```python
# src/model-serving/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Prediction metrics
prediction_counter = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_name', 'model_version', 'prediction_class']
)

prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Model prediction latency',
    ['model_name', 'model_version'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

feature_retrieval_latency = Histogram(
    'feature_retrieval_latency_seconds',
    'Feature retrieval latency',
    ['feature_store']
)

model_errors = Counter(
    'model_errors_total',
    'Total number of model errors',
    ['model_name', 'error_type']
)

active_model_version = Gauge(
    'active_model_version',
    'Currently active model version',
    ['model_name']
)

# Data drift metrics
data_drift_score = Gauge(
    'data_drift_score',
    'Data drift score for feature',
    ['feature_name', 'drift_metric']
)

def track_prediction(model_name: str, model_version: str, prediction: int, start_time: float):
    """Track prediction metrics"""
    latency = time.time() - start_time

    prediction_counter.labels(
        model_name=model_name,
        model_version=model_version,
        prediction_class=str(prediction)
    ).inc()

    prediction_latency.labels(
        model_name=model_name,
        model_version=model_version
    ).observe(latency)
```

**ServiceMonitor for Prometheus**:
```yaml
# kubernetes/monitoring/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: fraud-detection-metrics
  namespace: ml-serving
spec:
  selector:
    matchLabels:
      app: fraud-detection
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

#### 5.2 Evidently AI for ML Monitoring

```python
# src/monitoring/drift_detection.py
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import *
import pandas as pd
from prometheus_client import Gauge
import schedule
import time

# Prometheus metrics for drift
data_drift_gauge = Gauge('data_drift_detected', 'Data drift detected', ['feature'])
target_drift_gauge = Gauge('target_drift_score', 'Target drift score')

class ModelMonitor:
    """Monitor model performance and data drift"""

    def __init__(self, reference_data: pd.DataFrame, feature_columns: list, target_column: str):
        self.reference_data = reference_data
        self.feature_columns = feature_columns
        self.target_column = target_column

        self.column_mapping = ColumnMapping(
            target=target_column,
            prediction='prediction',
            numerical_features=[col for col in feature_columns if reference_data[col].dtype in ['int64', 'float64']],
            categorical_features=[col for col in feature_columns if reference_data[col].dtype == 'object']
        )

    def check_drift(self, current_data: pd.DataFrame) -> dict:
        """Check for data drift"""

        # Create drift report
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset()
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        # Extract results
        results = report.as_dict()

        # Update Prometheus metrics
        for feature in self.feature_columns:
            drift_detected = results['metrics'][0]['result']['drift_by_columns'].get(feature, {}).get('drift_detected', False)
            data_drift_gauge.labels(feature=feature).set(1 if drift_detected else 0)

        target_drift_score = results['metrics'][1]['result'].get('drift_score', 0)
        target_drift_gauge.set(target_drift_score)

        return results

    def run_tests(self, current_data: pd.DataFrame) -> bool:
        """Run automated tests"""

        tests = TestSuite(tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedRows(),
            TestNumberOfDuplicatedColumns(),
            TestColumnsType(),
            TestNumberOfDriftedColumns(),
        ])

        tests.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        results = tests.as_dict()
        all_passed = all(test['status'] == 'SUCCESS' for test in results['tests'])

        return all_passed

def monitor_production_data():
    """Periodic monitoring of production data"""
    from sqlalchemy import create_engine

    engine = create_engine('postgresql://user:pass@postgres/mldb')

    # Load reference data (last month)
    reference_query = """
    SELECT * FROM model_predictions
    WHERE timestamp >= NOW() - INTERVAL '30 days'
    AND timestamp < NOW() - INTERVAL '7 days'
    LIMIT 10000
    """
    reference_data = pd.read_sql(reference_query, engine)

    # Load current data (last 24 hours)
    current_query = """
    SELECT * FROM model_predictions
    WHERE timestamp >= NOW() - INTERVAL '24 hours'
    """
    current_data = pd.read_sql(current_query, engine)

    if len(current_data) < 100:
        print("Not enough current data for drift detection")
        return

    # Initialize monitor
    monitor = ModelMonitor(
        reference_data=reference_data,
        feature_columns=['total_transactions', 'avg_transaction_amount', 'amount'],
        target_column='is_fraud'
    )

    # Check drift
    drift_results = monitor.check_drift(current_data)

    if drift_results['metrics'][0]['result']['dataset_drift']:
        print("⚠️  Data drift detected!")
        # Trigger alert or retraining

    # Run tests
    tests_passed = monitor.run_tests(current_data)

    if not tests_passed:
        print("⚠️  Data quality tests failed!")

if __name__ == '__main__':
    # Run monitoring every hour
    schedule.every(1).hours.do(monitor_production_data)

    while True:
        schedule.run_pending()
        time.sleep(60)
```

This capstone project implementation guide is quite extensive. Let me continue with the remaining phases in the next file to keep things organized.

Would you like me to continue with:
- Phase 6: Platform API & SDK
- Phase 7: Testing & Validation
- Phase 8: Documentation & Deployment

Or would you prefer to move on to Capstone 02 and 03?