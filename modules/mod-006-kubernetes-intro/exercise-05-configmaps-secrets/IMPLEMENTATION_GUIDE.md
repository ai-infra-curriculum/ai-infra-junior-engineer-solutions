# Implementation Guide: ConfigMaps & Secrets for ML Infrastructure

## Table of Contents

1. [Introduction](#introduction)
2. [ConfigMaps for Application Configuration](#configmaps-for-application-configuration)
3. [Secrets for Sensitive Data](#secrets-for-sensitive-data)
4. [Environment Variables vs Volume Mounts](#environment-variables-vs-volume-mounts)
5. [External Secrets Management](#external-secrets-management)
6. [Configuration Hot-Reload Patterns](#configuration-hot-reload-patterns)
7. [Multi-Environment Configuration](#multi-environment-configuration)
8. [Production ML Configuration Management](#production-ml-configuration-management)
9. [Security Best Practices](#security-best-practices)
10. [Troubleshooting Guide](#troubleshooting-guide)

## Introduction

Managing configuration and secrets in Kubernetes is critical for ML infrastructure. This guide covers practical patterns for managing model configurations, API keys, database credentials, feature store settings, and other sensitive data required for production ML systems.

### Why ConfigMaps and Secrets Matter for ML

**ML-specific challenges:**
- Model hyperparameters change frequently
- Multiple environments (dev, staging, prod) with different configurations
- Sensitive credentials for data sources, model registries, and APIs
- Large configuration files for complex ML pipelines
- Need for hot-reloading to update models without downtime

**Key principles:**
1. **Separation of concerns**: Code, configuration, and secrets are managed separately
2. **Environment isolation**: Each environment has its own configuration
3. **Security first**: Sensitive data is protected with proper RBAC and encryption
4. **Auditability**: All configuration changes are tracked
5. **Flexibility**: Support for both static and dynamic configuration

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                       │
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │  ConfigMaps  │      │   Secrets    │                     │
│  │              │      │              │                     │
│  │ - Model cfg  │      │ - API keys   │                     │
│  │ - Feature    │      │ - DB creds   │                     │
│  │ - Inference  │      │ - ML tokens  │                     │
│  └──────┬───────┘      └──────┬───────┘                     │
│         │                     │                              │
│         └──────────┬──────────┘                              │
│                    │                                         │
│         ┌──────────▼───────────┐                             │
│         │    ML Application    │                             │
│         │                      │                             │
│         │ - Training jobs      │                             │
│         │ - Inference servers  │                             │
│         │ - Feature pipelines  │                             │
│         └──────────────────────┘                             │
│                                                               │
│  ┌──────────────────────────────────────────────┐            │
│  │  External Secrets Management (Optional)      │            │
│  │  - HashiCorp Vault                           │            │
│  │  - AWS Secrets Manager                       │            │
│  │  - External Secrets Operator                 │            │
│  └──────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## ConfigMaps for Application Configuration

ConfigMaps store non-sensitive configuration data. For ML workloads, this includes model hyperparameters, feature definitions, inference configurations, and pipeline settings.

### Creating ConfigMaps

#### Method 1: From Literal Values (Simple Configuration)

Best for: Feature flags, simple settings, environment markers

```bash
# ML model configuration
kubectl create configmap ml-model-config \
  --from-literal=MODEL_NAME="bert-base-uncased" \
  --from-literal=MODEL_VERSION="1.0.0" \
  --from-literal=BATCH_SIZE="32" \
  --from-literal=MAX_LENGTH="512" \
  --from-literal=NUM_LABELS="3" \
  --from-literal=LEARNING_RATE="2e-5" \
  -n ml-production

# Inference configuration
kubectl create configmap inference-config \
  --from-literal=WORKERS="4" \
  --from-literal=TIMEOUT="30" \
  --from-literal=MAX_BATCH_SIZE="64" \
  --from-literal=GPU_ENABLED="true" \
  --from-literal=CACHE_TTL="300" \
  -n ml-production
```

#### Method 2: From Files (Complex Configuration)

Best for: Model hyperparameters, feature definitions, pipeline configs

```bash
# Create model configuration file
cat > model-config.yaml << 'EOF'
model:
  name: "bert-base-uncased"
  task: "text-classification"
  num_labels: 3

training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 3
  warmup_steps: 500
  weight_decay: 0.01

optimizer:
  type: "AdamW"
  betas: [0.9, 0.999]
  eps: 1e-8

scheduler:
  type: "linear"
  num_warmup_steps: 500
EOF

# Create ConfigMap from file
kubectl create configmap model-hyperparameters \
  --from-file=config.yaml=model-config.yaml \
  -n ml-production
```

#### Method 3: Declarative YAML (Version Control)

Best for: Production environments, GitOps workflows

```yaml
# ml-configs.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-training-config
  namespace: ml-production
  labels:
    app: ml-training
    environment: production
    version: v1
data:
  # Model configuration
  MODEL_NAME: "bert-base-uncased"
  MODEL_VERSION: "1.0.0"
  CHECKPOINT_DIR: "/models/checkpoints"

  # Training parameters
  BATCH_SIZE: "32"
  LEARNING_RATE: "2e-5"
  NUM_EPOCHS: "3"
  MAX_STEPS: "10000"

  # Data configuration
  TRAIN_DATA_PATH: "/data/train"
  VAL_DATA_PATH: "/data/val"
  MAX_SAMPLES: "100000"

  # Logging
  LOG_LEVEL: "INFO"
  WANDB_PROJECT: "ml-training"

  # Feature configuration (JSON)
  feature_config.json: |
    {
      "features": [
        {
          "name": "text",
          "type": "string",
          "preprocessing": "tokenize"
        },
        {
          "name": "label",
          "type": "categorical",
          "num_classes": 3
        }
      ],
      "feature_store": {
        "type": "feast",
        "project": "ml_features"
      }
    }

  # Inference configuration (YAML)
  inference.yaml: |
    server:
      host: "0.0.0.0"
      port: 8080
      workers: 4

    model:
      path: "/models/production"
      device: "cuda"
      precision: "fp16"

    preprocessing:
      max_length: 512
      truncation: true
      padding: true

    batching:
      enabled: true
      max_batch_size: 64
      timeout_ms: 100

    monitoring:
      enabled: true
      log_predictions: true
      metrics_port: 9090
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: feature-store-config
  namespace: ml-production
  labels:
    app: feature-store
    environment: production
data:
  # Feast feature store configuration
  feature_store.yaml: |
    project: ml_features
    registry: /registry/data/registry.db
    provider: local
    online_store:
      type: redis
      connection_string: "redis:6379"
    offline_store:
      type: file

  # Feature definitions
  features.py: |
    from feast import Entity, Feature, FeatureView, ValueType
    from datetime import timedelta

    user = Entity(name="user_id", value_type=ValueType.INT64)

    user_features = FeatureView(
        name="user_features",
        entities=["user_id"],
        ttl=timedelta(days=1),
        features=[
            Feature(name="age", dtype=ValueType.INT64),
            Feature(name="country", dtype=ValueType.STRING),
            Feature(name="subscription_type", dtype=ValueType.STRING),
        ],
    )
```

Apply the configuration:

```bash
kubectl apply -f ml-configs.yaml
```

### ML-Specific ConfigMap Patterns

#### Pattern 1: Model Registry Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-registry-config
  namespace: ml-production
data:
  MLFLOW_TRACKING_URI: "http://mlflow-server:5000"
  MLFLOW_EXPERIMENT_NAME: "production-models"
  MODEL_REGISTRY_URI: "s3://ml-models/registry"
  DEFAULT_ARTIFACT_ROOT: "s3://ml-models/artifacts"

  # Model serving configuration
  model_serving.json: |
    {
      "models": [
        {
          "name": "text-classifier",
          "version": "1.0.0",
          "path": "s3://ml-models/text-classifier/v1",
          "runtime": "torchserve",
          "replicas": 3,
          "resources": {
            "requests": {"memory": "2Gi", "cpu": "1"},
            "limits": {"memory": "4Gi", "cpu": "2"}
          }
        },
        {
          "name": "recommendation-engine",
          "version": "2.1.0",
          "path": "s3://ml-models/recommender/v2.1",
          "runtime": "triton",
          "replicas": 5,
          "resources": {
            "requests": {"nvidia.com/gpu": "1"},
            "limits": {"nvidia.com/gpu": "1"}
          }
        }
      ]
    }
```

#### Pattern 2: Data Pipeline Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: data-pipeline-config
  namespace: ml-production
data:
  # Airflow/Kubeflow configuration
  pipeline_config.yaml: |
    pipelines:
      - name: feature_engineering
        schedule: "0 */6 * * *"  # Every 6 hours
        sources:
          - type: postgresql
            connection: "postgres-readonly"
            query: "SELECT * FROM events WHERE timestamp > :last_run"
          - type: s3
            bucket: "raw-data"
            prefix: "events/"
        transforms:
          - type: feature_extraction
            config: /config/features.yaml
          - type: validation
            schema: /config/schema.json
        outputs:
          - type: feature_store
            destination: feast
          - type: s3
            bucket: "processed-data"

      - name: model_training
        schedule: "0 2 * * *"  # Daily at 2 AM
        triggers:
          - type: data_drift
            threshold: 0.15
          - type: performance_degradation
            metric: "accuracy"
            threshold: 0.85
        resources:
          gpu: 2
          memory: "32Gi"
```

#### Pattern 3: Multi-Model Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: multi-model-config
  namespace: ml-production
data:
  models.yaml: |
    models:
      text_classification:
        primary:
          name: "bert-classifier-v2"
          endpoint: "http://text-classifier:8080/v2/predict"
          timeout: 1000
          retry_attempts: 2
        fallback:
          name: "bert-classifier-v1"
          endpoint: "http://text-classifier-v1:8080/predict"
          timeout: 2000

      sentiment_analysis:
        primary:
          name: "sentiment-roberta-v1"
          endpoint: "http://sentiment:8080/predict"
          timeout: 500
        ab_test:
          enabled: true
          variant_a: "sentiment-roberta-v1"
          variant_b: "sentiment-distilbert-v2"
          traffic_split: 0.8  # 80% to variant A

      embeddings:
        primary:
          name: "sentence-transformers"
          endpoint: "http://embeddings:8080/encode"
          batch_size: 128
          cache_enabled: true
          cache_ttl: 3600
```

### Using ConfigMaps in Pods

#### Method 1: Environment Variables (Individual Keys)

Best for: Simple configuration values, feature flags

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-job
  namespace: ml-production
spec:
  containers:
  - name: trainer
    image: ml-training:v1.0.0
    env:
    # Individual ConfigMap keys
    - name: MODEL_NAME
      valueFrom:
        configMapKeyRef:
          name: ml-training-config
          key: MODEL_NAME
    - name: BATCH_SIZE
      valueFrom:
        configMapKeyRef:
          name: ml-training-config
          key: BATCH_SIZE
    - name: LEARNING_RATE
      valueFrom:
        configMapKeyRef:
          name: ml-training-config
          key: LEARNING_RATE
```

#### Method 2: Environment Variables (All Keys)

Best for: Importing entire configuration sets

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-server
  namespace: ml-production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
    spec:
      containers:
      - name: server
        image: ml-inference:v1.0.0
        envFrom:
        # Import all keys from ConfigMap as environment variables
        - configMapRef:
            name: inference-config
        # Import with prefix
        - prefix: MODEL_
          configMapRef:
            name: ml-model-config
```

#### Method 3: Volume Mounts (Recommended for Files)

Best for: Configuration files, model configs, feature definitions

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-pipeline
  namespace: ml-production
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-pipeline
  template:
    metadata:
      labels:
        app: ml-pipeline
    spec:
      containers:
      - name: pipeline
        image: ml-pipeline:v1.0.0
        volumeMounts:
        # Mount entire ConfigMap as directory
        - name: pipeline-config
          mountPath: /config
          readOnly: true
        # Mount specific file with subPath
        - name: feature-config
          mountPath: /app/config/features.yaml
          subPath: feature_config.json
          readOnly: true
        command:
        - python
        - train.py
        - --config=/config/pipeline_config.yaml
        - --features=/app/config/features.yaml
      volumes:
      # ConfigMap as volume
      - name: pipeline-config
        configMap:
          name: data-pipeline-config
          items:
          - key: pipeline_config.yaml
            path: pipeline_config.yaml
      - name: feature-config
        configMap:
          name: ml-training-config
          items:
          - key: feature_config.json
            path: feature_config.json
```

#### Method 4: Projected Volumes (Combined Sources)

Best for: Combining multiple ConfigMaps and Secrets

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-worker
  namespace: ml-production
spec:
  containers:
  - name: worker
    image: ml-worker:v1.0.0
    volumeMounts:
    - name: combined-config
      mountPath: /config
      readOnly: true
  volumes:
  - name: combined-config
    projected:
      sources:
      # Model configuration
      - configMap:
          name: ml-model-config
          items:
          - key: config.yaml
            path: model/config.yaml
      # Feature store configuration
      - configMap:
          name: feature-store-config
          items:
          - key: feature_store.yaml
            path: features/store.yaml
      # Inference configuration
      - configMap:
          name: inference-config
          items:
          - key: inference.yaml
            path: inference/config.yaml
```

## Secrets for Sensitive Data

Secrets store sensitive information like database passwords, API keys, model registry credentials, and cloud service tokens.

### Creating Secrets

#### Method 1: From Literal Values

```bash
# Database credentials
kubectl create secret generic db-credentials \
  --from-literal=username=ml_user \
  --from-literal=password='P@ssw0rd!Complex#2024' \
  --from-literal=host=postgres.ml-db.svc.cluster.local \
  --from-literal=port=5432 \
  --from-literal=database=ml_production \
  -n ml-production

# API keys for ML services
kubectl create secret generic ml-api-keys \
  --from-literal=openai-api-key='sk-...' \
  --from-literal=huggingface-token='hf_...' \
  --from-literal=wandb-api-key='...' \
  --from-literal=mlflow-token='...' \
  -n ml-production

# Cloud credentials
kubectl create secret generic aws-credentials \
  --from-literal=aws_access_key_id='AKIA...' \
  --from-literal=aws_secret_access_key='...' \
  --from-literal=aws_default_region='us-east-1' \
  -n ml-production
```

#### Method 2: From Files

```bash
# Create credentials file
cat > db-creds.env << 'EOF'
DB_USERNAME=ml_user
DB_PASSWORD=P@ssw0rd!Complex#2024
DB_HOST=postgres.ml-db.svc.cluster.local
DB_PORT=5432
DB_DATABASE=ml_production
CONNECTION_STRING=postgresql://ml_user:P@ssw0rd!Complex#2024@postgres.ml-db.svc.cluster.local:5432/ml_production
EOF

# Create secret from file
kubectl create secret generic db-connection \
  --from-env-file=db-creds.env \
  -n ml-production

# Clean up file (important!)
shred -u db-creds.env

# TLS certificates for model serving
kubectl create secret tls model-serving-tls \
  --cert=/path/to/tls.crt \
  --key=/path/to/tls.key \
  -n ml-production

# SSH keys for Git access
kubectl create secret generic git-ssh-key \
  --from-file=ssh-privatekey=/path/to/id_rsa \
  --from-file=ssh-publickey=/path/to/id_rsa.pub \
  -n ml-production
```

#### Method 3: Declarative YAML with stringData

```yaml
# ml-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ml-service-credentials
  namespace: ml-production
  labels:
    app: ml-infrastructure
    environment: production
type: Opaque
stringData:
  # Database
  database-url: "postgresql://ml_user:P@ssw0rd@postgres:5432/ml_db"

  # Model registry
  mlflow-tracking-uri: "http://mlflow:5000"
  mlflow-username: "admin"
  mlflow-password: "mlflow-secret-password"

  # Feature store
  redis-password: "redis-secret-password"
  feast-registry-path: "s3://ml-features/registry"

  # Object storage
  s3-access-key: "AKIA..."
  s3-secret-key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

  # API keys
  openai-api-key: "sk-..."
  huggingface-token: "hf_..."

  # Monitoring
  prometheus-token: "prom-token-..."
  grafana-api-key: "eyJrIjoiT..."
---
apiVersion: v1
kind: Secret
metadata:
  name: model-serving-secrets
  namespace: ml-production
type: Opaque
stringData:
  # JWT for authentication
  jwt-secret-key: "your-256-bit-secret-key-here"

  # Model encryption key
  model-encryption-key: "AES256-encryption-key-here"

  # Webhook secrets
  github-webhook-secret: "webhook-secret-..."
  slack-webhook-url: "https://hooks.slack.com/services/..."
---
# Docker registry secret for private images
apiVersion: v1
kind: Secret
metadata:
  name: ml-registry-credentials
  namespace: ml-production
type: kubernetes.io/dockerconfigjson
stringData:
  .dockerconfigjson: |
    {
      "auths": {
        "registry.example.com": {
          "username": "ml-deployer",
          "password": "registry-password",
          "email": "ml-team@example.com",
          "auth": "bWwtZGVwbG95ZXI6cmVnaXN0cnktcGFzc3dvcmQ="
        }
      }
    }
```

Apply secrets:

```bash
# Apply from file
kubectl apply -f ml-secrets.yaml

# Verify (data will be base64 encoded)
kubectl get secret ml-service-credentials -n ml-production -o yaml
```

### Using Secrets in Pods

#### Pattern 1: Database Credentials

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-training-service
  namespace: ml-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: training
  template:
    metadata:
      labels:
        app: training
    spec:
      containers:
      - name: trainer
        image: ml-training:v1.0.0
        env:
        # Individual secret keys
        - name: DB_USERNAME
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: host
        # Or construct connection string
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-connection
              key: CONNECTION_STRING
```

#### Pattern 2: API Keys as Volume Mounts (Recommended)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
  namespace: ml-production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
    spec:
      containers:
      - name: server
        image: ml-inference:v1.0.0
        volumeMounts:
        # Mount secrets as files
        - name: api-keys
          mountPath: /secrets/api
          readOnly: true
        - name: cloud-creds
          mountPath: /secrets/cloud
          readOnly: true
        env:
        # Reference secret file paths
        - name: OPENAI_API_KEY_FILE
          value: /secrets/api/openai-api-key
        - name: AWS_SHARED_CREDENTIALS_FILE
          value: /secrets/cloud/credentials
      volumes:
      - name: api-keys
        secret:
          secretName: ml-api-keys
          defaultMode: 0400  # Read-only for owner
          items:
          - key: openai-api-key
            path: openai-api-key
          - key: huggingface-token
            path: huggingface-token
      - name: cloud-creds
        secret:
          secretName: aws-credentials
          defaultMode: 0400
```

#### Pattern 3: Model Registry Authentication

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training-job
  namespace: ml-production
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: training
        image: ml-training:v1.0.0
        envFrom:
        # Import all MLflow secrets
        - secretRef:
            name: ml-service-credentials
            optional: false
        volumeMounts:
        # Mount AWS credentials for S3 access
        - name: aws-config
          mountPath: /root/.aws
          readOnly: true
        command:
        - python
        - train.py
        - --experiment-name=production-model
        - --tracking-uri=$(mlflow-tracking-uri)
      volumes:
      - name: aws-config
        secret:
          secretName: aws-credentials
          items:
          - key: credentials
            path: credentials
          - key: config
            path: config
```

#### Pattern 4: Private Docker Registry

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: private-model-server
  namespace: ml-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      # Use secret to pull from private registry
      imagePullSecrets:
      - name: ml-registry-credentials
      containers:
      - name: server
        image: registry.example.com/ml/inference:v2.0.0
        ports:
        - containerPort: 8080
```

## Environment Variables vs Volume Mounts

Choosing between environment variables and volume mounts is critical for ML infrastructure.

### Comparison Matrix

| Feature | Environment Variables | Volume Mounts |
|---------|----------------------|---------------|
| **Update Behavior** | Static (requires pod restart) | Dynamic (~60s sync) |
| **Security** | Visible in pod spec and logs | More secure, file permissions |
| **Size Limit** | Limited (1 MB total) | 1 MB per ConfigMap/Secret |
| **Best For** | Simple values, flags | Config files, credentials |
| **Visibility** | kubectl describe pod shows them | Hidden from describe |
| **Hot-Reload** | No | Yes (with app support) |
| **Performance** | Faster (in memory) | Slight I/O overhead |

### Decision Guide

**Use Environment Variables when:**
- Simple key-value pairs (model name, version, flags)
- Configuration is read once at startup
- Need compatibility with 12-factor apps
- Configuration rarely changes
- Size < 100 variables

**Use Volume Mounts when:**
- Large configuration files (model configs, feature definitions)
- Sensitive data (passwords, keys, tokens)
- Need hot-reload capability
- Multiple related configuration files
- Complex JSON/YAML structures

### Hybrid Approach (Recommended for ML)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-hybrid-config
  namespace: ml-production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
    spec:
      containers:
      - name: app
        image: ml-service:v1.0.0

        # Environment variables for simple config
        env:
        - name: MODEL_NAME
          valueFrom:
            configMapKeyRef:
              name: ml-model-config
              key: MODEL_NAME
        - name: MODEL_VERSION
          valueFrom:
            configMapKeyRef:
              name: ml-model-config
              key: MODEL_VERSION
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: ml-model-config
              key: LOG_LEVEL

        # Secrets as environment variables (use sparingly)
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: ml-api-keys
              key: openai-api-key

        # Volume mounts for complex config files
        volumeMounts:
        - name: model-config
          mountPath: /config/model
          readOnly: true
        - name: feature-config
          mountPath: /config/features
          readOnly: true

        # Volume mounts for secrets (recommended)
        - name: credentials
          mountPath: /secrets
          readOnly: true

        # Reference config files and secrets in command
        command:
        - python
        - serve.py
        - --model-config=/config/model/config.yaml
        - --features=/config/features/features.yaml
        - --credentials=/secrets/credentials

      volumes:
      # ConfigMaps as volumes
      - name: model-config
        configMap:
          name: ml-model-config
          items:
          - key: config.yaml
            path: config.yaml
      - name: feature-config
        configMap:
          name: feature-store-config
          items:
          - key: features.yaml
            path: features.yaml

      # Secrets as volumes (recommended)
      - name: credentials
        secret:
          secretName: ml-service-credentials
          defaultMode: 0400
```

### Hot-Reload Pattern Implementation

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-hot-reload
  namespace: ml-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-hot-reload
  template:
    metadata:
      labels:
        app: ml-hot-reload
    spec:
      containers:
      # Main application container
      - name: app
        image: ml-app:v1.0.0
        volumeMounts:
        - name: config
          mountPath: /config
          readOnly: true
        command:
        - python
        - app.py
        - --config=/config/config.yaml

      # Sidecar: Config watcher for hot-reload
      - name: config-watcher
        image: busybox:1.36
        volumeMounts:
        - name: config
          mountPath: /config
          readOnly: true
        command:
        - /bin/sh
        - -c
        - |
          CONFIG_FILE="/config/config.yaml"
          LAST_HASH=""

          echo "Config watcher started"

          while true; do
            if [ -f "$CONFIG_FILE" ]; then
              CURRENT_HASH=$(md5sum "$CONFIG_FILE" | awk '{print $1}')

              if [ "$LAST_HASH" != "$CURRENT_HASH" ]; then
                echo "$(date): Config changed (hash: $CURRENT_HASH)"

                # Notify application (via HTTP, file, signal, etc.)
                wget -O- --post-data='{"action":"reload"}' \
                  http://localhost:8080/admin/reload || true

                LAST_HASH="$CURRENT_HASH"
              fi
            fi

            sleep 10
          done

      volumes:
      - name: config
        configMap:
          name: ml-model-config
```

## External Secrets Management

For production ML systems, use external secret management systems instead of native Kubernetes Secrets.

### Why External Secrets Management?

**Limitations of native Secrets:**
- Base64 encoded, not encrypted by default
- Stored in etcd (security risk)
- No built-in rotation
- No audit trail
- No centralized management
- Manual secret synchronization across clusters

**Benefits of external systems:**
- True encryption at rest and in transit
- Centralized secret management
- Automated rotation
- Detailed audit logs
- Fine-grained access control
- Dynamic secret generation
- Cross-cluster secret sharing

### Option 1: External Secrets Operator (Recommended)

External Secrets Operator (ESO) synchronizes secrets from external sources to Kubernetes Secrets.

#### Installation

```bash
# Add Helm repository
helm repo add external-secrets https://charts.external-secrets.io

# Install operator
helm install external-secrets \
  external-secrets/external-secrets \
  -n external-secrets-system \
  --create-namespace \
  --set installCRDs=true
```

#### Configure AWS Secrets Manager Backend

```yaml
# aws-secret-store.yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: ml-production
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        secretRef:
          accessKeyIDSecretRef:
            name: aws-credentials
            key: access-key-id
          secretAccessKeySecretRef:
            name: aws-credentials
            key: secret-access-key
---
# Fetch secret from AWS Secrets Manager
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: ml-database-credentials
  namespace: ml-production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: db-credentials
    creationPolicy: Owner
  data:
  - secretKey: username
    remoteRef:
      key: ml-production/database
      property: username
  - secretKey: password
    remoteRef:
      key: ml-production/database
      property: password
  - secretKey: host
    remoteRef:
      key: ml-production/database
      property: host
```

Apply configuration:

```bash
kubectl apply -f aws-secret-store.yaml

# Verify
kubectl get secretstore -n ml-production
kubectl get externalsecret -n ml-production
kubectl get secret db-credentials -n ml-production
```

#### Configure HashiCorp Vault Backend

```yaml
# vault-secret-store.yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: ml-production
spec:
  provider:
    vault:
      server: "https://vault.example.com:8200"
      path: "ml-secrets"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "ml-production"
          serviceAccountRef:
            name: external-secrets-sa
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: ml-api-keys
  namespace: ml-production
spec:
  refreshInterval: 15m
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: api-keys
    creationPolicy: Owner
  dataFrom:
  - extract:
      key: ml-production/api-keys
```

### Option 2: Sealed Secrets (GitOps-friendly)

Sealed Secrets encrypts secrets that can safely be stored in Git.

#### Installation

```bash
# Install Sealed Secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Install kubeseal CLI
wget https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/kubeseal-0.24.0-linux-amd64.tar.gz
tar -xvzf kubeseal-0.24.0-linux-amd64.tar.gz
sudo install -m 755 kubeseal /usr/local/bin/kubeseal
```

#### Create Sealed Secret

```bash
# Create regular secret (not applied)
kubectl create secret generic ml-api-keys \
  --from-literal=openai-key='sk-...' \
  --from-literal=hf-token='hf_...' \
  --dry-run=client -o yaml > ml-secret.yaml

# Encrypt with kubeseal
kubeseal --format=yaml \
  --cert=pub-cert.pem \
  < ml-secret.yaml > ml-sealed-secret.yaml

# Safe to commit to Git
git add ml-sealed-secret.yaml
git commit -m "Add ML API keys (sealed)"

# Apply sealed secret
kubectl apply -f ml-sealed-secret.yaml

# Controller automatically creates the secret
kubectl get secret ml-api-keys -n ml-production
```

Example sealed secret:

```yaml
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: ml-api-keys
  namespace: ml-production
spec:
  encryptedData:
    openai-key: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEq...
    hf-token: AgCFBqQDfxlVMZw+8Q6lKKBYdH5p3QbH...
  template:
    metadata:
      name: ml-api-keys
      namespace: ml-production
    type: Opaque
```

### Option 3: HashiCorp Vault Direct Integration

Use Vault Agent Injector to inject secrets directly into pods.

#### Installation

```bash
# Add Vault Helm repository
helm repo add hashicorp https://helm.releases.hashicorp.com

# Install Vault
helm install vault hashicorp/vault \
  --namespace vault \
  --create-namespace \
  --set "injector.enabled=true"
```

#### Configure Vault

```bash
# Enable Kubernetes auth
vault auth enable kubernetes

# Configure Kubernetes auth
vault write auth/kubernetes/config \
  kubernetes_host="https://$KUBERNETES_SERVICE_HOST:$KUBERNETES_SERVICE_PORT"

# Create policy
vault policy write ml-production - <<EOF
path "secret/data/ml-production/*" {
  capabilities = ["read"]
}
EOF

# Create role
vault write auth/kubernetes/role/ml-production \
  bound_service_account_names=ml-app \
  bound_service_account_namespaces=ml-production \
  policies=ml-production \
  ttl=24h
```

#### Use Vault in Pod

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-with-vault
  namespace: ml-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-vault
  template:
    metadata:
      labels:
        app: ml-vault
      annotations:
        # Enable Vault injection
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "ml-production"

        # Inject database credentials
        vault.hashicorp.com/agent-inject-secret-database: "secret/ml-production/database"
        vault.hashicorp.com/agent-inject-template-database: |
          {{- with secret "secret/ml-production/database" -}}
          export DB_USERNAME="{{ .Data.data.username }}"
          export DB_PASSWORD="{{ .Data.data.password }}"
          export DB_HOST="{{ .Data.data.host }}"
          {{- end }}

        # Inject API keys
        vault.hashicorp.com/agent-inject-secret-api-keys: "secret/ml-production/api-keys"
        vault.hashicorp.com/agent-inject-template-api-keys: |
          {{- with secret "secret/ml-production/api-keys" -}}
          export OPENAI_API_KEY="{{ .Data.data.openai_key }}"
          export HF_TOKEN="{{ .Data.data.hf_token }}"
          {{- end }}
    spec:
      serviceAccountName: ml-app
      containers:
      - name: app
        image: ml-app:v1.0.0
        command:
        - /bin/sh
        - -c
        - |
          source /vault/secrets/database
          source /vault/secrets/api-keys
          python app.py
```

## Configuration Hot-Reload Patterns

Hot-reload allows updating configuration without restarting pods, critical for ML services.

### Pattern 1: inotify-based Watcher

```python
# config_watcher.py
import os
import time
import hashlib
import signal
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloadHandler(FileSystemEventHandler):
    def __init__(self, config_path, callback):
        self.config_path = config_path
        self.callback = callback
        self.last_hash = self._get_file_hash()

    def _get_file_hash(self):
        """Calculate MD5 hash of config file"""
        if not os.path.exists(self.config_path):
            return None

        with open(self.config_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def on_modified(self, event):
        if event.src_path == self.config_path:
            current_hash = self._get_file_hash()

            if current_hash != self.last_hash:
                print(f"Config file changed: {self.config_path}")
                self.last_hash = current_hash

                # Reload configuration
                try:
                    self.callback(self.config_path)
                    print("Configuration reloaded successfully")
                except Exception as e:
                    print(f"Error reloading config: {e}")

# Usage in ML application
def reload_model_config(config_path):
    """Reload model configuration"""
    import yaml

    with open(config_path, 'r') as f:
        new_config = yaml.safe_load(f)

    # Update model parameters
    global model_config
    model_config = new_config

    print(f"Updated config: {model_config}")

if __name__ == "__main__":
    config_path = "/config/model-config.yaml"

    # Set up watcher
    event_handler = ConfigReloadHandler(config_path, reload_model_config)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(config_path), recursive=False)
    observer.start()

    print(f"Watching {config_path} for changes...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
```

### Pattern 2: Polling-based Reload

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-polling-reload
  namespace: ml-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
    spec:
      containers:
      # Main application
      - name: app
        image: ml-app:v1.0.0
        volumeMounts:
        - name: config
          mountPath: /config
          readOnly: true
        env:
        - name: CONFIG_RELOAD_INTERVAL
          value: "30"  # seconds
        - name: CONFIG_PATH
          value: "/config/config.yaml"

      # Config reloader sidecar
      - name: config-reloader
        image: jimmidyson/configmap-reload:v0.8.0
        args:
        - --volume-dir=/config
        - --webhook-url=http://localhost:8080/api/reload
        - --webhook-method=POST
        - --webhook-retries=3
        volumeMounts:
        - name: config
          mountPath: /config
          readOnly: true

      volumes:
      - name: config
        configMap:
          name: ml-model-config
```

### Pattern 3: Signal-based Reload (SIGHUP)

```python
# signal_reload.py
import signal
import yaml
import os

class ConfigurableMLModel:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = None
        self.load_config()

        # Register signal handler
        signal.signal(signal.SIGHUP, self.reload_config_signal)

    def load_config(self):
        """Load configuration from file"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        print(f"Config loaded: {self.config}")

    def reload_config_signal(self, signum, frame):
        """Reload config on SIGHUP signal"""
        print("Received SIGHUP, reloading configuration...")
        try:
            self.load_config()
            # Reinitialize model with new config
            self.reinitialize_model()
        except Exception as e:
            print(f"Error reloading config: {e}")

    def reinitialize_model(self):
        """Reinitialize model with new configuration"""
        print("Reinitializing model with new configuration...")
        # Model reinitialization logic here
```

Kubernetes deployment with signal-based reload:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-signal-reload
  namespace: ml-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
      annotations:
        # Add checksum to force pod restart if needed
        configmap/checksum: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
    spec:
      shareProcessNamespace: true  # Allow signaling between containers
      containers:
      - name: app
        image: ml-app:v1.0.0
        volumeMounts:
        - name: config
          mountPath: /config
          readOnly: true
        command:
        - python
        - signal_reload.py
        - --config=/config/config.yaml

      # Sidecar sends SIGHUP on config change
      - name: config-watcher
        image: busybox:1.36
        volumeMounts:
        - name: config
          mountPath: /config
          readOnly: true
        command:
        - /bin/sh
        - -c
        - |
          CONFIG="/config/config.yaml"
          LAST_HASH=""

          while true; do
            CURRENT_HASH=$(md5sum "$CONFIG" | awk '{print $1}')

            if [ -n "$LAST_HASH" ] && [ "$LAST_HASH" != "$CURRENT_HASH" ]; then
              echo "Config changed, sending SIGHUP to app"
              killall -HUP python
            fi

            LAST_HASH="$CURRENT_HASH"
            sleep 15
          done

      volumes:
      - name: config
        configMap:
          name: ml-model-config
```

## Multi-Environment Configuration

Managing configuration across multiple environments (dev, staging, production) is essential for ML pipelines.

### Pattern 1: Namespace-based Isolation

```bash
# Create separate namespaces
kubectl create namespace ml-dev
kubectl create namespace ml-staging
kubectl create namespace ml-production

# Label namespaces
kubectl label namespace ml-dev environment=development
kubectl label namespace ml-staging environment=staging
kubectl label namespace ml-production environment=production
```

Create environment-specific ConfigMaps:

```yaml
# dev-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
  namespace: ml-dev
  labels:
    environment: development
data:
  ENV: "development"
  LOG_LEVEL: "DEBUG"
  MODEL_VERSION: "latest"
  BATCH_SIZE: "8"
  ENABLE_PROFILING: "true"
  DATABASE_HOST: "postgres-dev.ml-dev.svc.cluster.local"
  MLFLOW_TRACKING_URI: "http://mlflow-dev:5000"
  S3_BUCKET: "ml-models-dev"
---
# staging-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
  namespace: ml-staging
  labels:
    environment: staging
data:
  ENV: "staging"
  LOG_LEVEL: "INFO"
  MODEL_VERSION: "v1.2.0-rc1"
  BATCH_SIZE: "32"
  ENABLE_PROFILING: "false"
  DATABASE_HOST: "postgres-staging.ml-staging.svc.cluster.local"
  MLFLOW_TRACKING_URI: "http://mlflow-staging:5000"
  S3_BUCKET: "ml-models-staging"
---
# production-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
  namespace: ml-production
  labels:
    environment: production
data:
  ENV: "production"
  LOG_LEVEL: "WARNING"
  MODEL_VERSION: "v1.1.0"
  BATCH_SIZE: "64"
  ENABLE_PROFILING: "false"
  DATABASE_HOST: "postgres-prod.ml-production.svc.cluster.local"
  MLFLOW_TRACKING_URI: "http://mlflow-prod:5000"
  S3_BUCKET: "ml-models-production"
```

### Pattern 2: Layered Configuration (Base + Override)

```yaml
# base-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-base-config
  namespace: ml-production
data:
  # Common configuration across all environments
  APP_NAME: "ml-inference-service"
  MODEL_FRAMEWORK: "pytorch"
  METRICS_PORT: "9090"
  HEALTH_CHECK_PATH: "/health"
  MAX_WORKERS: "4"
---
# production-overrides.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-production-overrides
  namespace: ml-production
data:
  # Production-specific overrides
  LOG_LEVEL: "WARNING"
  ENABLE_DEBUG: "false"
  RATE_LIMIT: "1000"
  CACHE_TTL: "3600"
  DATABASE_POOL_SIZE: "20"
```

Use both ConfigMaps:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  namespace: ml-production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
    spec:
      containers:
      - name: app
        image: ml-service:v1.0.0
        envFrom:
        # Load base config first
        - configMapRef:
            name: ml-base-config
        # Override with environment-specific config
        - configMapRef:
            name: ml-production-overrides
```

### Pattern 3: Kustomize for Multi-Environment

Directory structure:

```
k8s/
├── base/
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   └── configmap.yaml
├── overlays/
│   ├── dev/
│   │   ├── kustomization.yaml
│   │   ├── configmap-patch.yaml
│   │   └── replicas-patch.yaml
│   ├── staging/
│   │   ├── kustomization.yaml
│   │   ├── configmap-patch.yaml
│   │   └── replicas-patch.yaml
│   └── production/
│       ├── kustomization.yaml
│       ├── configmap-patch.yaml
│       └── replicas-patch.yaml
```

Base configuration:

```yaml
# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- deployment.yaml
- configmap.yaml

commonLabels:
  app: ml-service

# base/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
data:
  APP_NAME: "ml-service"
  MODEL_FRAMEWORK: "pytorch"
```

Production overlay:

```yaml
# overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: ml-production

bases:
- ../../base

patchesStrategicMerge:
- configmap-patch.yaml
- replicas-patch.yaml

commonLabels:
  environment: production

# overlays/production/configmap-patch.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-config
data:
  ENV: "production"
  LOG_LEVEL: "WARNING"
  REPLICAS: "5"
  BATCH_SIZE: "64"
```

Deploy:

```bash
# Development
kubectl apply -k k8s/overlays/dev

# Staging
kubectl apply -k k8s/overlays/staging

# Production
kubectl apply -k k8s/overlays/production
```

## Production ML Configuration Management

Real-world patterns for managing ML infrastructure in production.

### Complete ML Inference Service

```yaml
# ml-inference-production.yaml
---
# Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: ml-production
  labels:
    environment: production
    team: ml-platform
---
# Model configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: ml-production
data:
  # Model metadata
  MODEL_NAME: "text-classifier-bert"
  MODEL_VERSION: "v2.1.0"
  MODEL_PATH: "/models/production"

  # Inference settings
  BATCH_SIZE: "32"
  MAX_LENGTH: "512"
  NUM_WORKERS: "4"
  DEVICE: "cuda"
  PRECISION: "fp16"

  # Performance tuning
  ENABLE_BATCHING: "true"
  MAX_BATCH_WAIT_MS: "100"
  CACHE_ENABLED: "true"
  CACHE_TTL_SECONDS: "300"

  # Monitoring
  METRICS_ENABLED: "true"
  METRICS_PORT: "9090"
  LOG_PREDICTIONS: "true"
  LOG_LATENCY: "true"

  # Model configuration file
  config.yaml: |
    model:
      name: "bert-base-uncased"
      task: "text-classification"
      num_labels: 3
      checkpoint: "/models/production/checkpoint-best.pt"

    preprocessing:
      tokenizer: "bert-base-uncased"
      max_length: 512
      padding: "max_length"
      truncation: true

    inference:
      batch_size: 32
      num_workers: 4
      device: "cuda"
      precision: "fp16"

      batching:
        enabled: true
        max_wait_ms: 100
        max_batch_size: 64

      caching:
        enabled: true
        ttl_seconds: 300
        max_size_mb: 1000

    monitoring:
      metrics:
        enabled: true
        port: 9090
        endpoint: "/metrics"

      logging:
        level: "INFO"
        log_predictions: true
        log_latency: true
        sample_rate: 0.1  # Log 10% of requests
---
# Database credentials (use external secret manager in real production)
apiVersion: v1
kind: Secret
metadata:
  name: database-credentials
  namespace: ml-production
type: Opaque
stringData:
  username: "ml_inference_user"
  password: "CHANGE_ME_IN_PRODUCTION"
  host: "postgres-prod.database.svc.cluster.local"
  port: "5432"
  database: "ml_production"
  connection_string: "postgresql://ml_inference_user:CHANGE_ME_IN_PRODUCTION@postgres-prod.database.svc.cluster.local:5432/ml_production"
---
# API keys and tokens
apiVersion: v1
kind: Secret
metadata:
  name: api-credentials
  namespace: ml-production
type: Opaque
stringData:
  mlflow_token: "MLFLOW_TOKEN_HERE"
  s3_access_key: "AWS_ACCESS_KEY"
  s3_secret_key: "AWS_SECRET_KEY"
  monitoring_token: "PROMETHEUS_TOKEN"
---
# Service account
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ml-inference
  namespace: ml-production
---
# Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
  namespace: ml-production
  labels:
    app: ml-inference
    version: v2.1.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-inference
  template:
    metadata:
      labels:
        app: ml-inference
        version: v2.1.0
      annotations:
        # Prometheus scraping
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: ml-inference

      # Security context
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault

      # Init container: Download model
      initContainers:
      - name: model-downloader
        image: amazon/aws-cli:latest
        command:
        - /bin/sh
        - -c
        - |
          echo "Downloading model from S3..."
          aws s3 sync s3://ml-models/text-classifier/v2.1.0/ /models/production/
          echo "Model downloaded successfully"
        env:
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: api-credentials
              key: s3_access_key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: api-credentials
              key: s3_secret_key
        - name: AWS_DEFAULT_REGION
          value: "us-east-1"
        volumeMounts:
        - name: model-storage
          mountPath: /models

      containers:
      # Main inference server
      - name: inference-server
        image: ml-inference:v2.1.0
        imagePullPolicy: Always

        # Environment variables from ConfigMap
        envFrom:
        - configMapRef:
            name: model-config

        # Sensitive env vars from Secrets
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: connection_string
        - name: MLFLOW_TRACKING_TOKEN
          valueFrom:
            secretKeyRef:
              name: api-credentials
              key: mlflow_token

        # Volume mounts
        volumeMounts:
        # Model files
        - name: model-storage
          mountPath: /models
          readOnly: true
        # Config files
        - name: model-config-files
          mountPath: /config
          readOnly: true
        # Secrets
        - name: credentials
          mountPath: /secrets
          readOnly: true
        # Temporary storage
        - name: tmp
          mountPath: /tmp

        # Ports
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP

        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2

        # Resources
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"

        # Security context
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL

      # Sidecar: Config hot-reload
      - name: config-reloader
        image: jimmidyson/configmap-reload:v0.8.0
        args:
        - --volume-dir=/config
        - --webhook-url=http://localhost:8080/api/reload
        - --webhook-method=POST
        volumeMounts:
        - name: model-config-files
          mountPath: /config
          readOnly: true
        resources:
          requests:
            memory: "32Mi"
            cpu: "50m"
          limits:
            memory: "64Mi"
            cpu: "100m"

      volumes:
      # Empty dir for model storage
      - name: model-storage
        emptyDir:
          sizeLimit: 10Gi

      # ConfigMap files
      - name: model-config-files
        configMap:
          name: model-config
          items:
          - key: config.yaml
            path: config.yaml

      # Secrets
      - name: credentials
        secret:
          secretName: api-credentials
          defaultMode: 0400

      # Temporary storage
      - name: tmp
        emptyDir: {}

      # Node selector for GPU nodes
      nodeSelector:
        node.kubernetes.io/instance-type: "g4dn.xlarge"

      # Tolerations for GPU nodes
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
---
# Service
apiVersion: v1
kind: Service
metadata:
  name: ml-inference
  namespace: ml-production
  labels:
    app: ml-inference
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  selector:
    app: ml-inference
---
# HPA for autoscaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
  namespace: ml-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference
  minReplicas: 3
  maxReplicas: 10
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
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max
```

Deploy:

```bash
kubectl apply -f ml-inference-production.yaml

# Verify deployment
kubectl get all -n ml-production
kubectl describe deployment ml-inference -n ml-production
kubectl logs -n ml-production deployment/ml-inference -c inference-server
```

## Security Best Practices

### 1. Enable Encryption at Rest

Configure etcd encryption:

```yaml
# encryption-config.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
    - secrets
    - configmaps
    providers:
    - aescbc:
        keys:
        - name: key1
          secret: <BASE64_ENCODED_32_BYTE_KEY>
    - identity: {}
```

Apply to API server:

```bash
# Update kube-apiserver manifest
--encryption-provider-config=/etc/kubernetes/enc/encryption-config.yaml
```

### 2. Implement RBAC

```yaml
# rbac.yaml
---
# Role: Read-only access to specific secrets
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ml-secrets-reader
  namespace: ml-production
rules:
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["api-credentials", "database-credentials"]
  verbs: ["get"]
---
# Role: ConfigMap management
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ml-config-manager
  namespace: ml-production
rules:
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
---
# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ml-app-secrets
  namespace: ml-production
subjects:
- kind: ServiceAccount
  name: ml-inference
  namespace: ml-production
roleRef:
  kind: Role
  name: ml-secrets-reader
  apiGroup: rbac.authorization.k8s.io
---
# RoleBinding for config
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ml-app-config
  namespace: ml-production
subjects:
- kind: ServiceAccount
  name: ml-inference
  namespace: ml-production
roleRef:
  kind: Role
  name: ml-config-manager
  apiGroup: rbac.authorization.k8s.io
```

### 3. Secret Rotation Strategy

```bash
# rotation-script.sh
#!/bin/bash

set -euo pipefail

NAMESPACE="ml-production"
SECRET_NAME="database-credentials"
DEPLOYMENT="ml-inference"

echo "Starting secret rotation for $SECRET_NAME"

# 1. Backup current secret
echo "Backing up current secret..."
kubectl get secret $SECRET_NAME -n $NAMESPACE -o yaml > \
  "backup-$SECRET_NAME-$(date +%Y%m%d-%H%M%S).yaml"

# 2. Generate new password
NEW_PASSWORD=$(openssl rand -base64 32)

# 3. Update database with new password (application-specific)
echo "Updating database password..."
# psql -h $DB_HOST -U postgres -c "ALTER USER ml_user WITH PASSWORD '$NEW_PASSWORD';"

# 4. Create new secret version
echo "Creating new secret..."
kubectl create secret generic $SECRET_NAME-new \
  --from-literal=username=ml_user \
  --from-literal=password="$NEW_PASSWORD" \
  --from-literal=host=postgres-prod.database.svc.cluster.local \
  --from-literal=port=5432 \
  --from-literal=database=ml_production \
  --dry-run=client -o yaml | kubectl apply -f -

# 5. Update deployment to use new secret
echo "Updating deployment..."
kubectl patch deployment $DEPLOYMENT -n $NAMESPACE -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"inference-server","env":[{"name":"DATABASE_URL","valueFrom":{"secretKeyRef":{"name":"'$SECRET_NAME-new'","key":"connection_string"}}}]}]}}}}'

# 6. Wait for rollout
echo "Waiting for rollout..."
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE

# 7. Verify new secret works
echo "Verifying new secret..."
POD=$(kubectl get pod -n $NAMESPACE -l app=ml-inference -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n $NAMESPACE $POD -- /bin/sh -c 'psql $DATABASE_URL -c "SELECT 1"'

# 8. Delete old secret
echo "Deleting old secret..."
kubectl delete secret $SECRET_NAME -n $NAMESPACE

# 9. Rename new secret
kubectl get secret $SECRET_NAME-new -n $NAMESPACE -o yaml | \
  sed 's/'$SECRET_NAME-new'/'$SECRET_NAME'/g' | \
  kubectl apply -f -

kubectl delete secret $SECRET_NAME-new -n $NAMESPACE

echo "Secret rotation completed successfully!"
```

### 4. Audit Logging

Enable audit logging in kube-apiserver:

```yaml
# audit-policy.yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
# Log secret access at RequestResponse level
- level: RequestResponse
  resources:
  - group: ""
    resources: ["secrets"]
  namespaces: ["ml-production"]

# Log ConfigMap changes at Metadata level
- level: Metadata
  resources:
  - group: ""
    resources: ["configmaps"]
  namespaces: ["ml-production"]
  verbs: ["create", "update", "patch", "delete"]

# Don't log reads of ConfigMaps
- level: None
  resources:
  - group: ""
    resources: ["configmaps"]
  verbs: ["get", "list", "watch"]
```

### 5. Security Checklist

```markdown
## ML Infrastructure Security Checklist

### Secrets Management
- [ ] Enable encryption at rest for etcd
- [ ] Use external secret management (Vault, AWS Secrets Manager)
- [ ] Implement secret rotation (90-day maximum)
- [ ] Never commit secrets to Git
- [ ] Use `.dockerconfigjson` for registry credentials
- [ ] Mount secrets as volumes, not environment variables
- [ ] Set restrictive permissions (0400) on secret files
- [ ] Use separate secrets per environment

### RBAC
- [ ] Implement least privilege access
- [ ] Create dedicated service accounts
- [ ] Limit secret access to specific resource names
- [ ] Regular RBAC audits
- [ ] Document all role bindings

### Configuration
- [ ] Separate configuration by environment
- [ ] Use immutable ConfigMaps in production
- [ ] Version all configurations
- [ ] Implement configuration validation
- [ ] Document all configuration keys

### Monitoring & Audit
- [ ] Enable audit logging for secret access
- [ ] Monitor for unexpected config changes
- [ ] Alert on failed authentication
- [ ] Track secret rotation dates
- [ ] Regular security reviews

### Pod Security
- [ ] Run containers as non-root (runAsNonRoot: true)
- [ ] Read-only root filesystem
- [ ] Drop all capabilities
- [ ] Use security contexts
- [ ] Scan images for vulnerabilities

### Network Security
- [ ] Implement network policies
- [ ] Use TLS for all communications
- [ ] Limit egress traffic
- [ ] Service mesh for mTLS
```

## Troubleshooting Guide

### Issue 1: Pod Can't Find ConfigMap

**Symptoms:**
```
Events:
  Warning  FailedMount  configmap "ml-config" not found
```

**Solution:**
```bash
# Check if ConfigMap exists
kubectl get configmap ml-config -n ml-production

# Check namespace
kubectl get configmaps --all-namespaces | grep ml-config

# Create if missing
kubectl apply -f ml-configs.yaml

# Verify pod can see it
kubectl describe pod <pod-name> -n ml-production
```

### Issue 2: Secret Not Decoded Properly

**Symptoms:**
```
Application error: Invalid base64 string
```

**Solution:**
```bash
# Check secret encoding
kubectl get secret api-credentials -n ml-production -o yaml

# Decode manually to verify
kubectl get secret api-credentials -n ml-production \
  -o jsonpath='{.data.api-key}' | base64 -d

# Recreate with stringData
kubectl delete secret api-credentials -n ml-production
kubectl create secret generic api-credentials \
  --from-literal=api-key='correct-key-here' \
  -n ml-production
```

### Issue 3: Configuration Changes Not Reflected

**Symptoms:**
- Environment variables don't update
- Old configuration still used

**Solution:**
```bash
# For environment variables: Restart pods
kubectl rollout restart deployment ml-inference -n ml-production

# For volume mounts: Wait 60-90 seconds for kubelet sync
kubectl exec -n ml-production <pod-name> -- cat /config/config.yaml

# Force immediate update
kubectl delete pod <pod-name> -n ml-production
```

### Issue 4: Permission Denied on Secret Files

**Symptoms:**
```
Error: Permission denied reading /secrets/api-key
```

**Solution:**
```bash
# Check file permissions
kubectl exec -n ml-production <pod-name> -- ls -la /secrets/

# Check security context
kubectl get pod <pod-name> -n ml-production -o yaml | grep -A 10 securityContext

# Fix: Update secret volume permissions
kubectl patch deployment ml-inference -n ml-production -p \
  '{"spec":{"template":{"spec":{"volumes":[{"name":"credentials","secret":{"secretName":"api-credentials","defaultMode":420}}]}}}}'
```

### Issue 5: ConfigMap Too Large

**Symptoms:**
```
Error: ConfigMap size exceeds 1 MB limit
```

**Solution:**
```bash
# Check ConfigMap size
kubectl get configmap ml-config -n ml-production -o yaml | wc -c

# Split into multiple ConfigMaps
kubectl create configmap ml-config-part1 --from-file=config1.yaml
kubectl create configmap ml-config-part2 --from-file=config2.yaml

# Or use external configuration service
```

## Conclusion

This implementation guide covered comprehensive patterns for managing configuration and secrets in Kubernetes for ML infrastructure. Key takeaways:

1. **Use ConfigMaps** for non-sensitive configuration (model parameters, features)
2. **Use Secrets** for sensitive data (API keys, passwords)
3. **Prefer volume mounts** over environment variables for security and hot-reload
4. **Implement external secret management** for production systems
5. **Enable hot-reload** to update models without downtime
6. **Isolate environments** with separate namespaces and configurations
7. **Follow security best practices**: RBAC, encryption, rotation, audit logging

For production ML systems, always use external secret managers (Vault, AWS Secrets Manager) and implement automated secret rotation.

## Additional Resources

- [Kubernetes ConfigMaps](https://kubernetes.io/docs/concepts/configuration/configmap/)
- [Kubernetes Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
- [External Secrets Operator](https://external-secrets.io/)
- [HashiCorp Vault](https://www.vaultproject.io/)
- [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets)
- [MLflow Configuration](https://mlflow.org/docs/latest/tracking.html)
- [Kubeflow Configuration](https://www.kubeflow.org/docs/)
