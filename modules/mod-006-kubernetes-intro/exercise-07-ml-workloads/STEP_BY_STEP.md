# Step-by-Step Implementation Guide: ML Workloads on Kubernetes

## Overview

Deploy production-ready machine learning workloads on Kubernetes, including model serving, training jobs, A/B testing, auto-scaling, and GPU management. Learn to build a complete ML infrastructure platform.

**Time**: 4-5 hours | **Difficulty**: Advanced

---

## Phase 1: Setup and Understanding (25 minutes)

### Step 1: Understand ML Workload Architecture

**Key Components**:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT REQUESTS                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Ingress Controller    │  (Load balancer + routing)
        │  - A/B testing         │
        │  - Canary routing      │
        └────────────┬───────────┘
                     │
        ┌────────────┴──────────────────┐
        ▼                               ▼
┌───────────────────┐         ┌───────────────────┐
│  Model v1 (90%)   │         │  Model v2 (10%)   │
│  3 replicas       │         │  1 replica        │
│  Production       │         │  Canary           │
└────────┬──────────┘         └─────────┬─────────┘
         │                              │
         └──────────────┬───────────────┘
                        │
            ┌───────────▼──────────┐
            │  Model Storage       │
            │  (Persistent Volume) │
            │  - Versioned models  │
            │  - Metadata          │
            └──────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Batch Job   │  CronJob     │ Distributed  │  GPU Training  │
│  (One-time)  │  (Scheduled) │  (4 workers) │  (GPU nodes)   │
└──────────────┴──────────────┴──────────────┴────────────────┘
                     │
         ┌───────────▼──────────┐
         │  Training Data       │
         │  (Persistent Volume) │
         └──────────────────────┘
```

**Differences from Traditional Deployments**:

| Aspect | Traditional Apps | ML Workloads |
|--------|------------------|--------------|
| **Resource Usage** | Predictable | Variable (spikes during inference) |
| **Scaling** | Based on requests | Based on queue depth, model load |
| **Storage** | Small configs | Large models (GBs), training data (TBs) |
| **Versioning** | Code versions | Model + code + data versions |
| **Testing** | Unit/integration | A/B testing, shadow testing |
| **GPU** | Not needed | Critical for training, optional for inference |

### Step 2: Create Namespace with Resource Quotas

**`manifests/01-namespace.yaml`**:
```yaml
# Namespace for ML workloads
apiVersion: v1
kind: Namespace
metadata:
  name: ml-workloads
  labels:
    name: ml-workloads
    environment: production
---
# Resource quota to prevent resource exhaustion
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-quota
  namespace: ml-workloads
spec:
  hard:
    requests.cpu: "20"           # Max 20 CPUs requested
    requests.memory: 40Gi        # Max 40GB memory requested
    requests.nvidia.com/gpu: "4" # Max 4 GPUs requested
    limits.cpu: "40"             # Max 40 CPUs limit
    limits.memory: 80Gi          # Max 80GB memory limit
    limits.nvidia.com/gpu: "4"   # Max 4 GPUs limit
    persistentvolumeclaims: "5"  # Max 5 PVCs
    pods: "50"                   # Max 50 pods
---
# Limit ranges to set defaults and constraints
apiVersion: v1
kind: LimitRange
metadata:
  name: ml-limit-range
  namespace: ml-workloads
spec:
  limits:
  - max:
      cpu: "8"
      memory: 16Gi
    min:
      cpu: "100m"
      memory: 128Mi
    default:
      cpu: "1"
      memory: 1Gi
    defaultRequest:
      cpu: "500m"
      memory: 512Mi
    type: Container
  - max:
      cpu: "16"
      memory: 32Gi
    min:
      cpu: "100m"
      memory: 128Mi
    type: Pod
```

**Apply namespace configuration**:
```bash
kubectl apply -f manifests/01-namespace.yaml

# Verify resource quotas
kubectl describe resourcequota ml-quota -n ml-workloads

# Expected output shows limits for CPU, memory, GPU, pods

# Verify limit ranges
kubectl describe limitrange ml-limit-range -n ml-workloads
```

---

## Phase 2: Model Storage Setup (30 minutes)

### Step 3: Create Persistent Storage for Models

**`manifests/02-model-storage.yaml`**:
```yaml
# PVC for model storage (versioned models)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
  namespace: ml-workloads
spec:
  accessModes:
    - ReadWriteMany  # Multiple pods can read/write
  storageClassName: standard  # Adjust for your cluster
  resources:
    requests:
      storage: 50Gi
---
# PVC for training data
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-pvc
  namespace: ml-workloads
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: standard
  resources:
    requests:
      storage: 100Gi
---
# Init job to create model directory structure
apiVersion: batch/v1
kind: Job
metadata:
  name: storage-init
  namespace: ml-workloads
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: init
        image: busybox:1.36
        command:
        - /bin/sh
        - -c
        - |
          echo "Initializing model storage..."

          # Create directory structure
          mkdir -p /models/sentiment-classifier/1
          mkdir -p /models/sentiment-classifier/2
          mkdir -p /models/image-classifier/1
          mkdir -p /models/registry

          # Create metadata
          cat > /models/sentiment-classifier/1/metadata.json <<EOF
          {
            "name": "sentiment-classifier",
            "version": "1",
            "framework": "transformers",
            "accuracy": 0.92,
            "created": "$(date -Iseconds)",
            "description": "BERT-based sentiment analysis model"
          }
          EOF

          cat > /models/sentiment-classifier/2/metadata.json <<EOF
          {
            "name": "sentiment-classifier",
            "version": "2",
            "framework": "transformers",
            "accuracy": 0.94,
            "created": "$(date -Iseconds)",
            "description": "Improved sentiment analysis with RoBERTa"
          }
          EOF

          # Create registry index
          cat > /models/registry/models.json <<EOF
          {
            "models": [
              {
                "name": "sentiment-classifier",
                "versions": ["1", "2"],
                "latest": "2"
              },
              {
                "name": "image-classifier",
                "versions": ["1"],
                "latest": "1"
              }
            ]
          }
          EOF

          echo "Storage initialization complete!"
          ls -laR /models/
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

**Apply and verify storage**:
```bash
kubectl apply -f manifests/02-model-storage.yaml

# Wait for PVCs to be bound
kubectl get pvc -n ml-workloads

# Expected output:
# NAME                 STATUS   VOLUME                                     CAPACITY   ACCESS MODES
# models-pvc           Bound    pvc-xxx-xxx                                50Gi       RWX
# training-data-pvc    Bound    pvc-yyy-yyy                                100Gi      RWX

# Wait for storage init job to complete
kubectl wait --for=condition=complete job/storage-init -n ml-workloads --timeout=60s

# Verify directory structure
kubectl logs job/storage-init -n ml-workloads

# Expected: Directory tree showing model structure
```

---

## Phase 3: Model Serving Deployment (45 minutes)

### Step 4: Deploy Model Serving (REST API)

**`manifests/03-model-serving.yaml`** (partial):
```yaml
# ConfigMap with model serving code
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-server-code
  namespace: ml-workloads
data:
  app.py: |
    from flask import Flask, request, jsonify
    import json
    import os
    import time
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    from prometheus_client import make_wsgi_app

    app = Flask(__name__)

    # Prometheus metrics
    REQUEST_COUNT = Counter('model_requests_total', 'Total requests', ['model', 'version', 'status'])
    REQUEST_LATENCY = Histogram('model_request_latency_seconds', 'Request latency', ['model', 'version'])
    PREDICTIONS = Counter('model_predictions_total', 'Total predictions', ['model', 'version'])

    MODEL_DIR = os.getenv('MODEL_DIR', '/models')
    MODEL_NAME = os.getenv('MODEL_NAME', 'sentiment-classifier')
    MODEL_VERSION = os.getenv('MODEL_VERSION', '1')

    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy', 'model': MODEL_NAME, 'version': MODEL_VERSION})

    @app.route('/ready')
    def ready():
        # Check if model is loaded
        model_path = f"{MODEL_DIR}/{MODEL_NAME}/{MODEL_VERSION}"
        if os.path.exists(model_path):
            return jsonify({'status': 'ready'}), 200
        return jsonify({'status': 'not ready'}), 503

    @app.route('/predict', methods=['POST'])
    def predict():
        start_time = time.time()

        try:
            data = request.get_json()
            text = data.get('text', '')

            # Simulate model inference
            # In production, load actual model and run inference
            prediction = {
                'text': text,
                'sentiment': 'positive' if len(text) % 2 == 0 else 'negative',
                'confidence': 0.92,
                'model': MODEL_NAME,
                'version': MODEL_VERSION
            }

            # Record metrics
            REQUEST_COUNT.labels(model=MODEL_NAME, version=MODEL_VERSION, status='success').inc()
            PREDICTIONS.labels(model=MODEL_NAME, version=MODEL_VERSION).inc()

            duration = time.time() - start_time
            REQUEST_LATENCY.labels(model=MODEL_NAME, version=MODEL_VERSION).observe(duration)

            return jsonify(prediction)

        except Exception as e:
            REQUEST_COUNT.labels(model=MODEL_NAME, version=MODEL_VERSION, status='error').inc()
            return jsonify({'error': str(e)}), 500

    @app.route('/models')
    def list_models():
        """List available models"""
        registry_path = f"{MODEL_DIR}/registry/models.json"
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            return jsonify(registry)
        except:
            return jsonify({'error': 'Registry not found'}), 404

    @app.route('/metrics')
    def metrics():
        return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=8080)
---
# Deployment for model v1 (production)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-v1
  namespace: ml-workloads
  labels:
    app: ml-model
    version: v1
    component: model-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
      version: v1
  template:
    metadata:
      labels:
        app: ml-model
        version: v1
        component: model-server
    spec:
      containers:
      - name: model-server
        image: python:3.11-slim
        env:
        - name: MODEL_NAME
          value: "sentiment-classifier"
        - name: MODEL_VERSION
          value: "1"
        - name: MODEL_DIR
          value: "/models"
        command:
        - /bin/bash
        - -c
        - |
          pip install -q flask prometheus-client
          python /app/app.py
        ports:
        - containerPort: 8080
          name: http
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "2Gi"
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: code
          mountPath: /app
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: code
        configMap:
          name: model-server-code
---
# Deployment for model v2 (canary)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-v2
  namespace: ml-workloads
  labels:
    app: ml-model
    version: v2
    component: model-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-model
      version: v2
  template:
    metadata:
      labels:
        app: ml-model
        version: v2
        component: model-server
    spec:
      containers:
      - name: model-server
        image: python:3.11-slim
        env:
        - name: MODEL_NAME
          value: "sentiment-classifier"
        - name: MODEL_VERSION
          value: "2"
        - name: MODEL_DIR
          value: "/models"
        command:
        - /bin/bash
        - -c
        - |
          pip install -q flask prometheus-client
          python /app/app.py
        ports:
        - containerPort: 8080
          name: http
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "2Gi"
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: code
          mountPath: /app
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: code
        configMap:
          name: model-server-code
---
# Service for model v1
apiVersion: v1
kind: Service
metadata:
  name: model-v1-service
  namespace: ml-workloads
spec:
  type: ClusterIP
  selector:
    app: ml-model
    version: v1
  ports:
  - port: 80
    targetPort: 8080
    name: http
---
# Service for model v2
apiVersion: v1
kind: Service
metadata:
  name: model-v2-service
  namespace: ml-workloads
spec:
  type: ClusterIP
  selector:
    app: ml-model
    version: v2
  ports:
  - port: 80
    targetPort: 8080
    name: http
---
# Horizontal Pod Autoscaler for model v1
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-v1-hpa
  namespace: ml-workloads
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-v1
  minReplicas: 2
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
---
# Pod Disruption Budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: model-v1-pdb
  namespace: ml-workloads
spec:
  minAvailable: 2  # Always keep at least 2 pods running
  selector:
    matchLabels:
      app: ml-model
      version: v1
```

**Deploy and test model serving**:
```bash
kubectl apply -f manifests/03-model-serving.yaml

# Wait for pods to be ready
kubectl get pods -n ml-workloads -l component=model-server -w

# Expected: 3 pods for v1, 1 pod for v2, all Running

# Test model v1
kubectl run test-v1 --rm -it --image=curlimages/curl -n ml-workloads --restart=Never \
  -- curl -X POST http://model-v1-service/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "This product is amazing!"}'

# Expected: JSON response with sentiment prediction and version=1

# Test model v2
kubectl run test-v2 --rm -it --image=curlimages/curl -n ml-workloads --restart=Never \
  -- curl -X POST http://model-v2-service/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "This product is terrible!"}'

# Expected: JSON response with version=2

# Check HPA status
kubectl get hpa -n ml-workloads

# Check PDB
kubectl get pdb -n ml-workloads

# View metrics
kubectl port-forward -n ml-workloads deployment/model-v1 8080:8080
# In another terminal:
curl http://localhost:8080/metrics | grep model_
```

---

## Phase 4: Training Jobs (50 minutes)

### Step 5: Create Simple Training Job

**`manifests/04-training-jobs.yaml`** (partial):
```yaml
# Simple batch training job
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training-simple
  namespace: ml-workloads
  labels:
    job-type: training
spec:
  backoffLimit: 3  # Retry 3 times on failure
  template:
    metadata:
      labels:
        job-type: training
    spec:
      restartPolicy: OnFailure
      containers:
      - name: trainer
        image: python:3.11-slim
        command:
        - /bin/bash
        - -c
        - |
          echo "Starting model training..."
          pip install -q numpy scikit-learn

          # Simulate training
          python <<EOF
          import json
          import time
          from datetime import datetime

          print("Loading training data...")
          time.sleep(2)

          print("Training model...")
          for epoch in range(1, 6):
              print(f"Epoch {epoch}/5 - Loss: {1.0 / epoch:.4f}")
              time.sleep(1)

          print("Training complete!")

          # Save model metadata
          metadata = {
              "name": "sentiment-classifier",
              "version": "3",
              "accuracy": 0.95,
              "trained_at": datetime.now().isoformat(),
              "epochs": 5,
              "framework": "scikit-learn"
          }

          with open('/models/sentiment-classifier/3/metadata.json', 'w') as f:
              json.dump(metadata, f, indent=2)

          print("Model saved to /models/sentiment-classifier/3/")
          EOF

          echo "Training job completed successfully!"
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        volumeMounts:
        - name: models
          mountPath: /models
        - name: training-data
          mountPath: /data
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: training-data
        persistentVolumeClaim:
          claimName: training-data-pvc
---
# Scheduled training (CronJob)
apiVersion: batch/v1
kind: CronJob
metadata:
  name: scheduled-model-training
  namespace: ml-workloads
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: trainer
            image: python:3.11-slim
            command:
            - /bin/bash
            - -c
            - |
              echo "Scheduled training started at $(date)"

              # Check if new data is available
              if [ -f /data/new_data_available ]; then
                echo "New data detected. Starting training..."
                # Run training logic here
                sleep 10
                echo "Training complete"
              else
                echo "No new data. Skipping training."
              fi
            resources:
              requests:
                cpu: "2"
                memory: "4Gi"
              limits:
                cpu: "4"
                memory: "8Gi"
            volumeMounts:
            - name: models
              mountPath: /models
            - name: training-data
              mountPath: /data
          volumes:
          - name: models
            persistentVolumeClaim:
              claimName: models-pvc
          - name: training-data
            persistentVolumeClaim:
              claimName: training-data-pvc
---
# Distributed training (4 workers)
apiVersion: batch/v1
kind: Job
metadata:
  name: distributed-training
  namespace: ml-workloads
  labels:
    job-type: distributed-training
spec:
  parallelism: 4  # 4 workers in parallel
  completions: 4  # All 4 must complete
  template:
    metadata:
      labels:
        job-type: distributed-training
    spec:
      restartPolicy: OnFailure
      containers:
      - name: worker
        image: python:3.11-slim
        env:
        - name: WORKER_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        command:
        - /bin/bash
        - -c
        - |
          echo "Distributed worker $WORKER_ID starting..."
          pip install -q numpy

          # Simulate distributed training
          python <<EOF
          import os
          import time

          worker_id = os.getenv('WORKER_ID', 'unknown')
          print(f"Worker {worker_id} processing batch...")
          time.sleep(10)
          print(f"Worker {worker_id} completed!")
          EOF
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
---
# GPU training job
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-training-job
  namespace: ml-workloads
  labels:
    job-type: gpu-training
spec:
  template:
    spec:
      restartPolicy: OnFailure
      nodeSelector:
        accelerator: nvidia-gpu  # Only schedule on GPU nodes
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: gpu-trainer
        image: nvidia/cuda:12.0.0-runtime-ubuntu22.04
        command:
        - /bin/bash
        - -c
        - |
          echo "GPU Training Job"
          echo "Checking GPU availability..."
          nvidia-smi || echo "No GPU detected (this is expected if no GPU nodes)"

          echo "Starting GPU-accelerated training..."
          sleep 10
          echo "GPU training complete!"
        resources:
          requests:
            nvidia.com/gpu: 1  # Request 1 GPU
            cpu: "4"
            memory: "16Gi"
          limits:
            nvidia.com/gpu: 1
            cpu: "8"
            memory: "32Gi"
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

**Run training jobs**:
```bash
kubectl apply -f manifests/04-training-jobs.yaml

# Check simple training job
kubectl get jobs -n ml-workloads

# Watch job progress
kubectl get pods -n ml-workloads -l job-type=training -w

# View training logs
kubectl logs job/model-training-simple -n ml-workloads -f

# Expected: Training progress output, model saved

# Check distributed training
kubectl get pods -n ml-workloads -l job-type=distributed-training

# Expected: 4 pods running in parallel

# View distributed worker logs
kubectl logs -n ml-workloads -l job-type=distributed-training --all-containers=true

# Check CronJob schedule
kubectl get cronjobs -n ml-workloads

# Manually trigger CronJob
kubectl create job --from=cronjob/scheduled-model-training test-run -n ml-workloads

# Check GPU job (if GPU nodes available)
kubectl describe job gpu-training-job -n ml-workloads

# Clean up completed jobs
kubectl delete jobs -n ml-workloads --field-selector status.successful=1
```

---

## Phase 5: A/B Testing and Canary Deployments (40 minutes)

### Step 6: Implement Weight-Based Canary

**`manifests/05-ab-testing-canary.yaml`**:
```yaml
# Production Ingress (primary - model v1)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-production
  namespace: ml-workloads
spec:
  ingressClassName: nginx
  rules:
  - host: ml-model.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: model-v1-service
            port:
              number: 80
---
# Canary Ingress (25% traffic to model v2)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-canary
  namespace: ml-workloads
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "25"  # 25% to canary
spec:
  ingressClassName: nginx
  rules:
  - host: ml-model.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: model-v2-service
            port:
              number: 80
---
# Header-based canary (for testing)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-header-canary
  namespace: ml-workloads
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-by-header: "X-Model-Version"
    nginx.ingress.kubernetes.io/canary-by-header-value: "v2"
spec:
  ingressClassName: nginx
  rules:
  - host: ml-model.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: model-v2-service
            port:
              number: 80
---
# Load testing job
apiVersion: batch/v1
kind: Job
metadata:
  name: model-load-test
  namespace: ml-workloads
spec:
  parallelism: 5  # 5 parallel clients
  completions: 5
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: load-tester
        image: curlimages/curl:latest
        command:
        - /bin/sh
        - -c
        - |
          echo "Starting load test..."

          for i in $(seq 1 100); do
            RESPONSE=$(curl -s -X POST http://model-v1-service/predict \
              -H 'Content-Type: application/json' \
              -d "{\"text\": \"Test message $i\"}")

            VERSION=$(echo $RESPONSE | grep -o '"version": "[^"]*"' | cut -d'"' -f4)
            echo "Request $i: Model version $VERSION"

            sleep 0.1
          done

          echo "Load test complete!"
```

**Deploy and test A/B setup**:
```bash
kubectl apply -f manifests/05-ab-testing-canary.yaml

# Add DNS entry (for testing)
INGRESS_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "$INGRESS_IP ml-model.example.com" | sudo tee -a /etc/hosts

# Test weight-based canary (expect 75% v1, 25% v2)
for i in {1..20}; do
  curl -s -X POST http://ml-model.example.com/predict \
    -H 'Content-Type: application/json' \
    -d '{"text": "test"}' | grep -o '"version": "[^"]*"'
  echo ""
done

# Test header-based routing (force v2)
curl -X POST http://ml-model.example.com/predict \
  -H 'Content-Type: application/json' \
  -H 'X-Model-Version: v2' \
  -d '{"text": "test"}' | grep version

# Run load test
kubectl create -f manifests/05-ab-testing-canary.yaml
kubectl logs -n ml-workloads -l job-name=model-load-test --all-containers=true

# Gradually increase canary to 50%
kubectl patch ingress model-canary -n ml-workloads \
  -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/canary-weight":"50"}}}'

# Monitor model performance
kubectl top pods -n ml-workloads -l app=ml-model

# Promote canary to 100% (full rollout)
kubectl patch ingress model-canary -n ml-workloads \
  -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/canary-weight":"100"}}}'

# Rollback to v1 if needed
kubectl patch ingress model-canary -n ml-workloads \
  -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/canary-weight":"0"}}}'
```

---

## Phase 6: Model Monitoring and Observability (30 minutes)

### Step 7: Monitor Model Metrics

```bash
# Port forward to view Prometheus metrics
kubectl port-forward -n ml-workloads deployment/model-v1 8080:8080

# In another terminal, view metrics
curl http://localhost:8080/metrics | grep ^model_

# Key metrics:
# - model_requests_total{model="sentiment-classifier",version="1",status="success"} 1234
# - model_request_latency_seconds_bucket{model="sentiment-classifier",version="1",le="0.1"} 1000
# - model_predictions_total{model="sentiment-classifier",version="1"} 1234

# Compare v1 vs v2 metrics
kubectl port-forward -n ml-workloads deployment/model-v2 8081:8080
curl http://localhost:8081/metrics | grep ^model_

# View resource usage
kubectl top pods -n ml-workloads -l component=model-server

# Check HPA scaling
kubectl get hpa model-v1-hpa -n ml-workloads

# Generate load to trigger autoscaling
kubectl run load-generator --rm -it --image=busybox -n ml-workloads --restart=Never \
  -- /bin/sh -c "while true; do wget -q -O- http://model-v1-service/predict; done"

# Watch HPA scale up
kubectl get hpa model-v1-hpa -n ml-workloads -w

# Expected: Replicas increase from 2 to higher count

# Stop load generator (Ctrl+C)
```

---

## Phase 7: GPU Training (Optional - 25 minutes)

### Step 8: Setup GPU Nodes (if available)

**Note**: Skip this section if GPU nodes are not available.

```bash
# Check GPU availability
kubectl get nodes -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.capacity.nvidia\.com/gpu}{"\n"}{end}'

# If GPUs are available:

# Label GPU nodes
kubectl label nodes <gpu-node-name> accelerator=nvidia-gpu

# Taint GPU nodes (optional - reserves for GPU workloads)
kubectl taint nodes <gpu-node-name> nvidia.com/gpu=present:NoSchedule

# Install NVIDIA device plugin (if not already installed)
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify device plugin
kubectl get pods -n kube-system | grep nvidia

# Run GPU training job
kubectl apply -f manifests/04-training-jobs.yaml

# Check GPU job status
kubectl describe job gpu-training-job -n ml-workloads

# View GPU job logs
kubectl logs -n ml-workloads -l job-type=gpu-training

# Expected: nvidia-smi output showing GPU info
```

---

## Phase 8: Production Best Practices (20 minutes)

### Step 9: Implement Production Patterns

**Model Versioning Best Practices**:
```bash
# View model registry
kubectl exec -n ml-workloads deployment/model-v1 \
  -- cat /models/registry/models.json

# Expected: JSON with all model versions

# Check model metadata
kubectl exec -n ml-workloads deployment/model-v1 \
  -- cat /models/sentiment-classifier/1/metadata.json

# Tag models for rollback
kubectl label deployment model-v1 -n ml-workloads \
  model-version=1.0.0 \
  commit-sha=abc123 \
  trained-date=2025-01-15
```

**Implement Model Rollback**:
```bash
# Save current deployment
kubectl get deployment model-v1 -n ml-workloads -o yaml > model-v1-backup.yaml

# Rollback to previous version
kubectl rollout undo deployment/model-v1 -n ml-workloads

# Check rollout history
kubectl rollout history deployment/model-v1 -n ml-workloads

# Rollback to specific revision
kubectl rollout undo deployment/model-v1 -n ml-workloads --to-revision=2
```

**Cost Monitoring**:
```bash
# View resource quotas
kubectl describe resourcequota ml-quota -n ml-workloads

# Calculate current usage
kubectl top nodes
kubectl top pods -n ml-workloads

# Cost estimation (example):
# 3 CPU pods × $0.04/hr × 730 hrs/month = $87.60/month
# 100GB storage × $0.10/GB/month = $10/month
# Total: ~$100/month (without GPU)
```

---

## Phase 9: Troubleshooting and Debugging (25 minutes)

### Step 10: Common Issues and Solutions

**Issue 1: Pod Pending - Insufficient Resources**:
```bash
# Check pod status
kubectl describe pod <pod-name> -n ml-workloads

# Look for events:
# "0/3 nodes are available: insufficient cpu"

# Solution 1: Reduce resource requests
kubectl edit deployment model-v1 -n ml-workloads
# Reduce requests.cpu from "500m" to "250m"

# Solution 2: Scale down other workloads
kubectl scale deployment model-v2 -n ml-workloads --replicas=0

# Solution 3: Add more nodes (cloud)
# For GKE:
# gcloud container clusters resize my-cluster --num-nodes=4
```

**Issue 2: Training Job OOMKilled**:
```bash
# Check job status
kubectl describe job model-training-simple -n ml-workloads

# Look for: "OOMKilled" in pod status

# View pod events
kubectl get events -n ml-workloads --sort-by='.lastTimestamp' | grep OOM

# Solution: Increase memory limits
kubectl edit job model-training-simple -n ml-workloads
# Increase limits.memory from "8Gi" to "16Gi"

# Re-run job
kubectl delete job model-training-simple -n ml-workloads
kubectl apply -f manifests/04-training-jobs.yaml
```

**Issue 3: Model Serving High Latency**:
```bash
# Check pod resource usage
kubectl top pods -n ml-workloads -l app=ml-model

# If CPU/memory is high:
# Solution 1: Scale up
kubectl scale deployment model-v1 -n ml-workloads --replicas=5

# Solution 2: Increase resource limits
kubectl edit deployment model-v1 -n ml-workloads

# Solution 3: Check HPA configuration
kubectl describe hpa model-v1-hpa -n ml-workloads
```

**Issue 4: PVC Not Binding**:
```bash
# Check PVC status
kubectl describe pvc models-pvc -n ml-workloads

# Common causes:
# - No storage class available
# - No available PVs
# - Access mode mismatch

# Solution 1: Check storage class
kubectl get storageclass

# Solution 2: Create PV manually (for local testing)
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: models-pv
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteMany
  hostPath:
    path: /mnt/data/models
EOF
```

---

## Phase 10: Cleanup and Summary (15 minutes)

### Step 11: Automated Cleanup

**`scripts/cleanup.sh`**:
```bash
#!/bin/bash

echo "Cleaning up ML workloads..."

# Delete Ingress
kubectl delete ingress --all -n ml-workloads

# Delete HPAs
kubectl delete hpa --all -n ml-workloads

# Delete PDBs
kubectl delete pdb --all -n ml-workloads

# Delete deployments
kubectl delete deployment --all -n ml-workloads

# Delete services
kubectl delete svc --all -n ml-workloads

# Delete jobs
kubectl delete jobs --all -n ml-workloads

# Delete cronjobs
kubectl delete cronjobs --all -n ml-workloads

# Delete configmaps
kubectl delete configmap --all -n ml-workloads

# Delete PVCs (WARNING: deletes data!)
read -p "Delete PVCs (will delete all model data)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kubectl delete pvc --all -n ml-workloads
fi

# Delete namespace
kubectl delete namespace ml-workloads

echo "Cleanup complete!"
```

```bash
chmod +x scripts/cleanup.sh
./scripts/cleanup.sh
```

---

## Summary

**What You Built**:
- ✅ Complete ML serving infrastructure with v1 (production) and v2 (canary)
- ✅ Persistent storage for models and training data
- ✅ REST API model serving with Prometheus metrics
- ✅ Batch training jobs (one-time)
- ✅ Scheduled training (CronJob)
- ✅ Distributed training (4 parallel workers)
- ✅ GPU training jobs (with GPU node support)
- ✅ Weight-based A/B testing (75% v1, 25% v2)
- ✅ Header-based canary routing
- ✅ Horizontal Pod Autoscaler (2-10 replicas)
- ✅ Pod Disruption Budget for high availability
- ✅ Resource quotas and limit ranges
- ✅ Model versioning and metadata
- ✅ Load testing and performance monitoring

**ML Kubernetes Patterns**:

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| **Model Serving** | Low-latency inference | Deployment + Service + HPA |
| **Batch Training** | One-time model training | Job with PVC mounts |
| **Scheduled Training** | Daily retraining | CronJob |
| **Distributed Training** | Large-scale training | Job with parallelism |
| **GPU Training** | Accelerated training | Job with nvidia.com/gpu requests |
| **A/B Testing** | Gradual model rollout | Canary Ingress with weights |
| **Auto-scaling** | Variable traffic | HPA based on CPU/memory |
| **High Availability** | Zero-downtime | Multiple replicas + PDB |

**Production Checklist**:
- [x] Resource requests and limits configured
- [x] Resource quotas enforced
- [x] Health checks (liveness, readiness) configured
- [x] Auto-scaling enabled
- [x] Pod Disruption Budget for HA
- [x] Model versioning implemented
- [x] Persistent storage for models
- [x] Monitoring and metrics exposed
- [x] A/B testing capability
- [x] Rollback procedure documented
- [ ] Alert rules configured (requires Prometheus)
- [ ] Backup and disaster recovery plan
- [ ] Cost monitoring enabled

**Key Commands**:
```bash
# Model Serving
kubectl get deployments -n ml-workloads -l component=model-server
kubectl scale deployment model-v1 -n ml-workloads --replicas=5
kubectl get hpa -n ml-workloads

# Training
kubectl get jobs -n ml-workloads
kubectl logs job/model-training-simple -n ml-workloads
kubectl create job --from=cronjob/scheduled-model-training manual-run -n ml-workloads

# A/B Testing
kubectl patch ingress model-canary -n ml-workloads \
  -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/canary-weight":"50"}}}'

# Monitoring
kubectl top pods -n ml-workloads
kubectl describe resourcequota ml-quota -n ml-workloads
```

**Next Steps**:
- Integrate with Kubeflow for ML pipelines
- Add model monitoring and drift detection
- Implement feature stores (Feast, Tecton)
- Set up experiment tracking (MLflow, Weights & Biases)
- Explore model optimization (ONNX, TensorRT)
- Implement shadow testing for safer deployments
- Add cost optimization strategies

---

**Congratulations!** You've completed Module 006: Kubernetes Introduction. You now have production-ready skills in deploying ML workloads on Kubernetes with serving, training, A/B testing, and auto-scaling.
