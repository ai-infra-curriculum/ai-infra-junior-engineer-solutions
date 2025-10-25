# Exercise 03: Model Deployment

## Overview

Deploy machine learning models to production with robust serving infrastructure, API development, containerization, monitoring, and scaling strategies. Learn to build production-ready ML inference services that handle real-world traffic at scale.

## Learning Objectives

- ✅ Build RESTful APIs for model inference
- ✅ Implement model serving with various frameworks
- ✅ Containerize ML applications with Docker
- ✅ Deploy models to production environments
- ✅ Monitor deployed models in production
- ✅ Implement A/B testing and canary deployments
- ✅ Handle model versioning and rollbacks
- ✅ Optimize inference performance

## Topics Covered

### 1. Model Serving Fundamentals

#### Model Serialization

```python
import torch
import pickle
import joblib
from pathlib import Path

class ModelSerializer:
    """Serialize and deserialize ML models"""

    @staticmethod
    def save_pytorch_model(model: torch.nn.Module, path: str):
        """Save PyTorch model"""
        # Save full model
        torch.save(model, path)

        # Or save state dict (preferred)
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_pytorch_model(path: str, model_class=None):
        """Load PyTorch model"""
        # Load full model
        model = torch.load(path, map_location='cpu')

        # Or load state dict
        if model_class:
            model = model_class()
            model.load_state_dict(torch.load(path, map_location='cpu'))

        model.eval()  # Set to evaluation mode
        return model

    @staticmethod
    def save_sklearn_model(model, path: str):
        """Save scikit-learn model"""
        joblib.dump(model, path)

    @staticmethod
    def load_sklearn_model(path: str):
        """Load scikit-learn model"""
        return joblib.load(path)

    @staticmethod
    def export_onnx(model: torch.nn.Module, dummy_input: torch.Tensor,
                    output_path: str):
        """Export PyTorch model to ONNX format"""
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

# Usage
model = MyModel()
ModelSerializer.save_pytorch_model(model, 'model.pt')

# Export to ONNX for production serving
dummy_input = torch.randn(1, 3, 224, 224)
ModelSerializer.export_onnx(model, dummy_input, 'model.onnx')
```

#### Model Loading and Caching

```python
from typing import Dict, Optional
import threading

class ModelRegistry:
    """Thread-safe model registry with caching"""

    def __init__(self):
        self._models: Dict[str, any] = {}
        self._lock = threading.Lock()

    def register(self, name: str, model: any, version: str = 'latest'):
        """Register a model"""
        key = f"{name}:{version}"
        with self._lock:
            self._models[key] = {
                'model': model,
                'loaded_at': datetime.now(),
                'version': version
            }

    def get(self, name: str, version: str = 'latest') -> Optional[any]:
        """Get model from registry"""
        key = f"{name}:{version}"
        with self._lock:
            if key in self._models:
                return self._models[key]['model']
        return None

    def unload(self, name: str, version: str = 'latest'):
        """Unload model from memory"""
        key = f"{name}:{version}"
        with self._lock:
            if key in self._models:
                del self._models[key]

    def list_models(self) -> List[str]:
        """List all registered models"""
        with self._lock:
            return list(self._models.keys())

# Global model registry
model_registry = ModelRegistry()

# Register models at startup
model = load_model('model.pt')
model_registry.register('my_model', model, version='v1.0')
```

### 2. Building Inference APIs

#### FastAPI Model Serving

```python
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
from PIL import Image
import io

app = FastAPI(
    title="ML Model API",
    description="Production ML model serving API",
    version="1.0.0"
)

# Request/Response models
class PredictionRequest(BaseModel):
    """Prediction request schema"""
    data: List[List[float]]
    model_version: Optional[str] = 'latest'

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    predictions: List[float]
    model_version: str
    inference_time_ms: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    gpu_available: bool

# Load model at startup
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global model
    model = torch.load('model.pt', map_location='cpu')
    model.eval()
    print("Model loaded successfully")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version="v1.0",
        gpu_available=torch.cuda.is_available()
    )

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions"""
    try:
        start_time = time.time()

        # Convert input to tensor
        input_tensor = torch.tensor(request.data, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            predictions = output.tolist()

        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # ms

        return PredictionResponse(
            predictions=predictions,
            model_version=request.model_version,
            inference_time_ms=inference_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Image prediction endpoint
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Predict from uploaded image"""
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        input_tensor = preprocess(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            predictions = output.softmax(dim=1).tolist()[0]

        # Get top 5 predictions
        top5 = sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)[:5]

        return {
            "predictions": [
                {"class_id": idx, "probability": prob}
                for idx, prob in top5
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    results = []

    for req in requests:
        result = await predict(req)
        results.append(result)

    return {"results": results}

# Model metadata endpoint
@app.get("/model/info")
async def model_info():
    """Get model metadata"""
    return {
        "model_name": "my_model",
        "version": "v1.0",
        "framework": "pytorch",
        "input_shape": [None, 3, 224, 224],
        "output_shape": [None, 1000],
        "created_at": "2024-01-01T00:00:00Z"
    }

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Flask Model Serving (Alternative)

```python
from flask import Flask, request, jsonify
import torch
import numpy as np

app = Flask(__name__)

# Load model
model = None

@app.before_first_request
def load_model():
    """Load model before first request"""
    global model
    model = torch.load('model.pt', map_location='cpu')
    model.eval()

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction"""
    try:
        data = request.get_json()
        input_tensor = torch.tensor(data['input'], dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            predictions = output.tolist()

        return jsonify({
            'predictions': predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### 3. Model Serving Frameworks

#### TorchServe

```python
# model_handler.py - Custom TorchServe handler

import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler

class MyModelHandler(BaseHandler):
    """Custom TorchServe handler"""

    def initialize(self, context):
        """Initialize handler"""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Load model
        self.model = torch.jit.load(f"{model_dir}/model.pt")
        self.model.eval()

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model.to(self.device)

        self.initialized = True

    def preprocess(self, data):
        """Preprocess input data"""
        # Extract input from request
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # Decode base64
                image = base64.b64decode(image)

            # Convert to PIL Image
            image = Image.open(io.BytesIO(image))

            # Preprocess
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            images.append(preprocess(image))

        # Batch images
        return torch.stack(images).to(self.device)

    def inference(self, data):
        """Run inference"""
        with torch.no_grad():
            output = self.model(data)
        return output

    def postprocess(self, data):
        """Postprocess output"""
        # Apply softmax
        probabilities = F.softmax(data, dim=1)

        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)

        # Convert to list
        results = []
        for i in range(top5_prob.shape[0]):
            results.append({
                'predictions': [
                    {'class_id': int(idx), 'probability': float(prob)}
                    for idx, prob in zip(top5_idx[i], top5_prob[i])
                ]
            })

        return results
```

```bash
# Package model for TorchServe
torch-model-archiver \
    --model-name my_model \
    --version 1.0 \
    --model-file model.py \
    --serialized-file model.pt \
    --handler model_handler.py \
    --export-path model_store/

# Start TorchServe
torchserve \
    --start \
    --model-store model_store \
    --models my_model=my_model.mar \
    --ncs

# Make prediction
curl -X POST http://localhost:8080/predictions/my_model \
    -T input_image.jpg
```

#### TensorFlow Serving

```python
# Export SavedModel for TensorFlow Serving
import tensorflow as tf

# Create model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Export
tf.saved_model.save(model, 'models/resnet50/1')
```

```bash
# Start TensorFlow Serving
docker run -p 8501:8501 \
    --mount type=bind,source=$(pwd)/models/resnet50,target=/models/resnet50 \
    -e MODEL_NAME=resnet50 \
    -t tensorflow/serving

# Make prediction
curl -X POST http://localhost:8501/v1/models/resnet50:predict \
    -d @input.json
```

### 4. Containerization with Docker

#### Dockerfile for ML Application

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download model (or mount as volume)
# COPY models/ /app/models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### requirements.txt

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
torchvision==0.16.0
pillow==10.1.0
numpy==1.26.0
pydantic==2.4.2
python-multipart==0.0.6
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  model-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.pt
      - WORKERS=4
      - LOG_LEVEL=info
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    restart: unless-stopped

volumes:
  redis-data:
  prometheus-data:
```

#### Build and Run

```bash
# Build image
docker build -t ml-model-api:latest .

# Run container
docker run -d \
    --name model-api \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    ml-model-api:latest

# Or use docker-compose
docker-compose up -d

# Check logs
docker logs model-api

# Test API
curl http://localhost:8000/health
```

### 5. Production Deployment

#### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-api
  labels:
    app: ml-model-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model-api
  template:
    metadata:
      labels:
        app: ml-model-api
    spec:
      containers:
      - name: api
        image: ml-model-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        env:
        - name: MODEL_PATH
          value: "/models/model.pt"
        - name: WORKERS
          value: "4"
        volumeMounts:
        - name: models
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-api
spec:
  selector:
    app: ml-model-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-api
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
```

```bash
# Apply deployment
kubectl apply -f deployment.yaml

# Check status
kubectl get pods
kubectl get svc

# Scale deployment
kubectl scale deployment ml-model-api --replicas=5
```

### 6. Model Monitoring

#### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import time

app = FastAPI()

# Define metrics
prediction_counter = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_version', 'status']
)

prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_version'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

model_load_gauge = Gauge(
    'model_loaded',
    'Whether model is loaded',
    ['model_version']
)

gpu_memory_usage = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_id']
)

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make prediction with metrics"""
    start_time = time.time()

    try:
        # Make prediction
        result = model.predict(request.data)

        # Record success
        prediction_counter.labels(
            model_version='v1.0',
            status='success'
        ).inc()

        # Record latency
        latency = time.time() - start_time
        prediction_latency.labels(model_version='v1.0').observe(latency)

        return {"predictions": result}

    except Exception as e:
        # Record failure
        prediction_counter.labels(
            model_version='v1.0',
            status='error'
        ).inc()
        raise

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # Update GPU metrics
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i)
            gpu_memory_usage.labels(gpu_id=str(i)).set(memory_allocated)

    return PlainTextResponse(generate_latest())
```

#### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "ML Model Monitoring",
    "panels": [
      {
        "title": "Predictions per Second",
        "targets": [
          {
            "expr": "rate(model_predictions_total[1m])"
          }
        ]
      },
      {
        "title": "Prediction Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, model_prediction_latency_seconds)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(model_predictions_total{status=\"error\"}[1m])"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "gpu_memory_usage_bytes"
          }
        ]
      }
    ]
  }
}
```

### 7. A/B Testing and Canary Deployments

#### A/B Testing Implementation

```python
import random
from typing import Optional

class ABTester:
    """A/B testing for model versions"""

    def __init__(self, models: Dict[str, any], traffic_split: Dict[str, float]):
        """
        Initialize A/B tester

        Args:
            models: Dictionary of model versions
            traffic_split: Traffic split percentages (must sum to 1.0)
        """
        self.models = models
        self.traffic_split = traffic_split

        # Validate traffic split
        assert abs(sum(traffic_split.values()) - 1.0) < 0.001

    def select_model(self, user_id: Optional[str] = None) -> Tuple[str, any]:
        """
        Select model version based on traffic split

        Args:
            user_id: Optional user ID for consistent routing

        Returns:
            Tuple of (version, model)
        """
        if user_id:
            # Consistent hashing for same user
            import hashlib
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            rand_val = (hash_val % 10000) / 10000
        else:
            # Random selection
            rand_val = random.random()

        # Select model based on cumulative probability
        cumulative = 0
        for version, probability in self.traffic_split.items():
            cumulative += probability
            if rand_val <= cumulative:
                return version, self.models[version]

        # Fallback (shouldn't reach here)
        return list(self.models.items())[0]

# Usage
ab_tester = ABTester(
    models={
        'v1.0': model_v1,
        'v2.0': model_v2
    },
    traffic_split={
        'v1.0': 0.9,  # 90% traffic
        'v2.0': 0.1   # 10% traffic
    }
)

@app.post("/predict")
async def predict(request: PredictionRequest, user_id: str = None):
    """Predict with A/B testing"""
    # Select model version
    version, model = ab_tester.select_model(user_id)

    # Make prediction
    result = model.predict(request.data)

    # Log version used
    prediction_counter.labels(model_version=version, status='success').inc()

    return {
        "predictions": result,
        "model_version": version
    }
```

#### Canary Deployment with Kubernetes

```yaml
# canary-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-api-stable
spec:
  replicas: 9
  selector:
    matchLabels:
      app: ml-model-api
      version: stable
  template:
    metadata:
      labels:
        app: ml-model-api
        version: stable
    spec:
      containers:
      - name: api
        image: ml-model-api:v1.0
        # ... container spec

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-api-canary
spec:
  replicas: 1  # 10% traffic
  selector:
    matchLabels:
      app: ml-model-api
      version: canary
  template:
    metadata:
      labels:
        app: ml-model-api
        version: canary
    spec:
      containers:
      - name: api
        image: ml-model-api:v2.0
        # ... container spec

---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-api
spec:
  selector:
    app: ml-model-api  # Routes to both stable and canary
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

### 8. Performance Optimization

#### Batch Inference

```python
import asyncio
from typing import List
from collections import deque
import time

class BatchPredictor:
    """Batch inference with dynamic batching"""

    def __init__(self, model, batch_size: int = 32, max_wait_ms: int = 100):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms

        self.queue = deque()
        self.processing = False

    async def predict(self, input_data):
        """Add request to queue and wait for result"""
        # Create future for this request
        future = asyncio.Future()

        # Add to queue
        self.queue.append((input_data, future))

        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_batch())

        # Wait for result
        return await future

    async def _process_batch(self):
        """Process batch of requests"""
        self.processing = True

        while len(self.queue) > 0:
            # Wait for batch to fill or timeout
            start_time = time.time()
            while len(self.queue) < self.batch_size:
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms >= self.max_wait_ms:
                    break
                await asyncio.sleep(0.001)

            # Get batch
            batch = []
            futures = []
            for _ in range(min(self.batch_size, len(self.queue))):
                if self.queue:
                    input_data, future = self.queue.popleft()
                    batch.append(input_data)
                    futures.append(future)

            if not batch:
                break

            # Process batch
            try:
                batch_tensor = torch.stack(batch)
                with torch.no_grad():
                    outputs = self.model(batch_tensor)

                # Set results
                for i, future in enumerate(futures):
                    future.set_result(outputs[i])

            except Exception as e:
                # Set exception for all futures
                for future in futures:
                    future.set_exception(e)

        self.processing = False

# Usage
batch_predictor = BatchPredictor(model, batch_size=32, max_wait_ms=100)

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict with batching"""
    input_tensor = torch.tensor(request.data)
    result = await batch_predictor.predict(input_tensor)
    return {"predictions": result.tolist()}
```

---

## Project: Production Model Deployment

Deploy a complete ML inference service with monitoring, scaling, and A/B testing.

### Requirements

**Deployment Components:**
1. RESTful API for model inference
2. Docker containerization
3. Kubernetes deployment configuration
4. Prometheus monitoring
5. A/B testing implementation
6. Load testing and benchmarking

**Features:**
- Multi-model serving
- Batch inference support
- Health checks and readiness probes
- Metrics and logging
- Horizontal auto-scaling
- Canary deployments

### Implementation

See `solutions/` directory for complete implementations:

1. **`model_api.py`** - FastAPI-based model serving API
2. **`deploy_model.sh`** - Automated deployment script
3. **`load_test.py`** - Load testing tool for deployed models
4. **`monitor_deployment.py`** - Deployment health monitoring

---

## Practice Problems

### Problem 1: Multi-Model API

Create an API that:
- Serves multiple model versions simultaneously
- Routes requests based on model version
- Implements model warming and caching
- Provides version comparison endpoint
- Generates performance reports

### Problem 2: Model Performance Optimizer

Build a tool that:
- Benchmarks different serving configurations
- Tests batch sizes and worker counts
- Measures latency and throughput
- Recommends optimal settings
- Generates performance report

### Problem 3: Deployment Monitor

Create a monitoring system that:
- Tracks deployment health
- Monitors prediction latency
- Detects anomalies in predictions
- Alerts on performance degradation
- Provides dashboard visualization

---

## Best Practices

### 1. API Design

```python
# Use versioned APIs
@app.post("/v1/predict")
async def predict_v1():
    pass

@app.post("/v2/predict")
async def predict_v2():
    pass

# Implement rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict():
    pass

# Add request validation
class PredictionRequest(BaseModel):
    data: List[float]

    @validator('data')
    def validate_data(cls, v):
        if len(v) == 0:
            raise ValueError("Data cannot be empty")
        return v
```

### 2. Error Handling

```python
from fastapi import HTTPException

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        result = model.predict(request.data)
        return {"predictions": result}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 3. Model Versioning

```python
# Include version in response
@app.post("/predict")
async def predict(request: PredictionRequest):
    result = model.predict(request.data)
    return {
        "predictions": result,
        "model_version": "v1.0",
        "api_version": "v1",
        "timestamp": datetime.now().isoformat()
    }
```

---

## Validation

Test your deployment:

```bash
# Build and run locally
docker build -t ml-model-api .
docker run -p 8000:8000 ml-model-api

# Test API
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"data": [[1.0, 2.0, 3.0]]}'

# Deploy to Kubernetes
kubectl apply -f deployment.yaml

# Run load test
python solutions/load_test.py --url http://localhost:8000/predict --requests 1000

# Monitor deployment
python solutions/monitor_deployment.py --namespace default
```

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TorchServe](https://pytorch.org/serve/)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Monitoring](https://prometheus.io/docs/)

---

## Next Steps

1. **Module 005: Docker & Containerization** - Deep dive into containers
2. Learn Kubernetes operators for ML
3. Implement blue-green deployments
4. Study model optimization techniques
5. Explore edge deployment strategies

---

**Deploy ML models with confidence! 🚀**
