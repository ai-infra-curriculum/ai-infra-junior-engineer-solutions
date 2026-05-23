# Step-by-Step Guide: Model Serving Frameworks

## Overview
Implement production model serving using BentoML, TorchServe, and TensorFlow Serving to compare different frameworks for deploying ML models at scale.

## Phase 1: BentoML Setup and Service (15 minutes)

### Install BentoML
```bash
# Create project directory
mkdir -p model-serving
cd model-serving

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install BentoML
pip install bentoml scikit-learn pandas
```

### Train and Save Model
Create `train_model.py`:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import bentoml

# Train model
iris = load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save with BentoML
saved_model = bentoml.sklearn.save_model(
    "iris_classifier",
    model,
    metadata={
        "accuracy": model.score(X, y),
        "features": iris.feature_names
    }
)

print(f"Model saved: {saved_model}")
```

Run: `python train_model.py`

### Create BentoML Service
Create `service.py`:
```python
import bentoml
import numpy as np
from bentoml.io import JSON, NumpyNdarray

# Load the latest model
iris_runner = bentoml.sklearn.get("iris_classifier:latest").to_runner()

# Create service
svc = bentoml.Service("iris_classifier", runners=[iris_runner])

@svc.api(input=NumpyNdarray(), output=JSON())
async def classify(input_array: np.ndarray) -> dict:
    """Classify iris species"""
    result = await iris_runner.predict.async_run(input_array)
    return {"prediction": int(result[0])}

@svc.api(input=JSON(), output=JSON())
async def classify_json(input_data: dict) -> dict:
    """Classify from JSON input"""
    features = np.array(input_data["features"]).reshape(1, -1)
    result = await iris_runner.predict.async_run(features)
    proba = await iris_runner.predict_proba.async_run(features)

    return {
        "prediction": int(result[0]),
        "probabilities": proba[0].tolist(),
        "species": ["setosa", "versicolor", "virginica"][result[0]]
    }
```

### Run BentoML Service
```bash
# Serve locally
bentoml serve service:svc --reload

# Test in another terminal
curl -X POST http://localhost:3000/classify_json \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**Validation**: Verify prediction response with probabilities.

## Phase 2: Build and Deploy Bento (15 minutes)

### Create bentofile.yaml
Create `bentofile.yaml`:
```yaml
service: "service:svc"
description: "Iris classification service"
labels:
  owner: ml-team
  project: iris-classifier
include:
  - "*.py"
python:
  packages:
    - scikit-learn
    - pandas
    - numpy
docker:
  distro: debian
  python_version: "3.11"
  cuda_version: null
```

### Build Bento
```bash
# Build the bento
bentoml build

# List bentos
bentoml list

# View bento details
bentoml get iris_classifier:latest
```

### Containerize with Docker
```bash
# Build Docker image
bentoml containerize iris_classifier:latest

# Run container
docker run -p 3000:3000 iris_classifier:latest

# Test containerized service
curl -X POST http://localhost:3000/classify_json \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**Validation**: Service runs successfully in container.

## Phase 3: TorchServe Setup (15 minutes)

### Install TorchServe
```bash
# Install PyTorch and TorchServe
pip install torch torchvision torchserve torch-model-archiver
```

### Create PyTorch Model
Create `pytorch_model.py`:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
import numpy as np

# Simple neural network
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train model
iris = load_iris()
X = torch.FloatTensor(iris.data)
y = torch.LongTensor(iris.target)

model = IrisNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Save model
torch.save(model.state_dict(), 'iris_model.pth')
print("PyTorch model saved")
```

Run: `python pytorch_model.py`

### Create TorchServe Handler
Create `handler.py`:
```python
import torch
import torch.nn as nn
import json
from ts.torch_handler.base_handler import BaseHandler

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class IrisClassifierHandler(BaseHandler):
    def __init__(self):
        super(IrisClassifierHandler, self).__init__()
        self.model = None

    def initialize(self, context):
        """Initialize model"""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Load model
        self.model = IrisNet()
        self.model.load_state_dict(
            torch.load(f"{model_dir}/iris_model.pth", map_location="cpu")
        )
        self.model.eval()

    def preprocess(self, data):
        """Preprocess input data"""
        preprocessed_data = []
        for row in data:
            input_data = row.get("data") or row.get("body")
            if isinstance(input_data, (bytes, bytearray)):
                input_data = input_data.decode('utf-8')
            input_data = json.loads(input_data)
            preprocessed_data.append(input_data["features"])

        return torch.FloatTensor(preprocessed_data)

    def inference(self, data):
        """Run inference"""
        with torch.no_grad():
            outputs = self.model(data)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities

    def postprocess(self, inference_output):
        """Format output"""
        species = ["setosa", "versicolor", "virginica"]
        results = []

        for probs in inference_output:
            prediction = torch.argmax(probs).item()
            results.append({
                "prediction": prediction,
                "species": species[prediction],
                "probabilities": probs.tolist()
            })

        return results

_service = IrisClassifierHandler()

def handle(data, context):
    return _service.handle(data, context)
```

### Package and Serve Model
```bash
# Create model archive
torch-model-archiver \
  --model-name iris_classifier \
  --version 1.0 \
  --model-file pytorch_model.py \
  --serialized-file iris_model.pth \
  --handler handler.py \
  --export-path model-store

# Start TorchServe
mkdir -p model-store logs
torchserve --start \
  --model-store model-store \
  --models iris_classifier=iris_classifier.mar \
  --ncs

# Test inference
curl -X POST http://localhost:8080/predictions/iris_classifier \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**Validation**: Verify TorchServe returns predictions.

## Phase 4: TensorFlow Serving (15 minutes)

### Install TensorFlow Serving
```bash
# Install TensorFlow
pip install tensorflow

# Pull TensorFlow Serving Docker image
docker pull tensorflow/serving
```

### Create and Save TensorFlow Model
Create `tf_model.py`:
```python
import tensorflow as tf
from sklearn.datasets import load_iris
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X, y, epochs=50, verbose=0)

# Save in TensorFlow SavedModel format
model_version = 1
export_path = f'./tf_models/iris_model/{model_version}'
tf.saved_model.save(model, export_path)

print(f"Model saved to {export_path}")
```

Run: `python tf_model.py`

### Serve with TensorFlow Serving
```bash
# Start TensorFlow Serving with Docker
docker run -p 8501:8501 \
  --mount type=bind,source=$(pwd)/tf_models/iris_model,target=/models/iris_model \
  -e MODEL_NAME=iris_model \
  -t tensorflow/serving

# Test REST API
curl -X POST http://localhost:8501/v1/models/iris_model:predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [[5.1, 3.5, 1.4, 0.2]]
  }'
```

**Validation**: Verify predictions from TensorFlow Serving.

## Phase 5: Performance Comparison (15 minutes)

### Create Benchmark Script
Create `benchmark.py`:
```python
import requests
import time
import statistics
import concurrent.futures

def benchmark_endpoint(url, data, num_requests=100):
    """Benchmark an endpoint"""
    latencies = []

    def make_request():
        start = time.time()
        response = requests.post(url, json=data)
        latency = time.time() - start
        return latency, response.status_code

    # Sequential requests
    print(f"\nBenchmarking {url}")
    print("Sequential requests...")
    for _ in range(num_requests):
        latency, status = make_request()
        if status == 200:
            latencies.append(latency)

    # Concurrent requests
    print("Concurrent requests...")
    concurrent_latencies = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        for future in concurrent.futures.as_completed(futures):
            latency, status = future.result()
            if status == 200:
                concurrent_latencies.append(latency)

    # Calculate statistics
    print(f"\nResults for {url}:")
    print(f"  Sequential - Mean: {statistics.mean(latencies)*1000:.2f}ms, "
          f"P95: {statistics.quantiles(latencies, n=20)[18]*1000:.2f}ms")
    print(f"  Concurrent - Mean: {statistics.mean(concurrent_latencies)*1000:.2f}ms, "
          f"P95: {statistics.quantiles(concurrent_latencies, n=20)[18]*1000:.2f}ms")

    return latencies, concurrent_latencies

# Test data
test_data = {"features": [5.1, 3.5, 1.4, 0.2]}

# Benchmark BentoML
bentoml_latencies = benchmark_endpoint(
    "http://localhost:3000/classify_json",
    test_data
)

# Benchmark TorchServe
torchserve_latencies = benchmark_endpoint(
    "http://localhost:8080/predictions/iris_classifier",
    test_data
)

# Benchmark TensorFlow Serving
tf_data = {"instances": [[5.1, 3.5, 1.4, 0.2]]}
tf_latencies = benchmark_endpoint(
    "http://localhost:8501/v1/models/iris_model:predict",
    tf_data
)
```

**Validation**: Compare latency and throughput across frameworks.

## Phase 6: Production Deployment (10 minutes)

### Create Kubernetes Deployment for BentoML
Create `k8s/bentoml-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-bentoml
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iris-bentoml
  template:
    metadata:
      labels:
        app: iris-bentoml
    spec:
      containers:
      - name: bentoml
        image: iris_classifier:latest
        ports:
        - containerPort: 3000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: iris-bentoml-service
spec:
  type: LoadBalancer
  selector:
    app: iris-bentoml
  ports:
  - port: 80
    targetPort: 3000
```

### Create Docker Compose for Multi-Framework
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  bentoml:
    image: iris_classifier:latest
    ports:
      - "3000:3000"

  torchserve:
    image: pytorch/torchserve:latest
    ports:
      - "8080:8080"
      - "8081:8081"
    volumes:
      - ./model-store:/home/model-server/model-store

  tensorflow-serving:
    image: tensorflow/serving
    ports:
      - "8501:8501"
    environment:
      - MODEL_NAME=iris_model
    volumes:
      - ./tf_models/iris_model:/models/iris_model
```

**Validation**: Deploy all frameworks and verify they're accessible.

## Summary

You've implemented model serving with three major frameworks:
- **BentoML**: Easy Python-native serving with automatic API generation and containerization
- **TorchServe**: Production-grade PyTorch model serving with custom handlers and model management
- **TensorFlow Serving**: High-performance TF model serving with gRPC and REST APIs
- **Performance benchmarking**: Comparing latency and throughput across frameworks
- **Production deployment**: Kubernetes and Docker Compose configurations

Each framework has strengths: BentoML for rapid prototyping, TorchServe for PyTorch production, and TensorFlow Serving for TensorFlow at scale.
