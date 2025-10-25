# Exercise 02: ML Model Serving Frameworks

## Overview

Implement production model serving using industry-standard frameworks: TorchServe, TensorFlow Serving, BentoML, and ONNX Runtime. Compare performance, features, and use cases for each framework.

## Learning Objectives

- Deploy PyTorch models with TorchServe
- Serve TensorFlow models with TF Serving
- Use BentoML for multi-framework serving
- Optimize models with ONNX Runtime
- Implement model versioning and A/B testing
- Benchmark serving performance
- Choose appropriate serving framework for use cases

## Prerequisites

- Completed Exercise 01 (FastAPI Fundamentals)
- Python 3.8+
- Docker installed
- Basic understanding of PyTorch and TensorFlow
- CUDA-capable GPU (optional, for GPU serving)

## Project Structure

```
exercise-02-model-serving/
├── torchserve/
│   ├── models/                      # Model artifacts
│   ├── model_store/                 # TorchServe model store
│   ├── handlers/
│   │   └── custom_handler.py       # Custom TorchServe handler
│   ├── config.properties            # TorchServe configuration
│   ├── Dockerfile
│   └── README.md
├── tfserving/
│   ├── models/                      # SavedModel format
│   │   └── model_name/
│   │       └── 1/                   # Version 1
│   ├── client.py                    # TF Serving client
│   ├── docker-compose.yml
│   └── README.md
├── bentoml/
│   ├── service.py                   # BentoML service definition
│   ├── train_and_save.py           # Train and save model
│   ├── bentofile.yaml              # Bento build config
│   └── README.md
├── onnx/
│   ├── convert_to_onnx.py          # Model conversion
│   ├── onnx_inference.py           # ONNX Runtime inference
│   ├── optimize.py                 # Model optimization
│   └── README.md
├── benchmarks/
│   ├── benchmark_all.py            # Compare all frameworks
│   ├── load_test.py                # Load testing
│   ├── latency_test.py             # Latency benchmarks
│   └── results/                    # Benchmark results
└── README.md
```

## Quick Start

### 1. TorchServe

```bash
# Train and archive model
cd torchserve
python train_model.py
torch-model-archiver --model-name my_model \
  --version 1.0 \
  --model-file model.py \
  --serialized-file model.pth \
  --handler handlers/custom_handler.py \
  --export-path model_store

# Start TorchServe
torchserve --start --model-store model_store \
  --models my_model=my_model.mar

# Test inference
curl -X POST http://localhost:8080/predictions/my_model \
  -H "Content-Type: application/json" \
  -d '{"data": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'
```

### 2. TensorFlow Serving

```bash
# Export SavedModel
cd tfserving
python export_model.py

# Start TF Serving with Docker
docker run -p 8501:8501 \
  -v $(pwd)/models:/models \
  -e MODEL_NAME=my_model \
  tensorflow/serving

# Test inference
curl -X POST http://localhost:8501/v1/models/my_model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'
```

### 3. BentoML

```bash
# Train and save model
cd bentoml
python train_and_save.py

# Build Bento
bentoml build

# Serve locally
bentoml serve service:svc --reload

# Test inference
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0]}'
```

### 4. ONNX Runtime

```bash
# Convert model to ONNX
cd onnx
python convert_to_onnx.py

# Run inference
python onnx_inference.py
```

## Framework Comparison

### TorchServe

**Pros:**
- ✅ Official PyTorch serving solution
- ✅ Built-in model versioning
- ✅ A/B testing support
- ✅ Auto-scaling capabilities
- ✅ Metrics and logging

**Cons:**
- ❌ PyTorch-only
- ❌ More complex setup
- ❌ Learning curve for custom handlers

**Best For:**
- PyTorch model deployment
- Multi-model serving
- Production PyTorch applications

**Performance:**
- Throughput: 🟢 High
- Latency: 🟡 Medium
- Resource Usage: 🟡 Medium

### TensorFlow Serving

**Pros:**
- ✅ Mature and battle-tested
- ✅ High performance
- ✅ gRPC and REST APIs
- ✅ Model versioning
- ✅ Batching support

**Cons:**
- ❌ TensorFlow-only
- ❌ Less flexible than custom solutions
- ❌ Configuration complexity

**Best For:**
- TensorFlow model deployment
- High-throughput serving
- Production TensorFlow applications

**Performance:**
- Throughput: 🟢 Very High
- Latency: 🟢 Low
- Resource Usage: 🟢 Low

### BentoML

**Pros:**
- ✅ Framework-agnostic (PyTorch, TF, scikit-learn, etc.)
- ✅ Simple Python-first API
- ✅ Built-in containerization
- ✅ Model registry integration
- ✅ Easy deployment to cloud

**Cons:**
- ❌ Newer, less battle-tested
- ❌ Community smaller than TF Serving

**Best For:**
- Multi-framework projects
- Rapid prototyping
- Python-centric teams
- Cloud deployments

**Performance:**
- Throughput: 🟢 High
- Latency: 🟡 Medium
- Resource Usage: 🟡 Medium

### ONNX Runtime

**Pros:**
- ✅ Framework-agnostic
- ✅ Hardware acceleration (CPU, GPU, NPU)
- ✅ Excellent performance
- ✅ Cross-platform
- ✅ Small footprint

**Cons:**
- ❌ Conversion required
- ❌ Not all ops supported
- ❌ No built-in serving (needs wrapper)

**Best For:**
- Cross-platform deployment
- Performance optimization
- Edge deployment
- Vendor-agnostic inference

**Performance:**
- Throughput: 🟢 Very High
- Latency: 🟢 Very Low
- Resource Usage: 🟢 Very Low

## TorchServe Implementation

### Custom Handler

```python
# handlers/custom_handler.py
import torch
from ts.torch_handler.base_handler import BaseHandler

class CustomClassifier(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        """Initialize model."""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Load model
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.model = torch.jit.load(model_pt_path)
        self.model.eval()

        self.initialized = True

    def preprocess(self, requests):
        """Preprocess input data."""
        inputs = []
        for request in requests:
            data = request.get("data") or request.get("body")
            if isinstance(data, str):
                data = json.loads(data)
            inputs.append(torch.tensor(data))
        return torch.stack(inputs)

    def inference(self, data):
        """Run inference."""
        with torch.no_grad():
            predictions = self.model(data)
        return predictions

    def postprocess(self, inference_output):
        """Postprocess predictions."""
        predictions = inference_output.cpu().numpy()
        return predictions.tolist()
```

### Configuration

```properties
# config.properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082

number_of_netty_threads=32
job_queue_size=1000
model_store=/models/model_store

# Metrics
enable_metrics_api=true
metrics_format=prometheus

# Logging
default_response_timeout=120
```

### Docker Deployment

```dockerfile
FROM pytorch/torchserve:latest

# Copy model store
COPY model_store /models/model_store

# Copy configuration
COPY config.properties /home/model-server/config.properties

# Expose ports
EXPOSE 8080 8081 8082

CMD ["torchserve", \
     "--start", \
     "--model-store", "/models/model_store", \
     "--models", "all"]
```

## TensorFlow Serving Implementation

### Export SavedModel

```python
# export_model.py
import tensorflow as tf

# Train model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train on data
model.fit(X_train, y_train, epochs=10)

# Export SavedModel
export_path = 'models/my_model/1'
tf.saved_model.save(model, export_path)

print(f"Model exported to {export_path}")
```

### Client

```python
# client.py
import requests
import numpy as np

def predict(data):
    """Make prediction using TF Serving."""
    url = 'http://localhost:8501/v1/models/my_model:predict'

    payload = {
        "instances": data.tolist()
    }

    response = requests.post(url, json=payload)
    predictions = response.json()['predictions']

    return np.array(predictions)

# Example
data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
result = predict(data)
print(f"Prediction: {result}")
```

### Docker Compose

```yaml
version: '3.8'

services:
  tfserving:
    image: tensorflow/serving:latest
    ports:
      - "8501:8501"  # REST API
      - "8500:8500"  # gRPC API
    volumes:
      - ./models:/models
    environment:
      - MODEL_NAME=my_model
      - MODEL_BASE_PATH=/models/my_model
    command:
      - "--rest_api_port=8501"
      - "--model_config_file=/models/models.config"
```

## BentoML Implementation

### Service Definition

```python
# service.py
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, JSON

# Load model
model = bentoml.sklearn.get("my_model:latest")

# Create service
svc = bentoml.Service("my_model_service", runners=[model.to_runner()])

@svc.api(input=NumpyNdarray(), output=JSON())
async def predict(input_array: np.ndarray) -> dict:
    """Prediction endpoint."""
    result = await model.async_run(input_array)

    return {
        "prediction": int(result[0]),
        "confidence": float(max(result[1]))
    }

@svc.api(input=JSON(), output=JSON())
async def predict_json(input_data: dict) -> dict:
    """JSON prediction endpoint."""
    features = np.array([input_data["features"]])
    result = await model.async_run(features)

    return {
        "prediction": int(result[0]),
        "probability": result[1].tolist()
    }
```

### Train and Save

```python
# train_and_save.py
import bentoml
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train model
X = np.random.rand(1000, 5)
y = np.random.randint(0, 2, 1000)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save with BentoML
bentoml.sklearn.save_model(
    "my_model",
    model,
    signatures={
        "predict": {
            "batchable": True,
            "batch_dim": 0
        }
    },
    labels={
        "framework": "sklearn",
        "task": "classification"
    },
    metadata={
        "accuracy": 0.95,
        "trained_on": "2025-10-24"
    }
)

print("Model saved to BentoML model store")
```

### Bentofile

```yaml
# bentofile.yaml
service: "service:svc"
labels:
  owner: ml-team
  stage: production
include:
  - "*.py"
  - "requirements.txt"
python:
  packages:
    - scikit-learn==1.3.2
    - numpy==1.26.2
docker:
  distro: debian
  python_version: "3.10"
  cuda_version: "11.8"
```

## ONNX Runtime Implementation

### Model Conversion

```python
# convert_to_onnx.py
import torch
import torch.onnx
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# PyTorch to ONNX
def convert_pytorch_to_onnx():
    model = torch.load('model.pth')
    model.eval()

    dummy_input = torch.randn(1, 5)

    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

# Scikit-learn to ONNX
def convert_sklearn_to_onnx():
    model = RandomForestClassifier()
    # ... train model ...

    initial_type = [('float_input', FloatTensorType([None, 5]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=13
    )

    with open("model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
```

### ONNX Inference

```python
# onnx_inference.py
import onnxruntime as ort
import numpy as np

class ONNXPredictor:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, data):
        """Run inference."""
        input_data = np.array(data, dtype=np.float32)

        result = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )

        return result[0]

# Usage
predictor = ONNXPredictor('model.onnx')
result = predictor.predict([[1.0, 2.0, 3.0, 4.0, 5.0]])
print(f"Prediction: {result}")
```

### Optimization

```python
# optimize.py
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Load model
model = onnx.load('model.onnx')

# Quantize to INT8
quantize_dynamic(
    'model.onnx',
    'model_quantized.onnx',
    weight_type=QuantType.QInt8
)

print("Model quantized and saved")
```

## Performance Benchmarking

### Benchmark Script

```python
# benchmarks/benchmark_all.py
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def benchmark_latency(predict_fn, n_samples=1000):
    """Measure prediction latency."""
    data = np.random.rand(1, 5)

    latencies = []
    for _ in range(n_samples):
        start = time.time()
        predict_fn(data)
        latencies.append(time.time() - start)

    return {
        "mean": np.mean(latencies),
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99)
    }

def benchmark_throughput(predict_fn, duration=60):
    """Measure throughput (requests/sec)."""
    data = np.random.rand(1, 5)

    count = 0
    start_time = time.time()
    end_time = start_time + duration

    while time.time() < end_time:
        predict_fn(data)
        count += 1

    elapsed = time.time() - start_time
    return count / elapsed

# Run benchmarks
results = {
    "torchserve": {
        "latency": benchmark_latency(torchserve_predict),
        "throughput": benchmark_throughput(torchserve_predict)
    },
    "tfserving": {
        "latency": benchmark_latency(tfserving_predict),
        "throughput": benchmark_throughput(tfserving_predict)
    },
    "bentoml": {
        "latency": benchmark_latency(bentoml_predict),
        "throughput": benchmark_throughput(bentoml_predict)
    },
    "onnx": {
        "latency": benchmark_latency(onnx_predict),
        "throughput": benchmark_throughput(onnx_predict)
    }
}
```

## Model Versioning & A/B Testing

### TorchServe A/B Testing

```bash
# Register two model versions
curl -X POST "http://localhost:8081/models?url=model_v1.mar"
curl -X POST "http://localhost:8081/models?url=model_v2.mar"

# Set traffic split (70% v1, 30% v2)
curl -X PUT "http://localhost:8081/models/my_model" \
  -d "min_worker=1&max_worker=4" \
  -d "versions=v1:0.7,v2:0.3"
```

### BentoML Version Management

```python
# List all versions
import bentoml
models = bentoml.models.list()

# Load specific version
model_v1 = bentoml.sklearn.get("my_model:v1.0.0")
model_v2 = bentoml.sklearn.get("my_model:v2.0.0")

# A/B test in service
@svc.api(input=JSON(), output=JSON())
async def predict_ab(input_data: dict) -> dict:
    import random

    # 50/50 split
    if random.random() < 0.5:
        result = await model_v1.async_run(input_data)
        version = "v1.0.0"
    else:
        result = await model_v2.async_run(input_data)
        version = "v2.0.0"

    return {
        "prediction": result,
        "model_version": version
    }
```

## Best Practices

### 1. Model Optimization

✅ Quantize models for faster inference
✅ Use ONNX for cross-platform deployment
✅ Enable batching for throughput
✅ Profile and optimize bottlenecks

### 2. Monitoring

✅ Track prediction latency
✅ Monitor model accuracy drift
✅ Log all predictions
✅ Set up alerting

### 3. Scaling

✅ Use horizontal scaling
✅ Implement load balancing
✅ Enable auto-scaling
✅ Use GPU when beneficial

### 4. Security

✅ Validate input data
✅ Rate limit requests
✅ Use authentication
✅ Encrypt model artifacts

## Resources

- [TorchServe Documentation](https://pytorch.org/serve/)
- [TensorFlow Serving Guide](https://www.tensorflow.org/tfx/guide/serving)
- [BentoML Documentation](https://docs.bentoml.org/)
- [ONNX Runtime Docs](https://onnxruntime.ai/)

## Next Steps

After completing this exercise:

1. ✅ Understand different serving frameworks
2. ✅ Deploy models with multiple frameworks
3. ✅ Benchmark performance
4. ✅ Implement model versioning
5. ✅ Choose appropriate framework for use case

**Move on to**: Exercise 03 - Production API Design
