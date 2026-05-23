# Step-by-Step Implementation Guide: Model Serving

## Overview

Serve ML models in production! Learn model loading, inference optimization, batch processing, model versioning, and integration with MLflow.

**Time**: 2-3 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Load and serve ML models
✅ Implement model versioning
✅ Optimize inference performance
✅ Handle batch predictions
✅ Integrate with MLflow
✅ Implement model warm-up
✅ Add caching strategies
✅ Monitor model performance

---

## Phase 1: Model Loading

### Simple Model Loader

```python
# ml_service.py
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import structlog

logger = structlog.get_logger()

class ModelLoader:
    """Load and cache ML models"""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models: Dict[str, any] = {}

    def load_sklearn_model(self, model_name: str, version: str = "latest"):
        """Load scikit-learn model"""
        model_key = f"{model_name}:{version}"

        if model_key in self.models:
            logger.info("model_cache_hit", model=model_key)
            return self.models[model_key]

        model_path = self.model_dir / f"{model_name}_{version}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.models[model_key] = model
        logger.info("model_loaded", model=model_key, path=str(model_path))
        return model

    def load_pytorch_model(self, model_class, model_name: str, version: str = "latest"):
        """Load PyTorch model"""
        model_key = f"{model_name}:{version}"

        if model_key in self.models:
            return self.models[model_key]

        model_path = self.model_dir / f"{model_name}_{version}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        self.models[model_key] = model
        logger.info("pytorch_model_loaded", model=model_key)
        return model
```

### Model Registry with MLflow

```python
# mlflow_service.py
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from typing import Optional

class MLflowModelService:
    """Serve models from MLflow registry"""

    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.models = {}

    def load_model(self, model_name: str, stage: str = "Production"):
        """Load model from MLflow registry"""
        model_key = f"{model_name}:{stage}"

        if model_key in self.models:
            return self.models[model_key]

        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)

        self.models[model_key] = model
        logger.info("mlflow_model_loaded", model=model_name, stage=stage)
        return model

    def get_model_version(self, model_name: str, stage: str = "Production"):
        """Get model version info"""
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise ValueError(f"No {stage} version found for {model_name}")

        return {
            "version": versions[0].version,
            "run_id": versions[0].run_id,
            "stage": stage
        }
```

---

## Phase 2: Inference Service

### Basic Inference

```python
# inference.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from ml_service import ModelLoader

app = FastAPI()
model_loader = ModelLoader()

class PredictionRequest(BaseModel):
    features: List[float]
    model_version: str = "v1"

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    inference_time_ms: float

@app.on_event("startup")
async def load_models():
    """Pre-load models on startup"""
    model_loader.load_sklearn_model("iris_classifier", "v1")
    logger.info("models_preloaded")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Run inference"""
    import time
    start_time = time.time()

    try:
        # Load model
        model = model_loader.load_sklearn_model("iris_classifier", request.model_version)

        # Prepare input
        features = np.array(request.features).reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]

        inference_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            prediction=float(prediction),
            model_version=request.model_version,
            inference_time_ms=inference_time
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("inference_error", error=str(e))
        raise HTTPException(status_code=500, detail="Inference failed")
```

### PyTorch Inference

```python
# pytorch_inference.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from io import BytesIO

class ImageClassifier(nn.Module):
    """Simple CNN for image classification"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class PyTorchInferenceService:
    """PyTorch model inference service"""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = ImageClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @torch.no_grad()
    def predict(self, image_bytes: bytes):
        """Predict image class"""
        # Load and preprocess image
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        output = self.model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities[0].cpu().numpy().tolist()
        }

# FastAPI integration
from fastapi import File, UploadFile

pytorch_service = PyTorchInferenceService("models/image_classifier.pth")

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Image classification endpoint"""
    contents = await file.read()
    result = pytorch_service.predict(contents)
    return result
```

---

## Phase 3: Batch Processing

### Batch Prediction

```python
# batch_inference.py
from typing import List
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio

class BatchInferenceService:
    """Batch inference with parallelization"""

    def __init__(self, model, batch_size: int = 32, max_workers: int = 4):
        self.model = model
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def predict_batch(self, features: np.ndarray):
        """Predict on batch"""
        return self.model.predict(features)

    async def predict_batch_async(self, all_features: List[List[float]]):
        """Async batch prediction with chunking"""
        all_features = np.array(all_features)
        predictions = []

        # Split into batches
        for i in range(0, len(all_features), self.batch_size):
            batch = all_features[i:i + self.batch_size]

            # Run prediction in thread pool
            loop = asyncio.get_event_loop()
            batch_predictions = await loop.run_in_executor(
                self.executor,
                self.predict_batch,
                batch
            )
            predictions.extend(batch_predictions.tolist())

        return predictions

# FastAPI endpoint
from pydantic import BaseModel

class BatchPredictionRequest(BaseModel):
    samples: List[List[float]]

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    total: int
    batch_size: int

batch_service = BatchInferenceService(model_loader.load_sklearn_model("iris_classifier", "v1"))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    predictions = await batch_service.predict_batch_async(request.samples)

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        batch_size=batch_service.batch_size
    )
```

---

## Phase 4: Model Versioning

### Version Management

```python
# version_manager.py
from typing import Dict, List
from datetime import datetime
from pydantic import BaseModel

class ModelMetadata(BaseModel):
    name: str
    version: str
    created_at: datetime
    accuracy: float
    framework: str
    path: str

class ModelVersionManager:
    """Manage multiple model versions"""

    def __init__(self):
        self.versions: Dict[str, Dict[str, ModelMetadata]] = {}
        self.active_versions: Dict[str, str] = {}

    def register_model(self, metadata: ModelMetadata):
        """Register new model version"""
        if metadata.name not in self.versions:
            self.versions[metadata.name] = {}

        self.versions[metadata.name][metadata.version] = metadata
        logger.info("model_registered", name=metadata.name, version=metadata.version)

    def set_active_version(self, model_name: str, version: str):
        """Set active version for model"""
        if model_name not in self.versions or version not in self.versions[model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")

        self.active_versions[model_name] = version
        logger.info("active_version_updated", model=model_name, version=version)

    def get_active_version(self, model_name: str):
        """Get active version"""
        return self.active_versions.get(model_name, "latest")

    def list_versions(self, model_name: str) -> List[ModelMetadata]:
        """List all versions for model"""
        if model_name not in self.versions:
            return []
        return list(self.versions[model_name].values())

# API endpoints
version_manager = ModelVersionManager()

@app.get("/models/{model_name}/versions")
async def list_model_versions(model_name: str):
    """List all versions of a model"""
    versions = version_manager.list_versions(model_name)
    return {"model": model_name, "versions": versions}

@app.post("/models/{model_name}/versions/{version}/activate")
async def activate_version(model_name: str, version: str):
    """Set active version"""
    version_manager.set_active_version(model_name, version)
    return {"model": model_name, "active_version": version}
```

---

## Phase 5: Caching

### Prediction Caching

```python
# cache.py
import hashlib
import json
from typing import Optional
import redis
from functools import wraps

class PredictionCache:
    """Cache predictions using Redis"""

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = ttl

    def _make_key(self, model_name: str, features: List[float]):
        """Generate cache key"""
        data = f"{model_name}:{json.dumps(features)}"
        return f"pred:{hashlib.md5(data.encode()).hexdigest()}"

    def get(self, model_name: str, features: List[float]) -> Optional[dict]:
        """Get cached prediction"""
        key = self._make_key(model_name, features)
        cached = self.redis_client.get(key)
        if cached:
            logger.info("cache_hit", model=model_name)
            return json.loads(cached)
        return None

    def set(self, model_name: str, features: List[float], prediction: dict):
        """Cache prediction"""
        key = self._make_key(model_name, features)
        self.redis_client.setex(key, self.ttl, json.dumps(prediction))

def cache_predictions(cache: PredictionCache):
    """Decorator to cache predictions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: PredictionRequest):
            # Check cache
            cached = cache.get("iris_classifier", request.features)
            if cached:
                return PredictionResponse(**cached)

            # Get fresh prediction
            response = await func(request)

            # Cache result
            cache.set("iris_classifier", request.features, response.dict())

            return response
        return wrapper
    return decorator

# Usage
cache = PredictionCache()

@app.post("/predict/cached")
@cache_predictions(cache)
async def predict_with_cache(request: PredictionRequest):
    """Prediction with caching"""
    # ... normal prediction logic ...
    pass
```

---

## Phase 6: Model Warm-up

### Pre-warming Models

```python
# warmup.py
import asyncio
import numpy as np

class ModelWarmer:
    """Warm up models with dummy requests"""

    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader

    async def warmup_model(self, model_name: str, version: str, num_warmup: int = 10):
        """Warm up specific model"""
        model = self.model_loader.load_sklearn_model(model_name, version)

        logger.info("warming_up_model", model=model_name, version=version)

        # Generate dummy data
        dummy_features = np.random.rand(num_warmup, 4)

        # Run predictions
        for i in range(num_warmup):
            _ = model.predict(dummy_features[i:i+1])

        logger.info("model_warmed_up", model=model_name, warmup_requests=num_warmup)

@app.on_event("startup")
async def warmup_models():
    """Warm up all models on startup"""
    warmer = ModelWarmer(model_loader)

    models_to_warmup = [
        ("iris_classifier", "v1"),
        ("iris_classifier", "v2"),
    ]

    tasks = [
        warmer.warmup_model(name, version)
        for name, version in models_to_warmup
    ]

    await asyncio.gather(*tasks)
    logger.info("all_models_warmed_up")
```

---

## Phase 7: Performance Monitoring

### Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

# Define metrics
prediction_counter = Counter(
    'model_predictions_total',
    'Total predictions',
    ['model', 'version']
)

prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency',
    ['model', 'version']
)

model_accuracy = Gauge(
    'model_accuracy',
    'Model accuracy',
    ['model', 'version']
)

cache_hit_counter = Counter(
    'cache_hits_total',
    'Cache hits',
    ['model']
)

# Instrument FastAPI
Instrumentator().instrument(app).expose(app)

@app.post("/predict/monitored")
async def predict_with_metrics(request: PredictionRequest):
    """Prediction with metrics"""
    model_name = "iris_classifier"
    version = request.model_version

    # Track prediction
    prediction_counter.labels(model=model_name, version=version).inc()

    # Measure latency
    with prediction_latency.labels(model=model_name, version=version).time():
        # ... prediction logic ...
        result = await predict(request)

    return result
```

---

## Phase 8: A/B Testing

### Traffic Splitting

```python
# ab_testing.py
import random
from typing import Dict

class ABTestRouter:
    """Route traffic for A/B testing"""

    def __init__(self):
        self.experiments: Dict[str, dict] = {}

    def create_experiment(self, name: str, variants: Dict[str, float]):
        """
        Create A/B test experiment

        variants: {"v1": 0.5, "v2": 0.5}  # 50/50 split
        """
        if sum(variants.values()) != 1.0:
            raise ValueError("Variant weights must sum to 1.0")

        self.experiments[name] = variants

    def get_variant(self, experiment_name: str, user_id: Optional[str] = None):
        """Get variant for user"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        variants = self.experiments[experiment_name]

        # Deterministic assignment based on user_id
        if user_id:
            hash_value = hash(user_id) % 100
            cumulative = 0
            for variant, weight in variants.items():
                cumulative += weight * 100
                if hash_value < cumulative:
                    return variant

        # Random assignment
        rand_val = random.random()
        cumulative = 0
        for variant, weight in variants.items():
            cumulative += weight
            if rand_val < cumulative:
                return variant

        return list(variants.keys())[0]

# Usage
ab_router = ABTestRouter()
ab_router.create_experiment("model_version", {"v1": 0.8, "v2": 0.2})

@app.post("/predict/ab")
async def predict_ab_test(request: PredictionRequest, user_id: Optional[str] = None):
    """Prediction with A/B testing"""
    variant = ab_router.get_variant("model_version", user_id)

    # Use selected variant
    request.model_version = variant

    result = await predict(request)
    result.variant = variant

    return result
```

---

## Complete Production Service

```python
# production_service.py
from fastapi import FastAPI, Depends, HTTPException
from ml_service import ModelLoader
from mlflow_service import MLflowModelService
from cache import PredictionCache
from metrics import prediction_counter, prediction_latency
import structlog

logger = structlog.get_logger()

app = FastAPI(title="Production ML API")

# Initialize services
model_loader = ModelLoader()
mlflow_service = MLflowModelService("http://mlflow:5000")
cache = PredictionCache()

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    # Load models
    model_loader.load_sklearn_model("iris_classifier", "v1")

    # Warm up
    warmer = ModelWarmer(model_loader)
    await warmer.warmup_model("iris_classifier", "v1")

    logger.info("service_ready")

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Production prediction endpoint"""
    # Check cache
    cached = cache.get("iris_classifier", request.features)
    if cached:
        return cached

    # Metrics
    prediction_counter.labels(model="iris", version=request.model_version).inc()

    # Predict
    with prediction_latency.labels(model="iris", version=request.model_version).time():
        model = model_loader.load_sklearn_model("iris_classifier", request.model_version)
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]

    result = PredictionResponse(
        prediction=float(prediction),
        model_version=request.model_version,
        inference_time_ms=0
    )

    # Cache
    cache.set("iris_classifier", request.features, result.dict())

    return result
```

---

## Best Practices

✅ Pre-load models on startup
✅ Implement model versioning
✅ Cache predictions when appropriate
✅ Use batch processing for throughput
✅ Monitor inference latency
✅ Implement A/B testing
✅ Warm up models before serving
✅ Handle errors gracefully
✅ Log all predictions
✅ Use connection pooling for databases

---

**Model Serving mastered!** 🎯

**Next Exercise**: Production API
