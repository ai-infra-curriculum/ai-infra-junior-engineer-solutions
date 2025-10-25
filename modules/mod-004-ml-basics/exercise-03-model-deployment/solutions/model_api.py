#!/usr/bin/env python3
"""
model_api.py - FastAPI-based model serving API

Description:
    Production-ready ML model serving API with multi-model support,
    batch inference, monitoring, and A/B testing capabilities.

Usage:
    # Development
    uvicorn model_api:app --reload

    # Production
    uvicorn model_api:app --host 0.0.0.0 --port 8000 --workers 4

Features:
    - Multi-model version serving
    - Batch inference optimization
    - Prometheus metrics
    - Health checks
    - A/B testing support
    - Request validation
    - Error handling
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Header
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import time
import logging
from datetime import datetime
from collections import deque
import asyncio
import random
import hashlib

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Application Configuration
# ============================================================================

app = FastAPI(
    title="ML Model Serving API",
    description="Production ML model inference service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# Prometheus Metrics
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Request metrics
    request_counter = Counter(
        'api_requests_total',
        'Total API requests',
        ['endpoint', 'method', 'status']
    )

    # Prediction metrics
    prediction_counter = Counter(
        'model_predictions_total',
        'Total number of predictions',
        ['model_name', 'model_version', 'status']
    )

    prediction_latency = Histogram(
        'model_prediction_latency_seconds',
        'Prediction latency in seconds',
        ['model_name', 'model_version'],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    )

    batch_size_histogram = Histogram(
        'batch_size',
        'Batch size distribution',
        ['model_name'],
        buckets=[1, 2, 4, 8, 16, 32, 64, 128]
    )

    # System metrics
    model_loaded_gauge = Gauge(
        'model_loaded',
        'Whether model is loaded (1=loaded, 0=not loaded)',
        ['model_name', 'model_version']
    )

    gpu_memory_usage = Gauge(
        'gpu_memory_usage_bytes',
        'GPU memory usage in bytes',
        ['gpu_id']
    )

    active_requests_gauge = Gauge(
        'active_requests',
        'Number of active requests',
        ['endpoint']
    )


# ============================================================================
# Data Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Prediction request schema"""
    data: List[List[float]]
    model_version: Optional[str] = 'latest'
    batch_size: Optional[int] = None

    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        return v


class PredictionResponse(BaseModel):
    """Prediction response schema"""
    predictions: List[Any]
    model_name: str
    model_version: str
    inference_time_ms: float
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    requests: List[PredictionRequest]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: int
    gpu_available: bool
    gpu_count: int
    uptime_seconds: float


class ModelInfo(BaseModel):
    """Model metadata"""
    name: str
    version: str
    framework: str
    input_shape: List[Optional[int]]
    output_shape: List[Optional[int]]
    loaded_at: str
    size_mb: float


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """Thread-safe model registry"""

    def __init__(self):
        self._models: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()

    async def register(self, name: str, model: nn.Module,
                      version: str = 'latest', metadata: Optional[Dict] = None):
        """Register a model"""
        async with self._lock:
            key = f"{name}:{version}"
            self._models[key] = {
                'model': model,
                'metadata': metadata or {},
                'loaded_at': datetime.now(),
                'version': version
            }

            # Update metrics
            if PROMETHEUS_AVAILABLE:
                model_loaded_gauge.labels(model_name=name, model_version=version).set(1)

            logger.info(f"Registered model: {key}")

    async def get(self, name: str, version: str = 'latest') -> Optional[nn.Module]:
        """Get model from registry"""
        key = f"{name}:{version}"
        async with self._lock:
            if key in self._models:
                return self._models[key]['model']
        return None

    async def get_info(self, name: str, version: str = 'latest') -> Optional[Dict]:
        """Get model metadata"""
        key = f"{name}:{version}"
        async with self._lock:
            if key in self._models:
                return self._models[key]
        return None

    async def list_models(self) -> List[str]:
        """List all registered models"""
        async with self._lock:
            return list(self._models.keys())

    async def unload(self, name: str, version: str = 'latest'):
        """Unload model from memory"""
        key = f"{name}:{version}"
        async with self._lock:
            if key in self._models:
                del self._models[key]

                if PROMETHEUS_AVAILABLE:
                    model_loaded_gauge.labels(model_name=name, model_version=version).set(0)

                logger.info(f"Unloaded model: {key}")


# Global model registry
model_registry = ModelRegistry()


# ============================================================================
# Batch Predictor
# ============================================================================

class BatchPredictor:
    """Dynamic batch inference"""

    def __init__(self, model: nn.Module, batch_size: int = 32,
                 max_wait_ms: int = 100):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms

        self.queue = deque()
        self.processing = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.model.eval()

    async def predict(self, input_data: torch.Tensor) -> torch.Tensor:
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
                batch_tensor = torch.stack(batch).to(self.device)

                if PROMETHEUS_AVAILABLE:
                    batch_size_histogram.labels(model_name='default').observe(len(batch))

                with torch.no_grad():
                    outputs = self.model(batch_tensor)

                # Set results
                for i, future in enumerate(futures):
                    future.set_result(outputs[i].cpu())

            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                # Set exception for all futures
                for future in futures:
                    future.set_exception(e)

        self.processing = False


# ============================================================================
# A/B Testing
# ============================================================================

class ABTester:
    """A/B testing for model versions"""

    def __init__(self, traffic_split: Dict[str, float]):
        """
        Initialize A/B tester

        Args:
            traffic_split: Traffic split percentages (must sum to 1.0)
        """
        self.traffic_split = traffic_split

        # Validate traffic split
        total = sum(traffic_split.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Traffic split must sum to 1.0, got {total}")

    def select_version(self, user_id: Optional[str] = None) -> str:
        """
        Select model version based on traffic split

        Args:
            user_id: Optional user ID for consistent routing

        Returns:
            Selected version
        """
        if user_id:
            # Consistent hashing for same user
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            rand_val = (hash_val % 10000) / 10000
        else:
            # Random selection
            rand_val = random.random()

        # Select model based on cumulative probability
        cumulative = 0
        for version, probability in sorted(self.traffic_split.items()):
            cumulative += probability
            if rand_val <= cumulative:
                return version

        # Fallback
        return list(self.traffic_split.keys())[0]


# Global A/B tester (configured at startup)
ab_tester = None


# ============================================================================
# Application Lifecycle
# ============================================================================

# Track startup time
startup_time = None


@app.on_event("startup")
async def startup_event():
    """Initialize application"""
    global startup_time
    startup_time = time.time()

    logger.info("Starting ML Model Serving API...")

    # Load default model (ResNet18 for demo)
    try:
        logger.info("Loading default model (ResNet18)...")
        model = models.resnet18(pretrained=True)
        model.eval()

        await model_registry.register(
            name='resnet18',
            model=model,
            version='v1.0',
            metadata={
                'framework': 'pytorch',
                'input_shape': [None, 3, 224, 224],
                'output_shape': [None, 1000],
                'size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            }
        )

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    # Initialize A/B tester
    global ab_tester
    ab_tester = ABTester(traffic_split={'v1.0': 1.0})

    logger.info("API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")


# ============================================================================
# Health and Info Endpoints
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint"""
    return {
        "service": "ML Model Serving API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models = await model_registry.list_models()

    uptime = time.time() - startup_time if startup_time else 0

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(models),
        gpu_available=torch.cuda.is_available(),
        gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        uptime_seconds=uptime
    )


@app.get("/models", response_model=List[str])
async def list_models():
    """List available models"""
    return await model_registry.list_models()


@app.get("/models/{model_name}/{version}", response_model=ModelInfo)
async def get_model_info(model_name: str, version: str = 'latest'):
    """Get model metadata"""
    info = await model_registry.get_info(model_name, version)

    if not info:
        raise HTTPException(status_code=404, detail="Model not found")

    metadata = info['metadata']

    return ModelInfo(
        name=model_name,
        version=version,
        framework=metadata.get('framework', 'unknown'),
        input_shape=metadata.get('input_shape', []),
        output_shape=metadata.get('output_shape', []),
        loaded_at=info['loaded_at'].isoformat(),
        size_mb=metadata.get('size_mb', 0.0)
    )


# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest,
                 user_id: Optional[str] = Header(None)):
    """Make prediction"""
    start_time = time.time()

    try:
        # Track active requests
        if PROMETHEUS_AVAILABLE:
            active_requests_gauge.labels(endpoint='predict').inc()

        # Select model version (A/B testing)
        version = ab_tester.select_version(user_id) if ab_tester else 'v1.0'

        # Get model
        model = await model_registry.get('resnet18', version)
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available")

        # Convert input to tensor
        input_tensor = torch.tensor(request.data, dtype=torch.float32)

        # Make prediction
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
            predictions = output.cpu().tolist()

        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # ms

        # Record metrics
        if PROMETHEUS_AVAILABLE:
            prediction_counter.labels(
                model_name='resnet18',
                model_version=version,
                status='success'
            ).inc()

            prediction_latency.labels(
                model_name='resnet18',
                model_version=version
            ).observe(inference_time / 1000)

        return PredictionResponse(
            predictions=predictions,
            model_name='resnet18',
            model_version=version,
            inference_time_ms=inference_time,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Prediction error: {e}")

        if PROMETHEUS_AVAILABLE:
            prediction_counter.labels(
                model_name='resnet18',
                model_version='v1.0',
                status='error'
            ).inc()

        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if PROMETHEUS_AVAILABLE:
            active_requests_gauge.labels(endpoint='predict').dec()


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Predict from uploaded image"""
    start_time = time.time()

    try:
        # Track active requests
        if PROMETHEUS_AVAILABLE:
            active_requests_gauge.labels(endpoint='predict_image').inc()

        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocess
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        input_tensor = preprocess(image).unsqueeze(0)

        # Get model
        model = await model_registry.get('resnet18', 'v1.0')
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available")

        # Make prediction
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)

        # Get top 5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)

        predictions = [
            {
                "class_id": int(idx),
                "probability": float(prob)
            }
            for idx, prob in zip(top5_idx[0].cpu(), top5_prob[0].cpu())
        ]

        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000

        return {
            "predictions": predictions,
            "model_name": "resnet18",
            "model_version": "v1.0",
            "inference_time_ms": inference_time,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if PROMETHEUS_AVAILABLE:
            active_requests_gauge.labels(endpoint='predict_image').dec()


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    results = []

    for req in request.requests:
        try:
            result = await predict(req)
            results.append(result.dict())
        except HTTPException as e:
            results.append({"error": e.detail})

    return {"results": results, "count": len(results)}


# ============================================================================
# Monitoring Endpoints
# ============================================================================

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if not PROMETHEUS_AVAILABLE:
        return PlainTextResponse("Prometheus client not available")

    # Update GPU metrics
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i)
            gpu_memory_usage.labels(gpu_id=str(i)).set(memory_allocated)

    return PlainTextResponse(generate_latest())


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "model_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
