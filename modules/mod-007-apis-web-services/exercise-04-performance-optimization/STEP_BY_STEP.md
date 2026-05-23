# Step-by-Step Guide: API Performance Optimization

## Overview
Optimize ML API performance using async processing, ONNX model optimization, Celery task queues, and comprehensive monitoring to handle production-scale traffic.

## Phase 1: Async Request Processing (15 minutes)

### Setup Async Environment
```bash
mkdir -p performance-optimization
cd performance-optimization

python3 -m venv venv
source venv/bin/activate

pip install fastapi uvicorn aiohttp asyncio httpx
pip freeze > requirements.txt
```

### Create Async Prediction Service
Create `async_service.py`:
```python
import asyncio
from typing import List
import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Async ML API")

class PredictionRequest(BaseModel):
    features: List[float]

class BatchRequest(BaseModel):
    requests: List[PredictionRequest]

# Simulated model prediction (async)
async def predict_async(features: List[float]) -> float:
    """Async prediction simulation"""
    await asyncio.sleep(0.1)  # Simulate model inference
    return sum(features) / len(features)

# Synchronous version for comparison
def predict_sync(features: List[float]) -> float:
    """Sync prediction simulation"""
    time.sleep(0.1)  # Simulate model inference
    return sum(features) / len(features)

@app.post("/predict-sync")
def predict_endpoint_sync(request: PredictionRequest):
    """Synchronous prediction endpoint"""
    start = time.time()
    result = predict_sync(request.features)
    duration = time.time() - start

    return {
        "prediction": result,
        "duration_ms": duration * 1000,
        "mode": "synchronous"
    }

@app.post("/predict-async")
async def predict_endpoint_async(request: PredictionRequest):
    """Asynchronous prediction endpoint"""
    start = time.time()
    result = await predict_async(request.features)
    duration = time.time() - start

    return {
        "prediction": result,
        "duration_ms": duration * 1000,
        "mode": "asynchronous"
    }

@app.post("/batch-predict")
async def batch_predict(request: BatchRequest):
    """Batch prediction with async processing"""
    start = time.time()

    # Process all predictions concurrently
    tasks = [
        predict_async(req.features)
        for req in request.requests
    ]
    results = await asyncio.gather(*tasks)

    duration = time.time() - start

    return {
        "predictions": results,
        "count": len(results),
        "total_duration_ms": duration * 1000,
        "avg_duration_ms": (duration * 1000) / len(results)
    }
```

### Benchmark Async vs Sync
Create `benchmark_async.py`:
```python
import asyncio
import httpx
import time

async def benchmark_endpoint(url: str, data: dict, num_requests: int = 10):
    """Benchmark an endpoint with concurrent requests"""
    async with httpx.AsyncClient() as client:
        start = time.time()

        tasks = [
            client.post(url, json=data)
            for _ in range(num_requests)
        ]
        responses = await asyncio.gather(*tasks)

        duration = time.time() - start

        print(f"\n{url}")
        print(f"  Total requests: {num_requests}")
        print(f"  Total time: {duration:.2f}s")
        print(f"  Requests/sec: {num_requests/duration:.2f}")
        print(f"  Avg latency: {(duration/num_requests)*1000:.2f}ms")

        return duration

async def main():
    test_data = {"features": [1.0, 2.0, 3.0, 4.0]}

    # Benchmark sync endpoint
    await benchmark_endpoint(
        "http://localhost:8000/predict-sync",
        test_data,
        num_requests=10
    )

    # Benchmark async endpoint
    await benchmark_endpoint(
        "http://localhost:8000/predict-async",
        test_data,
        num_requests=10
    )

if __name__ == "__main__":
    asyncio.run(main())
```

**Validation**: Run benchmark and observe async performance improvement.

## Phase 2: ONNX Model Optimization (15 minutes)

### Install ONNX Runtime
```bash
pip install onnx onnxruntime scikit-learn skl2onnx
```

### Convert Model to ONNX
Create `convert_to_onnx.py`:
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

# Train a model
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, 4]))]
onx = convert_sklearn(model, initial_types=initial_type)

# Save ONNX model
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("Model converted to ONNX format")

# Verify ONNX model
sess = rt.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# Test prediction
test_data = X[:5].astype(np.float32)
pred_onx = sess.run([label_name], {input_name: test_data})[0]
pred_sklearn = model.predict(X[:5])

print(f"ONNX predictions: {pred_onx}")
print(f"SKLearn predictions: {pred_sklearn}")
print(f"Match: {np.array_equal(pred_onx, pred_sklearn)}")
```

Run: `python convert_to_onnx.py`

### Create ONNX Inference Service
Create `onnx_service.py`:
```python
import onnxruntime as rt
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import time

app = FastAPI(title="ONNX Optimized API")

# Load ONNX model at startup
sess = None

@app.on_event("startup")
async def load_model():
    global sess
    sess = rt.InferenceSession("model.onnx")
    print("ONNX model loaded")

class PredictionRequest(BaseModel):
    features: List[List[float]]

@app.post("/predict-onnx")
async def predict_onnx(request: PredictionRequest):
    """ONNX optimized prediction"""
    start = time.time()

    # Prepare input
    input_data = np.array(request.features, dtype=np.float32)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    # Run inference
    predictions = sess.run([label_name], {input_name: input_data})[0]

    duration = time.time() - start

    return {
        "predictions": predictions.tolist(),
        "count": len(predictions),
        "duration_ms": duration * 1000,
        "framework": "ONNX Runtime"
    }

@app.post("/predict-sklearn")
async def predict_sklearn(request: PredictionRequest):
    """SKLearn prediction for comparison"""
    import joblib

    start = time.time()

    # Load sklearn model
    model = joblib.load("sklearn_model.pkl")
    input_data = np.array(request.features)

    # Run inference
    predictions = model.predict(input_data)

    duration = time.time() - start

    return {
        "predictions": predictions.tolist(),
        "count": len(predictions),
        "duration_ms": duration * 1000,
        "framework": "scikit-learn"
    }
```

**Validation**: Compare ONNX vs scikit-learn inference times.

## Phase 3: Celery Task Queue (15 minutes)

### Install Celery and Redis
```bash
pip install celery redis flower

# Start Redis (in Docker)
docker run -d -p 6379:6379 redis:alpine
```

### Create Celery Worker
Create `celery_app.py`:
```python
from celery import Celery
import time
import numpy as np

# Configure Celery
celery_app = Celery(
    'ml_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@celery_app.task(name='predict_task')
def predict_task(features):
    """Background prediction task"""
    time.sleep(2)  # Simulate long-running task
    prediction = sum(features) / len(features)
    return {
        "prediction": prediction,
        "status": "completed"
    }

@celery_app.task(name='batch_predict_task')
def batch_predict_task(batch_features):
    """Background batch prediction"""
    results = []
    for features in batch_features:
        time.sleep(0.5)  # Simulate inference
        prediction = sum(features) / len(features)
        results.append(prediction)

    return {
        "predictions": results,
        "count": len(results),
        "status": "completed"
    }

@celery_app.task(name='retrain_model_task')
def retrain_model_task():
    """Background model retraining"""
    time.sleep(10)  # Simulate training
    return {
        "status": "model_retrained",
        "accuracy": 0.95
    }
```

### Create API with Celery Integration
Create `celery_api.py`:
```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
from celery_app import predict_task, batch_predict_task
from celery.result import AsyncResult

app = FastAPI(title="Celery Task Queue API")

class PredictionRequest(BaseModel):
    features: List[float]

class BatchRequest(BaseModel):
    batch_features: List[List[float]]

@app.post("/predict/async")
async def predict_async_task(request: PredictionRequest):
    """Submit prediction task to queue"""
    task = predict_task.delay(request.features)

    return {
        "task_id": task.id,
        "status": "submitted",
        "message": "Task submitted to queue"
    }

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get task status and result"""
    task_result = AsyncResult(task_id)

    if task_result.ready():
        return {
            "task_id": task_id,
            "status": "completed",
            "result": task_result.result
        }
    else:
        return {
            "task_id": task_id,
            "status": "pending",
            "state": task_result.state
        }

@app.post("/batch/async")
async def batch_predict_async(request: BatchRequest):
    """Submit batch prediction to queue"""
    task = batch_predict_task.delay(request.batch_features)

    return {
        "task_id": task.id,
        "status": "submitted",
        "batch_size": len(request.batch_features)
    }

@app.get("/tasks/active")
async def get_active_tasks():
    """Get active tasks"""
    from celery_app import celery_app

    inspect = celery_app.control.inspect()
    active = inspect.active()
    scheduled = inspect.scheduled()

    return {
        "active_tasks": active,
        "scheduled_tasks": scheduled
    }
```

### Start Celery Worker
```bash
# In one terminal
celery -A celery_app worker --loglevel=info

# In another terminal (optional: Flower UI)
celery -A celery_app flower

# In third terminal - start API
uvicorn celery_api:app --reload
```

**Validation**: Submit tasks and check status via API and Flower UI.

## Phase 4: Request Batching (15 minutes)

### Implement Dynamic Batching
Create `batch_processor.py`:
```python
import asyncio
from typing import List, Dict, Any
from collections import deque
import time

class BatchProcessor:
    def __init__(self, max_batch_size: int = 32, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = deque()
        self.processing = False

    async def add_request(self, features: List[float]) -> Dict[str, Any]:
        """Add request to batch queue"""
        future = asyncio.Future()
        self.queue.append((features, future))

        # Start batch processor if not running
        if not self.processing:
            asyncio.create_task(self._process_batch())

        # Wait for result
        return await future

    async def _process_batch(self):
        """Process queued requests in batches"""
        self.processing = True
        await asyncio.sleep(self.max_wait_time)

        if not self.queue:
            self.processing = False
            return

        # Collect batch
        batch = []
        futures = []

        while self.queue and len(batch) < self.max_batch_size:
            features, future = self.queue.popleft()
            batch.append(features)
            futures.append(future)

        # Process batch
        results = await self._batch_predict(batch)

        # Return results to waiting requests
        for future, result in zip(futures, results):
            future.set_result(result)

        # Continue processing if queue not empty
        if self.queue:
            asyncio.create_task(self._process_batch())
        else:
            self.processing = False

    async def _batch_predict(self, batch: List[List[float]]) -> List[float]:
        """Perform batch prediction"""
        await asyncio.sleep(0.05)  # Simulate batched inference
        return [sum(features) / len(features) for features in batch]

# Global batch processor
batch_processor = BatchProcessor(max_batch_size=32, max_wait_time=0.1)
```

### Create Batching API
```python
from batch_processor import batch_processor

@app.post("/predict-batched")
async def predict_with_batching(request: PredictionRequest):
    """Prediction with automatic request batching"""
    start = time.time()

    result = await batch_processor.add_request(request.features)

    duration = time.time() - start

    return {
        "prediction": result,
        "duration_ms": duration * 1000,
        "batching": "enabled"
    }
```

**Validation**: Send concurrent requests and observe batching behavior.

## Phase 5: Connection Pooling and Caching (15 minutes)

### Implement Connection Pool
Create `connection_pool.py`:
```python
from typing import Optional
import asyncio
import aiohttp
from aiocache import Cache
from aiocache.serializers import JsonSerializer

class OptimizedClient:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = Cache(
            Cache.MEMORY,
            serializer=JsonSerializer(),
            ttl=300
        )

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create session with connection pooling"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,  # Max connections
                limit_per_host=30,
                ttl_dns_cache=300
            )
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session

    async def fetch(self, url: str, use_cache: bool = True):
        """Fetch with caching"""
        # Check cache
        if use_cache:
            cached = await self.cache.get(url)
            if cached:
                return cached

        # Fetch from source
        session = await self.get_session()
        async with session.get(url) as response:
            data = await response.json()

            # Cache result
            if use_cache:
                await self.cache.set(url, data)

            return data

    async def close(self):
        """Close session"""
        if self.session:
            await self.session.close()

client = OptimizedClient()
```

### Add Performance Monitoring
Create `performance_monitor.py`:
```python
from prometheus_client import Histogram, Counter, Gauge
import time
from functools import wraps
import psutil

# Metrics
request_duration = Histogram(
    'request_duration_seconds',
    'Request duration',
    ['endpoint', 'method']
)

throughput = Counter(
    'requests_total',
    'Total requests',
    ['endpoint', 'status']
)

concurrent_requests = Gauge(
    'concurrent_requests',
    'Concurrent requests'
)

system_memory = Gauge('system_memory_percent', 'System memory usage')
system_cpu = Gauge('system_cpu_percent', 'System CPU usage')

def monitor_performance(endpoint: str):
    """Decorator for performance monitoring"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            concurrent_requests.inc()
            start = time.time()

            try:
                result = await func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start
                request_duration.labels(
                    endpoint=endpoint,
                    method='POST'
                ).observe(duration)
                throughput.labels(
                    endpoint=endpoint,
                    status=status
                ).inc()
                concurrent_requests.dec()

                # Update system metrics
                system_memory.set(psutil.virtual_memory().percent)
                system_cpu.set(psutil.cpu_percent())

        return wrapper
    return decorator
```

**Validation**: Monitor metrics under load.

## Phase 6: Load Testing (10 minutes)

### Create Load Test Script
Create `load_test.py`:
```python
import asyncio
import httpx
import time
from statistics import mean, median, stdev

async def load_test(
    url: str,
    data: dict,
    num_requests: int = 1000,
    concurrency: int = 50
):
    """Run load test against endpoint"""
    latencies = []
    errors = 0

    async def make_request(client):
        nonlocal errors
        try:
            start = time.time()
            response = await client.post(url, json=data)
            latency = time.time() - start

            if response.status_code == 200:
                latencies.append(latency)
            else:
                errors += 1
        except Exception as e:
            errors += 1

    async with httpx.AsyncClient(timeout=30.0) as client:
        start_time = time.time()

        # Create batches of concurrent requests
        for i in range(0, num_requests, concurrency):
            batch_size = min(concurrency, num_requests - i)
            tasks = [make_request(client) for _ in range(batch_size)]
            await asyncio.gather(*tasks)

        total_time = time.time() - start_time

    # Calculate statistics
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        print(f"\n=== Load Test Results ===")
        print(f"Total requests: {num_requests}")
        print(f"Successful: {len(latencies)}")
        print(f"Failed: {errors}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Requests/sec: {num_requests/total_time:.2f}")
        print(f"\nLatency (ms):")
        print(f"  Mean: {mean(latencies)*1000:.2f}")
        print(f"  Median: {median(latencies)*1000:.2f}")
        print(f"  Std Dev: {stdev(latencies)*1000:.2f}")
        print(f"  P50: {p50*1000:.2f}")
        print(f"  P95: {p95*1000:.2f}")
        print(f"  P99: {p99*1000:.2f}")

if __name__ == "__main__":
    asyncio.run(load_test(
        "http://localhost:8000/predict-async",
        {"features": [1.0, 2.0, 3.0, 4.0]},
        num_requests=1000,
        concurrency=50
    ))
```

**Validation**: Run load test and analyze performance metrics.

## Summary

You've implemented comprehensive performance optimizations:
- **Async processing** with 5-10x throughput improvement for I/O-bound operations
- **ONNX optimization** reducing inference latency by 30-50%
- **Celery task queues** for long-running background jobs
- **Dynamic batching** maximizing GPU utilization and throughput
- **Connection pooling** reducing overhead for external requests
- **Performance monitoring** with Prometheus metrics tracking
- **Load testing** validating performance under production load

These optimizations enable your ML API to handle production-scale traffic efficiently with optimal resource utilization.
