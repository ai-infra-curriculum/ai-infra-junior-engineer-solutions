# Exercise 04: Performance & Optimization

## Overview

This exercise covers performance optimization techniques for production ML serving systems. You'll learn model optimization, async processing, load balancing, caching strategies, and performance monitoring to build high-throughput, low-latency ML APIs.

## Learning Objectives

- Optimize ML models for inference (quantization, pruning, ONNX conversion)
- Implement async processing with Celery and RabbitMQ
- Configure load balancing with Nginx and HAProxy
- Set up advanced caching strategies with Redis
- Monitor performance with Prometheus and Grafana
- Implement horizontal scaling and auto-scaling
- Use connection pooling for databases and external services
- Build comprehensive benchmarking suites

## Table of Contents

1. [Model Optimization](#model-optimization)
2. [Async Processing](#async-processing)
3. [Load Balancing](#load-balancing)
4. [Advanced Caching](#advanced-caching)
5. [Performance Monitoring](#performance-monitoring)
6. [Horizontal Scaling](#horizontal-scaling)
7. [Benchmarking](#benchmarking)

---

## 1. Model Optimization

### Techniques Covered

#### Quantization
Convert model weights from float32 to int8/int16 for faster inference.

**Benefits:**
- 2-4x speedup on CPU
- 75% reduction in model size
- Lower memory footprint

**Trade-offs:**
- Slight accuracy loss (typically <1%)
- Requires calibration dataset

#### Pruning
Remove less important weights/neurons from the model.

**Benefits:**
- Smaller model size
- Faster inference
- Lower memory usage

**Trade-offs:**
- Requires retraining/fine-tuning
- More complex optimization process

#### ONNX Optimization
Convert models to ONNX format and apply graph optimizations.

**Benefits:**
- Cross-framework compatibility
- Hardware-specific optimizations
- Simplified deployment

### Implementation Files

- `src/optimization/quantize_model.py` - Dynamic and static quantization
- `src/optimization/prune_model.py` - Weight and structured pruning
- `src/optimization/onnx_optimize.py` - ONNX conversion and optimization

### Example: Dynamic Quantization

```python
import torch
from torch.quantization import quantize_dynamic

# Original model
model = YourModel()
model.eval()

# Quantize
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# Benchmark
# Original: 45ms/inference
# Quantized: 12ms/inference (3.75x speedup)
```

---

## 2. Async Processing

### Architecture

Use Celery with RabbitMQ for async ML inference tasks.

**Benefits:**
- Handle long-running predictions asynchronously
- Better resource utilization
- Prevent API timeouts
- Enable batch processing

### Components

1. **Celery Worker** - Process async tasks
2. **RabbitMQ** - Message broker
3. **Redis** - Result backend
4. **FastAPI** - Submit tasks and retrieve results

### Workflow

```
Client → FastAPI → RabbitMQ → Celery Worker → Redis → FastAPI → Client
   ↓        ↓                      ↓           ↓
 Submit   Task ID              Process      Store Result
```

### Implementation Files

- `src/async_processing/celery_app.py` - Celery configuration
- `src/async_processing/tasks.py` - Async ML tasks
- `src/async_processing/api.py` - FastAPI endpoints for async processing

### Example Usage

```python
# Submit async prediction
response = requests.post("/api/v1/predict/async", json={"features": [...]})
task_id = response.json()["task_id"]

# Poll for result
result = requests.get(f"/api/v1/tasks/{task_id}")
# {"status": "completed", "result": {...}}
```

---

## 3. Load Balancing

### Strategies

#### Round Robin
Distribute requests evenly across backend servers.

**Use case:** Uniform request processing time

#### Least Connections
Route to server with fewest active connections.

**Use case:** Varying request complexity

#### IP Hash
Same client always routes to same server.

**Use case:** Session affinity required

#### Weighted Round Robin
Distribute based on server capacity.

**Use case:** Heterogeneous server specs (GPU vs CPU)

### Implementation

#### Nginx Configuration

```nginx
upstream ml_api {
    least_conn;
    server api1:8000 weight=3;  # GPU server
    server api2:8000 weight=1;  # CPU server
    server api3:8000 weight=1;  # CPU server
}

server {
    listen 80;

    location /api/ {
        proxy_pass http://ml_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

#### HAProxy Configuration

```
backend ml_api
    balance leastconn
    option httpchk GET /health
    server api1 10.0.1.10:8000 check weight 3
    server api2 10.0.1.11:8000 check weight 1
    server api3 10.0.1.12:8000 check weight 1
```

### Files

- `load_balancing/nginx.conf` - Nginx load balancer configuration
- `load_balancing/haproxy.cfg` - HAProxy configuration
- `load_balancing/docker-compose.yml` - Multi-instance setup

---

## 4. Advanced Caching

### Multi-Level Caching Strategy

```
Request → L1: In-Memory Cache → L2: Redis → L3: Database → Model
           (ms response)        (5-10ms)     (50-100ms)   (100-500ms)
```

### Caching Strategies

#### 1. Result Caching
Cache prediction results for identical inputs.

**TTL:** 5-30 minutes
**Hit Rate:** 30-50% for repeated queries

#### 2. Model Caching
Cache loaded models in memory to avoid repeated loading.

**TTL:** Infinite (until memory pressure)
**Benefit:** Avoid 2-5s model loading time

#### 3. Feature Caching
Cache preprocessed features for frequently queried data.

**TTL:** 10-60 minutes
**Hit Rate:** 40-60% for popular entities

#### 4. Negative Caching
Cache "not found" results to prevent repeated expensive lookups.

**TTL:** 1-5 minutes
**Benefit:** Reduce database load

### Cache Invalidation Patterns

1. **Time-based (TTL)** - Expire after fixed duration
2. **Event-driven** - Invalidate on model update
3. **LRU** - Evict least recently used when memory full
4. **Write-through** - Update cache on write
5. **Cache-aside** - Load from cache, populate on miss

### Implementation

See `src/cache/` directory with:
- `multilevel_cache.py` - L1 (in-memory) + L2 (Redis) caching
- `cache_strategies.py` - Different caching patterns
- `cache_warming.py` - Pre-populate cache with common queries

---

## 5. Performance Monitoring

### Metrics to Track

#### Application Metrics
- **Request rate** (requests/second)
- **Response time** (p50, p95, p99 latency)
- **Error rate** (4xx, 5xx errors)
- **Throughput** (predictions/second)

#### ML-Specific Metrics
- **Model inference time** (preprocessing + inference + postprocessing)
- **Batch size** (for batch inference)
- **Queue depth** (async tasks pending)
- **Cache hit rate** (% requests served from cache)

#### Infrastructure Metrics
- **CPU utilization** (%)
- **Memory usage** (MB)
- **GPU utilization** (%)
- **Network I/O** (bytes/sec)
- **Disk I/O** (IOPS)

### Stack

- **Prometheus** - Metrics collection
- **Grafana** - Visualization and dashboards
- **Prometheus Python Client** - Instrument FastAPI app
- **cAdvisor** - Container metrics
- **Node Exporter** - System metrics

### Implementation Files

- `src/monitoring/prometheus_metrics.py` - Custom Prometheus metrics
- `src/monitoring/middleware.py` - FastAPI monitoring middleware
- `docker/prometheus.yml` - Prometheus configuration
- `docker/grafana-dashboards/` - Pre-built Grafana dashboards

### Example Instrumentation

```python
from prometheus_client import Counter, Histogram

PREDICTION_COUNT = Counter(
    'ml_predictions_total',
    'Total ML predictions',
    ['model_version', 'status']
)

PREDICTION_LATENCY = Histogram(
    'ml_prediction_duration_seconds',
    'Time spent processing prediction',
    ['model_version']
)

@app.post("/predict")
async def predict(request: PredictionRequest):
    with PREDICTION_LATENCY.labels(model_version="v1").time():
        result = await model.predict(request.features)
        PREDICTION_COUNT.labels(model_version="v1", status="success").inc()
        return result
```

---

## 6. Horizontal Scaling

### Strategies

#### 1. Stateless API Design
- No session state in API servers
- All state in Redis/database
- Enable easy scaling

#### 2. Auto-Scaling
Scale based on metrics:
- CPU > 70% → Scale up
- Queue depth > 100 → Add workers
- Request rate > 1000 rps → Add API instances

#### 3. Container Orchestration
Use Kubernetes for:
- Horizontal Pod Autoscaler (HPA)
- Cluster Autoscaler
- Resource limits and requests

### Kubernetes HPA Example

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
```

### Connection Pooling

**Database Connection Pooling:**
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,           # Persistent connections
    max_overflow=10,        # Additional connections during spikes
    pool_timeout=30,        # Wait time for connection
    pool_recycle=3600,      # Recycle connections every hour
)
```

**Redis Connection Pooling:**
```python
import redis

pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=50,
    socket_connect_timeout=5,
    socket_keepalive=True,
)

redis_client = redis.Redis(connection_pool=pool)
```

---

## 7. Benchmarking

### Tools

1. **Locust** - Distributed load testing
2. **Apache Bench (ab)** - Simple HTTP benchmarking
3. **wrk** - High-performance HTTP benchmarking
4. **Vegeta** - HTTP load testing tool

### Metrics to Measure

- **Throughput** (requests/second)
- **Latency** (p50, p95, p99, p99.9)
- **Error rate** (%)
- **Resource utilization** (CPU, memory, GPU)
- **Cost per 1000 requests** ($)

### Implementation Files

- `benchmarks/locust_test.py` - Locust load test scenarios
- `benchmarks/ab_benchmark.sh` - Apache Bench scripts
- `benchmarks/wrk_scenarios/` - wrk benchmark scenarios
- `benchmarks/performance_report.py` - Generate performance reports

### Example: Locust Load Test

```python
from locust import HttpUser, task, between

class MLAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict_single(self):
        self.client.post("/api/v1/predict", json={
            "features": [1.0, 2.0, 3.0, 4.0, 5.0]
        })

    @task(1)
    def predict_batch(self):
        self.client.post("/api/v1/predict/batch", json={
            "instances": [[1.0, 2.0, 3.0, 4.0, 5.0]] * 10
        })

    @task(1)
    def health_check(self):
        self.client.get("/health")
```

Run: `locust -f locust_test.py --host=http://localhost:8000 --users=100 --spawn-rate=10`

### Example: Apache Bench

```bash
# Benchmark single endpoint
ab -n 10000 -c 100 -p payload.json -T application/json \
   http://localhost:8000/api/v1/predict

# Results:
# Requests per second: 450 [#/sec]
# Time per request (mean): 222 ms
# Time per request (p95): 380 ms
# Time per request (p99): 520 ms
```

### Example: wrk

```bash
# High-concurrency benchmark
wrk -t12 -c400 -d30s -s post.lua http://localhost:8000/api/v1/predict

# Results:
# Throughput: 12,500 requests/sec
# Latency (avg): 32ms
# Latency (p99): 85ms
```

---

## Performance Optimization Checklist

### Model Level
- [ ] Quantize model (dynamic or static)
- [ ] Prune unnecessary weights
- [ ] Convert to ONNX for optimized inference
- [ ] Use TensorRT for NVIDIA GPUs
- [ ] Batch predictions when possible
- [ ] Cache model in memory

### Application Level
- [ ] Implement async processing for long tasks
- [ ] Use connection pooling for databases
- [ ] Enable gzip compression
- [ ] Implement multi-level caching
- [ ] Use efficient serialization (msgpack > JSON)
- [ ] Optimize data preprocessing

### Infrastructure Level
- [ ] Configure load balancing
- [ ] Set up horizontal auto-scaling
- [ ] Use CDN for static assets
- [ ] Optimize Docker images (multi-stage builds)
- [ ] Configure resource limits
- [ ] Enable HTTP/2

### Monitoring Level
- [ ] Track all key metrics (latency, throughput, errors)
- [ ] Set up alerting for anomalies
- [ ] Create performance dashboards
- [ ] Regular load testing
- [ ] Profile production traffic

---

## Expected Performance Improvements

### Baseline (Unoptimized)
- Throughput: 50 requests/sec
- Latency (p95): 800ms
- Model size: 200MB
- Memory usage: 2GB

### After Optimization
- Throughput: **500 requests/sec** (10x improvement)
- Latency (p95): **80ms** (10x improvement)
- Model size: **50MB** (4x reduction)
- Memory usage: **500MB** (4x reduction)

### Optimization Breakdown

| Optimization | Throughput Gain | Latency Improvement |
|--------------|-----------------|---------------------|
| Model quantization | +2x | -60% |
| ONNX optimization | +1.5x | -30% |
| Redis caching (40% hit rate) | +2x | -50% (cached) |
| Load balancing (3 instances) | +3x | -20% |
| Async processing | +1.5x | N/A (for batch) |
| Connection pooling | +1.2x | -10% |

**Combined:** ~10x throughput, ~10x latency improvement

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Infrastructure

```bash
docker-compose up -d
```

This starts:
- RabbitMQ (message broker)
- Redis (cache + result backend)
- Prometheus (metrics)
- Grafana (dashboards)

### 3. Run Optimization Scripts

```bash
# Quantize model
python src/optimization/quantize_model.py

# Convert to ONNX
python src/optimization/onnx_optimize.py

# Prune model
python src/optimization/prune_model.py
```

### 4. Start Celery Worker

```bash
celery -A src.async_processing.celery_app worker --loglevel=info
```

### 5. Start API

```bash
uvicorn src.async_processing.api:app --reload
```

### 6. Run Benchmarks

```bash
# Locust (opens web UI at http://localhost:8089)
locust -f benchmarks/locust_test.py --host=http://localhost:8000

# Apache Bench
bash benchmarks/ab_benchmark.sh

# Generate performance report
python benchmarks/performance_report.py
```

### 7. View Metrics

- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

---

## Project Structure

```
exercise-04-performance-optimization/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── src/
│   ├── optimization/
│   │   ├── quantize_model.py
│   │   ├── prune_model.py
│   │   └── onnx_optimize.py
│   ├── async_processing/
│   │   ├── celery_app.py
│   │   ├── tasks.py
│   │   └── api.py
│   └── monitoring/
│       ├── prometheus_metrics.py
│       └── middleware.py
├── load_balancing/
│   ├── nginx.conf
│   ├── haproxy.cfg
│   └── docker-compose.yml
├── benchmarks/
│   ├── locust_test.py
│   ├── ab_benchmark.sh
│   ├── wrk_scenarios/
│   └── performance_report.py
└── docker/
    ├── prometheus.yml
    └── grafana-dashboards/
        └── ml-api-dashboard.json
```

---

## Resources

### Documentation
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/)
- [Celery Best Practices](https://docs.celeryproject.org/en/stable/userguide/tasks.html#tips-and-best-practices)
- [Nginx Load Balancing](https://docs.nginx.com/nginx/admin-guide/load-balancer/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)

### Tools
- [Locust](https://locust.io/) - Load testing
- [Grafana](https://grafana.com/) - Metrics visualization
- [TensorRT](https://developer.nvidia.com/tensorrt) - GPU inference optimization
- [wrk](https://github.com/wg/wrk) - HTTP benchmarking

### Articles
- [Optimizing ML Inference](https://huggingface.co/docs/transformers/perf_infer_gpu_one)
- [Scaling FastAPI Applications](https://fastapi.tiangolo.com/deployment/)
- [Redis Caching Strategies](https://redis.io/docs/manual/patterns/)

---

## Next Steps

After mastering performance optimization:
- **Module 009: Monitoring & Observability** - Advanced monitoring patterns
- **Module 010: ML Infrastructure at Scale** - Large-scale ML systems
- Apply optimizations to real-world production systems
- Benchmark and profile your own ML models
- Implement custom optimization strategies

---

## Key Takeaways

1. **Model optimization** can provide 2-4x speedup with minimal accuracy loss
2. **Async processing** enables handling long-running tasks without blocking API
3. **Load balancing** distributes traffic and improves availability
4. **Multi-level caching** dramatically improves response times for repeated queries
5. **Monitoring** is essential for identifying bottlenecks and optimizing performance
6. **Horizontal scaling** provides linear scalability with proper architecture
7. **Benchmarking** validates optimization efforts and guides further improvements

**Performance is a feature** - Prioritize it from the start!
