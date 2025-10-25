# Troubleshooting Guide: Observability Foundations Lab

## Overview

This guide covers common issues, debugging techniques, and solutions for the observability-instrumented inference gateway.

---

## Table of Contents

1. [Application Issues](#application-issues)
2. [Model Loading Issues](#model-loading-issues)
3. [Metrics Issues](#metrics-issues)
4. [Logging Issues](#logging-issues)
5. [Tracing Issues](#tracing-issues)
6. [Performance Issues](#performance-issues)
7. [Docker/Kubernetes Issues](#dockerkubernetes-issues)
8. [Network Issues](#network-issues)

---

## Application Issues

### Issue: Application Won't Start

**Symptoms**:
```
Error: ModuleNotFoundError: No module named 'fastapi'
```

**Diagnosis**:
```bash
# Check if dependencies are installed
pip list | grep fastapi

# Check Python version
python --version  # Should be 3.11+
```

**Solution**:
```bash
# Install dependencies
pip install -r requirements.txt

# Or rebuild Docker image
docker-compose build --no-cache
```

---

### Issue: Port Already in Use

**Symptoms**:
```
Error: [Errno 48] Address already in use
```

**Diagnosis**:
```bash
# Check what's using port 8000
lsof -i :8000

# Or on Linux
netstat -tulpn | grep 8000
```

**Solution**:
```bash
# Kill the process using the port
kill -9 <PID>

# Or change the port in .env
METRICS_PORT=8001
```

---

### Issue: Application Crashes on Startup

**Symptoms**:
```
docker-compose logs inference-gateway
# Shows exit code 137 or immediate restarts
```

**Diagnosis**:
```bash
# Check container resources
docker stats inference-gateway

# Check logs for OOM (Out of Memory)
docker-compose logs | grep -i "killed\|oom\|memory"
```

**Solution**:
```yaml
# Increase Docker memory limit
services:
  inference-gateway:
    deploy:
      resources:
        limits:
          memory: 6GB  # Increase from 4GB
```

---

## Model Loading Issues

### Issue: Model Not Loading

**Symptoms**:
```
GET /ready
{"status":"not_ready","model_loaded":false}
```

**Diagnosis**:
```bash
# Check logs for model loading
docker-compose logs inference-gateway | grep "loading_model"

# Check available memory
docker stats inference-gateway
```

**Root Causes**:
1. **Insufficient Memory**: ResNet-50 requires ~2GB RAM
2. **Network Issues**: Can't download pretrained weights
3. **Disk Space**: No space to cache model

**Solution 1: Increase Memory**:
```yaml
services:
  inference-gateway:
    deploy:
      resources:
        limits:
          memory: 6GB  # Increase
```

**Solution 2: Preload Model Weights**:
```bash
# Download weights manually
python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)"

# Cache in Docker image
COPY --from=builder /root/.cache /root/.cache
```

**Solution 3: Disable Model Warmup** (for testing):
```bash
# In .env
MODEL_WARMUP=false

# Model will load on first /predict request instead
```

---

### Issue: Model Inference Slow

**Symptoms**:
```
# P99 latency > 1 second
curl http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(model_inference_duration_seconds_bucket[5m]))
```

**Diagnosis**:
```bash
# Check CPU usage
docker stats

# Check if using CPU instead of GPU
docker-compose logs | grep "device=cpu"

# Check traces in Jaeger
open http://localhost:16686
# Look for slow "model_inference" spans
```

**Solutions**:

**1. Use GPU**:
```bash
# Install NVIDIA Docker runtime
# Update .env
MODEL_DEVICE=cuda

# Update docker-compose.yml
services:
  inference-gateway:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**2. Optimize Model**:
```python
# In app/models/inference.py
# Add TorchScript compilation
self.model = torch.jit.script(self.model)

# Or use ONNX Runtime
import onnxruntime as ort
```

**3. Reduce Image Size**:
```python
# Smaller input images
transforms.Resize(128)  # Instead of 256
transforms.CenterCrop(112)  # Instead of 224
```

---

### Issue: Prediction Errors

**Symptoms**:
```
POST /predict
500 Internal Server Error
{"detail":"Prediction failed: ..."}
```

**Diagnosis**:
```bash
# Check logs with full stack traces
docker-compose logs inference-gateway | grep -A 20 "prediction_failed"

# Check Jaeger for failed traces
# Filter by status=error
```

**Common Causes**:

**1. Invalid Image Format**:
```
Error: cannot identify image file
```

**Solution**:
```python
# Add better error handling in preprocess_image
try:
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
except Exception as e:
    raise InvalidInputError(f"Invalid image format: {e}")
```

**2. Image Too Large**:
```
Error: Exceeded max file size
```

**Solution**:
```python
# Add file size validation
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if len(image_bytes) > MAX_FILE_SIZE:
    raise InvalidInputError(f"Image too large. Max size: 10MB")
```

**3. Model Not Loaded**:
```
Error: Model not loaded
```

**Solution**:
```bash
# Check readiness endpoint
curl http://localhost:8000/ready

# If not ready, check model loading logs
docker-compose logs | grep "loading_model"
```

---

## Metrics Issues

### Issue: Metrics Endpoint Not Working

**Symptoms**:
```
curl http://localhost:8000/metrics
404 Not Found
```

**Diagnosis**:
```bash
# Check if metrics are enabled
curl http://localhost:8000/ | jq
# Look for ENABLE_METRICS setting

# Check routes
curl http://localhost:8000/docs
# Verify /metrics is listed
```

**Solution**:
```bash
# Enable metrics in .env
ENABLE_METRICS=true

# Restart service
docker-compose restart inference-gateway
```

---

### Issue: No Metrics Appearing

**Symptoms**:
```
curl http://localhost:8000/metrics
# Returns empty or minimal metrics
```

**Diagnosis**:
```bash
# Check if any requests have been made
curl -X POST http://localhost:8000/predict -F "file=@test_image.jpg"

# Then check metrics again
curl http://localhost:8000/metrics | grep http_requests_total
```

**Solution**:
Metrics are only recorded after requests. Send some test traffic:

```bash
# Generate test traffic
for i in {1..10}; do
  curl -X POST http://localhost:8000/predict \
    -F "file=@test_image.jpg" \
    -s -o /dev/null
done

# Verify metrics
curl http://localhost:8000/metrics | grep -E "http_requests_total|model_predictions_total"
```

---

### Issue: Prometheus Not Scraping

**Symptoms**:
```
# In Prometheus UI (http://localhost:9090/targets)
# Target shows as DOWN
```

**Diagnosis**:
```bash
# Check if Prometheus can reach the app
docker exec prometheus wget -O- http://inference-gateway:8000/metrics

# Check Prometheus config
docker exec prometheus cat /etc/prometheus/prometheus.yml
```

**Solutions**:

**1. Fix Network Connectivity**:
```yaml
# Ensure services are on same network
networks:
  monitoring:
    driver: bridge

# Both services must use this network
services:
  inference-gateway:
    networks:
      - monitoring
  prometheus:
    networks:
      - monitoring
```

**2. Fix Target Configuration**:
```yaml
# config/prometheus.yml
scrape_configs:
  - job_name: 'inference-gateway'
    static_configs:
      - targets: ['inference-gateway:8000']  # Use service name, not localhost
```

**3. Reload Prometheus Config**:
```bash
# Reload without restart
curl -X POST http://localhost:9090/-/reload

# Or restart
docker-compose restart prometheus
```

---

### Issue: High Cardinality Warnings

**Symptoms**:
```
# Prometheus logs show warnings
level=warn msg="Many time series in metric"
```

**Diagnosis**:
```bash
# Check cardinality
curl 'http://localhost:9090/api/v1/label/__name__/values' | jq

# Check specific metric cardinality
promtool tsdb analyze /prometheus
```

**Solution**:
Remove high-cardinality labels like user_id, request_id:

```python
# WRONG
http_requests_total.labels(
    user_id=user_id,  # ❌ High cardinality
    request_id=request_id  # ❌ High cardinality
).inc()

# CORRECT
http_requests_total.labels(
    endpoint="/predict",  # ✅ Low cardinality
    method="POST",  # ✅ Low cardinality
    status="2xx"  # ✅ Low cardinality
).inc()
```

---

## Logging Issues

### Issue: No Logs Appearing

**Symptoms**:
```
docker-compose logs inference-gateway
# Very minimal or no application logs
```

**Diagnosis**:
```bash
# Check log level
docker-compose exec inference-gateway env | grep LOG_LEVEL

# Check if logging is configured
docker-compose exec inference-gateway python -c "import app.instrumentation.logging"
```

**Solution**:
```bash
# Set appropriate log level in .env
LOG_LEVEL=INFO  # Or DEBUG for more verbose

# Restart
docker-compose restart inference-gateway
```

---

### Issue: Logs Not Structured (Not JSON)

**Symptoms**:
```
docker-compose logs inference-gateway
# Shows plain text instead of JSON
```

**Diagnosis**:
```bash
# Check LOG_FORMAT setting
docker-compose exec inference-gateway env | grep LOG_FORMAT
```

**Solution**:
```bash
# Set JSON format in .env
LOG_FORMAT=json

# Restart
docker-compose restart inference-gateway
```

---

### Issue: Missing Trace Context in Logs

**Symptoms**:
```json
{
  "timestamp": "...",
  "message": "...",
  // Missing trace_id and span_id
}
```

**Diagnosis**:
```bash
# Check if tracing is enabled
curl http://localhost:8000/ | jq '.enable_tracing'
```

**Solution**:
```bash
# Enable tracing in .env
ENABLE_TRACING=true

# Verify trace context processor is working
docker-compose logs | grep trace_id
```

---

## Tracing Issues

### Issue: No Traces in Jaeger

**Symptoms**:
```
# Jaeger UI shows no traces
http://localhost:16686
```

**Diagnosis**:
```bash
# Check if Jaeger is running
docker-compose ps jaeger

# Check if app can reach Jaeger
docker-compose exec inference-gateway ping jaeger

# Check Jaeger logs
docker-compose logs jaeger
```

**Solutions**:

**1. Fix OTLP Endpoint**:
```bash
# In .env
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318  # Not localhost!

# Restart
docker-compose restart inference-gateway
```

**2. Enable Tracing**:
```bash
# In .env
ENABLE_TRACING=true
```

**3. Send Test Requests**:
```bash
# Traces only appear after requests
curl -X POST http://localhost:8000/predict -F "file=@test_image.jpg"

# Wait 10-30 seconds for batch export
# Then check Jaeger UI
```

---

### Issue: Incomplete Traces (Missing Spans)

**Symptoms**:
```
# Jaeger shows trace but missing some spans
# Expected: request → preprocess → inference → response
# Actual: request → response (missing middle spans)
```

**Diagnosis**:
```bash
# Check if manual spans are being created
grep "tracer.start_as_current_span" app/models/inference.py
```

**Solution**:
Ensure spans are properly created and closed:

```python
# WRONG
span = tracer.start_span("model_inference")
# ... work ...
# Forgot to close!

# CORRECT
with tracer.start_as_current_span("model_inference") as span:
    # ... work ...
    span.set_attribute("key", "value")
# Automatically closed
```

---

### Issue: High Tracing Overhead

**Symptoms**:
```
# P99 latency increased significantly with tracing enabled
Without tracing: 200ms
With tracing: 250ms (+25% overhead)
```

**Diagnosis**:
```bash
# Compare metrics with/without tracing
# Disable tracing
ENABLE_TRACING=false

# Send 1000 requests, measure P99
# Re-enable tracing
ENABLE_TRACING=true

# Send 1000 requests, measure P99
```

**Solution**:

**1. Use Sampling** (production):
```python
# In app/instrumentation/tracing.py
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

sampler = TraceIdRatioBased(0.1)  # Sample 10% of traces
tracer_provider = TracerProvider(resource=resource, sampler=sampler)
```

**2. Reduce Span Attributes**:
```python
# Remove expensive attributes
# span.set_attribute("large_data", huge_dict)  # ❌ Don't do this
```

---

## Performance Issues

### Issue: High Latency

**Symptoms**:
```
# P99 > 1 second
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1.0
```

**Diagnosis**:
```bash
# Check Jaeger for slow traces
open http://localhost:16686
# Sort by duration, look for slowest traces

# Check which span is slow
# Common culprits:
# - model_inference (model too slow)
# - preprocess_image (image too large)
# - wait_time (queueing)
```

**Solutions**:

**1. Optimize Model Inference**:
```python
# Use TorchScript
model = torch.jit.script(model)

# Use ONNX Runtime
import onnxruntime

# Use quantization
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

**2. Scale Horizontally**:
```bash
# Increase replicas
kubectl scale deployment inference-gateway --replicas=10
```

**3. Use GPU**:
```bash
MODEL_DEVICE=cuda
```

---

### Issue: High Memory Usage

**Symptoms**:
```
docker stats
# Shows > 90% memory usage

# Or OOMKilled errors
kubectl get pods
# Shows CrashLoopBackOff
kubectl describe pod inference-gateway-xxx
# Shows "OOMKilled"
```

**Diagnosis**:
```bash
# Check model memory usage
curl http://localhost:8000/metrics | grep model_memory_usage_bytes

# Check Python memory usage
docker-compose exec inference-gateway python -c "
import psutil
print(f'Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.0f}MB')
"
```

**Solutions**:

**1. Increase Memory Limits**:
```yaml
# Kubernetes
resources:
  limits:
    memory: 8Gi  # Increase from 4Gi
```

**2. Use Smaller Model**:
```python
# Use MobileNet instead of ResNet-50
from torchvision.models import mobilenet_v2
model = mobilenet_v2(pretrained=True)
```

**3. Clear Cache**:
```python
# Add to inference code
import gc
import torch

# After prediction
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()
```

---

### Issue: High CPU Usage

**Symptoms**:
```
docker stats
# Shows 100% CPU usage

# Prometheus query
container_cpu_usage_seconds_total > 0.9
```

**Diagnosis**:
```bash
# Check number of workers
docker-compose exec inference-gateway env | grep WORKERS

# Check request rate
curl 'http://localhost:9090/api/v1/query?query=rate(http_requests_total[1m])'
```

**Solutions**:

**1. Increase CPU Limits**:
```yaml
resources:
  limits:
    cpu: "4"  # Increase from 2
```

**2. Reduce Workers**:
```bash
# In .env
WORKERS=4  # Reduce if too many
```

**3. Implement Request Batching**:
```python
# Batch multiple requests together
# Process 10 images at once instead of 1
```

---

## Docker/Kubernetes Issues

### Issue: Container Keeps Restarting

**Symptoms**:
```
docker-compose ps
# Shows "Restarting" status

kubectl get pods
# Shows CrashLoopBackOff
```

**Diagnosis**:
```bash
# Check logs
docker-compose logs --tail=100 inference-gateway

# Check exit code
docker inspect inference-gateway | grep ExitCode

# Common exit codes:
# 137 = OOMKilled
# 1 = Application error
# 0 = Clean exit (shouldn't restart)
```

**Solutions**: See specific error in logs

---

### Issue: Health Checks Failing

**Symptoms**:
```
kubectl describe pod inference-gateway-xxx
# Shows "Liveness probe failed" or "Readiness probe failed"
```

**Diagnosis**:
```bash
# Test health endpoint manually
curl -v http://localhost:8000/health

# Check probe configuration
kubectl get pod inference-gateway-xxx -o yaml | grep -A 10 livenessProbe
```

**Solutions**:

**1. Increase initialDelaySeconds**:
```yaml
livenessProbe:
  initialDelaySeconds: 90  # Increase from 30 (model loading takes time)
```

**2. Increase timeoutSeconds**:
```yaml
readinessProbe:
  timeoutSeconds: 10  # Increase from 5
```

**3. Fix Health Endpoint**:
```bash
# Check if endpoint actually works
curl -v http://localhost:8000/health
```

---

## Network Issues

### Issue: Can't Access Application

**Symptoms**:
```
curl http://localhost:8000/health
curl: (7) Failed to connect to localhost port 8000: Connection refused
```

**Diagnosis**:
```bash
# Check if container is running
docker-compose ps

# Check port mapping
docker-compose ps inference-gateway | grep 8000

# Check if app is listening
docker-compose exec inference-gateway netstat -tulpn | grep 8000
```

**Solutions**:

**1. Fix Port Mapping**:
```yaml
services:
  inference-gateway:
    ports:
      - "8000:8000"  # host:container
```

**2. Use Correct Host**:
```bash
# If using Docker Compose
curl http://localhost:8000/health

# If using Kubernetes
kubectl port-forward deployment/inference-gateway 8000:8000
curl http://localhost:8000/health
```

---

## Debugging Workflow

### Step 1: Check Service Status

```bash
# Docker Compose
docker-compose ps
docker-compose logs --tail=50 inference-gateway

# Kubernetes
kubectl get pods -l app=inference-gateway
kubectl logs -l app=inference-gateway --tail=50
```

### Step 2: Check Health Endpoints

```bash
curl -v http://localhost:8000/health
curl -v http://localhost:8000/ready
curl -v http://localhost:8000/metrics
```

### Step 3: Check Observability Stack

```bash
# Prometheus targets
open http://localhost:9090/targets

# Jaeger traces
open http://localhost:16686

# Check recent logs
docker-compose logs --tail=100 inference-gateway | jq
```

### Step 4: Test Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg" \
  -v
```

### Step 5: Check Metrics/Traces/Logs

```bash
# Metrics
curl http://localhost:8000/metrics | grep -E "http_requests|model_predictions"

# Traces (in Jaeger UI)
# Search for service: inference-gateway
# Look for errors or slow traces

# Logs
docker-compose logs inference-gateway | grep -E "ERROR|WARNING|prediction_failed"
```

---

## Getting Help

### Check Application Logs

```bash
# All logs
docker-compose logs inference-gateway

# Follow logs
docker-compose logs -f inference-gateway

# Last 100 lines
docker-compose logs --tail=100 inference-gateway

# Filter for errors
docker-compose logs inference-gateway | grep ERROR

# JSON logs (if LOG_FORMAT=json)
docker-compose logs inference-gateway | jq 'select(.level=="ERROR")'
```

### Collect Diagnostic Information

```bash
#!/bin/bash
# diagnostics.sh

echo "=== Container Status ==="
docker-compose ps

echo "=== Container Stats ==="
docker stats --no-stream

echo "=== Application Logs ==="
docker-compose logs --tail=100 inference-gateway

echo "=== Health Checks ==="
curl -s http://localhost:8000/health | jq
curl -s http://localhost:8000/ready | jq

echo "=== Metrics Sample ==="
curl -s http://localhost:8000/metrics | head -50

echo "=== Prometheus Targets ==="
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'
```

---

## Summary

**Common Issues Checklist**:
- [ ] Check container is running (`docker-compose ps`)
- [ ] Check logs for errors (`docker-compose logs`)
- [ ] Check health endpoints (`/health`, `/ready`)
- [ ] Check metrics are being exported (`/metrics`)
- [ ] Check Prometheus is scraping (http://localhost:9090/targets)
- [ ] Check traces are appearing in Jaeger
- [ ] Check resource usage (`docker stats`)
- [ ] Verify network connectivity between services
- [ ] Ensure environment variables are set correctly
- [ ] Check model is loaded (`/ready` returns `model_loaded:true`)

**Escalation Path**:
1. Check application logs
2. Check Jaeger traces for slow/failing requests
3. Check Prometheus metrics for anomalies
4. Collect diagnostics with script above
5. Create GitHub issue with diagnostics
