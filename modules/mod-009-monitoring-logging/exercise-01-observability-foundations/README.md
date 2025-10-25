# Exercise 01: Observability Foundations Lab - Complete Solution

**Module**: Monitoring & Logging Basics (Module 009)
**Difficulty**: Beginner → Intermediate
**Estimated Time**: 3-4 hours

## Overview

This solution demonstrates a production-ready FastAPI inference service with comprehensive observability:
- **Prometheus Metrics**: Four Golden Signals (latency, traffic, errors, saturation)
- **Structured Logging**: JSON logging with correlation IDs and trace context
- **OpenTelemetry Tracing**: Distributed tracing with automatic and manual instrumentation
- **SLIs/SLOs**: Defined service level indicators and objectives (99.5% availability, P99 < 300ms)
- **Health Checks**: Liveness and readiness endpoints

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Inference Gateway Service                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌─────────────┐  │
│  │   FastAPI    │─────▶│ PyTorch      │      │ Observability│  │
│  │   Endpoints  │      │ ResNet-50    │      │ Layer        │  │
│  │              │      │ Model        │      │              │  │
│  │ /predict     │      │              │      │ • Metrics    │  │
│  │ /health      │      │              │      │ • Logging    │  │
│  │ /ready       │      │              │      │ • Tracing    │  │
│  │ /metrics     │      │              │      │              │  │
│  └──────────────┘      └──────────────┘      └─────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │
                 ┌────────────┼────────────┐
                 │            │            │
                 ▼            ▼            ▼
         ┌───────────┐ ┌──────────┐ ┌──────────────┐
         │Prometheus │ │  Loki    │ │ Jaeger/Tempo │
         │(Metrics)  │ │ (Logs)   │ │  (Traces)    │
         └───────────┘ └──────────┘ └──────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- 4GB RAM minimum (for PyTorch model)

### 1. Setup Environment

```bash
# Create .env file from template
cp .env.example .env

# Review and customize settings
vim .env
```

### 2. Run with Docker Compose

```bash
# Start all services (app + Prometheus + Jaeger)
docker-compose up -d

# View logs
docker-compose logs -f inference-gateway

# Expected output:
# ✓ Inference Gateway starting on port 8000
# ✓ Prometheus metrics available at /metrics
# ✓ OpenTelemetry tracing enabled
# ✓ Health checks available at /health, /ready
```

### 3. Test the Service

```bash
# Health check
curl http://localhost:8000/health
# {"status":"healthy","timestamp":"2024-01-01T00:00:00Z"}

# Readiness check
curl http://localhost:8000/ready
# {"status":"ready","model_loaded":true}

# Make a prediction (requires an image)
curl -X POST http://localhost:8000/predict \
  -F "file=@test_image.jpg"

# View Prometheus metrics
curl http://localhost:8000/metrics
```

### 4. Access Monitoring UIs

- **Application**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Jaeger Tracing**: http://localhost:16686

## Service Level Objectives (SLOs)

### Availability SLO

- **Target**: 99.5% availability (4h 22m downtime/year)
- **Measurement Window**: Rolling 30 days
- **SLI**: `(successful_requests / total_requests) * 100`

### Latency SLO

- **Target**: P99 latency < 300ms for `/predict` endpoint
- **Measurement Window**: Rolling 7 days
- **SLI**: 99th percentile of request duration

### Error Budget

- **Monthly Error Budget**: 0.5% (216 minutes)
- **Daily Error Budget**: 7.2 minutes
- **Alert Threshold**: 50% error budget consumed

## Metrics Exposed

### Four Golden Signals

1. **Latency** (How long requests take)
   ```
   http_request_duration_seconds{method="POST",endpoint="/predict"}
   ```

2. **Traffic** (Rate of requests)
   ```
   http_requests_total{method="POST",endpoint="/predict"}
   ```

3. **Errors** (Rate of failed requests)
   ```
   http_requests_total{method="POST",endpoint="/predict",status="5xx"}
   ```

4. **Saturation** (How full the service is)
   ```
   inference_queue_size
   model_memory_usage_bytes
   ```

### Application-Specific Metrics

```
# Model Performance
model_inference_duration_seconds
model_predictions_total

# Resource Utilization
python_gc_objects_collected_total
process_resident_memory_bytes
process_cpu_seconds_total

# Business Metrics
predictions_by_class_total{class="cat"}
predictions_by_class_total{class="dog"}
```

## Logging Structure

All logs are structured JSON with the following fields:

```json
{
  "timestamp": "2024-01-01T12:00:00.123456Z",
  "level": "INFO",
  "logger": "inference_gateway",
  "message": "Prediction completed successfully",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "trace_id": "a1b2c3d4e5f6g7h8",
  "span_id": "12345678",
  "user_id": "user-123",
  "endpoint": "/predict",
  "method": "POST",
  "duration_ms": 245.5,
  "status_code": 200,
  "prediction_class": "golden_retriever",
  "confidence": 0.95
}
```

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages for non-critical issues
- **ERROR**: Error messages for failures
- **CRITICAL**: Critical errors requiring immediate attention

## Tracing

OpenTelemetry spans capture the complete request lifecycle:

```
Request [/predict]
├─ validate_input (5ms)
├─ preprocess_image (15ms)
├─ model_inference (200ms)
│  ├─ load_model (150ms)
│  └─ forward_pass (50ms)
└─ format_response (5ms)

Total: 225ms
```

### Trace Context Propagation

Trace context is automatically propagated:
- HTTP headers (W3C Trace Context)
- Logs (trace_id and span_id fields)
- Downstream services

## Project Structure

```
exercise-01-observability-foundations/
├── app/
│   ├── main.py                      # FastAPI application entry point
│   ├── api/
│   │   ├── routes.py                # API endpoints (/predict, /health, /ready)
│   │   └── dependencies.py          # Dependency injection
│   ├── core/
│   │   ├── config.py                # Configuration management (Pydantic Settings)
│   │   └── exceptions.py            # Custom exceptions
│   ├── instrumentation/
│   │   ├── metrics.py               # Prometheus metrics (counters, histograms, gauges)
│   │   ├── logging.py               # Structured logging setup (structlog + JSON)
│   │   ├── tracing.py               # OpenTelemetry tracing configuration
│   │   └── middleware.py            # Observability middleware
│   └── models/
│       └── inference.py             # PyTorch ResNet-50 inference logic
├── tests/
│   ├── unit/
│   │   ├── test_metrics.py
│   │   ├── test_logging.py
│   │   └── test_inference.py
│   └── integration/
│       └── test_api.py              # End-to-end API tests
├── docs/
│   ├── slo.md                       # SLO documentation and error budget tracking
│   ├── observability-readiness.md  # Production observability checklist
│   └── architecture.md              # Architecture diagrams and design decisions
├── config/
│   └── prometheus.yml               # Local Prometheus scrape configuration
├── scripts/
│   ├── setup.sh                     # Environment setup script
│   └── load_test.sh                 # Load testing with curl/wrk
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Development dependencies
├── Dockerfile                       # Multi-stage Docker build
├── docker-compose.yml               # Local development stack (app + Prometheus + Jaeger)
├── .env.example                     # Environment variable template
├── .dockerignore
├── pytest.ini                       # Pytest configuration
└── README.md
```

## Key Features Implemented

### 1. Prometheus Metrics

- ✅ HTTP request duration histogram
- ✅ HTTP request counter (with status codes)
- ✅ Model inference duration histogram
- ✅ Prediction class counter
- ✅ Queue size gauge
- ✅ Model memory usage gauge
- ✅ Error rate counter
- ✅ Cardinality best practices (limited label values)

### 2. Structured Logging

- ✅ JSON-formatted logs
- ✅ Correlation IDs (request_id)
- ✅ Trace context in logs (trace_id, span_id)
- ✅ Contextual metadata (endpoint, method, duration, status_code)
- ✅ Performance logging (request timing)
- ✅ Error logging with stack traces

### 3. OpenTelemetry Tracing

- ✅ Automatic FastAPI instrumentation
- ✅ Manual span creation for model inference
- ✅ Span attributes (model_name, prediction_class, confidence)
- ✅ Trace context propagation
- ✅ Jaeger exporter configuration
- ✅ Sampling strategy (always on for demo)

### 4. Health Checks

- ✅ Liveness probe (`/health`) - Is the service running?
- ✅ Readiness probe (`/ready`) - Is the service ready to accept traffic?
- ✅ Model warmup check
- ✅ Dependency health checks

### 5. Production Best Practices

- ✅ Multi-stage Docker build
- ✅ Non-root container user
- ✅ Graceful shutdown handling
- ✅ Configuration via environment variables
- ✅ Security headers
- ✅ CORS configuration
- ✅ Request timeout handling
- ✅ Error handling and retries

## Testing

### Unit Tests

```bash
# Run unit tests
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=app --cov-report=html
```

### Integration Tests

```bash
# Run integration tests (requires Docker)
docker-compose up -d
pytest tests/integration/ -v
```

### Load Testing

```bash
# Simple load test
./scripts/load_test.sh

# Advanced load testing with locust
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## Monitoring Queries

### Prometheus Queries

```promql
# Request rate
rate(http_requests_total[5m])

# P99 latency
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# SLO Compliance (99.5% availability)
(sum(rate(http_requests_total{status!~"5.."}[30d])) / sum(rate(http_requests_total[30d]))) * 100

# Error budget remaining
1 - ((sum(rate(http_requests_total{status=~"5.."}[30d])) / sum(rate(http_requests_total[30d]))) / 0.005)
```

## Configuration

### Environment Variables

```bash
# Application
APP_NAME=inference-gateway
APP_VERSION=1.0.0
LOG_LEVEL=INFO
WORKERS=4

# Model
MODEL_NAME=resnet50
MODEL_WARMUP=true

# Observability
ENABLE_METRICS=true
ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318
OTEL_SERVICE_NAME=inference-gateway

# Performance
MAX_QUEUE_SIZE=100
REQUEST_TIMEOUT=30
```

## Troubleshooting

### High Latency

```bash
# Check Jaeger traces
# Navigate to http://localhost:16686
# Search for slow requests (> 300ms)
# Identify bottleneck spans (preprocess, inference, etc.)
```

### Missing Metrics

```bash
# Verify Prometheus is scraping
curl http://localhost:8000/metrics | grep http_requests_total

# Check Prometheus targets
# Navigate to http://localhost:9090/targets
```

### Log Correlation

```bash
# Find all logs for a specific request
docker-compose logs inference-gateway | grep "request_id=550e8400-e29b-41d4-a716-446655440000"

# Find all logs for a trace
docker-compose logs inference-gateway | grep "trace_id=a1b2c3d4e5f6g7h8"
```

## Next Steps

After completing this exercise, you'll extend this observability foundation in:

1. **Exercise 02**: Deploy Prometheus stack with persistent storage
2. **Exercise 03**: Create Grafana dashboards for SLO monitoring
3. **Exercise 04**: Implement centralized logging with Loki/ELK
4. **Exercise 05**: Configure alerting and incident response

## Additional Resources

- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/instrumentation/python/)
- [Google SRE Book - SLIs, SLOs, SLAs](https://sre.google/sre-book/service-level-objectives/)
- [The Four Golden Signals](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Structured Logging Best Practices](https://www.structlog.org/en/stable/)

## License

MIT License - See LICENSE file for details
