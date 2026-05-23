# Step-by-Step Implementation Guide: Observability Foundations

## Overview

Build observability into ML infrastructure! Learn the three pillars (metrics, logs, traces), instrumentation, SLIs/SLOs, and observability best practices.

**Time**: 2 hours | **Difficulty**: Beginner to Intermediate

---

## Learning Objectives

✅ Understand observability fundamentals
✅ Implement metrics collection
✅ Set up centralized logging
✅ Add distributed tracing
✅ Define SLIs and SLOs
✅ Create dashboards
✅ Implement alerting

---

## Three Pillars of Observability

### 1. Metrics (What's happening)
- Quantitative measurements over time
- CPU, memory, request rate, latency
- Aggregatable and efficient

### 2. Logs (What happened)
- Discrete events with context
- Error messages, audit trails
- Searchable and filterable

### 3. Traces (How it happened)
- Request flow through system
- Performance bottlenecks
- Dependency mapping

---

## Metrics with Prometheus Client

```python
# app.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from flask import Flask, Response
import time

app = Flask(__name__)

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_REQUESTS = Gauge('http_requests_active', 'Active HTTP requests')
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total predictions', ['model_version'])
MODEL_LATENCY = Histogram('model_prediction_duration_seconds', 'Prediction latency')

@app.before_request
def before_request():
    ACTIVE_REQUESTS.inc()
    request.start_time = time.time()

@app.after_request
def after_request(response):
    ACTIVE_REQUESTS.dec()
    request_duration = time.time() - request.start_time
    REQUEST_DURATION.observe(request_duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.endpoint,
        status=response.status_code
    ).inc()
    return response

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()

    # Run inference
    result = model.predict(request.json)

    # Record metrics
    duration = time.time() - start
    MODEL_LATENCY.observe(duration)
    MODEL_PREDICTIONS.labels(model_version='v2.0').inc()

    return {'prediction': result}

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')
```

---

## Logging with Structlog

```python
import structlog
import logging

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()

# Structured logging
log.info("prediction_made",
    model_version="v2.0",
    prediction=result,
    latency_ms=latency * 1000,
    input_shape=input.shape
)

log.error("model_loading_failed",
    model_path=path,
    error=str(e),
    exc_info=True
)
```

---

## Distributed Tracing with OpenTelemetry

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Instrument code
@app.route('/predict', methods=['POST'])
def predict():
    with tracer.start_as_current_span("predict") as span:
        span.set_attribute("model.version", "v2.0")

        with tracer.start_as_current_span("preprocess"):
            data = preprocess(request.json)

        with tracer.start_as_current_span("inference"):
            result = model.predict(data)

        with tracer.start_as_current_span("postprocess"):
            output = postprocess(result)

        return output
```

---

## SLIs and SLOs

### Service Level Indicators (SLIs)

```yaml
# SLI definitions
slis:
  - name: availability
    description: "% of successful requests"
    query: "sum(rate(http_requests_total{status!~'5..'}[5m])) / sum(rate(http_requests_total[5m]))"

  - name: latency
    description: "95th percentile latency"
    query: "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"

  - name: error_rate
    description: "% of failed requests"
    query: "sum(rate(http_requests_total{status=~'5..'}[5m])) / sum(rate(http_requests_total[5m]))"
```

### Service Level Objectives (SLOs)

```yaml
# SLO targets
slos:
  - sli: availability
    target: 99.9%  # "Three nines"
    window: 30d

  - sli: latency
    target: 200ms  # p95
    window: 30d

  - sli: error_rate
    target: 0.1%
    window: 30d
```

---

## Health Checks

```python
@app.route('/health')
def health():
    """Liveness probe"""
    return {'status': 'healthy'}, 200

@app.route('/ready')
def ready():
    """Readiness probe"""
    checks = {
        'database': check_database(),
        'model_loaded': model is not None,
        'redis': check_redis()
    }

    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503

    return {'ready': all_ready, 'checks': checks}, status_code
```

---

## Best Practices

✅ Instrument all critical paths
✅ Use structured logging
✅ Define clear SLIs/SLOs
✅ Monitor error budgets
✅ Implement health checks
✅ Add context to logs
✅ Use consistent metric naming
✅ Tag metrics appropriately
✅ Implement distributed tracing
✅ Monitor the 4 golden signals (latency, traffic, errors, saturation)

---

**Observability Foundations mastered!** 👁️
