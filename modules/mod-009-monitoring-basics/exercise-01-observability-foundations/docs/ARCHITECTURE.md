# Architecture: Observability Foundations Lab

## Overview

This document details the architecture decisions, design patterns, and trade-offs for the observability-instrumented inference gateway. The service demonstrates production-ready observability using the three pillars: **metrics**, **logs**, and **traces**.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Observability Architecture](#observability-architecture)
3. [Design Decisions](#design-decisions)
4. [Data Flow](#data-flow)
5. [Technology Choices](#technology-choices)
6. [Performance Considerations](#performance-considerations)
7. [Security Architecture](#security-architecture)
8. [Scalability](#scalability)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Client Layer                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │   Web    │   │  Mobile  │   │   API    │   │ Internal │    │
│  │  Client  │   │   App    │   │  Client  │   │ Services │    │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘    │
└───────┼──────────────┼──────────────┼──────────────┼───────────┘
        │              │              │              │
        └──────────────┴──────────────┴──────────────┘
                           │
                    ┌──────▼──────┐
                    │ Load Balancer │
                    │  (ALB/NLB)    │
                    └──────┬────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
    ┌───▼───┐         ┌───▼───┐         ┌───▼───┐
    │ Pod 1 │         │ Pod 2 │         │ Pod 3 │
    │       │         │       │         │       │
    │ App   │         │ App   │         │ App   │
    │ +     │         │ +     │         │ +     │
    │ Model │         │ Model │         │ Model │
    └───┬───┘         └───┬───┘         └───┬───┘
        │                  │                  │
        │  Metrics (Pull)  │                  │
        ├──────────────────┴──────────────────┤
        │                                     │
        ▼                                     │
┌─────────────┐                              │
│ Prometheus  │                              │
│   Server    │                              │
└─────────────┘                              │
                                             │
        │  Traces (Push)                     │
        ├────────────────────────────────────┤
        │                                    │
        ▼                                    │
┌─────────────┐                             │
│   Jaeger    │                             │
│  Collector  │                             │
└─────────────┘                             │
                                            │
        │  Logs (Push)                      │
        └───────────────────────────────────┘
                │
        ┌───────▼────────┐
        │  Log Aggregator │
        │  (Loki/ELK)     │
        └─────────────────┘
```

### Component Breakdown

| Component | Technology | Purpose | Scalability |
|-----------|-----------|---------|-------------|
| API Gateway | FastAPI | HTTP endpoints, request routing | Horizontal (stateless) |
| Model Inference | PyTorch ResNet-50 | Image classification | Horizontal (model per pod) |
| Metrics Collection | Prometheus Client | Expose metrics | N/A (library) |
| Logging | structlog | Structured JSON logs | N/A (library) |
| Tracing | OpenTelemetry | Distributed tracing | N/A (library) |
| Metrics Storage | Prometheus | Time-series database | Vertical + federation |
| Trace Storage | Jaeger | Trace storage and UI | Horizontal (backends) |
| Log Storage | Loki/ELK | Centralized logging | Horizontal |

---

## Observability Architecture

### Three Pillars Integration

```
┌──────────────────────────────────────────────────────────────┐
│                   Inference Gateway Service                   │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 FastAPI Application                      │ │
│  │                                                          │ │
│  │  ┌──────────────┐     ┌──────────────┐                │ │
│  │  │  /predict    │     │   /health    │                │ │
│  │  │  /metrics    │     │   /ready     │                │ │
│  │  └──────┬───────┘     └──────┬───────┘                │ │
│  │         │                    │                         │ │
│  │         │                    │                         │ │
│  │  ┌──────▼────────────────────▼───────┐                │ │
│  │  │  Observability Middleware         │                │ │
│  │  │                                   │                │ │
│  │  │  ┌──────────┐  ┌───────┐  ┌────┐│                │ │
│  │  │  │ Metrics  │  │  Logs │  │Trace││                │ │
│  │  │  │ Record   │  │ Struct│  │ Ctx ││                │ │
│  │  │  └────┬─────┘  └───┬───┘  └─┬──┘│                │ │
│  │  │       │            │        │   │                │ │
│  │  └───────┼────────────┼────────┼───┘                │ │
│  │          │            │        │                     │ │
│  │          │            │        │                     │ │
│  │  ┌───────▼────────────▼────────▼───────┐            │ │
│  │  │     Model Inference Engine          │            │ │
│  │  │                                      │            │ │
│  │  │  - Preprocess image                 │            │ │
│  │  │  - Run ResNet-50 forward pass       │            │ │
│  │  │  - Extract predictions              │            │ │
│  │  │                                      │            │ │
│  │  │  [Instrumented with spans]          │            │ │
│  │  └──────────────────────────────────────┘            │ │
│  │                                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐      │
│  │ Prometheus  │  │  structlog  │  │ OpenTelemetry│      │
│  │   Client    │  │  (JSON fmt) │  │     SDK      │      │
│  │             │  │             │  │              │      │
│  │ /metrics    │  │  stdout     │  │  OTLP Export │      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘      │
└─────────┼─────────────────┼─────────────────┼──────────────┘
          │                 │                 │
          │ Pull            │ Stream          │ Push
          │ (15s)           │ (realtime)      │ (batch)
          │                 │                 │
    ┌─────▼─────┐    ┌──────▼──────┐   ┌─────▼─────┐
    │Prometheus │    │    Loki     │   │  Jaeger   │
    │  Server   │    │   / ELK     │   │ Collector │
    └───────────┘    └─────────────┘   └───────────┘
```

### Observability Data Types

#### 1. Metrics (Prometheus)

**What**: Numerical measurements over time
**When**: Always, low overhead
**Use Cases**: SLO tracking, alerting, capacity planning

```
Metric Types Used:
- Counter: http_requests_total, model_predictions_total
- Histogram: http_request_duration_seconds, model_inference_duration_seconds
- Gauge: inference_queue_size, model_memory_usage_bytes
```

#### 2. Logs (structlog + JSON)

**What**: Discrete events with context
**When**: Important events, errors, debugging
**Use Cases**: Debugging, audit trails, error investigation

```json
{
  "timestamp": "2025-10-24T12:00:00.123Z",
  "level": "INFO",
  "message": "prediction_completed",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "trace_id": "a1b2c3d4e5f6g7h8",
  "prediction": "golden_retriever",
  "confidence": 0.95,
  "duration_ms": 245.5
}
```

#### 3. Traces (OpenTelemetry)

**What**: Request flow across services and functions
**When**: All requests (with sampling in production)
**Use Cases**: Latency investigation, dependency mapping, performance optimization

```
Trace Structure:
/predict [POST] (250ms total)
├─ validate_input (5ms)
├─ preprocess_image (15ms)
├─ model_inference (200ms)
│  ├─ load_model (cached)
│  └─ forward_pass (195ms)
└─ format_response (5ms)
```

---

## Design Decisions

### 1. Why FastAPI?

**Decision**: Use FastAPI as the web framework

**Rationale**:
- **Performance**: Async/await support for high concurrency
- **Type Safety**: Pydantic models for request/response validation
- **Auto-Documentation**: OpenAPI/Swagger docs out-of-the-box
- **Observability**: Easy to integrate middleware for metrics/tracing
- **Modern Python**: Python 3.11+ with type hints

**Alternatives Considered**:
- Flask: Simpler but synchronous, no built-in validation
- Django: Overkill for a microservice, heavier
- Starlette: FastAPI is built on top of it

**Trade-offs**:
- ✅ Fast development, excellent DX
- ✅ Built-in async support
- ❌ Smaller ecosystem than Flask
- ❌ Learning curve for async patterns

### 2. Why Prometheus for Metrics?

**Decision**: Use Prometheus for metrics collection and storage

**Rationale**:
- **Industry Standard**: De facto standard for Kubernetes monitoring
- **Pull Model**: Scrapes metrics from services (simpler than push)
- **PromQL**: Powerful query language for aggregations
- **Label-Based**: Flexible dimensional data model
- **Ecosystem**: Grafana, Alertmanager integration

**Alternatives Considered**:
- StatsD: Push-based, requires additional collector
- InfluxDB: Time-series DB but requires more infrastructure
- CloudWatch: AWS-specific, less flexible

**Trade-offs**:
- ✅ No client-side buffering/queueing
- ✅ Built-in service discovery
- ✅ Excellent for Kubernetes
- ❌ Pull model requires network reachability
- ❌ Long-term storage requires federation/remote write

### 3. Why Structured Logging (JSON)?

**Decision**: Use structlog with JSON formatting

**Rationale**:
- **Machine-Readable**: Easy to parse and query
- **Contextual**: Include request_id, trace_id, user_id, etc.
- **Correlation**: Link logs to traces and metrics
- **Search**: Efficient log queries in Loki/Elasticsearch
- **Standards**: JSON is universal

**Alternatives Considered**:
- Plain text logs: Human-readable but hard to parse
- Binary formats (Protocol Buffers): Faster but less debuggable

**Trade-offs**:
- ✅ Easy to query and filter
- ✅ Rich context
- ❌ Larger log size (~2-3x vs plain text)
- ❌ Less human-readable in raw form

### 4. Why OpenTelemetry for Tracing?

**Decision**: Use OpenTelemetry SDK for distributed tracing

**Rationale**:
- **Vendor-Neutral**: Works with Jaeger, Zipkin, Tempo, etc.
- **Auto-Instrumentation**: Automatic FastAPI instrumentation
- **Context Propagation**: W3C Trace Context standard
- **Future-Proof**: Industry standard (CNCF project)
- **Multi-Signal**: Can also collect metrics and logs

**Alternatives Considered**:
- Jaeger client directly: Vendor lock-in
- AWS X-Ray: AWS-only
- DataDog APM: Commercial, expensive

**Trade-offs**:
- ✅ Vendor-neutral, portable
- ✅ Rich ecosystem
- ❌ More complex setup than vendor SDKs
- ❌ Performance overhead (~1-5ms per request)

### 5. Why ResNet-50?

**Decision**: Use ResNet-50 for image classification

**Rationale**:
- **Well-Known**: Standard benchmark model
- **Pretrained**: ImageNet weights available
- **Moderate Size**: 25M parameters (not too large)
- **Good Accuracy**: 76% top-1 on ImageNet
- **Fast Inference**: ~100-200ms on CPU

**Alternatives Considered**:
- MobileNet: Smaller but less accurate
- EfficientNet: More accurate but slower
- Vision Transformer: State-of-the-art but very slow on CPU

**Trade-offs**:
- ✅ Good balance of speed and accuracy
- ✅ Fits in memory easily
- ❌ Not state-of-the-art
- ❌ Requires preprocessing

---

## Data Flow

### Request Flow with Observability

```
1. Client Request
   ↓
2. Load Balancer
   ↓
3. ObservabilityMiddleware.dispatch()
   │
   ├─ Generate request_id
   ├─ Start timer
   ├─ Increment inference_queue_size (Gauge)
   ├─ Log "request_received"
   │
   ↓
4. FastAPI Route Handler (/predict)
   │
   ├─ OpenTelemetry creates root span
   ├─ Validate file upload
   │
   ↓
5. classifier.predict()
   │
   ├─ Create "preprocess_image" span
   ├─ Preprocess image (Pillow + transforms)
   ├─ Close span
   │
   ├─ Create "model_inference" span
   ├─ Run model.forward() [PyTorch]
   ├─ Record model_inference_duration_seconds (Histogram)
   ├─ Increment model_predictions_total (Counter)
   ├─ Observe model_prediction_confidence (Histogram)
   ├─ Log "prediction_completed"
   ├─ Close span
   │
   ↓
6. Return Response
   ↓
7. ObservabilityMiddleware (continued)
   │
   ├─ Calculate duration
   ├─ Increment http_requests_total (Counter)
   ├─ Observe http_request_duration_seconds (Histogram)
   ├─ Decrement inference_queue_size (Gauge)
   ├─ Log "request_completed"
   ├─ Add X-Request-ID header
   │
   ↓
8. Response to Client
   ↓
9. Background Export
   │
   ├─ Prometheus scrapes /metrics (every 15s)
   ├─ OpenTelemetry exports spans to Jaeger (batched)
   ├─ Logs streamed to Loki/stdout
```

### Metrics Cardinality

**High Cardinality** (avoid):
- ❌ User IDs as labels
- ❌ Request IDs as labels
- ❌ Timestamps as labels

**Low Cardinality** (good):
- ✅ HTTP method: GET, POST, PUT, DELETE (4 values)
- ✅ Endpoint: /predict, /health, /ready (3 values)
- ✅ Status: 2xx, 3xx, 4xx, 5xx (4 values)
- ✅ Model name: resnet50 (1 value)

**Total Cardinality Calculation**:
```
http_request_duration_seconds:
  method (4) × endpoint (3) × status (4) = 48 time series

model_predictions_total:
  model (1) × prediction_class (1000) = 1000 time series

Total: ~1,050 active time series (well within limits)
```

---

## Technology Choices

### Python Libraries

| Library | Version | Purpose | Why Chosen |
|---------|---------|---------|------------|
| FastAPI | 0.109.0 | Web framework | Modern, fast, auto-docs |
| PyTorch | 2.1.2 | ML framework | Industry standard, flexible |
| prometheus-client | 0.19.0 | Metrics | Official Prometheus client |
| structlog | 24.1.0 | Structured logging | Best Python structured logging |
| opentelemetry-sdk | 1.22.0 | Tracing | Vendor-neutral standard |
| pydantic | 2.5.3 | Validation | Type-safe config/models |

### Infrastructure Choices

| Component | Technology | Why Chosen |
|-----------|-----------|------------|
| Container Runtime | Docker | Universal, portable |
| Orchestration | Kubernetes | Cloud-native standard |
| Metrics Backend | Prometheus | K8s standard, PromQL |
| Tracing Backend | Jaeger | Open-source, CNCF project |
| Logs Backend | Loki | LogQL, Grafana integration |
| Dashboards | Grafana | Best-in-class, free |
| Alerts | Alertmanager | Native Prometheus integration |

---

## Performance Considerations

### Latency Breakdown

**Target P99 Latency**: < 300ms

```
Typical Request Breakdown:
┌──────────────────────────────┬──────────┬──────────┐
│ Component                    │ Latency  │ % Total  │
├──────────────────────────────┼──────────┼──────────┤
│ Network (client → server)    │ 10ms     │ 4%       │
│ Load Balancer               │ 5ms      │ 2%       │
│ FastAPI routing             │ 2ms      │ 1%       │
│ Request validation          │ 3ms      │ 1%       │
│ Image preprocessing         │ 15ms     │ 6%       │
│ Model inference (ResNet-50) │ 180ms    │ 72%      │
│ Response formatting         │ 5ms      │ 2%       │
│ Observability overhead      │ 5ms      │ 2%       │
│ Network (server → client)   │ 10ms     │ 4%       │
├──────────────────────────────┼──────────┼──────────┤
│ Total P50                   │ ~235ms   │ 100%     │
│ Total P99                   │ ~280ms   │          │
└──────────────────────────────┴──────────┴──────────┘
```

**Optimization Opportunities**:
1. **Model Optimization**: Use TorchScript or ONNX Runtime (-30% latency)
2. **Batching**: Group requests for batch inference (-40% latency at scale)
3. **GPU**: Use GPU for inference (-60% latency)
4. **Model Distillation**: Use smaller model like MobileNet (-50% latency)

### Observability Overhead

```
Baseline (no observability):     210ms P99
+ Prometheus metrics:            +2ms  (+1%)
+ Structured logging:            +1ms  (+0.5%)
+ OpenTelemetry tracing:         +3ms  (+1.5%)
─────────────────────────────────────────────
Total with observability:        216ms P99
Total overhead:                  +6ms  (+3%)
```

**Acceptable overhead**: < 5% is industry standard

---

## Security Architecture

### Authentication & Authorization (Future)

```
┌──────────┐
│  Client  │
└────┬─────┘
     │
     │ 1. JWT Token
     ↓
┌──────────────┐
│  API Gateway │
│              │
│  ┌────────┐  │
│  │  Auth  │  │ 2. Validate JWT
│  │Middleware│  │
│  └────────┘  │
└──────┬───────┘
       │
       │ 3. Authorized Request
       ↓
   [Route Handler]
```

### Secrets Management

- **Development**: `.env` file (gitignored)
- **Staging/Production**:
  - Kubernetes Secrets
  - AWS Secrets Manager
  - HashiCorp Vault

### Network Security

- **TLS/HTTPS**: Terminate at load balancer
- **Network Policies**: Restrict pod-to-pod communication
- **Security Groups**: Firewall rules for egress/ingress

---

## Scalability

### Horizontal Scaling

**Stateless Design**:
- Each pod loads its own model copy
- No shared state between instances
- Scale based on CPU/memory/request rate

**Auto-Scaling Triggers**:
1. CPU > 70%
2. Memory > 80%
3. Request rate > 1000 req/s per pod
4. Queue size > 50

**Scaling Limits**:
- Min replicas: 3 (production)
- Max replicas: 20
- Scale up: +100% every 30s
- Scale down: -50% every 5min (with stabilization)

### Vertical Scaling

**Pod Resources**:
```yaml
Requests:
  CPU: 2 cores
  Memory: 4GB

Limits:
  CPU: 4 cores
  Memory: 8GB
```

**Why these values**:
- Model size: ~100MB
- PyTorch overhead: ~1GB
- Request buffers: ~500MB
- Headroom: 2.5GB

---

## Cost Analysis

### Per-Request Cost Breakdown

```
Infrastructure Cost (Kubernetes):
- 1 pod on m5.xlarge: $0.19/hr = $137/month
- 1000 requests/hour capacity
- Cost per 1000 requests: $0.19

Model Inference:
- CPU time: 180ms × $0.000048/vCPU-second = $0.00000864
- Cost per request: $0.0000086

Observability:
- Metrics storage (1 month): $0.001 per 1000 metrics
- Log storage (1 month): $0.50 per GB
- Trace storage (1 month): $0.20 per GB
- Cost per request: ~$0.00001

Total cost per request: $0.00002
Total cost per 1M requests: $20
```

### Cost Optimization

1. **Use Spot Instances**: Save 70% on compute
2. **Compress Logs**: Save 50% on log storage
3. **Sample Traces**: Sample 10% = save 90% on trace storage
4. **Batch Inference**: Reduce compute time by 40%
5. **Use GPU for High Volume**: Amortize GPU cost over many requests

---

## Summary

**Architecture Principles**:
- ✅ Stateless design for horizontal scaling
- ✅ Comprehensive observability (metrics, logs, traces)
- ✅ Production-ready (health checks, graceful shutdown)
- ✅ Modular (easy to swap components)
- ✅ Standards-based (OpenTelemetry, Prometheus)

**Key Metrics**:
- Latency: P99 < 300ms
- Availability: 99.5% SLO
- Observability overhead: < 5%
- Scalability: 3-20 replicas (auto)

**Future Enhancements**:
- Add authentication/authorization
- Implement request batching
- Add model versioning
- Deploy on GPU instances
- Implement A/B testing framework
