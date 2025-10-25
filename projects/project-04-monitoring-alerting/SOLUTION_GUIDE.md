# Project 04: Monitoring & Alerting System - Solution Guide

## Overview

This solution implements a comprehensive observability stack for ML infrastructure using Prometheus, Grafana, Elasticsearch, Logstash, and Kibana (ELK Stack).

## Architecture

```
Applications → Prometheus (Metrics) → Grafana (Visualization)
         ↓
     Filebeat → Logstash → Elasticsearch → Kibana (Logs)
         ↓
   Alertmanager → Notifications (Email/Slack/PagerDuty)
```

## Key Components

### 1. Prometheus (`prometheus/prometheus.yml`)

**Configuration:**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'ml-cluster'
    environment: 'production'

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # ML API
  - job_name: 'model-api'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - ml-serving
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

  # Node Exporter
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # Kubernetes
  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
```

**Metrics Collected:**
- **Infrastructure**: CPU, memory, disk, network
- **Application**: Request rate, latency, errors
- **ML Models**: Inference time, predictions/sec, accuracy
- **Kubernetes**: Pod status, resource usage

### 2. Alert Rules (`prometheus/alerts.yml`)

**Infrastructure Alerts:**
```yaml
groups:
  - name: infrastructure
    interval: 30s
    rules:
      - alert: HighCPUUsage
        expr: 100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
          description: "CPU usage is {{ $value }}% on {{ $labels.instance }}"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Memory usage is {{ $value }}% on {{ $labels.instance }}"

      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 15
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space on {{ $labels.instance }}"
          description: "Only {{ $value }}% disk space remaining on {{ $labels.mountpoint }}"

      - alert: ServiceDown
        expr: up == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "{{ $labels.instance }} has been down for more than 2 minutes"
```

**Application Alerts:**
```yaml
  - name: application
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on {{ $labels.job }}"
          description: "Error rate is {{ $value | humanizePercentage }} on {{ $labels.instance }}"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency on {{ $labels.job }}"
          description: "P95 latency is {{ $value }}s on {{ $labels.instance }}"

      - alert: LowThroughput
        expr: rate(http_requests_total[5m]) < 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low throughput on {{ $labels.job }}"
          description: "Only {{ $value }} requests/sec on {{ $labels.instance }}"
```

**ML-Specific Alerts:**
```yaml
  - name: ml-models
    interval: 30s
    rules:
      - alert: ModelAccuracyDrop
        expr: model_accuracy < 0.85
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy dropped below 85%"
          description: "{{ $labels.model_name }} accuracy is {{ $value }}"

      - alert: DataDriftDetected
        expr: data_drift_score > 0.3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected on {{ $labels.model_name }}"
          description: "Drift score is {{ $value }}"

      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency on {{ $labels.model_name }}"
          description: "P95 inference latency is {{ $value }}s"

      - alert: LowPredictionConfidence
        expr: avg_over_time(prediction_confidence[10m]) < 0.7
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Low prediction confidence on {{ $labels.model_name }}"
          description: "Average confidence is {{ $value }}"
```

### 3. Alertmanager (`prometheus/alertmanager.yml`)

**Routing Configuration:**
```yaml
global:
  resolve_timeout: 5m
  slack_api_url: 'YOUR_SLACK_WEBHOOK'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true

    - match:
        severity: warning
      receiver: 'slack-warnings'

    - match:
        alertname: ModelAccuracyDrop
      receiver: 'ml-team'

receivers:
  - name: 'default'
    email_configs:
      - to: 'ops-team@example.com'
        from: 'alertmanager@example.com'
        smarthost: 'smtp.example.com:587'
        auth_username: 'alertmanager@example.com'
        auth_password: 'password'

  - name: 'slack-warnings'
    slack_configs:
      - channel: '#ml-alerts'
        title: 'Warning: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'

  - name: 'ml-team'
    email_configs:
      - to: 'ml-team@example.com'
    slack_configs:
      - channel: '#ml-team'
```

### 4. Grafana Dashboards

**Infrastructure Dashboard:**
- CPU usage per node
- Memory usage per node
- Disk usage
- Network I/O
- Pod count and status

**Application Dashboard:**
- Request rate
- Response time (P50, P95, P99)
- Error rate
- Active connections
- Request duration distribution

**ML Model Dashboard:**
```json
{
  "title": "ML Model Performance",
  "panels": [
    {
      "title": "Predictions per Second",
      "targets": [
        {
          "expr": "rate(predictions_total[5m])"
        }
      ]
    },
    {
      "title": "Inference Latency",
      "targets": [
        {
          "expr": "histogram_quantile(0.50, rate(model_inference_duration_seconds_bucket[5m]))",
          "legendFormat": "P50"
        },
        {
          "expr": "histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m]))",
          "legendFormat": "P95"
        },
        {
          "expr": "histogram_quantile(0.99, rate(model_inference_duration_seconds_bucket[5m]))",
          "legendFormat": "P99"
        }
      ]
    },
    {
      "title": "Model Accuracy",
      "targets": [
        {
          "expr": "model_accuracy"
        }
      ]
    },
    {
      "title": "Prediction Confidence Distribution",
      "targets": [
        {
          "expr": "histogram_quantile(0.50, rate(prediction_confidence_bucket[5m]))"
        }
      ]
    }
  ]
}
```

### 5. Application Instrumentation (`src/metrics.py`)

**Prometheus Metrics:**
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

PREDICTIONS_COUNT = Counter(
    'predictions_total',
    'Total predictions',
    ['model', 'status']
)

INFERENCE_DURATION = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model']
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Current model accuracy',
    ['model']
)

PREDICTION_CONFIDENCE = Histogram(
    'prediction_confidence',
    'Prediction confidence distribution',
    ['model']
)

# Instrument Flask app
from flask import request

@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - request.start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.endpoint,
        status=response.status_code
    ).inc()

    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.endpoint
    ).observe(duration)

    return response

# Instrument model inference
def predict_with_metrics(image, model_name):
    start = time.time()
    try:
        predictions = model_loader.predict(image)
        duration = time.time() - start

        PREDICTIONS_COUNT.labels(model=model_name, status='success').inc()
        INFERENCE_DURATION.labels(model=model_name).observe(duration)
        PREDICTION_CONFIDENCE.labels(model=model_name).observe(predictions[0]['confidence'])

        return predictions
    except Exception as e:
        PREDICTIONS_COUNT.labels(model=model_name, status='error').inc()
        raise

# Metrics endpoint
@app.route('/metrics')
def metrics():
    return generate_latest()
```

### 6. ELK Stack Configuration

**Logstash Pipeline (`logstash.conf`):**
```
input {
  beats {
    port => 5044
  }
}

filter {
  if [kubernetes] {
    mutate {
      add_field => {
        "pod_name" => "%{[kubernetes][pod][name]}"
        "namespace" => "%{[kubernetes][namespace]}"
      }
    }
  }

  json {
    source => "message"
    target => "log"
  }

  mutate {
    add_field => {
      "[@metadata][index_prefix]" => "ml-logs"
    }
  }

  date {
    match => ["timestamp", "ISO8601"]
    target => "@timestamp"
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "%{[@metadata][index_prefix]}-%{+YYYY.MM.dd}"
  }
}
```

## Deployment

```bash
# Deploy monitoring stack
kubectl apply -f monitoring/prometheus.yaml
kubectl apply -f monitoring/grafana.yaml
kubectl apply -f monitoring/alertmanager.yaml

# Access UIs
kubectl port-forward -n monitoring svc/prometheus 9090:9090
kubectl port-forward -n monitoring svc/grafana 3000:3000
kubectl port-forward -n monitoring svc/alertmanager 9093:9093
```

## Best Practices

### 1. Alert Design
- Alert on symptoms, not causes
- Include context in annotations
- Use appropriate severity levels
- Avoid alert fatigue

### 2. Metric Naming
- Use consistent naming conventions
- Include units in metric names
- Use labels for dimensions
- Don't create high-cardinality labels

### 3. Dashboard Design
- One purpose per dashboard
- Use consistent colors
- Include time ranges
- Add context and documentation

### 4. Log Management
- Use structured logging (JSON)
- Include request IDs for tracing
- Set appropriate log levels
- Implement log rotation

## Conclusion

This monitoring solution provides:
- **Comprehensive metrics** collection
- **Proactive alerting** for issues
- **Visual dashboards** for analysis
- **Centralized logging** for debugging
- **ML-specific monitoring** for models

The implementation demonstrates production observability practices essential for maintaining reliable ML systems.
