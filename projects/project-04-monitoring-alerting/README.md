# Project 04: Monitoring & Alerting System - Solution

Comprehensive observability stack for ML infrastructure with Prometheus, Grafana, ELK Stack, and Alertmanager.

## Quick Start

```bash
# Deploy monitoring stack
kubectl apply -f prometheus/
kubectl apply -f grafana/
kubectl apply -f alertmanager/

# Or using docker-compose
docker-compose up -d

# Access UIs
kubectl port-forward -n monitoring svc/prometheus 9090:9090
kubectl port-forward -n monitoring svc/grafana 3000:3000
kubectl port-forward -n monitoring svc/alertmanager 9093:9093

# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Alertmanager: http://localhost:9093
```

## Features

- **Metrics Collection**: Prometheus scrapes from all services
- **Visualization**: Grafana dashboards for all components
- **Alerting**: 12+ alert rules with smart routing
- **Log Aggregation**: ELK Stack for centralized logging
- **ML Monitoring**: Custom metrics for model performance
- **Notifications**: Slack, Email, PagerDuty integration

## Architecture

```
Applications → Prometheus → Grafana
           ↓
       Filebeat → Logstash → Elasticsearch → Kibana
           ↓
    Alertmanager → Notifications
```

## Dashboards

### 1. Infrastructure Dashboard
- CPU usage per node
- Memory usage per node
- Disk I/O and space
- Network traffic

### 2. Application Dashboard
- Request rate (req/sec)
- Response time (P50, P95, P99)
- Error rate
- Active connections

### 3. ML Model Dashboard
- Predictions per second
- Inference latency
- Model accuracy
- Data drift scores
- Prediction confidence

## Alert Rules

### Infrastructure (4 alerts)
- High CPU usage (>80%)
- High memory usage (>85%)
- Low disk space (<15%)
- Service down

### Application (4 alerts)
- High error rate (>5%)
- High latency (P95 >500ms)
- Low throughput (<1 req/sec)
- High response time

### ML Models (4 alerts)
- Model accuracy drop (<85%)
- Data drift detected
- High inference latency (>1s)
- Low prediction confidence (<70%)

## Project Structure

```
project-04-monitoring-alerting/
├── prometheus/
│   ├── prometheus.yml      # Prometheus config
│   └── alerts.yml          # Alert rules
├── grafana/
│   └── dashboards/         # Dashboard JSONs
├── src/
│   ├── instrumentation.py  # Prometheus client
│   └── custom_metrics.py   # ML metrics
├── tests/
│   └── test_metrics.py     # Metric tests
├── README.md              # This file
└── SOLUTION_GUIDE.md      # Detailed guide
```

## Instrumentation

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
requests = Counter('http_requests_total', 'Total requests', ['method', 'endpoint'])
latency = Histogram('http_request_duration_seconds', 'Request duration')
accuracy = Gauge('model_accuracy', 'Model accuracy', ['model'])

# Use in application
@app.route('/predict')
def predict():
    with latency.time():
        result = model.predict(data)
        requests.labels(method='POST', endpoint='/predict').inc()
        accuracy.labels(model='resnet50').set(0.95)
    return result

# Expose metrics
@app.route('/metrics')
def metrics():
    return generate_latest()
```

## Alert Configuration

```yaml
# High error rate alert
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High error rate detected"
    description: "Error rate is {{ $value | humanizePercentage }}"
```

## Testing

```bash
# Test metrics collection
curl http://localhost:5000/metrics

# Trigger test alerts
./scripts/trigger-test-alerts.sh

# Verify alert routing
curl http://localhost:9093/api/v1/alerts
```

## Key Metrics

### The Four Golden Signals
1. **Latency**: Request duration
2. **Traffic**: Requests per second
3. **Errors**: Error rate
4. **Saturation**: Resource utilization

### ML-Specific Metrics
- Inference latency
- Predictions per second
- Model accuracy
- Data drift score
- Feature distribution

## Notification Channels

Configure in Alertmanager:
- **Email**: ops-team@example.com
- **Slack**: #ml-alerts channel
- **PagerDuty**: Critical alerts only
- **Webhook**: Custom integrations

## Documentation

See [SOLUTION_GUIDE.md](SOLUTION_GUIDE.md) for:
- Full architecture
- Alert design principles
- Dashboard best practices
- Troubleshooting guide
- Advanced configuration

## Requirements

- Prometheus 2.47+
- Grafana 10.2+
- Elasticsearch 8.11+
- Kubernetes (for K8s deployment)
- Docker Compose (for local deployment)

## License

Educational use only - AI Infrastructure Curriculum
