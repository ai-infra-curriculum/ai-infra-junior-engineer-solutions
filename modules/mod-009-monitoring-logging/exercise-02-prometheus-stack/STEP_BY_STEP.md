# Exercise 02: Step-by-Step Implementation Guide

## Overview

This guide walks you through implementing a production-ready Prometheus monitoring stack from scratch. Follow these steps to understand how each component works and how they integrate together.

## Phase 1: Core Prometheus Setup (30 minutes)

### Step 1: Create Project Structure

```bash
mkdir -p prometheus-stack/{config/{prometheus,alertmanager,exporters},scripts,docs,data}
cd prometheus-stack
```

### Step 2: Create Basic Prometheus Configuration

File: `config/prometheus/prometheus.yml`

Start with minimal configuration:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

**Why this matters**: This establishes the foundation. `scrape_interval` determines how often Prometheus collects metrics. 15s is a good balance between freshness and storage.

### Step 3: Create Docker Compose for Prometheus

File: `docker-compose.yml`

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.48.0
    container_name: prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

volumes:
  prometheus_data:
```

### Step 4: Start and Validate

```bash
# Create data directory
mkdir -p data/prometheus
chmod 777 data/prometheus

# Start Prometheus
docker-compose up -d prometheus

# Check health
curl http://localhost:9090/-/healthy

# View web UI
open http://localhost:9090
```

**Validation Checkpoint**:
- Prometheus UI accessible at http://localhost:9090
- Status > Targets shows "prometheus" endpoint as UP
- Can execute query: `up` returns result

## Phase 2: Add Infrastructure Monitoring (20 minutes)

### Step 5: Add Node Exporter

Add to `docker-compose.yml`:

```yaml
  node-exporter:
    image: prom/node-exporter:v1.7.0
    container_name: node-exporter
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--path.rootfs=/host/root'
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/host/root:ro,rslave
    ports:
      - "9100:9100"
    pid: host
```

Add scrape config in `prometheus.yml`:

```yaml
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Step 6: Add cAdvisor for Container Metrics

Add to `docker-compose.yml`:

```yaml
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.47.2
    container_name: cadvisor
    privileged: true
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    ports:
      - "8080:8080"
```

Add scrape config:

```yaml
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
```

### Step 7: Restart and Validate

```bash
docker-compose up -d

# Check new targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Query node metrics
curl -s 'http://localhost:9090/api/v1/query?query=node_cpu_seconds_total' | jq .

# Query container metrics
curl -s 'http://localhost:9090/api/v1/query?query=container_memory_usage_bytes' | jq .
```

**Validation Checkpoint**:
- All 3 targets (prometheus, node-exporter, cadvisor) showing as UP
- Can query `node_cpu_seconds_total`
- Can query `container_memory_usage_bytes`

## Phase 3: Recording Rules for SLO Tracking (30 minutes)

### Step 8: Create Recording Rules

File: `config/prometheus/recording_rules.yml`

```yaml
groups:
  - name: slo_availability
    interval: 30s
    rules:
      - record: slo:http_requests:total:rate5m
        expr: |
          sum(rate(http_requests_total{service="inference-gateway"}[5m]))

      - record: slo:http_requests:success:rate5m
        expr: |
          sum(rate(http_requests_total{service="inference-gateway", status!~"5.."}[5m]))

      - record: slo:availability:ratio_rate5m
        expr: |
          (
            sum(rate(http_requests_total{service="inference-gateway", status!~"5.."}[5m]))
            /
            sum(rate(http_requests_total{service="inference-gateway"}[5m]))
          ) * 100
```

**Why recording rules?**
- Pre-compute expensive queries
- Make dashboards faster
- Consistent calculations across all queries
- Reduce load on Prometheus

### Step 9: Configure Prometheus to Use Rules

Update `prometheus.yml`:

```yaml
rule_files:
  - '/etc/prometheus/recording_rules.yml'
```

Update `docker-compose.yml` to mount the rules file:

```yaml
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./config/prometheus/recording_rules.yml:/etc/prometheus/recording_rules.yml:ro
```

### Step 10: Validate Rules

```bash
# Validate rules syntax
docker run --rm \
  -v $(pwd)/config/prometheus:/etc/prometheus \
  prom/prometheus:v2.48.0 \
  promtool check rules /etc/prometheus/recording_rules.yml

# Restart Prometheus
docker-compose restart prometheus

# Check rules loaded
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[].name'
```

**Validation Checkpoint**:
- Rules validation passes
- Rules appear in Status > Rules in Prometheus UI
- Can query `slo:availability:ratio_rate5m` (will return data once inference-gateway is running)

## Phase 4: Alerting Rules (30 minutes)

### Step 11: Create Alerting Rules

File: `config/prometheus/alerting_rules.yml`

```yaml
groups:
  - name: slo_critical_alerts
    interval: 30s
    rules:
      # Fast burn rate alert
      - alert: SLOAvailabilityFastBurn
        expr: |
          slo:availability:burn_rate:1h > 14.4
          and
          slo:availability:burn_rate:6h > 14.4
        for: 2m
        labels:
          severity: critical
          page: "true"
        annotations:
          summary: "SLO fast burn detected"
          description: "Error budget consuming at 14.4x rate"

      # Service down alert
      - alert: ServiceDown
        expr: up{service=~"inference-gateway|prometheus"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.service }} is down"
```

### Step 12: Add Alertmanager

Add to `docker-compose.yml`:

```yaml
  alertmanager:
    image: prom/alertmanager:v0.26.0
    container_name: alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
    volumes:
      - ./config/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
    ports:
      - "9093:9093"
```

Create `config/alertmanager/alertmanager.yml`:

```yaml
global:
  resolve_timeout: 5m

route:
  receiver: 'default-notifications'
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 4h

receivers:
  - name: 'default-notifications'
    # Configure your notification channels here
```

### Step 13: Configure Prometheus to Use Alertmanager

Update `prometheus.yml`:

```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - '/etc/prometheus/recording_rules.yml'
  - '/etc/prometheus/alerting_rules.yml'
```

Update volumes in `docker-compose.yml`:

```yaml
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./config/prometheus/recording_rules.yml:/etc/prometheus/recording_rules.yml:ro
      - ./config/prometheus/alerting_rules.yml:/etc/prometheus/alerting_rules.yml:ro
```

### Step 14: Validate Alerting

```bash
# Validate alert rules
docker run --rm \
  -v $(pwd)/config/prometheus:/etc/prometheus \
  prom/prometheus:v2.48.0 \
  promtool check rules /etc/prometheus/alerting_rules.yml

# Validate Alertmanager config
docker run --rm \
  -v $(pwd)/config/alertmanager:/etc/alertmanager \
  prom/alertmanager:v0.26.0 \
  amtool check-config /etc/alertmanager/alertmanager.yml

# Restart stack
docker-compose up -d

# Check Alertmanager
curl http://localhost:9093/-/healthy

# View configured alerts
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[] | select(.type=="alerting") | .name'
```

**Validation Checkpoint**:
- Alertmanager accessible at http://localhost:9093
- Alert rules appear in Prometheus UI (Status > Rules)
- Alertmanager shows in Status > Runtime & Build Information

## Phase 5: Custom Exporters (30 minutes)

### Step 15: Build ML Model Exporter

Create `exporters/ml-model-exporter/exporter.py`:

```python
#!/usr/bin/env python3
import os
import time
from prometheus_client import start_http_server, Gauge
from prometheus_client.core import GaugeMetricFamily, REGISTRY

class MLModelCollector:
    def collect(self):
        # Model info metric
        model_info = GaugeMetricFamily(
            'ml_model_loaded',
            'Whether model is loaded',
            labels=['model_name']
        )
        model_info.add_metric(['resnet50'], 1)
        yield model_info

# Register and start
REGISTRY.register(MLModelCollector())
start_http_server(9101)

while True:
    time.sleep(1)
```

Create `exporters/ml-model-exporter/requirements.txt`:

```
prometheus-client==0.19.0
requests==2.31.0
```

Create `exporters/ml-model-exporter/Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY exporter.py .
EXPOSE 9101
CMD ["python", "-u", "exporter.py"]
```

### Step 16: Add Exporter to Stack

Add to `docker-compose.yml`:

```yaml
  ml-model-exporter:
    build:
      context: ./exporters/ml-model-exporter
    container_name: ml-model-exporter
    ports:
      - "9101:9101"
```

Add scrape config:

```yaml
  - job_name: 'ml-model-exporter'
    static_configs:
      - targets: ['ml-model-exporter:9101']
```

### Step 17: Build and Start

```bash
# Build exporter
docker build -t ml-model-exporter:latest ./exporters/ml-model-exporter

# Start full stack
docker-compose up -d

# Check exporter metrics
curl http://localhost:9101/metrics

# Verify Prometheus scrapes it
curl -s http://localhost:9090/api/v1/query?query=ml_model_loaded | jq .
```

**Validation Checkpoint**:
- ML exporter accessible at http://localhost:9101
- Can see `ml_model_loaded` metric
- Exporter shows as UP in Prometheus targets

## Phase 6: Integration Testing (20 minutes)

### Step 18: Create Test Script

Create `scripts/test-alerts.sh`:

```bash
#!/bin/bash
# Generate load to trigger alerts

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8000}"

echo "Generating high load..."
for i in {1..1000}; do
    curl -s ${GATEWAY_URL}/health > /dev/null || true
done

echo "Simulating errors..."
for i in {1..100}; do
    curl -s ${GATEWAY_URL}/nonexistent > /dev/null 2>&1 || true
done

echo "Check alerts in 2 minutes at http://localhost:9090/alerts"
```

```bash
chmod +x scripts/test-alerts.sh
```

### Step 19: End-to-End Test

```bash
# 1. Check all services
docker-compose ps

# 2. View all targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# 3. Test a query
curl -s 'http://localhost:9090/api/v1/query?query=up' | jq '.data.result[] | {job: .metric.job, value: .value[1]}'

# 4. Check recording rules work
curl -s 'http://localhost:9090/api/v1/query?query={__name__=~"slo:.*"}' | jq '.data.result[] | .metric.__name__'

# 5. Verify Alertmanager integration
curl -s http://localhost:9090/api/v1/alertmanagers | jq .
```

## Phase 7: Production Hardening (30 minutes)

### Step 20: Add Health Checks

Update `docker-compose.yml` with health checks:

```yaml
  prometheus:
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Step 21: Add Persistence

Ensure data persistence by using bind mounts:

```yaml
volumes:
  prometheus_data:
    driver: local
    driver_opts:
      type: none
      device: ${PWD}/data/prometheus
      o: bind
```

### Step 22: Configure Retention

```yaml
    command:
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.retention.size=10GB'
```

### Step 23: Security Hardening

Run as non-root:

```yaml
  prometheus:
    user: "65534"  # nobody user
```

### Step 24: Create Setup Automation

Create `scripts/setup.sh` that:
1. Creates data directories
2. Sets permissions
3. Validates all configurations
4. Builds custom exporters
5. Displays next steps

## Common Issues and Solutions

### Issue 1: Permission Denied on Data Directory

**Symptom**: Prometheus fails to start with "permission denied" on `/prometheus`

**Solution**:
```bash
chmod 777 data/prometheus
# Or run as root temporarily
docker-compose exec prometheus chown -R 65534:65534 /prometheus
```

### Issue 2: Targets Not Appearing

**Symptom**: Services don't show in Prometheus targets

**Solution**:
- Check Docker network connectivity
- Verify service names match `static_configs.targets`
- Check container logs: `docker-compose logs prometheus`

### Issue 3: Recording Rules Not Working

**Symptom**: Queries return no data

**Solution**:
- Check rule syntax: `promtool check rules`
- Verify rules are loaded: `curl http://localhost:9090/api/v1/rules`
- Check evaluation: Status > Rules in UI

### Issue 4: Alerts Not Firing

**Symptom**: Conditions met but no alerts

**Solution**:
- Check `for` duration hasn't elapsed yet
- Verify Alertmanager is connected: Status > Runtime & Build Info
- Check alert rules syntax: `promtool check rules`
- View pending alerts: `curl http://localhost:9090/api/v1/alerts`

## Performance Optimization

### 1. Cardinality Control

Use metric relabeling to drop high-cardinality labels:

```yaml
metric_relabel_configs:
  - source_labels: [endpoint]
    regex: '/predict/.*'
    replacement: '/predict/{id}'
    target_label: endpoint
```

### 2. Recording Rules for Expensive Queries

Pre-compute dashboard queries:

```yaml
- record: container:cpu_usage:percent
  expr: |
    sum(rate(container_cpu_usage_seconds_total[5m])) by (container) * 100
```

### 3. Adjust Scrape Intervals

Critical services: 10s
Standard services: 15s
Slow-changing metrics: 30-60s

## Next Steps

1. **Exercise 03**: Build Grafana dashboards using these metrics
2. **Exercise 04**: Add centralized logging with Loki
3. **Exercise 05**: Configure complete alerting workflows

## Key Takeaways

✅ **Monitoring Stack Architecture** - Understand how components integrate
✅ **SLO-Based Observability** - Metrics that matter for reliability
✅ **Recording Rules** - Pre-compute for performance
✅ **Multi-Window Alerts** - Reduce false positives
✅ **Custom Exporters** - Domain-specific metrics collection
✅ **Production Practices** - Validation, testing, hardening

---

**Congratulations! You've built a production-ready Prometheus monitoring stack!** 🎉
