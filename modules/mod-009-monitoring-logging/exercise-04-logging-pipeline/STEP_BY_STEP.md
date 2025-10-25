# Step-by-Step Implementation Guide: Centralized Logging Pipeline

## Overview

Build a production-ready centralized logging pipeline using **Grafana Loki** and **Promtail** for log aggregation, structured log parsing, and correlation with metrics and traces.

**Time**: 2-3 hours | **Difficulty**: Intermediate

---

## Phase 1: Loki Setup (30 minutes)

### Step 1: Configure Loki

**`config/loki/loki-config.yaml`**:
```yaml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
  chunk_idle_period: 5m
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2023-01-01
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/index
    cache_location: /loki/cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h  # 7 days
  ingestion_rate_mb: 10
  ingestion_burst_size_mb: 20

chunk_store_config:
  max_look_back_period: 720h  # 30 days

table_manager:
  retention_deletes_enabled: true
  retention_period: 720h  # 30 days
```

### Step 2: Start Loki

**`docker-compose.yml`** (add to existing):
```yaml
services:
  loki:
    image: grafana/loki:2.9.3
    container_name: loki
    ports:
      - "3100:3100"
    volumes:
      - ./config/loki:/etc/loki
      - loki_data:/loki
    command: -config.file=/etc/loki/loki-config.yaml
    networks:
      - monitoring

volumes:
  loki_data:
```

```bash
docker-compose up -d loki

# Test Loki
curl http://localhost:3100/ready
# Output: ready
```

---

## Phase 2: Promtail Configuration (40 minutes)

### Step 3: Configure Promtail for Docker Logs

**`config/promtail/promtail-config.yaml`**:
```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # Docker container logs
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      # Container name as label
      - source_labels: ['__meta_docker_container_name']
        regex: '/(.*)'
        target_label: 'container'
      # Container image
      - source_labels: ['__meta_docker_container_image']
        target_label: 'image'
      # Docker labels
      - source_labels: ['__meta_docker_container_label_com_docker_compose_service']
        target_label: 'service'

    # Multi-stage pipeline
    pipeline_stages:
      # Parse JSON logs
      - json:
          expressions:
            level: level
            timestamp: timestamp
            message: message
            request_id: request_id
            trace_id: trace_id
            duration_ms: duration_ms

      # Extract labels from JSON
      - labels:
          level:
          request_id:

      # Redact PII
      - replace:
          expression: '([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
          replace: '[EMAIL_REDACTED]'
      - replace:
          expression: '\b\d{13,19}\b'  # Credit card numbers
          replace: '[CC_REDACTED]'

      # Extract metrics from logs
      - metrics:
          log_errors_total:
            type: Counter
            description: "Total log errors"
            source: level
            config:
              value: error
              action: inc
```

### Step 4: Start Promtail

```yaml
# Add to docker-compose.yml
promtail:
  image: grafana/promtail:2.9.3
  container_name: promtail
  volumes:
    - ./config/promtail:/etc/promtail
    - /var/run/docker.sock:/var/run/docker.sock
    - /var/log:/var/log
  command: -config.file=/etc/promtail/promtail-config.yaml
  networks:
    - monitoring
```

```bash
docker-compose up -d promtail

# Check logs are flowing
curl 'http://localhost:3100/loki/api/v1/query?query={container="inference-gateway"}'
```

---

## Phase 3: LogQL Queries (30 minutes)

### Step 5: Basic LogQL Queries

**In Grafana Explore (Data Source: Loki)**:

```logql
# All logs from inference-gateway
{container="inference-gateway"}

# Errors only
{container="inference-gateway"} |= "ERROR"

# JSON log filtering
{container="inference-gateway"} | json | level="ERROR"

# Filter by request_id
{container="inference-gateway"} | json | request_id="550e8400-e29b-41d4-a716-446655440000"

# Trace correlation
{container="inference-gateway"} | json | trace_id="a1b2c3d4e5f6g7h8"

# Rate of errors
rate({container="inference-gateway"} | json | level="ERROR" [5m])

# Pattern extraction
{container="inference-gateway"} | pattern `<timestamp> <level> <_> request_id=<request_id> <_>`

# Log-based metric
sum(rate({container="inference-gateway"} | json | level="ERROR" [5m])) by (container)
```

---

## Phase 4: Log-Based Alerting (30 minutes)

### Step 6: Create Log-Based Alerts

**`config/loki/rules.yml`**:
```yaml
groups:
  - name: log_alerts
    interval: 1m
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate({container="inference-gateway"} | json | level="ERROR" [5m])) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in logs"
          description: "Error rate: {{ $value }} errors/sec"

      - alert: PredictionFailures
        expr: |
          sum(rate({container="inference-gateway"} |= "prediction_failed" [5m])) > 5
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High prediction failure rate"
```

**Enable Loki Ruler** (add to loki-config.yaml):
```yaml
ruler:
  alertmanager_url: http://alertmanager:9093
  enable_api: true
  enable_alertmanager_v2: true
  rule_path: /loki/rules
  storage:
    type: local
    local:
      directory: /loki/rules
```

---

## Phase 5: Grafana Log Dashboard (30 minutes)

### Step 7: Create Log Panel in Grafana

**Panel Configuration**:
```json
{
  "type": "logs",
  "title": "Application Logs",
  "targets": [
    {
      "expr": "{container=\"inference-gateway\"} | json",
      "refId": "A"
    }
  ],
  "options": {
    "showTime": true,
    "showLabels": true,
    "showCommonLabels": false,
    "wrapLogMessage": true,
    "prettifyLogMessage": true,
    "enableLogDetails": true,
    "dedupStrategy": "none",
    "sortOrder": "Descending"
  }
}
```

**Add to SLO Dashboard**:
- Create new row: "Logs"
- Add Logs panel with query: `{container="inference-gateway"} | json | level=~"ERROR|WARN"`
- Add Table panel with top error messages

---

## Phase 6: Log Retention and Compaction (20 minutes)

### Step 8: Configure Retention

**Already configured in loki-config.yaml**:
```yaml
table_manager:
  retention_deletes_enabled: true
  retention_period: 720h  # 30 days

limits_config:
  retention_period: 720h
```

**Manual Deletion**:
```bash
# Delete logs older than 30 days
curl -X POST 'http://localhost:3100/loki/api/v1/delete?query={container="old-service"}&start=0&end=1640000000'
```

---

## Validation

### Test Log Pipeline

```bash
# 1. Generate logs
docker exec inference-gateway python -c "
import logging
logger = logging.getLogger()
logger.error('Test error message', extra={'request_id': '123', 'trace_id': 'abc'})
"

# 2. Query in Loki (wait 10-30 seconds)
curl 'http://localhost:3100/loki/api/v1/query?query={container="inference-gateway"}%20|%20json%20|%20request_id="123"'

# 3. Check in Grafana Explore
# Data Source: Loki
# Query: {container="inference-gateway"} | json | request_id="123"
```

---

## Troubleshooting

### No Logs Appearing

1. Check Promtail is running: `docker logs promtail`
2. Check Loki is receiving: `curl http://localhost:3100/metrics | grep loki_ingester_streams_created_total`
3. Verify Docker socket mount: `docker exec promtail ls /var/run/docker.sock`

### High Memory Usage

1. Reduce retention: `retention_period: 168h` (7 days)
2. Limit ingestion rate: `ingestion_rate_mb: 5`
3. Increase chunk idle period: `chunk_idle_period: 15m`

---

## Summary

**What You Built**:
- ✅ Loki centralized log aggregation
- ✅ Promtail for Docker log collection
- ✅ JSON log parsing pipeline
- ✅ PII redaction
- ✅ Log-based metrics and alerts
- ✅ Grafana log visualization
- ✅ 30-day retention with auto-cleanup

**Key Queries**:
- `{container="app"}` - All logs from container
- `{container="app"} | json | level="ERROR"` - Parse JSON, filter errors
- `rate({container="app"} |= "error" [5m])` - Error rate
- `{container="app"} | json | trace_id="xyz"` - Trace correlation
