# Exercise 04: Centralized Logging Pipeline - Complete Solution

## Overview

This is a **production-ready centralized logging pipeline** using Grafana Loki and Promtail, providing comprehensive log aggregation, parsing, querying, and correlation with metrics and traces. The solution completes the unified observability stack from Exercises 01-03.

## 🎯 Solution Highlights

### Complete Logging Stack
- ✅ **Loki 2.9.3** - Scalable log aggregation system
- ✅ **Promtail 2.9.3** - Log collection and shipping agent
- ✅ **Docker Log Discovery** - Automatic container log collection
- ✅ **JSON Log Parsing** - Structured log extraction and enrichment
- ✅ **Log-Based Metrics** - Derive metrics from log patterns
- ✅ **30-Day Retention** - Configurable storage policies

### Advanced Features
- ✅ **Trace-Log Correlation** - Link logs to traces via trace_id
- ✅ **PII Redaction** - Automatic email and credit card masking
- ✅ **Multi-Stage Pipeline** - JSON parsing, label extraction, metrics derivation
- ✅ **Label-Based Indexing** - Efficient log querying with LogQL
- ✅ **Log-Based Alerting** - Ruler integration with Alertmanager
- ✅ **Compaction & Retention** - Automated cleanup and storage optimization

## 📁 Project Structure

```
exercise-04-logging-pipeline/
├── config/
│   ├── loki/
│   │   └── loki-config.yaml            # Loki server configuration
│   ├── promtail/
│   │   └── promtail-config.yaml         # Log collection & parsing
│   └── grafana/
│       └── provisioning/                # Grafana data sources
├── scripts/
│   ├── setup.sh                         # Environment setup
│   └── test-logging.sh                  # Log collection verification
├── docs/
│   └── logql-queries.md                 # LogQL query reference
├── docker-compose.yml                   # Full observability stack
├── .env.example                         # Environment configuration
└── README.md                            # This file
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Exercises 01-03 completed (or inference gateway available)
- 10GB free disk space for log storage

### 1. Setup

```bash
# Run setup script
./scripts/setup.sh

# Edit .env file if needed
cp .env.example .env
```

### 2. Start the Stack

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f loki promtail
```

### 3. Verify Log Collection

```bash
# Run test script
./scripts/test-logging.sh

# Check Loki is ready
curl http://localhost:3100/ready

# Check Promtail targets
curl http://localhost:9080/targets | jq

# Query logs
curl 'http://localhost:3100/loki/api/v1/query?query={container="loki"}&limit=10' | jq
```

### 4. Explore Logs in Grafana

```bash
# Open Grafana
open http://localhost:3000/explore

# Login: admin / admin
# Select data source: Loki
# Try query: {container="inference-gateway"}
```

## 📊 Architecture

### Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    Log Producers                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Inference   │  │  Prometheus  │  │   Grafana    │        │
│  │  Gateway     │  │              │  │              │        │
│  │ (JSON logs)  │  │  (text logs) │  │  (text logs) │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
│         │                 │                 │                  │
│         └─────────────────┴─────────────────┘                 │
│                           │                                    │
└───────────────────────────┼────────────────────────────────────┘
                            │
                   ┌────────▼────────┐
                   │    Promtail     │
                   │                 │
                   │ - Discovery     │
                   │ - Parsing       │
                   │ - Labels        │
                   │ - PII Redaction │
                   └────────┬────────┘
                            │ Push (HTTP)
                   ┌────────▼────────┐
                   │      Loki       │
                   │                 │
                   │ - Ingestion     │
                   │ - Indexing      │
                   │ - Compression   │
                   │ - Retention     │
                   └────────┬────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
    ┌─────▼─────┐    ┌──────▼──────┐   ┌─────▼─────┐
    │  Grafana  │    │  Prometheus │   │   Jaeger  │
    │  (Query)  │    │  (Metrics)  │   │  (Traces) │
    └───────────┘    └─────────────┘   └───────────┘
```

### Component Responsibilities

| Component | Purpose | Port |
|-----------|---------|------|
| Loki | Log storage and indexing | 3100 |
| Promtail | Log collection and shipping | 9080 |
| Grafana | Log visualization and querying | 3000 |

## 🔧 Configuration Details

### Loki Configuration Highlights

**Storage** (`loki-config.yaml`):
```yaml
storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
  filesystem:
    directory: /loki/chunks
```

**Retention**:
```yaml
limits_config:
  retention_period: 30d

compactor:
  retention_enabled: true
  compaction_interval: 10m
```

**Ingestion Limits**:
```yaml
limits_config:
  ingestion_rate_mb: 10
  ingestion_burst_size_mb: 20
  max_line_size: 256kb
  max_streams_per_user: 10000
```

### Promtail Configuration Highlights

**Docker Discovery** (`promtail-config.yaml`):
```yaml
docker_sd_configs:
  - host: unix:///var/run/docker.sock
    filters:
      - name: label
        values: ["logging=promtail"]
```

**JSON Log Parsing**:
```yaml
pipeline_stages:
  - docker: {}
  - json:
      expressions:
        timestamp: timestamp
        level: level
        trace_id: trace_id
        message: message
        duration_ms: duration_ms
  - labels:
      level: level
      trace_id: trace_id
```

**PII Redaction**:
```yaml
- replace:
    expression: '(?P<email>[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
    replace: '***REDACTED_EMAIL***'
```

## 📈 LogQL Query Examples

### Basic Queries

```logql
# All logs from inference-gateway
{container="inference-gateway"}

# Error logs only
{container="inference-gateway"} |= "ERROR"

# Logs with JSON parsing
{container="inference-gateway"} | json | status_code >= 500
```

### Metric Queries

```logql
# Request rate
rate({container="inference-gateway"}[5m])

# Error rate
sum(rate({container="inference-gateway"} |= "ERROR" [5m]))

# P99 latency from logs
quantile_over_time(0.99,
  {container="inference-gateway"}
  | json
  | unwrap duration_ms [5m]
)
```

### Trace Correlation

```logql
# Find all logs for a trace
{container="inference-gateway"}
  | json
  | trace_id="550e8400-e29b-41d4-a716-446655440000"
```

**See `docs/logql-queries.md` for comprehensive query examples!**

## 🔗 Integration with Other Exercises

### With Exercise 01 (Observability Foundations)
- Collects JSON logs from instrumented inference gateway
- Extracts trace_id, span_id, request_id from structured logs
- Correlates logs with OpenTelemetry traces

### With Exercise 02 (Prometheus Stack)
- Derives log-based metrics (request rate, error count)
- Complements Prometheus metrics with log details
- Provides context for metric spikes

### With Exercise 03 (Grafana Dashboards)
- Loki pre-configured as Grafana data source
- Explore view for interactive log querying
- Dashboard panels with log streams
- Drill-down from metrics to logs

### Unified Observability Flow

```
Incident Alert (Prometheus)
    ↓
View Dashboard (Grafana) - See metric spike
    ↓
Click on time range
    ↓
Explore Logs (Loki) - Filter by time + service
    ↓
Find trace_id in logs
    ↓
View Trace (Jaeger) - See full request flow
    ↓
Root Cause Identified!
```

## 🛠️ Operations

### Viewing Logs

```bash
# Via Grafana Explore (Recommended)
open http://localhost:3000/explore

# Via API
curl 'http://localhost:3100/loki/api/v1/query?query={container="inference-gateway"}&limit=100' | jq

# Via CLI (logcli)
docker run -it --rm --network host grafana/logcli:latest \
  --addr="http://localhost:3100" \
  query '{container="inference-gateway"}' --limit=10
```

### Monitoring Loki

```bash
# Check Loki metrics
curl http://localhost:3100/metrics | grep loki_ingester

# Check ingestion rate
curl http://localhost:3100/metrics | grep loki_ingester_chunks_created_total

# Check storage usage
du -sh data/loki/
```

### Monitoring Promtail

```bash
# Check Promtail metrics
curl http://localhost:9080/metrics | grep promtail

# View active targets
curl http://localhost:9080/targets | jq '.activeTargets[] | {job: .labels.job, health: .health}'

# Check positions file (what's been read)
cat data/promtail-positions/positions.yaml
```

### Debugging Log Collection

```bash
# 1. Check Promtail is discovering containers
docker-compose logs promtail | grep "found new target"

# 2. Verify Promtail is reading logs
docker-compose logs promtail | grep "Successfully sent batch"

# 3. Check Loki is receiving logs
curl http://localhost:3100/metrics | grep loki_distributor_lines_received_total

# 4. Query Loki for recent logs
curl 'http://localhost:3100/loki/api/v1/query?query={job="docker"}&limit=5' | jq
```

### Performance Tuning

**Increase Ingestion Rate**:
```yaml
# In loki-config.yaml
limits_config:
  ingestion_rate_mb: 50        # Increase from 10
  ingestion_burst_size_mb: 100 # Increase from 20
```

**Adjust Batch Size**:
```yaml
# In promtail-config.yaml
clients:
  - url: http://loki:3100/loki/api/v1/push
    batchsize: 2097152  # 2MB (up from 1MB)
    batchwait: 500ms    # Send more frequently
```

**Reduce Retention**:
```yaml
# In loki-config.yaml
limits_config:
  retention_period: 7d  # Reduce from 30d for less storage
```

## 📊 Log-Based Metrics

Promtail automatically generates metrics from logs:

### Counter Metrics

```promql
# Total log lines by level
promtail_log_lines_total{level="ERROR"}

# Rate of log lines
rate(promtail_log_lines_total[5m])
```

### Histogram Metrics

```promql
# HTTP request duration from logs
histogram_quantile(0.99,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
)
```

## 🔒 Security & Compliance

### PII Redaction

Promtail automatically redacts sensitive information:

- **Email addresses**: `user@example.com` → `***REDACTED_EMAIL***`
- **Credit cards**: `4111-1111-1111-1111` → `***REDACTED_CC***`

**Customize in `promtail-config.yaml`**:
```yaml
pipeline_stages:
  - replace:
      expression: 'your-custom-pattern'
      replace: '***REDACTED***'
```

### Access Control

For production, enable Loki multi-tenancy:

```yaml
# In loki-config.yaml
auth_enabled: true
```

Then send `X-Scope-OrgID` header with requests:
```bash
curl -H "X-Scope-OrgID: team-a" http://localhost:3100/loki/api/v1/query...
```

### Retention Policies

Logs are automatically deleted after retention period:

```yaml
limits_config:
  retention_period: 30d  # Customize per compliance requirements
```

## 🚨 Log-Based Alerting

Create alert rules in Loki (`config/loki/rules/`):

```yaml
groups:
  - name: log_alerts
    interval: 1m
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate({container="inference-gateway"} |= "ERROR" [5m]))
          > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in logs"
```

## 📚 Learning Outcomes

This solution demonstrates:

✅ **Loki Deployment** - Production configuration with retention and compaction
✅ **Log Collection** - Promtail with Docker service discovery
✅ **Structured Logging** - JSON parsing and field extraction
✅ **Label Extraction** - Efficient indexing for fast queries
✅ **LogQL Queries** - From basic filtering to complex aggregations
✅ **Log-Based Metrics** - Derive counters and histograms from logs
✅ **Trace Correlation** - Link logs to distributed traces
✅ **PII Redaction** - Compliance and privacy protection
✅ **Retention Management** - Automated cleanup policies
✅ **Unified Observability** - Metrics + Logs + Traces integration

## 🎓 Advanced Topics

### Multi-Tenancy

Enable tenant isolation:
```yaml
auth_enabled: true
```

### S3 Storage Backend

For production scale:
```yaml
storage_config:
  aws:
    s3: s3://region/bucket-name
    dynamodb:
      dynamodb_url: dynamodb://region
```

### Horizontal Scaling

Run multiple Loki components:
- Read path: queriers, query-frontend
- Write path: distributors, ingesters
- Background: compactors

### LogQL Advanced Patterns

**Pattern matching**:
```logql
{container="inference-gateway"}
  | pattern `<timestamp> <level> <message>`
  | level = "ERROR"
```

**Line format**:
```logql
{container="inference-gateway"}
  | json
  | line_format "{{.timestamp}} [{{.level}}] {{.endpoint}} {{.message}}"
```

## 🎉 Success Criteria

- ✅ Loki accessible at http://localhost:3100
- ✅ Promtail collecting logs from all containers
- ✅ Logs queryable in Grafana Explore
- ✅ JSON logs parsed correctly
- ✅ Trace IDs extracted and linkable
- ✅ PII automatically redacted
- ✅ 30-day retention configured
- ✅ Log-based metrics generated

---

**🎓 Ready for Exercise 05: Alerting & Incident Response!**
