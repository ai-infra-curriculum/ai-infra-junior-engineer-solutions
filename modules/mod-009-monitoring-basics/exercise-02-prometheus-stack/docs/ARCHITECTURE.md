# Architecture: Prometheus Monitoring Stack

## Overview

This document details the architecture of the production-ready Prometheus monitoring stack, including all components, data flows, design decisions, and scalability considerations.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Component Details](#component-details)
3. [Data Flow](#data-flow)
4. [SLO Tracking Architecture](#slo-tracking-architecture)
5. [Alerting Architecture](#alerting-architecture)
6. [Design Decisions](#design-decisions)
7. [Scalability](#scalability)
8. [High Availability](#high-availability)

---

## System Architecture

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      Monitored Targets                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Inference   │  │    Node      │  │   cAdvisor   │         │
│  │   Gateway    │  │  Exporter    │  │  (Container  │         │
│  │              │  │  (Host       │  │   Metrics)   │         │
│  │  /metrics    │  │  Metrics)    │  │              │         │
│  │  :8000       │  │  :9100       │  │  :8080       │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                 │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐         │
│  │  Blackbox    │  │   ML Model   │  │  Pushgateway │         │
│  │   Exporter   │  │   Exporter   │  │  (Batch Jobs)│         │
│  │  (Probing)   │  │  (Custom)    │  │              │         │
│  │  :9115       │  │  :9101       │  │  :9091       │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          │    HTTP Pull Scrape (every 15s)    │
          └──────────────────┼──────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │  Prometheus Server  │
                  │                     │
                  │ ┌─────────────────┐ │
                  │ │  TSDB Storage   │ │
                  │ │  30-day         │ │
                  │ │  retention      │ │
                  │ └─────────────────┘ │
                  │                     │
                  │ ┌─────────────────┐ │
                  │ │ Recording Rules │ │
                  │ │ (SLO metrics)   │ │
                  │ │ Eval every 30s  │ │
                  │ └─────────────────┘ │
                  │                     │
                  │ ┌─────────────────┐ │
                  │ │ Alerting Rules  │ │
                  │ │ MWMBR SLO alerts│ │
                  │ │ Eval every 15s  │ │
                  │ └─────────────────┘ │
                  │                     │
                  │ ┌─────────────────┐ │
                  │ │  PromQL Engine  │ │
                  │ │  Query API      │ │
                  │ │  :9090          │ │
                  │ └─────────────────┘ │
                  └──────────┬──────────┘
                             │ Alerts
                             │ (when conditions met)
                  ┌──────────▼──────────┐
                  │   Alertmanager      │
                  │                     │
                  │ ┌─────────────────┐ │
                  │ │ Alert Routing   │ │
                  │ │ - Critical →    │ │
                  │ │   PagerDuty     │ │
                  │ │ - Warning →     │ │
                  │ │   Slack         │ │
                  │ └─────────────────┘ │
                  │                     │
                  │ ┌─────────────────┐ │
                  │ │   Grouping &    │ │
                  │ │   Throttling    │ │
                  │ └─────────────────┘ │
                  │                     │
                  │ ┌─────────────────┐ │
                  │ │   Silences &    │ │
                  │ │   Inhibition    │ │
                  │ └─────────────────┘ │
                  └──────────┬──────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐      ┌─────▼─────┐     ┌─────▼─────┐
    │ PagerDuty │      │   Slack   │     │   Email   │
    │ (Critical)│      │ (Warning) │     │ (Fallback)│
    └───────────┘      └───────────┘     └───────────┘
                             │
                  ┌──────────▼──────────┐
                  │   On-Call Engineer  │
                  │   Incident Response │
                  └─────────────────────┘
```

---

## Component Details

### 1. Prometheus Server

**Purpose**: Central metrics collection, storage, and query engine

**Key Features**:
- Time-series database (TSDB) with efficient compression
- Pull-based metric collection (scraping)
- PromQL query language
- Recording and alerting rule evaluation
- Service discovery for dynamic targets

**Configuration**:
```yaml
Scrape interval: 15s
Evaluation interval: 15s (recording rules)
Retention: 30 days
Storage path: /prometheus
Config reload: HTTP POST /-/reload
```

**Resource Requirements**:
```
CPU: 2-4 cores
Memory: 8-16GB (depends on cardinality and retention)
Storage: 50GB for 30 days (with moderate cardinality)
Network: Low bandwidth (metrics are small)
```

**Metrics Exposed**:
- `prometheus_tsdb_*` - TSDB storage metrics
- `prometheus_rule_*` - Rule evaluation metrics
- `prometheus_target_*` - Scrape target health
- `prometheus_notifications_*` - Alert delivery

---

### 2. Alertmanager

**Purpose**: Alert routing, grouping, and notification delivery

**Key Features**:
- Alert aggregation and deduplication
- Routing based on labels (severity, team, service)
- Grouping to reduce alert noise
- Silences for maintenance windows
- Inhibition rules to suppress dependent alerts

**Routing Strategy**:
```
┌─────────────────┐
│ Incoming Alerts │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Match Labels   │
│  - severity     │
│  - service      │
│  - team         │
└────────┬────────┘
         │
         ├─ severity=critical ──▶ PagerDuty + Slack
         ├─ severity=warning  ──▶ Slack
         ├─ team=ml-platform  ──▶ #ml-alerts channel
         └─ team=infrastructure ─▶ #infra-alerts channel
```

**Grouping Rules**:
- Group by: `alertname`, `service`, `severity`
- Group wait: 30s (wait for more alerts before sending)
- Group interval: 5m (send new alerts in same group)
- Repeat interval: 4h (resend if not resolved)

---

### 3. Node Exporter

**Purpose**: Host-level system metrics (CPU, memory, disk, network)

**Metrics Provided**:
```
CPU:
- node_cpu_seconds_total
- node_load1, node_load5, node_load15

Memory:
- node_memory_MemTotal_bytes
- node_memory_MemAvailable_bytes
- node_memory_SwapTotal_bytes

Disk:
- node_filesystem_size_bytes
- node_filesystem_avail_bytes
- node_disk_read_bytes_total
- node_disk_written_bytes_total

Network:
- node_network_receive_bytes_total
- node_network_transmit_bytes_total
```

**Why It's Needed**:
- Foundation for infrastructure monitoring
- Capacity planning (disk space, memory trends)
- Correlation with application metrics
- Alerting on resource exhaustion

---

### 4. cAdvisor (Container Advisor)

**Purpose**: Container-level resource usage metrics

**Metrics Provided**:
```
container_cpu_usage_seconds_total{name="inference-gateway"}
container_memory_usage_bytes{name="inference-gateway"}
container_network_receive_bytes_total{name="inference-gateway"}
container_fs_usage_bytes{name="inference-gateway"}
```

**Use Cases**:
- Per-container CPU/memory tracking
- Detecting container resource leaks
- Right-sizing container resource limits
- Multi-tenant resource accounting

---

### 5. Custom ML Model Exporter

**Purpose**: Expose ML-specific metrics not available from application

**Custom Metrics**:
```python
# Model freshness
ml_model_age_seconds{model="resnet50"}

# Model file size
ml_model_size_bytes{model="resnet50"}

# Training dataset metrics
ml_training_dataset_size{model="resnet50"}
ml_training_dataset_last_updated{model="resnet50"}

# Model performance (offline)
ml_model_accuracy{model="resnet50", dataset="validation"}
ml_model_f1_score{model="resnet50"}
```

**Why Custom Exporter**:
- Metrics not available from inference service
- Batch job results (training, evaluation)
- External data sources (S3, model registry)

---

### 6. Blackbox Exporter

**Purpose**: Probe external endpoints (HTTP, TCP, ICMP, DNS)

**Probe Types**:
```yaml
HTTP Probe:
  - GET /health (expect 200)
  - GET /ready (expect 200)
  - SSL certificate expiry check

TCP Probe:
  - Port connectivity check

ICMP Probe:
  - Host reachability (ping)

DNS Probe:
  - DNS resolution check
```

**Metrics**:
```
probe_success{job="blackbox", instance="http://inference-gateway:8000/health"}
probe_duration_seconds
probe_http_status_code
probe_ssl_earliest_cert_expiry
```

---

### 7. Pushgateway

**Purpose**: Receive metrics from short-lived batch jobs

**Use Cases**:
- Training jobs (push final metrics before exit)
- ETL pipelines
- Cron jobs
- Serverless functions

**Workflow**:
```
1. Batch Job Runs
   ↓
2. Pushes Metrics to Pushgateway
   POST /metrics/job/<job_name>/instance/<instance>
   ↓
3. Prometheus Scrapes Pushgateway
   (metrics persist until next push)
```

**Example**:
```bash
# Training job pushes metrics
cat <<EOF | curl --data-binary @- http://pushgateway:9091/metrics/job/training/instance/run-123
# TYPE training_duration_seconds gauge
training_duration_seconds 3600
# TYPE training_loss gauge
training_loss 0.05
# TYPE training_accuracy gauge
training_accuracy 0.95
EOF
```

---

## Data Flow

### Metrics Collection Flow

```
┌────────────────────────────────────────────────────────┐
│ 1. SCRAPE TARGETS (Pull Model)                         │
│                                                         │
│ Prometheus Server                                      │
│   │                                                     │
│   ├─ Every 15s: HTTP GET http://target:port/metrics   │
│   │              (Prometheus text format)              │
│   │                                                     │
│   ├─ Parse Metrics                                     │
│   │   # HELP http_requests_total Total requests       │
│   │   # TYPE http_requests_total counter              │
│   │   http_requests_total{status="200"} 1234          │
│   │                                                     │
│   ├─ Add Metadata                                      │
│   │   - job: "inference-gateway"                       │
│   │   - instance: "10.0.1.5:8000"                      │
│   │   - __scrape_timestamp__: 1698765432               │
│   │                                                     │
│   └─ Store in TSDB                                     │
│       - Compress samples                               │
│       - Create index                                   │
└─────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ 2. RECORDING RULES (Pre-aggregation)                   │
│                                                         │
│ Every 30s:                                             │
│   │                                                     │
│   ├─ Evaluate Recording Rules                          │
│   │   slo:availability:ratio_rate30d =                 │
│   │     sum(rate(http_requests_total{status!~"5.."}[30d])) │
│   │     /                                               │
│   │     sum(rate(http_requests_total[30d]))           │
│   │                                                     │
│   └─ Store Computed Metrics                            │
│       (as new time series)                             │
└─────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ 3. ALERTING RULES (Alert Evaluation)                   │
│                                                         │
│ Every 15s:                                             │
│   │                                                     │
│   ├─ Evaluate Alert Rules                              │
│   │   alert: SLOAvailabilityFastBurn                   │
│   │   expr: slo:availability:burn_rate:1h > 14.4       │
│   │   for: 2m                                           │
│   │                                                     │
│   ├─ Alert Fires (if condition true for 2m)            │
│   │   - Create alert object                            │
│   │   - Add labels and annotations                     │
│   │   - Send to Alertmanager                           │
│   │                                                     │
│   └─ POST http://alertmanager:9093/api/v1/alerts      │
└─────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ 4. ALERT PROCESSING (Alertmanager)                     │
│                                                         │
│ Alertmanager receives alert:                           │
│   │                                                     │
│   ├─ Deduplicate (if already firing)                   │
│   │                                                     │
│   ├─ Group (by alertname, service, severity)           │
│   │   Wait 30s for more alerts in group                │
│   │                                                     │
│   ├─ Route (based on labels)                           │
│   │   severity=critical → PagerDuty receiver           │
│   │                                                     │
│   ├─ Inhibit (suppress dependent alerts)               │
│   │   Fast burn firing → suppress slow burn            │
│   │                                                     │
│   └─ Notify                                            │
│       - PagerDuty: POST to integration key             │
│       - Slack: POST webhook                            │
│       - Email: SMTP send                               │
└─────────────────────────────────────────────────────────┘
```

### Query Flow

```
User/Dashboard Query:
   │
   ▼
Prometheus PromQL API
   │
   ├─ Parse Query: histogram_quantile(0.99, ...)
   │
   ├─ Fetch Time Series from TSDB
   │   - Read index for matching labels
   │   - Retrieve samples from chunks
   │
   ├─ Execute Query
   │   - Apply functions (rate, sum, etc.)
   │   - Aggregate across labels
   │
   └─ Return JSON Result
       {
         "status": "success",
         "data": {
           "resultType": "vector",
           "result": [
             {"metric": {...}, "value": [timestamp, "0.245"]}
           ]
         }
       }
```

---

## SLO Tracking Architecture

### Recording Rules Hierarchy

```
Level 1: Raw Rate Calculations (30s window)
slo:http:request_rate:5m = rate(http_requests_total[5m])
slo:http:error_rate:5m = rate(http_requests_total{status=~"5.."}[5m])

Level 2: Success Ratio (5min, 30day)
slo:availability:ratio_rate5m =
  sum(rate(http_requests_total{status!~"5.."}[5m])) /
  sum(rate(http_requests_total[5m]))

slo:availability:ratio_rate30d =
  sum(rate(http_requests_total{status!~"5.."}[30d])) /
  sum(rate(http_requests_total[30d]))

Level 3: Burn Rates (for MWMBR alerts)
slo:availability:burn_rate:1h =
  (1 - slo:availability:ratio_rate1h) / (1 - 0.995)

slo:availability:burn_rate:6h =
  (1 - slo:availability:ratio_rate6h) / (1 - 0.995)

Level 4: Error Budget
slo:availability:error_budget_remaining =
  1 - (
    (sum(rate(http_requests_total{status=~"5.."}[30d])) /
     sum(rate(http_requests_total[30d]))) / 0.005
  )
```

**Why Hierarchical**:
- Reuse intermediate calculations
- Faster dashboard queries
- Consistency across alerts and dashboards

---

## Alerting Architecture

### Multi-Window Multi-Burn-Rate (MWMBR) Alerts

```
┌─────────────────────────────────────────────────────────┐
│ Fast Burn Alert (Critical)                              │
│                                                          │
│ Burn Rate: 14.4x normal (2% error budget per hour)      │
│ Windows: 1 hour AND 6 hours                             │
│ For: 2 minutes                                           │
│                                                          │
│ Logic:                                                   │
│   IF burn_rate_1h > 14.4 AND burn_rate_6h > 14.4       │
│   FOR 2 minutes                                          │
│   THEN page on-call                                      │
│                                                          │
│ Rationale:                                               │
│   - Detects severe outages quickly                       │
│   - Dual windows reduce false positives                  │
│   - 2min for duration = balance speed vs noise          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Slow Burn Alert (Warning)                               │
│                                                          │
│ Burn Rate: 6x normal (10% error budget per 3 days)      │
│ Windows: 3 days AND 7 days                              │
│ For: 15 minutes                                          │
│                                                          │
│ Logic:                                                   │
│   IF burn_rate_3d > 6 AND burn_rate_7d > 6             │
│   FOR 15 minutes                                         │
│   THEN send Slack alert                                  │
│                                                          │
│ Rationale:                                               │
│   - Catches gradual degradation                          │
│   - Time to investigate before crisis                    │
│   - Longer for duration = avoid noise                    │
└─────────────────────────────────────────────────────────┘
```

### Alert Routing Logic

```yaml
Routing Tree:

├─ match: severity=critical
│  ├─ receiver: pagerduty
│  └─ continue: true (also send to Slack)
│
├─ match: severity=critical (continued from above)
│  └─ receiver: slack-critical
│
├─ match: severity=warning, team=ml-platform
│  └─ receiver: slack-ml-team
│
├─ match: severity=warning, team=infrastructure
│  └─ receiver: slack-infra-team
│
└─ default
   └─ receiver: email-fallback
```

---

## Design Decisions

### 1. Why Pull-Based Scraping (vs Push)?

**Decision**: Use Prometheus pull model for metrics collection

**Rationale**:
- **Service Discovery**: Prometheus discovers targets automatically
- **Simpler Targets**: Services just expose `/metrics`, don't need to know where to push
- **Network Failure Handling**: If scrape fails, Prometheus knows (vs push silently fails)
- **Debugging**: Can manually curl `/metrics` to see what Prometheus sees
- **No Buffering**: Targets don't need to buffer metrics if collector is down

**Trade-offs**:
- ✅ Simpler target implementation
- ✅ Better failure detection
- ✅ Easier debugging
- ❌ Requires network reachability (Prometheus must reach targets)
- ❌ Short-lived jobs need Pushgateway

---

### 2. Why 15-Second Scrape Interval?

**Decision**: Scrape metrics every 15 seconds

**Rationale**:
- **Balance**: Good balance between freshness and storage cost
- **SLO Detection**: Can detect SLO violations within 30-60 seconds
- **Storage Cost**: Reasonable storage growth (~50GB for 30 days)
- **Network**: Low overhead (metrics are small)

**Alternatives Considered**:
- 10s: More granular but 50% more storage
- 30s: Less storage but slower alerting
- 60s: Much less storage but too coarse for SLO tracking

**Trade-offs**:
- ✅ Fast enough for SLO alerting (< 1 min detection)
- ✅ Reasonable storage costs
- ❌ Not suitable for sub-second anomaly detection

---

### 3. Why Multi-Window Multi-Burn-Rate (MWMBR) Alerts?

**Decision**: Use Google SRE's MWMBR methodology for SLO alerting

**Rationale**:
- **Fewer False Positives**: Dual windows filter transient spikes
- **Fast Detection**: Fast burn alerts fire within 2-5 minutes
- **Contextual Severity**: Different burn rates = different urgency
- **Error Budget Aware**: Alerts based on actual SLO consumption

**Alternatives Considered**:
- Simple threshold: `availability < 99.5%`
  - Problem: Too slow (only fires after SLO violated)
- Single burn rate: `burn_rate_1h > 14.4`
  - Problem: Too many false positives from spikes

**Trade-offs**:
- ✅ Very few false positives
- ✅ Actionable alerts (directly tied to SLO)
- ✅ Different urgency levels (fast burn vs slow burn)
- ❌ More complex to understand initially
- ❌ Requires recording rules for efficiency

---

### 4. Why 30-Day Retention?

**Decision**: Retain metrics for 30 days in Prometheus TSDB

**Rationale**:
- **SLO Window**: Availability SLO uses 30-day window
- **Capacity Planning**: Need 30 days of history for trends
- **Incident Investigation**: Most incidents investigated within 30 days
- **Cost-Effective**: Fits on reasonable disk (50-100GB)

**Long-Term Storage**:
For longer retention, use remote write to:
- **Thanos**: Long-term storage in S3/GCS
- **Cortex**: Multi-tenant Prometheus-as-a-Service
- **Mimir**: Grafana's long-term storage
- **VictoriaMetrics**: Fast, cost-efficient storage

---

## Scalability

### Vertical Scaling (Single Prometheus)

**Capacity Limits**:
```
Prometheus can handle:
- 10 million active time series
- 1 million samples/second ingestion
- 1000+ targets

Resource Requirements (for 1M series):
- CPU: 8-16 cores
- Memory: 32-64GB RAM
- Disk: 500GB-1TB SSD
- IOPS: 1000+ (SSD recommended)
```

**When to Scale Horizontally**:
- More than 10M active series
- Query latency > 5 seconds
- Scrape duration > 1 minute
- Disk I/O saturation

### Horizontal Scaling (Federation)

```
┌─────────────────────────────────────────────────────┐
│                 Global Prometheus                    │
│               (Aggregated Metrics)                   │
│                                                      │
│  Scrapes federated endpoints from regional servers  │
└───────────────────────┬─────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼───────┐ ┌────▼──────┐ ┌──────▼──────┐
│ Prometheus US │ │Prometheus  │ │ Prometheus  │
│    East       │ │   EU       │ │   APAC      │
│               │ │            │ │             │
│ - 100 targets │ │- 80 targets│ │- 60 targets │
└───────────────┘ └────────────┘ └─────────────┘
```

**Federation Query**:
```yaml
# Global Prometheus scrapes federated metrics
scrape_configs:
  - job_name: 'federate-us-east'
    scrape_interval: 60s
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]':
        - '{job="inference-gateway"}'
        - '{__name__=~"slo:.*"}'  # Only SLO metrics
    static_configs:
      - targets: ['prometheus-us-east:9090']
```

---

## High Availability

### Active-Passive HA

```
┌───────────────┐          ┌───────────────┐
│ Prometheus A  │          │ Prometheus B  │
│  (Active)     │          │  (Passive)    │
│               │          │               │
│ Scrapes all   │          │ Scrapes all   │
│ targets       │          │ targets       │
│               │          │               │
│ Sends alerts  │          │ Standby       │
└───────┬───────┘          └───────┬───────┘
        │                          │
        └──────────┬───────────────┘
                   │
            ┌──────▼──────┐
            │Alertmanager │
            │  (HA pair)  │
            └─────────────┘
```

**Configuration**:
```yaml
# Both Prometheus instances scrape same targets
# Alertmanager deduplicates identical alerts

Alertmanager Cluster:
- alertmanager-1: 10.0.1.10:9093
- alertmanager-2: 10.0.1.11:9093

Gossip port: 9094 (cluster communication)
```

**Benefits**:
- Zero data loss (both instances scrape independently)
- Alert deduplication by Alertmanager
- Failover: If Prometheus A dies, B continues

---

## Summary

**Architecture Principles**:
- ✅ Pull-based metrics collection (Prometheus scrapes targets)
- ✅ Pre-aggregation with recording rules (fast queries)
- ✅ SLO-based alerting with MWMBR (actionable alerts)
- ✅ Hierarchical alert routing (right notification to right team)
- ✅ Modular exporters (easy to add new metrics sources)

**Key Metrics**:
- Scrape interval: 15s
- Retention: 30 days
- Exporters: 7 (Node, cAdvisor, Blackbox, ML, Pushgateway, App, Prometheus)
- Alert evaluation: 15s
- Recording rule evaluation: 30s

**Scalability**:
- Vertical: Up to 10M time series per Prometheus
- Horizontal: Federation for global aggregation
- HA: Active-passive with Alertmanager clustering

**Future Enhancements**:
- Add Thanos for long-term storage (>30 days)
- Implement remote write to Mimir/Cortex
- Add Prometheus Operator for Kubernetes
- Implement multi-cluster federation
