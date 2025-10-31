# Exercise 02: Prometheus Monitoring Stack - Complete Solution

## Overview

This is a **production-ready Prometheus monitoring stack** for ML infrastructure, implementing comprehensive monitoring, SLO tracking, and intelligent alerting. The solution builds upon Exercise 01's instrumented inference gateway and provides a complete observability platform.

## 🎯 Solution Highlights

### Complete Monitoring Stack
- ✅ **Prometheus Server** (v2.48.0) with 30-day retention
- ✅ **Alertmanager** (v0.26.0) with multi-channel routing
- ✅ **Node Exporter** for infrastructure metrics
- ✅ **cAdvisor** for container metrics
- ✅ **Custom ML Model Exporter** for ML-specific metrics
- ✅ **Blackbox Exporter** for endpoint probing
- ✅ **Pushgateway** for batch job metrics

### Advanced Features
- ✅ **Multi-Window Multi-Burn-Rate (MWMBR) Alerts** for SLO violations
- ✅ **Comprehensive Recording Rules** for efficient SLO queries
- ✅ **Intelligent Alert Routing** (PagerDuty, Slack, Email)
- ✅ **Service Discovery** with Docker labels
- ✅ **Metric Relabeling** for cardinality control
- ✅ **Federation Support** (commented, production-ready)
- ✅ **Remote Write** configuration for long-term storage

## 📁 Project Structure

```
exercise-02-prometheus-stack/
├── config/
│   ├── prometheus/
│   │   ├── prometheus.yml              # Main Prometheus configuration
│   │   ├── recording_rules.yml         # SLO recording rules
│   │   └── alerting_rules.yml          # Alert definitions
│   ├── alertmanager/
│   │   ├── alertmanager.yml            # Alertmanager routing
│   │   └── templates/
│   │       └── slack.tmpl              # Slack notification templates
│   └── exporters/
│       └── blackbox.yml                # Blackbox exporter config
├── exporters/
│   └── ml-model-exporter/
│       ├── exporter.py                 # Custom ML metrics exporter
│       ├── Dockerfile                  # Exporter container image
│       └── requirements.txt            # Python dependencies
├── docker/
├── scripts/
│   ├── setup.sh                        # Environment setup script
│   └── test-alerts.sh                  # Alert testing utility
├── docs/
│   ├── ARCHITECTURE.md                 # System architecture
│   ├── SLO_TRACKING.md                 # SLO implementation guide
│   └── RUNBOOKS.md                     # Incident response runbooks
├── docker-compose.yml                  # Full stack orchestration
├── .env.example                        # Environment configuration template
└── README.md                           # This file
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Exercise 01 inference-gateway image built
- Basic understanding of Prometheus and PromQL

### 1. Setup

```bash
# Run setup script
./scripts/setup.sh

# Edit .env file with your credentials
cp .env.example .env
nano .env

# Add your Slack webhook, PagerDuty key, SMTP credentials
```

### 2. Start the Stack

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f prometheus alertmanager
```

### 3. Access Services

| Service | URL | Description |
|---------|-----|-------------|
| Prometheus | http://localhost:9090 | Metrics database and query interface |
| Alertmanager | http://localhost:9093 | Alert management and routing |
| Inference Gateway | http://localhost:8000 | ML inference service (Exercise 01) |
| cAdvisor | http://localhost:8080 | Container metrics |
| Node Exporter | http://localhost:9100/metrics | Host metrics |
| ML Model Exporter | http://localhost:9101 | Custom ML metrics |
| Blackbox Exporter | http://localhost:9115 | Endpoint probing |
| Pushgateway | http://localhost:9091 | Batch job metrics |

## 📊 SLO Tracking

### Defined SLOs

#### 1. Availability SLO: 99.5%
- **Measurement**: `(successful_requests / total_requests) * 100`
- **Window**: Rolling 30 days
- **Error Budget**: 0.5% (21.6 minutes/month)
- **Query**:
  ```promql
  slo:availability:ratio_rate30d
  ```

#### 2. Latency SLO: P99 < 300ms
- **Measurement**: 99th percentile of `/predict` endpoint duration
- **Window**: Rolling 7 days
- **Query**:
  ```promql
  slo:http_request_duration:p99:rate7d
  ```

### Recording Rules

All SLO metrics are pre-computed via **recording rules** for fast dashboard queries:

```promql
# Availability metrics (updated every 30s)
slo:availability:ratio_rate5m          # 5-minute availability
slo:availability:ratio_rate30d         # 30-day availability (SLO compliance)
slo:availability:error_budget_remaining # Remaining error budget (0-1)

# Latency metrics
slo:http_request_duration:p50:rate5m  # P50 latency
slo:http_request_duration:p95:rate5m  # P95 latency
slo:http_request_duration:p99:rate5m  # P99 latency (SLO target)
slo:http_request_duration:p99:rate7d  # P99 latency (7-day SLO window)

# Burn rate metrics (for MWMBR alerts)
slo:availability:burn_rate:1h          # 1-hour burn rate
slo:availability:burn_rate:6h          # 6-hour burn rate
slo:availability:burn_rate:3d          # 3-day burn rate
```

## 🚨 Alerting

### Multi-Window Multi-Burn-Rate (MWMBR) Alerts

Google SRE-recommended approach for accurate SLO alerting:

#### Fast Burn Alert (Critical)
- **Condition**: 14.4x burn rate sustained for 1h and 6h
- **Meaning**: 2% of monthly error budget consumed in 1 hour
- **Impact**: Entire budget exhausted in ~2 hours at this rate
- **Action**: Page on-call engineer immediately
- **Notification**: PagerDuty + Slack

```yaml
alert: SLOAvailabilityFastBurn
expr: |
  slo:availability:burn_rate:1h > 14.4
  and
  slo:availability:burn_rate:6h > 14.4
for: 2m
```

#### Slow Burn Alert (Warning)
- **Condition**: 6x burn rate sustained for 3 days
- **Meaning**: 10% of monthly error budget consumed over 3 days
- **Impact**: Budget exhausted in ~5 days at this rate
- **Action**: Create incident, investigate
- **Notification**: Slack

```yaml
alert: SLOAvailabilitySlowBurn
expr: |
  slo:availability:burn_rate:3d > 6
for: 15m
```

### Alert Routing

Alertmanager routes alerts based on severity and type:

| Alert Type | Severity | Receivers | Repeat Interval |
|------------|----------|-----------|-----------------|
| SLO Fast Burn | Critical | PagerDuty + Slack | 15 minutes |
| SLO Slow Burn | Warning | Slack | 3 hours |
| Service Down | Critical | PagerDuty + Slack | 15 minutes |
| High Error Rate | Critical | PagerDuty + Slack | 15 minutes |
| Resource Alerts | Warning/Critical | Slack (infra channel) | 6 hours |
| Model Quality | Warning | Slack (ML team) | 12 hours |
| Monitoring Issues | Warning | Slack (platform team) | 4 hours |

### Inhibition Rules

Suppress noise with intelligent alert inhibition:

- **Warning suppressed if critical firing** for same service
- **Slow burn suppressed if fast burn firing**
- **Resource alerts suppressed if service down**
- **High usage suppressed if critical usage firing**

## 🛠️ Operations

### Validating Configuration

```bash
# Validate Prometheus config
docker run --rm \
  -v $(pwd)/config/prometheus:/etc/prometheus \
  prom/prometheus:v2.48.0 \
  promtool check config /etc/prometheus/prometheus.yml

# Validate recording rules
docker run --rm \
  -v $(pwd)/config/prometheus:/etc/prometheus \
  prom/prometheus:v2.48.0 \
  promtool check rules /etc/prometheus/recording_rules.yml

# Validate alerting rules
docker run --rm \
  -v $(pwd)/config/prometheus:/etc/prometheus \
  prom/prometheus:v2.48.0 \
  promtool check rules /etc/prometheus/alerting_rules.yml

# Validate Alertmanager config
docker run --rm \
  -v $(pwd)/config/alertmanager:/etc/alertmanager \
  prom/alertmanager:v0.26.0 \
  amtool check-config /etc/alertmanager/alertmanager.yml
```

### Reloading Configuration

```bash
# Reload Prometheus (no downtime)
curl -X POST http://localhost:9090/-/reload

# Reload Alertmanager (no downtime)
curl -X POST http://localhost:9093/-/reload

# Or restart services
docker-compose restart prometheus alertmanager
```

### Querying Metrics

```bash
# Check all targets status
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# View current availability SLO
curl -s 'http://localhost:9090/api/v1/query?query=slo:availability:ratio_rate30d' | jq '.data.result[0].value[1]'

# View P99 latency
curl -s 'http://localhost:9090/api/v1/query?query=slo:http_request_duration:p99:rate5m' | jq '.data.result[0].value[1]'

# View error budget remaining
curl -s 'http://localhost:9090/api/v1/query?query=slo:availability:error_budget_remaining' | jq '.data.result[0].value[1]'

# View active alerts
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing") | {alert: .labels.alertname, severity: .labels.severity}'
```

### Testing Alerts

```bash
# Run interactive alert testing
./scripts/test-alerts.sh

# Manual alert testing
# 1. Generate high load
for i in {1..1000}; do curl -s http://localhost:8000/health > /dev/null; done

# 2. Simulate errors
for i in {1..100}; do curl -s http://localhost:8000/nonexistent > /dev/null; done

# 3. Check alerts fired (wait 2-5 minutes)
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing")'

# 4. Check Alertmanager received them
curl -s http://localhost:9093/api/v2/alerts | jq .
```

### Managing Silences

```bash
# Create silence via API
curl -X POST http://localhost:9093/api/v2/silences \
  -H 'Content-Type: application/json' \
  -d '{
    "matchers": [
      {"name": "alertname", "value": "HighCPUUsage", "isRegex": false}
    ],
    "startsAt": "2025-10-23T12:00:00Z",
    "endsAt": "2025-10-23T14:00:00Z",
    "createdBy": "ops-team",
    "comment": "Planned maintenance"
  }'

# View active silences
curl -s http://localhost:9093/api/v2/silences | jq .

# Or use the web UI
open http://localhost:9093/#/silences
```

## 🔧 Customization

### Adding New Scrape Targets

Edit `config/prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'my-new-service'
    static_configs:
      - targets: ['my-service:8080']
        labels:
          service: 'my-service'
          environment: 'production'
```

### Adding New Alerts

Edit `config/prometheus/alerting_rules.yml`:

```yaml
groups:
  - name: custom_alerts
    rules:
      - alert: MyCustomAlert
        expr: my_metric > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "My metric is too high"
          description: "Value: {{ $value }}"
```

### Customizing Alert Notifications

Edit `config/alertmanager/alertmanager.yml`:

```yaml
receivers:
  - name: 'my-custom-receiver'
    slack_configs:
      - channel: '#my-channel'
        username: 'MyBot'
        text: 'Custom notification template'
```

## 📈 Example PromQL Queries

### SLO Queries

```promql
# Current availability (last 5 minutes)
slo:availability:ratio_rate5m

# 30-day availability (SLO compliance)
slo:availability:ratio_rate30d

# Error budget consumption rate
rate(http_requests_total{status=~"5.."}[1h]) / (1 - 0.995)

# P99 latency current
slo:http_request_duration:p99:rate5m

# Requests under 300ms (latency SLO compliance)
sum(rate(http_request_duration_seconds_bucket{le="0.3"}[5m])) / sum(rate(http_request_duration_seconds_count[5m])) * 100
```

### Infrastructure Queries

```promql
# CPU usage by container
container:cpu_usage:percent

# Memory usage by container
container:memory_usage:percent

# Disk space remaining
(node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100

# Network traffic rate
container:network_transmit:rate5m_mb
```

### Application Queries

```promql
# Request rate by endpoint
http:requests:rate5m

# Error rate percentage
http:error_rate:percent:rate5m

# Average request duration
http:request_duration:avg:rate5m

# Top 5 slowest endpoints
topk(5, http:request_duration:avg:rate5m)
```

## 🏗️ Architecture

### Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                      Monitored Targets                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │  Inference  │  │    Node     │  │   Custom    │           │
│  │   Gateway   │  │  Exporter   │  │  Exporters  │           │
│  │   :8000     │  │   :9100     │  │  :9101-9115 │           │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │
│         │                 │                 │                  │
│         └─────────────────┴─────────────────┘                 │
│                           │ HTTP Pull (15s interval)          │
└───────────────────────────┼───────────────────────────────────┘
                            │
                 ┌──────────▼──────────┐
                 │  Prometheus Server  │
                 │                     │
                 │  - TSDB Storage     │
                 │  - Recording Rules  │
                 │  - Alert Evaluation │
                 │  - PromQL Engine    │
                 └──────────┬──────────┘
                            │ Alerts
                 ┌──────────▼──────────┐
                 │    Alertmanager     │
                 │                     │
                 │  - Grouping         │
                 │  - Routing          │
                 │  - Silencing        │
                 │  - Inhibition       │
                 └──────────┬──────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
    ┌─────▼─────┐    ┌──────▼──────┐   ┌─────▼─────┐
    │ PagerDuty │    │    Slack    │   │   Email   │
    │  (Page)   │    │ (Warnings)  │   │(Fallback) │
    └───────────┘    └─────────────┘   └───────────┘
```

### Component Responsibilities

| Component | Purpose | Metrics Exposed |
|-----------|---------|-----------------|
| Prometheus | Metrics storage, query, alerting | prometheus_* |
| Alertmanager | Alert routing and notification | alertmanager_* |
| Node Exporter | Host-level metrics | node_* |
| cAdvisor | Container metrics | container_* |
| ML Model Exporter | ML-specific metrics | ml_* |
| Blackbox Exporter | Endpoint availability | probe_* |
| Pushgateway | Batch job metrics | push_* |

## 📚 Learning Outcomes

This solution demonstrates:

✅ **Comprehensive Monitoring Stack** - Production-ready Prometheus deployment
✅ **SLO-Based Alerting** - Multi-window multi-burn-rate (Google SRE methodology)
✅ **Recording Rules** - Efficient pre-computation of SLO metrics
✅ **Alert Routing** - Intelligent notification delivery (PagerDuty, Slack)
✅ **Service Discovery** - Dynamic target discovery with Docker labels
✅ **Metric Relabeling** - Cardinality control and label normalization
✅ **Custom Exporters** - Building domain-specific metrics collectors
✅ **Infrastructure Monitoring** - Complete stack from hardware to application
✅ **Operational Excellence** - Validation, testing, and incident response tools

## 🔗 Integration with Other Exercises

- **Exercise 01**: Monitors instrumented inference gateway
- **Exercise 03**: Metrics feed into Grafana dashboards
- **Exercise 04**: Logs correlation with metrics via trace IDs
- **Exercise 05**: Alerts trigger incident response workflows

## 📖 Additional Resources

- **Prometheus Documentation**: https://prometheus.io/docs/
- **Alertmanager Guide**: https://prometheus.io/docs/alerting/latest/alertmanager/
- **SRE Workbook (Google)**: https://sre.google/workbook/alerting-on-slos/
- **PromQL Cheat Sheet**: https://promlabs.com/promql-cheat-sheet/

## 🎉 Success Criteria

- ✅ All services running and healthy
- ✅ Prometheus scraping all targets successfully
- ✅ Recording rules computing SLO metrics correctly
- ✅ Alerts firing and routing to correct channels
- ✅ Metrics retained for 30 days
- ✅ Custom ML exporter providing model metrics
- ✅ Configuration validated and production-ready

---

**🎓 Ready for Exercise 03: Grafana Dashboards!**
