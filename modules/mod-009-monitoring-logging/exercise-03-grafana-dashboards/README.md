# Exercise 03: Grafana Dashboards - Complete Solution

## Overview

This is a **production-ready Grafana dashboard suite** for ML infrastructure monitoring, providing comprehensive visualization of SLOs, application performance, and infrastructure health. The solution builds upon Exercises 01-02 to transform raw Prometheus metrics into actionable insights.

## 🎯 Solution Highlights

### Complete Dashboard Suite
- ✅ **SLO Overview Dashboard** - Real-time SLO compliance tracking
- ✅ **Application Performance Dashboard** - Request rates, latency, errors
- ✅ **Infrastructure Health Dashboard** - CPU, memory, network metrics
- ✅ **Programmatic Dashboard Generation** - Python script for maintainability
- ✅ **Multi-Data Source Integration** - Prometheus, Loki, Jaeger

### Advanced Features
- ✅ **Dashboard-as-Code** - Version-controlled JSON configurations
- ✅ **Automatic Provisioning** - Data sources and dashboards auto-configured
- ✅ **Trace-Metrics-Logs Correlation** - Unified observability
- ✅ **Responsive Design** - Optimized for different screen sizes
- ✅ **Custom Time Ranges** - Flexible time window selection

## 📁 Project Structure

```
exercise-03-grafana-dashboards/
├── config/
│   ├── grafana/
│   │   └── provisioning/
│   │       ├── datasources/
│   │       │   └── datasources.yml          # Prometheus, Loki, Jaeger configs
│   │       ├── dashboards/
│   │       │   └── dashboards.yml           # Dashboard provisioning config
│   │       ├── alerting/                    # Grafana unified alerting
│   │       └── notifiers/                   # Notification channels
│   └── dashboards/
│       ├── ml-platform/
│       │   ├── slo-overview.json            # SLO tracking dashboard
│       │   └── application-performance.json  # App metrics dashboard
│       ├── infrastructure/
│       │   └── infrastructure-health.json    # Infrastructure dashboard
│       ├── executive/                        # Executive summary dashboards
│       └── default/                          # Home dashboards
├── scripts/
│   ├── generate-dashboards.py               # Dashboard generator (Python)
│   └── setup.sh                             # Environment setup script
├── docker-compose.yml                       # Full stack with Grafana
├── .env.example                             # Environment configuration template
└── README.md                                # This file
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Exercises 01-02 completed (or Exercise 02 Prometheus stack running)
- Python 3.7+ (for dashboard generation)

### 1. Setup

```bash
# Run setup script
./scripts/setup.sh

# Edit .env file
cp .env.example .env
nano .env

# Set admin credentials and SMTP settings if needed
```

### 2. Start the Stack

```bash
# Start all services (Grafana, Prometheus, Loki, Jaeger)
docker-compose up -d

# Check service health
docker-compose ps

# View Grafana logs
docker-compose logs -f grafana
```

### 3. Access Grafana

```bash
# Open in browser
open http://localhost:3000

# Default login
# Username: admin
# Password: admin
# (Change password on first login)
```

### 4. Explore Dashboards

Navigate to **Dashboards → Browse** and explore:

1. **ML Platform** folder:
   - SLO Overview
   - Application Performance

2. **Infrastructure** folder:
   - Infrastructure Health

## 📊 Dashboard Details

### 1. SLO Overview Dashboard

**Purpose**: Real-time SLO compliance tracking for on-call engineers

**Panels**:
- **Availability SLO (30d)** - Gauge showing current 30-day availability vs 99.5% target
- **Latency P99 (7d)** - Gauge showing P99 latency vs 300ms target
- **Error Budget Remaining** - Percentage of monthly error budget left
- **Burn Rate (1h)** - Current error budget consumption rate
- **Availability Trends** - Time series of 5-minute availability percentages
- **Error Budget Consumption** - Burn rates across 1h, 6h, and 3d windows
- **Latency Percentiles** - P50, P95, P99 latency over time

**Use Cases**:
- Monitor SLO compliance during incidents
- Track error budget consumption
- Identify availability and latency degradation trends
- Determine if deployments impact SLOs

**Screenshot Equivalent**:
```
┌─────────────────────────────────────────────────────────────────┐
│ SLO Overview - ML Inference Platform              Last 6h  ▼   │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│ Availability│  Latency P99│ Error Budget│    Burn Rate (1h)   │
│             │             │             │                     │
│   99.87%    │   245 ms    │   76.2%     │       2.1x          │
│   ██████    │   ██████    │   ████      │       Low           │
│   Target:   │   Target:   │   Status:   │    Fast: 14.4x      │
│   99.5%     │   < 300ms   │   Healthy   │    Slow: 6.0x       │
└─────────────┴─────────────┴─────────────┴─────────────────────┘
```

### 2. Application Performance Dashboard

**Purpose**: Monitor inference gateway request metrics and errors

**Panels**:
- **Request Rate (QPS)** - Queries per second
- **Error Rate** - Percentage of failed requests
- **Avg Response Time** - Mean latency in milliseconds
- **Active Requests** - Current in-flight requests
- **Request Rate by Endpoint** - Traffic breakdown by API endpoint
- **Error Rate Over Time** - 5xx errors by status code

**Use Cases**:
- Diagnose traffic spikes or drops
- Identify high-error endpoints
- Track latency degradation
- Monitor system saturation

### 3. Infrastructure Health Dashboard

**Purpose**: Monitor container and host resource utilization

**Panels**:
- **CPU Usage by Container** - CPU percentage per service
- **Memory Usage by Container** - Memory percentage per service
- **Network Receive Rate** - Inbound network traffic (MB/s)
- **Network Transmit Rate** - Outbound network traffic (MB/s)

**Use Cases**:
- Identify resource-constrained services
- Detect memory leaks
- Monitor network saturation
- Plan capacity upgrades

## 🔧 Operations

### Regenerating Dashboards

```bash
# Edit dashboard generator
nano scripts/generate-dashboards.py

# Regenerate dashboard JSON files
python3 scripts/generate-dashboards.py

# Restart Grafana to reload
docker-compose restart grafana
```

### Adding Custom Dashboards

```python
# In scripts/generate-dashboards.py, add a new function:

def create_custom_dashboard():
    db = DashboardBuilder(
        title="My Custom Dashboard",
        uid="custom-dashboard",
        tags=["Custom"]
    )

    db.add_stat_panel(
        title="My Metric",
        query="my_custom_metric",
        unit="short",
        x=0, y=0, w=6, h=4
    )

    return db.to_json()

# Add to main() function:
dashboards = {
    # ... existing dashboards
    "default/custom-dashboard.json": create_custom_dashboard(),
}
```

### Checking Data Sources

```bash
# Via API
curl -u admin:admin http://localhost:3000/api/datasources | jq .

# Test Prometheus connection
curl -u admin:admin http://localhost:3000/api/datasources/proxy/1/api/v1/query?query=up | jq .
```

### Exporting Dashboards

```bash
# Export dashboard JSON via API
DASHBOARD_UID="slo-overview"
curl -u admin:admin "http://localhost:3000/api/dashboards/uid/${DASHBOARD_UID}" | jq .dashboard > exported-dashboard.json
```

### Importing Dashboards

```bash
# Via UI: Dashboards → Import → Upload JSON file

# Via API:
curl -X POST -H "Content-Type: application/json" \
  -u admin:admin \
  -d @dashboard.json \
  http://localhost:3000/api/dashboards/db
```

## 📈 PromQL Queries Used

### SLO Queries

```promql
# Availability (30-day)
slo:availability:ratio_rate30d

# Latency P99 (7-day)
slo:http_request_duration:p99:rate7d

# Error budget remaining
slo:availability:error_budget_remaining * 100

# Burn rate (1h)
slo:availability:burn_rate:1h
```

### Application Queries

```promql
# Request rate
sum(rate(http_requests_total{service="inference-gateway"}[5m]))

# Error rate percentage
sum(rate(http_requests_total{service="inference-gateway",status=~"5.."}[5m]))
/
sum(rate(http_requests_total{service="inference-gateway"}[5m]))
* 100

# Average latency
sum(rate(http_request_duration_seconds_sum{service="inference-gateway"}[5m]))
/
sum(rate(http_request_duration_seconds_count{service="inference-gateway"}[5m]))
* 1000
```

### Infrastructure Queries

```promql
# CPU usage by container
container:cpu_usage:percent

# Memory usage by container
container:memory_usage:percent

# Network receive rate
container:network_receive:rate5m_mb

# Network transmit rate
container:network_transmit:rate5m_mb
```

## 🏗️ Architecture

### Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                     Data Sources                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│  │ Prometheus │  │    Loki    │  │   Jaeger   │          │
│  │  :9090     │  │   :3100    │  │  :16686    │          │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘          │
│         │                │                │               │
│         └────────────────┴────────────────┘               │
│                          │ Queries                        │
└──────────────────────────┼────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │      Grafana        │
                │                     │
                │  - Dashboard Engine │
                │  - Query Processor  │
                │  - Data Aggregation │
                │  - Visualization    │
                └──────────┬──────────┘
                           │ HTTP (Port 3000)
                  ┌────────▼────────┐
                  │   Web Browser   │
                  │   (Dashboard)   │
                  └─────────────────┘
```

### Component Responsibilities

| Component | Purpose | Port |
|-----------|---------|------|
| Grafana | Dashboard visualization and queries | 3000 |
| Prometheus | Metrics data source | 9090 |
| Loki | Logs data source | 3100 |
| Jaeger | Traces data source | 16686 |

## 🎨 Dashboard Design Principles

### Best Practices Implemented

1. **Golden Signals First** - Latency, traffic, errors, saturation prominently displayed
2. **SLO-Driven** - Dashboards focused on what matters for reliability
3. **Actionable** - Each panel enables specific operational decisions
4. **Consistent Layout** - Stats at top, graphs below, critical metrics left-to-right
5. **Appropriate Visualization** - Gauges for current state, time series for trends
6. **Color-Blind Friendly** - Uses shapes and thresholds, not just colors

### Panel Types Used

- **Stat Panels**: Single values with thresholds (QPS, error rate, budget)
- **Gauge Panels**: Progress toward goals (SLO compliance, resource usage)
- **Time Series**: Trends over time (latency, availability, traffic)
- **Rows**: Collapsible sections for organization

## 🔗 Integration Points

### With Exercise 01 (Observability Foundations)
- Visualizes Four Golden Signals metrics
- Displays SLIs defined in Exercise 01
- Queries exposed `/metrics` endpoint

### With Exercise 02 (Prometheus Stack)
- Uses recording rules for fast queries
- Displays alert state from Alertmanager
- Leverages service discovery labels

### With Exercise 04 (Logging Pipeline)
- Loki data source pre-configured
- Trace ID drill-down to logs
- Log panel templates ready

### With Exercise 05 (Incident Response)
- Dashboards linked from alert annotations
- Incident timeline overlays
- Root cause analysis views

## 📚 Learning Outcomes

This solution demonstrates:

✅ **Grafana Deployment** - Production configuration with Docker
✅ **Data Source Configuration** - Multi-source integration (Prometheus, Loki, Jaeger)
✅ **Dashboard Design** - UX best practices and visualization theory
✅ **Dashboard-as-Code** - Programmatic generation with Python
✅ **Provisioning** - Automatic configuration on startup
✅ **PromQL Mastery** - Complex queries for SLO tracking
✅ **Correlation** - Metrics, logs, and traces linked
✅ **Operational Dashboards** - Role-specific views (on-call, ops, exec)

## 🎓 Advanced Topics

### Custom Panel Plugins

```bash
# Install custom panel plugins
docker-compose exec grafana grafana-cli plugins install grafana-piechart-panel
docker-compose restart grafana
```

### Dashboard Variables

Add dynamic filtering with template variables:

```python
db.add_variable(
    name="service",
    query='label_values(up, service)',
    label="Service",
    all_value=True
)
```

Use in queries:
```promql
up{service="$service"}
```

### Annotations

Display deployment events on dashboards via annotations API:

```bash
curl -X POST http://localhost:3000/api/annotations \
  -H "Content-Type: application/json" \
  -u admin:admin \
  -d '{
    "dashboardUID": "slo-overview",
    "time": '$(date +%s)'000,
    "text": "Deployment v1.2.3",
    "tags": ["deployment"]
  }'
```

## 🎉 Success Criteria

- ✅ Grafana accessible at http://localhost:3000
- ✅ All 3 data sources (Prometheus, Loki, Jaeger) healthy
- ✅ 3 dashboards auto-loaded and functional
- ✅ SLO metrics displaying correctly
- ✅ Drill-down from metrics to traces working
- ✅ Dashboards regenerable via Python script

---

**🎓 Ready for Exercise 04: Logging Pipeline!**
