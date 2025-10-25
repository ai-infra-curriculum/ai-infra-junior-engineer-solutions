# Step-by-Step Implementation Guide: Grafana Dashboards

## Overview

This guide walks you through creating production-ready Grafana dashboards for ML infrastructure monitoring, from basic setup through advanced dashboard-as-code practices.

**Estimated Time**: 2-3 hours
**Difficulty**: Intermediate

---

## Phase 1: Grafana Setup (20 minutes)

### Step 1: Start Grafana with Docker Compose

```bash
cd exercise-03-grafana-dashboards

# Create docker-compose.yml
cat > docker-compose.yml <<'EOF'
version: '3.8'

services:
  grafana:
    image: grafana/grafana:10.2.2
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
    volumes:
      - ./config/grafana/provisioning:/etc/grafana/provisioning
      - grafana_data:/var/lib/grafana
    networks:
      - monitoring

  prometheus:
    image: prom/prometheus:v2.48.0
    container_name: prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
    volumes:
      - ./config/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - monitoring

volumes:
  grafana_data:
  prometheus_data:

networks:
  monitoring:
EOF

# Start services
docker-compose up -d

# Access Grafana
open http://localhost:3000
# Login: admin / admin
```

---

## Phase 2: Data Source Configuration (15 minutes)

### Step 2: Auto-Provision Prometheus Data Source

Create **`config/grafana/provisioning/datasources/datasources.yml`**:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      httpMethod: POST
      timeInterval: 15s
    editable: true

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    jsonData:
      maxLines: 1000
    editable: true

  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    jsonData:
      tracesToLogsV2:
        datasourceUid: 'loki'
        tags: ['trace_id']
    editable: true
```

**Restart Grafana**:
```bash
docker-compose restart grafana

# Verify data source in Grafana UI
# Configuration > Data sources > Prometheus (should show green checkmark)
```

---

## Phase 3: Create SLO Overview Dashboard (30 minutes)

### Step 3: Create Dashboard Manually (Understanding UI)

1. **Create New Dashboard**:
   - Grafana UI → Dashboards → New Dashboard → Add Visualization

2. **Add Availability SLO Panel**:
```json
Panel Title: Availability SLO (99.5% target)
Visualization: Stat
Query:
  slo:availability:ratio_rate30d * 100

Options:
  - Unit: Percent (0-100)
  - Thresholds:
    - Red: < 99.5
    - Yellow: 99.5-99.9
    - Green: > 99.9
  - Value Mappings: None
```

3. **Add Error Budget Panel**:
```json
Panel Title: Error Budget Remaining
Visualization: Gauge
Query:
  slo:availability:error_budget_remaining * 100

Options:
  - Unit: Percent (0-100)
  - Min: 0, Max: 100
  - Thresholds:
    - Red: < 10
    - Yellow: 10-50
    - Green: > 50
```

4. **Add P99 Latency Panel**:
```json
Panel Title: P99 Latency (300ms target)
Visualization: Time series
Query A:
  histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{endpoint="/predict"}[5m])) * 1000

Options:
  - Unit: milliseconds (ms)
  - Legend: {{endpoint}}
  - Thresholds:
    - Green: < 300ms
    - Yellow: 300-500ms
    - Red: > 500ms
```

5. **Save Dashboard**:
   - Click Save icon
   - Name: "SLO Overview"
   - Folder: ML Platform

---

## Phase 4: Dashboard as Code (45 minutes)

### Step 4: Export Dashboard JSON

```bash
# In Grafana UI, open dashboard settings
# → JSON Model → Copy JSON

# Save to file
mkdir -p config/dashboards/ml-platform
cat > config/dashboards/ml-platform/slo-overview.json <<'EOF'
{
  "annotations": {
    "list": []
  },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 1,
  "id": null,
  "links": [],
  "liveNow": false,
  "panels": [
    {
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {"color": "red", "value": 0},
              {"color": "yellow", "value": 99.5},
              {"color": "green", "value": 99.9}
            ]
          },
          "unit": "percent"
        }
      },
      "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
      "id": 1,
      "options": {
        "colorMode": "value",
        "graphMode": "area",
        "justifyMode": "auto",
        "orientation": "horizontal",
        "reduceOptions": {
          "calcs": ["lastNotNull"],
          "fields": "",
          "values": false
        },
        "textMode": "auto"
      },
      "pluginVersion": "10.2.2",
      "targets": [
        {
          "expr": "slo:availability:ratio_rate30d * 100",
          "refId": "A"
        }
      ],
      "title": "Availability SLO (99.5% target)",
      "type": "stat"
    }
  ],
  "refresh": "30s",
  "schemaVersion": 38,
  "style": "dark",
  "tags": ["ml-platform", "slo"],
  "templating": {
    "list": []
  },
  "time": {"from": "now-24h", "to": "now"},
  "timepicker": {},
  "timezone": "",
  "title": "SLO Overview",
  "uid": "slo-overview",
  "version": 1
}
EOF
```

### Step 5: Auto-Provision Dashboard

Create **`config/grafana/provisioning/dashboards/dashboards.yml`**:

```yaml
apiVersion: 1

providers:
  - name: 'ML Platform'
    orgId: 1
    folder: 'ML Platform'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards/ml-platform
      foldersFromFilesStructure: true
```

**Mount dashboards in docker-compose.yml**:
```yaml
grafana:
  volumes:
    - ./config/grafana/provisioning:/etc/grafana/provisioning
    - ./config/dashboards:/etc/grafana/provisioning/dashboards  # Add this
```

**Restart and verify**:
```bash
docker-compose restart grafana

# Dashboard should auto-appear in Grafana UI
# → Dashboards → ML Platform → SLO Overview
```

---

## Phase 5: Programmatic Dashboard Generation (40 minutes)

### Step 6: Create Dashboard Generator Script

Create **`scripts/generate-dashboards.py`**:

```python
#!/usr/bin/env python3
"""
Generate Grafana dashboards programmatically using grafanalib.
"""
import json
from grafanalib.core import (
    Dashboard, TimeSeries, Target, GridPos,
    Stat, Gauge, Threshold, MILLISECONDS,
    PERCENT_UNIT, SECONDS_FORMAT, RowPanel
)
from grafanalib._gen import DashboardEncoder


def create_slo_dashboard():
    """Create SLO Overview Dashboard."""

    dashboard = Dashboard(
        title="SLO Overview (Generated)",
        tags=["ml-platform", "slo", "generated"],
        timezone="browser",
        refresh="30s",
        panels=[
            # Row: SLO Compliance
            RowPanel(gridPos=GridPos(h=1, w=24, x=0, y=0), title="SLO Compliance"),

            # Availability SLO
            Stat(
                title="Availability SLO (99.5% target)",
                dataSource="Prometheus",
                targets=[
                    Target(expr="slo:availability:ratio_rate30d * 100", refId="A")
                ],
                gridPos=GridPos(h=8, w=6, x=0, y=1),
                unit=PERCENT_UNIT,
                thresholds=[
                    Threshold(value=0, color="red"),
                    Threshold(value=99.5, color="yellow"),
                    Threshold(value=99.9, color="green"),
                ],
            ),

            # Error Budget
            Gauge(
                title="Error Budget Remaining",
                dataSource="Prometheus",
                targets=[
                    Target(expr="slo:availability:error_budget_remaining * 100", refId="A")
                ],
                gridPos=GridPos(h=8, w=6, x=6, y=1),
                unit=PERCENT_UNIT,
                min=0,
                max=100,
                thresholds=[
                    Threshold(value=0, color="red"),
                    Threshold(value=10, color="yellow"),
                    Threshold(value=50, color="green"),
                ],
            ),

            # P99 Latency
            TimeSeries(
                title="P99 Latency (300ms target)",
                dataSource="Prometheus",
                targets=[
                    Target(
                        expr='histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{endpoint="/predict"}[5m])) * 1000',
                        refId="A",
                        legendFormat="P99 Latency"
                    )
                ],
                gridPos=GridPos(h=8, w=12, x=12, y=1),
                unit=MILLISECONDS,
                thresholds=[
                    Threshold(value=300, color="green"),
                    Threshold(value=500, color="yellow"),
                    Threshold(value=1000, color="red"),
                ],
            ),

            # Row: Request Metrics
            RowPanel(gridPos=GridPos(h=1, w=24, x=0, y=9), title="Request Metrics"),

            # Request Rate
            TimeSeries(
                title="Request Rate",
                dataSource="Prometheus",
                targets=[
                    Target(
                        expr='sum(rate(http_requests_total[5m])) by (status)',
                        refId="A",
                        legendFormat="{{status}}"
                    )
                ],
                gridPos=GridPos(h=8, w=12, x=0, y=10),
                unit="reqps",
            ),

            # Error Rate
            TimeSeries(
                title="Error Rate",
                dataSource="Prometheus",
                targets=[
                    Target(
                        expr='sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100',
                        refId="A",
                        legendFormat="Error Rate %"
                    )
                ],
                gridPos=GridPos(h=8, w=12, x=12, y=10),
                unit=PERCENT_UNIT,
                thresholds=[
                    Threshold(value=0, color="green"),
                    Threshold(value=0.5, color="yellow"),
                    Threshold(value=1, color="red"),
                ],
            ),
        ],
    ).auto_panel_ids()

    return dashboard


def main():
    """Generate and save dashboard."""
    dashboard = create_slo_dashboard()

    # Save to JSON
    output_file = "config/dashboards/ml-platform/slo-overview-generated.json"
    with open(output_file, 'w') as f:
        json.dump(dashboard.to_json_data(), f, cls=DashboardEncoder, indent=2)

    print(f"✅ Dashboard generated: {output_file}")


if __name__ == "__main__":
    main()
```

**Install dependencies**:
```bash
pip install grafanalib
```

**Generate dashboard**:
```bash
python scripts/generate-dashboards.py

# Dashboard appears in Grafana UI automatically
```

---

## Phase 6: Advanced Panel Types (30 minutes)

### Step 7: Create Infrastructure Dashboard

**Heatmap Panel** (Request Duration Distribution):
```json
{
  "type": "heatmap",
  "title": "Request Duration Heatmap",
  "targets": [
    {
      "expr": "sum(rate(http_request_duration_seconds_bucket[5m])) by (le)",
      "format": "heatmap",
      "legendFormat": "{{le}}"
    }
  ],
  "dataFormat": "tsbuckets"
}
```

**Table Panel** (Top 10 Slowest Endpoints):
```json
{
  "type": "table",
  "title": "Top 10 Slowest Endpoints",
  "targets": [
    {
      "expr": "topk(10, histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))) * 1000",
      "format": "table",
      "instant": true
    }
  ],
  "transformations": [
    {"id": "organize", "options": {"renameByName": {"endpoint": "Endpoint", "Value": "P99 Latency (ms)"}}}
  ]
}
```

---

## Phase 7: Variables and Templating (25 minutes)

### Step 8: Add Dashboard Variables

```json
{
  "templating": {
    "list": [
      {
        "name": "environment",
        "type": "query",
        "datasource": "Prometheus",
        "query": "label_values(http_requests_total, environment)",
        "multi": false,
        "includeAll": false,
        "current": {"text": "production", "value": "production"}
      },
      {
        "name": "service",
        "type": "query",
        "datasource": "Prometheus",
        "query": "label_values(http_requests_total{environment=\"$environment\"}, service)",
        "multi": true,
        "includeAll": true
      },
      {
        "name": "percentile",
        "type": "custom",
        "query": "0.50,0.95,0.99",
        "multi": false,
        "current": {"text": "0.99", "value": "0.99"}
      }
    ]
  }
}
```

**Use variables in queries**:
```promql
histogram_quantile($percentile, rate(http_request_duration_seconds_bucket{environment="$environment",service=~"$service"}[5m]))
```

---

## Phase 8: Alerts in Grafana (20 minutes)

### Step 9: Create Grafana Unified Alerts

```yaml
# config/grafana/provisioning/alerting/rules.yml
apiVersion: 1

groups:
  - name: SLO Alerts
    interval: 1m
    rules:
      - uid: slo-availability-violation
        title: SLO Availability Violation
        condition: A
        data:
          - refId: A
            datasourceUid: prometheus
            model:
              expr: slo:availability:ratio_rate30d < 0.995
              intervalMs: 1000
              maxDataPoints: 43200
        for: 5m
        annotations:
          summary: "Availability SLO violated"
          description: "Availability is {{ $values.A.Value }}%, below 99.5% target"
        labels:
          severity: critical
        noDataState: NoData
        execErrState: Error
```

---

## Phase 9: Testing and Validation (15 minutes)

### Step 10: Validate Dashboards

```bash
# 1. Check dashboard loads correctly
curl -s http://admin:admin@localhost:3000/api/dashboards/uid/slo-overview | jq '.dashboard.title'

# 2. Test queries return data
curl -s 'http://localhost:9090/api/v1/query?query=slo:availability:ratio_rate30d' | jq '.data.result'

# 3. Verify panel rendering
# Open dashboard in UI and check for:
# - No "No Data" messages
# - Thresholds coloring correctly
# - Variables working
# - Time range selector working
```

---

## Phase 10: Production Deployment (20 minutes)

### Step 11: Export All Dashboards

```bash
# Export script
#!/bin/bash
# export-dashboards.sh

GRAFANA_URL="http://admin:admin@localhost:3000"
OUTPUT_DIR="config/dashboards/backup"

mkdir -p $OUTPUT_DIR

# Get all dashboard UIDs
DASHBOARDS=$(curl -s "$GRAFANA_URL/api/search?type=dash-db" | jq -r '.[].uid')

for uid in $DASHBOARDS; do
  echo "Exporting $uid..."
  curl -s "$GRAFANA_URL/api/dashboards/uid/$uid" | jq '.dashboard' > "$OUTPUT_DIR/${uid}.json"
done

echo "✅ Exported $(echo "$DASHBOARDS" | wc -l) dashboards to $OUTPUT_DIR"
```

### Step 12: Version Control

```bash
# Initialize git repo
git init
git add config/dashboards/*.json
git commit -m "Add Grafana dashboards"

# Push to remote
git remote add origin git@github.com:company/grafana-dashboards.git
git push -u origin main
```

---

## Summary

**What You Built**:
- ✅ Grafana with auto-provisioned data sources
- ✅ SLO Overview dashboard with 6+ panels
- ✅ Dashboard-as-code with JSON and Python
- ✅ Template variables for dynamic filtering
- ✅ Grafana unified alerts
- ✅ Production deployment workflow

**Key Skills Learned**:
- Manual dashboard creation in Grafana UI
- JSON dashboard structure
- Programmatic dashboard generation (grafanalib)
- Data source provisioning
- Template variables for reusability
- Alert creation in Grafana

**Next Steps**:
- Exercise 04: Centralized logging with Loki/ELK
- Exercise 05: Alerting and incident response workflows
- Add more advanced visualizations (heatmaps, flame graphs)
- Integrate with other data sources (Tempo for traces)
