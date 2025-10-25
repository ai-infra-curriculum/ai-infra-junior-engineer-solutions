#!/usr/bin/env python3
"""
Grafana Dashboard Generator for ML Infrastructure

Generates dashboard JSON files programmatically for better maintainability.
Dashboards include SLO tracking, infrastructure monitoring, and executive overviews.
"""

import json
import os
from pathlib import Path


class DashboardBuilder:
    """Builder for creating Grafana dashboard JSON"""

    def __init__(self, title, uid, tags=None):
        self.dashboard = {
            "uid": uid,
            "title": title,
            "tags": tags or [],
            "timezone": "browser",
            "schemaVersion": 39,
            "version": 1,
            "refresh": "30s",
            "time": {
                "from": "now-6h",
                "to": "now"
            },
            "timepicker": {
                "refresh_intervals": ["10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"]
            },
            "panels": [],
            "templating": {
                "list": []
            },
            "annotations": {
                "list": [
                    {
                        "builtIn": 1,
                        "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                        "enable": True,
                        "hide": True,
                        "iconColor": "rgba(0, 211, 255, 1)",
                        "name": "Annotations & Alerts",
                        "type": "dashboard"
                    }
                ]
            },
            "editable": True,
            "fiscalYearStartMonth": 0,
            "graphTooltip": 1,
            "links": [],
            "liveNow": False,
            "style": "dark"
        }
        self.panel_id = 1
        self.current_y = 0

    def add_variable(self, name, query, label=None, datasource="Prometheus", all_value=True):
        """Add a template variable"""
        variable = {
            "name": name,
            "type": "query",
            "label": label or name.replace("_", " ").title(),
            "datasource": {"type": "prometheus", "uid": "prometheus"},
            "query": query,
            "refresh": 1,
            "regex": "",
            "allValue": ".*" if all_value else None,
            "includeAll": all_value,
            "multi": False,
            "current": {},
            "options": [],
            "sort": 1
        }
        self.dashboard["templating"]["list"].append(variable)

    def add_stat_panel(self, title, query, unit="short", thresholds=None, x=0, y=None, w=6, h=4):
        """Add a stat panel (single value)"""
        if y is None:
            y = self.current_y

        panel = {
            "id": self.panel_id,
            "title": title,
            "type": "stat",
            "datasource": {"type": "prometheus", "uid": "prometheus"},
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{
                "expr": query,
                "refId": "A",
                "instant": True
            }],
            "options": {
                "graphMode": "area",
                "colorMode": "value",
                "justifyMode": "auto",
                "textMode": "auto",
                "orientation": "auto",
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"]
                }
            },
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "decimals": 2,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": thresholds or [
                            {"value": None, "color": "green"},
                            {"value": 80, "color": "red"}
                        ]
                    }
                }
            }
        }
        self.dashboard["panels"].append(panel)
        self.panel_id += 1
        return panel

    def add_gauge_panel(self, title, query, unit="percent", min_val=0, max_val=100, thresholds=None, x=0, y=None, w=6, h=4):
        """Add a gauge panel"""
        if y is None:
            y = self.current_y

        panel = {
            "id": self.panel_id,
            "title": title,
            "type": "gauge",
            "datasource": {"type": "prometheus", "uid": "prometheus"},
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{
                "expr": query,
                "refId": "A",
                "instant": True
            }],
            "options": {
                "showThresholdLabels": False,
                "showThresholdMarkers": True,
                "orientation": "auto"
            },
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "min": min_val,
                    "max": max_val,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": thresholds or [
                            {"value": None, "color": "green"},
                            {"value": 95, "color": "yellow"},
                            {"value": 99, "color": "red"}
                        ]
                    }
                }
            }
        }
        self.dashboard["panels"].append(panel)
        self.panel_id += 1
        return panel

    def add_time_series_panel(self, title, targets, unit="short", legend_display=True, x=0, y=None, w=12, h=8):
        """Add a time series graph panel"""
        if y is None:
            y = self.current_y

        formatted_targets = []
        for i, target in enumerate(targets):
            if isinstance(target, str):
                formatted_targets.append({
                    "expr": target,
                    "refId": chr(65 + i),  # A, B, C, ...
                    "legendFormat": f"Query {chr(65 + i)}"
                })
            else:
                formatted_targets.append(target)

        panel = {
            "id": self.panel_id,
            "title": title,
            "type": "timeseries",
            "datasource": {"type": "prometheus", "uid": "prometheus"},
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": formatted_targets,
            "options": {
                "legend": {
                    "calcs": ["mean", "lastNotNull", "max"],
                    "displayMode": "table" if legend_display else "hidden",
                    "placement": "bottom",
                    "showLegend": legend_display
                },
                "tooltip": {
                    "mode": "multi",
                    "sort": "desc"
                }
            },
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "smooth",
                        "barAlignment": 0,
                        "lineWidth": 1,
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "showPoints": "never",
                        "pointSize": 5,
                        "stacking": {
                            "mode": "none",
                            "group": "A"
                        },
                        "axisPlacement": "auto",
                        "axisLabel": "",
                        "scaleDistribution": {
                            "type": "linear"
                        }
                    },
                    "unit": unit,
                    "decimals": 2
                }
            }
        }
        self.dashboard["panels"].append(panel)
        self.panel_id += 1
        return panel

    def add_row_panel(self, title, y=None, collapsed=False):
        """Add a collapsible row"""
        if y is None:
            y = self.current_y

        panel = {
            "id": self.panel_id,
            "title": title,
            "type": "row",
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
            "collapsed": collapsed,
            "panels": []
        }
        self.dashboard["panels"].append(panel)
        self.panel_id += 1
        self.current_y = y + 1
        return panel

    def to_json(self):
        """Export dashboard as JSON"""
        return json.dumps({"dashboard": self.dashboard, "overwrite": True}, indent=2)


def create_slo_overview_dashboard():
    """Create SLO Overview Dashboard"""
    db = DashboardBuilder(
        title="SLO Overview - ML Inference Platform",
        uid="slo-overview",
        tags=["SLO", "ML Platform", "Production"]
    )

    # Row: SLO Status
    db.add_row_panel("SLO Compliance Status", y=0)

    # Availability SLO
    db.add_gauge_panel(
        title="Availability SLO (30d)",
        query="slo:availability:ratio_rate30d",
        unit="percent",
        min_val=99,
        max_val=100,
        thresholds=[
            {"value": None, "color": "red"},
            {"value": 99.5, "color": "green"}
        ],
        x=0, y=1, w=6, h=6
    )

    # Latency P99 SLO
    db.add_gauge_panel(
        title="Latency P99 (7d) - Target < 300ms",
        query="slo:http_request_duration:p99:rate7d",
        unit="ms",
        min_val=0,
        max_val=500,
        thresholds=[
            {"value": None, "color": "green"},
            {"value": 300, "color": "yellow"},
            {"value": 400, "color": "red"}
        ],
        x=6, y=1, w=6, h=6
    )

    # Error Budget Remaining
    db.add_stat_panel(
        title="Error Budget Remaining",
        query="slo:availability:error_budget_remaining * 100",
        unit="percent",
        thresholds=[
            {"value": None, "color": "red"},
            {"value": 10, "color": "yellow"},
            {"value": 50, "color": "green"}
        ],
        x=12, y=1, w=6, h=6
    )

    # Burn Rate (1h)
    db.add_stat_panel(
        title="Burn Rate (1h)",
        query="slo:availability:burn_rate:1h",
        unit="short",
        thresholds=[
            {"value": None, "color": "green"},
            {"value": 6, "color": "yellow"},
            {"value": 14, "color": "red"}
        ],
        x=18, y=1, w=6, h=6
    )

    # Row: Availability Trends
    db.add_row_panel("Availability & Error Budget Trends", y=7)

    # Availability over time
    db.add_time_series_panel(
        title="Availability Percentage (5m windows)",
        targets=[{
            "expr": "slo:availability:ratio_rate5m",
            "refId": "A",
            "legendFormat": "Availability %"
        }],
        unit="percent",
        x=0, y=8, w=12, h=8
    )

    # Error budget consumption
    db.add_time_series_panel(
        title="Error Budget Consumption Rate",
        targets=[
            {"expr": "slo:availability:burn_rate:1h", "refId": "A", "legendFormat": "1h burn rate"},
            {"expr": "slo:availability:burn_rate:6h", "refId": "B", "legendFormat": "6h burn rate"},
            {"expr": "slo:availability:burn_rate:3d", "refId": "C", "legendFormat": "3d burn rate"}
        ],
        unit="short",
        x=12, y=8, w=12, h=8
    )

    # Row: Latency Metrics
    db.add_row_panel("Latency Performance", y=16)

    # Latency percentiles
    db.add_time_series_panel(
        title="Request Latency Percentiles",
        targets=[
            {"expr": "slo:http_request_duration:p50:rate5m", "refId": "A", "legendFormat": "P50"},
            {"expr": "slo:http_request_duration:p95:rate5m", "refId": "B", "legendFormat": "P95"},
            {"expr": "slo:http_request_duration:p99:rate5m", "refId": "C", "legendFormat": "P99"}
        ],
        unit="ms",
        x=0, y=17, w=24, h=8
    )

    return db.to_json()


def create_application_performance_dashboard():
    """Create Application Performance Dashboard"""
    db = DashboardBuilder(
        title="Application Performance - Inference Gateway",
        uid="app-performance",
        tags=["Application", "ML Platform", "Performance"]
    )

    # Row: Request Metrics
    db.add_row_panel("Request Metrics", y=0)

    # Request rate
    db.add_stat_panel(
        title="Request Rate (QPS)",
        query="sum(rate(http_requests_total{service=\"inference-gateway\"}[5m]))",
        unit="reqps",
        x=0, y=1, w=6, h=4
    )

    # Error rate
    db.add_stat_panel(
        title="Error Rate",
        query='sum(rate(http_requests_total{service="inference-gateway",status=~"5.."}[5m])) / sum(rate(http_requests_total{service="inference-gateway"}[5m])) * 100',
        unit="percent",
        thresholds=[
            {"value": None, "color": "green"},
            {"value": 1, "color": "yellow"},
            {"value": 5, "color": "red"}
        ],
        x=6, y=1, w=6, h=4
    )

    # Avg latency
    db.add_stat_panel(
        title="Avg Response Time",
        query='sum(rate(http_request_duration_seconds_sum{service="inference-gateway"}[5m])) / sum(rate(http_request_duration_seconds_count{service="inference-gateway"}[5m])) * 1000',
        unit="ms",
        x=12, y=1, w=6, h=4
    )

    # Active requests
    db.add_stat_panel(
        title="Active Requests",
        query='http_requests_in_progress{service="inference-gateway"}',
        unit="short",
        x=18, y=1, w=6, h=4
    )

    # Request rate by endpoint
    db.add_time_series_panel(
        title="Request Rate by Endpoint",
        targets=[{
            "expr": 'sum(rate(http_requests_total{service="inference-gateway"}[5m])) by (endpoint)',
            "refId": "A",
            "legendFormat": "{{endpoint}}"
        }],
        unit="reqps",
        x=0, y=5, w=12, h=8
    )

    # Error rate over time
    db.add_time_series_panel(
        title="Error Rate Over Time",
        targets=[{
            "expr": 'sum(rate(http_requests_total{service="inference-gateway",status=~"5.."}[5m])) by (status)',
            "refId": "A",
            "legendFormat": "{{status}}"
        }],
        unit="reqps",
        x=12, y=5, w=12, h=8
    )

    return db.to_json()


def create_infrastructure_dashboard():
    """Create Infrastructure Health Dashboard"""
    db = DashboardBuilder(
        title="Infrastructure Health",
        uid="infrastructure-health",
        tags=["Infrastructure", "System", "Resources"]
    )

    # Row: CPU & Memory
    db.add_row_panel("CPU & Memory", y=0)

    # CPU usage by container
    db.add_time_series_panel(
        title="CPU Usage by Container",
        targets=[{
            "expr": "container:cpu_usage:percent",
            "refId": "A",
            "legendFormat": "{{container_label_com_docker_compose_service}}"
        }],
        unit="percent",
        x=0, y=1, w=12, h=8
    )

    # Memory usage by container
    db.add_time_series_panel(
        title="Memory Usage by Container",
        targets=[{
            "expr": "container:memory_usage:percent",
            "refId": "A",
            "legendFormat": "{{container_label_com_docker_compose_service}}"
        }],
        unit="percent",
        x=12, y=1, w=12, h=8
    )

    # Row: Network
    db.add_row_panel("Network", y=9)

    # Network receive
    db.add_time_series_panel(
        title="Network Receive Rate",
        targets=[{
            "expr": "container:network_receive:rate5m_mb",
            "refId": "A",
            "legendFormat": "{{container_label_com_docker_compose_service}}"
        }],
        unit="MBs",
        x=0, y=10, w=12, h=8
    )

    # Network transmit
    db.add_time_series_panel(
        title="Network Transmit Rate",
        targets=[{
            "expr": "container:network_transmit:rate5m_mb",
            "refId": "A",
            "legendFormat": "{{container_label_com_docker_compose_service}}"
        }],
        unit="MBs",
        x=12, y=10, w=12, h=8
    )

    return db.to_json()


def main():
    """Generate all dashboards"""
    output_dir = Path(__file__).parent.parent / "config" / "dashboards"

    dashboards = {
        "ml-platform/slo-overview.json": create_slo_overview_dashboard(),
        "ml-platform/application-performance.json": create_application_performance_dashboard(),
        "infrastructure/infrastructure-health.json": create_infrastructure_dashboard(),
    }

    for path, content in dashboards.items():
        file_path = output_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        print(f"✓ Created: {file_path}")

    print(f"\n✅ Generated {len(dashboards)} dashboards successfully!")


if __name__ == "__main__":
    main()
