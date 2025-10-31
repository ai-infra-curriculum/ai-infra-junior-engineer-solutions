# Step-by-Step Implementation Guide: Prometheus Stack

## Overview

Deploy complete Prometheus monitoring stack! Learn Prometheus, Alertmanager, service discovery, recording rules, and production monitoring patterns.

**Time**: 2-3 hours | **Difficulty**: Intermediate

---

## Learning Objectives

✅ Deploy Prometheus on Kubernetes
✅ Configure service discovery
✅ Write PromQL queries
✅ Create recording and alerting rules
✅ Set up Alertmanager
✅ Monitor ML workloads
✅ Implement high availability

---

## Install Prometheus Stack

```bash
# Add Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
  --set alertmanager.enabled=true \
  --set grafana.enabled=true

# Access Prometheus
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
```

---

## Service Discovery

```yaml
# prometheus-config.yaml
scrape_configs:
  # Kubernetes pods
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
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

  # ML inference service
  - job_name: 'ml-inference'
    static_configs:
      - targets: ['ml-api:8080']
    metric_relabel_configs:
      - source_labels: [__name__]
        regex: 'http_.*|model_.*'
        action: keep
```

---

## PromQL Queries

```promql
# Request rate
sum(rate(http_requests_total[5m])) by (endpoint)

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) /
sum(rate(http_requests_total[5m]))

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# GPU utilization
avg(nvidia_gpu_duty_cycle) by (gpu_uuid)

# Model prediction rate
rate(model_predictions_total[5m])

# Memory usage
container_memory_usage_bytes{pod=~"ml-.*"} /
container_spec_memory_limit_bytes{pod=~"ml-.*"}
```

---

## Recording Rules

```yaml
# prometheus-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ml-recording-rules
  namespace: monitoring
spec:
  groups:
  - name: ml_metrics
    interval: 30s
    rules:
    - record: job:http_request_duration_seconds:p95
      expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

    - record: job:http_requests:rate5m
      expr: sum(rate(http_requests_total[5m])) by (job, endpoint)

    - record: job:http_errors:rate5m
      expr: sum(rate(http_requests_total{status=~"5.."}[5m])) by (job)
```

---

## Alerting Rules

```yaml
# alerts.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ml-alerts
  namespace: monitoring
spec:
  groups:
  - name: ml_alerts
    rules:
    - alert: HighErrorRate
      expr: |
        sum(rate(http_requests_total{status=~"5.."}[5m])) /
        sum(rate(http_requests_total[5m])) > 0.05
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value | humanizePercentage }}"

    - alert: HighLatency
      expr: |
        histogram_quantile(0.95,
          rate(http_request_duration_seconds_bucket[5m])
        ) > 1
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "High latency detected"
        description: "p95 latency is {{ $value }}s"

    - alert: ModelPredictionsDrop
      expr: |
        rate(model_predictions_total[5m]) <
        rate(model_predictions_total[1h] offset 1h) * 0.5
      for: 15m
      labels:
        severity: warning
      annotations:
        summary: "Model predictions dropped significantly"
```

---

## Alertmanager Configuration

```yaml
# alertmanager-config.yaml
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-config
  namespace: monitoring
stringData:
  alertmanager.yaml: |
    global:
      resolve_timeout: 5m
      slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'

    route:
      group_by: ['alertname', 'cluster']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 12h
      receiver: 'slack'
      routes:
      - match:
          severity: critical
        receiver: 'pagerduty'
        continue: true

    receivers:
    - name: 'slack'
      slack_configs:
      - channel: '#ml-alerts'
        title: 'Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

    - name: 'pagerduty'
      pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
```

---

## Best Practices

✅ Use service discovery
✅ Implement recording rules for complex queries
✅ Set appropriate alert thresholds
✅ Use AlertManager for deduplication
✅ Monitor Prometheus itself
✅ Implement retention policies
✅ Use remote storage for long-term data
✅ Tag metrics consistently

---

**Prometheus Stack mastered!** 📊
