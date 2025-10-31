# Step-by-Step Implementation Guide: Grafana Dashboards

## Overview

Create production dashboards with Grafana! Learn visualization, dashboard design, ML-specific metrics, alerts, and dashboard-as-code.

**Time**: 2 hours | **Difficulty**: Beginner to Intermediate

---

## Learning Objectives

✅ Install and configure Grafana
✅ Create dashboards
✅ Visualize ML metrics
✅ Set up alerts
✅ Import/export dashboards
✅ Use dashboard variables
✅ Implement dashboard-as-code

---

## Access Grafana

```bash
# Port forward
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Default credentials
# Username: admin
# Password: (get from secret)
kubectl get secret -n monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 -d
```

---

## ML Dashboard JSON

```json
{
  "dashboard": {
    "title": "ML Inference Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [{
          "expr": "sum(rate(http_requests_total[5m])) by (endpoint)"
        }]
      },
      {
        "title": "Latency (p95)",
        "type": "graph",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [{
          "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))"
        }],
        "format": "percentunit"
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [{
          "expr": "avg(nvidia_gpu_duty_cycle) by (gpu_uuid)"
        }]
      }
    ]
  }
}
```

---

## Dashboard Variables

```json
{
  "templating": {
    "list": [
      {
        "name": "namespace",
        "type": "query",
        "query": "label_values(kube_pod_info, namespace)",
        "refresh": 1
      },
      {
        "name": "pod",
        "type": "query",
        "query": "label_values(kube_pod_info{namespace=\"$namespace\"}, pod)",
        "refresh": 1
      }
    ]
  }
}
```

---

## Dashboard as Code

```yaml
# grafana-dashboard.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  ml-metrics.json: |
    {
      "title": "ML Metrics",
      "panels": [...]
    }
```

---

## Best Practices

✅ Group related metrics
✅ Use meaningful panel titles
✅ Add descriptions to panels
✅ Use template variables
✅ Set appropriate time ranges
✅ Configure alerts
✅ Version control dashboards
✅ Use dashboard folders

---

**Grafana Dashboards mastered!** 📈
