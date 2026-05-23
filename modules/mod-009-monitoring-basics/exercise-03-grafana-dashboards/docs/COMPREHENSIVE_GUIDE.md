# Comprehensive Guide: Grafana Dashboards

## Architecture

### Dashboard Architecture

```
┌──────────────────────────────────────────────────┐
│          Grafana Dashboard Layer                  │
│                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ SLO         │  │ Application │  │Infrastructure│
│  │ Overview    │  │ Performance │  │  Health   │ │
│  └──────┬──────┘  └──────┬──────┘  └─────┬────┘ │
└─────────┼─────────────────┼────────────────┼──────┘
          │                 │                │
          └─────────────────┴────────────────┘
                            │
                 ┌──────────┴──────────┐
                 │   Data Sources      │
                 │                     │
                 │  - Prometheus       │
                 │  - Loki (Logs)      │
                 │  - Jaeger (Traces)  │
                 └─────────────────────┘
```

### Dashboard Provisioning

**Declarative Configuration**:
- Data sources auto-configured via YAML
- Dashboards loaded from JSON files on startup
- Version-controlled dashboard definitions
- Automatic updates on file changes

---

## Deployment

### Development Deployment

```bash
# Start with Docker Compose
docker-compose up -d

# Access at http://localhost:3000
# Default credentials: admin/admin
```

### Production Deployment (Kubernetes)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:10.2.2
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: grafana-admin
              key: password
        volumeMounts:
        - name: provisioning
          mountPath: /etc/grafana/provisioning
        - name: dashboards
          mountPath: /var/lib/grafana/dashboards
      volumes:
      - name: provisioning
        configMap:
          name: grafana-provisioning
      - name: dashboards
        configMap:
          name: grafana-dashboards
```

**Cost**: ~$50-100/month (AWS)

---

## Troubleshooting

### Issue: Dashboard Shows "No Data"

**Solutions**:
1. Check data source connection: Configuration → Data Sources → Test
2. Verify query returns data in Prometheus UI
3. Check time range matches data availability
4. Verify metric name is correct (`label_values(__name__)`)

### Issue: Provisioned Dashboards Don't Appear

**Solutions**:
1. Check provisioning config: `/etc/grafana/provisioning/dashboards/`
2. Verify JSON files are valid: `jq . dashboard.json`
3. Check Grafana logs: `docker logs grafana | grep provisioning`
4. Ensure dashboard UIDs are unique

### Issue: Slow Dashboard Loading

**Solutions**:
1. Reduce query time range
2. Use recording rules for expensive queries
3. Limit number of panels (< 20 per dashboard)
4. Use query caching: `min_interval` in panel

### Issue: Variables Not Working

**Solutions**:
1. Check variable query syntax: `label_values(metric, label)`
2. Verify data source is selected
3. Test variable in Explore view first
4. Check variable dependencies (order matters)

---

## Best Practices

### Dashboard Design

**Organization**:
- Group related panels with Row panels
- Use consistent color schemes
- Limit dashboards to 10-15 panels
- Create separate dashboards for different audiences (dev, ops, exec)

**Panel Configuration**:
- Always set panel titles
- Add descriptions for complex queries
- Use appropriate visualization types
- Set meaningful thresholds
- Configure units (ms, %, GB)

**Variables**:
- Use `$variable` syntax in queries
- Chain variables (environment → service → endpoint)
- Provide sensible defaults
- Use `All` option when appropriate

### Performance

**Query Optimization**:
```promql
# BAD: Returns all label combinations
http_requests_total

# GOOD: Aggregate early
sum(rate(http_requests_total[5m])) by (status)

# BETTER: Use recording rule
http:requests:rate5m  # Pre-computed
```

**Caching**:
- Set `min_interval` to match recording rule interval
- Use `$__interval` for dynamic step size
- Enable query result caching in Grafana config

---

## Summary

**Key Features**:
- ✅ Auto-provisioned data sources and dashboards
- ✅ Dashboard-as-code (JSON version control)
- ✅ Programmatic generation with grafanalib
- ✅ Template variables for dynamic filtering
- ✅ Multi-data source integration
- ✅ Grafana unified alerting

**Production Readiness**:
- High availability (2+ replicas)
- Persistent storage for dashboards
- Authentication and authorization
- Automated backups
- Version-controlled configurations
