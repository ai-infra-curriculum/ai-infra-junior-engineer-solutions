# Implementation Guide: PromQL and Recording Rules

Step-by-step guide to implementing production-grade Prometheus monitoring for ML infrastructure.

**Estimated Time:** 2-3 hours
**Difficulty:** Intermediate-Advanced

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Phase 1: Deploy Recording Rules](#phase-1-deploy-recording-rules-30-minutes)
3. [Phase 2: Deploy Alert Rules](#phase-2-deploy-alert-rules-30-minutes)
4. [Phase 3: Configure AlertManager](#phase-3-configure-alertmanager-20-minutes)
5. [Phase 4: Create Grafana Dashboards](#phase-4-create-grafana-dashboards-40-minutes)
6. [Phase 5: Performance Validation](#phase-5-performance-validation-20-minutes)
7. [Phase 6: Production Hardening](#phase-6-production-hardening-30-minutes)

---

## Prerequisites

### Required Knowledge

- ✅ Basic Kubernetes concepts (Pods, ConfigMaps, Services)
- ✅ Prometheus fundamentals (metrics, queries)
- ✅ Basic PromQL syntax
- ✅ Grafana dashboard creation

### Required Tools

```bash
# Verify installations
kubectl version --client
helm version
promtool --version # From Prometheus release
curl --version
```

### Existing Infrastructure

You should have:
- ✅ Kubernetes cluster (1.20+)
- ✅ Prometheus installed (2.30+)
- ✅ Grafana installed (8.0+)
- ✅ AlertManager installed (optional for Phase 3)
- ✅ ML platform exporting metrics

### Verify Prometheus

```bash
# Port-forward Prometheus
kubectl port-forward -n monitoring svc/prometheus-server 9090:80

# Check Prometheus is reachable
curl http://localhost:9090/-/healthy

# Expected: Prometheus is Healthy.
```

### Verify Metrics Exist

```bash
# Check for ML metrics
curl -s 'http://localhost:9090/api/v1/label/__name__/values' | \
    grep model_predictions_total

# If empty, you need to instrument your ML platform first
# See: https://prometheus.io/docs/instrumenting/clientlibs/
```

---

## Phase 1: Deploy Recording Rules (30 minutes)

### Step 1.1: Validate Recording Rules Syntax

Before deploying, validate the rules file:

```bash
# Navigate to exercise directory
cd modules/mod-009-monitoring-basics/exercise-07

# Validate syntax
promtool check rules kubernetes/recording-rules.yaml

# Expected output:
# SUCCESS: 6 groups, 40+ rules
```

**If validation fails:**
- Check YAML indentation (use spaces, not tabs)
- Verify metric names match your instrumentation
- Ensure all PromQL expressions are valid

### Step 1.2: Create ConfigMap

```bash
# Create the ConfigMap in monitoring namespace
kubectl create configmap prometheus-recording-rules \
  --from-file=recording_rules.yml=kubernetes/recording-rules.yaml \
  -n monitoring \
  --dry-run=client -o yaml | kubectl apply -f -

# Verify ConfigMap created
kubectl get configmap prometheus-recording-rules -n monitoring

# View contents
kubectl get configmap prometheus-recording-rules -n monitoring -o yaml
```

### Step 1.3: Mount ConfigMap in Prometheus

Edit your Prometheus deployment to mount the ConfigMap:

```bash
# Edit Prometheus deployment
kubectl edit deployment prometheus-server -n monitoring
```

Add volume and volume mount:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus-server
spec:
  template:
    spec:
      containers:
      - name: prometheus
        volumeMounts:
        # Add this:
        - name: recording-rules
          mountPath: /etc/config/recording_rules.yml
          subPath: recording_rules.yml
      volumes:
      # Add this:
      - name: recording-rules
        configMap:
          name: prometheus-recording-rules
```

**Alternative:** If using Prometheus Operator:

```bash
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ml-platform-recording-rules
  namespace: monitoring
spec:
  groups:
$(cat kubernetes/recording-rules.yaml | yq '.data.recording_rules.yml' | sed 's/^/    /')
EOF
```

### Step 1.4: Update Prometheus Configuration

Edit Prometheus configuration to load the rules:

```bash
# Edit Prometheus ConfigMap
kubectl edit configmap prometheus-server -n monitoring
```

Add to `prometheus.yml`:

```yaml
global:
  evaluation_interval: 30s  # Match recording rule intervals

rule_files:
  - /etc/config/recording_rules.yml  # Add this line
```

### Step 1.5: Reload Prometheus

```bash
# Restart Prometheus to load new rules
kubectl rollout restart deployment prometheus-server -n monitoring

# Wait for rollout
kubectl rollout status deployment prometheus-server -n monitoring

# Check logs for errors
kubectl logs -n monitoring deployment/prometheus-server --tail=50 | grep -i "rule\|error"
```

**Expected log output:**
```
level=info msg="Loading configuration file" filename=/etc/config/prometheus.yml
level=info msg="Completed loading of configuration file" filename=/etc/config/prometheus.yml
level=info msg="Rule evaluation started" group=ml_model_serving_rules
```

### Step 1.6: Verify Recording Rules

```bash
# Port-forward if not already done
kubectl port-forward -n monitoring svc/prometheus-server 9090:80

# Check rules are loaded
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[].name'

# Should see:
# "ml_model_serving_rules"
# "ml_platform_aggregations"
# "ml_resource_efficiency_rules"
# "ml_business_rules"
# "ml_trend_analysis_rules"
# "ml_slo_tracking_rules"
```

### Step 1.7: Verify Recording Rules Are Evaluating

Wait 60 seconds for recording rules to start populating data, then:

```bash
# Query a recording rule
curl -s 'http://localhost:9090/api/v1/query?query=model:predictions:rate5m' | \
    jq '.data.result | length'

# Should return > 0 (number of models)
```

**If 0 results:**
1. Check base metrics exist: `model_predictions_total`
2. Wait another 30-60 seconds
3. Check Prometheus logs for errors
4. Verify recording rule interval matches your data

### Step 1.8: Test Query Performance

```bash
# Make benchmark scripts executable
chmod +x scripts/benchmark_queries.sh

# Run benchmarks
./scripts/benchmark_queries.sh

# Review results
cat benchmark_results.txt
```

**Expected improvement:** 5-15x faster queries

---

## Phase 2: Deploy Alert Rules (30 minutes)

### Step 2.1: Validate Alert Rules

```bash
# Validate alert rules syntax
promtool check rules kubernetes/alert-rules.yaml

# Expected: SUCCESS with 0 errors
```

**Common errors:**
- Undefined recording rules (deploy recording rules first!)
- Invalid PromQL expressions
- Missing required labels

### Step 2.2: Customize Alert Thresholds

Before deploying, review and adjust thresholds for your environment:

```bash
# Edit alert rules
vi kubernetes/alert-rules.yaml
```

**Key thresholds to review:**

| Alert | Default | Adjust If... |
|-------|---------|--------------|
| ModelHighErrorRate | >5% | Your baseline error rate is different |
| ModelHighLatency | >500ms | Your SLO is different |
| ModelHighCPUUsage | >85% | You prefer different headroom |
| SLOErrorBudgetBurn | 99.9% | Your SLO target is different |

### Step 2.3: Create Alert Rules ConfigMap

```bash
# Create ConfigMap
kubectl create configmap prometheus-alert-rules \
  --from-file=alert_rules.yml=kubernetes/alert-rules.yaml \
  -n monitoring \
  --dry-run=client -o yaml | kubectl apply -f -

# Verify
kubectl get configmap prometheus-alert-rules -n monitoring
```

### Step 2.4: Mount Alert Rules ConfigMap

```bash
# Edit Prometheus deployment
kubectl edit deployment prometheus-server -n monitoring
```

Add volume and mount:

```yaml
spec:
  template:
    spec:
      containers:
      - name: prometheus
        volumeMounts:
        - name: alert-rules
          mountPath: /etc/config/alert_rules.yml
          subPath: alert_rules.yml
      volumes:
      - name: alert-rules
        configMap:
          name: prometheus-alert-rules
```

### Step 2.5: Update Prometheus Config for Alerts

```bash
kubectl edit configmap prometheus-server -n monitoring
```

Add to `prometheus.yml`:

```yaml
rule_files:
  - /etc/config/recording_rules.yml
  - /etc/config/alert_rules.yml  # Add this line

# Configure AlertManager endpoint
alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093  # Adjust if your service name differs
```

### Step 2.6: Reload and Verify

```bash
# Restart Prometheus
kubectl rollout restart deployment prometheus-server -n monitoring

# Wait for rollout
kubectl rollout status deployment prometheus-server -n monitoring

# Verify alerts loaded
curl -s http://localhost:9090/api/v1/rules | \
    jq '.data.groups[] | select(.name | contains("alert")) | .name'

# Should see:
# "ml_model_alerts_critical"
# "ml_model_alerts_warning"
# "ml_resource_alerts"
# "ml_platform_alerts"
# "ml_slo_alerts"
# "ml_business_alerts"
```

### Step 2.7: Check Alert Status

```bash
# View currently firing alerts
curl -s http://localhost:9090/api/v1/alerts | \
    jq '.data.alerts[] | {alert: .labels.alertname, state: .state}'

# Or via Prometheus UI:
# http://localhost:9090/alerts
```

**Expected:** Some alerts may be firing initially (normal)

### Step 2.8: Test Alert Firing (Optional)

Simulate a condition to test alerts:

```bash
# Simulate high error rate by stopping a model pod
kubectl delete pod -l app=ml-model-server --force --grace-period=0

# Wait 5 minutes, then check alerts
curl -s http://localhost:9090/api/v1/alerts | \
    jq '.data.alerts[] | select(.labels.alertname == "ModelNoPredictions")'

# Restore the pod
kubectl scale deployment ml-model-server --replicas=1
```

---

## Phase 3: Configure AlertManager (20 minutes)

### Step 3.1: Review AlertManager Config Template

```bash
# Review the config
cat kubernetes/alertmanager-config.yaml
```

### Step 3.2: Add Notification Receivers

Edit `kubernetes/alertmanager-config.yaml` with your endpoints:

```yaml
receivers:
- name: 'pagerduty-critical'
  pagerduty_configs:
  - service_key: 'YOUR_PAGERDUTY_KEY'  # Replace!
    description: '{{ .CommonAnnotations.summary }}'

- name: 'slack-warnings'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK'  # Replace!
    channel: '#ml-platform-alerts'
    title: '{{ .GroupLabels.alertname }}'
    text: '{{ .CommonAnnotations.description }}'
```

### Step 3.3: Deploy AlertManager Config

```bash
# Create/update ConfigMap
kubectl create configmap alertmanager-config \
  --from-file=alertmanager.yml=kubernetes/alertmanager-config.yaml \
  -n monitoring \
  --dry-run=client -o yaml | kubectl apply -f -

# Reload AlertManager
kubectl exec -n monitoring deployment/alertmanager -- \
    wget --post-data="" http://localhost:9093/-/reload
```

### Step 3.4: Verify AlertManager

```bash
# Port-forward AlertManager
kubectl port-forward -n monitoring svc/alertmanager 9093:9093

# Check status
curl http://localhost:9093/api/v1/status

# View alert groups
curl http://localhost:9093/api/v1/alerts
```

### Step 3.5: Test Notifications

```bash
# Send test alert
curl -X POST http://localhost:9093/api/v1/alerts -d '[{
  "labels": {
    "alertname": "TestAlert",
    "severity": "warning"
  },
  "annotations": {
    "summary": "Test notification from AlertManager"
  }
}]'

# Check your Slack/PagerDuty for the test alert
```

---

## Phase 4: Create Grafana Dashboards (40 minutes)

### Step 4.1: Import Dashboard Template

```bash
# Port-forward Grafana
kubectl port-forward -n monitoring svc/grafana 3000:80

# Open browser to http://localhost:3000
```

### Step 4.2: Create Model Health Dashboard

**Create New Dashboard → Add Panel:**

**Panel 1: Request Rate**
- Query: `model:predictions:rate5m`
- Visualization: Time series
- Unit: req/s
- Legend: `{{model}}`

**Panel 2: Error Rate**
- Query: `model:error_ratio:rate5m * 100`
- Visualization: Time series
- Unit: percent
- Legend: `{{model}}`
- Threshold: 5% (warning), 20% (critical)

**Panel 3: P95 Latency**
- Query: `model:latency:p95 * 1000`
- Visualization: Time series
- Unit: ms
- Legend: `{{model}}`
- Threshold: 500ms (warning), 2000ms (critical)

**Panel 4: Availability Gauge**
- Query: `model:availability:rate5m * 100`
- Visualization: Gauge
- Unit: percent
- Threshold: <99% (warning), <99.9% (critical)

**Panel 5: Top Models by Traffic**
- Query: `topk(10, model:predictions:rate5m)`
- Visualization: Bar chart
- Unit: req/s

### Step 4.3: Create Platform Overview Dashboard

**Panel 1: Total Throughput**
- Query: `platform:predictions:rate5m`
- Visualization: Stat
- Unit: req/s

**Panel 2: Platform Error Rate**
- Query: `platform:error_ratio:rate5m * 100`
- Visualization: Stat
- Unit: percent

**Panel 3: Platform P95 Latency**
- Query: `platform:latency:p95 * 1000`
- Visualization: Stat
- Unit: ms

**Panel 4: Active Models**
- Query: `count(model:predictions:rate5m > 0)`
- Visualization: Stat

**Panel 5: Daily Predictions**
- Query: `sum(increase(model_predictions_total[24h]))`
- Visualization: Stat

### Step 4.4: Create SLO Dashboard

**Panel 1: Error Budget Remaining**
- Query: `model:error_budget:remaining_30d * 100`
- Visualization: Gauge
- Unit: percent
- Thresholds: <10% (critical), <25% (warning)

**Panel 2: Burn Rate**
- Query: `model:error_budget:burn_rate_1h`
- Visualization: Time series
- Threshold: 1.0 (unsustainable)

**Panel 3: Availability (30 days)**
- Query: `avg_over_time(model:availability:rate5m[30d]) * 100`
- Visualization: Stat
- Unit: percent

**Panel 4: Latency SLO Compliance**
- Query: `count(model:latency:p95 < 0.2) / count(model:latency:p95) * 100`
- Visualization: Gauge
- Unit: percent

### Step 4.5: Configure Dashboard Variables

Add template variables for filtering:

```
Variable: model
Type: Query
Query: label_values(model:predictions:rate5m, model)
```

Update queries to use variable:
```promql
model:predictions:rate5m{model="$model"}
```

### Step 4.6: Set Dashboard Refresh Rate

Dashboard Settings → Time options:
- Refresh: 30s
- Time range: Last 6 hours
- Timezone: Browser

### Step 4.7: Configure Alerts in Grafana

For each panel showing critical metrics:
1. Click "Alert" tab
2. Create alert rule
3. Condition: `WHEN last() OF query() IS ABOVE threshold`
4. Contact point: Link to AlertManager

---

## Phase 5: Performance Validation (20 minutes)

### Step 5.1: Run Performance Benchmarks

```bash
# Run benchmark suite
./scripts/benchmark_queries.sh

# Review results
cat benchmark_results.txt
```

**Expected Results:**
- Request rate: 10x faster
- Histogram queries: 15x faster
- Complex dashboard: 11x faster

### Step 5.2: Check Cardinality

```bash
# Run cardinality check
chmod +x scripts/check_cardinality.sh
./scripts/check_cardinality.sh

# Review report
cat cardinality_report.txt
```

**Target Metrics:**
- Total series: <100k (good), <500k (acceptable)
- Recording rule series: ~1,500
- Storage increase: ~10%

### Step 5.3: Validate Dashboard Load Times

1. Open Grafana dashboard
2. Open browser DevTools (F12)
3. Go to Network tab
4. Refresh dashboard
5. Check timing:
   - Query time: <100ms per query
   - Dashboard load: <1s total

### Step 5.4: Monitor Prometheus Performance

```bash
# Check Prometheus metrics
curl -s http://localhost:9090/metrics | grep prometheus_rule

# Key metrics:
# - prometheus_rule_evaluation_duration_seconds (should be <1s)
# - prometheus_rule_group_iterations_missed_total (should be 0)
# - prometheus_tsdb_symbol_table_size_bytes (storage usage)
```

---

## Phase 6: Production Hardening (30 minutes)

### Step 6.1: Add Recording Rule Alerts

Create alerts for recording rule failures:

```yaml
- alert: RecordingRuleFailing
  expr: up{job="prometheus"} == 1 and absent(model:predictions:rate5m)
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Recording rule model:predictions:rate5m not evaluating"
```

### Step 6.2: Enable Recording Rule Metrics

Add to Prometheus config:

```yaml
global:
  evaluation_interval: 30s

# Enable detailed rule metrics
prometheus:
  metrics:
    rules:
      enabled: true
```

### Step 6.3: Configure Retention

```yaml
# In Prometheus config
storage:
  tsdb:
    retention.time: 30d  # Keep 30 days of data
    retention.size: 100GB  # Or limit by size
```

### Step 6.4: Set Up Backup

```bash
# Create CronJob to snapshot Prometheus data
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: prometheus-backup
  namespace: monitoring
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: alpine:latest
            command:
            - /bin/sh
            - -c
            - |
              # Snapshot Prometheus data
              wget --post-data="" http://prometheus-server:80/api/v1/admin/tsdb/snapshot
              # Copy to backup location
              # ... (add your backup logic)
          restartPolicy: OnFailure
EOF
```

### Step 6.5: Document Runbooks

Create runbook links for all critical alerts:

```yaml
annotations:
  runbook_url: "https://wiki.company.com/runbooks/{{ $labels.alertname }}"
```

Create runbook pages with:
- Alert description
- Investigation steps
- Resolution procedures
- Escalation contacts

### Step 6.6: Set Up On-Call Rotation

Configure PagerDuty schedules:

1. Create escalation policy
2. Add on-call rotation
3. Test page notifications
4. Document escalation procedures

### Step 6.7: Production Checklist

Before declaring production-ready:

- [ ] All recording rules loading successfully
- [ ] All alert rules configured
- [ ] AlertManager routing working
- [ ] Grafana dashboards created
- [ ] Performance benchmarks passing (>5x improvement)
- [ ] Cardinality within limits (<500k series)
- [ ] Backups configured
- [ ] Runbooks documented
- [ ] On-call rotation configured
- [ ] Team trained on new dashboards

---

## Troubleshooting

### Issue 1: Recording Rules Not Evaluating

**Symptoms:**
```bash
curl http://localhost:9090/api/v1/query?query=model:predictions:rate5m
# Returns: {"data":{"result":[]}}
```

**Solution:**
1. Check base metrics exist:
   ```bash
   curl 'http://localhost:9090/api/v1/query?query=model_predictions_total'
   ```
2. Check Prometheus logs:
   ```bash
   kubectl logs -n monitoring deployment/prometheus-server | grep -i "rule\|error"
   ```
3. Verify recording rule interval:
   ```bash
   curl http://localhost:9090/api/v1/rules | jq '.data.groups[] | {name: .name, interval: .interval}'
   ```

### Issue 2: High Cardinality Warnings

**Symptoms:**
```
WARN: Found metrics with >10k series
```

**Solution:**
1. Identify high-cardinality metrics:
   ```bash
   ./scripts/check_cardinality.sh
   ```
2. Drop unnecessary labels via relabeling:
   ```yaml
   metric_relabel_configs:
   - source_labels: [high_cardinality_label]
     action: labeldrop
   ```
3. Aggregate high-cardinality dimensions in recording rules

### Issue 3: Alert Fatigue

**Symptoms:**
Too many alerts firing, team ignoring notifications

**Solution:**
1. Review alert thresholds:
   ```bash
   # Count firing alerts
   curl -s http://localhost:9090/api/v1/alerts | \
       jq '.data.alerts | map(select(.state=="firing")) | length'
   ```
2. Increase `for` duration for noisy alerts
3. Add silences for known issues:
   ```bash
   amtool silence add alertname=NoisyAlert
   ```
4. Use alert inhibition rules

### Issue 4: Slow Dashboard Performance

**Symptoms:**
Dashboard takes >5s to load even with recording rules

**Solution:**
1. Check if queries using recording rules:
   ```promql
   # Bad: sum by (model) (rate(model_predictions_total[5m]))
   # Good: model:predictions:rate5m
   ```
2. Reduce time range (6h instead of 24h)
3. Increase dashboard refresh interval (60s instead of 10s)
4. Check Prometheus query performance:
   ```bash
   curl http://localhost:9090/metrics | grep prometheus_engine_query_duration
   ```

---

## Validation Checklist

### Recording Rules
- [ ] All 40+ rules loaded successfully
- [ ] Queries return data: `model:predictions:rate5m`
- [ ] Performance improvement >5x
- [ ] Storage overhead <15%

### Alert Rules
- [ ] All alerts loaded successfully
- [ ] Test alerts firing correctly
- [ ] AlertManager routing working
- [ ] Notifications received (Slack/PagerDuty)

### Dashboards
- [ ] Model health dashboard created
- [ ] Platform overview dashboard created
- [ ] SLO dashboard created
- [ ] Dashboard load time <1s
- [ ] Variables working correctly

### Production
- [ ] Backups configured
- [ ] Runbooks documented
- [ ] On-call rotation active
- [ ] Team trained
- [ ] Monitoring metrics monitored

---

## Next Steps

After successful deployment:

1. **Week 1:** Monitor alert noise, adjust thresholds
2. **Week 2:** Add custom recording rules for your use cases
3. **Week 3:** Create business dashboards for stakeholders
4. **Month 2:** Review and optimize cardinality
5. **Month 3:** Implement long-term storage (Thanos/Cortex)

---

## Resources

- [Recording Rules Documentation](https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/)
- [Alerting Rules Documentation](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
- [AlertManager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)
- [Grafana Provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)

**Congratulations!** You've successfully deployed production-grade Prometheus monitoring for your ML platform. 🎉
