# Troubleshooting Guide: Prometheus Monitoring Stack

## Overview

This guide covers common issues with Prometheus, Alertmanager, exporters, and their solutions.

---

## Table of Contents

1. [Prometheus Issues](#prometheus-issues)
2. [Alertmanager Issues](#alertmanager-issues)
3. [Scraping Issues](#scraping-issues)
4. [Query Performance Issues](#query-performance-issues)
5. [Storage Issues](#storage-issues)
6. [Alert Issues](#alert-issues)
7. [Exporter Issues](#exporter-issues)

---

## Prometheus Issues

### Issue: Prometheus Won't Start

**Symptoms**:
```bash
docker-compose logs prometheus
# Error: invalid configuration file
```

**Diagnosis**:
```bash
# Validate configuration
docker run --rm -v $(pwd)/config/prometheus:/etc/prometheus \
  prom/prometheus:v2.48.0 \
  promtool check config /etc/prometheus/prometheus.yml
```

**Common Errors**:

**1. YAML Syntax Error**:
```
Error: yaml: line 10: mapping values are not allowed here
```

**Solution**:
```bash
# Check indentation (use spaces, not tabs)
# Validate with yamllint
yamllint config/prometheus/prometheus.yml
```

**2. Invalid Scrape Config**:
```
Error: scrape timeout greater than scrape interval
```

**Solution**:
```yaml
scrape_configs:
  - job_name: 'my-service'
    scrape_interval: 15s
    scrape_timeout: 10s  # Must be < scrape_interval
```

---

### Issue: High Memory Usage

**Symptoms**:
```bash
docker stats prometheus
# Memory: 28GB / 32GB (90%)

# Or Kubernetes
kubectl top pod prometheus-0 -n monitoring
# Memory: 28Gi / 32Gi
```

**Diagnosis**:
```promql
# Check number of time series
prometheus_tsdb_symbol_table_size_bytes

# Check cardinality
count({__name__=~".+"})

# Top metrics by series count
topk(10, count by (__name__)({__name__=~".+"}))
```

**Root Causes & Solutions**:

**1. High Cardinality Labels**:
```promql
# BAD: User ID as label (millions of values)
http_requests_total{user_id="12345"}

# GOOD: Aggregated without high-cardinality labels
http_requests_total{endpoint="/api/users"}
```

**Solution**:
```bash
# Find high-cardinality metrics
promtool tsdb analyze /prometheus

# Remove offending metrics via relabeling
# In prometheus.yml:
metric_relabel_configs:
  - source_labels: [user_id]
    regex: '.*'
    action: labeldrop
```

**2. Too Many Targets**:
```bash
# Check target count
curl 'http://localhost:9090/api/v1/targets' | jq '.data.activeTargets | length'

# If > 1000 targets, consider federation or sharding
```

**3. Long Retention**:
```bash
# Reduce retention
--storage.tsdb.retention.time=15d  # Instead of 30d
```

---

### Issue: Prometheus Crashes (OOMKilled)

**Symptoms**:
```bash
kubectl describe pod prometheus-0
# Last State: Terminated, Reason: OOMKilled
```

**Solution**:
```yaml
# Increase memory limits
resources:
  limits:
    memory: 64Gi  # Increase from 32Gi

# Or reduce data ingestion
# - Increase scrape_interval (15s → 30s)
# - Remove unused metrics
# - Use recording rules for expensive queries
```

---

## Alertmanager Issues

### Issue: Alerts Not Being Sent

**Symptoms**:
```bash
# Alerts show as FIRING in Prometheus
curl http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing")'

# But no notifications received
```

**Diagnosis**:
```bash
# 1. Check if Alertmanager received alerts
curl http://localhost:9093/api/v2/alerts | jq

# 2. Check Alertmanager logs
docker-compose logs alertmanager | grep -i error

# 3. Verify Prometheus → Alertmanager connection
curl 'http://localhost:9090/api/v1/status/config' | jq '.data.yaml' | grep alertmanagers
```

**Common Issues**:

**1. Alertmanager Unreachable**:
```yaml
# In prometheus.yml
alerting:
  alertmanagers:
    - static_configs:
      - targets:
        - alertmanager:9093  # Use service name in Docker, not localhost!
```

**2. Invalid Alertmanager Config**:
```bash
# Validate Alertmanager config
docker run --rm -v $(pwd)/config/alertmanager:/etc/alertmanager \
  prom/alertmanager:v0.26.0 \
  amtool check-config /etc/alertmanager/alertmanager.yml
```

**3. Alerts Silenced**:
```bash
# Check active silences
curl http://localhost:9093/api/v2/silences | jq

# Delete silence
curl -X DELETE http://localhost:9093/api/v2/silence/<silence_id>
```

**4. Receiver Configuration Error**:
```yaml
# Check receiver exists
receivers:
  - name: 'pagerduty'  # Must match route receiver
    pagerduty_configs:
      - service_key: 'YOUR_KEY'

route:
  receiver: 'pagerduty'  # Must match receiver name
```

---

### Issue: Too Many Duplicate Alerts

**Symptoms**:
```
Receiving 10 identical Slack messages for same alert
```

**Solution**:
```yaml
# Configure proper grouping in alertmanager.yml
route:
  group_by: ['alertname', 'service', 'severity']
  group_wait: 30s  # Wait 30s before sending first notification
  group_interval: 5m  # Wait 5m before sending new alerts in group
  repeat_interval: 4h  # Repeat notification every 4h if not resolved
```

---

## Scraping Issues

### Issue: Target Down

**Symptoms**:
```
# In Prometheus UI: Status > Targets
# Shows target as DOWN (red)
```

**Diagnosis**:
```bash
# 1. Check if target is reachable
curl http://target-host:port/metrics

# 2. Check Prometheus logs
docker-compose logs prometheus | grep "target-host"

# Common errors:
# - "context deadline exceeded" = timeout
# - "connection refused" = port not open
# - "no route to host" = network issue
```

**Solutions**:

**1. Target Not Responding**:
```bash
# Verify service is running
docker-compose ps target-service

# Check if metrics endpoint exists
curl -v http://target:port/metrics
```

**2. Network Connectivity**:
```bash
# Test from Prometheus container
docker-compose exec prometheus wget -O- http://target:port/metrics

# Check Docker network
docker network ls
docker network inspect monitoring_default
```

**3. Timeout Too Short**:
```yaml
scrape_configs:
  - job_name: 'slow-service'
    scrape_timeout: 30s  # Increase from default 10s
```

---

### Issue: Metrics Not Appearing

**Symptoms**:
```promql
# Query returns no data
http_requests_total
# (empty result)
```

**Diagnosis**:
```bash
# 1. Verify target is UP
curl 'http://localhost:9090/api/v1/targets' | jq '.data.activeTargets[] | select(.labels.job=="my-service")'

# 2. Check raw metrics from target
curl http://target:port/metrics | grep http_requests_total

# 3. Check for metric relabeling
curl 'http://localhost:9090/api/v1/status/config' | jq '.data.yaml' | grep -A 10 metric_relabel
```

**Solutions**:

**1. Metric Dropped by Relabeling**:
```yaml
# Check for action: drop rules
metric_relabel_configs:
  - source_labels: [__name__]
    regex: 'http_requests_total'  # This drops the metric!
    action: drop  # Remove this if you need the metric
```

**2. Wrong Metric Name**:
```bash
# List all available metrics
curl 'http://localhost:9090/api/v1/label/__name__/values' | jq
```

---

## Query Performance Issues

### Issue: Queries Timing Out

**Symptoms**:
```promql
# Query takes > 30 seconds or times out
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[30d]))
```

**Diagnosis**:
```bash
# Check query execution time in Prometheus UI
# Enable query logging
--query.log-file=/var/log/prometheus/queries.log
```

**Solutions**:

**1. Use Recording Rules**:
```yaml
# Pre-compute expensive queries
# recording_rules.yml
groups:
  - name: http_latency
    interval: 30s
    rules:
      - record: http:request_duration:p99:rate30d
        expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[30d]))

# Then query the recorded metric
http:request_duration:p99:rate30d  # Fast!
```

**2. Reduce Time Range**:
```promql
# Instead of 30 days
rate(http_requests_total[7d])  # Use 7 days

# Or use subquery
max_over_time(http:requests:rate5m[30d:5m])  # Only evaluate every 5m
```

**3. Limit Cardinality**:
```promql
# Aggregate early
sum by (status) (rate(http_requests_total[5m]))  # Good

# Instead of
rate(http_requests_total[5m])  # Returns all label combinations
```

---

## Storage Issues

### Issue: Disk Space Full

**Symptoms**:
```bash
df -h /prometheus
# /dev/sda1   100G  98G  0  100% /prometheus

docker-compose logs prometheus
# Error: no space left on device
```

**Solutions**:

**1. Reduce Retention**:
```bash
# Restart with shorter retention
--storage.tsdb.retention.time=15d  # Instead of 30d
```

**2. Increase Disk Size** (Kubernetes):
```bash
# Resize PVC
kubectl patch pvc prometheus-data-prometheus-0 -n monitoring \
  -p '{"spec":{"resources":{"requests":{"storage":"1Ti"}}}}'

# Resize underlying volume (AWS EBS)
aws ec2 modify-volume --volume-id vol-xxx --size 1000
```

**3. Clean Up Old Data**:
```bash
# Delete data older than 15 days (dangerous! backup first)
curl -X POST http://localhost:9090/api/v1/admin/tsdb/delete_series?match[]={__name__=~".+"}&start=0&end=$(date -d '15 days ago' +%s)

# Clean tombstones
curl -X POST http://localhost:9090/api/v1/admin/tsdb/clean_tombstones
```

---

### Issue: TSDB Corruption

**Symptoms**:
```bash
docker-compose logs prometheus
# Error: opening storage failed: corruption in block
```

**Solution**:
```bash
# 1. Stop Prometheus
docker-compose stop prometheus

# 2. Check and repair TSDB
docker run --rm -v prometheus_data:/prometheus \
  prom/prometheus:v2.48.0 \
  promtool tsdb analyze /prometheus

# 3. If corrupt, remove broken block
# (data loss for that block's time range)
rm -rf /prometheus/01HXXX

# 4. Restart Prometheus
docker-compose start prometheus
```

---

## Alert Issues

### Issue: Alerts Flapping (Firing/Resolved Repeatedly)

**Symptoms**:
```
Alert fires, resolves 30 seconds later, fires again...
```

**Solution**:
```yaml
# Increase 'for' duration
alert: HighErrorRate
expr: rate(http_errors_total[5m]) > 0.05
for: 10m  # Increase from 2m - must be true for 10min before firing
```

---

### Issue: Alert Not Firing

**Symptoms**:
```
Condition is true but alert doesn't fire
```

**Diagnosis**:
```bash
# 1. Test alert expression in Prometheus UI
# Run the exact 'expr' from alert rule

# 2. Check if rule is loaded
curl 'http://localhost:9090/api/v1/rules' | jq '.data.groups[].rules[] | select(.name=="MyAlert")'

# 3. Check evaluation interval
# Alert must be true for duration specified in 'for:'
```

**Solutions**:

**1. Expression Never True**:
```bash
# Test expression
curl 'http://localhost:9090/api/v1/query?query=rate(http_errors_total[5m]) > 0.05'
# If empty, condition is never true
```

**2. 'for' Duration Too Long**:
```yaml
alert: HighErrorRate
expr: rate(http_errors_total[5m]) > 0.05
for: 5m  # Reduce from 30m if condition is transient
```

**3. Rules File Not Loaded**:
```bash
# Check Prometheus config includes rules file
curl 'http://localhost:9090/api/v1/status/config' | jq '.data.yaml' | grep rule_files

# Reload Prometheus after adding rules
curl -X POST http://localhost:9090/-/reload
```

---

## Exporter Issues

### Issue: Node Exporter Not Collecting Metrics

**Symptoms**:
```
Missing node_* metrics in Prometheus
```

**Diagnosis**:
```bash
# Check if exporter is running
curl http://localhost:9100/metrics

# Check Prometheus target
curl 'http://localhost:9090/api/v1/targets' | jq '.data.activeTargets[] | select(.labels.job=="node-exporter")'
```

**Solution**:
```yaml
# Ensure correct volume mounts (for Docker)
node-exporter:
  volumes:
    - /proc:/host/proc:ro
    - /sys:/host/sys:ro
    - /:/host/root:ro,rslave
  command:
    - '--path.procfs=/host/proc'
    - '--path.sysfs=/host/sys'
    - '--path.rootfs=/host/root'
```

---

### Issue: cAdvisor Permission Denied

**Symptoms**:
```
Error: failed to get container stats: permission denied
```

**Solution**:
```yaml
# Run cAdvisor as privileged
cadvisor:
  privileged: true
  volumes:
    - /:/rootfs:ro
    - /var/run:/var/run:ro
    - /sys:/sys:ro
    - /var/lib/docker/:/var/lib/docker:ro
```

---

## Debugging Workflow

### Step 1: Check Component Health

```bash
# Prometheus
curl http://localhost:9090/-/healthy
curl http://localhost:9090/-/ready

# Alertmanager
curl http://localhost:9093/-/healthy

# Exporters
curl http://localhost:9100/metrics  # Node Exporter
curl http://localhost:8080/metrics  # cAdvisor
```

### Step 2: Check Targets

```bash
# List all targets and their health
curl 'http://localhost:9090/api/v1/targets' | jq '.data.activeTargets[] | {job: .labels.job, instance: .labels.instance, health: .health, error: .lastError}'
```

### Step 3: Check Logs

```bash
# Prometheus
docker-compose logs prometheus | tail -100

# Alertmanager
docker-compose logs alertmanager | tail -100

# Filter for errors
docker-compose logs | grep -i error
```

### Step 4: Test Queries

```bash
# Simple query
curl 'http://localhost:9090/api/v1/query?query=up'

# Alert expression
curl 'http://localhost:9090/api/v1/query?query=slo:availability:burn_rate:1h > 14.4'
```

### Step 5: Check Alerts

```bash
# Firing alerts in Prometheus
curl 'http://localhost:9090/api/v1/alerts' | jq '.data.alerts[] | select(.state=="firing")'

# Alerts in Alertmanager
curl 'http://localhost:9093/api/v2/alerts' | jq
```

---

## Diagnostics Script

```bash
#!/bin/bash
# prometheus-diagnostics.sh

echo "=== Prometheus Health ==="
curl -s http://localhost:9090/-/healthy
curl -s http://localhost:9090/-/ready

echo -e "\n=== Target Status ==="
curl -s 'http://localhost:9090/api/v1/targets' | \
  jq -r '.data.activeTargets[] | "\(.labels.job) - \(.health)"'

echo -e "\n=== TSDB Stats ==="
curl -s 'http://localhost:9090/api/v1/status/tsdb' | jq

echo -e "\n=== Active Alerts ==="
curl -s 'http://localhost:9090/api/v1/alerts' | \
  jq -r '.data.alerts[] | select(.state=="firing") | "\(.labels.alertname) - \(.labels.severity)"'

echo -e "\n=== Disk Usage ==="
df -h | grep prometheus

echo -e "\n=== Memory Usage ==="
docker stats --no-stream prometheus

echo -e "\n=== Recent Errors ==="
docker-compose logs prometheus | grep -i error | tail -20
```

---

## Summary

**Common Issues Checklist**:
- [ ] Prometheus is running and healthy (`/-/healthy`)
- [ ] All targets are UP (`/targets`)
- [ ] Queries return expected data
- [ ] Recording rules are evaluating (`/rules`)
- [ ] Alert rules are evaluating (`/rules`)
- [ ] Alerts are being sent to Alertmanager
- [ ] Alertmanager is delivering notifications
- [ ] Disk space is sufficient (`df -h`)
- [ ] Memory usage is not at limit (`docker stats`)
- [ ] No errors in logs

**When to Escalate**:
1. TSDB corruption (data loss possible)
2. Persistent OOM kills
3. Critical alerts not being delivered
4. Widespread target scraping failures

**Resources**:
- [Prometheus Troubleshooting](https://prometheus.io/docs/prometheus/latest/troubleshooting/)
- [Alertmanager Troubleshooting](https://prometheus.io/docs/alerting/latest/troubleshooting/)
- [PromQL Debugging](https://promlabs.com/blog/2020/06/18/debugging-promql-queries/)
