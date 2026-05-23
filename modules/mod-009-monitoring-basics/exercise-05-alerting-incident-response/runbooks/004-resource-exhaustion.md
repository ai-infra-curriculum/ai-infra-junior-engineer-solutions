# Runbook: Resource Exhaustion Alert

**Alert Name**: `HighCPUUsage` / `HighMemoryUsage` / `HighDiskUsage`
**Severity**: Warning → Critical (escalates)
**Page On-Call**: Critical only
**SLO Impact**: Medium to High (degraded performance)

---

## Alert Description

The inference gateway container is experiencing resource saturation, which can lead to degraded performance, request timeouts, or service crashes. Resource exhaustion is often a leading indicator of more serious incidents.

### Alert Queries

**High CPU Usage**:
```promql
(
  rate(container_cpu_usage_seconds_total{container="inference-gateway"}[5m])
  * 100
) > 80
```

**High Memory Usage**:
```promql
(
  container_memory_usage_bytes{container="inference-gateway"}
  /
  container_spec_memory_limit_bytes{container="inference-gateway"}
  * 100
) > 85
```

**High Disk Usage**:
```promql
(
  1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)
) * 100 > 85
```

### Severity Levels

| Resource | Warning | Critical | Emergency |
|----------|---------|----------|-----------|
| CPU      | >70%    | >85%     | >95%      |
| Memory   | >75%    | >90%     | >98%      |
| Disk     | >80%    | >90%     | >95%      |

---

## Triage (First 5 Minutes)

### Step 1: Identify Resource Type

```bash
# Check all resources at once
docker stats inference-gateway --no-stream

# Output shows:
# CONTAINER    CPU %   MEM USAGE / LIMIT   MEM %   NET I/O   BLOCK I/O   PIDS
# inference... 87.5%   1.8GiB / 2GiB       90%     ...       ...         42
```

**Determine severity**:
- Single resource >85%: Warning (investigate)
- Single resource >90%: Critical (immediate action)
- Multiple resources >80%: Critical (system strain)
- Any resource >98%: Emergency (imminent failure)

### Step 2: Check Service Health

```bash
# Is service still responding?
time curl http://localhost:8000/health
# Look for slow response time (>1s indicates strain)

# Check error rate
curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])' | jq
```

### Step 3: Quick Timeline Assessment

```bash
# When did this start?
# Check resource usage over last hour
curl -s 'http://localhost:9090/api/v1/query_range?query=rate(container_cpu_usage_seconds_total{container="inference-gateway"}[5m])*100&start='$(date -u -d '1 hour ago' +%s)'&end='$(date -u +%s)'&step=60' | jq

# Sudden spike vs gradual increase?
# Sudden: Likely traffic surge or specific request
# Gradual: Likely resource leak or growing load
```

---

## CPU Exhaustion Investigation

### Step 4: Identify CPU-Intensive Operations

```bash
# Check current CPU usage
docker stats inference-gateway --no-stream | grep CPU

# Get process tree inside container
docker exec inference-gateway ps aux --sort=-%cpu | head -20

# Check for CPU-intensive threads
docker exec inference-gateway top -bn1 | head -20
```

**Common causes**:
- Model inference with large inputs
- Infinite loops or deadlocks
- Regex operations on large text
- Cryptographic operations
- JSON parsing of huge payloads

### Step 5: Check Request Pattern

```bash
# Request rate
curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{job="inference-gateway"}[5m])' | jq

# Request duration (slow requests?)
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(http_request_duration_seconds_bucket[5m]))' | jq

# Check for unusually large requests
docker exec inference-gateway tail -100 /var/log/app.log | grep -i "large\|size\|payload"
```

### Step 6: Profile CPU Usage (If Available)

```bash
# Python profiling (if app supports it)
curl http://localhost:8000/debug/pprof/profile?seconds=10 > cpu_profile.prof

# Or use py-spy for live profiling
docker exec inference-gateway py-spy top --pid 1 --duration 10

# Check for hot loops
docker exec inference-gateway py-spy dump --pid 1
```

---

## Memory Exhaustion Investigation

### Step 7: Check Memory Usage Pattern

```bash
# Current memory usage
docker stats inference-gateway --no-stream | grep MEM

# Memory usage over time (check for leak)
curl -s 'http://localhost:9090/api/v1/query_range?query=container_memory_usage_bytes{container="inference-gateway"}&start='$(date -u -d '6 hours ago' +%s)'&end='$(date -u +%s)'&step=300' | jq

# Is it steadily increasing? → Memory leak
# Is it spiky? → Large request processing
# Is it flat at limit? → Normal high usage
```

### Step 8: Identify Memory-Intensive Operations

```bash
# Check process memory inside container
docker exec inference-gateway ps aux --sort=-%mem | head -20

# Check Python heap usage (if applicable)
docker exec inference-gateway python3 -c "
import sys
import gc
print(f'Objects: {len(gc.get_objects())}')
print(f'Memory: {sys.getsizeof(gc.get_objects()) / 1024 / 1024} MB')
"

# Check for large objects in memory
docker exec inference-gateway python3 -c "
import gc
import sys
objects = gc.get_objects()
large = sorted([(sys.getsizeof(o), type(o)) for o in objects], reverse=True)[:10]
for size, obj_type in large:
    print(f'{size/1024/1024:.2f} MB - {obj_type}')
"
```

**Common causes**:
- Large model weights not released
- In-memory caching without limits
- Request data not garbage collected
- Large numpy arrays
- Connection pool leaks

### Step 9: Check for Memory Leaks

```bash
# Compare memory before and after request
BEFORE=$(docker stats inference-gateway --no-stream --format "{{.MemUsage}}")
curl -X POST http://localhost:8000/predict -d '{"data":[[1,2,3]]}'
sleep 2
AFTER=$(docker stats inference-gateway --no-stream --format "{{.MemUsage}}")
echo "Before: $BEFORE, After: $AFTER"

# If memory doesn't decrease after requests, likely a leak
```

---

## Disk Exhaustion Investigation

### Step 10: Identify Disk Usage

```bash
# Check disk usage
df -h

# Find large directories
du -sh /* 2>/dev/null | sort -hr | head -10

# Find large files
find / -type f -size +100M 2>/dev/null | xargs ls -lh | sort -k5 -hr

# Check Docker disk usage
docker system df
```

**Common causes**:
- Log files not rotated
- Temporary files not cleaned
- Model checkpoints accumulating
- Container image layers
- Old Docker volumes

### Step 11: Check Log File Growth

```bash
# Find large log files
find /var/log -type f -size +100M 2>/dev/null | xargs ls -lh

# Check application logs
docker logs inference-gateway --tail=1000 | wc -l
du -sh /var/lib/docker/containers/*/

# Check log rotation configuration
cat /etc/logrotate.d/* 2>/dev/null
```

---

## Mitigation Strategies

### Strategy 1: Immediate Relief - Restart Service

**When to use**: Resource usage >95%, service degraded but not down

```bash
# Graceful restart (releases memory/CPU)
docker-compose restart inference-gateway

# Monitor recovery
watch -n 1 'docker stats inference-gateway --no-stream'

# Expected: Resources should drop to baseline after restart
```

**Trade-off**: Brief downtime but immediate resource relief.

### Strategy 2: Horizontal Scaling

**When to use**: High load, but individual requests are normal

```bash
# Scale to more replicas (Kubernetes)
kubectl scale deployment/inference-gateway --replicas=5

# Or with Docker Compose (if configured)
docker-compose up -d --scale inference-gateway=3

# Verify load distribution
curl -s http://localhost:9090/api/v1/query?query=sum(rate(http_requests_total{job="inference-gateway"}[5m]))by(instance) | jq
```

### Strategy 3: Vertical Scaling (Increase Limits)

**When to use**: Single instance consistently hits limits

```bash
# Edit docker-compose.yml
# Before:
#   deploy:
#     resources:
#       limits:
#         cpus: '2'
#         memory: 2G

# After:
#   deploy:
#     resources:
#       limits:
#         cpus: '4'
#         memory: 4G

# Apply changes
docker-compose up -d inference-gateway

# Verify new limits
docker inspect inference-gateway | jq '.[0].HostConfig.Memory'
```

### Strategy 4: Rate Limiting (Traffic Control)

**When to use**: Excessive request volume causing strain

```bash
# Enable rate limiting (application level)
docker-compose exec inference-gateway \
  curl -X POST http://localhost:8000/admin/config \
  -d '{"rate_limit_enabled": true, "rate_limit_rps": 100}'

# Or use nginx rate limiting (infrastructure level)
# Add to nginx.conf:
# limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
# limit_req zone=api burst=20;
```

### Strategy 5: Request Shedding (Load Shedding)

**When to use**: Critical resource exhaustion, prevent total failure

```bash
# Enable graceful degradation
docker-compose exec inference-gateway \
  curl -X POST http://localhost:8000/admin/mode \
  -d '{"mode": "degraded", "accept_rate": 0.5}'

# Returns 503 for 50% of requests with Retry-After header
# Allows system to recover while serving some traffic
```

### Strategy 6: Clear Caches and Temporary Data

**When to use**: Disk exhaustion or memory bloat

```bash
# Clear application cache (if supported)
docker-compose exec inference-gateway \
  curl -X POST http://localhost:8000/admin/clear-cache

# Clear old logs
docker-compose exec inference-gateway \
  find /var/log -name "*.log" -mtime +7 -delete

# Clear temporary files
docker-compose exec inference-gateway \
  find /tmp -type f -mtime +1 -delete

# Docker system cleanup
docker system prune -f
docker volume prune -f
```

### Strategy 7: Kill Resource-Heavy Requests

**When to use**: Specific requests identified as culprit

```bash
# Find long-running requests
docker exec inference-gateway ps aux | grep python | awk '{if ($10 > 300) print $2}'

# Kill specific process
docker exec inference-gateway kill -9 <PID>

# Or implement request timeout (application level)
# Set timeout: 30s for all requests
```

---

## CPU-Specific Mitigations

### Optimize Inference Configuration

```bash
# Reduce model batch size (less CPU per request)
curl -X POST http://localhost:8000/admin/config \
  -d '{"max_batch_size": 8}'  # Was 32

# Reduce worker threads
curl -X POST http://localhost:8000/admin/config \
  -d '{"num_workers": 4}'  # Was 8

# Enable model quantization (if not already)
curl -X POST http://localhost:8000/admin/config \
  -d '{"use_quantization": true}'
```

### Enable CPU Affinity

```bash
# Pin to specific CPUs to avoid context switching
docker update --cpuset-cpus="0-3" inference-gateway

# Or in docker-compose.yml:
# cpuset: "0-3"
```

---

## Memory-Specific Mitigations

### Force Garbage Collection

```bash
# Trigger Python garbage collection
docker exec inference-gateway python3 -c "import gc; gc.collect()"

# Or via admin endpoint (if implemented)
curl -X POST http://localhost:8000/admin/gc
```

### Clear Model Cache

```bash
# Clear cached models (will reload on next request)
docker exec inference-gateway rm -rf /tmp/model_cache/*

# Or via admin endpoint
curl -X POST http://localhost:8000/admin/clear-model-cache
```

### Implement Memory Limits Per Request

```bash
# Configure max request payload size
curl -X POST http://localhost:8000/admin/config \
  -d '{"max_payload_size": "10MB"}'  # Was 100MB

# Limit in-memory result size
curl -X POST http://localhost:8000/admin/config \
  -d '{"max_result_size": "5MB"}'
```

---

## Disk-Specific Mitigations

### Rotate and Compress Logs

```bash
# Immediate log rotation
docker exec inference-gateway logrotate -f /etc/logrotate.conf

# Compress old logs
docker exec inference-gateway \
  find /var/log -name "*.log" -mtime +1 -exec gzip {} \;

# Move logs to external storage
docker exec inference-gateway \
  find /var/log -name "*.log.gz" -mtime +7 \
  -exec aws s3 cp {} s3://logs-archive/ \; \
  -delete
```

### Clean Docker Resources

```bash
# Remove unused images
docker image prune -a -f

# Remove unused volumes
docker volume prune -f

# Remove build cache
docker builder prune -a -f

# Remove stopped containers
docker container prune -f
```

---

## Monitoring and Verification

### Continuous Monitoring (During Mitigation)

```bash
# Watch resources every 2 seconds
watch -n 2 'docker stats inference-gateway --no-stream'

# Monitor request success rate
watch -n 5 'curl -s http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"2.."}[1m])/rate(http_requests_total[1m]) | jq'

# Monitor error rate
watch -n 5 'curl -s http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[1m]) | jq'
```

### Verify Resolution

**Resource levels normalized**:
```bash
# CPU should be <70%
# Memory should be <75%
# Disk should be <80%
docker stats inference-gateway --no-stream
```

**Service health restored**:
```bash
# Health endpoint responsive
curl http://localhost:8000/health
# Expected: HTTP 200, <100ms response

# Error rate normal
curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])/rate(http_requests_total[5m])*100' | jq
# Expected: <1%

# Latency normal
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(http_request_duration_seconds_bucket[5m]))' | jq
# Expected: <0.5s
```

---

## Communication

### Warning Alert (CPU >70%, Memory >75%)

**Post in #monitoring** (not #incidents):
```
⚠️  WARNING: Resource Pressure - Inference Gateway
CPU: 78% (threshold: 70%)
Memory: 82% (threshold: 75%)
Status: Monitoring, no user impact yet

Action: Observing for 10 minutes before intervention
Dashboard: http://localhost:3000/d/infrastructure-health
```

### Critical Alert (CPU >85%, Memory >90%)

**Post in #incidents**:
```
🔴 CRITICAL: Resource Exhaustion - Inference Gateway
Severity: P1
IC: @your-name
Started: 2025-10-23 16:30 UTC

Resources:
- CPU: 92% (critical threshold: 85%)
- Memory: 94% (critical threshold: 90%)
- Disk: 76% (ok)

Impact: Degraded performance, ~15% slow responses
Action: Scaling replicas 3 → 5
ETA: 2 minutes

Dashboard: http://localhost:3000/d/infrastructure-health
Next Update: 16:35 UTC
```

---

## Prevention

### Proactive Monitoring Alerts

```yaml
# Add to alerting_rules.yml

# Predictive memory alert (trending toward exhaustion)
- alert: MemoryUsageTrendingHigh
  expr: |
    predict_linear(
      container_memory_usage_bytes{container="inference-gateway"}[1h],
      3600
    ) >
    container_spec_memory_limit_bytes * 0.95
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Memory will hit limit in ~1 hour"
    description: "Current: {{ $value | humanize }}"

# CPU sustained high usage
- alert: CPUSustainedHigh
  expr: |
    avg_over_time(
      rate(container_cpu_usage_seconds_total{container="inference-gateway"}[5m])[30m:]
    ) > 0.75
  labels:
    severity: warning
  annotations:
    summary: "CPU sustained >75% for 30 minutes"
```

### Auto-Scaling Configuration

```yaml
# Kubernetes HPA (Horizontal Pod Autoscaler)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-gateway
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75
```

### Resource Limits and Requests

```yaml
# Set appropriate limits in docker-compose.yml
services:
  inference-gateway:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G
        reservations:
          cpus: '2'
          memory: 2G
```

### Log Rotation

```bash
# Configure log rotation
# /etc/logrotate.d/inference-gateway
/var/log/inference-gateway/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 app app
    postrotate
        docker kill -s USR1 inference-gateway
    endscript
}
```

---

## Additional Resources

- **Resource Monitoring Dashboard**: http://localhost:3000/d/infrastructure-health
- **Prometheus Queries**: docs/monitoring/prometheus-queries.md
- **Performance Tuning Guide**: docs/optimization/performance-tuning.md
- **Capacity Planning**: docs/capacity/planning.md

---

## Runbook Metadata

**Last Updated**: 2025-10-23
**Owner**: SRE Team
**Reviewers**: Platform Team, ML Infrastructure Team
**Next Review**: 2025-11-23
**Average Resolution Time**: 12 minutes
**Incident Frequency**: 2-3/month (warning level), <1/month (critical)
