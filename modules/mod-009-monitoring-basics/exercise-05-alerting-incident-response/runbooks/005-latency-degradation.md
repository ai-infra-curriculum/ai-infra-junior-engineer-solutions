# Runbook: Latency Degradation Alert

**Alert Name**: `SLOLatencyP99Violation` / `HighLatency`
**Severity**: Warning → Critical
**Page On-Call**: Critical only (P99 > 1s)
**SLO Impact**: High (directly impacts user experience)

---

## Alert Description

The inference gateway is experiencing elevated response times, with P99 latency exceeding the 300ms SLO threshold. Latency degradation directly impacts user experience and can cascade into timeout errors if unresolved.

### Alert Queries

**P99 Latency SLO Violation**:
```promql
histogram_quantile(0.99,
  rate(http_request_duration_seconds_bucket{service="inference-gateway"}[5m])
) > 0.3  # 300ms SLO threshold
```

**P50 Latency (Median)**:
```promql
histogram_quantile(0.50,
  rate(http_request_duration_seconds_bucket{service="inference-gateway"}[5m])
) > 0.1  # 100ms threshold
```

**P95 Latency**:
```promql
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket{service="inference-gateway"}[5m])
) > 0.2  # 200ms threshold
```

### Severity Thresholds

| Percentile | Target | Warning | Critical |
|------------|--------|---------|----------|
| P50        | <50ms  | >100ms  | >200ms   |
| P95        | <200ms | >300ms  | >500ms   |
| P99        | <300ms | >500ms  | >1000ms  |

**SLO**: 95% of requests complete within 300ms (P95 < 300ms)

---

## Triage (First 5 Minutes)

### Step 1: Assess Latency Scope

```bash
# Check current P50/P95/P99 latency
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(http_request_duration_seconds_bucket{service="inference-gateway"}[5m]))' | jq '.data.result[0].value[1]'

# Are all endpoints affected or specific ones?
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(http_request_duration_seconds_bucket{service="inference-gateway"}[5m]))by(endpoint)' | jq
```

**Determine scope**:
- All endpoints slow: System-wide issue (resource, dependency)
- Specific endpoint slow: Endpoint-specific problem (query, logic)
- Intermittent slowness: Load spikes or periodic background tasks

### Step 2: Check User Impact

```bash
# What's the request success rate?
curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"2.."}[5m])/rate(http_requests_total[5m])*100' | jq

# Are requests timing out?
curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status="504"}[5m])' | jq

# Current request volume
curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])' | jq
```

### Step 3: Timeline Analysis

```bash
# When did latency start increasing?
# Compare current vs 1 hour ago
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(http_request_duration_seconds_bucket[5m]))&time='$(date -u -d '1 hour ago' +%s) | jq

# Sudden spike or gradual increase?
# Graph over last 2 hours
curl -s 'http://localhost:9090/api/v1/query_range?query=histogram_quantile(0.99,rate(http_request_duration_seconds_bucket[5m]))&start='$(date -u -d '2 hours ago' +%s)'&end='$(date -u +%s)'&step=60' | jq
```

**Pattern recognition**:
- **Sudden spike**: Deployment, traffic surge, dependency failure
- **Gradual increase**: Resource leak, cache exhaustion, query inefficiency
- **Periodic spikes**: Background jobs, scheduled tasks, traffic patterns

---

## Investigation (Next 10 Minutes)

### Step 4: Identify Latency Breakdown

**Use distributed tracing** to see where time is spent:

```bash
# Find a slow trace
# 1. Go to Jaeger UI: http://localhost:16686
# 2. Service: inference-gateway
# 3. Min Duration: 500ms
# 4. Limit: 20 traces
# 5. Find Traces

# Analyze span durations:
# - Which span is slowest?
# - Is it consistent across traces?
# - Are there outliers?
```

**Common slow spans**:
- Database queries
- External API calls (model service, feature store)
- Model inference
- Data preprocessing
- Network I/O
- Lock contention

### Step 5: Check Database Performance

```bash
# Query latency to database
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(pg_query_duration_seconds_bucket[5m]))' | jq

# Active database connections
curl -s 'http://localhost:9090/api/v1/query?query=pg_stat_activity_count' | jq

# Long-running queries
docker exec postgres psql -U user -d db -c "
  SELECT pid, now() - query_start as duration, query
  FROM pg_stat_activity
  WHERE state = 'active' AND now() - query_start > interval '1 second'
  ORDER BY duration DESC;
"

# Check for locks
docker exec postgres psql -U user -d db -c "
  SELECT blocked_locks.pid AS blocked_pid,
         blocking_locks.pid AS blocking_pid,
         blocked_activity.query AS blocked_query
  FROM pg_catalog.pg_locks blocked_locks
  JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
  JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
  JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
  WHERE NOT blocked_locks.granted;
"
```

### Step 6: Check External Dependencies

```bash
# Model service latency
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(http_request_duration_seconds{job="model-service"}[5m]))' | jq

# Feature store latency
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(http_request_duration_seconds{job="feature-store"}[5m]))' | jq

# Check dependency health
curl http://model-service:8080/health
curl http://feature-store:8081/health

# Network latency to dependencies
docker exec inference-gateway ping -c 3 model-service
docker exec inference-gateway ping -c 3 feature-store
```

### Step 7: Check Resource Saturation

```bash
# CPU usage (high CPU = slow processing)
docker stats inference-gateway --no-stream | grep CPU

# Memory usage (memory pressure = swapping = slow)
docker stats inference-gateway --no-stream | grep MEM

# I/O wait (disk bottleneck)
docker exec inference-gateway iostat -x 1 5

# Network saturation
docker exec inference-gateway iftop -t -s 5
```

### Step 8: Check Application Logs

```bash
# Look for slow query warnings
docker logs inference-gateway --tail=200 | grep -i "slow\|timeout\|latency"

# Look for retry attempts
docker logs inference-gateway --tail=200 | grep -i "retry\|backoff"

# Look for errors that might cause retries
docker logs inference-gateway --tail=200 | grep -i "error\|exception" | tail -20
```

---

## Common Root Causes

### Cause 1: Database Query Performance

**Symptoms**:
- Database span in traces is slow
- Increasing query time over days/weeks
- High database CPU usage

**Investigation**:
```bash
# Find slowest queries
docker exec postgres psql -U user -d db -c "
  SELECT query, calls, total_time, mean_time
  FROM pg_stat_statements
  ORDER BY mean_time DESC
  LIMIT 10;
"

# Check missing indexes
docker exec postgres psql -U user -d db -c "
  SELECT schemaname, tablename, attname, n_distinct, correlation
  FROM pg_stats
  WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
  AND n_distinct > 100
  ORDER BY abs(correlation) ASC
  LIMIT 10;
"

# Table bloat (needs VACUUM)
docker exec postgres psql -U user -d db -c "
  SELECT schemaname, tablename,
         pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
  FROM pg_tables
  WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
  ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
  LIMIT 10;
"
```

**Mitigation**:
```bash
# Add missing index (example)
docker exec postgres psql -U user -d db -c "
  CREATE INDEX CONCURRENTLY idx_features_user_id ON features(user_id);
"

# VACUUM bloated tables
docker exec postgres psql -U user -d db -c "VACUUM ANALYZE features;"

# Kill long-running query (if blocking)
docker exec postgres psql -U user -d db -c "SELECT pg_terminate_backend(PID);"
```

### Cause 2: External Dependency Slowness

**Symptoms**:
- External API spans slow in traces
- Dependency metrics show increased latency
- Correlation with dependency alerts

**Investigation**:
```bash
# Check dependency latency trends
curl -s 'http://localhost:9090/api/v1/query_range?query=histogram_quantile(0.99,rate(http_request_duration_seconds{job="model-service"}[5m]))&start='$(date -u -d '1 hour ago' +%s)'&end='$(date -u +%s)'&step=60' | jq

# Check dependency error rate
curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{job="model-service",status=~"5.."}[5m])' | jq

# Test direct call to dependency
time curl -X POST http://model-service:8080/predict -d '{"data":[[1,2,3]]}'
```

**Mitigation**:
```bash
# Enable timeout and circuit breaker
curl -X POST http://localhost:8000/admin/config -d '{
  "model_service_timeout_ms": 1000,
  "circuit_breaker_enabled": true,
  "circuit_breaker_threshold": 5
}'

# Use cached responses (if applicable)
curl -X POST http://localhost:8000/admin/config -d '{
  "enable_response_cache": true,
  "cache_ttl_seconds": 60
}'

# Reduce dependency call frequency
curl -X POST http://localhost:8000/admin/config -d '{
  "feature_fetch_batch_size": 100
}'
```

### Cause 3: Model Inference Slowness

**Symptoms**:
- Model inference span slow in traces
- Specific request patterns (large inputs) are slow
- GPU/CPU saturation during inference

**Investigation**:
```bash
# Check model inference duration metric
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(ml_model_inference_duration_seconds_bucket[5m]))' | jq

# Check GPU utilization (if applicable)
docker exec inference-gateway nvidia-smi

# Check batch size and throughput
curl -s 'http://localhost:9090/api/v1/query?query=ml_model_batch_size' | jq
curl -s 'http://localhost:9090/api/v1/query?query=rate(ml_model_predictions_total[5m])' | jq

# Check input size distribution
docker logs inference-gateway --tail=100 | grep "input_size" | awk '{print $NF}' | sort -n
```

**Mitigation**:
```bash
# Reduce batch size for lower latency
curl -X POST http://localhost:8000/admin/config -d '{
  "max_batch_size": 8,
  "max_batch_wait_ms": 10
}'

# Use quantized model (faster inference)
curl -X POST http://localhost:8000/admin/config -d '{
  "use_quantized_model": true
}'

# Reject oversized requests
curl -X POST http://localhost:8000/admin/config -d '{
  "max_input_size": 1000
}'
```

### Cause 4: Resource Contention

**Symptoms**:
- High CPU or memory usage
- I/O wait spikes
- Slow across all operations

**Investigation**:
```bash
# Check resource metrics
docker stats inference-gateway --no-stream

# Check for resource throttling
docker exec inference-gateway cat /sys/fs/cgroup/cpu/cpu.stat

# Check I/O statistics
docker exec inference-gateway iostat -x 2 5
```

**Mitigation**: See runbook 004-resource-exhaustion.md

### Cause 5: Cache Misses

**Symptoms**:
- Latency improves after warmup
- Cold start slowness
- Cache hit rate low

**Investigation**:
```bash
# Check cache hit rate
curl -s 'http://localhost:9090/api/v1/query?query=rate(cache_hits_total[5m])/(rate(cache_hits_total[5m])+rate(cache_misses_total[5m]))*100' | jq

# Check cache size
curl http://localhost:8000/admin/cache/stats
```

**Mitigation**:
```bash
# Increase cache size
curl -X POST http://localhost:8000/admin/config -d '{
  "cache_max_size_mb": 1024
}'

# Pre-warm cache
curl -X POST http://localhost:8000/admin/cache/warmup

# Increase cache TTL
curl -X POST http://localhost:8000/admin/config -d '{
  "cache_ttl_seconds": 3600
}'
```

---

## Mitigation Actions

### Immediate Actions (< 2 minutes)

**Option 1: Increase Timeout Tolerance**
```bash
# If requests complete but slowly, increase client timeouts temporarily
# This prevents cascading timeout failures
curl -X POST http://localhost:8000/admin/config -d '{
  "request_timeout_seconds": 10
}'
```

**Option 2: Enable Graceful Degradation**
```bash
# Return cached or default responses for slow operations
curl -X POST http://localhost:8000/admin/config -d '{
  "degraded_mode": true,
  "use_stale_cache": true
}'
```

**Option 3: Rate Limiting**
```bash
# Reduce load to allow system to recover
curl -X POST http://localhost:8000/admin/config -d '{
  "rate_limit_rps": 50
}'
```

### Short-term Actions (< 10 minutes)

**Option 4: Horizontal Scaling**
```bash
# Add more instances to distribute load
kubectl scale deployment/inference-gateway --replicas=6

# Or with Docker Compose
docker-compose up -d --scale inference-gateway=3

# Verify load distribution after 2 minutes
curl -s http://localhost:9090/api/v1/query?query=rate(http_requests_total[1m])by(instance) | jq
```

**Option 5: Database Query Optimization**
```bash
# If database is bottleneck, add index or optimize query
# (Requires code deployment or database change)

# Temporary: Reduce query frequency
curl -X POST http://localhost:8000/admin/config -d '{
  "db_query_cache_ttl_seconds": 300
}'
```

**Option 6: Circuit Breaker for Slow Dependencies**
```bash
# If external dependency is slow, fail fast instead of waiting
curl -X POST http://localhost:8000/admin/config -d '{
  "circuit_breaker_enabled": true,
  "circuit_breaker_timeout_ms": 500,
  "circuit_breaker_failure_threshold": 5
}'
```

### Long-term Actions (< 1 hour)

**Option 7: Code Optimization**
```bash
# Deploy optimized version with performance improvements
# - Database query optimization
# - Caching layer addition
# - Algorithm improvement
# - Async processing

# Rollout with canary deployment
kubectl set image deployment/inference-gateway \
  app=inference-gateway:v1.2.4-optimized \
  --record

# Monitor latency improvement
watch -n 5 'curl -s http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(http_request_duration_seconds_bucket[5m])) | jq'
```

---

## Communication

### Warning Alert (P99 > 500ms, no user impact)

**Post in #monitoring**:
```
⚠️  WARNING: Latency Degradation - Inference Gateway
P99 Latency: 520ms (SLO: 300ms)
P95 Latency: 280ms (ok)
P50 Latency: 85ms (ok)

Status: Monitoring, no errors yet
Affected: Slowest 1% of requests
Action: Investigating root cause

Dashboard: http://localhost:3000/d/app-performance
```

### Critical Alert (P99 > 1s, user impact likely)

**Post in #incidents**:
```
🔴 CRITICAL: High Latency - Inference Gateway
Severity: P1
IC: @your-name
Started: 2025-10-23 17:15 UTC

Latency (P99): 1.2s (SLO: 300ms, +300% over target)
Latency (P95): 850ms
Latency (P50): 250ms

Impact:
- Timeout rate: 3% (normally 0%)
- User experience: Degraded
- Error rate: Still <1%

Root Cause: Under investigation (checking DB, dependencies, resources)
Action: Scaling replicas 3 → 6
ETA: 3 minutes

Dashboard: http://localhost:3000/d/app-performance
Next Update: 17:20 UTC (5 min)
```

### Status Update

```
🔄 UPDATE [17:20 UTC] - 5 MIN ELAPSED

Status: Partially mitigated
Latency (P99): 650ms (was 1.2s, target 300ms)
Latency (P50): 120ms (acceptable)

Root Cause: Database query slowness on features table
Action Taken:
- Scaled to 6 replicas ✅
- Added database index on user_id ✅
- Enabled query result caching (TTL: 5min) ✅

Next Steps:
- Monitor for 10 minutes
- Prepare VACUUM on features table

Next Update: 17:30 UTC
```

---

## Resolution & Verification

### Verify Latency Normalized

```bash
# Check P99 latency
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(http_request_duration_seconds_bucket[5m]))' | jq
# Target: <0.3 (300ms)

# Check P95 latency
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))' | jq
# Target: <0.2 (200ms)

# Check median (P50)
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.50,rate(http_request_duration_seconds_bucket[5m]))' | jq
# Target: <0.05 (50ms)
```

### Check SLO Impact

```bash
# Calculate SLO compliance
curl -s 'http://localhost:9090/api/v1/query?query=slo:latency:ratio_rate30d' | jq

# Error budget remaining
curl -s 'http://localhost:9090/api/v1/query?query=slo:latency:error_budget_remaining' | jq
```

### Monitor for Stability (15 minutes)

```bash
# Watch latency trend
watch -n 30 'curl -s http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,rate(http_request_duration_seconds_bucket[5m])) | jq -r ".data.result[0].value[1]"'

# Ensure no spikes
# Should stay consistently <300ms
```

---

## Prevention

### Performance Testing

```yaml
# Add to CI/CD pipeline
performance-test:
  script:
    - locust -f tests/load_test.py --headless -u 100 -r 10 -t 5m
    - python scripts/check_latency_slo.py --p99-max=300ms
```

### Latency Budget Alerts

```yaml
# Add to alerting_rules.yml
- alert: LatencyBudgetBurning
  expr: |
    slo:latency:burn_rate:1h > 10
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Latency budget burning fast"
    description: "At current rate, latency budget exhausted in {{ $value | humanizeDuration }}"
```

### Query Performance Monitoring

```yaml
# Add to recording_rules.yml
- record: db:query_duration:p99
  expr: |
    histogram_quantile(0.99,
      rate(pg_query_duration_seconds_bucket[5m])
    )
```

---

## Additional Resources

- **Latency Dashboard**: http://localhost:3000/d/app-performance
- **Jaeger Traces**: http://localhost:16686
- **Database Performance**: http://localhost:3000/d/postgres-performance
- **Performance Tuning Guide**: docs/performance/tuning.md

---

## Runbook Metadata

**Last Updated**: 2025-10-23
**Owner**: ML Platform Team
**Reviewers**: SRE Team
**Next Review**: 2025-11-23
**MTTR Goal**: <15 minutes
**Average MTTR**: 12 minutes
**Incident Frequency**: 1-2/month
