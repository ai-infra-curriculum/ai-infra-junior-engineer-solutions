# Runbook: Service Down Alert

**Alert Name**: `ServiceDown`
**Severity**: Critical (P0)
**Page On-Call**: Yes
**SLO Impact**: Extreme (100% error rate)

---

## Alert Description

The inference gateway service is completely unreachable. Prometheus cannot scrape metrics from the service endpoint, indicating a total outage. This represents the most severe incident type, with all user requests failing.

### Alert Query

```promql
up{job="inference-gateway"} == 0
```

**Translation**: The inference-gateway target is down (no successful scrape in last check).

### What This Means

- **Service Status**: Completely unavailable
- **User Impact**: 100% of requests failing
- **Revenue Impact**: Total service disruption
- **SLO Impact**: Immediate and severe error budget consumption

---

## Triage (First 2 Minutes)

### Step 1: Acknowledge Immediately

**This is a P0 incident**. Acknowledge within 1 minute.

```bash
# PagerDuty: Click "Acknowledge"
# Slack: Post in #incidents IMMEDIATELY
```

**Declare Major Incident**:
```
🚨 MAJOR INCIDENT: INFERENCE GATEWAY DOWN
Severity: P0 (All-hands)
Incident Commander: @your-name
Started: [TIME] UTC
Status: Total outage - investigating

ALL HANDS ON DECK
War Room: Zoom https://zoom.us/j/emergency
```

### Step 2: Quick Health Checks

```bash
# Test basic connectivity
curl -I http://localhost:8000/health
# Expected: Connection refused or timeout

# Check if container is running
docker ps | grep inference-gateway
# Expected: May not appear if crashed

# Check container status
docker ps -a | grep inference-gateway
# Look for exit code or restart status
```

### Step 3: Identify Failure Mode

```bash
# Check recent container events
docker events --filter 'container=inference-gateway' --since 10m

# Check exit code
docker inspect inference-gateway --format='{{.State.ExitCode}}'
```

**Common exit codes**:
- `137`: OOM killed (out of memory)
- `139`: Segmentation fault
- `143`: SIGTERM (graceful shutdown)
- `1`: General error

---

## Investigation (Next 5 Minutes)

### Step 4: Check Container Logs

```bash
# Last 100 lines
docker logs inference-gateway --tail=100

# Follow live logs (if restarting)
docker logs inference-gateway -f

# With timestamps
docker logs inference-gateway --tail=100 -t
```

**Look for**:
- Panic messages
- Exception stack traces
- Port binding errors
- Configuration errors
- Connection failures to dependencies

### Step 5: Check System Resources

```bash
# Check memory availability
free -h

# Check disk space
df -h

# Check CPU load
uptime

# Check system logs for OOM
dmesg | grep -i 'out of memory' | tail -20
dmesg | grep -i 'kill' | tail -20
```

### Step 6: Check Dependencies

```bash
# Check if Prometheus is up
curl http://localhost:9090/-/healthy

# Check if other services are affected
docker-compose ps

# Check network connectivity
docker network inspect exercise-05-alerting-incident-response_default

# Test internal DNS resolution
docker-compose exec prometheus ping inference-gateway -c 3
```

### Step 7: Check Recent Changes

```bash
# Check recent deployments
kubectl rollout history deployment/inference-gateway 2>/dev/null || \
  docker inspect inference-gateway --format='{{.Created}}'

# Check recent commits
git log --since="1 hour ago" --oneline

# Check configuration changes
git log --since="1 hour ago" --oneline -- docker-compose.yml config/
```

---

## Mitigation Strategies

### Strategy 1: Restart Service (Fast Recovery)

**If no obvious cause identified, try immediate restart**:

```bash
# Stop and restart
docker-compose restart inference-gateway

# Watch startup logs
docker-compose logs -f inference-gateway

# Verify health after 30 seconds
sleep 30
curl http://localhost:8000/health
```

**If restart fails**, proceed to Strategy 2.

### Strategy 2: Recreate Container (Clean Restart)

**If restart didn't work, try full recreate**:

```bash
# Stop and remove container
docker-compose stop inference-gateway
docker-compose rm -f inference-gateway

# Recreate from image
docker-compose up -d inference-gateway

# Watch startup
docker-compose logs -f inference-gateway

# Verify health after 30 seconds
sleep 30
curl http://localhost:8000/health
```

### Strategy 3: Rollback to Previous Version

**If recent deployment caused the issue**:

```bash
# Kubernetes rollback
kubectl rollout undo deployment/inference-gateway
kubectl rollout status deployment/inference-gateway

# Docker Compose rollback (if using tags)
# Edit docker-compose.yml to use previous image tag
docker-compose pull inference-gateway
docker-compose up -d inference-gateway
```

### Strategy 4: Increase Resources (OOM Kill)

**If killed due to out of memory**:

```bash
# Edit docker-compose.yml
# Increase memory limit:
# deploy:
#   resources:
#     limits:
#       memory: 4G  # Was 2G

# Recreate with new limits
docker-compose up -d inference-gateway

# Monitor memory usage
docker stats inference-gateway
```

### Strategy 5: Fix Configuration Error

**If logs show configuration error**:

```bash
# Common issues:
# 1. Port already in use
lsof -i :8000
# Kill conflicting process or change port

# 2. Environment variable missing
docker-compose config
# Verify all required env vars

# 3. Volume mount issue
docker inspect inference-gateway --format='{{json .Mounts}}' | jq

# Fix configuration in docker-compose.yml or .env
# Then recreate:
docker-compose up -d inference-gateway
```

### Strategy 6: Bypass and Route Around

**If service cannot be restored quickly**:

```bash
# Option A: Activate backup instance
docker-compose up -d inference-gateway-backup

# Option B: Route to different cluster/region
# Update load balancer or DNS

# Option C: Activate maintenance page
# Return 503 with retry-after header
```

---

## Communication

### Initial Notification (within 2 minutes)

**Post in #incidents**:
```
🚨 MAJOR INCIDENT: Inference Gateway Complete Outage
Severity: P0 - All Hands
Incident Commander: @your-name
Started: 2025-10-23 15:45 UTC

Status: TOTAL OUTAGE
- Service: 100% down
- Impact: ALL users affected
- ETA: Investigating (update in 5 min)

Actions:
- Attempting immediate restart
- Checking for OOM/crash
- Preparing rollback

War Room: https://zoom.us/j/emergency
Next Update: 15:50 UTC (5 min)
```

### Status Updates (every 5 minutes)

```
🔄 UPDATE [15:50 UTC] - 5 MIN ELAPSED

Status: Still down
Root Cause: OOM kill identified (exit code 137)
Action: Increasing memory limit from 2G → 4G
ETA: 3 minutes (container recreating)
Impact: All users still affected

Next Update: 15:55 UTC
```

### Executive Summary (for leadership)

```
MAJOR INCIDENT UPDATE [15:50 UTC]

Service: ML Inference Gateway
Impact: CRITICAL - Complete Outage
Duration: 5 minutes so far
Affected Users: 100% (all users)

Root Cause: Out of memory (OOM) kill
Mitigation: Increasing container memory 2G → 4G

Financial Impact: Estimated $X/minute revenue loss
ETA to Recovery: 3-5 minutes

IC: @your-name
War Room: https://zoom.us/j/emergency
```

### Customer Communication (status page)

```
Investigating Major Outage - ML Inference API

We are currently experiencing a complete outage of the ML Inference API.
All prediction requests are failing. Our engineering team is actively
working on restoration.

Status: Identified - Memory exhaustion
ETA: 5-10 minutes

We will provide updates every 5 minutes.

Last Updated: 15:50 UTC
```

---

## Resolution

### Step 1: Verify Service Restored

```bash
# Health check
curl http://localhost:8000/health
# Expected: HTTP 200 {"status": "healthy"}

# Test inference endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[1,2,3,4]]}'
# Expected: HTTP 200 with prediction

# Check Prometheus scraping
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="inference-gateway")'
# Expected: health="up"
```

### Step 2: Monitor for Stability

**Watch for 10-15 minutes**:

```bash
# Monitor metrics
curl -s 'http://localhost:9090/api/v1/query?query=up{job="inference-gateway"}' | jq '.data.result[0].value[1]'
# Should consistently return "1"

# Monitor error rate
curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])' | jq

# Monitor memory usage
docker stats inference-gateway --no-stream
```

### Step 3: Calculate Impact

```bash
# Calculate downtime
# Start: 15:45 UTC
# End: 15:53 UTC
# Duration: 8 minutes

# Calculate requests affected
# Query Prometheus for request rate before outage:
curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{job="inference-gateway"}[5m])&time=2025-10-23T15:40:00Z' | jq

# Estimate total failed requests:
# Rate (req/sec) × Downtime (seconds) = Failed requests
```

### Step 4: Communicate Resolution

**Post in #incidents**:
```
✅ RESOLVED [15:53 UTC]

Incident Duration: 8 minutes
Root Cause: OOM kill due to memory leak in v1.2.3
Resolution: Increased memory limit 2G → 4G and restarted service

Service Status: OPERATIONAL
- Health checks: PASS
- Error rate: 0% (normal)
- Latency: Normal

Impact:
- Failed requests: ~2,400
- Affected users: All users (100%)
- SLO impact: 0.6% error budget consumed

Post-Mortem: Will be completed by EOD
Next Steps:
1. Investigate memory leak (assign @dev-lead)
2. Implement memory monitoring alert (assign @sre)
3. Add auto-restart on OOM (assign @platform)

Status: MONITORING (30-minute observation period)
```

### Step 5: Update Status Page

```
Resolved - ML Inference API Restored

The ML Inference API has been fully restored and is operating normally.

Root Cause: Memory resource exhaustion
Resolution: Increased memory allocation
Duration: 8 minutes (15:45-15:53 UTC)

We apologize for the inconvenience. A detailed post-mortem will be
published within 24 hours.

Status: Resolved
Last Updated: 15:55 UTC
```

---

## Post-Incident Actions

### Immediate (< 2 hours)

- [ ] **@oncall**: Complete incident report using template
- [ ] **@dev-lead**: Analyze memory usage trends leading to OOM
- [ ] **@sre**: Add memory usage alert (>85% threshold)
- [ ] **@platform**: Configure auto-restart on OOM

### Short-term (< 24 hours)

- [ ] **@dev-lead**: Fix memory leak in v1.2.3
- [ ] **@qa**: Add memory leak testing to CI/CD
- [ ] **@sre**: Implement graceful degradation on resource pressure
- [ ] **@oncall**: Conduct post-mortem meeting

### Long-term (< 1 week)

- [ ] **@platform**: Implement health-based auto-scaling
- [ ] **@sre**: Add resource pressure monitoring dashboard
- [ ] **@dev-lead**: Implement memory profiling in staging
- [ ] **@platform**: Set up multi-region failover

---

## Prevention

### Monitoring Improvements

```yaml
# Add to alerting_rules.yml
- alert: HighMemoryUsage
  expr: |
    (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.85
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Container {{ $labels.container }} high memory usage"
    description: "Memory usage at {{ $value | humanizePercentage }}"

- alert: CriticalMemoryUsage
  expr: |
    (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.95
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Container {{ $labels.container }} critical memory usage"
    description: "Memory usage at {{ $value | humanizePercentage }} - OOM imminent"
```

### Auto-Restart Configuration

```yaml
# Add to docker-compose.yml
services:
  inference-gateway:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
```

### Health Checks

```yaml
# Add to docker-compose.yml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 10s
  timeout: 5s
  retries: 3
  start_period: 30s
```

### Resource Monitoring

```bash
# Add to monitoring stack
# Enable cAdvisor for container metrics
# Add memory usage panels to Grafana
# Set up alerts for memory trends
```

---

## Common Failure Patterns

### Pattern 1: OOM Kill (Memory Exhaustion)

**Symptoms**:
- Exit code 137
- Sudden stop without graceful shutdown
- `dmesg` shows OOM killer

**Root Causes**:
- Memory leak in application
- Insufficient memory limit
- Memory-intensive request spike

**Resolution**:
- Increase memory limit
- Fix memory leak
- Add memory monitoring

### Pattern 2: Crash Loop (Repeated Failures)

**Symptoms**:
- Container continuously restarting
- Short uptime before crash
- Increasing restart count

**Root Causes**:
- Dependency unavailable (database, cache)
- Configuration error
- Port conflict

**Resolution**:
- Fix dependency availability
- Validate configuration
- Check port conflicts

### Pattern 3: Deployment Failure

**Symptoms**:
- New version fails to start
- Old version was healthy
- Started after deployment

**Root Causes**:
- Breaking code change
- Missing environment variable
- Incompatible dependency version

**Resolution**:
- Rollback to previous version
- Fix code issue
- Update configuration

### Pattern 4: Resource Contention

**Symptoms**:
- Service down during high load
- Other services also affected
- System resource exhaustion

**Root Causes**:
- Insufficient host resources
- No resource limits set
- Traffic spike

**Resolution**:
- Scale infrastructure
- Set resource limits
- Implement rate limiting

---

## Escalation Procedure

### When to Escalate

- Service not restored within 15 minutes
- Root cause not identified within 10 minutes
- Multiple services affected
- Data loss or corruption suspected

### Escalation Path

1. **Tier 1** (0-5 min): On-call engineer investigates
2. **Tier 2** (5-15 min): Escalate to service owner + SRE lead
3. **Tier 3** (15-30 min): Escalate to engineering manager + platform lead
4. **Executive** (30+ min): Notify VP Engineering + CTO

### Escalation Template

```
ESCALATION REQUEST

Incident: Inference Gateway Down
Duration: 15 minutes
Severity: P0
IC: @oncall-engineer

Situation:
- Service completely down since 15:45 UTC
- Restart/rollback attempts failed
- Root cause not identified

Actions Taken:
- Restarted service (failed)
- Rolled back deployment (failed)
- Checked logs and system resources

Request:
Need senior engineer support for deeper investigation
Possible database corruption or infrastructure issue

War Room: https://zoom.us/j/emergency
```

---

## Additional Resources

- **Service Repository**: https://github.com/company/inference-gateway
- **Deployment Docs**: docs/deployment/production.md
- **Architecture Diagram**: docs/architecture/system-design.md
- **Runbook Repository**: https://github.com/company/runbooks

---

## Runbook Metadata

**Last Updated**: 2025-10-23
**Owner**: SRE Team
**Reviewers**: ML Platform Team, DevOps Team
**Next Review**: 2025-11-23
**MTTR Goal**: <10 minutes
**Actual MTTR**: 8 minutes (average last 3 incidents)
**False Positive Rate**: <1% (highly reliable alert)

---

## Validation Checklist

After incident resolution, verify:

- [ ] Service health endpoint returns 200
- [ ] Prometheus shows target as "up"
- [ ] Inference endpoint accepts requests
- [ ] Error rate <1% for 10+ minutes
- [ ] Latency within normal range
- [ ] No error spikes in logs
- [ ] Memory/CPU usage normal
- [ ] All health checks passing
- [ ] SLO dashboard shows recovery
- [ ] Incident report created
- [ ] Post-mortem scheduled
