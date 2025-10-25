# Runbook: High Error Rate Alert

**Alert Name**: `HighErrorRate`
**Severity**: Critical
**Page On-Call**: Yes
**SLO Impact**: High (directly impacts availability SLO)

---

## Alert Description

The inference gateway is experiencing an elevated error rate (>5% of requests returning 5xx errors over the last 5 minutes). This directly threatens our 99.5% availability SLO and indicates a serious service degradation.

### Alert Query

```promql
(
  sum(rate(http_requests_total{service="inference-gateway", status=~"5.."}[5m]))
  /
  sum(rate(http_requests_total{service="inference-gateway"}[5m]))
) * 100 > 5
```

### Thresholds

- **Warning**: >1% error rate for 5 minutes
- **Critical**: >5% error rate for 2 minutes (pages on-call)
- **SLO Breach**: Error rate sustained at >0.5% over 30 days

---

## Triage (First 5 Minutes)

### Step 1: Acknowledge the Alert

```bash
# In PagerDuty: Click "Acknowledge"
# In Slack: React with 👀 emoji to indicate you're investigating
```

**Why**: Lets team know someone is handling the incident and stops escalation.

### Step 2: Check Service Status

**Dashboard**: http://localhost:3000/d/app-performance

```bash
# Quick health check
curl http://localhost:8000/health

# Check if service is responding
curl -I http://localhost:8000/
```

**Expected**: Service should return HTTP 200.
**If service is down**: Skip to "Service Completely Down" section below.

### Step 3: Determine Error Scope

**Grafana Query**:
```promql
# Error rate by endpoint
sum(rate(http_requests_total{service="inference-gateway", status=~"5.."}[5m])) by (endpoint)
```

**Questions to answer**:
- Is this affecting all endpoints or specific ones?
- Is this a sudden spike or gradual increase?
- What's the current error rate vs. baseline?

### Step 4: Check Recent Changes

```bash
# Check for recent deployments
kubectl rollout history deployment/inference-gateway

# Check recent commits
git log --since="1 hour ago" --oneline

# Check for configuration changes
git log --since="1 hour ago" --oneline -- config/
```

**Common causes**:
- Recent deployment
- Configuration change
- Infrastructure change
- Upstream dependency failure

---

## Investigation (Next 10 Minutes)

### Step 5: Examine Error Logs

**Loki Query** (in Grafana Explore):
```logql
{container="inference-gateway"}
  | json
  | status_code >= 500
  | line_format "{{.timestamp}} [{{.status_code}}] {{.method}} {{.endpoint}} - {{.message}}"
```

**Look for patterns**:
- Specific error messages (e.g., "Connection refused", "Timeout", "Out of memory")
- Specific endpoints failing
- Error clustering around specific time

### Step 6: Check Resource Utilization

**CPU Usage**:
```promql
container:cpu_usage:percent{container_label_com_docker_compose_service="inference-gateway"}
```

**Memory Usage**:
```promql
container:memory_usage:percent{container_label_com_docker_compose_service="inference-gateway"}
```

**Thresholds**:
- CPU >90%: Likely resource saturation
- Memory >95%: Risk of OOM kills

### Step 7: Check Dependencies

**Database Health**:
```bash
# Check database connections
curl http://localhost:8000/ready | jq '.dependencies'
```

**External Services**:
```bash
# Check model service availability
curl http://model-service:8080/health

# Check feature store
curl http://feature-store:8081/health
```

### Step 8: Review Traces

**Find slow or failed traces**:

1. Go to Jaeger UI: http://localhost:16686
2. Service: `inference-gateway`
3. Tags: `error=true`
4. Lookback: Last 15 minutes
5. Find traces with errors

**Analyze**:
- Where in the request flow is it failing?
- Are external calls timing out?
- Is the model inference step failing?

---

## Mitigation Actions

### Option 1: Rollback Recent Deployment

**If error rate started after recent deployment**:

```bash
# Rollback to previous version
kubectl rollout undo deployment/inference-gateway

# Watch rollback progress
kubectl rollout status deployment/inference-gateway

# Verify error rate drops
# Check dashboard after 2-3 minutes
```

### Option 2: Scale Up Resources

**If CPU/Memory saturation**:

```bash
# Scale horizontally
kubectl scale deployment/inference-gateway --replicas=10

# Or scale vertically (requires restart)
kubectl set resources deployment/inference-gateway \
  --limits=cpu=2000m,memory=4Gi

# Watch for pods to be ready
kubectl get pods -l app=inference-gateway -w
```

### Option 3: Disable Problematic Feature

**If specific endpoint is failing**:

```bash
# Set feature flag to disable endpoint
kubectl set env deployment/inference-gateway \
  FEATURE_ENDPOINT_XYZ_ENABLED=false

# Or update ConfigMap
kubectl edit configmap inference-gateway-config
```

### Option 4: Circuit Breaker Activation

**If external dependency is failing**:

```bash
# Enable circuit breaker for failing dependency
kubectl set env deployment/inference-gateway \
  CIRCUIT_BREAKER_THRESHOLD=5 \
  CIRCUIT_BREAKER_TIMEOUT=60
```

---

## Communication

### Internal Communication (Slack)

**Initial Notification** (post in #incidents):
```
🔥 INCIDENT: High Error Rate - Inference Gateway
Severity: P1 (Critical)
Incident Commander: @your-name
Started: 2025-10-23 14:23 UTC
Current Status: Investigating

Error Rate: 12% (target: <0.5%)
Dashboard: http://localhost:3000/d/app-performance
Zoom bridge: https://zoom.us/j/123456789

Updates will be posted here every 10 minutes.
```

**Status Update** (every 10 minutes):
```
🔄 UPDATE [14:33 UTC]
Status: Mitigation in progress
Action: Rolling back deployment v1.2.3 → v1.2.2
ETA: 5 minutes
Error Rate: 8% (decreasing)
```

**Resolution Notification**:
```
✅ RESOLVED [14:45 UTC]
Incident Duration: 22 minutes
Root Cause: NPE in v1.2.3 prediction handler
Mitigation: Rollback to v1.2.2
Error Rate: 0.3% (normal)

Post-mortem: Will be completed by EOD
```

### External Communication (Status Page)

**If user-facing**:
```
Investigating - We're investigating elevated error rates on the ML inference API.
Users may experience slower response times or failures. We're actively working on a fix.

Last Updated: 14:25 UTC
```

---

## Service Completely Down

### Immediate Actions

**Step 1: Check container status**:
```bash
docker ps | grep inference-gateway
docker logs inference-gateway --tail=50
```

**Step 2: Check for crash loop**:
```bash
# If container keeps restarting
docker events --filter 'container=inference-gateway'
```

**Step 3: Restart service**:
```bash
docker-compose restart inference-gateway

# Or force recreate
docker-compose up -d --force-recreate inference-gateway
```

**Step 4: Check startup errors**:
```bash
# Watch logs during startup
docker-compose logs -f inference-gateway
```

### Common Failure Modes

**Out of Memory**:
```bash
# Check for OOM kills
dmesg | grep -i kill

# Increase memory limit
docker-compose down
# Edit docker-compose.yml: increase memory limit
docker-compose up -d
```

**Configuration Error**:
```bash
# Validate configuration
docker-compose config

# Check environment variables
docker-compose exec inference-gateway env | grep -i config
```

**Port Conflict**:
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill conflicting process or change port
```

---

## Resolution & Cleanup

### Step 1: Verify Metrics

**Check error rate has normalized**:
```promql
sum(rate(http_requests_total{service="inference-gateway", status=~"5.."}[5m])) / sum(rate(http_requests_total{service="inference-gateway"}[5m])) * 100
```

**Target**: <1% error rate for 10+ minutes

### Step 2: Check SLO Impact

**Availability SLO**:
```promql
slo:availability:ratio_rate30d
```

**Error Budget**:
```promql
slo:availability:error_budget_remaining
```

**Calculate burn**: How much error budget was consumed during incident?

### Step 3: Document Actions Taken

Create incident file: `incidents/2025-01/incident-YYYYMMDD-HHmm-high-error-rate.md`

Include:
- Timeline of events
- Actions taken
- What worked / didn't work
- Relevant logs, traces, metrics screenshots

### Step 4: Resolve PagerDuty Alert

```bash
# In PagerDuty: Click "Resolve"
# Add resolution notes
```

### Step 5: Schedule Post-Mortem

```bash
# Create calendar invite within 48 hours
# Invite: Incident Commander, on-call engineer, service owner, observability lead
# Prep: Share incident doc before meeting
```

---

## Prevention

### Monitoring Gaps Identified

- [ ] Add pre-deployment integration tests
- [ ] Add canary deployment with automatic rollback
- [ ] Add rate limiting to prevent cascading failures
- [ ] Add synthetic monitoring for critical paths

### Code Changes Needed

- [ ] Improve error handling in prediction handler
- [ ] Add circuit breaker for external dependencies
- [ ] Implement graceful degradation for ML model failures

### Process Improvements

- [ ] Update deployment checklist
- [ ] Add rollback procedure to deployment docs
- [ ] Create pre-flight checks script

---

## Additional Resources

- **Dashboard**: http://localhost:3000/d/app-performance
- **Logs**: http://localhost:3000/explore (Loki)
- **Traces**: http://localhost:16686 (Jaeger)
- **Runbook Repo**: https://github.com/company/runbooks
- **On-Call Guide**: https://wiki.company.com/oncall

---

## Runbook Metadata

**Last Updated**: 2025-10-23
**Owner**: ML Platform Team
**Reviewers**: SRE Team
**Next Review**: 2025-11-23
**Incident Count**: 3 (last 30 days)
**MTTR**: 18 minutes (average)
