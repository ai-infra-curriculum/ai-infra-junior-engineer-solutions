# Runbook: SLO Fast Burn Rate Alert

**Alert Name**: `SLOAvailabilityFastBurn`
**Severity**: Critical
**Page On-Call**: Yes
**SLO Impact**: Extreme (entire error budget at risk)

---

## Alert Description

The inference gateway is consuming error budget at 14.4x the normal rate. At this pace, the entire monthly error budget will be exhausted in approximately 2 hours, violating our 99.5% availability SLO.

This is a **multi-window multi-burn-rate (MWMBR) alert** designed to catch SLO violations early while minimizing false positives.

### Alert Query

```promql
slo:availability:burn_rate:1h > 14.4
and
slo:availability:burn_rate:6h > 14.4
```

**Translation**: Error rate is 14.4x higher than tolerable for BOTH:
- Last 1 hour
- Last 6 hours

### What This Means

- **Normal error budget consumption**: 0.5% per month = 0.0167% per day
- **Current burn rate**: 14.4x = 0.24% per day
- **Time to exhaust budget**: ~2 hours if sustained
- **SLO Breach ETA**: ~48 hours if not mitigated

---

## Triage (First 3 Minutes)

### Step 1: Acknowledge Immediately

**This is the highest-severity alert**. Acknowledge within 2 minutes to prevent auto-escalation.

```bash
# PagerDuty: Click "Acknowledge"
# Slack: Post in #incidents immediately
```

### Step 2: Check Current Error Budget

**Grafana Dashboard**: http://localhost:3000/d/slo-overview

```promql
# Error budget remaining (0-1, where 1 = 100%)
slo:availability:error_budget_remaining

# Monthly availability (target: 99.5%)
slo:availability:ratio_rate30d
```

**Evaluate severity**:
- Budget >80% remaining: Not yet critical, but trending bad
- Budget 50-80%: Concerning, immediate action needed
- Budget <50%: Critical, deployment freeze likely needed
- Budget <10%: Emergency, error budget freeze activated

### Step 3: Identify Error Pattern

```promql
# Error rate over last 6 hours
sum(rate(http_requests_total{service="inference-gateway", status=~"5.."}[5m]))
/
sum(rate(http_requests_total{service="inference-gateway"}[5m]))
* 100
```

**Compare to**:
- 1 hour ago
- 6 hours ago
- Same time yesterday

**Pattern types**:
- **Sudden spike**: Recent deployment or incident
- **Gradual increase**: Resource exhaustion, memory leak
- **Intermittent**: Dependency flakiness, retry storms

---

## Investigation (Next 10 Minutes)

### Step 4: Check for Active Incidents

```bash
# Check if other alerts are firing
curl http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing")'
```

**Common co-alerts**:
- `HighErrorRate`: Confirms elevated errors
- `HighCPUUsage`: Resource saturation
- `HighMemoryUsage`: Memory pressure
- `ServiceDown`: Complete outage

**If no co-alerts**: This may be a low-rate but sustained error condition.

### Step 5: Analyze Burn Rate Trend

```promql
# 1-hour burn rate
slo:availability:burn_rate:1h

# 6-hour burn rate
slo:availability:burn_rate:6h

# 3-day burn rate
slo:availability:burn_rate:3d
```

**Interpretation**:
- 1h >> 6h: Recent spike, may self-resolve
- 1h ≈ 6h: Sustained problem, action required
- 6h >> 3d: Problem started in last few hours
- All windows high: Long-running issue

### Step 6: Identify Root Cause Category

**Use logs to categorize**:

```logql
# Count errors by type
sum(count_over_time({container="inference-gateway"} |= "ERROR" [1h])) by (error_type)
```

**Categories**:
1. **4xx Client Errors**: Bad requests from clients (not SLO impacting usually)
2. **5xx Server Errors**: Service failures (SLO impacting)
3. **Timeouts**: Latency causing failures
4. **Dependency Failures**: Upstream service issues

### Step 7: Calculate Impact

**Queries**:

```promql
# Requests affected in last hour
sum(increase(http_requests_total{service="inference-gateway", status=~"5.."}[1h]))

# Total requests
sum(increase(http_requests_total{service="inference-gateway"}[1h]))

# Error budget consumed in last hour
(1 - slo:availability:burn_rate:1h) * 100
```

**Document**:
- Failed requests: X
- Affected users: ~Y (estimate)
- Error budget consumed: Z%

---

## Mitigation Strategies

### Strategy 1: Emergency Rollback

**If error rate started after recent deployment**:

```bash
# Check recent deployments
git log --since="6 hours ago" --oneline

# Immediate rollback
kubectl rollout undo deployment/inference-gateway

# Verify burn rate decreasing
# Wait 5 minutes, then check:
curl -s 'http://localhost:9090/api/v1/query?query=slo:availability:burn_rate:1h' | jq '.data.result[0].value[1]'
```

**Expected**: Burn rate should drop below 14.4 within 5-10 minutes.

### Strategy 2: Traffic Shedding

**If overload causing failures**:

```bash
# Enable rate limiting
kubectl set env deployment/inference-gateway \
  RATE_LIMIT_ENABLED=true \
  RATE_LIMIT_RPS=100

# Or implement graceful degradation
kubectl set env deployment/inference-gateway \
  GRACEFUL_DEGRADATION=true
```

**Trade-off**: May reject some requests, but prevents total service failure.

### Strategy 3: Isolate Failing Component

**If specific endpoint causing errors**:

```bash
# Disable problematic endpoint
kubectl set env deployment/inference-gateway \
  ENDPOINT_PREDICT_V2_ENABLED=false

# Or redirect traffic
kubectl patch service inference-gateway -p '{"spec":{"selector":{"version":"v1"}}}'
```

### Strategy 4: Scale Resources

**If resource exhaustion**:

```bash
# Horizontal scaling
kubectl scale deployment/inference-gateway --replicas=20

# Check if burn rate decreases
# Wait 3-5 minutes for new pods to be ready
```

### Strategy 5: Error Budget Freeze

**If error budget critically low (<10%)**:

```bash
# Announce freeze in #engineering
# Post in Slack:
🚨 ERROR BUDGET FREEZE 🚨
Inference Gateway error budget: 8% remaining
All non-critical deployments FROZEN until budget recovers.
Only emergency fixes allowed.
Duration: Until error budget > 50%
```

**Freeze actions**:
- Halt all deployments
- Cancel non-critical changes
- Focus on reliability improvements
- Daily error budget reviews

---

## Communication

### Critical Incident Declaration

**Post in #incidents**:
```
🔴 CRITICAL INCIDENT: SLO Fast Burn - Inference Gateway
Severity: P0 (All-hands)
Incident Commander: @your-name
Started: 2025-10-23 15:45 UTC

Error Budget: 12% remaining (was 65% 2h ago)
Burn Rate: 14.4x (critical threshold)
Impact: ~5,000 failed requests in last hour

Dashboard: http://localhost:3000/d/slo-overview
War Room: Zoom https://zoom.us/j/123456789

All hands on deck. Stand by for assignments.
```

### Executive Summary (for leadership)

```
INCIDENT SUMMARY [15:50 UTC]

Service: ML Inference Gateway
Impact: HIGH - SLO breach imminent
Status: Actively mitigating

Error Budget: 12% remaining (burned 53% in 2 hours)
At current rate: Budget exhausted in 30 minutes

Actions Taken:
• Rolled back deployment v1.2.3 → v1.2.2
• Scaled replicas 5 → 15
• Enabled circuit breaker for Model API

Next Update: 16:00 UTC (10 minutes)
```

---

## Resolution

### Step 1: Verify Burn Rate Normalized

**Wait 15 minutes after mitigation**, then check:

```promql
# Should be <2.0 (ideally <1.0)
slo:availability:burn_rate:1h

# Should be decreasing
slo:availability:burn_rate:6h
```

### Step 2: Project Error Budget Recovery

```promql
# Current error budget
slo:availability:error_budget_remaining

# If no further incidents, how long to recover to 80%?
# Depends on remaining days in month and current burn rate
```

### Step 3: Determine Post-Incident Actions

**If error budget <20%**:
- Error budget freeze remains in effect
- Daily status meetings
- Focus on reliability improvements

**If error budget 20-50%**:
- Partial freeze (only critical changes)
- Incident review required before new deployments

**If error budget >50%**:
- Resume normal operations
- Schedule post-mortem

---

## Prevention

### Detection Improvements

- [ ] Add intermediate burn rate alert (6x threshold) for earlier warning
- [ ] Add slow burn alert (3-day window) for chronic issues
- [ ] Implement pre-deployment SLO impact prediction

### Mitigation Automation

- [ ] Automated rollback on SLO fast burn
- [ ] Automated traffic shedding at burn rate >10x
- [ ] Circuit breaker auto-activation

### Process Changes

- [ ] Mandatory canary deployment for changes
- [ ] SLO review required in deployment checklist
- [ ] Error budget discussion in weekly SRE sync

---

## Understanding Multi-Window Multi-Burn-Rate

### Why This Alert Design?

**Traditional alerts**: Simple threshold on error rate
- Problem: False positives on temporary blips
- Problem: Slow to detect chronic issues

**MWMBR**: Validates across multiple time windows
- **1-hour window**: Catches rapid changes
- **6-hour window**: Filters out temporary blips
- **Must fire in BOTH**: Reduces false positives by 90%+

### Burn Rate Calculation

```
Burn Rate = (Current Error Rate) / (Acceptable Error Rate)

Acceptable Error Rate = (1 - SLO Target) = 1 - 0.995 = 0.005 = 0.5%

If current error rate = 7.2%:
Burn Rate = 7.2% / 0.5% = 14.4x
```

**Interpretation**: Consuming error budget 14.4x faster than sustainable.

### Time to Exhaustion

```
Error Budget Duration = 30 days (monthly SLO)
Time to Exhaustion = Budget Duration / Burn Rate
                   = 30 days / 14.4
                   = 2.08 days
                   = ~50 hours
```

But we trigger alert when burn rate detected in **1 hour**, giving us 49 hours to fix.

---

## Additional Resources

- **Google SRE Workbook**: Chapter 5 - Alerting on SLOs
- **SLO Dashboard**: http://localhost:3000/d/slo-overview
- **Error Budget Policy**: docs/policies/error-budget-policy.md
- **Deployment Freeze Process**: docs/policies/deployment-freeze.md

---

## Runbook Metadata

**Last Updated**: 2025-10-23
**Owner**: SRE Team
**Reviewers**: ML Platform Team Lead
**Next Review**: 2025-11-23
**False Positive Rate**: <2% (well-tuned)
**Mean Time to Resolve**: 32 minutes
