# Step-by-Step Implementation Guide: Alerting & Incident Response

## Overview

Implement a comprehensive alerting and incident response framework following SRE best practices: SLO-based MWMBR alerts, actionable runbooks, error budget management, and structured incident response.

**Time**: 3-4 hours | **Difficulty**: Advanced

---

## Phase 1: Alert Design Principles (30 minutes)

### Step 1: Define Alert Categories

**Symptom-Based Alerts** (GOOD):
```yaml
# User-facing issue
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.01

# Latency degradation
- alert: HighLatency
  expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 0.5
```

**Cause-Based Alerts** (AVOID for pages):
```yaml
# Infrastructure issue (use for warnings, not pages)
- alert: HighCPU
  expr: node_cpu_usage > 0.9
  labels:
    severity: warning  # NOT critical
```

**Actionable Alerts** (Required):
- Clear summary: "What is wrong?"
- Runbook link: "How to fix?"
- Context: Request rate, error rate, affected services

---

## Phase 2: MWMBR SLO Alerts (45 minutes)

### Step 2: Implement Multi-Window Multi-Burn-Rate Alerts

**`config/prometheus/alerting_rules.yml`**:

```yaml
groups:
  - name: slo_alerts
    interval: 15s
    rules:
      # === CRITICAL: Fast Burn (Page On-Call) ===
      - alert: SLOAvailabilityFastBurn
        expr: |
          (
            slo:availability:burn_rate:1h > 14.4
            and
            slo:availability:burn_rate:6h > 14.4
          )
        for: 2m
        labels:
          severity: critical
          slo: availability
          runbook: https://runbooks.company.com/slo-fast-burn
        annotations:
          summary: "Fast SLO burn detected - Page On-Call"
          description: |
            Availability burn rate: {{ $value }}x normal
            Current availability: {{ printf "%.2f" (query "slo:availability:ratio_rate5m * 100") }}%
            Error budget will be exhausted in ~2 hours at this rate

            **Immediate Actions**:
            1. Check recent deployments/changes
            2. Review error logs for spike
            3. Check infrastructure health (CPU, memory, network)
            4. Consider rollback if recent deploy

            Runbook: https://runbooks.company.com/slo-fast-burn

      # === WARNING: Slow Burn (Slack Alert) ===
      - alert: SLOAvailabilitySlowBurn
        expr: |
          (
            slo:availability:burn_rate:3d > 6
            and
            slo:availability:burn_rate:7d > 6
          )
        for: 15m
        labels:
          severity: warning
          slo: availability
          runbook: https://runbooks.company.com/slo-slow-burn
        annotations:
          summary: "Slow SLO burn detected - Investigate"
          description: |
            Availability burn rate: {{ $value }}x normal over 3 days
            Current availability: {{ printf "%.2f" (query "slo:availability:ratio_rate30d * 100") }}%
            Error budget will be exhausted in ~5 days at this rate

            **Actions**:
            1. Review error trends over past 3 days
            2. Identify recurring error patterns
            3. Create ticket to investigate root cause
            4. Schedule fix in upcoming sprint

            Runbook: https://runbooks.company.com/slo-slow-burn

      # === CRITICAL: Error Budget Low ===
      - alert: ErrorBudgetCritical
        expr: slo:availability:error_budget_remaining < 0.1
        for: 5m
        labels:
          severity: critical
          slo: error_budget
        annotations:
          summary: "Error budget critically low - Freeze launches"
          description: |
            Error budget remaining: {{ printf "%.1f" (mul $value 100) }}%

            **Policy Actions**:
            1. FREEZE all non-critical feature launches
            2. Focus on reliability improvements
            3. Conduct incident review
            4. Create recovery plan

            Error budget policy: https://wiki.company.com/error-budget-policy
```

---

## Phase 3: Alertmanager Configuration (40 minutes)

### Step 3: Configure Alert Routing

**`config/alertmanager/alertmanager.yml`**:

```yaml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

# Alert routing tree
route:
  receiver: 'default'
  group_by: ['alertname', 'service', 'severity']
  group_wait: 30s        # Wait before sending first notification
  group_interval: 5m     # Wait before sending new alerts in group
  repeat_interval: 4h    # Resend if not resolved

  routes:
    # === CRITICAL ALERTS → Page + Slack ===
    - match:
        severity: critical
      receiver: pagerduty-critical
      continue: true  # Also send to Slack

    - match:
        severity: critical
      receiver: slack-critical

    # === WARNING ALERTS → Slack only ===
    - match:
        severity: warning
      receiver: slack-warnings
      group_wait: 5m  # Longer wait for warnings

    # === ML Platform Team ===
    - match:
        team: ml-platform
      receiver: slack-ml-platform

    # === Infrastructure Team ===
    - match:
        team: infrastructure
      receiver: slack-infrastructure

# Inhibition rules (suppress dependent alerts)
inhibit_rules:
  # Critical suppresses warning for same alert
  - source_match:
      severity: critical
    target_match:
      severity: warning
    equal: ['alertname', 'service']

  # Fast burn suppresses slow burn
  - source_match:
      alertname: SLOAvailabilityFastBurn
    target_match:
      alertname: SLOAvailabilitySlowBurn
    equal: ['service']

# Notification receivers
receivers:
  - name: 'default'
    slack_configs:
      - channel: '#alerts-default'
        title: "{{ .GroupLabels.alertname }}"
        text: "{{ range .Alerts }}{{ .Annotations.description }}{{ end }}"

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: "{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}"
        details:
          firing: "{{ .Alerts.Firing | len }}"
          resolved: "{{ .Alerts.Resolved | len }}"

  - name: 'slack-critical'
    slack_configs:
      - channel: '#alerts-critical'
        color: 'danger'
        title: "🚨 CRITICAL: {{ .GroupLabels.alertname }}"
        text: |
          {{ range .Alerts }}
          *Summary*: {{ .Annotations.summary }}

          *Details*:
          {{ .Annotations.description }}

          *Runbook*: {{ .Labels.runbook }}
          {{ end }}

  - name: 'slack-warnings'
    slack_configs:
      - channel: '#alerts-warnings'
        color: 'warning'
        title: "⚠️  WARNING: {{ .GroupLabels.alertname }}"

  - name: 'slack-ml-platform'
    slack_configs:
      - channel: '#ml-platform-alerts'

  - name: 'slack-infrastructure'
    slack_configs:
      - channel: '#infrastructure-alerts'
```

---

## Phase 4: Runbook Creation (45 minutes)

### Step 4: Create Actionable Runbooks

**`docs/runbooks/slo-fast-burn.md`**:

```markdown
# Runbook: SLO Fast Burn Alert

**Alert**: `SLOAvailabilityFastBurn`
**Severity**: CRITICAL
**On-Call Response Time**: 15 minutes

---

## 1. Immediate Triage (First 5 minutes)

### Check Recent Deployments
```bash
# List recent deployments
kubectl get events -n production --sort-by='.lastTimestamp' | grep Deployment | head -10

# Check current deployment status
kubectl rollout status deployment/inference-gateway -n production
```

**If recent deployment (< 1 hour ago)**:
→ **ACTION**: Initiate rollback immediately

```bash
kubectl rollout undo deployment/inference-gateway -n production
kubectl rollout status deployment/inference-gateway -n production
```

### Check Error Spike
```promql
# In Prometheus/Grafana
rate(http_requests_total{status=~"5.."}[5m])
```

**If error rate spike**:
→ **ACTION**: Proceed to Step 2 (Error Investigation)

### Check Infrastructure Health
```bash
# CPU/Memory
kubectl top nodes
kubectl top pods -n production

# Network
ping prometheus.company.com
curl -I https://api.company.com/health
```

**If infrastructure issue**:
→ **ACTION**: Page infrastructure on-call, proceed to Step 3

---

## 2. Error Investigation (Next 10 minutes)

### Check Error Logs
```bash
# Grafana Explore (Loki)
{container="inference-gateway"} | json | level="ERROR" | line_format "{{.message}}"

# Or kubectl
kubectl logs -n production deployment/inference-gateway --tail=100 | grep ERROR
```

### Common Error Patterns

#### Database Connection Errors
```
Error: connection timeout to postgres
```
**Action**:
1. Check database health: `kubectl get pods -n data | grep postgres`
2. Check connection pool: `psql -h postgres -c "SELECT count(*) FROM pg_stat_activity"`
3. Restart database proxy if needed

#### Model Loading Failures
```
Error: failed to load model from S3
```
**Action**:
1. Check S3 connectivity: `aws s3 ls s3://models/`
2. Check IAM permissions: `aws sts get-caller-identity`
3. Verify model file exists

#### Dependency Service Down
```
Error: failed to call feature-service
```
**Action**:
1. Check dependency health: `curl http://feature-service/health`
2. Page dependent service on-call if down
3. Enable circuit breaker if repeated failures

---

## 3. Mitigation Actions

### Option 1: Rollback Recent Change
```bash
kubectl rollout undo deployment/inference-gateway -n production
# Monitor for 5 minutes, verify error rate drops
```

### Option 2: Scale Up Resources
```bash
# If CPU/memory saturated
kubectl scale deployment/inference-gateway --replicas=10 -n production
```

### Option 3: Disable Problematic Feature
```bash
# Feature flag
kubectl set env deployment/inference-gateway FEATURE_X_ENABLED=false -n production
```

### Option 4: Failover to Backup
```bash
# Switch traffic to backup cluster
kubectl patch service inference-gateway -p '{"spec":{"selector":{"version":"backup"}}}'
```

---

## 4. Verification (After 10 minutes)

### Check SLO Recovery
```promql
# Burn rate should drop below 14.4
slo:availability:burn_rate:1h

# Availability should improve
slo:availability:ratio_rate5m * 100
```

### Check Error Rate
```promql
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])
# Should be < 1%
```

---

## 5. Communication

### Update Incident Channel
```
#incident-2024-001

[RESOLVED] SLO Fast Burn Alert
- Root Cause: Recent deployment introduced database connection leak
- Mitigation: Rolled back deployment v1.2.3 → v1.2.2
- Error rate dropped from 5% → 0.1%
- SLO burn rate: 14.4x → 1.2x (normal)
- Error budget consumed: 2% (remaining: 48%)

Next Steps:
- Post-incident review scheduled for tomorrow
- Fix PR created: https://github.com/company/app/pull/123
```

---

## 6. Post-Incident (Within 24 hours)

1. **Create Incident Report**: https://wiki.company.com/incidents/2024-001
2. **Schedule Post-Mortem**: Invite team, stakeholders
3. **Identify Action Items**: Prevention, detection, mitigation improvements
4. **Update Runbook**: Add new patterns discovered
```

---

## Phase 5: Error Budget Policy (30 minutes)

### Step 5: Implement Error Budget Decision-Making

**`docs/policies/error-budget-policy.md`**:

```markdown
# Error Budget Policy

## Error Budget Thresholds

| Error Budget Remaining | Actions |
|------------------------|---------|
| > 75% | **Normal Operations** - Proceed with feature launches |
| 50-75% | **Caution** - Review error trends, reduce launch velocity |
| 25-50% | **Alert** - Freeze non-critical launches, focus on reliability |
| 10-25% | **Critical** - Freeze ALL launches except critical fixes |
| < 10% | **Lockdown** - All hands on reliability, incident declared |

## Automated Actions

**When error budget < 25%**:
1. Automated Slack message to #eng-leadership
2. Pause CI/CD deployments (except hotfixes)
3. Create high-priority reliability ticket
4. Schedule emergency SRE review

**When error budget < 10%**:
1. Page SRE lead and Engineering Director
2. Declare incident (severity: P1)
3. Convene war room
4. Halt all non-emergency changes

## Budget Reset

- Error budget resets monthly (1st of month)
- Partial resets granted for:
  - Platform-wide outages (not service-specific)
  - Valid measurement bugs
  - Pre-approved maintenance windows
```

---

## Phase 6: Testing Alert Delivery (30 minutes)

### Step 6: Test End-to-End Alert Flow

```bash
# 1. Fire test alert
curl -X POST http://localhost:9093/api/v1/alerts -d '[
  {
    "labels": {
      "alertname": "TestAlert",
      "severity": "critical"
    },
    "annotations": {
      "summary": "Test alert - please acknowledge"
    }
  }
]'

# 2. Verify in Alertmanager UI
open http://localhost:9093/#/alerts

# 3. Check Slack channel
# Should receive message in #alerts-critical

# 4. Create silence
curl -X POST http://localhost:9093/api/v2/silences -d '{
  "matchers": [{"name": "alertname", "value": "TestAlert", "isRegex": false}],
  "startsAt": "2025-01-01T00:00:00Z",
  "endsAt": "2025-01-01T01:00:00Z",
  "createdBy": "test@company.com",
  "comment": "Test silence"
}'

# 5. Verify alert is silenced
```

---

## Validation

### Alert Checklist

- [ ] Alert fires when condition met
- [ ] Alert resolves when condition clears
- [ ] PagerDuty notification received (critical)
- [ ] Slack notification received (all severities)
- [ ] Runbook link is accessible
- [ ] Silence functionality works
- [ ] Inhibition rules work (critical suppresses warning)
- [ ] Group wait/interval configured correctly

---

## Summary

**What You Built**:
- ✅ SLO-based MWMBR alerts (fast burn + slow burn)
- ✅ Multi-channel alerting (PagerDuty + Slack)
- ✅ Actionable runbooks with triage steps
- ✅ Error budget policy with automated actions
- ✅ Alert routing and inhibition rules
- ✅ Incident response workflow

**Best Practices Implemented**:
- Symptom-based alerting (not cause-based)
- Actionable alerts (clear next steps)
- Runbook links in every alert
- Multi-window burn rate (reduce false positives)
- Error budget-driven decision making
- Structured incident response

**On-Call Readiness**:
- Response time: 15 minutes for critical
- Escalation path: Engineer → Lead → Director
- Runbooks for top 10 alert types
- Post-incident review process
- Continuous improvement loop
