# Step-by-Step Implementation Guide: Alerting & Incident Response

## Overview

Implement production alerting and incident response! Learn alert design, on-call rotation, runbooks, incident management, and postmortems.

**Time**: 2 hours | **Difficulty**: Intermediate to Advanced

---

## Learning Objectives

✅ Design effective alerts
✅ Configure multi-channel alerting
✅ Create runbooks
✅ Implement on-call rotation
✅ Manage incidents
✅ Conduct postmortems
✅ Reduce alert fatigue

---

## Alert Design Principles

### Good Alert Characteristics
- Actionable
- Urgent
- Real impact on users
- Novel (not repetitive)

### Alert Thresholds

```yaml
# Good: Based on SLO
alert: SLOViolation
expr: |
  (
    sum(rate(http_requests_total{status!~"5.."}[5m])) /
    sum(rate(http_requests_total[5m]))
  ) < 0.999
for: 5m

# Bad: Static threshold
alert: HighCPU
expr: cpu_usage > 80
```

---

## Multi-Channel Alerting

### Slack Integration

```yaml
# alertmanager.yaml
receivers:
- name: 'slack-critical'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
    channel: '#incidents'
    title: '🚨 CRITICAL: {{ .GroupLabels.alertname }}'
    text: |
      *Alert:* {{ .GroupLabels.alertname }}
      *Severity:* {{ .GroupLabels.severity }}
      *Description:* {{ .CommonAnnotations.description }}
      *Runbook:* {{ .CommonAnnotations.runbook_url }}
    send_resolved: true
```

### PagerDuty Integration

```yaml
receivers:
- name: 'pagerduty'
  pagerduty_configs:
  - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
    description: '{{ .GroupLabels.alertname }}'
    details:
      severity: '{{ .GroupLabels.severity }}'
      description: '{{ .CommonAnnotations.description }}'
      runbook_url: '{{ .CommonAnnotations.runbook_url }}'
```

---

## Runbooks

### Example Runbook

```markdown
# Runbook: High Error Rate

## Alert
**Alert Name:** HighErrorRate
**Severity:** Critical
**SLO Impact:** Yes

## Symptoms
- Error rate > 5% for 5 minutes
- Users experiencing 500 errors
- Possible impact on SLA

## Investigation

### 1. Check Recent Deployments
\`\`\`bash
kubectl rollout history deployment/ml-api
\`\`\`

### 2. Check Logs
\`\`\`bash
kubectl logs -l app=ml-api --tail=100 | grep ERROR
\`\`\`

### 3. Check Dependencies
\`\`\`bash
# Database
kubectl exec -it postgres-0 -- pg_isready
# Redis
kubectl exec -it redis-0 -- redis-cli ping
\`\`\`

### 4. Check Metrics
- CPU/Memory usage
- Request rate
- Database connections

## Resolution

### Option 1: Rollback Recent Deployment
\`\`\`bash
kubectl rollout undo deployment/ml-api
\`\`\`

### Option 2: Scale Up
\`\`\`bash
kubectl scale deployment ml-api --replicas=10
\`\`\`

### Option 3: Restart Pods
\`\`\`bash
kubectl rollout restart deployment/ml-api
\`\`\`

## Escalation
If not resolved in 15 minutes:
- Escalate to: @ml-platform-team
- War room: #incident-room
- Incident commander: On-call SRE

## Postmortem
- Create incident ticket
- Schedule postmortem within 48 hours
- Document root cause
```

---

## On-Call Rotation

### PagerDuty Schedule

```python
import pdpyras

session = pdpyras.APISession('YOUR_API_KEY')

# Create schedule
schedule = session.rpost('/schedules', json={
    'schedule': {
        'name': 'ML Platform On-Call',
        'time_zone': 'America/Los_Angeles',
        'schedule_layers': [{
            'name': 'Weekly Rotation',
            'start': '2024-01-01T00:00:00',
            'rotation_virtual_start': '2024-01-01T00:00:00',
            'rotation_turn_length_seconds': 604800,  # 1 week
            'users': [
                {'user': {'id': 'USER1', 'type': 'user_reference'}},
                {'user': {'id': 'USER2', 'type': 'user_reference'}},
                {'user': {'id': 'USER3', 'type': 'user_reference'}}
            ]
        }]
    }
})
```

---

## Incident Management

### Incident Severity Levels

```
SEV-1 (Critical)
- Total service outage
- Data loss
- Security breach
- Response: Immediate, 24/7

SEV-2 (High)
- Partial outage
- Degraded performance
- SLO violation
- Response: 1 hour

SEV-3 (Medium)
- Minor degradation
- Non-critical features affected
- Response: Next business day

SEV-4 (Low)
- Cosmetic issues
- Nice-to-have features
- Response: Planned maintenance
```

### Incident Response Workflow

```yaml
1. Acknowledge
   - Accept page
   - Join incident channel

2. Assess
   - Determine severity
   - Check impact
   - Review recent changes

3. Mitigate
   - Follow runbook
   - Implement temporary fix
   - Communicate status

4. Resolve
   - Verify fix
   - Monitor metrics
   - Close incident

5. Document
   - Create postmortem
   - Document timeline
   - Identify action items
```

---

## Postmortem Template

```markdown
# Postmortem: [Incident Title]

**Date:** YYYY-MM-DD
**Duration:** X hours
**Severity:** SEV-X
**Impact:** X users affected

## Summary
Brief description of the incident.

## Timeline (UTC)
- 14:32 - Alert triggered
- 14:35 - On-call responded
- 14:40 - Root cause identified
- 14:45 - Fix deployed
- 14:50 - Service recovered
- 15:00 - Incident closed

## Root Cause
Detailed explanation of what caused the incident.

## Impact
- Users affected: 1,000
- Requests failed: 50,000
- Revenue impact: $X
- SLO budget consumed: 0.5%

## Detection
How was the incident detected?

## Response
What actions were taken?

## Resolution
How was the incident resolved?

## Lessons Learned
### What Went Well
- Quick detection
- Effective runbook

### What Didn't Go Well
- Slow rollback
- Missing monitoring

## Action Items
- [ ] Add monitoring for X (Owner: @person, Due: DATE)
- [ ] Update runbook (Owner: @person, Due: DATE)
- [ ] Implement circuit breaker (Owner: @person, Due: DATE)

## Timeline
Detailed timeline of events.
```

---

## Best Practices

✅ Alert on symptoms, not causes
✅ Make alerts actionable
✅ Provide runbooks for all alerts
✅ Implement on-call rotation
✅ Conduct blameless postmortems
✅ Track action items
✅ Review and update runbooks
✅ Measure MTTR and MTTD
✅ Reduce alert fatigue
✅ Use severity levels

---

**Alerting & Incident Response mastered!** 🚨
