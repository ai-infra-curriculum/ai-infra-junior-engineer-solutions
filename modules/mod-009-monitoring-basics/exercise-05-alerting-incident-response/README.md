# Exercise 05: Alerting & Incident Response

**Module**: MOD-009 Monitoring & Logging Basics
**Exercise**: 05 of 05
**Difficulty**: Advanced
**Estimated Time**: 4-6 hours

---

## Overview

This exercise establishes a comprehensive alerting and incident response framework for the ML infrastructure platform. You'll learn how to design effective alerts, create actionable runbooks, manage error budgets, and respond to production incidents following industry best practices (SRE principles, MWMBR alerts, incident management).

### Learning Objectives

By completing this exercise, you will:

1. **Alert Design**: Understand symptom-based vs. cause-based alerts
2. **SLO-Based Alerting**: Implement Multi-Window Multi-Burn-Rate (MWMBR) alerts
3. **Runbook Creation**: Write actionable incident response runbooks
4. **Incident Management**: Follow structured incident response workflows
5. **Error Budget Policy**: Balance reliability with development velocity
6. **On-Call Operations**: Understand on-call responsibilities and best practices

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Incident Response Flow                       │
└─────────────────────────────────────────────────────────────────────┘

  Alert Fires (Prometheus)
         ↓
  Alertmanager (routing)
         ↓
  ┌──────────┬──────────┬──────────┐
  │          │          │          │
  v          v          v          v
PagerDuty  Slack    Email    Webhook
  │
  └─→ On-Call Engineer
         │
         ↓
   1. Acknowledge Alert (< 2 min for P0)
   2. Open Runbook
   3. Check Dashboard
   4. Follow Triage Steps
   5. Investigate (Logs, Traces, Metrics)
   6. Mitigate (Runbook Actions)
   7. Communicate (Status Updates)
   8. Resolve & Document
         │
         ↓
   Post-Incident Review
   - Incident Report
   - Root Cause Analysis
   - Action Items
   - Runbook Updates
```

---

## Directory Structure

```
exercise-05-alerting-incident-response/
├── README.md                           # This file
├── runbooks/                           # Incident response runbooks
│   ├── 001-high-error-rate.md         # High error rate (>5%) response
│   ├── 002-slo-burn-rate.md           # Fast burn rate (error budget) response
│   ├── 003-service-down.md            # Complete service outage response
│   ├── 004-resource-exhaustion.md     # CPU/memory/disk saturation response
│   └── 005-latency-degradation.md     # Elevated latency response
├── incidents/                          # Incident documentation
│   └── templates/
│       └── incident-template.md       # Post-mortem template
├── scripts/                            # Operational scripts
│   └── simulate-incident.sh           # Incident simulation for training
├── docs/                               # Policy documentation
│   └── policies/
│       ├── alerting-policy.md         # Alert design and management policy
│       ├── error-budget-policy.md     # Error budget and freeze policy
│       └── on-call-rotation.md        # On-call rotation and procedures
├── config/                             # Configuration files (from Exercise 02)
│   ├── prometheus/
│   │   ├── alerting_rules.yml         # Alert definitions
│   │   ├── recording_rules.yml        # SLO recording rules
│   │   └── prometheus.yml             # Prometheus configuration
│   └── alertmanager/
│       └── alertmanager.yml           # Alert routing configuration
└── COMPLETION_SUMMARY.md               # Exercise completion summary
```

---

## Prerequisites

Before starting this exercise, you should have completed:

- **Exercise 01**: Observability Foundations (Structured logging, tracing)
- **Exercise 02**: Prometheus Stack (Metrics collection, alerts)
- **Exercise 03**: Grafana Dashboards (Visualization, SLO dashboards)
- **Exercise 04**: Logging Pipeline (Log aggregation with Loki)

You should understand:
- Prometheus alerting rules and Alertmanager
- SLO-based monitoring (availability, latency)
- Grafana dashboards and data sources
- Distributed tracing with Jaeger
- Log aggregation with Loki

---

## Quick Start

### 1. Start Observability Stack

Ensure the complete stack from Exercises 01-04 is running:

```bash
# Navigate to Exercise 02 directory (Prometheus stack)
cd ../exercise-02-prometheus-stack

# Start the monitoring stack
docker-compose up -d

# Verify services are running
docker-compose ps

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Check Alertmanager status
curl http://localhost:9093/api/v2/status | jq
```

Expected services:
- Prometheus: http://localhost:9090
- Alertmanager: http://localhost:9093
- Grafana: http://localhost:3000
- Loki: http://localhost:3100
- Jaeger: http://localhost:16686
- Inference Gateway: http://localhost:8000

---

### 2. Review Alert Configuration

Check the alerting rules are loaded:

```bash
# View firing alerts
curl http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | {name: .labels.alertname, state: .state}'

# View alert rules
curl http://localhost:9090/api/v1/rules | jq '.data.groups[] | {name: .name, rules: [.rules[].name]}'
```

Expected alert rules:
- **Availability**: `ServiceDown`, `HighErrorRate`, `SLOAvailabilityFastBurn`, `SLOAvailabilitySlowBurn`
- **Latency**: `SLOLatencyP99Violation`, `HighLatency`
- **Resources**: `HighCPUUsage`, `HighMemoryUsage`, `DiskSpaceLow`
- **Model Quality**: `ModelDriftDetected`, `LowModelConfidence`

---

### 3. Simulate an Incident

Use the incident simulation script to practice incident response:

```bash
cd exercise-05-alerting-incident-response

# Make script executable
chmod +x scripts/simulate-incident.sh

# Run simulation (interactive menu)
./scripts/simulate-incident.sh
```

**Available scenarios**:
1. **High Error Rate**: Generate 5xx errors to trigger `HighErrorRate` alert
2. **Latency Spike**: Generate slow requests to trigger latency alerts
3. **Traffic Surge**: Generate high request volume
4. **Resource Exhaustion**: Stress CPU/memory to trigger resource alerts
5. **Complete Service Outage**: Stop the service (requires confirmation)
6. **SLO Fast Burn**: Sustained error rate to burn error budget fast

---

### 4. Practice Incident Response

**Workflow**:

1. **Alert Fires**: Simulate incident (e.g., scenario #1 High Error Rate)
2. **Acknowledge**: Observe alert in Prometheus: http://localhost:9090/alerts
3. **Open Runbook**: Read `runbooks/001-high-error-rate.md`
4. **Investigate**:
   - Check dashboard: http://localhost:3000/d/app-performance
   - Check logs: http://localhost:3000/explore (Loki)
   - Check traces: http://localhost:16686 (Jaeger)
5. **Mitigate**: Follow runbook mitigation steps
6. **Verify**: Confirm metrics return to normal
7. **Document**: Create incident report using template

---

## Runbook Guide

### Runbook 001: High Error Rate

**Triggers**: `HighErrorRate` alert (error rate >5% for 2 minutes)

**Symptoms**:
- Elevated 5xx errors
- Users experiencing failures
- SLO impact likely

**Common Causes**:
- Recent deployment
- Dependency failure
- Resource saturation
- Database issues

**Mitigation Actions**:
- Rollback recent deployment
- Scale resources
- Disable failing endpoint
- Circuit breaker activation

**MTTR Goal**: 15 minutes

**See**: `runbooks/001-high-error-rate.md`

---

### Runbook 002: SLO Fast Burn Rate

**Triggers**: `SLOAvailabilityFastBurn` alert (burn rate >14.4x in 1h AND 6h windows)

**Symptoms**:
- Error budget burning rapidly
- SLO breach imminent (within hours)
- Sustained elevated error rate

**Common Causes**:
- Deployment issues
- Sustained traffic spike
- Dependency degradation
- Resource leak

**Mitigation Actions**:
- Emergency rollback
- Traffic shedding
- Error budget freeze activation
- All-hands incident declaration

**MTTR Goal**: 30 minutes (critical urgency)

**See**: `runbooks/002-slo-burn-rate.md`

---

### Runbook 003: Service Down

**Triggers**: `ServiceDown` alert (service completely unreachable)

**Symptoms**:
- 100% error rate
- No successful health checks
- Prometheus cannot scrape metrics

**Common Causes**:
- Out of memory (OOM kill)
- Crash loop
- Configuration error
- Port conflict

**Mitigation Actions**:
- Restart service
- Recreate container
- Rollback version
- Fix configuration
- Increase resources

**MTTR Goal**: 10 minutes (P0 incident)

**See**: `runbooks/003-service-down.md`

---

### Runbook 004: Resource Exhaustion

**Triggers**: `HighCPUUsage` / `HighMemoryUsage` / `DiskSpaceLow`

**Symptoms**:
- CPU >85%, Memory >90%, or Disk >85%
- Degraded performance
- Potential service crash imminent

**Common Causes**:
- Traffic spike
- Resource leak
- Inefficient queries
- Log file growth

**Mitigation Actions**:
- Horizontal scaling
- Vertical scaling (increase limits)
- Rate limiting
- Cache clearing
- Kill resource-heavy processes

**MTTR Goal**: 20 minutes

**See**: `runbooks/004-resource-exhaustion.md`

---

### Runbook 005: Latency Degradation

**Triggers**: `SLOLatencyP99Violation` (P99 latency >300ms)

**Symptoms**:
- Slow response times
- User complaints
- Potential timeout errors

**Common Causes**:
- Database slow queries
- External dependency slowness
- Model inference bottleneck
- Resource contention
- Cache misses

**Mitigation Actions**:
- Query optimization
- Circuit breaker for slow dependencies
- Reduce model batch size
- Horizontal scaling
- Cache warming

**MTTR Goal**: 15 minutes

**See**: `runbooks/005-latency-degradation.md`

---

## Policy Documents

### Alerting Policy

Defines how alerts are designed, routed, and managed.

**Key Topics**:
- Alert severity levels (P0-P3)
- Alert design principles (symptom-based, actionable)
- Multi-Window Multi-Burn-Rate (MWMBR) methodology
- Alert routing and escalation
- Alert quality metrics
- Alert review process

**See**: `docs/policies/alerting-policy.md`

---

### Error Budget Policy

Defines how error budgets are calculated and what happens when they're depleted.

**Key Topics**:
- Error budget calculation (availability, latency)
- Error budget thresholds (healthy, warning, critical, emergency)
- Deployment freeze policy
- Error budget review process
- Exception request process

**Key Thresholds**:
- **>50%**: Normal operations, full velocity
- **30-50%**: Warning, elevated caution
- **10-30%**: Critical, partial deployment freeze
- **<10%**: Emergency, full deployment freeze

**See**: `docs/policies/error-budget-policy.md`

---

### On-Call Rotation Policy

Defines on-call responsibilities, rotation schedule, and compensation.

**Key Topics**:
- On-call schedule (1-week rotations, 24/7 coverage)
- Roles and responsibilities
- Escalation path
- Handoff procedures
- Compensation and time off
- Training and onboarding
- Best practices

**See**: `docs/policies/on-call-rotation.md`

---

## Incident Management Workflow

### 1. Alert Detection

**Automatic**:
- Prometheus evaluates alerting rules
- Alert fires when condition met
- Alertmanager receives alert
- Notification sent via PagerDuty, Slack, email

---

### 2. Acknowledgment

**On-Call Engineer**:
- Acknowledge alert in PagerDuty (within SLA)
- Post in #incidents channel
- Open relevant dashboard and runbook

**SLA by Severity**:
- P0 (Critical): 2 minutes
- P1 (High): 5 minutes
- P2 (Medium): 30 minutes

---

### 3. Triage (First 5 Minutes)

**Goals**:
- Understand impact (how many users affected?)
- Determine scope (which services affected?)
- Identify timeline (when did it start?)

**Actions**:
- Check service health dashboard
- Review error rate and latency metrics
- Check recent deployments
- Review logs for error patterns

---

### 4. Investigation (Next 10 Minutes)

**Goals**:
- Identify root cause
- Determine mitigation strategy

**Tools**:
- **Metrics**: Prometheus queries, Grafana dashboards
- **Logs**: Loki queries, log aggregation
- **Traces**: Jaeger distributed tracing
- **Resources**: Docker stats, system metrics

**Techniques**:
- Timeline analysis (when did it start?)
- Change correlation (what changed recently?)
- Dependency checks (are external services healthy?)
- Resource checks (CPU, memory, disk)

---

### 5. Mitigation

**Goals**:
- Restore service quickly
- Minimize user impact

**Common Actions**:
- Rollback deployment
- Scale resources
- Disable failing feature
- Activate circuit breaker
- Apply configuration change

**Priority**: Mitigation first, root cause later.

---

### 6. Communication

**Internal**:
- Post incident declaration in #incidents
- Status updates every 10 minutes (P0) or 30 minutes (P1)
- Post resolution message when fixed

**External** (if user-facing):
- Update status page
- Notify affected customers
- Executive summary for leadership

**Template**: See runbooks for communication templates.

---

### 7. Resolution

**Criteria**:
- Metrics returned to normal
- Service health checks passing
- Error rate <1% for 10+ minutes
- No new alerts firing

**Actions**:
- Monitor for stability (15 minutes)
- Calculate SLO impact
- Resolve alert in PagerDuty
- Post resolution message

---

### 8. Post-Incident Review

**Timeline**:
- Incident report: Within 24 hours
- Post-mortem meeting: Within 48 hours

**Deliverables**:
- Completed incident report (see template)
- Root cause analysis (Five Whys)
- Action items assigned
- Runbook updates
- Monitoring improvements

**Template**: `incidents/templates/incident-template.md`

**Blameless Culture**: Focus on systems and processes, not individuals.

---

## Testing and Practice

### 1. Incident Simulation Drills

Run quarterly drills to practice incident response:

```bash
# Simulate high error rate
./scripts/simulate-incident.sh
# Select: 1 (High Error Rate)

# Follow runbook 001-high-error-rate.md
# Practice triage, investigation, mitigation
# Document actions and timing

# Debrief:
# - What went well?
# - What needs improvement?
# - Were runbook steps clear?
# - Did tools work as expected?
```

---

### 2. Chaos Engineering

Intentionally inject failures to test resilience:

**Examples**:
- Kill random pods (test auto-restart)
- Inject latency to dependencies (test timeouts)
- Fill disk space (test monitoring alerts)
- Max out CPU (test resource alerts)

**Tools**:
- Chaos Monkey
- LitmusChaos
- Gremlin
- Manual (./scripts/simulate-incident.sh)

---

### 3. Runbook Validation

Regularly test runbooks:

1. Trigger alert artificially
2. Follow runbook step-by-step
3. Verify each command works
4. Measure time to resolution
5. Update runbook with learnings

**Schedule**: Quarterly per runbook

---

### 4. Alert Quality Review

Monthly review of alert quality:

```bash
# Get alert statistics for last 30 days
curl -s 'http://localhost:9090/api/v1/query?query=ALERTS' | jq

# Review:
# - Fire count per alert
# - False positive rate
# - Average time to resolution
# - Escalation rate

# Actions:
# - Tune noisy alerts
# - Remove dead alerts
# - Add missing alerts
# - Update thresholds
```

---

## Key Concepts

### Multi-Window Multi-Burn-Rate (MWMBR) Alerts

**Problem**: Simple threshold alerts cause alert fatigue (false positives).

**Solution**: Require alert condition in MULTIPLE time windows.

**Example**:
```yaml
- alert: SLOAvailabilityFastBurn
  expr: |
    slo:availability:burn_rate:1h > 14.4
    and
    slo:availability:burn_rate:6h > 14.4
  for: 2m
```

**Benefits**:
- 1-hour window catches recent changes
- 6-hour window filters temporary blips
- Both must trigger → reduces false positives by 90%+

**Reference**: Google SRE Workbook Chapter 5

---

### Error Budget

The **maximum allowed unreliability** while still meeting SLO.

**Formula**:
```
Error Budget = 100% - SLO Target
```

**Example** (99.5% availability SLO):
```
Error Budget = 100% - 99.5% = 0.5%
Monthly (30 days) = 216 minutes allowed downtime
```

**Usage**:
- **>50% remaining**: Deploy freely
- **30-50% remaining**: Extra caution
- **10-30% remaining**: Partial freeze (critical fixes only)
- **<10% remaining**: Full freeze

---

### Symptom-Based Alerts

**Prefer**: Alert on user-visible symptoms

**Example** (Good):
```yaml
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
  annotations:
    summary: "Users experiencing elevated error rate"
```

**Avoid**: Alert on internal component states

**Example** (Bad):
```yaml
- alert: ProcessDown
  expr: up{job="api"} == 0
```

**Why**: Process might auto-restart with zero user impact. Alert on what users experience.

---

### Blameless Post-Mortems

**Focus on**:
- What happened (timeline)
- Why it happened (root cause, contributing factors)
- What can prevent it (action items)

**Avoid**:
- Blaming individuals
- Focusing on who made the mistake
- Punishment or reprimands

**Goal**: Learn and improve systems, not punish people.

---

## Integration with Previous Exercises

### Exercise 01: Observability Foundations

- **Structured logs** provide context during incidents
- **Trace IDs** link logs, metrics, and traces for root cause analysis
- **OpenTelemetry** instrumentation enables distributed tracing

**Usage in Incident Response**:
- Search logs by trace_id during investigation
- View full request context in Jaeger
- Correlate errors across services

---

### Exercise 02: Prometheus Stack

- **Alerting rules** define when to page on-call
- **Recording rules** pre-compute SLO metrics
- **Prometheus** is the source of truth for metrics

**Usage in Incident Response**:
- Check Prometheus dashboard for firing alerts
- Query metrics to understand impact
- View alert history for patterns

---

### Exercise 03: Grafana Dashboards

- **SLO dashboard** shows error budget status
- **Application performance dashboard** shows error rate, latency
- **Infrastructure dashboard** shows resource usage

**Usage in Incident Response**:
- Open relevant dashboard immediately
- Visualize trends (sudden spike vs gradual)
- Drill down to specific timeframes

---

### Exercise 04: Logging Pipeline

- **Loki** aggregates logs from all services
- **LogQL** queries find relevant logs quickly
- **PII redaction** ensures compliance

**Usage in Incident Response**:
- Query logs by service, level, or trace_id
- Identify error patterns
- Reconstruct request flow

---

## Common Mistakes to Avoid

### 1. Alert Fatigue

**Symptom**: Too many alerts, engineers ignore them

**Causes**:
- Alerting on non-actionable conditions
- Thresholds too sensitive
- No false positive review process

**Fix**:
- Monthly alert quality review
- Remove or tune noisy alerts
- Increase `for` duration to filter transients

---

### 2. Missing Runbooks

**Symptom**: Alerts fire but engineers don't know what to do

**Causes**:
- No runbook link in alert
- Runbook outdated or vague
- Runbook not tested

**Fix**:
- Every alert must have runbook_url
- Test runbooks quarterly
- Update after each incident

---

### 3. No Error Budget Discipline

**Symptom**: Deployments continue during low error budget

**Causes**:
- Error budget policy not enforced
- Engineers unaware of budget status
- No freeze process

**Fix**:
- Automated freeze announcements
- Error budget dashboard visible to all
- Clear escalation for freeze exceptions

---

### 4. Poor Communication During Incidents

**Symptom**: Stakeholders don't know incident status

**Causes**:
- Infrequent status updates
- Updates lack details
- Wrong communication channels

**Fix**:
- Set update cadence (every 10 min for P0)
- Use templates for consistency
- Post in #incidents channel (not DMs)

---

### 5. Skipping Post-Mortems

**Symptom**: Same incidents repeat

**Causes**:
- No time allocated for post-mortem
- Action items not tracked
- Learnings not shared

**Fix**:
- Schedule post-mortem within 48h
- Assign action items with owners
- Share post-mortem widely

---

## Advanced Topics

### 1. Automated Incident Response

**Tools**:
- Kubernetes auto-scaling (HPA)
- Circuit breakers (automatic failover)
- Auto-rollback on alert

**Example** (auto-rollback):
```yaml
# ArgoCD Rollout with automatic rollback
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: inference-gateway
spec:
  analysis:
    - templateName: error-rate-check
      args:
        - name: error-rate-threshold
          value: "0.05"
  strategy:
    canary:
      canaryService: inference-gateway-canary
      stableService: inference-gateway-stable
      analysis:
        failureLimit: 1  # Rollback after 1 failed analysis
```

---

### 2. Multi-Region Incident Response

**Challenges**:
- Incidents may affect specific regions
- Failover to backup region
- Global vs regional SLOs

**Strategies**:
- Regional runbooks
- Automated traffic shifting
- Regional on-call rotations

---

### 3. Incident Command System (ICS)

For major incidents, adopt formal ICS roles:

- **Incident Commander (IC)**: Coordinates response
- **Operations Lead**: Executes mitigation actions
- **Communications Lead**: Manages stakeholder updates
- **Planning Lead**: Documents incident, tracks action items

**When to use**: P0 incidents lasting >30 minutes

---

## Troubleshooting

### Alerts Not Firing

```bash
# Check Prometheus is scraping targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'

# Check alert rules are loaded
curl http://localhost:9090/api/v1/rules | jq

# Check Alertmanager is reachable
curl http://localhost:9093/api/v2/status

# Check alert routing configuration
docker exec prometheus cat /etc/prometheus/prometheus.yml | grep alertmanagers -A 5
```

---

### Alert Fires But No Notification

```bash
# Check Alertmanager received the alert
curl http://localhost:9093/api/v2/alerts | jq

# Check routing rules
docker exec alertmanager cat /etc/alertmanager/alertmanager.yml

# Check receiver configuration (PagerDuty, Slack, etc.)
# Verify API keys are correct
```

---

### Runbook Commands Fail

```bash
# Check you have access to required systems
docker ps  # Can you access Docker?
kubectl get pods  # Can you access Kubernetes?

# Check VPN is connected
ping <internal-service>

# Check you have required permissions
# May need to be added to ops group or granted AWS/GCP access
```

---

## Additional Resources

### Books
- **Site Reliability Engineering** (Google SRE Book)
  - Chapter 6: Monitoring Distributed Systems
  - Chapter 14: Managing Incidents
- **The Site Reliability Workbook** (Google SRE Workbook)
  - Chapter 5: Alerting on SLOs
  - Chapter 9: Incident Response

### Online Courses
- **PagerDuty University**: Incident Response Training
- **Google Cloud Coursera**: Site Reliability Engineering
- **Linux Foundation**: Certified Kubernetes Administrator

### Tools Documentation
- [Prometheus Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)
- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)
- [Grafana Alerting](https://grafana.com/docs/grafana/latest/alerting/)
- [PagerDuty Incident Response](https://response.pagerduty.com/)

---

## Success Criteria

You have successfully completed this exercise when you can:

- [ ] Understand the difference between symptom-based and cause-based alerts
- [ ] Explain Multi-Window Multi-Burn-Rate (MWMBR) alerting
- [ ] Follow a runbook to mitigate a simulated incident in <15 minutes
- [ ] Calculate error budget remaining and explain deployment freeze policy
- [ ] Create an incident report using the post-mortem template
- [ ] Identify alert quality issues (false positives, missing alerts)
- [ ] Communicate incident status effectively to stakeholders
- [ ] Explain on-call responsibilities and escalation procedures

---

## Next Steps

After completing this exercise:

1. **Practice**: Run monthly incident simulation drills
2. **Customize**: Adapt runbooks and policies to your organization
3. **Expand**: Add runbooks for additional failure modes
4. **Automate**: Implement automated incident response where possible
5. **Share**: Train team members on incident response procedures

---

## Feedback and Improvements

This is a living curriculum. If you find issues or have suggestions:

1. Document learnings after real incidents
2. Update runbooks with new troubleshooting steps
3. Add missing scenarios to simulation script
4. Improve alert thresholds based on experience
5. Share post-mortems for peer learning

---

## Congratulations! 🎉

You've completed all 5 exercises in Module 009: Monitoring & Logging Basics. You now have a production-ready observability stack with metrics, logs, traces, dashboards, alerts, and incident response procedures.

**You've built**:
- Exercise 01: Structured logging and distributed tracing
- Exercise 02: Prometheus metrics collection and alerting
- Exercise 03: Grafana dashboards for visualization
- Exercise 04: Centralized logging with Loki
- Exercise 05: Alerting and incident response framework

**Total**: 80+ files, 20,000+ lines of production-ready code and documentation!

**Next Module**: MOD-010 Infrastructure as Code (Terraform/Pulumi)
