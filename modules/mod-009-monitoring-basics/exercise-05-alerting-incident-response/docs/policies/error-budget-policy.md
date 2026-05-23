# Error Budget Policy

**Version**: 1.0
**Last Updated**: 2025-10-23
**Owner**: SRE Team
**Reviewers**: ML Platform Team, Product Management
**Next Review**: 2025-11-23

---

## Table of Contents

1. [Overview](#overview)
2. [Error Budget Fundamentals](#error-budget-fundamentals)
3. [Error Budget Calculation](#error-budget-calculation)
4. [Error Budget Freeze Policy](#error-budget-freeze-policy)
5. [Deployment Guidelines](#deployment-guidelines)
6. [Error Budget Review Process](#error-budget-review-process)
7. [Roles and Responsibilities](#roles-and-responsibilities)

---

## Overview

This document defines the error budget policy for the ML infrastructure platform. The error budget quantifies how much unreliability is acceptable in exchange for development velocity. This policy establishes rules for managing the error budget to balance innovation with reliability.

### Purpose

- **Balance velocity and reliability**: Allow fast iteration while maintaining user trust
- **Data-driven decisions**: Objective criteria for deployment freezes
- **Shared responsibility**: Development and operations share reliability goals
- **Proactive risk management**: Prevent SLO breaches before they occur

### Scope

This policy applies to all production services with defined SLOs:
- ML Inference Gateway
- Feature Store
- Model Registry
- Training Infrastructure
- API Gateway

---

## Error Budget Fundamentals

### What is an Error Budget?

The **error budget** is the maximum amount of unreliability allowed while still meeting your SLO.

**Formula**:
```
Error Budget = 100% - SLO Target
```

**Example**:
- SLO: 99.5% availability
- Error Budget: 0.5% allowed unreliability
- Monthly (30 days): 216 minutes of downtime allowed
- Daily: 7.2 minutes of downtime allowed

### Why Error Budgets Matter

**Traditional approach** (no error budget):
- Every outage is a failure
- Risk-averse culture
- Slow innovation
- "Zero downtime" is impossible and expensive

**Error budget approach**:
- Some unreliability is expected and budgeted
- Enables calculated risk-taking
- Fast deployment when budget is healthy
- Automatic brake when budget is low
- Objective decision-making

---

## Error Budget Calculation

### Availability Error Budget

**SLO**: 99.5% availability (success rate)

**Calculation**:
```promql
# Success ratio (0-1, where 1 = 100%)
slo:availability:ratio_rate30d =
  sum(rate(http_requests_total{status=~"2..|3.."}[30d]))
  /
  sum(rate(http_requests_total[30d]))

# Error budget remaining (0-1)
slo:availability:error_budget_remaining =
  1 - ((1 - slo:availability:ratio_rate30d) / (1 - 0.995))

# As percentage (0-100%)
slo:availability:error_budget_remaining * 100
```

**Interpretation**:
- **100% budget remaining**: No failures this month (perfect)
- **50% budget remaining**: Used half of allowed failures (healthy)
- **0% budget remaining**: SLO breached (critical)
- **Negative budget**: Over budget, SLO violated

**Example**:
```
SLO Target: 99.5%
Actual: 99.7%
Error Budget Used: 40%
Error Budget Remaining: 60%

Interpretation: Healthy. Can deploy with normal velocity.
```

### Latency Error Budget

**SLO**: 95% of requests complete in <300ms

**Calculation**:
```promql
# Latency compliance ratio
slo:latency:ratio_rate30d =
  sum(rate(http_request_duration_seconds_bucket{le="0.3"}[30d]))
  /
  sum(rate(http_request_duration_seconds_count[30d]))

# Error budget remaining
slo:latency:error_budget_remaining =
  1 - ((1 - slo:latency:ratio_rate30d) / (1 - 0.95))
```

---

## Error Budget Freeze Policy

### Error Budget Thresholds

| Error Budget Remaining | Status | Action |
|------------------------|--------|--------|
| **>50%** | Healthy | Normal operations, full velocity |
| **30-50%** | Warning | Elevated caution, optional review |
| **10-30%** | Critical | Partial freeze, critical fixes only |
| **<10%** | Emergency | Full freeze, all hands on reliability |

---

### Healthy: >50% Budget Remaining

**Status**: ✅ Normal operations

**Deployment Policy**:
- ✅ Feature releases: Approved
- ✅ Experiments: Approved
- ✅ Refactoring: Approved
- ✅ Infrastructure changes: Approved

**Requirements**:
- Standard code review process
- Automated tests passing
- Deployment approval by service owner

**Example Message**:
```
📊 Error Budget Status: HEALTHY (72%)
Status: Normal operations
Deployments: All approved
Next Review: Weekly
```

---

### Warning: 30-50% Budget Remaining

**Status**: ⚠️ Elevated caution

**Deployment Policy**:
- ✅ Critical bug fixes: Approved
- ✅ Feature releases: Approved with extra review
- ⚠️ Experiments: Requires justification
- ⚠️ Refactoring: Defer unless high-value
- ⚠️ Infrastructure changes: Requires SRE approval

**Requirements**:
- Senior engineer review required
- Rollback plan documented
- Incremental rollout (canary → 50% → 100%)
- Extended monitoring period (24h)

**Action Items**:
- Daily error budget review
- Investigate root causes of recent errors
- Identify reliability improvements

**Example Message**:
```
⚠️  Error Budget Status: WARNING (42%)
Status: Elevated caution
Deployments: Extra review required
Action: Daily error budget sync at 10am
Next Review: Daily until >50%
```

---

### Critical: 10-30% Budget Remaining

**Status**: 🔴 Partial freeze

**Deployment Policy**:
- ✅ P0 hotfixes (production outages): Approved
- ✅ P1 critical bugs: Approved with SRE sign-off
- ❌ Feature releases: BLOCKED
- ❌ Experiments: BLOCKED
- ❌ Refactoring: BLOCKED
- ❌ Non-critical changes: BLOCKED

**Requirements**:
- SRE team must approve all deployments
- Incident Commander assigned
- War room open
- Hourly error budget monitoring
- All teams notified

**Mandatory Actions**:
- **Immediate**: Post freeze announcement in #engineering
- **Within 2 hours**: Incident review meeting
- **Within 24 hours**: Reliability action plan created
- **Daily**: Error budget standup

**Example Message**:
```
🔴 ERROR BUDGET FREEZE: CRITICAL (18%)

Status: PARTIAL FREEZE
Budget Remaining: 18% (was 42% yesterday)
Projected Depletion: 3 days at current rate

DEPLOYMENT FREEZE:
❌ Feature releases: BLOCKED
❌ Experiments: BLOCKED
✅ P0/P1 hotfixes: Approved with SRE sign-off only

Actions:
• War room: https://zoom.us/j/emergency
• Daily standup: 9am, 3pm (until >30%)
• All teams: Focus on reliability improvements

Incident Commander: @sre-lead
Next Review: In 6 hours
```

---

### Emergency: <10% Budget Remaining

**Status**: 🚨 Full freeze

**Deployment Policy**:
- ✅ Production outage hotfixes ONLY: Requires VP Engineering approval
- ❌ Everything else: BLOCKED

**Requirements**:
- VP Engineering must approve any change
- Multiple senior engineers review
- Detailed rollback plan
- Real-time monitoring during deployment

**Mandatory Actions**:
- **Immediate**:
  - All-hands freeze announcement
  - Cancel all planned deployments
  - Engineering manager notified
  - VP Engineering notified
- **Within 1 hour**:
  - Emergency incident review
  - Identify top 3 reliability issues
  - Assign action items to teams
- **Within 4 hours**:
  - Detailed recovery plan created
  - Communication to stakeholders
- **Every 6 hours**:
  - Executive status update

**Example Message**:
```
🚨 ERROR BUDGET FREEZE: EMERGENCY (7%)

Status: FULL FREEZE
Budget Remaining: 7%
SLO Breach: IMMINENT (projected 18 hours)

ALL DEPLOYMENTS BLOCKED
Exception: Production outage hotfixes only (VP approval required)

Recent Incidents (last 24h):
1. 15:30 UTC - HighErrorRate (12%) - 18min - Cost 4% budget
2. 19:45 UTC - ServiceDown (100%) - 8min - Cost 6% budget
3. 02:10 UTC - LatencySpike (P99=2s) - 22min - Cost 3% budget

Top Actions (PRIORITY):
1. Fix memory leak in v1.2.3 (@dev-lead)
2. Add database read replica (@platform)
3. Implement circuit breaker (@sre)

War Room: https://zoom.us/j/emergency (ACTIVE 24/7)
IC: @sre-lead
Next Update: Every 6 hours
```

---

## Deployment Guidelines

### Pre-Deployment Checklist

Before any production deployment:

- [ ] Check current error budget status
- [ ] Verify deployment is allowed under current budget status
- [ ] Required approvals obtained
- [ ] Automated tests passing
- [ ] Rollback plan documented
- [ ] Monitoring plan defined
- [ ] On-call engineer notified

### Deployment Risk Assessment

**Low Risk** (allowed with >10% budget):
- Configuration changes (non-breaking)
- Static content updates
- Documentation updates
- Logging changes

**Medium Risk** (allowed with >30% budget):
- Feature releases (gradual rollout)
- Database schema changes (backwards-compatible)
- Dependency updates (minor versions)
- Performance optimizations

**High Risk** (allowed with >50% budget):
- Major refactoring
- Infrastructure changes
- Database migrations (breaking)
- Dependency updates (major versions)
- Architecture changes

### Deployment Process by Budget Status

#### >50% Budget: Standard Process

```bash
1. Merge PR to main branch
2. Automated CI/CD pipeline deploys
3. Canary deployment (5%)
4. Monitor for 15 minutes
5. Gradual rollout (5% → 25% → 50% → 100%)
6. Monitor for 1 hour post-deployment
```

#### 30-50% Budget: Enhanced Process

```bash
1. Merge PR with senior engineer approval
2. Manual deployment trigger (no auto-deploy)
3. Canary deployment (5%)
4. Monitor for 30 minutes
5. Gradual rollout (5% → 10% → 25% → 50% → 100%)
6. Monitor for 4 hours post-deployment
7. SRE sign-off required before 100%
```

#### 10-30% Budget: Emergency Process

```bash
1. Merge PR with SRE + VP Engineering approval
2. Manual deployment (no automation)
3. Canary deployment (2%)
4. Monitor for 1 hour
5. Gradual rollout (2% → 5% → 10% → 25% → 50%)
6. Pause at 50% for 24 hours
7. Monitor for 24 hours before full rollout
8. Multiple approvals at each stage
```

---

## Error Budget Review Process

### Weekly Review (Normal Operations)

**When**: Every Monday 10am
**Attendees**: Service owners, SRE on-call
**Duration**: 15 minutes

**Agenda**:
1. Review current error budget (all services)
2. Review incidents from past week
3. Identify trends (improving/degrading)
4. Assign action items if budget <50%

**Output**: Email summary to team

---

### Daily Review (Budget <50%)

**When**: Every day 10am
**Attendees**: Service owners, SRE team, engineering leads
**Duration**: 30 minutes

**Agenda**:
1. Error budget status (current and trend)
2. Incident review (last 24 hours)
3. Reliability action items progress
4. Deployment requests review
5. Forecast: Days to budget recovery or depletion

**Output**: Slack status update in #incidents

---

### Emergency Review (Budget <10%)

**When**: Every 6 hours (24/7)
**Attendees**: All hands (service owners, SRE, engineering managers, VP)
**Duration**: 1 hour

**Agenda**:
1. Critical error budget status
2. Recent incident deep-dive
3. Reliability action plan review
4. Resource allocation decisions
5. Stakeholder communication

**Output**: Executive summary email + Slack update

---

## Roles and Responsibilities

### Service Owner

**Responsibilities**:
- Monitor error budget for their service
- Approve deployments within policy guidelines
- Lead incident response for their service
- Drive reliability improvements when budget low

**Authority**:
- Approve deployments when budget >30%
- Escalate to SRE when budget 10-30%
- Request freeze exception (with justification)

---

### SRE Team

**Responsibilities**:
- Monitor error budget across all services
- Enforce freeze policy
- Approve critical deployments during partial freeze
- Facilitate error budget review meetings

**Authority**:
- Declare error budget freeze
- Approve/reject deployments during freeze
- Escalate to VP Engineering for policy exceptions

---

### Engineering Manager

**Responsibilities**:
- Ensure team awareness of error budget status
- Prioritize reliability work when budget low
- Balance feature velocity with reliability
- Communicate with stakeholders

**Authority**:
- Approve resource allocation for reliability
- Escalate to VP for freeze exceptions

---

### VP Engineering / CTO

**Responsibilities**:
- Final authority on freeze exceptions
- Strategic reliability decisions
- Stakeholder communication (customers, board)

**Authority**:
- Override freeze policy (with documentation)
- Approve high-risk deployments during emergency freeze

---

## Error Budget Recovery Strategies

### Short-term (During Freeze)

1. **Stop the bleeding**: Identify and fix recent issues
2. **Defer deployments**: Focus on stability
3. **Increase monitoring**: Watch for new issues
4. **Quick wins**: Low-risk reliability improvements

### Medium-term (Post-Freeze)

1. **Root cause analysis**: Investigate major incidents
2. **Technical debt paydown**: Fix known reliability issues
3. **Testing improvements**: Add tests for failure modes
4. **Monitoring gaps**: Add alerts for blind spots

### Long-term (Preventive)

1. **Architecture improvements**: Reduce single points of failure
2. **Capacity planning**: Right-size infrastructure
3. **Chaos engineering**: Proactive failure testing
4. **Process improvements**: Update runbooks, improve deploy process

---

## Exception Process

### Requesting a Freeze Exception

**When**: You need to deploy during a freeze due to business-critical reasons

**Process**:
1. Create exception request ticket
2. Document business justification
3. Assess risk and mitigation plan
4. Get approval from required level:
   - 10-30% budget: SRE + Engineering Manager
   - <10% budget: VP Engineering

**Exception Request Template**:
```markdown
## Freeze Exception Request

**Requester**: @name
**Service**: inference-gateway
**Change**: Deploy v1.2.4 (critical security patch)

**Business Justification**:
- CVE-2025-1234 high-severity vulnerability
- Customer contractual requirement
- Revenue impact: $X/day

**Risk Assessment**:
- Change complexity: Low (single file change)
- Blast radius: Isolated to authentication module
- Rollback time: <5 minutes

**Mitigation Plan**:
- Deploy to 1% canary
- Monitor for 2 hours
- Rollback trigger: Any error rate increase
- On-call engineer dedicated to monitoring

**Approvals**:
- [ ] Service Owner: @owner
- [ ] SRE Lead: @sre-lead
- [ ] Engineering Manager: @em
- [ ] VP Engineering: @vp (if <10% budget)
```

---

## Appendix A: Error Budget Calculation Examples

### Example 1: Availability SLO

**Service**: Inference Gateway
**SLO**: 99.5% availability (monthly)
**Time Period**: 30 days = 43,200 minutes

**Allowed Downtime**:
```
Error Budget = (1 - 0.995) × 43,200 minutes
             = 0.005 × 43,200
             = 216 minutes
             = 3 hours 36 minutes per month
```

**Incident Impact**:
- 8-minute outage = 3.7% of monthly budget
- 30-minute outage = 13.9% of monthly budget
- 2-hour outage = 55.6% of monthly budget

### Example 2: Latency SLO

**Service**: Inference Gateway
**SLO**: 95% of requests <300ms
**Traffic**: 1,000 requests/minute = 43.2M requests/month

**Allowed Slow Requests**:
```
Error Budget = (1 - 0.95) × 43.2M
             = 0.05 × 43.2M
             = 2.16M requests per month
```

**Incident Impact**:
- 10 minutes at 10% slow (100 req/min slow) = 0.05% of budget
- 1 hour at 50% slow (500 req/min slow) = 1.4% of budget
- 1 day at 10% slow = 6.7% of budget

---

## Appendix B: Error Budget Dashboard

**Grafana Dashboard**: `http://localhost:3000/d/error-budget`

**Panels**:
1. Current error budget remaining (gauge)
2. Error budget trend (30 days)
3. Error budget burn rate (current)
4. Days until budget exhaustion (projection)
5. Major incidents (annotations)
6. Deployment velocity vs budget

---

## Document Change Log

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-10-23 | 1.0 | SRE Team | Initial policy creation |

---

## Approval

**Policy Owner**: SRE Team Lead
**Approved By**: VP of Engineering, CTO
**Effective Date**: 2025-10-23
**Next Review**: 2025-11-23

---

## References

- [Google SRE Book - Embracing Risk](https://sre.google/sre-book/embracing-risk/)
- [Google SRE Workbook - SLO Engineering](https://sre.google/workbook/implementing-slos/)
- [Alerting Policy](./alerting-policy.md)
- [On-Call Rotation](./on-call-rotation.md)
