# On-Call Rotation Policy

**Version**: 1.0
**Last Updated**: 2025-10-23
**Owner**: SRE Team
**Reviewers**: Engineering Management, HR
**Next Review**: 2025-11-23

---

## Table of Contents

1. [Overview](#overview)
2. [On-Call Schedule](#on-call-schedule)
3. [Roles and Responsibilities](#roles-and-responsibilities)
4. [Escalation Path](#escalation-path)
5. [Handoff Procedures](#handoff-procedures)
6. [Compensation and Time Off](#compensation-and-time-off)
7. [Training and Onboarding](#training-and-onboarding)
8. [On-Call Best Practices](#on-call-best-practices)

---

## Overview

This document defines the on-call rotation policy for the ML infrastructure platform, establishing expectations, responsibilities, and procedures for engineers participating in the on-call rotation.

### Purpose

- **Ensure coverage**: 24/7 coverage for production incidents
- **Fair distribution**: Equitable rotation among qualified engineers
- **Response quality**: Well-rested, well-equipped engineers respond to incidents
- **Work-life balance**: Sustainable on-call practices

### Scope

This policy applies to:
- ML Platform Engineers
- SRE Team
- Backend Engineers (secondary escalation)

---

## On-Call Schedule

### Rotation Schedule

**Primary On-Call**:
- **Duration**: 1 week (Monday 9am PT → Monday 9am PT)
- **Frequency**: Approximately every 6-8 weeks per engineer
- **Team Size**: 6-8 engineers in rotation

**Secondary On-Call**:
- **Duration**: 1 week (same schedule as primary)
- **Frequency**: Alternates with primary rotation
- **Purpose**: Escalation path if primary is unreachable or needs help

### Example 8-Week Rotation

| Week | Primary On-Call | Secondary On-Call | Tertiary Backup |
|------|----------------|-------------------|-----------------|
| 1    | Alice          | Bob               | Carol           |
| 2    | Bob            | Carol             | David           |
| 3    | Carol          | David             | Alice           |
| 4    | David          | Alice             | Bob             |
| 5    | Eve            | Frank             | Alice           |
| 6    | Frank          | Alice             | Bob             |
| 7    | Alice          | Bob               | Carol           |
| 8    | Bob            | Carol             | David           |

**Rotation managed in**: PagerDuty

---

### Coverage Hours

**24/7 On-Call Coverage**:
- **Business Hours** (9am-5pm local time): Expected to respond within 5 minutes
- **After Hours** (5pm-9am local time): Expected to respond within 15 minutes
- **Weekends**: Same as after-hours expectations

**Exception**:
- **Planned unavailability**: Swap with teammate (see Swap Process below)
- **Emergency unavailability**: Secondary takes over

---

### Geographic Distribution

**Goal**: Follow-the-sun rotation to minimize after-hours pages

**Current Distribution**:
- **US West Coast** (3 engineers): 6am-2pm UTC coverage
- **US East Coast** (2 engineers): 9am-5pm UTC coverage
- **Europe** (2 engineers): 12pm-8pm UTC coverage
- **Asia-Pacific** (1 engineer): Coverage TBD

**Note**: Currently not full follow-the-sun. Most incidents handled by US team.

---

## Roles and Responsibilities

### Primary On-Call Engineer

**Responsibilities**:

1. **Incident Response**:
   - Acknowledge alerts within SLA (2 min for P0, 5 min for P1)
   - Investigate and mitigate incidents following runbooks
   - Escalate to secondary if issue is complex or outside expertise
   - Document actions in incident thread

2. **Communication**:
   - Post status updates in #incidents channel
   - Notify stakeholders for major incidents
   - Complete incident reports after resolution

3. **Availability**:
   - Carry PagerDuty phone/device at all times during shift
   - Have laptop with VPN access available
   - Be in location with reliable internet
   - Respond to pages within SLA

4. **Handoff**:
   - Conduct handoff call with incoming on-call
   - Document ongoing incidents and action items
   - Ensure PagerDuty schedule is updated

**Authority**:
- Declare incidents (P0-P3)
- Execute runbook mitigation actions
- Approve rollbacks during incidents
- Request assistance from any team member

---

### Secondary On-Call Engineer

**Responsibilities**:

1. **Backup Coverage**:
   - Be available if primary is unreachable (5-minute escalation)
   - Assist primary with complex incidents
   - Take over if primary is overwhelmed (multiple incidents)

2. **Availability**:
   - Same as primary, but pages only if primary doesn't respond
   - Help triage during major incidents

3. **Handoff**:
   - Participate in handoff call (optional but recommended)

**Authority**:
- Same as primary once escalated

---

### Service Owner (Subject Matter Expert)

**Responsibilities**:

1. **Expertise**:
   - Available for escalation during business hours
   - Provide deep technical knowledge for complex issues
   - Approve high-risk mitigation actions (e.g., database changes)

2. **Availability**:
   - No expectation of 24/7 availability
   - Best-effort response for after-hours P0 incidents
   - Formal escalation for incidents >30 minutes

**Authority**:
- Approve deployment rollbacks for their service
- Make architectural decisions during incidents
- Assign follow-up action items

---

### SRE Lead

**Responsibilities**:

1. **Escalation Point**:
   - Available for major incidents (P0, long-running P1)
   - Coordinate multi-team response
   - Communicate with engineering management

2. **Post-Incident**:
   - Review all P0/P1 incidents
   - Approve runbook updates
   - Identify reliability improvements

**Availability**:
- Business hours (primary)
- After-hours for P0 escalations only

---

## Escalation Path

### P0 - Critical Incident

```
Minute 0:  Alert fires → PagerDuty → Primary On-Call
Minute 5:  If not acknowledged → Secondary On-Call
Minute 15: Automatic escalation → Service Owner + SRE Lead
Minute 30: Automatic escalation → Engineering Manager
Minute 60: Automatic escalation → VP Engineering
```

### P1 - High Priority Incident

```
Minute 0:  Alert fires → PagerDuty → Primary On-Call
Minute 15: If not acknowledged → Secondary On-Call
Minute 30: Escalation → Service Owner
Minute 60: Escalation → SRE Lead
```

### P2 - Medium Priority

```
No automatic escalation.
Slack notification to #monitoring.
On-call reviews during next business day.
```

### Manual Escalation

**When to escalate**:
- Issue outside your expertise
- Need approval for high-risk action
- Incident not resolving within MTTR goal
- Multiple simultaneous incidents
- Uncertainty about mitigation approach

**How to escalate**:
```
1. Post in #incidents: "@secondary-oncall need assistance with [issue]"
2. If urgent, call secondary directly via PagerDuty
3. For subject matter expertise, page service owner
4. For major incidents, request all-hands via #incidents
```

---

## Handoff Procedures

### Handoff Call (Monday 9am PT)

**Duration**: 30 minutes
**Attendees**: Outgoing on-call + Incoming on-call + (Optional) SRE lead

**Agenda**:

1. **Past Week Review** (10 minutes):
   - Total alerts received
   - Incidents handled (P0, P1, P2)
   - Average MTTR
   - Interesting/tricky incidents
   - Lessons learned

2. **Ongoing Issues** (10 minutes):
   - Active incidents or degraded services
   - Error budget status
   - Scheduled maintenance windows
   - Known issues to watch

3. **Action Items** (5 minutes):
   - Follow-up tickets from incidents
   - Runbook updates needed
   - Monitoring gaps identified

4. **Questions** (5 minutes):
   - Incoming on-call asks clarifying questions
   - Review any recent changes to on-call procedures

**Output**: Handoff notes posted in #on-call-handoff Slack channel

---

### Handoff Template

```markdown
## On-Call Handoff: [Outgoing] → [Incoming]
**Date**: 2025-10-23 9:00 AM PT
**Week**: Oct 16-23, 2025

### Incidents Summary
- **P0 Critical**: 1 incident
  - 10/18 15:45 - ServiceDown (8 min) - Memory leak, rolled back
- **P1 High**: 2 incidents
  - 10/17 10:30 - HighErrorRate (12 min) - Bad deployment
  - 10/20 03:15 - HighLatency (22 min) - Database slow query
- **P2 Medium**: 5 incidents (see ticket links below)
- **Total Pages**: 8
- **Average MTTR**: 14 minutes

### Current Status
- ✅ All services healthy
- ⚠️  Error budget: 58% (was 72% last week)
- ⚠️  Watch: Database CPU trending up (Action item: #1234)

### Ongoing Issues
- None currently

### Scheduled Maintenance
- 10/25 02:00 AM PT - Database minor version upgrade (planned downtime: 10 min)

### Action Items
- [ ] Update runbook for memory leak detection (#INC-456)
- [ ] Add alert for database query duration (#MON-789)

### Notes
- New deployment process: Now requires canary for 30min (was 15min)
- Runbook 001-high-error-rate updated with new mitigation steps

### Questions?
Ask in #incidents or DM me @outgoing-engineer
```

---

## Compensation and Time Off

### On-Call Pay

**Compensation Structure**:
- **On-call stipend**: $X per week on-call (whether paged or not)
- **Incident pay**: $Y per hour actively working on incidents after-hours
- **Weekend/holiday multiplier**: 1.5x incident pay

**Payment Schedule**: Monthly, separate from regular salary

**Tracking**: Automatically tracked in PagerDuty

---

### Time Off

#### Compensatory Time Off

**After-hours incident response**:
- **>2 hours**: Take comp time next business day
- **>4 hours**: Take full day off
- **Overnight incident (10pm-6am)**: Take day off, no questions asked

**Process**:
1. Log hours in PagerDuty (automatic)
2. Notify manager of comp time plan
3. Update team calendar
4. No approval needed (pre-approved policy)

#### Planned Time Off (PTO)

**Swapping shifts**:
1. Request swap in PagerDuty (minimum 2 weeks notice)
2. Find teammate to swap with (or volunteer from pool)
3. Update PagerDuty schedule
4. Confirm swap in #on-call channel

**Emergency unavailability**:
1. Notify #incidents channel immediately
2. Secondary automatically takes over
3. Update manager
4. Arrange swap for remaining shift

---

### On-Call Load Limits

**Maximum load per engineer**:
- **Target**: <5 pages per week
- **Warning**: 5-10 pages per week (review alert quality)
- **Critical**: >10 pages per week (immediate action required)

**If consistently overloaded**:
1. Review alert quality (false positives?)
2. Improve runbooks (faster resolution)
3. Add engineers to rotation
4. Implement alert quality improvements

---

## Training and Onboarding

### On-Call Readiness Requirements

Before joining on-call rotation, engineers must:

1. **Complete Onboarding** (4-6 weeks):
   - [ ] Shadow 2 on-call shifts (one business hours, one after-hours)
   - [ ] Participate in 3 incident responses (as observer)
   - [ ] Complete incident response training
   - [ ] Read all runbooks
   - [ ] Pass on-call readiness quiz

2. **Technical Prerequisites**:
   - [ ] Access to all production systems
   - [ ] VPN configured on personal device
   - [ ] PagerDuty app installed and tested
   - [ ] Familiarity with monitoring dashboards
   - [ ] Know how to execute common mitigation actions

3. **First Week On-Call** (Supported):
   - Shadow pairing: Experienced engineer shadows first shift
   - Dedicated Slack channel for questions
   - SRE lead on standby

---

### On-Call Training Resources

**Required Reading**:
- [ ] Alerting Policy
- [ ] Error Budget Policy
- [ ] All Runbooks (001-005)
- [ ] Incident Response Guide
- [ ] Escalation Procedures

**Recommended Reading**:
- [ ] Google SRE Book: "Being On-Call"
- [ ] PagerDuty Incident Response Guide
- [ ] Post-mortems from last 6 months

**Hands-On Training**:
- [ ] Chaos Engineering exercise (simulate incidents)
- [ ] Runbook walkthrough session
- [ ] Dashboard navigation training
- [ ] Communication templates practice

---

### Incident Simulation Drills

**Frequency**: Quarterly

**Purpose**:
- Practice incident response in safe environment
- Test runbooks and tooling
- Build muscle memory
- Identify gaps

**Example Drill**:
```bash
# Use incident simulation script
./scripts/simulate-incident.sh

# Select scenario:
# 1. High Error Rate
# 2. Latency Spike
# 3. Complete Outage

# Follow runbook as if real incident
# Debrief afterwards: What went well? What needs improvement?
```

---

## On-Call Best Practices

### Before Your Shift

- [ ] Check calendar for conflicts (travel, appointments)
- [ ] Test PagerDuty (send test page to yourself)
- [ ] Review recent incidents
- [ ] Attend handoff call
- [ ] Check error budget status
- [ ] Review scheduled maintenance

---

### During Your Shift

#### Daily Check-In (5 minutes)

**Morning routine**:
1. Review overnight alerts (if any)
2. Check service health dashboard
3. Review error budget trend
4. Check #incidents channel for updates
5. Respond to any non-urgent tickets

#### Incident Response

**When page arrives**:
1. **Acknowledge immediately** (within SLA)
2. **Open runbook** for alert type
3. **Post in #incidents** (even if minor)
4. **Follow runbook** step-by-step
5. **Document actions** as you go
6. **Escalate early** if stuck
7. **Communicate** status updates
8. **Resolve** and document learnings

#### Communication Tips

**During incident**:
- Over-communicate (post updates even if no progress)
- Use clear, concise language
- Include data (metrics, logs, traces)
- Set expectations for next update

**Example**:
```
🔴 Investigating: HighErrorRate

Status: Error rate 8% (target: <1%)
Started: 16:45 UTC
Duration: 5 minutes

Actions taken:
- Checked recent deployments (none in last hour)
- Checking resource usage (CPU: 45%, Memory: 62% - normal)
- Reviewing logs for patterns

Next update: 16:55 UTC (10 min)
```

---

### After Your Shift

- [ ] Attend handoff call with incoming on-call
- [ ] Complete incident reports for all P0/P1 incidents
- [ ] File tickets for runbook updates
- [ ] Submit comp time if applicable
- [ ] Retrospect: What could be improved?

---

### Self-Care During On-Call

**Healthy practices**:
- ✅ Keep laptop charged and accessible
- ✅ Have backup internet (mobile hotspot)
- ✅ Sleep with phone nearby (but not vibrating under pillow!)
- ✅ Limit alcohol during on-call week
- ✅ Plan light social activities (easy to leave if paged)
- ✅ Communicate on-call status to family/friends

**Avoid**:
- ❌ Traveling without laptop and VPN
- ❌ Committing to events you can't leave (weddings, theater, etc.)
- ❌ Excessive caffeine before bed (you need quality sleep!)
- ❌ Ignoring alerts (even false positives - acknowledge and investigate)

**Burnout prevention**:
- Take comp time after rough shifts
- Flag excessive pages to SRE lead
- Request rotation adjustments if needed
- Remember: You're not alone (escalate early!)

---

## On-Call Metrics

**Tracked monthly**:
- Alert volume per engineer
- Average MTTR per engineer
- Response time (acknowledgment latency)
- Escalation rate
- Comp time taken
- On-call satisfaction (quarterly survey)

**Review in monthly SRE meeting**:
- Identify engineers with high load
- Discuss alert quality issues
- Recognize excellent incident response
- Adjust rotation if needed

---

## Opting Out

**If on-call is not for you**:
- On-call participation is typically **expected** for ML Platform and SRE roles
- However, **exceptions can be made**:
  - Personal circumstances (family, health)
  - Role adjustment (focus on non-operational work)
  - Transfer to team without on-call requirements

**Process**:
1. Discuss with manager
2. Create plan (reduce rotation, opt out, role change)
3. Update team capacity planning
4. No penalties (everyone has different circumstances)

---

## Frequently Asked Questions

### What if I'm sick during my on-call shift?

Notify #incidents immediately. Secondary takes over. No need to find replacement yourself.

### Can I travel while on-call?

Yes, but ensure you have laptop, VPN, and reliable internet. Avoid international travel (timezone challenges).

### What if I miss a page?

Don't panic. Secondary will be paged after 5 minutes. Acknowledge as soon as you see it and communicate what happened.

### How do I handle multiple simultaneous incidents?

Triage: Address P0 first. Post in #incidents requesting assistance. SRE lead will coordinate additional responders.

### Do I need to fix all incidents or just mitigate?

Mitigation is the priority. Restore service first, root cause analysis can happen later.

### What if I break something during incident response?

Happens to everyone. Focus on recovery. Document what happened. Post-mortem will identify improvements (blameless culture).

---

## Document Change Log

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-10-23 | 1.0 | SRE Team | Initial policy creation |

---

## Approval

**Policy Owner**: SRE Team Lead
**Approved By**: Engineering Manager, VP of Engineering, HR
**Effective Date**: 2025-10-23
**Next Review**: 2025-11-23

---

## Additional Resources

- **PagerDuty**: https://company.pagerduty.com
- **On-Call Schedule**: https://company.pagerduty.com/schedules
- **Runbooks**: https://runbooks.company.com
- **Incident Channel**: #incidents (Slack)
- **On-Call Support**: #on-call-support (Slack)
- **SRE Team**: sre@company.com
