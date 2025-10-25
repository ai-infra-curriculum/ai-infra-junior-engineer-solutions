# Incident Report: [TITLE]

**Incident ID**: INC-YYYYMMDD-####
**Date**: YYYY-MM-DD
**Severity**: [P0-Critical / P1-High / P2-Medium / P3-Low]
**Status**: [Investigating / Mitigating / Resolved / Closed]
**Incident Commander**: @name
**Duration**: HH:MM (from detection to resolution)

---

## Executive Summary

**One-line summary**: [What happened in one sentence]

**Impact**:
- **Users affected**: [Number or percentage]
- **Duration**: [How long]
- **Services impacted**: [Which services]
- **Revenue impact**: [$amount or N/A]
- **SLO impact**: [Error budget consumed: X%]

**Root Cause**: [One sentence root cause]

**Resolution**: [One sentence how it was fixed]

---

## Timeline

All times in UTC.

| Time | Event | Actor |
|------|-------|-------|
| 14:23 | Alert fired: HighErrorRate | Prometheus |
| 14:24 | Alert acknowledged | @oncall-engineer |
| 14:25 | Incident declared in #incidents | @oncall-engineer |
| 14:27 | Dashboard shows 12% error rate | @oncall-engineer |
| 14:30 | Identified recent deployment as cause | @oncall-engineer |
| 14:32 | Initiated rollback to v1.2.2 | @oncall-engineer |
| 14:35 | Rollback completed | Kubernetes |
| 14:38 | Error rate dropped to 0.8% | Prometheus |
| 14:40 | Monitoring for 10 minutes | @oncall-engineer |
| 14:50 | Incident resolved | @oncall-engineer |

**Total Duration**: 27 minutes (detection to resolution)

---

## Detailed Description

### What Happened

[Detailed description of the incident]

Example:
At 14:23 UTC, Prometheus detected an elevated error rate on the inference-gateway service. The error rate quickly climbed from baseline 0.3% to 12% within 2 minutes. Investigation revealed that deployment v1.2.3, which went live at 14:20 UTC, introduced a NullPointerException in the prediction request handler when processing images without EXIF metadata.

### Blast Radius

**Affected Services**:
- inference-gateway: 12% error rate
- feature-store: Minor increase in cache misses
- model-registry: No impact

**User Impact**:
- Approximately 840 failed prediction requests (out of 7,000 total in 30 minutes)
- Affected users: ~200 unique clients
- Geography: All regions

**SLO Impact**:
- Monthly availability SLO: 99.72% (target: 99.5%) ✅
- Error budget consumed: 8.4% (acceptable)
- No SLO breach

---

## Root Cause Analysis

### The Five Whys

1. **Why did inference requests fail?**
   - Because the prediction handler threw NullPointerException

2. **Why did it throw NullPointerException?**
   - Because code tried to access `image.exif.datetime` without null check

3. **Why was there no null check?**
   - Because the developer assumed all images have EXIF data

4. **Why wasn't this caught in testing?**
   - Because test images all had EXIF metadata

5. **Why didn't test images cover this case?**
   - Because test data generation didn't include edge cases

**Root Cause**: Insufficient test coverage for edge cases (images without EXIF metadata)

### Contributing Factors

1. **Code Review**: Reviewer didn't catch missing null check
2. **CI/CD**: Integration tests didn't cover null EXIF scenario
3. **Deployment Process**: No canary deployment to catch issue before full rollout
4. **Monitoring**: Alert fired correctly, but 2-minute delay before page

---

## Resolution

### Immediate Mitigation

**Action Taken**: Rollback deployment from v1.2.3 to v1.2.2

```bash
kubectl rollout undo deployment/inference-gateway
kubectl rollout status deployment/inference-gateway
```

**Why It Worked**: v1.2.2 had proper null checking for EXIF data

**Time to Mitigation**: 12 minutes (from detection)

### Verification

- Error rate dropped from 12% to 0.8% (baseline 0.3%)
- Monitored for 10 minutes to ensure stability
- Spot-checked logs for residual errors
- Verified SLO dashboard showed recovery

---

## Action Items

### Immediate (< 24 hours)

- [x] **@dev-lead**: Add null check to EXIF access in v1.2.4 (INC-001)
- [x] **@qa-lead**: Add test case for images without EXIF metadata (INC-002)
- [ ] **@oncall-engineer**: Complete post-mortem document (INC-003)

### Short-term (< 1 week)

- [ ] **@sre-lead**: Implement canary deployment for inference-gateway (INC-004)
- [ ] **@qa-lead**: Expand test data generator to include edge cases (INC-005)
- [ ] **@dev-lead**: Add defensive coding guideline to style guide (INC-006)

### Long-term (< 1 month)

- [ ] **@platform-lead**: Implement automated rollback on error spike (INC-007)
- [ ] **@qa-lead**: Fuzzing test suite for prediction handler (INC-008)
- [ ] **@sre-lead**: Reduce alert detection delay to <1 minute (INC-009)

---

## Lessons Learned

### What Went Well

✅ **Alert Detection**: Prometheus detected issue within 2 minutes of deployment
✅ **Response Time**: On-call engineer acknowledged and began investigation immediately
✅ **Diagnosis Speed**: Root cause identified in 8 minutes
✅ **Mitigation**: Rollback executed cleanly without additional issues
✅ **Communication**: Clear updates in #incidents channel every 5-10 minutes

### What Went Wrong

❌ **Deployment Safety**: No canary deployment to catch issue before full rollout
❌ **Test Coverage**: Edge case not covered in integration tests
❌ **Code Review**: Null pointer risk not identified during review
❌ **Detection Delay**: 2-minute delay between deployment and alert

### Where We Got Lucky

🍀 **Timing**: Incident occurred during business hours with full team available
🍀 **Severity**: Error rate 12% (bad) but not 100% (catastrophic)
🍀 **Rollback**: Previous version was stable and rollback worked cleanly
🍀 **SLO**: Had sufficient error budget to absorb incident

---

## Supporting Data

### Metrics

**Error Rate**:
```promql
sum(rate(http_requests_total{service="inference-gateway", status=~"5.."}[5m]))
/
sum(rate(http_requests_total{service="inference-gateway"}[5m])) * 100
```
- Baseline: 0.3%
- Peak: 12.4%
- Recovery: 0.8% (within 10 minutes of rollback)

**Request Volume**:
- Total requests during incident: 7,000
- Failed requests: 840
- Success rate: 88%

### Logs

**Sample Error**:
```json
{
  "timestamp": "2025-10-23T14:25:12.345Z",
  "level": "ERROR",
  "service": "inference-gateway",
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "NullPointerException in predict_handler",
  "stack_trace": "at PredictionHandler.extract_datetime(predict.py:45)\n...",
  "endpoint": "/predict",
  "method": "POST",
  "status_code": 500
}
```

### Traces

**Sample Failed Trace**: http://localhost:16686/trace/550e8400-e29b-41d4-a716-446655440000

- Total duration: 245ms
- Failed at: EXIF extraction step
- Error: NullPointerException

---

## References

- **Alert**: http://localhost:9090/alerts (HighErrorRate)
- **Dashboard**: http://localhost:3000/d/app-performance
- **Runbook**: runbooks/001-high-error-rate.md
- **Pull Request**: https://github.com/company/inference-gateway/pull/456 (v1.2.3)
- **Rollback Commit**: https://github.com/company/inference-gateway/commit/abc123

---

## Sign-off

**Incident Commander**: @oncall-engineer (approved)
**Service Owner**: @dev-lead (approved)
**SRE Lead**: @sre-lead (approved)

**Post-Mortem Meeting**:
- Date: 2025-10-24 10:00 UTC
- Attendees: IC, service owner, SRE lead, QA lead
- Recording: [link]
- Notes: [link]

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2025-10-23 | @oncall-engineer | Initial draft |
| 2025-10-23 | @dev-lead | Added code review notes |
| 2025-10-24 | @sre-lead | Added action items and sign-off |
