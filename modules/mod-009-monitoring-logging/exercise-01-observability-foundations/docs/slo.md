# Service Level Objectives (SLOs)

## Overview

This document defines the Service Level Objectives (SLOs) for the Inference Gateway service. SLOs are targets for service reliability that balance user expectations with engineering costs.

## SLO Definitions

### 1. Availability SLO

**Target**: 99.5% availability over a rolling 30-day window

**Measurement**:
- **SLI** (Service Level Indicator): Ratio of successful requests to total requests
- **Formula**: `(successful_requests / total_requests) × 100`
- **Success Definition**: HTTP status codes 2xx, 3xx, 4xx (client errors don't count against availability)
- **Failure Definition**: HTTP status codes 5xx (server errors)

**Prometheus Query**:
```promql
# Current availability (30-day rolling window)
(
  sum(rate(http_requests_total{status!~"5.."}[30d]))
  /
  sum(rate(http_requests_total[30d]))
) * 100
```

**Error Budget**:
- **Monthly**: 0.5% = 216 minutes = 3.6 hours
- **Daily**: 7.2 minutes
- **Hourly**: 18 seconds

### 2. Latency SLO

**Target**: P99 latency < 300ms for `/predict` endpoint over a rolling 7-day window

**Measurement**:
- **SLI**: 99th percentile of request duration
- **Endpoint**: `/predict` (core inference endpoint)
- **Exclusions**: `/health`, `/ready`, `/metrics` (monitoring endpoints)

**Prometheus Query**:
```promql
# P99 latency for /predict endpoint (7-day rolling window)
histogram_quantile(
  0.99,
  rate(http_request_duration_seconds_bucket{endpoint="/predict"}[7d])
)
```

**Additional Latency Targets**:
- **P95**: < 200ms
- **P50**: < 100ms

**Prometheus Queries**:
```promql
# P95 latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{endpoint="/predict"}[7d]))

# P50 latency
histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{endpoint="/predict"}[7d]))
```

## Error Budget Policy

### Error Budget Calculation

```
Error Budget Remaining = 1 - (Actual Error Rate / SLO Error Rate)
```

**Prometheus Query**:
```promql
# Error budget remaining (1.0 = 100%, 0.0 = 0%)
1 - (
  (
    sum(rate(http_requests_total{status=~"5.."}[30d]))
    /
    sum(rate(http_requests_total[30d]))
  ) / 0.005
)
```

### Error Budget Alerts

| Error Budget Consumed | Action |
|-----------------------|--------|
| < 50% | Normal operations |
| 50-75% | Alert SRE team, review incidents |
| 75-90% | Halt feature launches, focus on reliability |
| > 90% | Incident declared, all hands on reliability |

### Error Budget Reset

- Error budgets reset monthly
- Partial resets may be granted for:
  - Platform-wide outages (not service-specific)
  - Scheduled maintenance (with advance notice)
  - Valid bugs in SLO measurement

## SLO Monitoring

### Dashboards

**Grafana Dashboard**: `Inference Gateway - SLO Overview`
- Current availability vs target
- P99 latency vs target
- Error budget remaining
- Error budget burn rate
- Historical SLO compliance

### Key Metrics

1. **Availability Metrics**:
   - `http_requests_total` - Total requests
   - `http_requests_total{status=~"5.."}` - Server errors
   - Availability percentage
   - Error budget remaining

2. **Latency Metrics**:
   - `http_request_duration_seconds` - Request latency histogram
   - P50, P95, P99 percentiles
   - Latency trends over time

3. **Traffic Metrics**:
   - Request rate (requests/second)
   - Request volume trends
   - Traffic patterns by hour/day

## SLO Reporting

### Daily SLO Report

Generated automatically and sent to team Slack channel:

```
🎯 Inference Gateway SLO Report - [Date]

Availability SLO (99.5% target):
✅ Current: 99.87% (Above target)
📊 Error Budget: 74% remaining

Latency SLO (P99 < 300ms target):
✅ Current P99: 245ms (Below target)
📊 Current P95: 180ms
📊 Current P50: 95ms

🔥 Error Budget Burn Rate: 0.26%/day (Normal)
```

### Weekly SLO Review

Team review every Monday:
1. Review SLO compliance
2. Analyze SLO violations (if any)
3. Review error budget consumption
4. Identify reliability improvements
5. Adjust SLOs if needed (quarterly)

## SLO Violations

### Violation Response

When SLO is violated (falls below target):

1. **Immediate** (< 1 hour):
   - Alert fires to on-call engineer
   - Create incident in incident management system
   - Begin investigation

2. **Short-term** (< 24 hours):
   - Root cause analysis
   - Implement immediate mitigation
   - Update status page

3. **Long-term** (< 1 week):
   - Post-incident review
   - Identify preventive measures
   - Update runbooks
   - Implement fixes

### Recent Violations

| Date | SLO | Actual | Duration | Root Cause | Resolution |
|------|-----|--------|----------|------------|------------|
| - | - | - | - | - | - |

*(No violations to date)*

## SLO Evolution

### Adjustment Criteria

SLOs may be adjusted quarterly based on:
- User feedback and expectations
- Business requirements
- System maturity
- Cost considerations
- Historical performance data

### Historical SLO Changes

| Date | SLO | Old Target | New Target | Reason |
|------|-----|------------|------------|--------|
| 2024-01-01 | Availability | - | 99.5% | Initial SLO |
| 2024-01-01 | Latency (P99) | - | 300ms | Initial SLO |

## Appendix

### Prometheus Recording Rules

```yaml
# /etc/prometheus/rules/slo-rules.yml
groups:
  - name: slo_rules
    interval: 30s
    rules:
      # Availability SLI
      - record: slo:availability:ratio_30d
        expr: |
          sum(rate(http_requests_total{status!~"5.."}[30d]))
          /
          sum(rate(http_requests_total[30d]))

      # Error budget remaining
      - record: slo:error_budget:remaining_30d
        expr: |
          1 - (
            (
              sum(rate(http_requests_total{status=~"5.."}[30d]))
              /
              sum(rate(http_requests_total[30d]))
            ) / 0.005
          )

      # Latency P99
      - record: slo:latency:p99_7d
        expr: |
          histogram_quantile(
            0.99,
            rate(http_request_duration_seconds_bucket{endpoint="/predict"}[7d])
          )
```

### Alert Rules

```yaml
# /etc/prometheus/rules/slo-alerts.yml
groups:
  - name: slo_alerts
    rules:
      # Availability SLO violation
      - alert: SLOAvailabilityViolation
        expr: slo:availability:ratio_30d < 0.995
        for: 5m
        labels:
          severity: critical
          slo: availability
        annotations:
          summary: "Availability SLO violated"
          description: "Availability is {{ $value | humanizePercentage }}, below 99.5% target"

      # Error budget critical
      - alert: ErrorBudgetCritical
        expr: slo:error_budget:remaining_30d < 0.1
        for: 5m
        labels:
          severity: critical
          slo: error_budget
        annotations:
          summary: "Error budget critically low"
          description: "Only {{ $value | humanizePercentage }} error budget remaining"

      # Latency SLO violation
      - alert: SLOLatencyViolation
        expr: slo:latency:p99_7d > 0.3
        for: 10m
        labels:
          severity: warning
          slo: latency
        annotations:
          summary: "Latency SLO violated"
          description: "P99 latency is {{ $value }}s, above 300ms target"
```

## Resources

- [Google SRE Book - SLIs, SLOs, SLAs](https://sre.google/sre-book/service-level-objectives/)
- [Implementing SLOs](https://sre.google/workbook/implementing-slos/)
- [Error Budgets](https://sre.google/sre-book/embracing-risk/)
