# PromQL Query Library for ML Infrastructure

Comprehensive collection of useful PromQL queries for monitoring ML platforms.

## Table of Contents

1. [Basic Metrics](#basic-metrics)
2. [Latency Analysis](#latency-analysis)
3. [Error Tracking](#error-tracking)
4. [Resource Usage](#resource-usage)
5. [Business Metrics](#business-metrics)
6. [Trend Analysis](#trend-analysis)
7. [Anomaly Detection](#anomaly-detection)
8. [SLO Tracking](#slo-tracking)
9. [Dashboard Queries](#dashboard-queries)
10. [Troubleshooting Queries](#troubleshooting-queries)

---

## Basic Metrics

### Request Rate

```promql
# Per-model request rate (requests/sec)
model:predictions:rate5m

# Platform-wide request rate
platform:predictions:rate5m

# Request rate by endpoint
sum by (endpoint) (rate(model_http_requests_total[5m]))

# Request rate trend (current vs 1 hour ago)
model:predictions:rate5m / (model:predictions:rate5m offset 1h)
```

### Error Rate

```promql
# Per-model error rate (errors/sec)
model:errors:rate5m

# Error ratio (percentage)
model:error_ratio:rate5m

# Models with >5% error rate
model:error_ratio:rate5m > 0.05

# Platform error rate
platform:error_ratio:rate5m
```

### Success Rate

```promql
# Per-model success rate
model:success_ratio:rate5m

# Platform success rate
1 - platform:error_ratio:rate5m

# Success rate over last hour
avg_over_time(model:success_ratio:rate5m[1h])
```

---

## Latency Analysis

### Percentile Queries

```promql
# P50 latency per model
model:latency:p50

# P95 latency per model
model:latency:p95

# P99 latency per model
model:latency:p99

# Average latency
model:latency:mean

# Platform P95 latency
platform:latency:p95
```

### Latency Distributions

```promql
# Raw histogram query (without recording rule)
histogram_quantile(0.95,
  sum by (le, model) (
    rate(model_prediction_duration_seconds_bucket[5m])
  )
)

# Multiple percentiles at once
histogram_quantile(0.50, sum by (le, model) (rate(model_prediction_duration_seconds_bucket[5m])))
  or
histogram_quantile(0.95, sum by (le, model) (rate(model_prediction_duration_seconds_bucket[5m])))
  or
histogram_quantile(0.99, sum by (le, model) (rate(model_prediction_duration_seconds_bucket[5m])))

# Latency bucket distribution
sum by (le, model) (rate(model_prediction_duration_seconds_bucket[5m]))
```

### Latency Trends

```promql
# Latency trend (current vs 1 hour ago)
model:latency:p95 / (model:latency:p95 offset 1h)

# Latency degradation (>20% slower)
model:latency:p95 / (model:latency:p95 offset 1h) > 1.2

# Average latency over last 24 hours
avg_over_time(model:latency:p95[24h])
```

### Latency SLO Compliance

```promql
# Models meeting 200ms SLO
model:latency:p95 < 0.2

# Models violating 200ms SLO
model:latency:p95 > 0.2

# Percentage of time meeting SLO (over 1 hour)
(count_over_time((model:latency:p95 < 0.2)[1h:])) / (count_over_time(model:latency:p95[1h:]))
```

---

## Error Tracking

### Error Types

```promql
# Errors by status code
sum by (status_code) (rate(model_http_requests_total{status_code=~"5.."}[5m]))

# 4xx vs 5xx errors
sum(rate(model_http_requests_total{status_code=~"4.."}[5m])) # Client errors
  vs
sum(rate(model_http_requests_total{status_code=~"5.."}[5m])) # Server errors

# Timeout errors
sum by (model) (rate(model_prediction_errors_total{error_type="timeout"}[5m]))

# OOM errors
sum by (model) (rate(model_prediction_errors_total{error_type="out_of_memory"}[5m]))
```

### Error Patterns

```promql
# Models with increasing error rates
deriv(model:error_ratio:rate5m[30m]) > 0

# Error spikes (>3x normal rate)
model:errors:rate5m > 3 * avg_over_time(model:errors:rate5m[1h])

# Consistent errors (>1% for 2+ hours)
avg_over_time(model:error_ratio:rate5m[2h]) > 0.01
```

### Error Impact

```promql
# Total error count (last hour)
sum(increase(model_prediction_errors_total[1h]))

# Errors per minute
sum(rate(model_prediction_errors_total[5m])) * 60

# Failed requests impact
sum(model:errors:rate5m) / sum(model:predictions:rate5m) * 100
```

---

## Resource Usage

### CPU Usage

```promql
# CPU utilization per model
model:cpu:utilization

# Models using >85% CPU
model:cpu:utilization > 0.85

# CPU usage trend
deriv(model:cpu:utilization[15m])

# Total platform CPU usage
sum(rate(container_cpu_usage_seconds_total{container="model-server"}[5m]))
```

### Memory Usage

```promql
# Memory utilization per model
model:memory:utilization

# Models using >90% memory
model:memory:utilization > 0.90

# Memory usage in GB
sum by (model) (container_memory_usage_bytes{container="model-server"}) / 1024 / 1024 / 1024

# Memory growth rate (MB per hour)
model:memory:growth_rate_1h / 1024 / 1024
```

### Efficiency Metrics

```promql
# Predictions per CPU second
model:efficiency:predictions_per_cpu

# Predictions per GB memory
model:efficiency:predictions_per_gb_memory

# Low efficiency models (<10 predictions/CPU-second)
model:efficiency:predictions_per_cpu < 10

# Resource efficiency comparison
topk(10, model:efficiency:predictions_per_cpu)
```

---

## Business Metrics

### Traffic Analysis

```promql
# Top 5 models by traffic
topk(5, model:predictions:rate5m)

# Bottom 5 models by traffic
bottomk(5, model:predictions:rate5m)

# Models with <10 req/min
model:predictions:rate5m * 60 < 10

# Total daily predictions
sum(increase(model_predictions_total[24h]))
```

### Cost Metrics

```promql
# Cost per 1000 predictions
model:cost:per_1k_predictions

# Most expensive models
topk(5, model:cost:per_1k_predictions)

# Total platform cost per hour
sum(model:cost:per_1k_predictions * model:predictions:rate5m * 3600 / 1000)

# Cost efficiency (predictions per dollar)
1 / model:cost:per_1k_predictions
```

### Revenue and Profit

```promql
# Revenue per model
model:revenue:rate5m * 3600 # per hour

# Profit per model
model:profit:rate5m * 3600 # per hour

# Unprofitable models
model:profit:rate5m < 0

# Most profitable models
topk(5, model:profit:rate5m)

# Profit margin percentage
(model:profit:rate5m / model:revenue:rate5m) * 100
```

---

## Trend Analysis

### Traffic Trends

```promql
# Hour-over-hour growth
model:predictions:trend_1h

# Day-over-day growth
model:predictions:trend_24h

# Growing models (>20% increase)
model:predictions:trend_1h > 1.2

# Declining models (>20% decrease)
model:predictions:trend_1h < 0.8

# Weekly trend
model:predictions:rate5m / (model:predictions:rate5m offset 7d)
```

### Performance Trends

```promql
# Latency trend
model:latency:trend_1h

# Error rate trend
model:error_ratio:trend_1h

# Cache hit rate trend
model:cache:hit_ratio / (model:cache:hit_ratio offset 1h)

# Degrading performance (latency +50%)
model:latency:trend_1h > 1.5
```

### Smoothing and Averaging

```promql
# 1-hour moving average
model:predictions:avg_1h

# Standard deviation (volatility)
model:predictions:stddev_1h

# Coefficient of variation (relative volatility)
model:predictions:stddev_1h / model:predictions:avg_1h
```

---

## Anomaly Detection

### Statistical Anomalies

```promql
# Deviations >2 standard deviations
abs(model:predictions:rate5m - model:predictions:avg_1h) > 2 * model:predictions:stddev_1h

# Z-score calculation
(model:predictions:rate5m - model:predictions:avg_1h) / model:predictions:stddev_1h

# Traffic outside 3-sigma range
abs(model:predictions:rate5m - avg_over_time(model:predictions:rate5m[1h])) >
  3 * stddev_over_time(model:predictions:rate5m[1h])
```

### Pattern Anomalies

```promql
# Unusual error patterns (errors but no traffic drop)
model:error_ratio:rate5m > 0.05 and model:predictions:rate5m > avg_over_time(model:predictions:rate5m[1h])

# Latency spike without traffic spike
model:latency:p95 > 2 * avg_over_time(model:latency:p95[1h])
  and
model:predictions:rate5m < 1.2 * avg_over_time(model:predictions:rate5m[1h])

# Cache miss spike
model:cache:miss_ratio > 3 * avg_over_time(model:cache:miss_ratio[1h])
```

### Change Detection

```promql
# Sudden change in request rate (>50% in 5 min)
abs(deriv(model:predictions:rate5m[5m])) > 0.5 * model:predictions:rate5m

# Rapid latency increase
deriv(model:latency:p95[10m]) > 0.1 # 100ms increase per minute

# Error rate spike
rate(model_prediction_errors_total[5m]) > 5 * rate(model_prediction_errors_total[1h] offset 1h)
```

---

## SLO Tracking

### Availability SLO

```promql
# Availability (99.9% target)
model:availability:rate5m

# Models below 99.9% availability
model:availability:rate5m < 0.999

# Platform-wide availability
(sum(model:predictions:rate5m) - sum(model:errors:rate5m)) / sum(model:predictions:rate5m)

# Availability over last 30 days
avg_over_time(model:availability:rate5m[30d])
```

### Error Budget

```promql
# Error budget remaining (30 days)
model:error_budget:remaining_30d

# Burn rate (1 hour window)
model:error_budget:burn_rate_1h

# Burning error budget too fast (>1.0 = unsustainable)
model:error_budget:burn_rate_1h > 1.0

# Error budget exhaustion time (days remaining)
model:error_budget:remaining_30d / model:error_budget:burn_rate_1h / 24
```

### Multi-Window Burn Rate Alerts

```promql
# Fast burn (1h and 5m windows both high)
(
  1 - (sum(increase(model_predictions_total[1h])) - sum(increase(model_prediction_errors_total[1h])))
      / sum(increase(model_predictions_total[1h]))
) > 0.002
and
(
  1 - (sum(increase(model_predictions_total[5m])) - sum(increase(model_prediction_errors_total[5m])))
      / sum(increase(model_predictions_total[5m]))
) > 0.002

# Slow burn (6h and 30m windows)
(
  1 - (sum(increase(model_predictions_total[6h])) - sum(increase(model_prediction_errors_total[6h])))
      / sum(increase(model_predictions_total[6h]))
) > 0.001
and
(
  1 - (sum(increase(model_predictions_total[30m])) - sum(increase(model_prediction_errors_total[30m])))
      / sum(increase(model_predictions_total[30m]))
) > 0.001
```

### Latency SLO

```promql
# Latency SLO compliance (p95 < 200ms)
model:latency_slo:compliance

# Latency SLO violation rate
count(model:latency:p95 > 0.2) / count(model:latency:p95)

# Models consistently violating latency SLO
avg_over_time(model:latency:p95[1h]) > 0.2
```

---

## Dashboard Queries

### Model Health Dashboard

```promql
# Request rate panel
model:predictions:rate5m

# Error rate panel
model:error_ratio:rate5m * 100 # Show as percentage

# Latency panel
model:latency:p95

# Availability panel
model:availability:rate5m * 100 # Show as percentage
```

### Platform Overview Dashboard

```promql
# Total throughput
platform:predictions:rate5m

# Platform error rate
platform:error_ratio:rate5m * 100

# Platform latency
platform:latency:p95

# Active models count
count(model:predictions:rate5m > 0)

# Total daily predictions
sum(increase(model_predictions_total[24h]))
```

### Resource Dashboard

```promql
# CPU usage by model
model:cpu:utilization * 100

# Memory usage by model
model:memory:utilization * 100

# Top CPU consumers
topk(10, model:cpu:utilization)

# Models near resource limits
model:memory:utilization > 0.8
```

### Business Dashboard

```promql
# Revenue by model (per hour)
topk(10, model:revenue:rate5m * 3600)

# Cost by model (per hour)
topk(10, model:cost:per_1k_predictions * model:predictions:rate5m * 3600 / 1000)

# Most profitable models
topk(10, model:profit:rate5m * 3600)

# Profit margin
(model:profit:rate5m / model:revenue:rate5m) * 100
```

---

## Troubleshooting Queries

### Debugging High Latency

```promql
# Models with p95 > 1 second
model:latency:p95 > 1.0

# Correlation with CPU usage
model:latency:p95 and model:cpu:utilization > 0.8

# Correlation with cache misses
model:latency:p95 and model:cache:miss_ratio > 0.5

# Latency by request size
histogram_quantile(0.95,
  sum by (le, model, request_size_bucket) (
    rate(model_prediction_duration_seconds_bucket[5m])
  )
)
```

### Debugging High Error Rates

```promql
# Error breakdown by type
sum by (error_type) (rate(model_prediction_errors_total[5m]))

# Errors correlated with memory usage
model:error_ratio:rate5m and model:memory:utilization > 0.9

# Errors correlated with traffic spikes
model:error_ratio:rate5m and (model:predictions:rate5m / (model:predictions:rate5m offset 1h) > 2)

# Recent deployment correlation
changes(model_info[5m]) and model:error_ratio:rate5m > 0.01
```

### Debugging Performance Degradation

```promql
# Models with degrading performance
model:latency:p95 / (model:latency:p95 offset 24h) > 1.5

# Memory leak detection
deriv(container_memory_usage_bytes{container="model-server"}[1h]) > 0

# CPU throttling
rate(container_cpu_cfs_throttled_seconds_total{container="model-server"}[5m]) > 0

# Disk I/O issues
rate(container_fs_reads_total{container="model-server"}[5m]) > 100
```

### Capacity Planning

```promql
# Current headroom (requests until 80% CPU)
(0.8 - model:cpu:utilization) * model:predictions:rate5m / model:cpu:utilization

# Predicted capacity exhaustion (days until 100% CPU at current growth)
(1 - model:cpu:utilization) / deriv(model:cpu:utilization[7d]) / 86400

# Traffic forecast (linear regression)
predict_linear(model:predictions:rate5m[7d], 86400 * 30) # 30 days ahead

# Resource requirement forecast
predict_linear(container_memory_usage_bytes{container="model-server"}[7d], 86400 * 30)
```

---

## Query Optimization Tips

### Use Recording Rules

❌ **Slow:**
```promql
histogram_quantile(0.95, sum by (le, model) (rate(model_prediction_duration_seconds_bucket[5m])))
```

✅ **Fast:**
```promql
model:latency:p95
```

### Avoid Large Time Ranges

❌ **Slow:**
```promql
rate(model_predictions_total[1d]) # Too large
```

✅ **Fast:**
```promql
rate(model_predictions_total[5m]) # 4x scrape interval
```

### Limit Label Cardinality

❌ **Slow:**
```promql
sum by (model, version, replica, pod) (rate(model_predictions_total[5m]))
```

✅ **Fast:**
```promql
sum by (model) (rate(model_predictions_total[5m]))
```

### Use Aggregation Early

❌ **Slow:**
```promql
sum(rate(model_predictions_total[5m])) by (model)
```

✅ **Fast:**
```promql
model:predictions:rate5m # Pre-aggregated
```

---

## Common Patterns

### Rate of Increase

```promql
# Requests per second
rate(model_predictions_total[5m])

# Requests per minute
rate(model_predictions_total[5m]) * 60

# Total increase over period
increase(model_predictions_total[1h])
```

### Ratios and Percentages

```promql
# Error rate as ratio
sum(rate(model_prediction_errors_total[5m])) / sum(rate(model_predictions_total[5m]))

# As percentage
(sum(rate(model_prediction_errors_total[5m])) / sum(rate(model_predictions_total[5m]))) * 100

# Success rate
1 - (errors / total)
```

### Aggregations

```promql
# Sum across dimensions
sum by (model) (rate(model_predictions_total[5m]))

# Average
avg by (model) (model_prediction_duration_seconds)

# Max/Min
max by (model) (model:latency:p95)
min by (model) (model:cache:hit_ratio)

# Count
count by (status) (model_info)
```

---

## Resources

- [PromQL Documentation](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [PromQL Functions](https://prometheus.io/docs/prometheus/latest/querying/functions/)
- [Query Examples](https://prometheus.io/docs/prometheus/latest/querying/examples/)
- [Best Practices](https://prometheus.io/docs/practices/naming/)
