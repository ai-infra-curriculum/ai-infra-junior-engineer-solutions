# LogQL Query Reference for ML Infrastructure

## Overview

LogQL is Loki's query language, similar to PromQL. This guide covers common queries for troubleshooting, analysis, and log-based metrics.

## Query Structure

```
{label_selectors} |= "search text" | parser | filter
```

## Basic Label Selectors

### By Container

```logql
# All logs from inference-gateway
{container="inference-gateway"}

# All logs from Prometheus
{container="prometheus"}

# All logs from a specific service
{service="inference-gateway"}
```

### By Component

```logql
# All application logs
{component="application"}

# All infrastructure logs
{component="infrastructure"}

# All monitoring system logs
{component="monitoring"}
```

### By Log Level

```logql
# Only ERROR logs
{container="inference-gateway"} |= `level":"ERROR"`

# Only WARNING and ERROR
{container="inference-gateway"} |~ `"level":"(ERROR|WARNING)"`

# Exclude DEBUG logs
{container="inference-gateway"} != `"level":"DEBUG"`
```

## Text Search and Filtering

### Simple Text Search

```logql
# Logs containing "prediction"
{container="inference-gateway"} |= "prediction"

# Logs NOT containing "health"
{container="inference-gateway"} != "health"

# Case-insensitive regex
{container="inference-gateway"} |~ `(?i)error`
```

### Combined Filters

```logql
# Errors during predictions
{container="inference-gateway"} |= "prediction" |= "ERROR"

# HTTP 5xx errors
{container="inference-gateway"} |= `"status_code":5`

# Slow requests (>1000ms)
{container="inference-gateway"} | json | duration_ms > 1000
```

## JSON Parsing

### Extract Fields

```logql
# Parse JSON and extract fields
{container="inference-gateway"}
  | json
  | line_format "{{.timestamp}} [{{.level}}] {{.message}}"
```

### Filter on Parsed Fields

```logql
# Requests to /predict endpoint
{container="inference-gateway"}
  | json
  | endpoint="/predict"

# Requests with specific trace ID
{container="inference-gateway"}
  | json
  | trace_id="abc123"

# High latency requests
{container="inference-gateway"}
  | json
  | duration_ms > 500
```

## Pattern Matching

### Extract Values with Regex

```logql
# Extract status code
{container="inference-gateway"}
  | regexp `"status_code":(?P<status>\d+)`
  | status >= 500
```

### Named Captures

```logql
# Extract request details
{container="inference-gateway"}
  | regexp `(?P<method>GET|POST)\s+(?P<path>/\S+)\s+(?P<status>\d+)`
  | method="POST"
  | status >= 400
```

## Log-Based Metrics

### Rate Queries

```logql
# Request rate (requests per second)
rate({container="inference-gateway"}[5m])

# Error rate
rate({container="inference-gateway"} |= "ERROR" [5m])

# Rate by endpoint
sum(rate({container="inference-gateway"} | json | __error__="" [5m])) by (endpoint)
```

### Count Queries

```logql
# Total log lines in last hour
count_over_time({container="inference-gateway"}[1h])

# Total errors in last hour
count_over_time({container="inference-gateway"} |= "ERROR" [1h])

# Count by log level
sum(count_over_time({container="inference-gateway"} | json [5m])) by (level)
```

### Percentile Queries (using parsed numeric fields)

```logql
# P99 request latency from logs
quantile_over_time(0.99,
  {container="inference-gateway"}
  | json
  | unwrap duration_ms [5m]
)

# Average request duration
avg_over_time(
  {container="inference-gateway"}
  | json
  | unwrap duration_ms [5m]
)
```

### Bytes Queries

```logql
# Total log volume
sum(bytes_over_time({container="inference-gateway"}[1h]))

# Log volume by container
sum(bytes_over_time({job="docker"}[1h])) by (container)
```

## Advanced Queries

### Trace Correlation

```logql
# Find all logs for a specific trace
{container="inference-gateway"}
  | json
  | trace_id="550e8400-e29b-41d4-a716-446655440000"

# Find traces with errors
{container="inference-gateway"}
  | json
  | level="ERROR"
  | trace_id != ""
```

### Request Analysis

```logql
# Slow requests (P95)
quantile_over_time(0.95,
  {container="inference-gateway"}
  | json
  | endpoint="/predict"
  | unwrap duration_ms [5m]
)

# Error rate by endpoint
sum(rate({container="inference-gateway"} | json | status_code >= 500 [5m])) by (endpoint)
/
sum(rate({container="inference-gateway"} | json [5m])) by (endpoint)
```

### Top-K Queries

```logql
# Top 10 slowest requests
topk(10,
  avg_over_time(
    {container="inference-gateway"}
    | json
    | unwrap duration_ms [5m]
  ) by (endpoint)
)

# Top 5 noisiest containers
topk(5,
  sum(rate({job="docker"}[5m])) by (container)
)
```

## Troubleshooting Queries

### Find Recent Errors

```logql
# Last 100 error logs
{container="inference-gateway"} |= "ERROR" | limit 100

# Errors in last 15 minutes
{container="inference-gateway"}
  |= "ERROR"
  | json
  | timestamp > now() - 15m
```

### Find Failed Requests

```logql
# HTTP 5xx errors
{container="inference-gateway"}
  | json
  | status_code >= 500
  | line_format "{{.timestamp}} [{{.status_code}}] {{.method}} {{.endpoint}} - {{.message}}"
```

### Find Missing Traces

```logql
# Requests without trace ID
{container="inference-gateway"}
  | json
  | trace_id="" or trace_id=~"^$"
```

### Debug Specific User Journey

```logql
# All logs for a specific request ID
{container="inference-gateway"}
  | json
  | request_id="550e8400-e29b-41d4-a716-446655440000"
```

## Performance Queries

### Identify Performance Bottlenecks

```logql
# Requests slower than 1 second
{container="inference-gateway"}
  | json
  | duration_ms > 1000
  | line_format "{{.timestamp}} {{.endpoint}} took {{.duration_ms}}ms"
```

### Monitor Error Budget

```logql
# Calculate error rate for SLO
(
  sum(rate({container="inference-gateway"} | json | status_code >= 500 [5m]))
  /
  sum(rate({container="inference-gateway"} | json [5m]))
) * 100
```

## Security and Audit Queries

### Find Access Attempts

```logql
# Unauthorized access attempts
{container="inference-gateway"}
  | json
  | status_code = 401 or status_code = 403

# List unique IP addresses
{container="inference-gateway"}
  | json
  | distinct client_ip
```

### Monitor Suspicious Activity

```logql
# High error rate from single source
sum(rate({container="inference-gateway"} | json | status_code >= 400 [5m])) by (client_ip)
  > 10
```

## Grafana Integration

### Creating Alerts from LogQL

```logql
# Alert on high error rate
sum(rate({container="inference-gateway"} |= "ERROR" [5m]))
  > 10
```

### Dashboard Queries

```logql
# Time series of request rate
sum(rate({container="inference-gateway"} | json [1m])) by (endpoint)

# Table of recent errors
{container="inference-gateway"}
  |= "ERROR"
  | json
  | line_format "{{.timestamp}} | {{.level}} | {{.message}}"
```

## Best Practices

### Label Selection

✅ **DO**: Use labels for high-cardinality filtering
```logql
{container="inference-gateway"} | json | endpoint="/predict"
```

❌ **DON'T**: Store high-cardinality data in labels
```logql
# BAD: request_id should not be a label
{request_id="abc123"}  # This won't work efficiently
```

### Query Optimization

✅ **DO**: Filter early with label selectors
```logql
{container="inference-gateway"} | json | status_code >= 500
```

❌ **DON'T**: Parse then filter
```logql
{job="docker"} | json | container="inference-gateway" | status_code >= 500
```

### Time Ranges

✅ **DO**: Use appropriate time ranges
```logql
rate({container="inference-gateway"}[5m])  # Good for dashboards
```

❌ **DON'T**: Query unnecessarily long ranges
```logql
rate({container="inference-gateway"}[30d])  # Too expensive
```

## Common Patterns

### Error Investigation Workflow

1. **Find the error**:
   ```logql
   {container="inference-gateway"} |= "ERROR"
   ```

2. **Extract trace ID**:
   ```logql
   {container="inference-gateway"} |= "ERROR" | json | line_format "{{.trace_id}}"
   ```

3. **Get full request context**:
   ```logql
   {container="inference-gateway"} | json | trace_id="extracted_trace_id"
   ```

4. **Check related services**:
   ```logql
   {job="docker"} | json | trace_id="extracted_trace_id"
   ```

### Performance Analysis Workflow

1. **Identify slow requests**:
   ```logql
   {container="inference-gateway"} | json | duration_ms > 1000
   ```

2. **Find patterns**:
   ```logql
   topk(10, avg_over_time({container="inference-gateway"} | json | unwrap duration_ms [5m]) by (endpoint))
   ```

3. **Drill down**:
   ```logql
   {container="inference-gateway"} | json | endpoint="/slow/endpoint" | unwrap duration_ms
   ```

## Resources

- **LogQL Documentation**: https://grafana.com/docs/loki/latest/logql/
- **LogQL Cheat Sheet**: https://grafana.com/docs/loki/latest/logql/log_queries/
- **Metric Queries**: https://grafana.com/docs/loki/latest/logql/metric_queries/

---

**Pro Tip**: Use Grafana's Explore view to build queries interactively with syntax highlighting and autocomplete!
