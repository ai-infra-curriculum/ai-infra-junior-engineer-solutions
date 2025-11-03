# Exercise 06: Log Analysis - Reflection Question Answers

## Question 1: What log levels should you monitor in production?

### Short Answer

In production ML systems, you should actively monitor **ERROR** and **CRITICAL** levels, regularly review **WARNING** levels, and selectively sample **INFO** levels. DEBUG logging should typically be disabled in production due to performance impact and log volume.

### Detailed Answer

#### Log Level Hierarchy and Production Monitoring Strategy

**1. CRITICAL (Priority: Immediate Action)**

CRITICAL log levels indicate system failures that require immediate intervention. These represent catastrophic failures that can bring down services or cause data loss.

**Examples in ML Systems:**
- System crash or core dump
- Database connection pool exhausted
- Critical resource unavailable (GPU, disk space exhausted)
- Model serving endpoint completely down
- Data pipeline critical failure

**Monitoring Strategy:**
```bash
# Real-time alerting for CRITICAL
tail -f /var/log/ml/app.log | grep --line-buffered "CRITICAL" | \
    while read line; do
        echo "$line" | mail -s "CRITICAL ALERT" oncall@example.com
        # Also send to PagerDuty/Ops Genie
    done
```

**Production Practices:**
- Alert immediately (PagerDuty, phone, SMS)
- Trigger automated incident response
- Wake up on-call engineer
- Log to incident management system
- Retention: 90+ days

**2. ERROR (Priority: High)**

ERROR indicates failures that don't crash the system but prevent successful operation. These are actionable issues that require investigation and resolution.

**Examples in ML Systems:**
- Model inference failure
- Failed prediction request
- CUDA out of memory
- Authentication failure
- Database query failure
- Failed checkpoint save
- Model file not found

**Monitoring Strategy:**
```bash
# Alert on error rate threshold
ERROR_COUNT=$(grep -c "ERROR" /var/log/ml/last_hour.log)
ERROR_THRESHOLD=50

if [ $ERROR_COUNT -gt $ERROR_THRESHOLD ]; then
    echo "Error rate exceeded: $ERROR_COUNT errors in last hour" | \
        mail -s "ML System Error Alert" team@example.com
fi
```

**Production Practices:**
- Alert on error rate (e.g., >10 errors/minute)
- Alert on specific error types (CUDA, auth, model)
- Create tickets automatically
- Aggregate similar errors
- Retention: 30-60 days
- Daily error reports

**3. WARNING (Priority: Medium)**

WARNING indicates potential problems or unusual conditions that don't prevent operation but may lead to errors if not addressed.

**Examples in ML Systems:**
- High memory usage (>80%)
- Slow inference time (>SLA threshold)
- Model accuracy degradation
- Approaching rate limits
- Deprecated API usage
- Configuration issues
- Retry attempts

**Monitoring Strategy:**
```bash
# Daily WARNING summary
grep "WARNING" /var/log/ml/today.log | \
    awk '{print $0}' | \
    sort | uniq -c | sort -rn | \
    head -20 > /reports/warnings_$(date +%Y%m%d).txt
```

**Production Practices:**
- Review daily or weekly
- Set threshold alerts (e.g., >100 warnings/hour)
- Track trends over time
- Include in team standup/retrospectives
- Retention: 14-30 days
- Weekly summary reports

**4. INFO (Priority: Low - Selective)**

INFO provides general operational information. In production, INFO logging should be selective and focus on business-critical events.

**Examples in ML Systems:**
- Training epoch completed
- Model loaded successfully
- API request completed
- Checkpoint saved
- Batch processing started/completed
- Configuration loaded

**Monitoring Strategy:**
```bash
# Sample INFO logs (not all)
# Extract key metrics only
grep "Epoch.*completed" /var/log/ml/training.log | \
    awk '{print $2, $3, $(NF-1), $NF}'  # Timestamp and metrics only
```

**Production Practices:**
- Selectively log important business events
- Use sampling (1% of requests)
- Extract metrics to separate system (Prometheus)
- Aggregate and summarize
- Short retention: 7 days
- Use for performance analysis

**5. DEBUG (Priority: Not in Production)**

DEBUG provides detailed diagnostic information useful for development but creates excessive volume and performance overhead in production.

**When to Use in Production:**
- Temporarily enable for specific troubleshooting
- Use feature flags to enable per-request
- Enable for specific user/request ID
- Route to separate log file
- Disable after investigation

**Example: Selective Debug Logging**
```python
import logging
import os

# Enable debug for specific request ID
if os.getenv('DEBUG_REQUEST_ID') == request_id:
    logger.setLevel(logging.DEBUG)
    logger.debug(f"Detailed diagnostics for {request_id}: {data}")
```

### Production Monitoring Matrix

| Log Level | Monitor Frequency | Alert Threshold | Retention | Action |
|-----------|------------------|-----------------|-----------|---------|
| CRITICAL  | Real-time        | Any occurrence  | 90+ days  | Immediate page |
| ERROR     | Real-time        | >10/min or specific types | 30-60 days | Alert team |
| WARNING   | Daily            | >100/hour trending up | 14-30 days | Review and track |
| INFO      | Weekly           | Sample 1%       | 7 days    | Extract metrics |
| DEBUG     | Never (disabled) | N/A             | N/A       | Temp enable only |

### ML-Specific Monitoring Considerations

**1. Model Performance Logs**

Monitor model-specific metrics beyond standard logs:

```python
# Model performance logging
logger.info(f"Inference completed: "
           f"latency={latency_ms}ms "
           f"confidence={confidence:.3f} "
           f"model_version={version} "
           f"gpu_memory={gpu_memory_mb}MB")
```

**Key Metrics to Log:**
- Inference latency
- Prediction confidence
- Model version
- GPU memory usage
- Batch size
- Queue depth

**2. Training Job Logs**

Training requires different monitoring:

```python
# Training progress logging
logger.info(f"Epoch {epoch}/{total}: "
           f"loss={loss:.4f} "
           f"accuracy={accuracy:.4f} "
           f"val_loss={val_loss:.4f} "
           f"learning_rate={lr:.6f} "
           f"time={epoch_time}s")

# Log warnings for potential issues
if val_loss > (train_loss * 1.2):
    logger.warning(f"Possible overfitting detected: "
                  f"val_loss={val_loss:.4f} > train_loss={train_loss:.4f}")
```

**3. Resource Usage Logs**

Track system resources:

```bash
# Log GPU utilization
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu \
    --format=csv,noheader | \
    awk '{print "INFO GPU: utilization="$1"% memory="$2" temp="$3"C"}'

# Log disk space warnings
df -h | awk '$5+0 > 80 {print "WARNING Disk space >80%:", $0}'
```

### Best Practices for Production Log Monitoring

**1. Structured Logging**

Use structured formats for easier parsing:

```python
import json
import logging

# Structured log format
log_data = {
    "timestamp": "2024-10-18T10:00:00Z",
    "level": "ERROR",
    "component": "model_server",
    "message": "Prediction failed",
    "request_id": "req_123",
    "model_version": "v2.1",
    "error_type": "ValidationError",
    "error_detail": "Invalid input shape"
}

logger.error(json.dumps(log_data))
```

**2. Correlation IDs**

Track requests across services:

```python
import uuid

# Generate correlation ID
correlation_id = str(uuid.uuid4())

# Include in all log messages
logger.info(f"[{correlation_id}] Request received")
logger.info(f"[{correlation_id}] Model prediction started")
logger.info(f"[{correlation_id}] Request completed")
```

**3. Error Rate Monitoring**

Set up automated monitoring:

```bash
#!/bin/bash
# monitor_errors.sh - Run every 5 minutes via cron

ERROR_RATE=$(grep -c "ERROR" /var/log/ml/last_5min.log)
THRESHOLD=25  # 5 errors/minute

if [ $ERROR_RATE -gt $THRESHOLD ]; then
    # Alert
    curl -X POST https://alerting.example.com/alert \
        -d "{\"severity\": \"high\", \"message\": \"Error rate: $ERROR_RATE in 5min\"}"
fi
```

**4. Log Aggregation**

Use centralized logging:

```yaml
# Fluentd configuration for ML logs
<source>
  @type tail
  path /var/log/ml/*.log
  tag ml.logs
  <parse>
    @type json
  </parse>
</source>

<filter ml.logs>
  @type record_transformer
  <record>
    hostname ${hostname}
    environment production
  </record>
</filter>

<match ml.logs>
  @type elasticsearch
  host elasticsearch.example.com
  port 9200
  index_name ml-logs
</match>
```

### Monitoring Tools and Dashboards

**1. Real-time Monitoring Dashboard**

Key metrics to display:
- Error rate (last 5min, 1hr, 24hr)
- Warning count trends
- Critical alerts (last 7 days)
- Top error types
- Service health status

**2. Alerting Rules**

```yaml
# Example Prometheus alerting rules
groups:
  - name: ml_system_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(log_messages{level="ERROR"}[5m]) > 0.1
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: CriticalError
        expr: log_messages{level="CRITICAL"} > 0
        labels:
          severity: critical
        annotations:
          summary: "Critical error occurred"
```

### Summary

**Production Monitoring Strategy:**

1. **CRITICAL**: Real-time alerts, immediate response
2. **ERROR**: Real-time monitoring with threshold alerts
3. **WARNING**: Daily review, trend monitoring
4. **INFO**: Selective logging, extract metrics
5. **DEBUG**: Disabled, enable temporarily for troubleshooting

**Key Principles:**
- Alert on what's actionable
- Reduce noise with thresholds and aggregation
- Use structured logging for parsing
- Implement correlation IDs
- Aggregate logs centrally
- Monitor trends, not just counts
- Balance detail with performance

---

## Question 2: How would you set up real-time alerting on errors?

### Short Answer

Real-time error alerting requires: (1) log aggregation and parsing, (2) error detection logic with thresholds, (3) alert routing to appropriate channels (PagerDuty, Slack, email), and (4) deduplication to prevent alert fatigue. Implement using tools like Prometheus Alertmanager, Grafana, or custom scripts with log streaming.

### Detailed Answer

#### Architecture Overview

A complete real-time alerting system consists of four layers:

```
[Logs] → [Collection] → [Processing/Detection] → [Alert Routing] → [Notification]
```

### 1. Log Collection and Streaming

**Option A: tail + grep (Simple)**

```bash
#!/bin/bash
# simple_alerter.sh - Basic real-time error alerting

LOG_FILE="/var/log/ml/api.log"
ALERT_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Follow log and alert on errors
tail -f "$LOG_FILE" | grep --line-buffered "ERROR" | \
    while read line; do
        # Extract key information
        timestamp=$(echo "$line" | awk '{print $1, $2}')
        error_msg=$(echo "$line" | sed 's/.*ERROR //')

        # Send alert
        curl -X POST "$ALERT_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"⚠️ ML System Error\n\nTime: $timestamp\nError: $error_msg\"}"

        # Also log to alert history
        echo "$timestamp | $error_msg" >> /var/log/ml/alerts.log
    done
```

**Run as systemd service:**

```ini
# /etc/systemd/system/ml-alerter.service
[Unit]
Description=ML Error Alerting Service
After=network.target

[Service]
Type=simple
User=mluser
ExecStart=/opt/ml/scripts/simple_alerter.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Option B: Fluentd/Fluent Bit (Production)**

```yaml
# fluent-bit.conf
[INPUT]
    Name              tail
    Path              /var/log/ml/*.log
    Parser            json
    Tag               ml.logs
    Refresh_Interval  5

[FILTER]
    Name              grep
    Match             ml.logs
    Regex             level ERROR|CRITICAL

[OUTPUT]
    Name              http
    Match             ml.logs
    Host              alerting-service.example.com
    Port              443
    URI               /alerts
    Format            json
    tls               On
```

### 2. Error Detection and Threshold Logic

**Simple Threshold Alerting:**

```bash
#!/bin/bash
# threshold_alerter.sh - Alert on error rate threshold

ERROR_THRESHOLD=10
TIME_WINDOW=300  # 5 minutes in seconds
ALERT_COOLDOWN=1800  # 30 minutes

LAST_ALERT_FILE="/tmp/last_alert_time"

# Count errors in time window
ERROR_COUNT=$(find /var/log/ml -name "*.log" -type f -mmin -5 -exec grep -c "ERROR" {} + | \
              awk '{sum+=$1} END {print sum}')

# Check if threshold exceeded
if [ "$ERROR_COUNT" -ge "$ERROR_THRESHOLD" ]; then
    # Check cooldown
    if [ -f "$LAST_ALERT_FILE" ]; then
        LAST_ALERT=$(cat "$LAST_ALERT_FILE")
        CURRENT_TIME=$(date +%s)
        TIME_DIFF=$((CURRENT_TIME - LAST_ALERT))

        if [ "$TIME_DIFF" -lt "$ALERT_COOLDOWN" ]; then
            echo "Alert on cooldown, skipping"
            exit 0
        fi
    fi

    # Send alert
    MESSAGE="⚠️ High error rate detected: $ERROR_COUNT errors in last 5 minutes"
    curl -X POST "https://api.pagerduty.com/incidents" \
        -H "Authorization: Token token=YOUR_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"incident\": {
                \"type\": \"incident\",
                \"title\": \"ML System: High Error Rate\",
                \"service\": {\"id\": \"YOUR_SERVICE_ID\"},
                \"body\": {\"type\": \"incident_body\", \"details\": \"$MESSAGE\"}
            }
        }"

    # Update cooldown
    date +%s > "$LAST_ALERT_FILE"
fi
```

**Run via cron:**

```cron
# Check every 5 minutes
*/5 * * * * /opt/ml/scripts/threshold_alerter.sh
```

**Advanced: Python Error Classifier:**

```python
#!/usr/bin/env python3
"""
error_classifier.py - Classify and prioritize errors for alerting
"""

import re
import json
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Error patterns and severities
ERROR_PATTERNS = {
    'critical': [
        r'CRITICAL',
        r'segmentation fault',
        r'core dumped',
        r'out of memory',
        r'disk.*full',
        r'cuda.*out of memory',
    ],
    'high': [
        r'authentication.*failed',
        r'connection refused',
        r'database.*error',
        r'model.*not found',
    ],
    'medium': [
        r'timeout',
        r'retry',
        r'invalid.*input',
    ]
}

# Alert thresholds
THRESHOLDS = {
    'critical': {'count': 1, 'window': 60},      # Alert on any
    'high': {'count': 5, 'window': 300},         # 5 in 5 min
    'medium': {'count': 20, 'window': 600},      # 20 in 10 min
}

# State tracking
error_counts = defaultdict(lambda: defaultdict(list))

def classify_error(log_line):
    """Classify error by severity."""
    for severity, patterns in ERROR_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, log_line, re.IGNORECASE):
                return severity
    return 'low'

def should_alert(severity, error_type):
    """Check if alert should be sent based on thresholds."""
    threshold = THRESHOLDS.get(severity, {'count': float('inf'), 'window': 60})
    now = datetime.now()
    window_start = now - timedelta(seconds=threshold['window'])

    # Get recent errors of this type
    recent_errors = error_counts[severity][error_type]

    # Remove old errors outside window
    recent_errors = [t for t in recent_errors if t > window_start]
    error_counts[severity][error_type] = recent_errors

    # Add current error
    recent_errors.append(now)

    # Check threshold
    return len(recent_errors) >= threshold['count']

def send_alert(severity, error_type, message, log_line):
    """Send alert via multiple channels."""
    alert_data = {
        'severity': severity,
        'error_type': error_type,
        'message': message,
        'log_line': log_line,
        'timestamp': datetime.now().isoformat(),
        'count': len(error_counts[severity][error_type])
    }

    # Send to Slack
    slack_webhook = "https://hooks.slack.com/services/YOUR/WEBHOOK"
    slack_payload = {
        'text': f":warning: {severity.upper()} Error Alert",
        'attachments': [{
            'color': 'danger' if severity == 'critical' else 'warning',
            'fields': [
                {'title': 'Error Type', 'value': error_type, 'short': True},
                {'title': 'Severity', 'value': severity, 'short': True},
                {'title': 'Message', 'value': message},
                {'title': 'Log Line', 'value': log_line[:200]}
            ]
        }]
    }
    requests.post(slack_webhook, json=slack_payload)

    # Send to PagerDuty for critical
    if severity == 'critical':
        pagerduty_api = "https://api.pagerduty.com/incidents"
        pagerduty_payload = {
            'incident': {
                'type': 'incident',
                'title': f"CRITICAL: {error_type}",
                'service': {'id': 'YOUR_SERVICE_ID'},
                'urgency': 'high',
                'body': {
                    'type': 'incident_body',
                    'details': json.dumps(alert_data)
                }
            }
        }
        headers = {
            'Authorization': 'Token token=YOUR_TOKEN',
            'Content-Type': 'application/json'
        }
        requests.post(pagerduty_api, json=pagerduty_payload, headers=headers)

    logger.info(f"Alert sent: {severity} - {error_type}")

def process_log_line(line):
    """Process a log line and alert if necessary."""
    if 'ERROR' not in line:
        return

    # Classify error
    severity = classify_error(line)

    # Extract error type
    match = re.search(r'ERROR\s+([^:]+)', line)
    error_type = match.group(1).strip() if match else 'unknown'

    # Extract message
    match = re.search(r'ERROR\s+(.+)', line)
    message = match.group(1).strip() if match else line

    # Check if alert should be sent
    if should_alert(severity, error_type):
        send_alert(severity, error_type, message, line)

if __name__ == '__main__':
    import sys
    import time

    # Read from stdin (use with tail -f)
    for line in sys.stdin:
        process_log_line(line.strip())
```

**Usage:**

```bash
# Run with log streaming
tail -f /var/log/ml/*.log | python3 error_classifier.py
```

### 3. Alert Routing and Channels

**Multi-Channel Alerting Script:**

```bash
#!/bin/bash
# multi_channel_alert.sh - Send alerts to multiple destinations

SEVERITY="$1"
MESSAGE="$2"
ERROR_DETAILS="$3"

# Configuration
SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK"
PAGERDUTY_TOKEN="YOUR_PAGERDUTY_TOKEN"
PAGERDUTY_SERVICE_ID="YOUR_SERVICE_ID"
EMAIL_TO="oncall@example.com"
SMS_API="https://api.twilio.com/2010-04-01/Accounts/YOUR_ACCOUNT/Messages.json"

send_slack() {
    local severity=$1
    local message=$2

    # Set color based on severity
    local color="warning"
    [[ "$severity" == "critical" ]] && color="danger"

    curl -X POST "$SLACK_WEBHOOK" \
        -H 'Content-Type: application/json' \
        -d "{
            \"text\": \"🚨 ML System Alert\",
            \"attachments\": [{
                \"color\": \"$color\",
                \"fields\": [
                    {\"title\": \"Severity\", \"value\": \"$severity\", \"short\": true},
                    {\"title\": \"Time\", \"value\": \"$(date)\", \"short\": true},
                    {\"title\": \"Message\", \"value\": \"$message\"}
                ]
            }]
        }"
}

send_pagerduty() {
    local severity=$1
    local message=$2

    curl -X POST "https://api.pagerduty.com/incidents" \
        -H "Authorization: Token token=$PAGERDUTY_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"incident\": {
                \"type\": \"incident\",
                \"title\": \"ML System: $message\",
                \"service\": {\"id\": \"$PAGERDUTY_SERVICE_ID\"},
                \"urgency\": \"high\",
                \"body\": {
                    \"type\": \"incident_body\",
                    \"details\": \"$ERROR_DETAILS\"
                }
            }
        }"
}

send_email() {
    local severity=$1
    local message=$2

    echo "Severity: $severity
Time: $(date)
Message: $message

Details:
$ERROR_DETAILS" | mail -s "ML System Alert: $severity" "$EMAIL_TO"
}

send_sms() {
    local message=$1

    curl -X POST "$SMS_API" \
        --data-urlencode "To=+1234567890" \
        --data-urlencode "From=+0987654321" \
        --data-urlencode "Body=ML Alert: $message" \
        -u YOUR_ACCOUNT_SID:YOUR_AUTH_TOKEN
}

# Route alerts based on severity
case "$SEVERITY" in
    critical)
        send_pagerduty "$SEVERITY" "$MESSAGE"
        send_slack "$SEVERITY" "$MESSAGE"
        send_email "$SEVERITY" "$MESSAGE"
        send_sms "$MESSAGE"
        ;;
    high)
        send_slack "$SEVERITY" "$MESSAGE"
        send_email "$SEVERITY" "$MESSAGE"
        ;;
    medium)
        send_slack "$SEVERITY" "$MESSAGE"
        ;;
    *)
        # Log only for low severity
        logger -t ml-alert "[$SEVERITY] $MESSAGE"
        ;;
esac
```

### 4. Alert Deduplication and Throttling

**Deduplication Script:**

```python
#!/usr/bin/env python3
"""
alert_deduplicator.py - Prevent duplicate alerts
"""

import hashlib
import time
import json
from pathlib import Path

ALERT_STATE_FILE = '/tmp/alert_state.json'
DEDUP_WINDOW = 1800  # 30 minutes

def load_alert_state():
    """Load previous alert state."""
    if Path(ALERT_STATE_FILE).exists():
        with open(ALERT_STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_alert_state(state):
    """Save alert state."""
    with open(ALERT_STATE_FILE, 'w') as f:
        json.dump(state, f)

def generate_alert_key(error_type, message):
    """Generate unique key for alert."""
    content = f"{error_type}:{message}"
    return hashlib.md5(content.encode()).hexdigest()

def should_send_alert(error_type, message):
    """Check if alert should be sent (not duplicate)."""
    alert_key = generate_alert_key(error_type, message)
    state = load_alert_state()

    current_time = time.time()

    if alert_key in state:
        last_alert_time = state[alert_key]
        time_since_last = current_time - last_alert_time

        if time_since_last < DEDUP_WINDOW:
            print(f"Suppressing duplicate alert (last sent {int(time_since_last)}s ago)")
            return False

    # Update state
    state[alert_key] = current_time

    # Clean up old entries
    state = {k: v for k, v in state.items() if current_time - v < DEDUP_WINDOW}

    save_alert_state(state)
    return True

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print("Usage: alert_deduplicator.py <error_type> <message>")
        sys.exit(1)

    error_type = sys.argv[1]
    message = sys.argv[2]

    if should_send_alert(error_type, message):
        print("SEND_ALERT")
        sys.exit(0)
    else:
        print("SUPPRESS_ALERT")
        sys.exit(1)
```

**Integration:**

```bash
# Use deduplicator before sending alerts
if python3 alert_deduplicator.py "$ERROR_TYPE" "$MESSAGE"; then
    ./multi_channel_alert.sh "$SEVERITY" "$MESSAGE" "$DETAILS"
fi
```

### 5. Production-Grade Solutions

**Option A: Prometheus + Alertmanager**

```yaml
# prometheus_rules.yml
groups:
  - name: ml_system_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          rate(log_errors_total{level="ERROR"}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
          component: ml_system
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"

      - alert: CriticalError
        expr: |
          log_errors_total{level="CRITICAL"} > 0
        labels:
          severity: critical
          component: ml_system
        annotations:
          summary: "Critical error occurred"
          description: "Critical error detected in ML system"

      - alert: ModelInferenceFailure
        expr: |
          rate(model_inference_errors_total[5m]) > 0.05
        for: 1m
        labels:
          severity: high
        annotations:
          summary: "Model inference failures detected"

      - alert: GPUMemoryExhausted
        expr: |
          increase(gpu_oom_errors_total[5m]) > 0
        labels:
          severity: critical
        annotations:
          summary: "GPU out of memory errors"
```

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  receiver: 'default'
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h

  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true

    - match:
        severity: high
      receiver: 'slack-high'

    - match:
        severity: warning
      receiver: 'slack-warnings'

receivers:
  - name: 'default'
    email_configs:
      - to: 'team@example.com'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_SERVICE_KEY'
        description: '{{ .CommonAnnotations.summary }}'

  - name: 'slack-high'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK'
        channel: '#ml-alerts'
        title: 'High Severity Alert'
        text: '{{ .CommonAnnotations.description }}'

  - name: 'slack-warnings'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK'
        channel: '#ml-warnings'
```

**Option B: Grafana Loki**

```yaml
# loki-config.yml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h

# Alert rules
ruler:
  alertmanager_url: http://alertmanager:9093
  ring:
    kvstore:
      store: inmemory
  rule_path: /tmp/loki/rules
  storage:
    type: local
    local:
      directory: /loki/rules
```

```yaml
# loki-rules.yml
groups:
  - name: ml_errors
    interval: 1m
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate({job="ml-app"} |= "ERROR" [5m])) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in ML application"

      - alert: CriticalErrors
        expr: |
          count_over_time({job="ml-app"} |= "CRITICAL" [1m]) > 0
        labels:
          severity: critical
        annotations:
          summary: "Critical errors detected"
```

### 6. Best Practices

**1. Alert Hierarchy**

```
CRITICAL → PagerDuty + Slack + Email + SMS
HIGH     → Slack + Email
MEDIUM   → Slack
LOW      → Log only
```

**2. Alert Content**

Include in every alert:
- Timestamp
- Severity level
- Error type/category
- Affected component
- Error message
- Count/frequency
- Runbook link
- Quick action buttons

**3. Testing**

```bash
# Test alert pipeline
./test_alert.sh critical "Test critical alert"
./test_alert.sh high "Test high priority alert"
```

**4. Monitoring the Monitor**

```bash
# Watchdog - ensure alerting is working
#!/bin/bash
# watchdog.sh

LAST_HEARTBEAT="/tmp/alerting_heartbeat"

# Check if alerting system is alive
if [ -f "$LAST_HEARTBEAT" ]; then
    LAST_TIME=$(cat "$LAST_HEARTBEAT")
    CURRENT_TIME=$(date +%s)
    DIFF=$((CURRENT_TIME - LAST_TIME))

    if [ $DIFF -gt 600 ]; then  # 10 minutes
        # Alerting system is down
        curl -X POST "https://backup-alert.example.com/deadman" \
            -d "Alerting system has not sent heartbeat for $DIFF seconds"
    fi
fi
```

### Summary

**Real-Time Alerting Architecture:**

1. **Collection**: tail -f, Fluentd, Promtail
2. **Processing**: grep, Python classifier, Prometheus
3. **Detection**: Thresholds, patterns, ML anomaly detection
4. **Deduplication**: State tracking, time windows
5. **Routing**: Severity-based multi-channel
6. **Delivery**: Slack, PagerDuty, Email, SMS

**Key Principles:**
- Alert on actionable items only
- Use severity-based routing
- Implement deduplication
- Monitor the monitoring system
- Test alert pipelines
- Include context in alerts
- Provide runbooks

---

## Question 3: Why is log rotation important?

### Short Answer

Log rotation is critical for: (1) preventing disk space exhaustion, (2) maintaining system performance, (3) managing retention policies, (4) enabling efficient log analysis, and (5) complying with data retention regulations. Without rotation, logs grow unbounded, eventually filling disks and causing system failures.

### Detailed Answer

#### 1. Disk Space Management

**The Problem:**

Without log rotation, logs grow indefinitely:

```bash
# ML training log growth example
$ du -sh /var/log/ml/training.log
150G    /var/log/ml/training.log

# Disk space exhausted
$ df -h /var/log
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1       200G  200G     0 100% /var/log
```

**Real-World Scenario:**

```
Day 1:   training.log = 100 MB
Day 7:   training.log = 700 MB
Day 30:  training.log = 3 GB
Day 90:  training.log = 9 GB
Day 365: training.log = 36 GB  ← Disk full, system failure
```

**Impact of Full Disk:**
- New logs cannot be written
- Applications crash
- Database writes fail
- Model checkpoints cannot be saved
- System becomes unresponsive

**Solution with Rotation:**

```bash
# With daily rotation, keeping 7 days
$ ls -lh /var/log/ml/
-rw-r--r-- 1 mluser mluser  120M Oct 18 23:59 training.log
-rw-r--r-- 1 mluser mluser   98M Oct 17 23:59 training.log.1.gz
-rw-r--r-- 1 mluser mluser  102M Oct 16 23:59 training.log.2.gz
-rw-r--r-- 1 mluser mluser  105M Oct 15 23:59 training.log.3.gz
...
Total: ~800 MB instead of 36 GB
```

#### 2. Performance Impact

**Large Files Slow Operations:**

```bash
# Searching a 10 GB log file
$ time grep "ERROR" large.log
real    2m 45s   ← Very slow

# Searching rotated logs (1 GB each)
$ time grep "ERROR" training.log
real    0m 15s   ← Much faster
```

**Performance Comparison:**

| File Size | grep Time | tail Time | gzip Time | Open Time |
|-----------|-----------|-----------|-----------|-----------|
| 100 MB    | 1s        | <1s       | 3s        | <1s       |
| 1 GB      | 12s       | 1s        | 35s       | 2s        |
| 10 GB     | 2m 15s    | 15s       | 6m        | 20s       |
| 100 GB    | 25m       | 3m        | 1h        | 5m        |

**Why Smaller Files Are Faster:**
- Less data to scan
- Better cache utilization
- Faster I/O operations
- Parallel processing possible
- Reduced memory footprint

#### 3. Retention Policy Management

**Compliance Requirements:**

Different data types have different retention needs:

```bash
# Security audit logs: 1 year
/var/log/auth.log {
    weekly
    rotate 52
    compress
}

# Application logs: 30 days
/var/log/ml/app.log {
    daily
    rotate 30
    compress
}

# Debug logs: 7 days
/var/log/ml/debug.log {
    daily
    rotate 7
    compress
}

# Training metrics: Permanent (archive)
/var/log/ml/training.log {
    daily
    rotate 365
    compress
    olddir /archive/ml/training/
    createolddir 755 mluser mluser
}
```

**Legal Compliance Examples:**

| Regulation | Retention Requirement | Log Types |
|------------|----------------------|-----------|
| GDPR       | Delete personal data on request | User activity, API requests |
| HIPAA      | 6 years | Healthcare ML application logs |
| SOX        | 7 years | Financial transaction logs |
| PCI-DSS    | 1 year (3 months readily available) | Payment processing logs |

#### 4. Efficient Log Analysis

**Targeted Analysis:**

```bash
# Without rotation: Search everything
$ grep "ERROR" huge.log | wc -l
15234  ← Mixed old and new errors

# With rotation: Search specific time period
$ zgrep "ERROR" training.log.2.gz | wc -l  # 2 days ago
45     ← Specific to that day

# Analyze trends over time
$ for i in {0..7}; do
    echo -n "Day $i: "
    zgrep -c "ERROR" training.log.$i.gz
done
Day 0: 23
Day 1: 45   ← Spike!
Day 2: 28
Day 3: 25
...
```

**Time-Based Queries:**

```bash
# Find errors in specific time window
$ zgrep "2024-10-15" training.log.3.gz | grep "ERROR"

# Compare performance across days
$ for log in training.log*.gz; do
    echo "$log:"
    zgrep "average_loss" "$log" | tail -1
done
```

#### 5. Backup and Archival

**Archival Strategy:**

```bash
#!/bin/bash
# archive_logs.sh - Archive old logs to S3

ARCHIVE_DIR="/var/log/ml/archive"
S3_BUCKET="s3://company-ml-logs"
RETENTION_DAYS=90

# Find logs older than retention period
find "$ARCHIVE_DIR" -name "*.log.gz" -mtime +$RETENTION_DAYS -type f | \
    while read log_file; do
        # Upload to S3
        aws s3 cp "$log_file" "$S3_BUCKET/$(basename $log_file)" \
            --storage-class GLACIER

        # Delete local copy after successful upload
        if [ $? -eq 0 ]; then
            rm "$log_file"
            echo "Archived and removed: $log_file"
        fi
    done
```

**Tiered Storage:**

```
Hot Storage (Local SSD):     Last 7 days   → Fast access
Warm Storage (Local HDD):    8-30 days     → Medium access
Cold Storage (S3 Standard):  31-90 days    → Slow access
Archive (S3 Glacier):        90+ days      → Very slow (hours)
```

#### 6. Log Rotation Configuration

**logrotate Configuration:**

```bash
# /etc/logrotate.d/ml-training
/var/log/ml/training/*.log {
    # Rotation frequency
    daily                    # Options: daily, weekly, monthly, yearly
                            # or size 100M (rotate when > 100MB)

    # Retention
    rotate 30                # Keep 30 rotations
    maxage 90                # Delete files older than 90 days

    # Compression
    compress                 # Compress rotated logs
    delaycompress            # Don't compress most recent rotation
    compresscmd /bin/gzip    # Compression command
    compressext .gz          # Compression extension
    compressoptions -9       # Best compression

    # File handling
    missingok                # Don't error if log is missing
    notifempty               # Don't rotate if empty
    create 0640 mluser mluser # Create new log with permissions
    sharedscripts            # Run postrotate once for all logs

    # Actions
    prerotate
        # Run before rotation
        /usr/bin/systemctl status ml-training > /dev/null || exit 1
    endscript

    postrotate
        # Reload application to use new log file
        /usr/bin/systemctl reload ml-training > /dev/null 2>&1 || true
    endscript

    # Advanced options
    dateext                  # Add date to rotated filename
    dateformat -%Y%m%d       # Date format: training.log-20241018
    extension .log           # Original file extension
    olddir /var/log/ml/archive # Move rotated logs to separate directory
    createolddir 0755 mluser mluser # Create olddir if missing
}
```

**Size-Based Rotation:**

```bash
# Rotate when file reaches 100MB
/var/log/ml/api.log {
    size 100M
    rotate 10
    compress
    delaycompress
    missingok
    notifempty
    create 0640 mluser mluser
}
```

**Hourly Rotation (High-Volume):**

```bash
# For very high-volume logs
/var/log/ml/high-traffic-api.log {
    hourly
    rotate 168    # 7 days * 24 hours
    compress
    delaycompress
    missingok
    notifempty
    create 0640 mluser mluser
    dateext
    dateformat -%Y%m%d-%H
}
```

#### 7. Testing Log Rotation

```bash
# Test configuration (dry-run)
sudo logrotate -d /etc/logrotate.d/ml-training

# Force rotation immediately
sudo logrotate -f /etc/logrotate.d/ml-training

# Check rotation status
cat /var/lib/logrotate/status | grep ml-training
```

#### 8. Custom Rotation Scripts

**Manual Rotation Script:**

```bash
#!/bin/bash
# manual_rotate.sh - Custom log rotation

LOG_FILE="/var/log/ml/training.log"
ARCHIVE_DIR="/var/log/ml/archive"
RETENTION_DAYS=30

# Create archive directory
mkdir -p "$ARCHIVE_DIR"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE_FILE="$ARCHIVE_DIR/training-$TIMESTAMP.log"

# Copy and compress current log
cp "$LOG_FILE" "$ARCHIVE_FILE"
gzip "$ARCHIVE_FILE"

# Truncate current log
> "$LOG_FILE"

# Reload application
systemctl reload ml-training

# Delete old archives
find "$ARCHIVE_DIR" -name "training-*.log.gz" -mtime +$RETENTION_DAYS -delete

echo "Rotated: $LOG_FILE → $ARCHIVE_FILE.gz"
```

**Python Rotation Script:**

```python
#!/usr/bin/env python3
"""
smart_rotate.py - Intelligent log rotation based on multiple criteria
"""

import os
import gzip
import shutil
from pathlib import Path
from datetime import datetime, timedelta

class LogRotator:
    def __init__(self, log_file, max_size_mb=100, max_age_days=7, max_count=10):
        self.log_file = Path(log_file)
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self.max_age = timedelta(days=max_age_days)
        self.max_count = max_count

    def should_rotate(self):
        """Check if rotation is needed."""
        if not self.log_file.exists():
            return False

        # Check size
        size = self.log_file.stat().st_size
        if size >= self.max_size:
            return True, 'size'

        # Check age
        mtime = datetime.fromtimestamp(self.log_file.stat().st_mtime)
        age = datetime.now() - mtime
        if age >= self.max_age:
            return True, 'age'

        return False, None

    def rotate(self):
        """Perform log rotation."""
        if not self.log_file.exists():
            return

        # Generate archive name
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        archive_name = f"{self.log_file}.{timestamp}.gz"

        # Compress and archive
        with open(self.log_file, 'rb') as f_in:
            with gzip.open(archive_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Truncate current log
        open(self.log_file, 'w').close()

        print(f"Rotated: {self.log_file} → {archive_name}")

        # Cleanup old archives
        self.cleanup_old()

    def cleanup_old(self):
        """Remove old log archives."""
        pattern = f"{self.log_file.name}.*.gz"
        archives = sorted(self.log_file.parent.glob(pattern))

        # Remove if exceeds count
        if len(archives) > self.max_count:
            for archive in archives[:-self.max_count]:
                archive.unlink()
                print(f"Removed old archive: {archive}")

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: smart_rotate.py <log_file>")
        sys.exit(1)

    rotator = LogRotator(sys.argv[1])
    should_rotate, reason = rotator.should_rotate()

    if should_rotate:
        print(f"Rotation needed: {reason}")
        rotator.rotate()
    else:
        print("Rotation not needed")
```

#### 9. Monitoring Log Rotation

```bash
# Check if rotation is working
#!/bin/bash
# check_rotation.sh

LOG_FILE="/var/log/ml/training.log"
ARCHIVE_DIR="/var/log/ml/archive"

# Check current log size
CURRENT_SIZE=$(du -m "$LOG_FILE" | cut -f1)
echo "Current log size: ${CURRENT_SIZE}MB"

# Check age
AGE_DAYS=$(echo "($(date +%s) - $(stat -c %Y "$LOG_FILE")) / 86400" | bc)
echo "Current log age: $AGE_DAYS days"

# Check archived logs
ARCHIVE_COUNT=$(ls -1 "$ARCHIVE_DIR"/*.gz 2>/dev/null | wc -l)
echo "Archived logs: $ARCHIVE_COUNT files"

# Check oldest archive
if [ $ARCHIVE_COUNT -gt 0 ]; then
    OLDEST=$(ls -t "$ARCHIVE_DIR"/*.gz | tail -1)
    OLDEST_DAYS=$(echo "($(date +%s) - $(stat -c %Y "$OLDEST")) / 86400" | bc)
    echo "Oldest archive: $OLDEST_DAYS days old"
fi

# Alert if issues
if [ $CURRENT_SIZE -gt 500 ]; then
    echo "WARNING: Log file is too large!"
fi

if [ $AGE_DAYS -gt 2 ]; then
    echo "WARNING: Log file is old, rotation may not be working!"
fi
```

#### 10. Best Practices

**1. Choose Appropriate Rotation Strategy:**

- **High-frequency training**: Daily or size-based (100MB)
- **API logs**: Hourly (if high traffic) or daily
- **Error logs**: Daily with longer retention
- **Debug logs**: Hourly with short retention (7 days)

**2. Compression:**

Always compress rotated logs:
```bash
compress
delaycompress      # Keep most recent uncompressed for active reading
compressoptions -9 # Maximum compression
```

**3. Separate Archives:**

```bash
olddir /var/log/ml/archive
createolddir 0755 mluser mluser
```

**4. Test Rotation:**

```bash
# Weekly rotation test
0 2 * * 0 /usr/sbin/logrotate -f /etc/logrotate.d/ml-app
```

**5. Monitor Disk Space:**

```bash
# Alert on low disk space
df -h /var/log | awk '$5+0 > 80 {print "WARNING: Disk space >80%:", $0}'
```

### Summary

**Why Log Rotation is Critical:**

1. **Prevents disk exhaustion** → System stability
2. **Maintains performance** → Fast log operations
3. **Manages retention** → Compliance and efficiency
4. **Enables analysis** → Targeted time-based queries
5. **Facilitates backups** → Manageable archive sizes

**Key Rotation Parameters:**
- **Frequency**: Daily (most common), hourly (high-volume), or size-based
- **Retention**: 7-90 days (application), 1+ years (compliance)
- **Compression**: Always enabled (saves 80-90% space)
- **Archive location**: Separate directory for rotated logs

**Essential Commands:**
- `logrotate -d`: Test configuration
- `logrotate -f`: Force rotation
- `zgrep`: Search compressed logs
- `zcat`: View compressed logs

**Without rotation**: System failure inevitable
**With rotation**: System runs smoothly indefinitely

---

## Question 4: How can you correlate logs across multiple services?

### Short Answer

Correlate logs across services using: (1) **correlation IDs** (unique request identifiers propagated through all services), (2) **centralized logging** (aggregate logs from all services into one system), (3) **structured logging** (consistent JSON format with shared fields), and (4) **timestamp synchronization** (ensure all services use synchronized time).

### Detailed Answer

#### 1. Correlation IDs (Request Tracing)

**The Problem:**

Without correlation, distributed requests are impossible to trace:

```
Service A: [10:00:01] User login attempt
Service B: [10:00:02] Database query executed
Service C: [10:00:02] Cache miss
Service B: [10:00:03] Query result returned
Service A: [10:00:04] Login successful

Question: Which logs belong to the same request? → Can't tell!
```

**Solution: Correlation IDs**

Generate a unique ID for each request and pass it through all services:

```python
# service_a.py - API Gateway
import uuid
import logging
import requests

def handle_login_request():
    # Generate correlation ID
    correlation_id = str(uuid.uuid4())

    # Include in all log messages
    logger.info(f"[{correlation_id}] User login attempt",
                extra={'correlation_id': correlation_id})

    # Pass to downstream services
    headers = {'X-Correlation-ID': correlation_id}
    response = requests.post('http://service-b/auth',
                            headers=headers,
                            json={'username': 'user'})

    logger.info(f"[{correlation_id}] Login successful",
                extra={'correlation_id': correlation_id})

    return response
```

```python
# service_b.py - Auth Service
def authenticate(request):
    # Extract correlation ID from headers
    correlation_id = request.headers.get('X-Correlation-ID', 'unknown')

    logger.info(f"[{correlation_id}] Database query started",
                extra={'correlation_id': correlation_id})

    # Query database
    result = db.query("SELECT * FROM users WHERE username = ?", username)

    logger.info(f"[{correlation_id}] Database query completed",
                extra={'correlation_id': correlation_id})

    # Pass correlation ID to cache service
    headers = {'X-Correlation-ID': correlation_id}
    cache_result = requests.get('http://service-c/cache',
                                headers=headers)

    return result
```

```python
# service_c.py - Cache Service
def get_cache(request):
    correlation_id = request.headers.get('X-Correlation-ID', 'unknown')

    logger.info(f"[{correlation_id}] Cache lookup",
                extra={'correlation_id': correlation_id})

    # Cache operations...

    return cached_data
```

**Correlated Logs:**

```
Service A: [10:00:01] [abc123] User login attempt
Service B: [10:00:02] [abc123] Database query started
Service C: [10:00:02] [abc123] Cache lookup
Service C: [10:00:02] [abc123] Cache miss
Service B: [10:00:03] [abc123] Database query completed
Service A: [10:00:04] [abc123] Login successful
```

**Searching Correlated Logs:**

```bash
# Find all logs for specific request
grep "abc123" /var/log/ml/*.log

# Or with centralized logging
curl "http://elasticsearch:9200/logs/_search" \
    -d '{"query": {"match": {"correlation_id": "abc123"}}}'
```

#### 2. Centralized Logging Architecture

**ELK Stack (Elasticsearch, Logstash, Kibana):**

```yaml
# filebeat.yml - Log shipper on each service
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/ml/*.log
    json.keys_under_root: true
    json.add_error_key: true
    fields:
      service: "ml-training"
      environment: "production"
    fields_under_root: true

output.logstash:
  hosts: ["logstash:5044"]
```

```conf
# logstash.conf - Log processor
input {
  beats {
    port => 5044
  }
}

filter {
  # Parse JSON logs
  json {
    source => "message"
  }

  # Extract correlation ID
  if [correlation_id] {
    mutate {
      add_tag => ["correlated"]
    }
  }

  # Add timestamp
  date {
    match => ["timestamp", "ISO8601"]
    target => "@timestamp"
  }

  # Enrich with service metadata
  mutate {
    add_field => {
      "service_name" => "%{[fields][service]}"
      "environment" => "%{[fields][environment]}"
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "logs-%{[service_name]}-%{+YYYY.MM.dd}"
  }
}
```

**Querying Correlated Logs in Kibana:**

```
# Search across all services for correlation ID
correlation_id: "abc123"

# Filter by time and service
correlation_id: "abc123" AND service: "ml-training" AND timestamp: [2024-10-18 TO 2024-10-19]

# Build request flow visualization
correlation_id: "abc123" | sort @timestamp
```

#### 3. Structured Logging

**Consistent Log Format:**

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, service_name):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)

    def log(self, level, message, **kwargs):
        """Create structured log entry."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'service': self.service_name,
            'level': level,
            'message': message,
            'correlation_id': kwargs.get('correlation_id', 'unknown'),
            'request_id': kwargs.get('request_id'),
            'user_id': kwargs.get('user_id'),
            'session_id': kwargs.get('session_id'),
            'duration_ms': kwargs.get('duration_ms'),
            'status_code': kwargs.get('status_code'),
            'error_type': kwargs.get('error_type'),
            'metadata': kwargs.get('metadata', {})
        }

        # Remove None values
        log_entry = {k: v for k, v in log_entry.items() if v is not None}

        # Output as JSON
        self.logger.log(
            getattr(logging, level),
            json.dumps(log_entry)
        )

# Usage
logger = StructuredLogger('ml-training')

logger.log('INFO', 'Training epoch completed',
          correlation_id='abc123',
          user_id='user_456',
          metadata={'epoch': 10, 'loss': 0.123})
```

**Log Output:**

```json
{
  "timestamp": "2024-10-18T10:00:00Z",
  "service": "ml-training",
  "level": "INFO",
  "message": "Training epoch completed",
  "correlation_id": "abc123",
  "user_id": "user_456",
  "metadata": {
    "epoch": 10,
    "loss": 0.123
  }
}
```

**Benefits:**
- Easy to parse programmatically
- Consistent fields across services
- Searchable in log aggregation systems
- Can be deserialized for analysis

#### 4. Distributed Tracing

**OpenTelemetry Integration:**

```python
# tracer.py - OpenTelemetry setup
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.flask import FlaskInstrumentor

# Set up tracer
resource = Resource(attributes={
    "service.name": "ml-training"
})

trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# Export to Jaeger
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Auto-instrument HTTP requests
RequestsInstrumentor().instrument()
FlaskInstrumentor().instrument_app(app)
```

```python
# ml_training.py - Instrumented code
from tracer import tracer
import requests

@app.route('/train', methods=['POST'])
def train_model():
    with tracer.start_as_current_span("train_model") as span:
        span.set_attribute("model.type", "resnet50")
        span.set_attribute("batch.size", 32)

        # This span is automatically included in the trace
        data = load_data()

        with tracer.start_as_current_span("model_inference"):
            result = model.fit(data)

        # Call downstream service (automatically traced)
        response = requests.post('http://model-registry/save',
                                json={'model': result})

        span.set_attribute("status", "success")
        return {"status": "completed"}

def load_data():
    with tracer.start_as_current_span("load_data") as span:
        # Fetch from data service (automatically traced)
        response = requests.get('http://data-service/dataset')
        span.set_attribute("dataset.size", len(response.json()))
        return response.json()
```

**Trace Visualization (Jaeger UI):**

```
Trace ID: abc123def456

┌─ train_model (200ms)
│  ├─ load_data (50ms)
│  │  └─ HTTP GET /dataset (45ms)  [data-service]
│  ├─ model_inference (120ms)
│  └─ HTTP POST /save (25ms)  [model-registry]
```

#### 5. Timestamp Synchronization

**Problem: Clock Skew**

```
Service A [10:00:05] Request sent
Service B [09:59:58] Request received  ← Clock 7 seconds behind!
```

**Solution: NTP Synchronization**

```bash
# Install and configure NTP
sudo apt install ntp

# Configure NTP servers
sudo nano /etc/ntp.conf
# Add:
server 0.pool.ntp.org iburst
server 1.pool.ntp.org iburst
server 2.pool.ntp.org iburst
server 3.pool.ntp.org iburst

# Restart NTP
sudo systemctl restart ntp

# Check sync status
ntpq -p
```

**Use UTC in All Logs:**

```python
from datetime import datetime, timezone

# Always use UTC
timestamp = datetime.now(timezone.utc).isoformat()

# Don't use local time!
# timestamp = datetime.now().isoformat()  ← Wrong!
```

#### 6. Practical Log Correlation Examples

**Example 1: ML Training Pipeline**

```python
# orchestrator.py - Training orchestrator
import uuid
import logging
import requests

class MLTrainingOrchestrator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def start_training(self, config):
        # Generate correlation ID
        training_id = str(uuid.uuid4())

        self.logger.info(f"[{training_id}] Training started",
                        extra={'correlation_id': training_id,
                              'config': config})

        # Step 1: Prepare data
        self.logger.info(f"[{training_id}] Data preparation")
        data_result = requests.post(
            'http://data-service/prepare',
            headers={'X-Correlation-ID': training_id},
            json=config
        )

        # Step 2: Train model
        self.logger.info(f"[{training_id}] Model training")
        training_result = requests.post(
            'http://training-service/train',
            headers={'X-Correlation-ID': training_id},
            json={'data_id': data_result.json()['id']}
        )

        # Step 3: Evaluate model
        self.logger.info(f"[{training_id}] Model evaluation")
        eval_result = requests.post(
            'http://eval-service/evaluate',
            headers={'X-Correlation-ID': training_id},
            json={'model_id': training_result.json()['id']}
        )

        # Step 4: Register model
        self.logger.info(f"[{training_id}] Model registration")
        registry_result = requests.post(
            'http://registry-service/register',
            headers={'X-Correlation-ID': training_id},
            json={'model_id': training_result.json()['id'],
                 'metrics': eval_result.json()}
        )

        self.logger.info(f"[{training_id}] Training completed",
                        extra={'correlation_id': training_id,
                              'model_id': registry_result.json()['id']})

        return training_id
```

**Correlated Log Output:**

```
[orchestrator]    [10:00:00] [train-abc123] Training started
[data-service]    [10:00:01] [train-abc123] Data preparation started
[data-service]    [10:00:45] [train-abc123] Data preparation completed
[training-service][10:00:46] [train-abc123] Model training started
[training-service][10:15:30] [train-abc123] Epoch 1/10 completed
[training-service][10:30:15] [train-abc123] Epoch 10/10 completed
[training-service][10:30:20] [train-abc123] Model training completed
[eval-service]    [10:30:21] [train-abc123] Evaluation started
[eval-service]    [10:32:45] [train-abc123] Evaluation completed
[registry-service][10:32:46] [train-abc123] Model registration started
[registry-service][10:32:50] [train-abc123] Model registered: model-xyz789
[orchestrator]    [10:32:51] [train-abc123] Training completed
```

**Searching:**

```bash
# Find all logs for this training job
grep "train-abc123" /var/log/ml/*/*.log

# Timeline view
grep "train-abc123" /var/log/ml/*/*.log | sort -k2
```

**Example 2: API Request Flow**

```python
# api_gateway.py
from flask import Flask, request
import uuid
import logging

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get or generate correlation ID
    correlation_id = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))

    logger.info(f"[{correlation_id}] Prediction request received",
               extra={'correlation_id': correlation_id,
                     'client_ip': request.remote_addr})

    # Authentication
    auth_result = authenticate(correlation_id, request.headers.get('Authorization'))

    # Load model
    model = load_model(correlation_id, request.json['model_id'])

    # Perform inference
    result = perform_inference(correlation_id, model, request.json['input'])

    # Log metrics
    log_metrics(correlation_id, result)

    logger.info(f"[{correlation_id}] Prediction completed",
               extra={'correlation_id': correlation_id,
                     'duration_ms': result['duration_ms']})

    return result

def authenticate(correlation_id, auth_token):
    logger.info(f"[{correlation_id}] Authentication check")
    # Auth logic...
    return True

def load_model(correlation_id, model_id):
    logger.info(f"[{correlation_id}] Loading model: {model_id}")
    # Model loading...
    return model

def perform_inference(correlation_id, model, input_data):
    logger.info(f"[{correlation_id}] Performing inference")
    # Inference logic...
    return result

def log_metrics(correlation_id, result):
    logger.info(f"[{correlation_id}] Metrics",
               extra={'correlation_id': correlation_id,
                     'confidence': result['confidence'],
                     'latency_ms': result['duration_ms']})
```

#### 7. Log Correlation Tools

**Grafana Loki Query:**

```logql
# Find all logs with correlation ID
{job="ml-app"} |= "abc123"

# Extract and visualize request flow
{job="ml-app"} |= "abc123"
| json
| line_format "{{.timestamp}} [{{.service}}] {{.message}}"
| sort_by timestamp
```

**Elasticsearch Query:**

```json
{
  "query": {
    "bool": {
      "must": [
        {"match": {"correlation_id": "abc123"}}
      ]
    }
  },
  "sort": [
    {"@timestamp": {"order": "asc"}}
  ],
  "_source": ["@timestamp", "service", "level", "message"]
}
```

**Jaeger Trace Query:**

```
# View complete trace
http://jaeger-ui:16686/trace/abc123def456

# See service dependencies
http://jaeger-ui:16686/dependencies
```

#### 8. Best Practices

**1. Consistent Correlation ID Format:**

```python
# Use UUIDs for uniqueness
correlation_id = str(uuid.uuid4())  # e.g., "550e8400-e29b-41d4-a716-446655440000"

# Or prefixed for clarity
correlation_id = f"req_{uuid.uuid4()}"  # e.g., "req_550e8400-..."
```

**2. Propagate Through All Service Calls:**

```python
def call_service(url, data, correlation_id):
    """Always include correlation ID in headers."""
    headers = {
        'X-Correlation-ID': correlation_id,
        'Content-Type': 'application/json'
    }
    return requests.post(url, json=data, headers=headers)
```

**3. Include in Error Messages:**

```python
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"[{correlation_id}] Operation failed: {str(e)}",
                extra={'correlation_id': correlation_id,
                      'error_type': type(e).__name__})
    raise
```

**4. Add to Response Headers:**

```python
@app.after_request
def add_correlation_id(response):
    correlation_id = g.get('correlation_id', 'unknown')
    response.headers['X-Correlation-ID'] = correlation_id
    return response
```

**5. Correlation in Async/Background Jobs:**

```python
from celery import Celery

app = Celery('tasks')

@app.task(bind=True)
def train_model_async(self, config, correlation_id):
    """Background training job with correlation."""
    logger.info(f"[{correlation_id}] Background training started",
               extra={'correlation_id': correlation_id,
                     'task_id': self.request.id})

    # Training logic...

    logger.info(f"[{correlation_id}] Background training completed",
               extra={'correlation_id': correlation_id})
```

### Summary

**Log Correlation Strategy:**

1. **Correlation IDs**: Unique identifiers propagated through all services
2. **Centralized Logging**: Aggregate logs in Elasticsearch, Loki, or similar
3. **Structured Logging**: Consistent JSON format with shared fields
4. **Timestamp Sync**: Use NTP to synchronize clocks
5. **Distributed Tracing**: Use OpenTelemetry/Jaeger for deep visibility

**Essential Fields in Every Log:**
- `correlation_id`: Request/transaction identifier
- `service`: Service name
- `timestamp`: UTC timestamp
- `level`: Log level
- `message`: Human-readable message

**Tools:**
- **Log Aggregation**: ELK Stack, Loki, Splunk
- **Distributed Tracing**: Jaeger, Zipkin, OpenTelemetry
- **Search**: Kibana, Grafana, Splunk UI

**Querying Correlated Logs:**

```bash
# Simple grep
grep "abc123" /var/log/ml/*.log | sort -k2

# Elasticsearch
curl "http://es:9200/logs/_search?q=correlation_id:abc123"

# Loki
logcli query '{job="ml-app"} |= "abc123"' --since=1h
```

---

## Question 5: What metrics can you extract from training logs?

### Short Answer

Extract these key metrics from ML training logs: (1) **performance metrics** (loss, accuracy, validation metrics), (2) **training progress** (epochs completed, time per epoch, ETA), (3) **resource utilization** (GPU memory, batch size, learning rate), (4) **model quality** (overfitting detection, convergence), and (5) **operational metrics** (checkpoint frequency, error rates, warnings).

### Detailed Answer

#### 1. Performance Metrics

**Primary Training Metrics:**

```python
# Training log example
2024-10-18 10:00:10 INFO Epoch 1/100 - loss: 2.3012 - accuracy: 0.1234 - val_loss: 2.1234 - val_accuracy: 0.1567
2024-10-18 10:01:30 INFO Epoch 2/100 - loss: 1.8765 - accuracy: 0.3456 - val_loss: 1.7890 - val_accuracy: 0.3789
```

**Extraction Script:**

```bash
#!/bin/bash
# extract_training_metrics.sh

LOG_FILE="$1"
OUTPUT_CSV="training_metrics.csv"

# Create CSV header
echo "epoch,loss,accuracy,val_loss,val_accuracy,time" > "$OUTPUT_CSV"

# Extract metrics
awk '/Epoch [0-9]+\/[0-9]+/ {
    match($0, /Epoch ([0-9]+)/, epoch);
    match($0, /loss: ([0-9.]+)/, loss);
    match($0, /accuracy: ([0-9.]+)/, acc);
    match($0, /val_loss: ([0-9.]+)/, val_loss);
    match($0, /val_accuracy: ([0-9.]+)/, val_acc);
    timestamp = $1 " " $2;

    printf "%d,%.4f,%.4f,%.4f,%.4f,%s\n",
        epoch[1], loss[1], acc[1], val_loss[1], val_acc[1], timestamp
}' "$LOG_FILE" >> "$OUTPUT_CSV"

echo "Metrics saved to: $OUTPUT_CSV"
```

**Metrics to Track:**

| Metric | Description | What It Tells You |
|--------|-------------|-------------------|
| Training Loss | Error on training data | Model learning progress |
| Training Accuracy | Correct predictions (train) | Model performance on seen data |
| Validation Loss | Error on validation data | Generalization ability |
| Validation Accuracy | Correct predictions (val) | Real-world performance estimate |
| Test Loss/Accuracy | Final evaluation metrics | Final model quality |
| Learning Rate | Current LR value | Optimization state |
| Batch Size | Current batch size | Memory/speed trade-off |
| Gradient Norm | Size of gradients | Training stability |

#### 2. Training Progress Metrics

**Time-Based Metrics:**

```bash
# Extract epoch timing
awk '
BEGIN {
    prev_time = 0;
}
/Epoch [0-9]+/ {
    match($0, /Epoch ([0-9]+)/, epoch);
    timestamp = $1 " " $2;

    # Convert timestamp to epoch seconds
    cmd = "date -d \"" timestamp "\" +%s";
    cmd | getline current_time;
    close(cmd);

    if (prev_time > 0) {
        duration = current_time - prev_time;
        printf "Epoch %d: %d seconds (%.2f min)\n",
            epoch[1], duration, duration/60;
    }

    prev_time = current_time;
}' training.log
```

**Progress Metrics:**

```python
#!/usr/bin/env python3
"""
Training Progress Analyzer
"""

import re
from datetime import datetime
import pandas as pd

def extract_progress_metrics(log_file):
    """Extract training progress metrics."""
    epochs = []
    times = []
    loss_values = []

    with open(log_file) as f:
        for line in f:
            if 'Epoch' in line and 'loss:' in line:
                # Extract epoch number
                epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    total_epochs = int(epoch_match.group(2))

                # Extract timestamp
                time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if time_match:
                    timestamp = datetime.strptime(time_match.group(1),
                                                 '%Y-%m-%d %H:%M:%S')

                # Extract loss
                loss_match = re.search(r'loss: ([\d.]+)', line)
                if loss_match:
                    loss = float(loss_match.group(1))

                epochs.append(current_epoch)
                times.append(timestamp)
                loss_values.append(loss)

    # Create DataFrame
    df = pd.DataFrame({
        'epoch': epochs,
        'timestamp': times,
        'loss': loss_values
    })

    # Calculate time per epoch
    df['time_per_epoch'] = df['timestamp'].diff().dt.total_seconds()

    # Calculate average time per epoch
    avg_time = df['time_per_epoch'].mean()

    # Estimate completion time
    epochs_remaining = total_epochs - current_epoch
    eta_seconds = epochs_remaining * avg_time
    eta = datetime.now() + pd.Timedelta(seconds=eta_seconds)

    # Calculate progress
    progress = current_epoch / total_epochs * 100

    return {
        'current_epoch': current_epoch,
        'total_epochs': total_epochs,
        'progress_percent': progress,
        'avg_time_per_epoch': avg_time,
        'eta': eta,
        'elapsed_time': (times[-1] - times[0]).total_seconds(),
        'metrics_df': df
    }

if __name__ == '__main__':
    import sys
    result = extract_progress_metrics(sys.argv[1])

    print(f"Training Progress:")
    print(f"  Epoch: {result['current_epoch']}/{result['total_epochs']}")
    print(f"  Progress: {result['progress_percent']:.1f}%")
    print(f"  Avg time/epoch: {result['avg_time_per_epoch']:.1f}s")
    print(f"  Elapsed: {result['elapsed_time']/3600:.2f} hours")
    print(f"  ETA: {result['eta']}")
```

**Extracted Progress Metrics:**

```
Training Progress:
  Epoch: 45/100
  Progress: 45.0%
  Avg time/epoch: 125.3s
  Elapsed: 1.57 hours
  ETA: 2024-10-18 12:45:30
  Estimated remaining: 1.92 hours
```

#### 3. Resource Utilization Metrics

**GPU Metrics Extraction:**

```bash
# Extract GPU memory usage from logs
grep "GPU memory" training.log | \
    awk '{
        match($0, /([0-9.]+)GB/, mem);
        print $1, $2, mem[1] "GB"
    }'

# Or from nvidia-smi logged to file
awk '{
    gpu_util = $1;
    mem_used = $2;
    temp = $3;
    print $0;

    # Track peaks
    if (mem_used > max_mem || max_mem == 0) max_mem = mem_used;
    if (gpu_util > max_util || max_util == 0) max_util = gpu_util;
}
END {
    printf "\nPeak GPU utilization: %d%%\n", max_util;
    printf "Peak memory usage: %dMB\n", max_mem;
}' gpu_log.txt
```

**Resource Metrics to Track:**

| Metric | Source | Importance |
|--------|--------|------------|
| GPU Utilization % | nvidia-smi | Efficiency indicator |
| GPU Memory Used/Total | nvidia-smi | Memory optimization |
| CPU Utilization % | top/htop | Bottleneck detection |
| RAM Usage | free -h | Memory requirements |
| Disk I/O | iostat | Data loading performance |
| Network I/O | iftop | Distributed training |
| Batch Size | Training logs | Throughput/memory trade-off |
| Data Loading Time | Training logs | Pipeline efficiency |

**Comprehensive Resource Monitoring:**

```python
#!/usr/bin/env python3
"""
resource_monitor.py - Monitor and log resource usage during training
"""

import time
import psutil
import logging
from datetime import datetime

try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_AVAILABLE = True
except:
    NVIDIA_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_gpu_metrics():
    """Get GPU metrics using NVML."""
    if not NVIDIA_AVAILABLE:
        return None

    gpu_metrics = []
    device_count = pynvml.nvmlDeviceGetCount()

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

        # Get utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        # Get memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # Get temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle,
                                               pynvml.NVML_TEMPERATURE_GPU)

        gpu_metrics.append({
            'gpu_id': i,
            'utilization': util.gpu,
            'memory_used_mb': mem_info.used // (1024**2),
            'memory_total_mb': mem_info.total // (1024**2),
            'memory_percent': (mem_info.used / mem_info.total) * 100,
            'temperature': temp
        })

    return gpu_metrics

def get_system_metrics():
    """Get system resource metrics."""
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        'cpu_percent': cpu_percent,
        'memory_used_gb': mem.used / (1024**3),
        'memory_total_gb': mem.total / (1024**3),
        'memory_percent': mem.percent,
        'disk_used_gb': disk.used / (1024**3),
        'disk_total_gb': disk.total / (1024**3),
        'disk_percent': disk.percent
    }

def log_resource_metrics():
    """Log all resource metrics."""
    system_metrics = get_system_metrics()

    logger.info(f"System Resources - "
               f"CPU: {system_metrics['cpu_percent']:.1f}% | "
               f"RAM: {system_metrics['memory_used_gb']:.1f}/"
               f"{system_metrics['memory_total_gb']:.1f}GB "
               f"({system_metrics['memory_percent']:.1f}%) | "
               f"Disk: {system_metrics['disk_percent']:.1f}%")

    gpu_metrics = get_gpu_metrics()
    if gpu_metrics:
        for gpu in gpu_metrics:
            logger.info(f"GPU {gpu['gpu_id']} - "
                       f"Utilization: {gpu['utilization']}% | "
                       f"Memory: {gpu['memory_used_mb']}/"
                       f"{gpu['memory_total_mb']}MB "
                       f"({gpu['memory_percent']:.1f}%) | "
                       f"Temp: {gpu['temperature']}°C")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s %(levelname)s %(message)s')

    while True:
        log_resource_metrics()
        time.sleep(60)  # Log every minute
```

#### 4. Model Quality Metrics

**Overfitting Detection:**

```bash
# Detect overfitting from validation loss
awk -F',' 'NR>1 {
    train_loss = $2;
    val_loss = $4;
    gap = val_loss - train_loss;

    if (gap > 0.15) {
        printf "Epoch %d: Possible overfitting - gap: %.4f\n", $1, gap
    }
}' training_metrics.csv
```

**Convergence Analysis:**

```python
#!/usr/bin/env python3
"""
Analyze training convergence
"""

import pandas as pd
import numpy as np

def analyze_convergence(metrics_csv):
    """Analyze if training has converged."""
    df = pd.read_csv(metrics_csv)

    # Calculate loss change rate
    df['loss_change'] = df['loss'].diff().abs()

    # Check last N epochs
    window = 5
    recent_changes = df['loss_change'].tail(window)

    avg_change = recent_changes.mean()
    std_change = recent_changes.std()

    # Convergence criteria
    convergence_threshold = 0.001
    stability_threshold = 0.0005

    is_converged = (avg_change < convergence_threshold and
                   std_change < stability_threshold)

    return {
        'converged': is_converged,
        'avg_change': avg_change,
        'std_change': std_change,
        'recommendation': (
            "Training has converged" if is_converged
            else "Continue training or adjust learning rate"
        )
    }

if __name__ == '__main__':
    import sys
    result = analyze_convergence(sys.argv[1])
    print(f"Convergence Analysis:")
    print(f"  Converged: {result['converged']}")
    print(f"  Avg loss change: {result['avg_change']:.6f}")
    print(f"  Std loss change: {result['std_change']:.6f}")
    print(f"  Recommendation: {result['recommendation']}")
```

**Learning Rate Schedule Tracking:**

```bash
# Extract learning rate changes
grep "learning_rate\|Learning rate" training.log | \
    awk '{
        match($0, /([0-9.e-]+)/, lr);
        print $1, $2, "LR:", lr[1]
    }'
```

#### 5. Operational Metrics

**Checkpoint Metrics:**

```bash
# Track checkpoint frequency and size
grep "Checkpoint saved" training.log | \
    awk '{
        print $1, $2, $0;
        count++;
    }
    END {
        print "\nTotal checkpoints:", count
    }'

# Checkpoint sizes
ls -lh /models/checkpoints/ | \
    awk '{
        if (NR > 1) {
            sum += $5;
            print $9, $5;
        }
    }
    END {
        print "\nTotal size:", sum/1024/1024, "MB"
    }'
```

**Error Rate Tracking:**

```bash
# Calculate error rate per epoch
for i in {1..100}; do
    echo -n "Epoch $i: "
    grep "Epoch $i" training.log | grep -c "ERROR" || echo 0
done
```

**Warning Analysis:**

```bash
# Categorize warnings
grep "WARNING" training.log | \
    sed 's/.*WARNING //' | \
    cut -d':' -f1 | \
    sort | uniq -c | sort -rn
```

#### 6. Complete Metrics Dashboard

**Python Script for Comprehensive Analysis:**

```python
#!/usr/bin/env python3
"""
complete_metrics_dashboard.py - Complete training metrics analysis
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class TrainingMetricsAnalyzer:
    def __init__(self, log_file):
        self.log_file = log_file
        self.metrics = self.extract_all_metrics()

    def extract_all_metrics(self):
        """Extract all available metrics from log file."""
        epochs = []
        timestamps = []
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        learning_rates = []
        batch_sizes = []
        gpu_memory = []

        with open(self.log_file) as f:
            for line in f:
                # Extract epoch metrics
                if 'Epoch' in line and 'loss:' in line:
                    epoch = re.search(r'Epoch (\d+)', line)
                    time = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    loss = re.search(r'loss: ([\d.]+)', line)
                    acc = re.search(r'accuracy: ([\d.]+)', line)
                    v_loss = re.search(r'val_loss: ([\d.]+)', line)
                    v_acc = re.search(r'val_accuracy: ([\d.]+)', line)

                    if all([epoch, time, loss, acc, v_loss, v_acc]):
                        epochs.append(int(epoch.group(1)))
                        timestamps.append(datetime.strptime(time.group(1),
                                                           '%Y-%m-%d %H:%M:%S'))
                        train_loss.append(float(loss.group(1)))
                        train_acc.append(float(acc.group(1)))
                        val_loss.append(float(v_loss.group(1)))
                        val_acc.append(float(v_acc.group(1)))

                # Extract learning rate
                if 'learning_rate' in line or 'Learning rate' in line:
                    lr = re.search(r'([\d.e-]+)', line.split(':')[-1])
                    if lr:
                        learning_rates.append(float(lr.group(1)))

                # Extract batch size changes
                if 'batch_size' in line or 'batch size' in line:
                    bs = re.search(r'(\d+)', line.split('batch')[-1])
                    if bs:
                        batch_sizes.append(int(bs.group(1)))

                # Extract GPU memory
                if 'GPU memory' in line or 'gpu_memory' in line:
                    mem = re.search(r'([\d.]+)(?:GB|MB)', line)
                    if mem:
                        gpu_memory.append(float(mem.group(1)))

        df = pd.DataFrame({
            'epoch': epochs,
            'timestamp': timestamps,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })

        return {
            'dataframe': df,
            'learning_rates': learning_rates,
            'batch_sizes': batch_sizes,
            'gpu_memory': gpu_memory
        }

    def calculate_performance_metrics(self):
        """Calculate performance metrics."""
        df = self.metrics['dataframe']

        return {
            'best_train_loss': df['train_loss'].min(),
            'best_train_acc': df['train_accuracy'].max(),
            'best_val_loss': df['val_loss'].min(),
            'best_val_acc': df['val_accuracy'].max(),
            'final_train_loss': df['train_loss'].iloc[-1],
            'final_val_loss': df['val_loss'].iloc[-1],
            'loss_improvement': df['train_loss'].iloc[0] - df['train_loss'].iloc[-1],
            'acc_improvement': df['train_accuracy'].iloc[-1] - df['train_accuracy'].iloc[0]
        }

    def calculate_progress_metrics(self):
        """Calculate training progress metrics."""
        df = self.metrics['dataframe']

        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        avg_time_per_epoch = df['time_diff'].mean()

        total_time = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()

        return {
            'total_epochs': len(df),
            'avg_time_per_epoch': avg_time_per_epoch,
            'total_training_time': total_time,
            'total_training_time_hours': total_time / 3600
        }

    def detect_overfitting(self):
        """Detect overfitting."""
        df = self.metrics['dataframe']

        df['loss_gap'] = df['val_loss'] - df['train_loss']
        df['acc_gap'] = df['train_accuracy'] - df['val_accuracy']

        overfitting_epochs = df[df['loss_gap'] > 0.1]

        return {
            'overfitting_detected': len(overfitting_epochs) > 0,
            'overfitting_epochs': overfitting_epochs['epoch'].tolist(),
            'max_loss_gap': df['loss_gap'].max(),
            'avg_loss_gap': df['loss_gap'].mean()
        }

    def analyze_convergence(self):
        """Analyze training convergence."""
        df = self.metrics['dataframe']

        df['loss_change'] = df['train_loss'].diff().abs()

        window = 5
        recent_changes = df['loss_change'].tail(window)

        converged = (recent_changes.mean() < 0.001 and
                    recent_changes.std() < 0.0005)

        return {
            'converged': converged,
            'avg_recent_change': recent_changes.mean(),
            'std_recent_change': recent_changes.std()
        }

    def generate_report(self):
        """Generate complete metrics report."""
        performance = self.calculate_performance_metrics()
        progress = self.calculate_progress_metrics()
        overfitting = self.detect_overfitting()
        convergence = self.analyze_convergence()

        report = {
            'performance_metrics': performance,
            'progress_metrics': progress,
            'overfitting_analysis': overfitting,
            'convergence_analysis': convergence,
            'resource_metrics': {
                'avg_gpu_memory': (np.mean(self.metrics['gpu_memory'])
                                  if self.metrics['gpu_memory'] else None),
                'batch_sizes': self.metrics['batch_sizes'],
                'learning_rates': self.metrics['learning_rates']
            }
        }

        return report

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 complete_metrics_dashboard.py <log_file>")
        sys.exit(1)

    analyzer = TrainingMetricsAnalyzer(sys.argv[1])
    report = analyzer.generate_report()

    print("=" * 60)
    print(" Training Metrics Dashboard")
    print("=" * 60)

    print("\nPerformance Metrics:")
    for key, value in report['performance_metrics'].items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\nProgress Metrics:")
    for key, value in report['progress_metrics'].items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\nOverfitting Analysis:")
    for key, value in report['overfitting_analysis'].items():
        print(f"  {key}: {value}")

    print("\nConvergence Analysis:")
    for key, value in report['convergence_analysis'].items():
        print(f"  {key}: {value}")

    # Save report to JSON
    with open('training_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\nReport saved to: training_report.json")
```

#### 7. Visualization and Dashboards

**Create Real-Time Dashboard:**

```python
#!/usr/bin/env python3
"""
real_time_dashboard.py - Real-time training metrics dashboard
"""

import time
import pandas as pd
from flask import Flask, render_template, jsonify

app = Flask(__name__)

def get_latest_metrics():
    """Get latest metrics from log file."""
    # Read and parse latest log entries
    # Return metrics as JSON
    pass

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/metrics')
def api_metrics():
    metrics = get_latest_metrics()
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Summary

**Extractable Metrics Categories:**

1. **Performance**: Loss, accuracy, validation metrics
2. **Progress**: Epochs, time per epoch, ETA
3. **Resources**: GPU/CPU usage, memory, I/O
4. **Quality**: Overfitting, convergence, stability
5. **Operational**: Errors, warnings, checkpoints

**Key Extraction Methods:**
- grep/awk for structured text logs
- Python/pandas for complex analysis
- Real-time monitoring for live training
- Visualization for trend analysis

**Metrics to Track:**
- **Must Track**: Train/val loss, train/val accuracy
- **Should Track**: Time per epoch, GPU memory, learning rate
- **Nice to Have**: Gradient norms, batch statistics, convergence

**Tools:**
- Bash scripts for quick extraction
- Python for comprehensive analysis
- TensorBoard for real-time visualization
- Grafana for production monitoring

---

**End of Reflection Question Answers**
