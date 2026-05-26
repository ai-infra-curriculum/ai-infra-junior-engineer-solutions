# Exercise 06: Linux Log Management — Solution

## What the exercise asked for

Read, analyze, rotate, and ship logs for production ML
workloads on Linux: systemd journal, application logs,
syslog, log rotation, basic forwarding.

## Where logs live

| Source | Location | Tool |
|---|---|---|
| Boot / kernel | `/var/log/syslog` (Debian), `/var/log/messages` (RHEL) | `tail`, `less`, `journalctl` |
| systemd services | `journalctl -u <service>` | `journalctl` |
| Authentication | `/var/log/auth.log` | `grep` patterns |
| Application | `/var/log/<app>/` or container stdout | `docker logs`, `kubectl logs` |
| Audit | `/var/log/audit/audit.log` (auditd) | `aureport`, `ausearch` |

## journalctl: the modern Linux logging interface

```bash
# Last 100 lines of a service
journalctl -u myapp -n 100

# Follow live
journalctl -u myapp -f

# Time-bounded
journalctl --since "1 hour ago"
journalctl --since "2026-05-26 14:00" --until "2026-05-26 15:00"

# By priority (emergency=0 to debug=7)
journalctl -u myapp -p err

# JSON output (for piping to jq)
journalctl -u myapp -o json | jq '.MESSAGE'

# Disk usage
journalctl --disk-usage
sudo journalctl --vacuum-time=7d
```

## Reading container logs

```bash
# Docker
docker logs <container>
docker logs -f --tail 100 <container>

# Kubernetes
kubectl logs <pod>
kubectl logs -f <pod> -c <container>
kubectl logs --previous <pod>   # if it crashed and restarted
kubectl logs -l app=ml-serving --tail=50  # by label
```

## Log rotation

For files in `/var/log/` written by long-lived processes, use
`logrotate`:

```text
# /etc/logrotate.d/myapp
/var/log/myapp/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 640 myapp myapp
    sharedscripts
    postrotate
        systemctl reload myapp >/dev/null 2>&1 || true
    endscript
}
```

Without rotation: disks fill up at 3 AM and the on-call wakes up.

For container workloads, the runtime handles rotation (Docker:
`max-size` and `max-file` driver opts; Kubernetes: container
runtime config).

## Log shipping to a central store

### journald → ELK / Loki / Splunk

```bash
# Use fluent-bit, vector, or rsyslog as the shipper
# Example: vector config that reads journal and ships to Loki
sources:
  journald:
    type: journald

sinks:
  loki:
    type: loki
    endpoint: http://loki:3100
    inputs: [journald]
```

For ML platforms: structured JSON logs from applications,
shipped to Loki or Elasticsearch, queried in Grafana.

## ML-specific logging patterns

```python
import logging
import json

# Structured JSON logging for production ML
class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            # ML-specific structured fields:
            "model_version": getattr(record, "model_version", None),
            "tenant_id": getattr(record, "tenant_id", None),
            "request_id": getattr(record, "request_id", None),
        })
```

Structured logs are queryable; unstructured logs are not.

## Common mistakes

- Logging at `INFO` level in production for high-traffic
  endpoints → disk fills, monitoring tool chokes.
- Logging sensitive data (PII, API keys, model inputs that
  contain PII).
- Not rotating; disks fill up.
- Catching exceptions silently without logging.
- Logging without structured fields, so you can't query later.

## Cross-references

- Exercise prompt:
  `ai-infra-junior-engineer-learning/lessons/mod-002-linux-essentials/exercises/exercise-06-logs.md`
- mod-009-monitoring-basics covers observability at the
  application level.
- The Engineer-track has production-grade logging patterns in
  `engineer-solutions/mod-108`.
