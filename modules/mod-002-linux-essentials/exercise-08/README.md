# Exercise 08: System Automation and Maintenance

Complete ML infrastructure automation suite with backup, monitoring, cleanup, and health check capabilities.

## Overview

This exercise demonstrates production-ready system automation for ML infrastructure, including:

- **Automated Model Backups**: Timestamped, verified backups with optional S3 sync
- **GPU Health Monitoring**: Real-time monitoring with multi-channel alerts
- **Log Rotation**: Automated log management for all ML services
- **Artifact Cleanup**: Intelligent cleanup of old experiments and checkpoints
- **Health Checks**: Comprehensive system health validation
- **Task Orchestration**: Master script coordinating all maintenance tasks

## Directory Structure

```
exercise-08/
├── backups/
│   └── backup_models.sh           # Model backup with versioning
├── monitoring/
│   └── monitor_gpus.sh            # GPU health monitoring
├── cleanup/
│   └── cleanup_ml_artifacts.sh    # Artifact cleanup
├── health-check/
│   └── health_check.sh            # System health validation
├── config/
│   ├── crontab.example            # Cron scheduling examples
│   ├── systemd-timers.md          # Systemd timer configurations
│   └── logrotate-ml.conf          # Log rotation configuration
├── docs/
│   └── ANSWERS.md                 # Reflection questions
├── run_all_maintenance.sh         # Master orchestration script
└── README.md                      # This file
```

## Quick Start

### 1. Setup Environment Variables

```bash
# Model paths
export MODEL_DIR=/opt/ml/models
export BACKUP_DIR=/backup/models

# Thresholds
export TEMP_THRESHOLD=80
export MEMORY_THRESHOLD=90

# Retention policies
export RETENTION_DAYS=30
export EXPERIMENTS_RETENTION=30
export CHECKPOINTS_RETENTION=7
```

### 2. Run Individual Scripts

```bash
# Model backup
./backups/backup_models.sh

# GPU monitoring
./monitoring/monitor_gpus.sh

# Cleanup artifacts
./cleanup/cleanup_ml_artifacts.sh

# Health check
./health-check/health_check.sh

# Full maintenance suite
./run_all_maintenance.sh
```

### 3. Schedule Automation

**Option A: Using Cron**
```bash
# Edit crontab
crontab -e

# Add these lines:
0 2 * * * /home/mluser/exercise-08/backups/backup_models.sh
*/5 * * * * /home/mluser/exercise-08/monitoring/monitor_gpus.sh
0 3 * * 0 /home/mluser/exercise-08/cleanup/cleanup_ml_artifacts.sh
```

**Option B: Using Systemd Timers** (recommended)
```bash
# Copy service and timer files
sudo cp config/systemd/*.service /etc/systemd/system/
sudo cp config/systemd/*.timer /etc/systemd/system/

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable --now ml-backup.timer
sudo systemctl enable --now gpu-monitor.timer
```

See `config/systemd-timers.md` for complete systemd setup.

## Scripts Documentation

### 1. Model Backup Script

**Location**: `backups/backup_models.sh`

**Purpose**: Create timestamped, verified backups of ML models with optional cloud sync.

**Features**:
- Tar/gzip compression (configurable level)
- SHA256 checksum verification
- Automatic retention policy
- Optional S3 upload
- Dry-run mode for testing
- Backup integrity verification

**Usage**:
```bash
# Basic backup
./backups/backup_models.sh

# Custom configuration
MODEL_DIR=/opt/ml/models \
BACKUP_DIR=/backup/models \
RETENTION_DAYS=30 \
./backups/backup_models.sh

# Dry run
./backups/backup_models.sh --dry-run

# S3 sync
S3_BUCKET=my-ml-backups \
S3_STORAGE_CLASS=GLACIER \
./backups/backup_models.sh --s3-sync

# Verify existing backup
./backups/backup_models.sh --verify /backup/models/models_backup_20250131_020000.tar.gz
```

**Configuration Options**:
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `/opt/ml/models` | Source directory |
| `BACKUP_DIR` | `/backup/models` | Backup destination |
| `RETENTION_DAYS` | `30` | Days to keep backups |
| `COMPRESSION_LEVEL` | `6` | gzip compression (1-9) |
| `S3_BUCKET` | _(empty)_ | S3 bucket for upload |
| `S3_STORAGE_CLASS` | `STANDARD` | S3 storage class |
| `VERIFY_BACKUP` | `true` | Verify after backup |

**Exit Codes**:
- `0`: Success
- `1`: Backup failed
- `2`: Verification failed

**Logs**: `/var/log/ml-backup.log`

### 2. GPU Monitoring Script

**Location**: `monitoring/monitor_gpus.sh`

**Purpose**: Monitor GPU health and send alerts when thresholds are exceeded.

**Features**:
- nvidia-smi integration
- Configurable thresholds (temp, memory, power)
- Multiple alert channels (email, webhook, syslog)
- Alert cooldown to prevent spam
- Metrics logging to CSV
- GPU error detection

**Usage**:
```bash
# Basic monitoring
./monitoring/monitor_gpus.sh

# Custom thresholds
TEMP_THRESHOLD=75 \
MEMORY_THRESHOLD=85 \
./monitoring/monitor_gpus.sh

# Email alerts
ALERT_EMAIL=ml-ops@company.com \
./monitoring/monitor_gpus.sh

# Webhook alerts
ALERT_WEBHOOK=https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
./monitoring/monitor_gpus.sh

# Verbose output
./monitoring/monitor_gpus.sh --verbose
```

**Configuration Options**:
| Variable | Default | Description |
|----------|---------|-------------|
| `TEMP_THRESHOLD` | `80` | Temperature warning (°C) |
| `MEMORY_THRESHOLD` | `90` | Memory usage warning (%) |
| `POWER_THRESHOLD` | `95` | Power usage warning (%) |
| `ALERT_EMAIL` | _(empty)_ | Email for alerts |
| `ALERT_WEBHOOK` | _(empty)_ | Webhook URL |
| `ALERT_COOLDOWN` | `3600` | Seconds between alerts |
| `METRICS_FILE` | `/var/log/gpu-metrics.csv` | Metrics log |

**Alert Channels**:
1. **Email**: Requires `mail` command configured
2. **Webhook**: HTTP POST with JSON payload
3. **Syslog**: Logged to system journal

**Metrics CSV Format**:
```csv
timestamp,gpu_id,name,temp,util_gpu,util_mem,mem_used,mem_total,power_draw,power_limit
2025-01-31 14:30:00,0,Tesla V100,72,85,90,15000,16384,250,300
```

**Exit Codes**:
- `0`: All GPUs healthy
- `1`: Warning threshold exceeded
- `2`: Critical threshold exceeded

**Logs**: `/var/log/gpu-monitor.log`, `/var/log/gpu-metrics.csv`

### 3. Cleanup Script

**Location**: `cleanup/cleanup_ml_artifacts.sh`

**Purpose**: Clean up old ML experiments, checkpoints, and logs to free disk space.

**Features**:
- Multiple cleanup targets
- Configurable retention per target
- Size tracking and reporting
- Aggressive mode for deep cleanup
- Docker, pip, and apt cache cleaning
- Dry-run mode

**Usage**:
```bash
# Basic cleanup
./cleanup/cleanup_ml_artifacts.sh

# Custom retention
EXPERIMENTS_RETENTION=30 \
CHECKPOINTS_RETENTION=7 \
LOGS_RETENTION=14 \
./cleanup/cleanup_ml_artifacts.sh

# Aggressive cleanup (includes Docker, pip, apt)
./cleanup/cleanup_ml_artifacts.sh --aggressive

# Dry run
./cleanup/cleanup_ml_artifacts.sh --dry-run

# Specific targets only
./cleanup/cleanup_ml_artifacts.sh --experiments-only
./cleanup/cleanup_ml_artifacts.sh --checkpoints-only
```

**Configuration Options**:
| Variable | Default | Description |
|----------|---------|-------------|
| `EXPERIMENTS_RETENTION` | `30` | Days to keep experiments |
| `CHECKPOINTS_RETENTION` | `7` | Days to keep checkpoints |
| `LOGS_RETENTION` | `14` | Days to keep logs |
| `AGGRESSIVE` | `false` | Enable aggressive cleanup |

**Cleanup Targets**:
1. **Experiments**: `/opt/ml/experiments/*` (30 days default)
2. **Checkpoints**: `/opt/ml/checkpoints/*` (7 days default)
3. **Logs**: `/var/log/ml-*` (14 days default)
4. **Temp Files**: `/tmp/ml-*` (always)
5. **Aggressive Mode**:
   - Docker images (7 days old)
   - Pip cache
   - Apt cache
   - /tmp files (3 days old)

**Exit Codes**:
- `0`: Success
- `1`: Cleanup failed

**Logs**: `/var/log/ml-cleanup.log`

### 4. Health Check Script

**Location**: `health-check/health_check.sh`

**Purpose**: Comprehensive system health validation for ML infrastructure.

**Features**:
- Disk space and inode checks
- Memory usage (RAM and swap)
- CPU load average
- Service status validation
- GPU health checks
- Network connectivity tests
- Recent error analysis
- OOM killer detection
- JSON output for monitoring
- Health score calculation

**Usage**:
```bash
# Basic health check
./health-check/health_check.sh

# Custom thresholds
./health-check/health_check.sh --disk-warn 70 --disk-crit 85 --mem-warn 90

# Check specific services
./health-check/health_check.sh --services "docker nginx postgresql"

# JSON output (for monitoring tools)
./health-check/health_check.sh --json

# Verbose output
./health-check/health_check.sh --verbose

# Skip network checks
./health-check/health_check.sh --no-network
```

**Configuration Options**:
| Variable | Default | Description |
|----------|---------|-------------|
| `DISK_WARN_THRESHOLD` | `80` | Disk warning (%) |
| `DISK_CRIT_THRESHOLD` | `90` | Disk critical (%) |
| `MEM_WARN_THRESHOLD` | `85` | Memory warning (%) |
| `GPU_TEMP_WARN` | `75` | GPU temp warning (°C) |
| `GPU_TEMP_CRIT` | `85` | GPU temp critical (°C) |
| `SERVICES_TO_CHECK` | `docker` | Services to validate |

**Health Checks**:
1. **Disk Space**: Usage and inode counts
2. **Memory**: RAM and swap usage
3. **CPU**: Load average vs CPU count
4. **Services**: systemd service status
5. **GPU**: Temperature, memory, errors
6. **Network**: Connectivity and DNS
7. **System**: Recent errors, OOM events
8. **Uptime**: System uptime and reboot status

**Exit Codes**:
- `0`: All checks passed
- `1`: Warnings detected
- `2`: Critical issues detected

**Logs**: `/var/log/ml-health-check.log`

**JSON Output Example**:
```json
{
  "timestamp": "2025-01-31 14:30:00",
  "hostname": "ml-node-01",
  "checks": [
    {"check": "Disk usage", "status": "pass", "message": "Disk usage: 65%"},
    {"check": "Memory usage", "status": "pass", "message": "Memory usage: 72%"}
  ],
  "summary": {
    "total": 15,
    "passed": 13,
    "warnings": 2,
    "failed": 0,
    "health_score": 86.67
  }
}
```

### 5. Master Orchestration Script

**Location**: `run_all_maintenance.sh`

**Purpose**: Coordinate all maintenance tasks in the correct order.

**Features**:
- Task dependency management
- Individual task enable/disable
- Continue-on-error mode
- Task tracking and summary
- Duration calculation
- Comprehensive logging

**Usage**:
```bash
# Run all tasks
./run_all_maintenance.sh

# Skip specific tasks
./run_all_maintenance.sh --skip-backup
./run_all_maintenance.sh --skip-gpu --skip-cleanup

# Continue even if a task fails
./run_all_maintenance.sh --continue-on-error

# Custom log file
./run_all_maintenance.sh --log-file /var/log/custom-maintenance.log
```

**Task Execution Order**:
1. **Health Check** → Baseline system state
2. **GPU Monitoring** → Check GPU health
3. **Backups** → Backup before cleanup
4. **Cleanup** → Clean after backups are safe

**Configuration Options**:
| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_HEALTH_CHECK` | `true` | Run health check |
| `RUN_GPU_MONITOR` | `true` | Run GPU monitoring |
| `RUN_BACKUP` | `true` | Run backups |
| `RUN_CLEANUP` | `true` | Run cleanup |
| `CONTINUE_ON_ERROR` | `false` | Continue if task fails |
| `LOG_FILE` | `/var/log/ml-maintenance.log` | Log file path |

**Exit Codes**:
- `0`: All tasks succeeded
- `1`: Some tasks failed

**Logs**: `/var/log/ml-maintenance.log`

## Scheduling

### Cron vs Systemd Timers

| Feature | Cron | Systemd Timers |
|---------|------|----------------|
| **Scheduling** | Simple syntax | Calendar expressions |
| **Logging** | Mail/files | journald integration |
| **Dependencies** | None | Full systemd support |
| **Resource limits** | No | Yes (cgroups) |
| **Missed runs** | Skip | Persistent option |
| **Per-service logs** | No | Yes |
| **Randomization** | No | RandomizedDelaySec |

**Recommendation**: Use **systemd timers** for production. Better logging, resource control, and integration with systemd.

### Cron Setup

**1. Copy example configuration**:
```bash
cp config/crontab.example /tmp/ml-crontab
```

**2. Review and customize**:
```bash
# Edit variables at top
MAILTO=your-email@company.com
MODEL_DIR=/your/model/path
```

**3. Install**:
```bash
crontab /tmp/ml-crontab
```

**4. Verify**:
```bash
crontab -l
```

**5. Monitor execution**:
```bash
# View cron logs
grep CRON /var/log/syslog

# Check script logs
tail -f /var/log/ml-backup-cron.log
tail -f /var/log/gpu-monitor-cron.log
```

See `config/crontab.example` for complete configuration.

### Systemd Timer Setup

**1. Create service files** (see `config/systemd-timers.md` for examples):
```bash
sudo nano /etc/systemd/system/ml-backup.service
sudo nano /etc/systemd/system/ml-backup.timer
```

**2. Reload systemd**:
```bash
sudo systemctl daemon-reload
```

**3. Enable and start timers**:
```bash
sudo systemctl enable ml-backup.timer
sudo systemctl enable gpu-monitor.timer
sudo systemctl enable ml-cleanup.timer

sudo systemctl start ml-backup.timer
sudo systemctl start gpu-monitor.timer
sudo systemctl start ml-cleanup.timer
```

**4. Verify timers**:
```bash
# List all timers
systemctl list-timers

# Check specific timer
systemctl status ml-backup.timer
```

**5. Monitor execution**:
```bash
# View service logs
journalctl -u ml-backup.service -f

# View recent runs
journalctl -u ml-backup.service --since "1 day ago"
```

See `config/systemd-timers.md` for complete examples and troubleshooting.

## Log Rotation

### Setup

**1. Copy configuration**:
```bash
sudo cp config/logrotate-ml.conf /etc/logrotate.d/ml-infrastructure
```

**2. Test configuration**:
```bash
# Dry run
sudo logrotate -d /etc/logrotate.d/ml-infrastructure

# Force rotation (testing)
sudo logrotate -f /etc/logrotate.d/ml-infrastructure
```

**3. Verify**:
```bash
# Check status
cat /var/lib/logrotate/status | grep ml

# Monitor rotation
grep logrotate /var/log/syslog
```

### Configuration Overview

The logrotate configuration manages these logs:

| Log Type | Rotation | Retention | Compression |
|----------|----------|-----------|-------------|
| ML API logs | Daily | 14 days | Yes (delayed) |
| Training logs | Weekly | 8 weeks | Yes |
| Backup logs | Daily | 30 days | Yes |
| GPU monitor | Daily | 30 days | Yes |
| GPU metrics CSV | Weekly | 12 weeks | No |
| Cleanup logs | Weekly | 8 weeks | Yes |
| Health check | Daily | 14 days | Yes |
| Maintenance | Monthly | 12 months | Yes |

See `config/logrotate-ml.conf` for complete configuration.

## Testing

### Unit Testing

Test individual scripts before scheduling:

```bash
# 1. Backup script
MODEL_DIR=/tmp/test-models \
BACKUP_DIR=/tmp/test-backups \
./backups/backup_models.sh --dry-run

# 2. GPU monitoring (if GPUs available)
TEMP_THRESHOLD=50 \
./monitoring/monitor_gpus.sh --verbose

# 3. Cleanup
EXPERIMENTS_DIR=/tmp/test-experiments \
./cleanup/cleanup_ml_artifacts.sh --dry-run

# 4. Health check
./health-check/health_check.sh --verbose

# 5. Full suite
./run_all_maintenance.sh --skip-gpu --continue-on-error
```

### Integration Testing

Test scheduled execution:

```bash
# Create test cron job (runs every minute)
* * * * * /path/to/health_check.sh >> /tmp/test-health.log 2>&1

# Wait 2 minutes, then check
cat /tmp/test-health.log

# Remove test job
crontab -e  # Delete test line
```

### Monitoring Integration

Test with monitoring systems:

```bash
# JSON output for Prometheus/Datadog
./health-check/health_check.sh --json | jq

# Check exit codes
./health-check/health_check.sh
echo "Exit code: $?"

# Alert webhook test
ALERT_WEBHOOK=https://hooks.slack.com/test \
TEMP_THRESHOLD=0 \
./monitoring/monitor_gpus.sh
```

## Troubleshooting

### Common Issues

#### 1. Script Permission Denied

**Problem**: `bash: ./script.sh: Permission denied`

**Solution**:
```bash
chmod +x backups/*.sh
chmod +x monitoring/*.sh
chmod +x cleanup/*.sh
chmod +x health-check/*.sh
chmod +x *.sh
```

#### 2. Cron Job Not Running

**Problem**: Cron job doesn't execute

**Diagnosis**:
```bash
# Check cron service
systemctl status cron

# Check crontab
crontab -l

# Monitor cron execution
grep CRON /var/log/syslog

# Check script logs
tail -f /var/log/ml-backup-cron.log
```

**Solutions**:
- Verify absolute paths in crontab
- Check environment variables
- Ensure scripts are executable
- Verify user has permissions

#### 3. Systemd Timer Not Firing

**Problem**: Timer enabled but service doesn't run

**Diagnosis**:
```bash
# Check timer status
systemctl status ml-backup.timer

# List all timers
systemctl list-timers --all

# Check service
systemctl status ml-backup.service

# View logs
journalctl -u ml-backup.timer
journalctl -u ml-backup.service
```

**Solutions**:
```bash
# Reload systemd
sudo systemctl daemon-reload

# Restart timer
sudo systemctl restart ml-backup.timer

# Check time
timedatectl status

# Verify calendar expression
systemd-analyze calendar "daily"
```

#### 4. GPU Monitoring Fails

**Problem**: `nvidia-smi: command not found`

**Solutions**:
- Verify NVIDIA drivers: `nvidia-smi`
- Skip GPU checks: `CHECK_GPU=no ./health_check.sh`
- Install nvidia-utils: `sudo apt install nvidia-utils-*`

#### 5. Email Alerts Not Sent

**Problem**: Alerts configured but not received

**Diagnosis**:
```bash
# Test mail command
echo "Test" | mail -s "Test Subject" your@email.com

# Check mail logs
tail -f /var/log/mail.log
```

**Solutions**:
- Install mail: `sudo apt install mailutils`
- Configure SMTP in `/etc/postfix/main.cf`
- Use webhook alerts instead

#### 6. Disk Space Issues

**Problem**: Backups filling disk

**Solutions**:
```bash
# Check disk usage
df -h /backup

# Reduce retention
RETENTION_DAYS=7 ./backups/backup_models.sh

# Enable S3 sync
S3_BUCKET=my-backups ./backups/backup_models.sh --s3-sync

# Run cleanup
./cleanup/cleanup_ml_artifacts.sh --aggressive
```

#### 7. Logrotate Not Working

**Problem**: Logs growing, not rotating

**Diagnosis**:
```bash
# Test logrotate
sudo logrotate -d /etc/logrotate.d/ml-infrastructure

# Check status
cat /var/lib/logrotate/status | grep ml

# Force rotation
sudo logrotate -f /etc/logrotate.d/ml-infrastructure
```

**Solutions**:
- Fix syntax errors in config
- Ensure log files exist
- Check permissions: `ls -la /var/log/ml-*.log`
- Verify logrotate cron job: `/etc/cron.daily/logrotate`

### Debug Mode

Enable verbose output for debugging:

```bash
# Bash debug mode
bash -x ./backups/backup_models.sh

# Script verbose mode
./monitoring/monitor_gpus.sh --verbose
./health-check/health_check.sh --verbose

# Dry run (no changes)
./backups/backup_models.sh --dry-run
./cleanup/cleanup_ml_artifacts.sh --dry-run
```

### Logging

Check logs for errors:

```bash
# Script logs
tail -f /var/log/ml-backup.log
tail -f /var/log/gpu-monitor.log
tail -f /var/log/ml-cleanup.log
tail -f /var/log/ml-health-check.log

# Systemd logs
journalctl -u ml-backup.service -f
journalctl -u gpu-monitor.service --since "1 hour ago"

# Cron logs
grep CRON /var/log/syslog | tail -20

# System logs
dmesg -T | tail -50
```

## Best Practices

### 1. Backup Strategy

Follow the **3-2-1 backup rule**:
- **3** copies of data (original + 2 backups)
- **2** different storage types (local + cloud)
- **1** off-site copy (S3, GCS, Azure Blob)

```bash
# Configure S3 backup
export S3_BUCKET=ml-models-backup
export S3_STORAGE_CLASS=GLACIER  # Cost-effective for archives
export RETENTION_DAYS=30

./backups/backup_models.sh --s3-sync
```

### 2. Monitoring Alerts

Configure appropriate alert thresholds:

```bash
# Conservative (fewer false positives)
export TEMP_THRESHOLD=85
export MEMORY_THRESHOLD=95

# Aggressive (early warnings)
export TEMP_THRESHOLD=75
export MEMORY_THRESHOLD=85
```

Set alert cooldown to prevent spam:
```bash
export ALERT_COOLDOWN=3600  # 1 hour between alerts
```

### 3. Resource Limits

Use systemd resource controls:

```ini
[Service]
# Limit memory
MemoryLimit=2G

# Limit CPU
CPUQuota=50%

# Limit I/O
IOWeight=100
```

### 4. Testing in Production

Always test automation before deploying:

```bash
# 1. Dry run first
./backups/backup_models.sh --dry-run

# 2. Test with short cron schedule
# Run every 5 minutes for testing
*/5 * * * * /path/to/script.sh

# 3. Monitor for 24 hours
tail -f /var/log/ml-*.log

# 4. Adjust to production schedule
0 2 * * * /path/to/script.sh
```

### 5. Security

- Run scripts as dedicated user (not root)
- Use absolute paths
- Validate inputs
- Restrict file permissions:
  ```bash
  chmod 750 backups/*.sh  # rwxr-x---
  chmod 640 config/*      # rw-r-----
  ```

### 6. Documentation

Maintain runbooks for:
- Alert response procedures
- Backup restoration steps
- Disaster recovery plans
- Escalation procedures

### 7. Monitoring

Integrate with monitoring systems:

```bash
# Prometheus exporter
./health-check/health_check.sh --json | \
  jq -r '.summary.health_score'

# Datadog custom metrics
./monitoring/monitor_gpus.sh --json | \
  jq -r '.gpus[] | "\(.id): \(.temp)°C"'
```

## Production Deployment

### Pre-deployment Checklist

- [ ] Test all scripts with dry-run mode
- [ ] Configure environment variables
- [ ] Set up log directories with correct permissions
- [ ] Test alert mechanisms (email, webhook)
- [ ] Verify backup destination has sufficient space
- [ ] Configure logrotate
- [ ] Set up monitoring integration
- [ ] Document runbooks
- [ ] Schedule backups during low-traffic hours
- [ ] Test restoration from backup

### Deployment Steps

1. **Install scripts**:
```bash
# Create directory
sudo mkdir -p /opt/ml-automation

# Copy scripts
sudo cp -r exercise-08/* /opt/ml-automation/

# Set ownership
sudo chown -R mluser:mluser /opt/ml-automation
```

2. **Configure environment**:
```bash
# Create environment file
sudo nano /etc/ml-automation.conf

# Add configuration
MODEL_DIR=/opt/ml/models
BACKUP_DIR=/backup/models
RETENTION_DAYS=30
ALERT_EMAIL=ml-ops@company.com
```

3. **Set up scheduling**:
```bash
# Option A: Systemd (recommended)
sudo cp config/systemd/*.service /etc/systemd/system/
sudo cp config/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ml-backup.timer gpu-monitor.timer ml-cleanup.timer

# Option B: Cron
sudo crontab -u mluser /opt/ml-automation/config/crontab.example
```

4. **Configure log rotation**:
```bash
sudo cp config/logrotate-ml.conf /etc/logrotate.d/ml-infrastructure
sudo logrotate -d /etc/logrotate.d/ml-infrastructure  # Test
```

5. **Verify deployment**:
```bash
# Test each script
sudo -u mluser /opt/ml-automation/backups/backup_models.sh --dry-run
sudo -u mluser /opt/ml-automation/health-check/health_check.sh

# Check timers
systemctl list-timers

# Monitor for 24 hours
journalctl -u ml-backup.service -f
```

## Advanced Configuration

### S3 Lifecycle Policies

Automate backup archival:

```json
{
  "Rules": [{
    "Id": "archive-old-backups",
    "Status": "Enabled",
    "Transitions": [{
      "Days": 30,
      "StorageClass": "GLACIER"
    }],
    "Expiration": {
      "Days": 90
    }
  }]
}
```

### Custom Alert Handlers

Create custom alert handlers:

```bash
# /opt/ml-automation/handlers/custom-alert.sh
#!/bin/bash
# Custom alert handler for PagerDuty, Opsgenie, etc.

SEVERITY="$1"
MESSAGE="$2"

# Send to PagerDuty
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H 'Content-Type: application/json' \
  -d "{
    \"routing_key\": \"$PAGERDUTY_KEY\",
    \"event_action\": \"trigger\",
    \"payload\": {
      \"summary\": \"$MESSAGE\",
      \"severity\": \"$SEVERITY\",
      \"source\": \"ml-automation\"
    }
  }"
```

### Multi-node Automation

For distributed ML infrastructure:

```bash
# Run on all nodes
for node in ml-node-{01..10}; do
  ssh $node '/opt/ml-automation/health-check/health_check.sh --json'
done | jq -s '.'
```

## References

- [Exercise Specification](../../learning/mod-002-linux-essentials/exercises/exercise-08-system-automation.md)
- [Cron Documentation](https://man7.org/linux/man-pages/man5/crontab.5.html)
- [Systemd Timers](https://www.freedesktop.org/software/systemd/man/systemd.timer.html)
- [Logrotate Manual](https://linux.die.net/man/8/logrotate)
- [Bash Best Practices](https://google.github.io/styleguide/shellguide.html)

## License

Part of AI Infrastructure Junior Engineer curriculum.

## Support

For issues or questions, refer to:
- Troubleshooting section above
- `docs/ANSWERS.md` for conceptual questions
- Module instructor or TA
