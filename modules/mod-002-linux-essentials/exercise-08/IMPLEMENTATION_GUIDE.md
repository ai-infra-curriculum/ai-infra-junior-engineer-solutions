# Implementation Guide: System Automation and Maintenance for ML Infrastructure

## Overview

This guide teaches you to automate system maintenance tasks for ML infrastructure. Learn to create automated backup scripts, set up scheduled tasks, implement health checks, and configure log rotation—all critical for production ML systems.

**Estimated Time:** 120-150 minutes
**Difficulty:** Intermediate to Advanced

## Prerequisites

- Completed Exercises 01-07
- Understanding of systemd and cron
- Sudo privileges
- Basic ML workflow knowledge

## Phase 1: Automated Model Backup System (35 minutes)

### Step 1.1: Create Backup Script

```bash
mkdir -p ~/ml-automation
cd ~/ml-automation

cat > backup_models.sh << 'EOF'
#!/bin/bash
# Automated ML model backup script

set -e  # Exit on error

# Configuration
MODEL_DIR="${1:-/var/ml/models}"
BACKUP_DIR="${2:-/var/ml/backups}"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo "Starting model backup at $(date)"
echo "Source: $MODEL_DIR"
echo "Destination: $BACKUP_DIR"
echo ""

# Create compressed backup
BACKUP_FILE="$BACKUP_DIR/models_backup_$TIMESTAMP.tar.gz"

tar -czf "$BACKUP_FILE" \
    -C "$(dirname "$MODEL_DIR")" \
    "$(basename "$MODEL_DIR")" \
    2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ Backup created: $BACKUP_FILE"
    BACKUP_SIZE=$(du -h "$BACKUP_FILE" | awk '{print $1}')
    echo "  Size: $BACKUP_SIZE"
else
    echo "✗ Backup failed!"
    exit 1
fi

# Verify backup integrity
echo "Verifying backup integrity..."
if tar -tzf "$BACKUP_FILE" > /dev/null 2>&1; then
    echo "✓ Backup integrity verified"
else
    echo "✗ Backup corrupted!"
    exit 1
fi

# Clean old backups
echo "Cleaning backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -name "models_backup_*.tar.gz" \
    -mtime +$RETENTION_DAYS -delete

REMAINING=$(find "$BACKUP_DIR" -name "models_backup_*.tar.gz" | wc -l)
echo "✓ Cleanup complete. Remaining backups: $REMAINING"

# Report disk usage
echo ""
echo "Disk usage:"
du -sh "$BACKUP_DIR"
df -h "$BACKUP_DIR" | tail -1

echo ""
echo "Backup completed successfully at $(date)"
EOF

chmod +x backup_models.sh
```

### Step 1.2: Test Backup Script

```bash
# Create test model directory
mkdir -p /tmp/test_models
echo "model_v1.pt" > /tmp/test_models/model.pt

# Run backup
./backup_models.sh /tmp/test_models /tmp/test_backups

# Verify backup
ls -lh /tmp/test_backups/

# Test restoration
mkdir -p /tmp/restore_test
tar -xzf /tmp/test_backups/models_backup_*.tar.gz -C /tmp/restore_test
ls -la /tmp/restore_test/test_models/
```

### Step 1.3: Add Logging and Notifications

```bash
cat > backup_models_enhanced.sh << 'EOF'
#!/bin/bash
# Enhanced backup script with logging and notifications

MODEL_DIR="${1:-/var/ml/models}"
BACKUP_DIR="${2:-/var/ml/backups}"
LOG_FILE="/var/log/ml_backup.log"
EMAIL="${3:-admin@company.com}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Run backup
log "Starting backup from $MODEL_DIR"

if ./backup_models.sh "$MODEL_DIR" "$BACKUP_DIR" >> "$LOG_FILE" 2>&1; then
    log "Backup completed successfully"
    STATUS="SUCCESS"
else
    log "Backup FAILED"
    STATUS="FAILED"
fi

# Send notification (if mail is configured)
if command -v mail &> /dev/null; then
    echo "Backup status: $STATUS" | mail -s "ML Backup $STATUS" "$EMAIL"
fi

log "Backup process finished"
EOF

chmod +x backup_models_enhanced.sh
```

**Validation:**
- [ ] Backup script runs successfully
- [ ] Creates compressed archive
- [ ] Verifies backup integrity
- [ ] Cleans old backups
- [ ] Logs output

## Phase 2: Scheduled Tasks with Cron (25 minutes)

### Step 2.1: Cron Basics

```bash
# View current cron jobs
crontab -l

# Edit cron jobs
crontab -e

# Cron syntax:
# MIN HOUR DAY MONTH WEEKDAY COMMAND
#  *    *    *    *      *     command to execute
#
# Examples:
# 0 2 * * * /path/to/backup.sh        # Daily at 2 AM
# 0 */4 * * * /path/to/check.sh       # Every 4 hours
# 0 0 * * 0 /path/to/weekly.sh        # Weekly on Sunday
# 0 3 1 * * /path/to/monthly.sh       # Monthly on 1st

# Create cron schedule
cat > setup_cron.sh << 'EOF'
#!/bin/bash
# Set up automated tasks

# Backup models daily at 2 AM
(crontab -l 2>/dev/null; echo "0 2 * * * /home/mluser/ml-automation/backup_models.sh") | crontab -

# Clean temp files every 6 hours
(crontab -l 2>/dev/null; echo "0 */6 * * * /home/mluser/ml-automation/cleanup_temp.sh") | crontab -

# Health check every hour
(crontab -l 2>/dev/null; echo "0 * * * * /home/mluser/ml-automation/health_check.sh") | crontab -

# GPU monitoring every 5 minutes
(crontab -l 2>/dev/null; echo "*/5 * * * * /home/mluser/ml-automation/gpu_monitor.sh") | crontab -

echo "Cron jobs configured:"
crontab -l
EOF

chmod +x setup_cron.sh
```

### Step 2.2: Cron with Error Handling

```bash
cat > cron_wrapper.sh << 'EOF'
#!/bin/bash
# Wrapper for cron jobs with error handling

SCRIPT="$1"
LOG_DIR="/var/log/ml_automation"

mkdir -p "$LOG_DIR"

SCRIPT_NAME=$(basename "$SCRIPT" .sh)
LOG_FILE="$LOG_DIR/${SCRIPT_NAME}_$(date +%Y%m%d).log"

echo "=== Starting $SCRIPT_NAME at $(date) ===" >> "$LOG_FILE"

if bash "$SCRIPT" >> "$LOG_FILE" 2>&1; then
    echo "=== Completed successfully at $(date) ===" >> "$LOG_FILE"
    exit 0
else
    echo "=== FAILED at $(date) ===" >> "$LOG_FILE"
    # Send alert
    echo "Cron job $SCRIPT_NAME failed" | mail -s "Cron Failure" admin@company.com
    exit 1
fi
EOF

chmod +x cron_wrapper.sh

# Use in crontab:
# 0 2 * * * /path/to/cron_wrapper.sh /path/to/backup_models.sh
```

**Validation:**
- [ ] Cron jobs configured
- [ ] Can list cron schedule
- [ ] Understand cron syntax
- [ ] Error handling in place

## Phase 3: Systemd Timers (Alternative to Cron) (25 minutes)

### Step 3.1: Create Systemd Service

```bash
# Create service file
sudo cat > /etc/systemd/system/ml-backup.service << 'EOF'
[Unit]
Description=ML Model Backup Service
After=network.target

[Service]
Type=oneshot
User=mluser
ExecStart=/home/mluser/ml-automation/backup_models.sh
StandardOutput=journal
StandardError=journal
EOF
```

### Step 3.2: Create Systemd Timer

```bash
# Create timer file
sudo cat > /etc/systemd/system/ml-backup.timer << 'EOF'
[Unit]
Description=ML Model Backup Timer
Requires=ml-backup.service

[Timer]
OnCalendar=daily
OnCalendar=02:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable and start timer
sudo systemctl enable ml-backup.timer
sudo systemctl start ml-backup.timer

# Check timer status
systemctl status ml-backup.timer

# List all timers
systemctl list-timers
```

### Step 3.3: Test Service Manually

```bash
# Run service manually
sudo systemctl start ml-backup.service

# Check status
systemctl status ml-backup.service

# View logs
journalctl -u ml-backup.service -n 50
```

**Validation:**
- [ ] Service file created
- [ ] Timer file created
- [ ] Timer enabled and running
- [ ] Can view logs with journalctl

## Phase 4: Health Check System (30 minutes)

### Step 4.1: Create Comprehensive Health Check

```bash
cat > health_check.sh << 'EOF'
#!/bin/bash
# ML Infrastructure Health Check

ALERT_EMAIL="admin@company.com"
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEM=90
ALERT_THRESHOLD_DISK=85

# Health check result
HEALTH_STATUS="OK"
ALERTS=()

# Check disk space
check_disk() {
    echo "Checking disk space..."

    while read -r line; do
        USAGE=$(echo "$line" | awk '{print $5}' | sed 's/%//')
        MOUNT=$(echo "$line" | awk '{print $6}')

        if [ "$USAGE" -gt $ALERT_THRESHOLD_DISK ]; then
            ALERTS+=("DISK: $MOUNT is ${USAGE}% full")
            HEALTH_STATUS="WARNING"
        fi
    done < <(df -h | grep -E "^/dev/")
}

# Check memory
check_memory() {
    echo "Checking memory..."

    MEM_USAGE=$(free | grep Mem | awk '{print int($3/$2 * 100)}')

    if [ "$MEM_USAGE" -gt $ALERT_THRESHOLD_MEM ]; then
        ALERTS+=("MEMORY: Usage at ${MEM_USAGE}%")
        HEALTH_STATUS="WARNING"
    fi
}

# Check CPU load
check_cpu() {
    echo "Checking CPU load..."

    CPU_LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    CPU_CORES=$(nproc)
    CPU_PERCENT=$(echo "$CPU_LOAD / $CPU_CORES * 100" | bc)

    if [ "$CPU_PERCENT" -gt $ALERT_THRESHOLD_CPU ]; then
        ALERTS+=("CPU: Load average ${CPU_LOAD} (${CPU_PERCENT}% of capacity)")
        HEALTH_STATUS="WARNING"
    fi
}

# Check GPU (if available)
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "Checking GPU..."

        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits | while read line; do
            GPU_ID=$(echo "$line" | cut -d',' -f1)
            GPU_UTIL=$(echo "$line" | cut -d',' -f2)
            MEM_USED=$(echo "$line" | cut -d',' -f3)
            MEM_TOTAL=$(echo "$line" | cut -d',' -f4)

            MEM_PERCENT=$(echo "$MEM_USED / $MEM_TOTAL * 100" | bc)

            if [ "$MEM_PERCENT" -gt 95 ]; then
                ALERTS+=("GPU$GPU_ID: Memory at ${MEM_PERCENT}%")
                HEALTH_STATUS="WARNING"
            fi
        done
    fi
}

# Check critical services
check_services() {
    echo "Checking services..."

    SERVICES=("docker" "ssh")

    for service in "${SERVICES[@]}"; do
        if ! systemctl is-active --quiet "$service"; then
            ALERTS+=("SERVICE: $service is not running")
            HEALTH_STATUS="CRITICAL"
        fi
    done
}

# Run all checks
check_disk
check_memory
check_cpu
check_gpu
check_services

# Report results
echo ""
echo "=== Health Check Results ==="
echo "Status: $HEALTH_STATUS"
echo "Timestamp: $(date)"
echo ""

if [ ${#ALERTS[@]} -eq 0 ]; then
    echo "✓ All checks passed"
else
    echo "⚠ Alerts:"
    for alert in "${ALERTS[@]}"; do
        echo "  - $alert"
    done

    # Send alert email
    if command -v mail &> /dev/null; then
        (
            echo "Health Check Status: $HEALTH_STATUS"
            echo ""
            for alert in "${ALERTS[@]}"; do
                echo "- $alert"
            done
        ) | mail -s "ML Infrastructure Health Alert" "$ALERT_EMAIL"
    fi
fi

# Exit with error code if critical
if [ "$HEALTH_STATUS" = "CRITICAL" ]; then
    exit 1
fi
EOF

chmod +x health_check.sh
./health_check.sh
```

### Step 4.2: Service-Specific Health Checks

```bash
cat > check_ml_services.sh << 'EOF'
#!/bin/bash
# Check ML-specific services

# Check if model API is responding
check_api() {
    if curl -s -f http://localhost:8080/health > /dev/null 2>&1; then
        echo "✓ Model API is healthy"
        return 0
    else
        echo "✗ Model API is down"
        return 1
    fi
}

# Check Redis connection
check_redis() {
    if redis-cli ping > /dev/null 2>&1; then
        echo "✓ Redis is healthy"
        return 0
    else
        echo "✗ Redis is down"
        return 1
    fi
}

# Check database connection
check_database() {
    if pg_isready -h localhost > /dev/null 2>&1; then
        echo "✓ PostgreSQL is healthy"
        return 0
    else
        echo "✗ PostgreSQL is down"
        return 1
    fi
}

# Run all service checks
check_api
check_redis
check_database
EOF

chmod +x check_ml_services.sh
```

**Validation:**
- [ ] Health check runs successfully
- [ ] Checks all critical resources
- [ ] Sends alerts when thresholds exceeded
- [ ] Logs results

## Phase 5: Log Rotation and Cleanup (20 minutes)

### Step 5.1: Configure Log Rotation

```bash
# Create logrotate config
sudo cat > /etc/logrotate.d/ml-services << 'EOF'
/var/log/ml_automation/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 mluser mluser
    postrotate
        # Optional: restart service after rotation
        # systemctl reload ml-service
    endscript
}

/var/ml/training/logs/*.log {
    size 100M
    rotate 10
    compress
    missingok
    notifempty
    create 0644 mluser mluser
}
EOF

# Test logrotate config
sudo logrotate -d /etc/logrotate.d/ml-services

# Force rotation (testing)
sudo logrotate -f /etc/logrotate.d/ml-services
```

### Step 5.2: Custom Cleanup Script

```bash
cat > cleanup_temp.sh << 'EOF'
#!/bin/bash
# Clean temporary ML files

TEMP_DIRS=(
    "/tmp/ml_cache"
    "/var/ml/checkpoints/temp"
    "/home/mluser/.cache/torch"
)

RETENTION_DAYS=7

echo "Starting cleanup at $(date)"

for dir in "${TEMP_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Cleaning $dir..."

        # Remove old files
        find "$dir" -type f -mtime +$RETENTION_DAYS -delete

        # Remove empty directories
        find "$dir" -type d -empty -delete

        echo "  Current size: $(du -sh "$dir" 2>/dev/null | awk '{print $1}')"
    fi
done

# Clean pip cache
pip cache purge > /dev/null 2>&1

echo "Cleanup completed at $(date)"
EOF

chmod +x cleanup_temp.sh
```

**Validation:**
- [ ] Logrotate configured
- [ ] Can test rotation manually
- [ ] Cleanup script works
- [ ] Old files removed

## Phase 6: GPU Monitoring Automation (20 minutes)

### Step 6.1: GPU Monitoring Script

```bash
cat > gpu_monitor.sh << 'EOF'
#!/bin/bash
# Monitor GPU usage and log metrics

LOG_FILE="/var/log/gpu_metrics.log"
ALERT_TEMP=85
ALERT_MEM=95

if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found"
    exit 1
fi

# Collect metrics
nvidia-smi --query-gpu=timestamp,index,name,temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader | while IFS=',' read -r timestamp gpu_id name temp util mem_used mem_total; do

    # Clean values
    temp=$(echo "$temp" | tr -d ' ')
    util=$(echo "$util" | tr -d ' %')
    mem_used=$(echo "$mem_used" | tr -d ' MiB')
    mem_total=$(echo "$mem_total" | tr -d ' MiB')

    # Calculate memory percentage
    mem_percent=$(echo "scale=2; $mem_used / $mem_total * 100" | bc)

    # Log metrics
    echo "$timestamp,GPU$gpu_id,$temp°C,$util%,$mem_percent%" >> "$LOG_FILE"

    # Check for alerts
    if [ "$temp" -gt $ALERT_TEMP ]; then
        echo "WARNING: GPU$gpu_id temperature is ${temp}°C" | logger -t gpu-monitor
    fi

    if (( $(echo "$mem_percent > $ALERT_MEM" | bc -l) )); then
        echo "WARNING: GPU$gpu_id memory usage is ${mem_percent}%" | logger -t gpu-monitor
    fi
done

# Keep only last 7 days of logs
find "$(dirname "$LOG_FILE")" -name "gpu_metrics.log.*" -mtime +7 -delete
EOF

chmod +x gpu_monitor.sh
```

### Step 6.2: GPU Alert Script

```bash
cat > gpu_alert.sh << 'EOF'
#!/bin/bash
# Alert on GPU issues

nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits | while IFS=',' read -r gpu_id temp util mem_used mem_total; do

    mem_percent=$(echo "$mem_used / $mem_total * 100" | bc)

    if [ "$temp" -gt 85 ] || [ "$mem_percent" -gt 95 ]; then
        MESSAGE="GPU$gpu_id Alert: Temp=${temp}°C, Memory=${mem_percent}%"
        echo "$MESSAGE" | mail -s "GPU Alert" admin@company.com
        logger -p user.warning "$MESSAGE"
    fi
done
EOF

chmod +x gpu_alert.sh
```

**Validation:**
- [ ] GPU monitoring logs metrics
- [ ] Alerts on high temperature/memory
- [ ] Logs rotate properly

## Best Practices Summary

### Automation Design

✅ Make scripts idempotent (safe to run multiple times)
✅ Add comprehensive error handling
✅ Log all operations
✅ Send alerts for failures
✅ Test scripts before scheduling

### Scheduling

✅ Use cron for simple periodic tasks
✅ Use systemd timers for complex workflows
✅ Avoid overlapping executions
✅ Set appropriate timeouts
✅ Monitor execution success

### Backups

✅ Verify backup integrity
✅ Test restoration regularly
✅ Implement retention policies
✅ Monitor backup storage space
✅ Encrypt sensitive backups

### Monitoring

✅ Check all critical resources
✅ Set reasonable alert thresholds
✅ Avoid alert fatigue
✅ Log metrics for trending
✅ Test alert mechanisms

### Maintenance

✅ Clean temporary files regularly
✅ Rotate logs before they fill disk
✅ Update automation scripts as system changes
✅ Document automation workflows
✅ Review and optimize schedules

## Completion Checklist

### Scripts Created
- [ ] Automated backup script
- [ ] Health check script
- [ ] Cleanup script
- [ ] GPU monitoring script
- [ ] Service check script

### Scheduling
- [ ] Configured cron jobs or systemd timers
- [ ] Automated daily backups
- [ ] Scheduled cleanup tasks
- [ ] Regular health checks

### Monitoring
- [ ] Health checks running
- [ ] GPU monitoring active
- [ ] Alert notifications working
- [ ] Logs being collected

### Maintenance
- [ ] Log rotation configured
- [ ] Cleanup automation in place
- [ ] Backup retention enforced
- [ ] All scripts tested

## Next Steps

1. **Production Implementation:**
   - Deploy automation to production servers
   - Set up centralized monitoring
   - Configure proper alerting channels

2. **Advanced Automation:**
   - Ansible for multi-server automation
   - Infrastructure as Code (Terraform)
   - CI/CD pipelines for model deployment

3. **Enterprise Features:**
   - Backup to cloud storage (S3, GCS)
   - Distributed monitoring (Prometheus)
   - Centralized logging (ELK stack)

## Resources

- [Cron Documentation](https://man7.org/linux/man-pages/man5/crontab.5.html)
- [Systemd Timers](https://www.freedesktop.org/software/systemd/man/systemd.timer.html)
- [logrotate Manual](https://linux.die.net/man/8/logrotate)
- [Bash Scripting Guide](https://www.gnu.org/software/bash/manual/)

Congratulations! You can now automate ML infrastructure maintenance.
