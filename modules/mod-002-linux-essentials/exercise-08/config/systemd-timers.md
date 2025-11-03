# Systemd Timers Configuration

Systemd timers are a modern alternative to cron jobs, offering better logging, dependency management, and integration with systemd.

## Benefits Over Cron

- **Better logging**: Integrated with journald
- **Dependencies**: Can depend on other services
- **Resource control**: Can limit CPU, memory usage
- **Calendar expressions**: More flexible scheduling
- **Missed runs**: Can catch up on missed runs
- **Per-service logs**: `journalctl -u service-name`

## Installation

All systemd files go in `/etc/systemd/system/`

```bash
# Copy service and timer files
sudo cp *.service *.timer /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable timers to start on boot
sudo systemctl enable ml-backup.timer
sudo systemctl enable gpu-monitor.timer
sudo systemctl enable ml-cleanup.timer

# Start timers now
sudo systemctl start ml-backup.timer
sudo systemctl start gpu-monitor.timer
sudo systemctl start ml-cleanup.timer
```

## ML Backup Service and Timer

### File: `/etc/systemd/system/ml-backup.service`

```ini
[Unit]
Description=ML Model Backup Service
After=network.target
Documentation=https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions

[Service]
Type=oneshot
User=mluser
Group=mluser
WorkingDirectory=/home/mluser/exercise-08/backups
ExecStart=/home/mluser/exercise-08/backups/backup_models.sh
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryLimit=2G
CPUQuota=50%

# Environment
Environment="MODEL_DIR=/opt/ml/models"
Environment="BACKUP_DIR=/backup/models"
Environment="RETENTION_DAYS=30"

# Restart policy (for failures)
Restart=on-failure
RestartSec=300

[Install]
WantedBy=multi-user.target
```

### File: `/etc/systemd/system/ml-backup.timer`

```ini
[Unit]
Description=Run ML backup daily
Requires=ml-backup.service
Documentation=https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions

[Timer]
# Run daily at 2 AM
OnCalendar=daily
OnCalendar=02:00

# If system was down, run within 1 hour of boot
Persistent=true

# Randomize start time by up to 10 minutes (reduces load spikes)
RandomizedDelaySec=600

# Don't run if system is on battery
ConditionACPower=true

[Install]
WantedBy=timers.target
```

## GPU Monitor Service and Timer

### File: `/etc/systemd/system/gpu-monitor.service`

```ini
[Unit]
Description=GPU Health Monitoring Service
After=network.target nvidia-persistenced.service
Documentation=https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions

[Service]
Type=oneshot
User=mluser
Group=mluser
WorkingDirectory=/home/mluser/exercise-08/monitoring
ExecStart=/home/mluser/exercise-08/monitoring/monitor_gpus.sh
StandardOutput=journal
StandardError=journal

# Environment
Environment="TEMP_THRESHOLD=80"
Environment="MEMORY_THRESHOLD=90"
Environment="ALERT_EMAIL=ml-ops@company.com"

# Don't fail if GPU not available
SuccessExitStatus=0 1

[Install]
WantedBy=multi-user.target
```

### File: `/etc/systemd/system/gpu-monitor.timer`

```ini
[Unit]
Description=Run GPU monitoring every 5 minutes
Requires=gpu-monitor.service
Documentation=https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions

[Timer]
# Run every 5 minutes
OnCalendar=*:0/5
OnBootSec=2min

# Start immediately if missed
Persistent=true

[Install]
WantedBy=timers.target
```

## ML Cleanup Service and Timer

### File: `/etc/systemd/system/ml-cleanup.service`

```ini
[Unit]
Description=ML Artifacts Cleanup Service
After=network.target
Documentation=https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions

[Service]
Type=oneshot
User=mluser
Group=mluser
WorkingDirectory=/home/mluser/exercise-08/cleanup
ExecStart=/home/mluser/exercise-08/cleanup/cleanup_ml_artifacts.sh
StandardOutput=journal
StandardError=journal

# Resource limits (cleanup can be I/O intensive)
IOWeight=100
CPUQuota=30%

# Environment
Environment="EXPERIMENTS_RETENTION=30"
Environment="CHECKPOINTS_RETENTION=7"
Environment="AGGRESSIVE=false"

# Don't restart on failure
Restart=no

[Install]
WantedBy=multi-user.target
```

### File: `/etc/systemd/system/ml-cleanup.timer`

```ini
[Unit]
Description=Run ML cleanup weekly
Requires=ml-cleanup.service
Documentation=https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions

[Timer]
# Run every Sunday at 3 AM
OnCalendar=Sun 03:00

# Run within 1 day if missed
Persistent=true

# Random delay up to 30 minutes
RandomizedDelaySec=1800

[Install]
WantedBy=timers.target
```

## Full Maintenance Suite Service and Timer

### File: `/etc/systemd/system/ml-maintenance.service`

```ini
[Unit]
Description=Complete ML Infrastructure Maintenance
After=network.target
Documentation=https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions

[Service]
Type=oneshot
User=mluser
Group=mluser
WorkingDirectory=/home/mluser/exercise-08
ExecStart=/home/mluser/exercise-08/run_all_maintenance.sh
StandardOutput=journal
StandardError=journal
TimeoutStartSec=3600

# Continue even if some tasks fail
SuccessExitStatus=0 1

[Install]
WantedBy=multi-user.target
```

### File: `/etc/systemd/system/ml-maintenance.timer`

```ini
[Unit]
Description=Run complete maintenance suite weekly
Requires=ml-maintenance.service
Documentation=https://github.com/ai-infra-curriculum/ai-infra-junior-engineer-solutions

[Timer]
# Run every Sunday at 1 AM
OnCalendar=Sun 01:00
Persistent=true
RandomizedDelaySec=1800

[Install]
WantedBy=timers.target
```

## Management Commands

### Check Timer Status

```bash
# List all timers
systemctl list-timers

# Check specific timer
systemctl status ml-backup.timer

# View timer details
systemctl show ml-backup.timer

# See when timer last ran and when it will run next
systemctl list-timers ml-backup.timer
```

### Run Service Manually

```bash
# Trigger service immediately (won't wait for timer)
sudo systemctl start ml-backup.service

# Check service status
systemctl status ml-backup.service

# View service logs
journalctl -u ml-backup.service

# Follow logs in real-time
journalctl -u ml-backup.service -f

# View logs since last hour
journalctl -u ml-backup.service --since "1 hour ago"
```

### Enable/Disable Timers

```bash
# Enable timer (start on boot)
sudo systemctl enable ml-backup.timer

# Disable timer
sudo systemctl disable ml-backup.timer

# Start timer now
sudo systemctl start ml-backup.timer

# Stop timer
sudo systemctl stop ml-backup.timer

# Restart timer
sudo systemctl restart ml-backup.timer
```

### Debugging

```bash
# Check if timer is active
systemctl is-active ml-backup.timer

# Check if timer is enabled
systemctl is-enabled ml-backup.timer

# View timer configuration
systemctl cat ml-backup.timer

# View service configuration
systemctl cat ml-backup.service

# Check for errors
systemctl status ml-backup.service --no-pager -l

# View full logs with debug info
journalctl -u ml-backup.service -b --no-pager
```

## Calendar Expression Examples

```ini
# Every 5 minutes
OnCalendar=*:0/5

# Every hour at minute 30
OnCalendar=*:30

# Every 6 hours
OnCalendar=0/6:00

# Daily at 2 AM
OnCalendar=daily
OnCalendar=02:00

# Weekly on Sunday at 3 AM
OnCalendar=Sun 03:00

# Monthly on the 1st at midnight
OnCalendar=*-*-01 00:00

# Weekdays at 9 AM
OnCalendar=Mon..Fri 09:00

# Multiple times
OnCalendar=Mon,Wed,Fri 10:00
OnCalendar=Tue,Thu 14:00
```

## Best Practices

1. **Use `Type=oneshot`**: For tasks that run and complete
2. **Set resource limits**: Prevent runaway processes
3. **Use `Persistent=true`**: Catch up on missed runs
4. **Add `RandomizedDelaySec`**: Avoid load spikes
5. **Use absolute paths**: Don't rely on PATH
6. **Log to journald**: Integrated logging
7. **Set appropriate user**: Don't run as root unless needed
8. **Add documentation**: Help future maintainers
9. **Test services**: Run manually before enabling timer
10. **Monitor logs**: Regularly check journalctl output

## Troubleshooting

### Timer not firing

```bash
# Check timer is enabled and active
systemctl list-timers --all | grep ml-backup

# Verify timer configuration
systemd-analyze calendar "daily"
systemd-analyze calendar "02:00"

# Check system time
timedatectl status
```

### Service fails

```bash
# Check service status
systemctl status ml-backup.service

# View recent logs
journalctl -u ml-backup.service -n 50

# Test script manually
/home/mluser/exercise-08/backups/backup_models.sh

# Check permissions
ls -la /home/mluser/exercise-08/backups/backup_models.sh

# Verify user can run
sudo -u mluser /home/mluser/exercise-08/backups/backup_models.sh
```

### Service stuck

```bash
# Check if service is running
systemctl is-active ml-backup.service

# View process
systemctl status ml-backup.service

# Kill if stuck
sudo systemctl kill ml-backup.service

# Restart
sudo systemctl restart ml-backup.service
```

## Comparison: Cron vs Systemd Timers

| Feature | Cron | Systemd Timers |
|---------|------|----------------|
| Scheduling | Simple syntax | Calendar expressions |
| Logging | Mail/files | journald |
| Dependencies | None | Full systemd integration |
| Resource limits | No | Yes (cgroups) |
| Missed runs | Skip | Persistent option |
| Per-service logs | No | Yes |
| Email on failure | Yes | Via OnFailure |
| Randomization | No | RandomizedDelaySec |
| Requires | cron daemon | systemd (usually present) |

## References

- [systemd.timer man page](https://www.freedesktop.org/software/systemd/man/systemd.timer.html)
- [systemd.service man page](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
- [systemd.time man page](https://www.freedesktop.org/software/systemd/man/systemd.time.html)
