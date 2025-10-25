# Quick Start Guide

## Exercise 04: System Administration Scripts

This directory contains 8 production-ready system administration scripts for ML infrastructure management, totaling over 5,000 lines of robust, tested code.

## Installation

```bash
cd solutions
chmod +x *.sh
```

## Quick Test

```bash
# Test all scripts
cd tests
./test_scripts.sh
```

## 5-Minute Tour

### 1. System Monitoring
```bash
# Check system health
./solutions/system_monitor.sh --check

# Generate monitoring report
./solutions/system_monitor.sh --report
```

### 2. User Management
```bash
# List all users
./solutions/user_management.sh list-users

# View user info
./solutions/user_management.sh info $USER
```

### 3. Disk Management
```bash
# Check disk usage
./solutions/disk_manager.sh check

# Find large files
./solutions/disk_manager.sh large-files /home
```

### 4. Security Audit
```bash
# Run security audit
./solutions/security_audit.sh --audit
```

### 5. Service Management
```bash
# View service status
./solutions/manage_services.sh status
```

### 6. Backup System
```bash
# List backups
./solutions/backup_automation.sh list

# Dry run backup
./solutions/backup_automation.sh backup --dry-run
```

### 7. Log Management
```bash
# Analyze logs
./solutions/log_rotation.sh analyze
```

### 8. System Maintenance
```bash
# Disk analysis
./solutions/system_maintenance.sh --disk
```

## Common Tasks

### Daily Operations
```bash
# Morning health check
./solutions/system_monitor.sh --check
./solutions/disk_manager.sh check
./solutions/manage_services.sh status
```

### Weekly Maintenance
```bash
# Run full maintenance
sudo ./solutions/system_maintenance.sh --all
sudo ./solutions/log_rotation.sh all
sudo ./solutions/security_audit.sh --audit --report
```

### User Administration
```bash
# Add new ML engineer
sudo ./solutions/user_management.sh add-user mluser --groups docker,sudo

# Setup SSH access
sudo ./solutions/user_management.sh ssh-key mluser

# Generate user audit
sudo ./solutions/user_management.sh audit
```

### Backup & Recovery
```bash
# Full backup
sudo ./solutions/backup_automation.sh backup --type full

# Incremental backup
sudo ./solutions/backup_automation.sh backup

# Verify backup
./solutions/backup_automation.sh verify backup-YYYYMMDD-HHMMSS

# Restore if needed
sudo ./solutions/backup_automation.sh restore backup-YYYYMMDD-HHMMSS
```

## Automation Setup

### Cron Jobs
```bash
# Edit crontab
crontab -e

# Add these lines for automated management:

# Daily monitoring at 2 AM
0 2 * * * /path/to/solutions/system_monitor.sh --check --report

# Daily backup at 1 AM
0 1 * * * /path/to/solutions/backup_automation.sh backup

# Weekly security audit (Sunday 3 AM)
0 3 * * 0 /path/to/solutions/security_audit.sh --audit --report

# Weekly cleanup (Sunday 4 AM)
0 4 * * 0 /path/to/solutions/log_rotation.sh all
0 4 * * 0 /path/to/solutions/disk_manager.sh cleanup
```

## Important Notes

1. **Permissions**: Most operations require root (sudo)
2. **Testing**: Always test with `--dry-run` first
3. **Logs**: Check `/var/log/*` for detailed logs
4. **Help**: All scripts have `--help` option
5. **Safety**: Scripts include error handling and validation

## Script Features

All scripts include:
- Comprehensive error handling
- Input validation
- Detailed logging
- Verbose mode
- Dry-run capability
- Color-coded output
- Help documentation
- Safe defaults

## Monitoring Thresholds

Default alert thresholds (customizable):
- CPU: 80%
- Memory: 80%
- Disk: 80%

## File Locations

```
solutions/               # All executable scripts
tests/                   # Test suite
/var/log/               # Log files
/var/backups/           # Backup storage
/var/log/security-reports/  # Security audit reports
```

## Need Help?

1. Check script help: `./script.sh --help`
2. Read full documentation: `README.md`
3. Check logs: `tail -f /var/log/script-name.log`
4. Test syntax: `bash -n script.sh`
5. Debug mode: `bash -x script.sh`

## Scripts Overview

| Script | Lines | Purpose |
|--------|-------|---------|
| system_monitor.sh | 784 | System health monitoring |
| user_management.sh | 1128 | User & group administration |
| backup_automation.sh | 962 | Backup & recovery system |
| log_rotation.sh | 258 | Log management |
| security_audit.sh | 426 | Security scanning |
| disk_manager.sh | 352 | Disk usage management |
| manage_services.sh | 670 | Service management |
| system_maintenance.sh | 617 | System maintenance |

**Total: 5,197 lines of production code**

## What's Next?

1. Customize scripts for your environment
2. Set up automated schedules
3. Configure alerting
4. Test disaster recovery
5. Train your team

**Ready to manage your ML infrastructure like a pro!**
