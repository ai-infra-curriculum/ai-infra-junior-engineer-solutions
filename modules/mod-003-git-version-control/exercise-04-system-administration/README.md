# Exercise 04: System Administration

## Overview

Master Linux system administration for ML infrastructure. Learn to manage services, automate tasks, maintain systems, handle packages, and ensure system reliability for production ML workloads.

## Learning Objectives

- ✅ Manage system services with systemd
- ✅ Schedule tasks with cron and systemd timers
- ✅ Perform system maintenance and updates
- ✅ Manage software packages
- ✅ Configure system startup and boot
- ✅ Handle system logs and log rotation
- ✅ Monitor system health and performance
- ✅ Automate administrative tasks

## Topics Covered

### 1. Service Management with systemd

#### Basic Service Commands

```bash
# Start a service
sudo systemctl start service_name

# Stop a service
sudo systemctl stop service_name

# Restart a service
sudo systemctl restart service_name

# Reload service configuration
sudo systemctl reload service_name

# Service status
systemctl status service_name
systemctl is-active service_name
systemctl is-enabled service_name

# Enable service (start on boot)
sudo systemctl enable service_name

# Disable service
sudo systemctl disable service_name

# Enable and start
sudo systemctl enable --now service_name
```

#### Managing ML Services

```bash
# Common ML-related services
systemctl status docker          # Docker daemon
systemctl status nvidia-persistenced  # NVIDIA persistence daemon
systemctl status jupyter         # Jupyter server (if installed as service)
systemctl status mlflow          # MLflow tracking server

# View all running services
systemctl list-units --type=service --state=running

# View all services (active and inactive)
systemctl list-units --type=service --all

# View failed services
systemctl list-units --type=service --state=failed
```

#### Creating Custom Service Files

```bash
# Service file location
/etc/systemd/system/service_name.service

# Example: ML Training Service
cat > /etc/systemd/system/ml-training.service <<'EOF'
[Unit]
Description=ML Training Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=ml-user
Group=ml-group
WorkingDirectory=/opt/ml/training
Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="PYTHONPATH=/opt/ml/lib"
ExecStart=/usr/bin/python3 /opt/ml/training/train.py
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/ml-training/output.log
StandardError=append:/var/log/ml-training/error.log

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Start service
sudo systemctl start ml-training

# Enable on boot
sudo systemctl enable ml-training
```

#### Example: Jupyter Notebook Service

```bash
# /etc/systemd/system/jupyter.service
[Unit]
Description=Jupyter Notebook Server
After=network.target

[Service]
Type=simple
User=ml-user
WorkingDirectory=/home/ml-user/notebooks
Environment="PATH=/home/ml-user/.local/bin:/usr/local/bin:/usr/bin"
ExecStart=/home/ml-user/.local/bin/jupyter notebook --no-browser --port=8888
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Example: MLflow Tracking Server

```bash
# /etc/systemd/system/mlflow.service
[Unit]
Description=MLflow Tracking Server
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=mlflow
Group=mlflow
WorkingDirectory=/opt/mlflow
Environment="MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:password@localhost/mlflow"
Environment="MLFLOW_DEFAULT_ARTIFACT_ROOT=/data/mlflow/artifacts"
ExecStart=/usr/local/bin/mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT}
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

### 2. Task Scheduling

#### Cron Jobs

```bash
# Edit crontab
crontab -e

# List cron jobs
crontab -l

# Remove all cron jobs
crontab -r

# Edit cron jobs for another user
sudo crontab -u username -e

# Cron syntax
# ┌───────────── minute (0-59)
# │ ┌─────────── hour (0-23)
# │ │ ┌───────── day of month (1-31)
# │ │ │ ┌─────── month (1-12)
# │ │ │ │ ┌───── day of week (0-6, Sunday=0)
# │ │ │ │ │
# * * * * * command

# Examples:
# Daily backup at 2 AM
0 2 * * * /opt/scripts/backup_models.sh

# Cleanup logs every Sunday at 3 AM
0 3 * * 0 /opt/scripts/cleanup_logs.sh

# Model training every 6 hours
0 */6 * * * /opt/scripts/train_models.sh

# Health check every 15 minutes
*/15 * * * * /opt/scripts/health_check.sh

# Monthly report on 1st of month
0 0 1 * * /opt/scripts/monthly_report.sh

# Weekday backups (Mon-Fri) at 11 PM
0 23 * * 1-5 /opt/scripts/daily_backup.sh

# GPU memory cleanup every hour
0 * * * * /usr/bin/nvidia-smi --gpu-reset

# Special strings
@reboot /opt/scripts/startup.sh
@hourly /opt/scripts/hourly_task.sh
@daily /opt/scripts/daily_task.sh
@weekly /opt/scripts/weekly_task.sh
@monthly /opt/scripts/monthly_task.sh
```

#### Systemd Timers (Modern Alternative to Cron)

```bash
# Timer file: /etc/systemd/system/ml-backup.timer
[Unit]
Description=ML Backup Timer
Requires=ml-backup.service

[Timer]
OnCalendar=daily
OnCalendar=02:00
Persistent=true
Unit=ml-backup.service

[Install]
WantedBy=timers.target

# Service file: /etc/systemd/system/ml-backup.service
[Unit]
Description=ML Backup Service
After=network.target

[Service]
Type=oneshot
User=backup
ExecStart=/opt/scripts/backup_models.sh

# Enable and start timer
sudo systemctl enable ml-backup.timer
sudo systemctl start ml-backup.timer

# View timer status
systemctl list-timers
systemctl status ml-backup.timer

# OnCalendar examples:
# OnCalendar=hourly
# OnCalendar=daily
# OnCalendar=weekly
# OnCalendar=monthly
# OnCalendar=*-*-* 02:00:00  # Daily at 2 AM
# OnCalendar=Mon *-*-* 09:00:00  # Monday 9 AM
# OnCalendar=*-01-01 00:00:00  # January 1st
```

### 3. Package Management

#### APT (Debian/Ubuntu)

```bash
# Update package lists
sudo apt update

# Upgrade packages
sudo apt upgrade
sudo apt full-upgrade  # Handle dependencies

# Install package
sudo apt install package_name

# Install specific version
sudo apt install package_name=version

# Remove package
sudo apt remove package_name

# Remove package and configuration
sudo apt purge package_name

# Remove unnecessary packages
sudo apt autoremove

# Search for packages
apt search package_name
apt-cache search package_name

# Show package information
apt show package_name

# List installed packages
apt list --installed

# Hold package at current version
sudo apt-mark hold package_name
sudo apt-mark unhold package_name

# Clean package cache
sudo apt clean
sudo apt autoclean
```

#### YUM/DNF (RHEL/CentOS/Fedora)

```bash
# Update package lists
sudo yum check-update
sudo dnf check-update

# Update all packages
sudo yum update
sudo dnf upgrade

# Install package
sudo yum install package_name
sudo dnf install package_name

# Remove package
sudo yum remove package_name
sudo dnf remove package_name

# Search packages
yum search package_name
dnf search package_name

# Package information
yum info package_name
dnf info package_name

# List installed packages
yum list installed
dnf list installed
```

#### Python Packages for ML

```bash
# Install pip packages
pip3 install package_name
pip3 install -r requirements.txt

# Install in user directory
pip3 install --user package_name

# Upgrade package
pip3 install --upgrade package_name

# Uninstall package
pip3 uninstall package_name

# List installed packages
pip3 list

# Show package details
pip3 show package_name

# Create virtual environment
python3 -m venv ml-env
source ml-env/bin/activate

# Install ML packages
pip3 install torch torchvision torchaudio
pip3 install tensorflow
pip3 install scikit-learn pandas numpy
pip3 install jupyter jupyterlab
pip3 install mlflow tensorboard

# Freeze requirements
pip3 freeze > requirements.txt
```

### 4. System Startup and Boot

#### Boot Process

```bash
# View boot messages
dmesg
dmesg | less
dmesg | grep -i error

# View boot log
journalctl -b
journalctl -b -1  # Previous boot

# Set default boot target
sudo systemctl set-default multi-user.target  # Console
sudo systemctl set-default graphical.target   # GUI

# View default target
systemctl get-default

# Temporarily switch target
sudo systemctl isolate multi-user.target

# Reboot system
sudo reboot
sudo systemctl reboot

# Power off
sudo poweroff
sudo systemctl poweroff

# View startup time
systemd-analyze
systemd-analyze blame  # Show time per service
systemd-analyze critical-chain  # Show dependency chain
```

### 5. Log Management

#### Viewing Logs with journalctl

```bash
# View all logs
journalctl

# View logs since boot
journalctl -b

# Follow logs (like tail -f)
journalctl -f

# Filter by service
journalctl -u service_name
journalctl -u docker.service
journalctl -u ml-training.service

# Filter by time
journalctl --since "2024-01-24 10:00:00"
journalctl --since "1 hour ago"
journalctl --since today
journalctl --since yesterday

# Filter by priority
journalctl -p err          # Error and above
journalctl -p warning      # Warning and above

# Combine filters
journalctl -u nginx -p err --since today

# Show kernel messages
journalctl -k

# Disk usage
journalctl --disk-usage

# Vacuum old logs
sudo journalctl --vacuum-time=7d   # Keep 7 days
sudo journalctl --vacuum-size=100M # Keep 100MB

# Export logs
journalctl -u ml-training.service -o json > logs.json
```

#### Traditional Log Files

```bash
# System logs
/var/log/syslog           # System messages
/var/log/auth.log         # Authentication logs
/var/log/kern.log         # Kernel logs
/var/log/dmesg            # Boot messages
/var/log/messages         # General messages

# Application logs
/var/log/apache2/         # Apache web server
/var/log/nginx/           # Nginx web server
/var/log/mysql/           # MySQL database

# View log files
tail -f /var/log/syslog
less /var/log/auth.log
grep "error" /var/log/syslog
```

#### Log Rotation

```bash
# Logrotate configuration
/etc/logrotate.conf
/etc/logrotate.d/

# Example: /etc/logrotate.d/ml-training
/var/log/ml-training/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ml-user ml-group
    postrotate
        systemctl reload ml-training >/dev/null 2>&1 || true
    endscript
}

# Test logrotate configuration
sudo logrotate -d /etc/logrotate.conf

# Force rotation
sudo logrotate -f /etc/logrotate.conf
```

### 6. System Monitoring

#### Resource Monitoring

```bash
# System information
uname -a                  # System info
hostname                  # Hostname
uptime                    # Uptime and load

# CPU information
lscpu
cat /proc/cpuinfo

# Memory information
free -h
cat /proc/meminfo

# Disk information
df -h
lsblk
fdisk -l

# PCI devices (GPUs)
lspci | grep -i vga
lspci | grep -i nvidia

# USB devices
lsusb

# Hardware information
sudo lshw
sudo lshw -short
sudo dmidecode
```

#### Performance Monitoring

```bash
# Real-time monitoring
top
htop
atop

# I/O monitoring
iotop
iostat

# Network monitoring
iftop
nethogs

# Process accounting
sa
lastcomm

# System activity
sar                       # System activity reporter
sar -u                    # CPU usage
sar -r                    # Memory usage
sar -b                    # I/O stats
```

### 7. User and Permission Management

#### User Management

```bash
# Add user
sudo adduser username
sudo useradd -m -s /bin/bash username

# Add user to groups
sudo usermod -aG sudo username
sudo usermod -aG docker username
sudo usermod -aG video username  # For GPU access

# Delete user
sudo userdel username
sudo userdel -r username  # Also remove home directory

# Modify user
sudo usermod -l newname oldname  # Rename
sudo usermod -s /bin/zsh username  # Change shell

# Lock/unlock user
sudo passwd -l username  # Lock
sudo passwd -u username  # Unlock

# View user info
id username
groups username
finger username
```

#### Group Management

```bash
# Create group
sudo groupadd groupname

# Delete group
sudo groupdel groupname

# Add user to group
sudo gpasswd -a username groupname

# Remove user from group
sudo gpasswd -d username groupname

# View group members
getent group groupname
```

#### sudo Configuration

```bash
# Edit sudoers file (always use visudo)
sudo visudo

# Grant sudo access
username ALL=(ALL:ALL) ALL

# Allow specific commands without password
username ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart ml-training

# Allow group
%groupname ALL=(ALL:ALL) ALL

# View sudo access
sudo -l
```

---

## Project: ML Infrastructure Automation Suite

Build a comprehensive automation suite for ML infrastructure management.

### Requirements

**Components to Create:**
1. Service management automation
2. Automated backup system with rotation
3. System health monitoring and alerting
4. Package update automation
5. Log management and rotation
6. Scheduled task orchestration

**Features:**
- Automated service recovery
- Scheduled backups with retention
- System health checks
- Automated updates with rollback
- Centralized logging
- Alert notifications

### Implementation

See `solutions/` directory for complete implementations.

---

## Practice Problems

### Problem 1: Service Manager

Create a script that:
- Manages multiple ML services
- Monitors service health
- Auto-restarts failed services
- Logs all events
- Sends alerts on failures

### Problem 2: Backup Automation

Create a system that:
- Schedules automated backups
- Rotates old backups
- Verifies backup integrity
- Manages storage space
- Provides restoration capability

### Problem 3: System Maintenance

Create a script that:
- Updates system packages
- Cleans up old files
- Rotates logs
- Monitors disk space
- Generates maintenance reports

---

## Best Practices

### 1. Service Management

```bash
# Always test services before enabling
sudo systemctl start service
# Verify it works
sudo systemctl enable service

# Use service dependencies
After=network.target
Requires=postgresql.service

# Always set restart policies
Restart=on-failure
RestartSec=10
```

### 2. Task Scheduling

```bash
# Use absolute paths in cron jobs
0 2 * * * /usr/bin/python3 /opt/scripts/backup.py

# Redirect output
0 2 * * * /opt/scripts/backup.sh >> /var/log/backup.log 2>&1

# Set environment variables
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin
```

### 3. System Updates

```bash
# Always backup before major updates
sudo apt update && sudo apt upgrade

# Test updates in staging first
# Schedule updates during maintenance windows
# Keep kernel updated for security

# Use unattended-upgrades for security updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

---

## Validation

Test your administration skills:

```bash
# Create and test a service
sudo systemctl start test-service
systemctl status test-service

# Schedule a cron job
crontab -e
# Add: */5 * * * * echo "Test" >> /tmp/cron-test.log

# Monitor logs
journalctl -f -u test-service

# Update system
sudo apt update && sudo apt list --upgradable

# Check service startup
systemd-analyze blame
```

---

## Resources

- [systemd Documentation](https://www.freedesktop.org/software/systemd/man/)
- [Cron Guide](https://crontab.guru/)
- [Linux System Administration](https://www.tldp.org/LDP/sag/html/index.html)
- [Package Management](https://www.debian.org/doc/manuals/debian-reference/ch02.html)

---

## Next Steps

1. **Module 004: ML Basics** - Dive into machine learning fundamentals
2. Automate your ML infrastructure
3. Set up monitoring and alerting
4. Implement backup strategies
5. Create runbooks for common tasks

---

**Automate everything! 🤖**

---

## Solutions: Production-Ready Scripts

The `solutions/` directory contains comprehensive, production-ready system administration scripts for ML infrastructure management.

### Script Overview

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `system_monitor.sh` | System monitoring | CPU, memory, disk, GPU monitoring; alerting; reporting |
| `user_management.sh` | User & group management | Add/delete users; SSH keys; password policies; auditing |
| `backup_automation.sh` | Backup & recovery | Full/incremental backups; rotation; verification; restore |
| `log_rotation.sh` | Log management | Rotation; compression; cleanup; analysis |
| `security_audit.sh` | Security auditing | User accounts; SSH; firewall; permissions; threats |
| `disk_manager.sh` | Disk management | Usage monitoring; large file finder; cleanup; SMART |
| `manage_services.sh` | Service management | Start/stop/restart; monitoring; auto-restart; health checks |
| `system_maintenance.sh` | System maintenance | Updates; cleanup; log rotation; reports |

### 1. System Monitor (`system_monitor.sh`)

Comprehensive system monitoring with real-time metrics and alerting.

**Features:**
- CPU, memory, and disk usage monitoring
- GPU monitoring (NVIDIA GPUs)
- Process monitoring and analysis
- Service health checks
- Network monitoring
- Alert generation
- Detailed reporting

**Usage Examples:**

```bash
# Run health check
./system_monitor.sh --check

# Continuous monitoring
./system_monitor.sh --monitor

# Generate report
./system_monitor.sh --report

# Monitor with alerts
./system_monitor.sh --monitor --alert

# Custom thresholds
./system_monitor.sh --check --threshold 90

# Verbose output
./system_monitor.sh --check --verbose
```

**Alert Configuration:**
- Default thresholds: CPU/Memory/Disk at 80%
- Customizable alert mechanisms
- Integration points for email, Slack, PagerDuty

### 2. User Management (`user_management.sh`)

Complete user and group management solution.

**Features:**
- User creation with ML-specific groups
- SSH key management
- Password policy enforcement
- User auditing and reporting
- Account locking/unlocking
- Group management

**Usage Examples:**

```bash
# Add ML engineer with docker access
./user_management.sh add-user mluser --groups docker,sudo

# Add user with custom shell
./user_management.sh add-user datauser --shell /bin/zsh

# Delete user and home directory
./user_management.sh del-user olduser --remove-home

# Manage SSH keys
./user_management.sh ssh-key mluser

# Add user to group
./user_management.sh add-to-group mluser video

# Lock account
./user_management.sh lock mluser

# Generate audit report
./user_management.sh audit

# List all users
./user_management.sh list-users

# Display user info
./user_management.sh info mluser
```

**Security Best Practices:**
- All operations logged with timestamps
- Automatic backups before user deletion
- SSH key management with validation
- Password policy enforcement
- User activity auditing

### 3. Backup Automation (`backup_automation.sh`)

Automated backup solution with versioning and verification.

**Features:**
- Full and incremental backups
- Backup rotation with retention policies
- Integrity verification with checksums
- Compression support
- Encryption support (GPG)
- Scheduled backups via cron
- Restore functionality

**Usage Examples:**

```bash
# Full backup with defaults
./backup_automation.sh backup --type full

# Incremental backup
./backup_automation.sh backup

# Backup specific directories
./backup_automation.sh backup --source /opt,/home --type full

# List available backups
./backup_automation.sh list

# Verify backup integrity
./backup_automation.sh verify backup-20240124-140000

# Restore backup
./backup_automation.sh restore backup-20240124-140000

# Restore to specific location
./backup_automation.sh restore backup-20240124-140000 /tmp/restore

# Cleanup old backups
./backup_automation.sh cleanup --keep 5

# Setup scheduled backups
./backup_automation.sh schedule

# Dry run
./backup_automation.sh backup --dry-run
```

**Backup Strategy:**
- Incremental backups save storage and time
- Automatic rotation prevents disk filling
- Checksum verification ensures integrity
- Snapshot-based incremental tracking

### 4. Log Rotation (`log_rotation.sh`)

Automated log management and rotation.

**Features:**
- Log rotation for large files
- Compression of old logs
- Automatic cleanup based on age
- Log analysis and reporting
- Systemd journal management

**Usage Examples:**

```bash
# Rotate large log files
./log_rotation.sh rotate

# Cleanup old logs
./log_rotation.sh cleanup

# Compress uncompressed logs
./log_rotation.sh compress

# Analyze log usage
./log_rotation.sh analyze

# Vacuum systemd journal
./log_rotation.sh vacuum

# Run all tasks
./log_rotation.sh all

# Verbose output
./log_rotation.sh rotate --verbose

# Dry run mode
./log_rotation.sh all --dry-run
```

**Configuration:**
- Default log rotation: files > 100MB
- Log retention: 30 days
- Journal vacuum: 30 days / 500MB
- Archive directory: `/var/log/archives`

### 5. Security Audit (`security_audit.sh`)

Comprehensive security auditing and vulnerability scanning.

**Features:**
- System update checking
- User account security analysis
- SSH configuration audit
- Firewall status verification
- File permission checking
- Suspicious process detection
- Failed login analysis
- Open port scanning
- Rootkit detection integration

**Usage Examples:**

```bash
# Perform full security audit
./security_audit.sh --audit

# Generate detailed report
./security_audit.sh --report

# Both audit and report
./security_audit.sh --audit --report

# Verbose output
./security_audit.sh --audit --verbose
```

**Security Checks:**
- UID 0 accounts (non-root)
- Accounts without passwords
- Root SSH login status
- Password authentication status
- Firewall configuration
- Critical file permissions
- World-writable files
- Failed login attempts
- Open network ports

### 6. Disk Manager (`disk_manager.sh`)

Disk usage monitoring and management.

**Features:**
- Disk usage monitoring with alerts
- Large file and directory finder
- Temporary file cleanup
- ML directory analysis
- SMART disk health monitoring
- I/O statistics
- Comprehensive reporting

**Usage Examples:**

```bash
# Check disk usage
./disk_manager.sh check

# Find largest files in /home
./disk_manager.sh large-files /home

# Find largest directories
./disk_manager.sh large-dirs /opt

# Cleanup temporary files
./disk_manager.sh cleanup

# Analyze ML directories
./disk_manager.sh analyze

# Check SMART disk health
./disk_manager.sh smart

# Show I/O statistics
./disk_manager.sh iostat

# Generate comprehensive report
./disk_manager.sh report

# Continuous monitoring
./disk_manager.sh monitor
```

**Alert Thresholds:**
- Critical: >= 80% disk usage
- Warning: >= 70% disk usage
- Large files: > 100MB

### 7. Service Manager (`manage_services.sh`)

ML infrastructure service management with health monitoring.

**Features:**
- Start/stop/restart services
- Service health monitoring
- Auto-restart failed services
- Service logs viewing
- Enable/disable services on boot
- Bulk operations

**Usage Examples:**

```bash
# Show status of all services
./manage_services.sh status

# Start a service
./manage_services.sh start docker

# Start all services
./manage_services.sh start --all

# Stop a service
./manage_services.sh stop jupyter

# Restart a service
./manage_services.sh restart mlflow

# Enable service on boot
./manage_services.sh enable docker

# View service logs
./manage_services.sh logs jupyter

# Follow logs
./manage_services.sh logs mlflow --follow

# Monitor services (auto-restart)
./manage_services.sh monitor
```

**Monitored Services:**
- docker
- nvidia-persistenced
- jupyter
- mlflow
- postgresql
- redis-server

### 8. System Maintenance (`system_maintenance.sh`)

Automated system maintenance and updates.

**Features:**
- Package updates (APT/YUM/DNF)
- System cleanup
- Log rotation
- Disk usage analysis
- Maintenance reporting

**Usage Examples:**

```bash
# Run all maintenance tasks
./system_maintenance.sh --all

# Update packages only
./system_maintenance.sh --update

# Cleanup system
./system_maintenance.sh --cleanup

# Rotate logs
./system_maintenance.sh --logs

# Analyze disk usage
./system_maintenance.sh --disk

# Generate report
./system_maintenance.sh --report maintenance.txt

# Dry run
./system_maintenance.sh --all --dry-run

# Verbose output
./system_maintenance.sh --all --verbose
```

---

## Testing

### Running Tests

The `tests/` directory contains a comprehensive test suite.

```bash
cd tests
./test_scripts.sh
```

**Test Coverage:**
- Script existence validation
- Executable permissions
- Syntax validation
- Help option functionality
- Basic command execution
- Integration tests
- Code quality checks

**Expected Output:**
```
========================================
System Administration Scripts Test Suite
========================================
Testing: system_monitor.sh
  Testing: Script exists... ✓ PASS
  Testing: Script is executable... ✓ PASS
  Testing: Script syntax is valid... ✓ PASS
  ...

Test Summary
Total: 40+ tests
Pass Rate: 100%
✓ ALL TESTS PASSED
```

---

## Automation & Scheduling

### Cron Setup

Add scripts to crontab for automated execution:

```bash
# Edit crontab
crontab -e

# Example cron jobs
# Daily system monitoring at 2 AM
0 2 * * * /path/to/system_monitor.sh --check --report >> /var/log/daily-monitor.log 2>&1

# Weekly security audit every Sunday at 3 AM
0 3 * * 0 /path/to/security_audit.sh --audit --report >> /var/log/security-audit.log 2>&1

# Daily backup at 1 AM
0 1 * * * /path/to/backup_automation.sh backup --type incremental >> /var/log/backup.log 2>&1

# Weekly cleanup every Sunday at 4 AM
0 4 * * 0 /path/to/disk_manager.sh cleanup >> /var/log/cleanup.log 2>&1

# Weekly log rotation every Sunday at 5 AM
0 5 * * 0 /path/to/log_rotation.sh all >> /var/log/log-rotation.log 2>&1

# Hourly system health check
0 * * * * /path/to/system_monitor.sh --check >> /var/log/hourly-check.log 2>&1
```

### Systemd Timers

Create systemd timers for more control:

```bash
# Create timer for daily monitoring
cat > /etc/systemd/system/daily-monitor.timer << EOF
[Unit]
Description=Daily System Monitor
Requires=daily-monitor.service

[Timer]
OnCalendar=daily
OnCalendar=02:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Create service
cat > /etc/systemd/system/daily-monitor.service << EOF
[Unit]
Description=Daily System Monitoring

[Service]
Type=oneshot
ExecStart=/path/to/system_monitor.sh --check --report
EOF

# Enable and start
sudo systemctl enable daily-monitor.timer
sudo systemctl start daily-monitor.timer
```

---

## Configuration

### Environment Variables

Scripts support configuration via environment variables:

```bash
# System Monitor
export MONITOR_CPU_THRESHOLD=85
export MONITOR_MEM_THRESHOLD=85
export MONITOR_DISK_THRESHOLD=85

# Backup Automation
export BACKUP_ROOT="/mnt/backups"
export BACKUP_KEEP=14
export BACKUP_COMPRESSION=true

# Log Rotation
export LOG_RETENTION_DAYS=60
export MAX_LOG_SIZE="500M"

# Run with custom config
./system_monitor.sh --check
```

### Configuration Files

Some scripts support configuration files:

```bash
# System monitor config
cat > /etc/sysadmin/monitor.conf << EOF
CPU_THRESHOLD=85
MEMORY_THRESHOLD=85
DISK_THRESHOLD=85
ALERT_EMAIL=admin@example.com
EOF

# Backup config
cat > /etc/sysadmin/backup.conf << EOF
BACKUP_ROOT=/mnt/backups
KEEP_BACKUPS=14
COMPRESSION=true
ENCRYPTION=false
EOF
```

---

## Troubleshooting

### Common Issues

**Permission Denied**
```bash
# Run with sudo
sudo ./script_name.sh

# Or fix permissions
chmod +x script_name.sh
```

**Log File Access**
```bash
# Create log directory
sudo mkdir -p /var/log
sudo chmod 755 /var/log

# Fix log file permissions
sudo touch /var/log/script-name.log
sudo chmod 644 /var/log/script-name.log
```

**Command Not Found**
```bash
# Add to PATH
export PATH=$PATH:/path/to/solutions

# Or use full path
/full/path/to/script_name.sh
```

**Dry Run First**
```bash
# Test without making changes
./script_name.sh --dry-run
```

### Debugging

Enable verbose output for troubleshooting:

```bash
# Verbose mode
./script_name.sh --verbose

# Bash debugging
bash -x ./script_name.sh

# Check syntax
bash -n ./script_name.sh
```

### Log Locations

```
/var/log/system-monitor.log      # System monitoring logs
/var/log/user-management.log     # User management logs
/var/log/backup-automation.log   # Backup logs
/var/log/log-rotation.log        # Log rotation logs
/var/log/security-audit.log      # Security audit logs
/var/log/disk-manager.log        # Disk management logs
/var/log/service-manager.log     # Service management logs
/var/log/system-maintenance.log  # System maintenance logs
```

---

## Best Practices

### 1. Always Use Sudo for System Operations

```bash
# Correct
sudo ./system_maintenance.sh --all

# May fail without permissions
./system_maintenance.sh --all
```

### 2. Test Before Production

```bash
# Always test with dry-run first
./backup_automation.sh backup --dry-run

# Then run actual operation
./backup_automation.sh backup
```

### 3. Monitor Logs

```bash
# Watch logs in real-time
tail -f /var/log/system-monitor.log

# Check for errors
grep ERROR /var/log/*.log
```

### 4. Regular Backups

```bash
# Schedule regular backups
0 1 * * * /path/to/backup_automation.sh backup --type incremental

# Weekly full backups
0 2 * * 0 /path/to/backup_automation.sh backup --type full
```

### 5. Security Audits

```bash
# Regular security audits
0 3 * * 0 /path/to/security_audit.sh --audit --report

# Review reports
ls -lh /var/log/security-reports/
```

### 6. Resource Monitoring

```bash
# Continuous monitoring in production
./system_monitor.sh --monitor --alert

# Or use cron for periodic checks
*/15 * * * * /path/to/system_monitor.sh --check
```

---

## Integration with ML Workflows

### GPU Monitoring

```bash
# Monitor GPU usage for ML workloads
./system_monitor.sh --check

# Alert when GPU memory high
./system_monitor.sh --monitor --alert --threshold 90
```

### User Management for ML Teams

```bash
# Add data scientist with GPU access
./user_management.sh add-user datascientist --groups docker,video,sudo

# Setup SSH keys for remote Jupyter access
./user_management.sh ssh-key datascientist
```

### Automated Model Backups

```bash
# Backup ML models and data
./backup_automation.sh backup --source /opt/ml,/data/models --type full

# Schedule daily incremental backups
0 1 * * * /path/to/backup_automation.sh backup --source /data/models
```

### Service Management for ML Infrastructure

```bash
# Start ML services
./manage_services.sh start jupyter
./manage_services.sh start mlflow

# Monitor ML services
./manage_services.sh monitor

# Auto-restart failed services
nohup ./manage_services.sh monitor &
```

---

## Production Deployment Checklist

- [ ] Review and customize all scripts for your environment
- [ ] Set up log rotation and monitoring
- [ ] Configure backup schedules
- [ ] Set up security auditing
- [ ] Configure alerting mechanisms
- [ ] Test disaster recovery procedures
- [ ] Document your customizations
- [ ] Train team members on script usage
- [ ] Set up centralized logging (optional)
- [ ] Configure monitoring dashboards (optional)

---

## Additional Resources

### Documentation
- [systemd Documentation](https://www.freedesktop.org/software/systemd/man/)
- [Cron Guide](https://crontab.guru/)
- [Bash Scripting Guide](https://www.gnu.org/software/bash/manual/)

### Tools
- `htop` - Interactive process viewer
- `iotop` - I/O monitoring
- `smartmontools` - Disk health monitoring
- `rkhunter` - Rootkit detection
- `logwatch` - Log analysis

### Related Scripts
- Prometheus exporters for metrics
- Grafana for visualization
- ELK stack for log aggregation
- Ansible playbooks for deployment automation

---

## Contributing

To improve these scripts:

1. Test thoroughly in non-production environment
2. Follow existing code style and conventions
3. Add appropriate error handling
4. Update documentation
5. Add test cases
6. Submit changes with clear descriptions

---

## License

These scripts are provided as educational material for AI infrastructure engineering.
Use at your own risk and always test in non-production environments first.

---

**Questions? Issues? Improvements?**

Check the logs, read the documentation, and test your changes!

**Happy System Administration!**
