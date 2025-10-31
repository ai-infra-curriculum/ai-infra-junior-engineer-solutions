# Step-by-Step Implementation Guide: System Administration for ML Infrastructure

## Overview

Master Linux system administration by building comprehensive automation tools for managing ML infrastructure. Learn to manage system services, automate backups, perform security audits, handle user management, and maintain system health for production ML environments.

**Time**: 5-6 hours | **Difficulty**: Intermediate to Advanced

---

## Prerequisites

```bash
# Verify system tools
which systemctl journalctl cron useradd usermod

# Install required packages
sudo apt update
sudo apt install -y \
    systemd \
    cron \
    logrotate \
    fail2ban \
    auditd \
    rsyslog \
    sudo \
    acl

# Verify sudo access
sudo -v

# Check systemd version
systemctl --version
```

---

## Learning Objectives

By completing this exercise, you will be able to:

✅ Manage systemd services for ML applications
✅ Automate system backups and maintenance tasks
✅ Implement security audits and hardening
✅ Manage users and permissions securely
✅ Configure log rotation and retention
✅ Monitor system health and performance
✅ Create automated maintenance schedules
✅ Handle system emergencies and recovery

---

## Phase 1: Service Management Script (90 minutes)

### Step 1: Understanding systemd

Systemd is the modern init system and service manager for Linux:

```bash
# View all services
systemctl list-units --type=service

# Check service status
systemctl status nginx

# Start/stop/restart service
sudo systemctl start myservice
sudo systemctl stop myservice
sudo systemctl restart myservice

# Enable/disable service (auto-start on boot)
sudo systemctl enable myservice
sudo systemctl disable myservice

# View service logs
journalctl -u myservice -f  # Follow logs
journalctl -u myservice --since "1 hour ago"
```

### Step 2: Create manage_services.sh

Start with the script structure:

```bash
#!/bin/bash
#
# manage_services.sh - Manage systemd services for ML infrastructure
#
# Description:
#   Manage, monitor, and control ML-related systemd services
#   Includes health checks, auto-restart, and dependency management
#
# Usage:
#   ./manage_services.sh [OPTIONS] COMMAND [SERVICE]
#
# Commands:
#   start SERVICE       Start a service
#   stop SERVICE        Stop a service
#   restart SERVICE     Restart a service
#   status SERVICE      Show service status
#   logs SERVICE        Show service logs
#   enable SERVICE      Enable service on boot
#   disable SERVICE     Disable service on boot
#   list                List all ML services
#   health              Check health of all services
#   create NAME         Create new service file
#
# Options:
#   -f, --follow        Follow logs in real-time
#   -n, --lines NUM     Number of log lines (default: 50)
#   -v, --verbose       Verbose output
#   -h, --help          Show help
#

set -euo pipefail

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly SERVICE_DIR="/etc/systemd/system"
readonly ML_SERVICES=(
    "ml-api"
    "ml-training"
    "mlflow-server"
    "tensorboard"
    "jupyter"
)

FOLLOW_LOGS=false
LOG_LINES=50
VERBOSE=false

# Logging
log() {
    local level="$1"
    shift
    echo "[$level] $*" >&2
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        log "DEBUG" "$@"
    fi
}
```

### Step 3: Implement Service Control Functions

Add service management:

```bash
check_service_exists() {
    local service="$1"

    if systemctl list-unit-files | grep -q "^${service}.service"; then
        return 0
    else
        log "ERROR" "Service $service not found"
        return 1
    fi
}

start_service() {
    local service="$1"

    check_service_exists "$service" || return 1

    log "INFO" "Starting service: $service"

    if sudo systemctl start "$service"; then
        log "SUCCESS" "Service $service started"
        sleep 2
        show_service_status "$service"
        return 0
    else
        log "ERROR" "Failed to start $service"
        show_service_logs "$service" 20
        return 1
    fi
}

stop_service() {
    local service="$1"

    check_service_exists "$service" || return 1

    log "INFO" "Stopping service: $service"

    if sudo systemctl stop "$service"; then
        log "SUCCESS" "Service $service stopped"
        return 0
    else
        log "ERROR" "Failed to stop $service"
        return 1
    fi
}

restart_service() {
    local service="$1"

    check_service_exists "$service" || return 1

    log "INFO" "Restarting service: $service"

    if sudo systemctl restart "$service"; then
        log "SUCCESS" "Service $service restarted"
        sleep 2
        show_service_status "$service"
        return 0
    else
        log "ERROR" "Failed to restart $service"
        show_service_logs "$service" 20
        return 1
    fi
}

enable_service() {
    local service="$1"

    check_service_exists "$service" || return 1

    log "INFO" "Enabling service: $service (auto-start on boot)"

    if sudo systemctl enable "$service"; then
        log "SUCCESS" "Service $service enabled"
        return 0
    else
        log "ERROR" "Failed to enable $service"
        return 1
    fi
}

disable_service() {
    local service="$1"

    check_service_exists "$service" || return 1

    log "INFO" "Disabling service: $service"

    if sudo systemctl disable "$service"; then
        log "SUCCESS" "Service $service disabled"
        return 0
    else
        log "ERROR" "Failed to disable $service"
        return 1
    fi
}
```

### Step 4: Implement Status and Logging Functions

Add monitoring capabilities:

```bash
show_service_status() {
    local service="$1"

    if systemctl is-active "$service" &>/dev/null; then
        local status="✓ ACTIVE"
        local color="\033[32m"  # Green
    else
        local status="✗ INACTIVE"
        local color="\033[31m"  # Red
    fi

    local enabled="disabled"
    if systemctl is-enabled "$service" &>/dev/null; then
        enabled="enabled"
    fi

    echo -e "${color}${status}\033[0m - $service ($enabled)"

    if [[ "$VERBOSE" == true ]]; then
        echo ""
        systemctl status "$service" --no-pager
    fi
}

show_service_logs() {
    local service="$1"
    local lines="${2:-$LOG_LINES}"

    log "INFO" "Showing logs for $service (last $lines lines)"

    if [[ "$FOLLOW_LOGS" == true ]]; then
        sudo journalctl -u "$service" -n "$lines" -f
    else
        sudo journalctl -u "$service" -n "$lines" --no-pager
    fi
}

list_all_services() {
    log "INFO" "ML Infrastructure Services"
    log "INFO" "========================================"

    for service in "${ML_SERVICES[@]}"; do
        if systemctl list-unit-files | grep -q "^${service}.service"; then
            show_service_status "$service"
        else
            echo "○ NOT INSTALLED - $service"
        fi
    done
}

check_all_health() {
    log "INFO" "Health Check for ML Services"
    log "INFO" "========================================"

    local failed_count=0

    for service in "${ML_SERVICES[@]}"; do
        if systemctl list-unit-files | grep -q "^${service}.service"; then
            if systemctl is-active "$service" &>/dev/null; then
                log "SUCCESS" "$service is running"
            else
                log "ERROR" "$service is not running"
                ((failed_count++))

                # Attempt restart
                log "INFO" "Attempting to restart $service..."
                if sudo systemctl restart "$service"; then
                    log "SUCCESS" "$service restarted successfully"
                else
                    log "ERROR" "Failed to restart $service"
                fi
            fi
        fi
    done

    echo ""
    if [[ $failed_count -eq 0 ]]; then
        log "SUCCESS" "All services are healthy"
    else
        log "WARNING" "$failed_count service(s) had issues"
    fi

    return $failed_count
}
```

### Step 5: Implement Service Creation

Add template for new services:

```bash
create_service_file() {
    local service_name="$1"
    local exec_start="${2:-/usr/local/bin/$service_name}"
    local user="${3:-ml-user}"
    local description="${4:-ML Infrastructure Service}"

    local service_file="$SERVICE_DIR/${service_name}.service"

    log "INFO" "Creating service file: $service_file"

    sudo tee "$service_file" > /dev/null <<EOF
[Unit]
Description=$description
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$user
Group=$user
WorkingDirectory=/opt/ml/$service_name
ExecStart=$exec_start
Restart=on-failure
RestartSec=10s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$service_name

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Environment
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=-/etc/default/$service_name

[Install]
WantedBy=multi-user.target
EOF

    if [[ $? -eq 0 ]]; then
        log "SUCCESS" "Service file created"

        # Reload systemd
        log "INFO" "Reloading systemd..."
        sudo systemctl daemon-reload

        log "INFO" "To start the service, run:"
        echo "  sudo systemctl start $service_name"
        echo "  sudo systemctl enable $service_name"

        return 0
    else
        log "ERROR" "Failed to create service file"
        return 1
    fi
}
```

---

## Phase 2: Backup Automation Script (90 minutes)

### Summary

The `backup_automation.sh` script creates comprehensive automated backup system.

**Features**:
- Scheduled backups via cron
- Multiple backup types (full, incremental, differential)
- Backup to local and remote destinations
- Verification and integrity checks
- Automated cleanup of old backups

**Core Implementation**:

```bash
create_backup_job() {
    local source_dir="$1"
    local backup_dir="$2"
    local schedule="$3"  # cron format

    # Create backup script
    local backup_script="/usr/local/bin/backup-$(basename "$source_dir").sh"

    cat > "$backup_script" <<'SCRIPT_EOF'
#!/bin/bash
set -euo pipefail

SOURCE_DIR="'"$source_dir"'"
BACKUP_DIR="'"$backup_dir"'"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="backup-${TIMESTAMP}.tar.gz"

# Create backup
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")"

# Create checksum
sha256sum "${BACKUP_DIR}/${BACKUP_NAME}" > "${BACKUP_DIR}/${BACKUP_NAME}.sha256"

# Log
logger -t backup "Backup created: ${BACKUP_NAME}"

# Cleanup old backups (keep last 30 days)
find "$BACKUP_DIR" -name "backup-*.tar.gz" -mtime +30 -delete
SCRIPT_EOF

    chmod +x "$backup_script"

    # Add to crontab
    (crontab -l 2>/dev/null; echo "$schedule $backup_script") | crontab -

    log "SUCCESS" "Backup job created: $backup_script"
}

verify_backup() {
    local backup_file="$1"
    local checksum_file="${backup_file}.sha256"

    if [[ ! -f "$checksum_file" ]]; then
        log "ERROR" "Checksum file not found"
        return 1
    fi

    local expected=$(cat "$checksum_file" | awk '{print $1}')
    local actual=$(sha256sum "$backup_file" | awk '{print $1}')

    if [[ "$expected" == "$actual" ]]; then
        log "SUCCESS" "Backup integrity verified"
        return 0
    else
        log "ERROR" "Backup integrity check failed!"
        return 1
    fi
}
```

---

## Phase 3: Security Audit Script (60 minutes)

### Summary

The `security_audit.sh` script performs comprehensive security checks.

**Audit Areas**:
- File permissions
- User accounts and passwords
- Sudo configuration
- Open ports
- Running processes
- System updates
- Firewall rules

**Implementation**:

```bash
audit_file_permissions() {
    log "INFO" "Auditing critical file permissions..."

    # Check /etc/passwd
    local passwd_perms=$(stat -c %a /etc/passwd)
    if [[ "$passwd_perms" != "644" ]]; then
        log "WARNING" "/etc/passwd has incorrect permissions: $passwd_perms (should be 644)"
    fi

    # Check /etc/shadow
    local shadow_perms=$(stat -c %a /etc/shadow)
    if [[ "$shadow_perms" != "640" ]] && [[ "$shadow_perms" != "000" ]]; then
        log "WARNING" "/etc/shadow has incorrect permissions: $shadow_perms (should be 640 or 000)"
    fi

    # Check for world-writable files
    log "INFO" "Searching for world-writable files..."
    find / -type f -perm -002 ! -path "/proc/*" ! -path "/sys/*" 2>/dev/null | head -20
}

audit_user_accounts() {
    log "INFO" "Auditing user accounts..."

    # Check for users without passwords
    awk -F: '($2 == "") {print $1}' /etc/shadow

    # Check for UID 0 users (should only be root)
    awk -F: '($3 == 0) {print $1}' /etc/passwd

    # List users with login shells
    awk -F: '($7 != "/usr/sbin/nologin" && $7 != "/bin/false") {print $1, $7}' /etc/passwd
}

check_system_updates() {
    log "INFO" "Checking for system updates..."

    if command -v apt &> /dev/null; then
        sudo apt update &> /dev/null
        local updates=$(apt list --upgradable 2>/dev/null | wc -l)
        log "INFO" "$updates packages can be updated"
    fi
}
```

---

## Phase 4: User Management Script (60 minutes)

### Summary

The `user_management.sh` script manages users and groups for ML infrastructure.

**Features**:
- Create ML users with proper permissions
- Manage SSH access
- Set up user quotas
- Configure user environments
- Audit user activities

**Implementation**:

```bash
create_ml_user() {
    local username="$1"
    local groups="${2:-ml-users}"

    log "INFO" "Creating ML user: $username"

    # Create user
    sudo useradd -m -s /bin/bash -G "$groups" "$username"

    # Set up SSH directory
    sudo mkdir -p "/home/$username/.ssh"
    sudo chmod 700 "/home/$username/.ssh"

    # Create default directories
    sudo mkdir -p "/home/$username"/{experiments,datasets,models,notebooks}

    # Set ownership
    sudo chown -R "$username:$username" "/home/$username"

    # Configure bashrc
    sudo tee -a "/home/$username/.bashrc" > /dev/null <<'EOF'
# ML Environment
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/ml/lib:$PYTHONPATH"

# Aliases
alias gpu='nvidia-smi'
alias jup='jupyter lab --no-browser --port=8888'
EOF

    log "SUCCESS" "User $username created"
}

add_ssh_key() {
    local username="$1"
    local public_key="$2"

    local auth_keys="/home/$username/.ssh/authorized_keys"

    sudo mkdir -p "/home/$username/.ssh"
    echo "$public_key" | sudo tee -a "$auth_keys" > /dev/null
    sudo chmod 600 "$auth_keys"
    sudo chown "$username:$username" "$auth_keys"

    log "SUCCESS" "SSH key added for $username"
}
```

---

## Phase 5: System Maintenance Script (60 minutes)

### Summary

The `system_maintenance.sh` script automates routine maintenance tasks.

**Tasks**:
- Clean package cache
- Remove old kernels
- Clear temporary files
- Optimize disk usage
- Update locate database
- Rotate logs

**Implementation**:

```bash
clean_package_cache() {
    log "INFO" "Cleaning package cache..."

    if command -v apt &> /dev/null; then
        sudo apt clean
        sudo apt autoclean
        sudo apt autoremove -y
    fi
}

remove_old_kernels() {
    log "INFO" "Removing old kernels..."

    local current_kernel=$(uname -r)
    log "INFO" "Current kernel: $current_kernel"

    if command -v dpkg &> /dev/null; then
        # List installed kernels
        dpkg -l | grep linux-image | grep -v "$current_kernel"

        # Remove old kernels (keeping current)
        sudo apt autoremove --purge -y
    fi
}

clean_temp_files() {
    log "INFO" "Cleaning temporary files..."

    # Clear /tmp (files older than 7 days)
    sudo find /tmp -type f -atime +7 -delete

    # Clear /var/tmp
    sudo find /var/tmp -type f -atime +30 -delete

    # Clear user cache
    find ~/.cache -type f -atime +30 -delete 2>/dev/null || true
}

optimize_disk() {
    log "INFO" "Optimizing disk usage..."

    # Find and remove duplicate files
    # Analyze large files
    # Clear old logs
}
```

---

## Phase 6: Log Rotation Configuration (30 minutes)

### Summary

Configure `logrotate` for ML application logs.

**Configuration**:

```bash
create_logrotate_config() {
    local log_file="$1"
    local config_name=$(basename "$log_file" .log)

    sudo tee "/etc/logrotate.d/$config_name" > /dev/null <<EOF
$log_file {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ml-user ml-user
    sharedscripts
    postrotate
        systemctl reload $config_name 2>/dev/null || true
    endscript
}
EOF

    log "SUCCESS" "Logrotate config created for $log_file"
}
```

---

## Testing & Validation

```bash
# Test service management
./solutions/manage_services.sh list
./solutions/manage_services.sh health

# Create test service
./solutions/manage_services.sh create test-ml-service

# Test backup automation
./solutions/backup_automation.sh create /data /backup "0 2 * * *"

# Run security audit
sudo ./solutions/security_audit.sh

# Test user management
sudo ./solutions/user_management.sh create testuser ml-users

# Run system maintenance
sudo ./solutions/system_maintenance.sh

# Verify log rotation
sudo logrotate -d /etc/logrotate.d/ml-api
```

---

## Best Practices

1. **Use systemd for services**: Modern, reliable, with built-in logging
2. **Automate with cron**: Schedule regular maintenance tasks
3. **Regular security audits**: Weekly scans for vulnerabilities
4. **Backup verification**: Always verify backup integrity
5. **Least privilege**: Users only have necessary permissions
6. **Monitor everything**: Services, disk, logs, users
7. **Document changes**: Keep change logs for auditing

---

## Next Steps

1. Integrate with monitoring tools (Prometheus, Nagios)
2. Implement centralized logging (ELK stack)
3. Add automated alerting
4. Create disaster recovery procedures
5. Implement configuration management (Ansible, Puppet)

---

## Resources

- [systemd Manual](https://www.freedesktop.org/software/systemd/man/)
- [Linux System Administration](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/configuring_basic_system_settings/)
- [Security Hardening Guide](https://www.cisecurity.org/cis-benchmarks/)
- [Backup Best Practices](https://www.backblaze.com/blog/the-3-2-1-backup-strategy/)

---

**Congratulations!** You've mastered Linux system administration for ML infrastructure! 🎉
