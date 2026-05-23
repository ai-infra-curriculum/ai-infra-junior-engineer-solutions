# Module 003: Linux & Command Line

## Overview

Master Linux system administration and command-line proficiency essential for AI Infrastructure Engineering. This module covers shell scripting, system operations, networking, and infrastructure management - core skills for managing ML infrastructure.

## Learning Objectives

By completing this module, you will:

- ✅ Write effective Bash scripts for automation
- ✅ Understand Linux filesystem and permissions
- ✅ Manage processes and system resources
- ✅ Configure SSH and secure remote access
- ✅ Perform network diagnostics and troubleshooting
- ✅ Administer Linux systems for ML workloads
- ✅ Automate infrastructure tasks with shell scripts

## Module Structure

### Exercise 01: Bash Scripting (6-8 hours)
**Complexity:** ⭐⭐ Medium

Master shell scripting for infrastructure automation.

**Topics:**
- Variables and data types in Bash
- Control structures (if/else, loops)
- Functions and script organization
- Input/output and file operations
- Error handling and exit codes
- Command substitution and pipelines
- Script debugging techniques

**Project:** Infrastructure automation toolkit

---

### Exercise 02: Filesystem & Processes (6-8 hours)
**Complexity:** ⭐⭐ Medium

Deep dive into Linux filesystem, permissions, and process management.

**Topics:**
- Filesystem hierarchy standard (FHS)
- File permissions and ownership
- Links (hard and symbolic)
- Process lifecycle and states
- Process monitoring and management
- System resources (CPU, memory, disk)
- Job control and background processes

**Project:** System monitoring and cleanup scripts

---

### Exercise 03: SSH & Networking (6-8 hours)
**Complexity:** ⭐⭐⭐ Hard

Configure secure remote access and troubleshoot network issues.

**Topics:**
- SSH configuration and key management
- SSH tunneling and port forwarding
- Network interfaces and routing
- DNS and hostname resolution
- Firewall configuration (iptables/ufw)
- Network diagnostics (ping, traceroute, netstat)
- Secure file transfer (scp, rsync)

**Project:** Secure multi-server infrastructure setup

---

### Exercise 04: System Administration (8-10 hours)
**Complexity:** ⭐⭐⭐ Hard

Administer Linux systems for production ML infrastructure.

**Topics:**
- User and group management
- Package management (apt, yum)
- Service management (systemd)
- Disk management and partitioning
- Log management and rotation
- Cron jobs and scheduling
- System performance tuning
- Backup and recovery

**Project:** Production-ready ML server configuration

---

## Prerequisites

**From Previous Modules:**
- Module 001: Development environment set up
- Module 002: Python programming proficiency
- Basic terminal familiarity

**System Requirements:**
- Linux environment (Ubuntu 22.04+ recommended)
- WSL2 (if using Windows)
- macOS terminal (commands mostly compatible)
- Root/sudo access for some exercises

---

## Success Criteria

After completing this module, you should be able to:

✅ Write maintainable Bash scripts for automation
✅ Navigate and manage Linux filesystems confidently
✅ Diagnose and resolve process issues
✅ Configure secure SSH access
✅ Troubleshoot network connectivity
✅ Administer Linux servers
✅ Set up scheduled tasks and services
✅ Monitor and optimize system performance

---

## Estimated Time

**Total Module Time:** 26-34 hours

- Exercise 01: Bash Scripting (6-8 hours)
- Exercise 02: Filesystem & Processes (6-8 hours)
- Exercise 03: SSH & Networking (6-8 hours)
- Exercise 04: System Administration (8-10 hours)

---

## Key Concepts for AI Infrastructure

### 1. Automation is Critical

```bash
#!/bin/bash
# Automate ML model deployment
set -euo pipefail

MODEL_PATH="/models/classifier_v1.pkl"
DEPLOY_DIR="/opt/ml/models"

# Validate model
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Deploy with permissions
sudo cp "$MODEL_PATH" "$DEPLOY_DIR/"
sudo chown ml-user:ml-group "$DEPLOY_DIR/$(basename $MODEL_PATH)"
sudo chmod 644 "$DEPLOY_DIR/$(basename $MODEL_PATH)"

echo "Model deployed successfully"
```

### 2. Monitoring System Resources

```bash
# Monitor GPU usage for ML workloads
watch -n 1 nvidia-smi

# Check memory usage
free -h
vmstat 1 10

# Disk I/O for training jobs
iostat -x 1 10

# Process monitoring
top -u ml-user
htop
```

### 3. Secure Remote Access

```bash
# Configure SSH for secure access
cat >> ~/.ssh/config <<EOF
Host ml-server
    HostName 10.0.1.50
    User ml-admin
    Port 22
    IdentityFile ~/.ssh/ml-server-key
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF

# Set correct permissions
chmod 600 ~/.ssh/ml-server-key
chmod 644 ~/.ssh/config
```

### 4. Log Management

```bash
# View training logs
tail -f /var/log/ml/training.log

# Search for errors
grep -i error /var/log/ml/*.log

# Compress old logs
find /var/log/ml -name "*.log" -mtime +30 -exec gzip {} \;
```

---

## Essential Commands Reference

### File Operations
```bash
ls -lah              # List files with details
cd /path/to/dir      # Change directory
pwd                  # Print working directory
cp source dest       # Copy files
mv source dest       # Move/rename files
rm -rf dir/          # Remove directory recursively
mkdir -p path/to/dir # Create directory with parents
touch file.txt       # Create empty file
cat file.txt         # Display file contents
less file.txt        # Paginated file viewer
head -n 10 file      # First 10 lines
tail -f file         # Follow file (live updates)
```

### Text Processing
```bash
grep "pattern" file  # Search for pattern
grep -r "text" dir/  # Recursive search
sed 's/old/new/g'    # Replace text
awk '{print $1}'     # Print first column
cut -d',' -f1        # Cut by delimiter
sort file            # Sort lines
uniq                 # Remove duplicates
wc -l file           # Count lines
```

### Process Management
```bash
ps aux               # List all processes
top                  # Interactive process monitor
htop                 # Better process monitor
kill PID             # Terminate process
kill -9 PID          # Force kill
killall name         # Kill by name
bg                   # Background job
fg                   # Foreground job
nohup cmd &          # Run detached from terminal
```

### System Information
```bash
uname -a             # System information
df -h                # Disk space
du -sh dir/          # Directory size
free -h              # Memory usage
uptime               # System uptime
who                  # Logged in users
last                 # Login history
```

### Networking
```bash
ip addr              # Show IP addresses
ip route             # Show routing table
ping host            # Test connectivity
traceroute host      # Trace route
netstat -tulpn       # Show listening ports
ss -tulpn            # Socket statistics
curl url             # HTTP request
wget url             # Download file
```

### Permissions
```bash
chmod 755 file       # Change permissions
chmod +x script      # Make executable
chown user:group     # Change ownership
sudo command         # Run as root
```

---

## Tools & Technologies

### Command Line Tools
- **bash** - Shell scripting
- **zsh** - Advanced shell (Oh My Zsh)
- **tmux** - Terminal multiplexer
- **vim/nano** - Text editors
- **grep/sed/awk** - Text processing
- **jq** - JSON processor

### System Monitoring
- **htop** - Process monitor
- **iotop** - I/O monitor
- **nethogs** - Network monitor
- **glances** - System monitor
- **dstat** - Versatile resource stats

### Network Tools
- **ssh** - Secure shell
- **scp** - Secure copy
- **rsync** - File synchronization
- **netcat** - Network utility
- **nmap** - Network scanner

### Administration
- **systemctl** - Service management
- **journalctl** - Log viewing
- **cron** - Task scheduling
- **ufw** - Firewall
- **fail2ban** - Intrusion prevention

---

## Best Practices

### 1. Script Safety

```bash
#!/bin/bash
# Always include these at the top
set -euo pipefail

# -e: Exit on error
# -u: Exit on undefined variable
# -o pipefail: Catch errors in pipelines
```

### 2. Use Functions

```bash
#!/bin/bash

# Define functions for reusability
check_requirements() {
    local cmd="$1"
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: $cmd is not installed"
        return 1
    fi
}

deploy_model() {
    local model_path="$1"
    local deploy_dir="$2"

    check_requirements "python3" || exit 1

    # Deployment logic here
    echo "Deploying $model_path to $deploy_dir"
}

# Main
deploy_model "/models/model.pkl" "/opt/ml"
```

### 3. Error Handling

```bash
#!/bin/bash

if ! some_command; then
    echo "Command failed"
    exit 1
fi

# Or with error handling
if ! some_command 2>/tmp/error.log; then
    echo "Error occurred. See /tmp/error.log"
    cat /tmp/error.log
    exit 1
fi
```

### 4. Logging

```bash
#!/bin/bash

LOG_FILE="/var/log/deployment.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "Starting deployment"
deploy_model
log "Deployment complete"
```

---

## Common Pitfalls

### 1. Not Quoting Variables

```bash
# Bad - will break with spaces
file=$1
cat $file

# Good - always quote
file="$1"
cat "$file"
```

### 2. Not Checking Exit Codes

```bash
# Bad - continues even on error
command
do_something_else

# Good - check exit code
if ! command; then
    echo "Command failed"
    exit 1
fi
do_something_else
```

### 3. Using Root Unnecessarily

```bash
# Bad - runs everything as root
sudo bash script.sh

# Good - use sudo only when needed
#!/bin/bash
normal_command
sudo privileged_command
normal_command_2
```

---

## Testing Your Skills

### Self-Assessment Checklist

After each exercise, verify you can:

- [ ] Navigate the filesystem efficiently
- [ ] Create and edit files with vim/nano
- [ ] Write and execute shell scripts
- [ ] Understand and modify permissions
- [ ] Monitor and manage processes
- [ ] Configure SSH access
- [ ] Diagnose network issues
- [ ] Schedule automated tasks
- [ ] Read and interpret logs

---

## Resources

### Official Documentation
- [GNU Bash Manual](https://www.gnu.org/software/bash/manual/)
- [Linux Documentation Project](https://tldp.org/)
- [Ubuntu Server Guide](https://ubuntu.com/server/docs)
- [Red Hat Enterprise Linux Docs](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/)

### Books
- "The Linux Command Line" by William Shotts
- "Linux Administration Handbook" by Evi Nemeth et al.
- "Unix and Linux System Administration Handbook" by Nemeth et al.
- "Bash Cookbook" by Carl Albing & JP Vossen

### Online Resources
- [ExplainShell](https://explainshell.com/) - Explain shell commands
- [ShellCheck](https://www.shellcheck.net/) - Shell script linter
- [Linux Journey](https://linuxjourney.com/) - Interactive tutorials
- [OverTheWire Bandit](https://overthewire.org/wargames/bandit/) - Command line game

### Practice
- [Linux Survival](https://linuxsurvival.com/) - Interactive tutorial
- [Terminus](https://web.mit.edu/mprat/Public/web/Terminus/Web/main.html) - Terminal game
- [Command Line Challenge](https://cmdchallenge.com/) - Bash challenges

---

## Module Projects

### Project 1: Infrastructure Automation Toolkit (Exercise 01)
Build a collection of Bash scripts for common infrastructure tasks.

**Skills:** Shell scripting, automation, error handling

### Project 2: System Health Monitor (Exercise 02)
Create a monitoring system that tracks system resources and alerts on issues.

**Skills:** Process management, resource monitoring, alerting

### Project 3: Secure Multi-Server Setup (Exercise 03)
Configure secure SSH access across multiple servers with jump hosts.

**Skills:** SSH configuration, key management, networking

### Project 4: ML Server Configuration (Exercise 04)
Set up a production-ready Linux server for ML workloads.

**Skills:** System administration, service management, security hardening

---

## Getting Started

### 1. Verify Linux Environment

```bash
# Check OS version
cat /etc/os-release

# Check kernel version
uname -r

# Check shell
echo $SHELL
```

### 2. Install Essential Tools

```bash
# Update package list
sudo apt update

# Install tools
sudo apt install -y \
    vim \
    tmux \
    htop \
    net-tools \
    jq \
    curl \
    wget \
    git

# Verify installations
vim --version
tmux -V
htop --version
```

### 3. Start with Exercise 01

```bash
cd exercise-01-bash-scripting
cat README.md
```

---

## Next Steps

After completing this module:

1. **Module 004: ML Basics** - Understand machine learning fundamentals
2. Practice Linux administration daily
3. Set up your own Linux server (local VM or cloud)
4. Contribute to DevOps open-source projects
5. Obtain Linux certifications (LPIC-1, RHCSA)

---

## Module Completion Checklist

- [ ] Exercise 01: Bash Scripting completed
- [ ] Exercise 02: Filesystem & Processes completed
- [ ] Exercise 03: SSH & Networking completed
- [ ] Exercise 04: System Administration completed
- [ ] All projects functional and documented
- [ ] Can confidently administer Linux systems
- [ ] Ready for Module 004

---

**Master the command line and become a Linux power user! 🐧💻**
