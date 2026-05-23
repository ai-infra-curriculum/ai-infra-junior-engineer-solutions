# Exercise 02: Filesystem & Processes

## Overview

Master Linux filesystem operations and process management for ML infrastructure. Learn to navigate the filesystem, manage processes, monitor resources, and understand system internals critical for running AI/ML workloads.

## Learning Objectives

- ✅ Understand Linux filesystem hierarchy and structure
- ✅ Navigate and manipulate files and directories efficiently
- ✅ Master file permissions and ownership
- ✅ Manage processes and system resources
- ✅ Monitor system performance
- ✅ Work with the /proc filesystem
- ✅ Handle disk space and storage
- ✅ Use system utilities for ML infrastructure management

## Topics Covered

### 1. Linux Filesystem Hierarchy

#### Standard Directory Structure

```
/                    Root directory
├── bin/            Essential user binaries
├── boot/           Boot loader files
├── dev/            Device files
├── etc/            System configuration
├── home/           User home directories
├── lib/            Shared libraries
├── media/          Removable media mount points
├── mnt/            Temporary mount points
├── opt/            Optional software
├── proc/           Process information (virtual)
├── root/           Root user home
├── run/            Runtime data
├── sbin/           System binaries
├── srv/            Service data
├── sys/            System information (virtual)
├── tmp/            Temporary files
├── usr/            User programs and data
│   ├── bin/        User binaries
│   ├── lib/        User libraries
│   ├── local/      Locally installed software
│   └── share/      Shared data
└── var/            Variable data
    ├── log/        Log files
    ├── cache/      Application cache
    ├── lib/        State information
    └── tmp/        Temporary files (preserved across reboots)
```

#### Key Directories for ML Infrastructure

```bash
# Model storage
/opt/ml/models/              # Production models
/data/training/              # Training datasets
/data/validation/            # Validation datasets

# Logs and monitoring
/var/log/ml-api/            # API service logs
/var/log/training/          # Training logs
/var/log/inference/         # Inference logs

# Configuration
/etc/ml-config/             # ML service configuration
/etc/systemd/system/        # Service definitions

# Temporary data
/tmp/ml-scratch/            # Temporary processing
/var/cache/ml/              # Cached data

# User data
/home/ml-user/experiments/  # User experiments
/home/ml-user/notebooks/    # Jupyter notebooks
```

### 2. File Operations

#### Basic Navigation

```bash
# Print working directory
pwd

# List files
ls                  # Basic list
ls -l               # Long format with details
ls -la              # Include hidden files
ls -lh              # Human-readable sizes
ls -lS              # Sort by size
ls -lt              # Sort by modification time
ls -ltr             # Reverse time order (oldest first)

# Change directory
cd /path/to/dir     # Absolute path
cd ../              # Parent directory
cd ~                # Home directory
cd -                # Previous directory

# Create directory
mkdir new_dir                    # Single directory
mkdir -p path/to/nested/dir      # Create parent directories
mkdir -m 755 secure_dir          # With permissions

# Remove directory
rmdir empty_dir                  # Remove empty directory
rm -r dir                        # Remove directory and contents
rm -rf dir                       # Force remove (be careful!)
```

#### File Manipulation

```bash
# Copy files
cp source.txt dest.txt                    # Copy file
cp -r source_dir/ dest_dir/               # Copy directory recursively
cp -p file1 file2                         # Preserve attributes
cp -u source dest                         # Update (copy if newer)
cp -v source dest                         # Verbose output

# Move/Rename files
mv old_name.txt new_name.txt              # Rename
mv file.txt /new/location/                # Move
mv -i file.txt dest/                      # Interactive (prompt before overwrite)
mv -n file.txt dest/                      # No overwrite

# Remove files
rm file.txt                               # Remove file
rm -i file.txt                            # Interactive removal
rm -f file.txt                            # Force remove
rm *.log                                  # Remove all .log files

# Create empty file or update timestamp
touch file.txt                            # Create or update timestamp
touch -t 202401241200 file.txt            # Set specific time

# Create symbolic links
ln -s /path/to/target link_name           # Symbolic link
ln target_file hard_link                  # Hard link
```

#### Viewing File Contents

```bash
# Display entire file
cat file.txt                              # Print to stdout
cat file1.txt file2.txt > combined.txt    # Concatenate files

# View with pagination
less file.txt                             # Scroll through file
more file.txt                             # Simpler pager

# View beginning/end
head file.txt                             # First 10 lines
head -n 20 file.txt                       # First 20 lines
tail file.txt                             # Last 10 lines
tail -n 50 file.txt                       # Last 50 lines
tail -f logfile.log                       # Follow (watch for new content)

# Search in files
grep "pattern" file.txt                   # Search for pattern
grep -r "error" logs/                     # Recursive search
grep -i "Error" file.txt                  # Case-insensitive
grep -v "debug" file.txt                  # Invert match (exclude)
grep -n "error" file.txt                  # Show line numbers
grep -c "error" file.txt                  # Count matches
```

#### File Information

```bash
# File statistics
stat file.txt                             # Detailed file information
file image.png                            # Determine file type

# Disk usage
du -h file.txt                            # Human-readable size
du -sh directory/                         # Directory size summary
du -h --max-depth=1 /data/                # First-level subdirectories

# Disk space
df -h                                     # Filesystem usage
df -h /data                               # Specific filesystem
df -i                                     # Inode usage

# Find files
find /path -name "*.log"                  # Find by name
find /path -type f                        # Find files only
find /path -type d                        # Find directories only
find /path -size +100M                    # Files larger than 100MB
find /path -mtime -7                      # Modified in last 7 days
find /path -empty                         # Empty files/directories
find /path -name "*.tmp" -delete          # Find and delete

# Locate files (uses database, faster)
locate filename                           # Quick file search
updatedb                                  # Update locate database
```

### 3. File Permissions

#### Understanding Permissions

```
-rwxr-xr-x  1 user group 4096 Jan 24 12:00 script.sh
│││││││││││
│││└┴┴┴┴┴┴┴── Permissions (owner, group, others)
││└────────── Number of hard links
│└─────────── Owner
└──────────── File type (- = file, d = directory, l = link)

Permission bits:
r (read)    = 4
w (write)   = 2
x (execute) = 1

rwx = 4 + 2 + 1 = 7
r-x = 4 + 0 + 1 = 5
r-- = 4 + 0 + 0 = 4
```

#### Changing Permissions

```bash
# Symbolic mode
chmod u+x script.sh              # Add execute for user
chmod g+w file.txt               # Add write for group
chmod o-r file.txt               # Remove read for others
chmod a+r file.txt               # Add read for all
chmod u=rwx,g=rx,o=r file.txt    # Set explicit permissions

# Numeric mode
chmod 755 script.sh              # rwxr-xr-x
chmod 644 file.txt               # rw-r--r--
chmod 600 secrets.txt            # rw-------
chmod 700 private_dir/           # rwx------

# Recursive
chmod -R 755 directory/          # Apply to all files/subdirs

# Special permissions
chmod u+s binary                 # Set SUID
chmod g+s directory/             # Set SGID
chmod +t /tmp/                   # Set sticky bit
```

#### Changing Ownership

```bash
# Change owner
chown user file.txt                       # Change owner
chown user:group file.txt                 # Change owner and group
chown -R user:group directory/            # Recursive

# Change group only
chgrp group file.txt                      # Change group
chgrp -R group directory/                 # Recursive
```

#### Access Control Lists (ACLs)

```bash
# View ACLs
getfacl file.txt

# Set ACLs
setfacl -m u:username:rw file.txt         # Grant user permissions
setfacl -m g:groupname:r file.txt         # Grant group permissions
setfacl -x u:username file.txt            # Remove user permissions
setfacl -b file.txt                       # Remove all ACLs
setfacl -R -m u:mluser:rwx /data/models   # Recursive ACL
```

### 4. Process Management

#### Viewing Processes

```bash
# Process snapshot
ps                               # Current shell processes
ps aux                           # All processes (BSD style)
ps -ef                           # All processes (Unix style)
ps -u username                   # User's processes
ps -p 1234                       # Specific process ID

# Process tree
pstree                           # Process hierarchy
pstree -p                        # Show PIDs

# Real-time monitoring
top                              # Interactive process viewer
htop                             # Enhanced top (if available)

# Top shortcuts:
# P - Sort by CPU
# M - Sort by memory
# k - Kill process
# r - Renice process
# q - Quit
```

#### Process Details

```bash
# Process information
ps -o pid,ppid,cmd,pcpu,pmem 1234        # Custom columns
ps -p 1234 -o etime                      # Process running time

# All threads
ps -eLf                                   # All threads
ps -T -p 1234                             # Threads for specific process
```

#### Managing Processes

```bash
# Starting processes
command &                        # Run in background
nohup command &                  # Immune to hangups
nohup command > output.log 2>&1 &  # Redirect output

# Job control
jobs                             # List background jobs
fg %1                            # Bring job to foreground
bg %1                            # Continue job in background
Ctrl+Z                           # Suspend current process
Ctrl+C                           # Kill current process

# Stopping processes
kill PID                         # Terminate process (SIGTERM)
kill -9 PID                      # Force kill (SIGKILL)
kill -15 PID                     # Graceful shutdown (SIGTERM)
killall process_name             # Kill by name
pkill pattern                    # Kill by pattern

# Process priority
nice -n 10 command               # Start with priority
renice -n 5 -p PID               # Change priority
renice -n -5 -u username         # Change user's processes

# Priority levels:
# -20 (highest) to 19 (lowest)
# Default: 0
```

#### Process Signals

```bash
# Common signals
kill -l                          # List all signals

SIGHUP   (1)  - Hangup
SIGINT   (2)  - Interrupt (Ctrl+C)
SIGQUIT  (3)  - Quit
SIGKILL  (9)  - Kill (cannot be caught)
SIGTERM (15)  - Terminate (default)
SIGSTOP (19)  - Stop (cannot be caught)
SIGCONT (18)  - Continue

# Sending signals
kill -SIGHUP PID
kill -1 PID                      # Same as SIGHUP
```

### 5. System Monitoring

#### CPU and Load

```bash
# CPU information
lscpu                            # CPU details
cat /proc/cpuinfo                # Detailed CPU info

# System load
uptime                           # System uptime and load average
w                                # Who is logged in and load
cat /proc/loadavg                # Load averages

# Load average interpretation:
# Three numbers: 1-min, 5-min, 15-min averages
# Compare to number of CPUs:
#   < # of CPUs: System not fully utilized
#   = # of CPUs: Fully loaded
#   > # of CPUs: Processes waiting for CPU
```

#### Memory

```bash
# Memory usage
free                             # Basic memory info
free -h                          # Human-readable
free -m                          # In megabytes
free -s 5                        # Update every 5 seconds

# Detailed memory
cat /proc/meminfo                # Detailed memory statistics
vmstat                           # Virtual memory statistics
vmstat 1 10                      # Update every second, 10 times

# Memory by process
ps aux --sort=-%mem | head       # Top memory consumers
```

#### Disk I/O

```bash
# Disk statistics
iostat                           # CPU and I/O statistics
iostat -x                        # Extended statistics
iostat -x 1 5                    # Every second, 5 times

# I/O monitoring
iotop                            # Interactive I/O monitor
iotop -o                         # Show only active I/O

# Disk activity
dstat                            # Versatile resource statistics
```

#### Network

```bash
# Network interfaces
ifconfig                         # Interface configuration
ip addr                          # Modern alternative
ip link                          # Link layer info

# Network statistics
netstat -tuln                    # Listening ports
netstat -tupln                   # With process names (needs root)
ss -tuln                         # Modern alternative

# Network activity
iftop                            # Network bandwidth monitor
nethogs                          # Network usage by process
```

### 6. The /proc Filesystem

#### Overview

The `/proc` filesystem is a virtual filesystem providing process and system information.

```bash
# Process directories
/proc/[PID]/                     # Per-process information
/proc/[PID]/cmdline              # Command line
/proc/[PID]/cwd                  # Current working directory (symlink)
/proc/[PID]/environ              # Environment variables
/proc/[PID]/exe                  # Executable (symlink)
/proc/[PID]/fd/                  # File descriptors
/proc/[PID]/maps                 # Memory mappings
/proc/[PID]/status               # Process status
/proc/[PID]/stat                 # Process statistics

# System information
/proc/cpuinfo                    # CPU information
/proc/meminfo                    # Memory information
/proc/loadavg                    # Load averages
/proc/uptime                     # System uptime
/proc/version                    # Kernel version
/proc/filesystems                # Supported filesystems
/proc/mounts                     # Mounted filesystems
/proc/net/                       # Network information
```

#### Practical Examples

```bash
# View process command line
cat /proc/1234/cmdline | tr '\0' ' '

# Check process environment
cat /proc/1234/environ | tr '\0' '\n'

# View process status
cat /proc/1234/status

# See open files
ls -l /proc/1234/fd

# Memory maps
cat /proc/1234/maps

# System uptime
cat /proc/uptime
# First number: uptime in seconds
# Second number: idle time in seconds

# CPU count
grep -c processor /proc/cpuinfo

# Total memory
grep MemTotal /proc/meminfo
```

### 7. System Resources for ML Workloads

#### GPU Monitoring

```bash
# NVIDIA GPUs
nvidia-smi                                # GPU status
nvidia-smi -l 1                           # Update every second
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
nvidia-smi dmon                           # Device monitoring

# GPU processes
nvidia-smi pmon                           # Process monitoring
fuser -v /dev/nvidia*                     # Processes using GPU
```

#### Memory Management

```bash
# Check memory pressure
cat /proc/pressure/memory                 # Memory pressure (if available)

# OOM (Out of Memory) killer
dmesg | grep -i "out of memory"           # OOM events
cat /proc/sys/vm/oom_kill_allocating_task

# Memory limits for processes (cgroups)
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
cat /sys/fs/cgroup/memory/memory.usage_in_bytes

# Clear caches (careful!)
sync                                      # Flush filesystem buffers
echo 3 > /proc/sys/vm/drop_caches         # Clear caches (requires root)
```

#### CPU Affinity

```bash
# Set CPU affinity
taskset -c 0,1 python train.py            # Run on CPUs 0 and 1
taskset -p 0x3 1234                       # Set affinity for PID 1234

# Check CPU affinity
taskset -p 1234                           # Show affinity for process
```

#### I/O Priority

```bash
# Set I/O priority
ionice -c 2 -n 7 python process_data.py   # Best-effort, priority 7
ionice -c 3 backup.sh                     # Idle priority

# I/O priority classes:
# 0: None (inherit)
# 1: Real-time
# 2: Best-effort (default)
# 3: Idle
```

### 8. Disk Management

#### Mounting Filesystems

```bash
# View mounts
mount                                     # All mounted filesystems
mount | column -t                         # Formatted view
cat /proc/mounts                          # Kernel view
df -h                                     # Disk space usage

# Mount filesystem
mount /dev/sdb1 /mnt/data                 # Mount device
mount -t ext4 /dev/sdb1 /mnt/data         # Specify filesystem type
mount -o ro /dev/sdb1 /mnt/data           # Mount read-only

# Unmount
umount /mnt/data                          # Unmount
umount -l /mnt/data                       # Lazy unmount
fuser -km /mnt/data                       # Kill processes and unmount
```

#### Persistent Mounts

```bash
# /etc/fstab format:
# <device> <mount point> <type> <options> <dump> <pass>

# Example entries:
UUID=12345678  /data  ext4  defaults  0  2
/dev/sdb1      /mnt/models  xfs  defaults  0  2

# Mount all fstab entries
mount -a

# Test fstab
mount -fav                                # Fake verbose all
```

#### Storage Information

```bash
# Block devices
lsblk                                     # List block devices
lsblk -f                                  # Show filesystems
blkid                                     # Block device UUIDs

# Disk usage by directory
du -h --max-depth=1 /data | sort -hr      # Sorted by size
ncdu /data                                # Interactive disk usage

# Find large files
find /data -type f -size +1G              # Files > 1GB
find /data -type f -size +1G -exec ls -lh {} \; | sort -k5 -hr

# Check filesystem
fsck /dev/sdb1                            # Filesystem check (unmounted!)
```

---

## Project: ML Infrastructure Monitoring Suite

Build a comprehensive monitoring and management toolkit for ML infrastructure.

### Requirements

**Tools to Create:**
1. Resource monitoring dashboard
2. Process management utility
3. Disk space analyzer for datasets
4. GPU utilization tracker
5. Log file cleanup utility

**Technical Requirements:**
- Real-time monitoring with updates
- Alert thresholds for resources
- Process filtering and management
- Detailed reporting with statistics
- Cleanup automation with safety checks
- Support for multiple GPUs
- Integration with system logs

### Implementation

See `solutions/` directory for complete implementations.

### Example Scripts

#### 1. Resource Monitor

```bash
#!/bin/bash
set -euo pipefail

# monitor_resources.sh - Real-time resource monitoring

readonly UPDATE_INTERVAL=2
readonly CPU_THRESHOLD=80
readonly MEM_THRESHOLD=85

monitor_loop() {
    while true; do
        clear

        echo "======================================"
        echo "ML Infrastructure Resource Monitor"
        echo "======================================"
        echo "Timestamp: $(date)"
        echo ""

        # CPU usage
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
        echo "CPU Usage: ${cpu_usage}%"

        # Memory usage
        mem_info=$(free -h | awk 'NR==2 {print $3 "/" $2}')
        echo "Memory: $mem_info"

        # Disk usage
        disk_usage=$(df -h / | awk 'NR==2 {print $5}')
        echo "Disk Usage: $disk_usage"

        # GPU usage
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            echo "GPU Information:"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
                --format=csv,noheader | while IFS=',' read -r idx name util mem_used mem_total; do
                echo "  GPU $idx: $util (Mem: $mem_used / $mem_total)"
            done
        fi

        sleep "$UPDATE_INTERVAL"
    done
}

monitor_loop
```

#### 2. Process Manager

```bash
#!/bin/bash
set -euo pipefail

# manage_ml_processes.sh - Manage ML training processes

list_ml_processes() {
    echo "ML Training Processes:"
    echo "======================"

    ps aux | grep -E "(python.*train|jupyter)" | grep -v grep | \
        awk '{printf "PID: %5s | CPU: %5s%% | MEM: %5s%% | %s\n", $2, $3, $4, $11}'
}

kill_process() {
    local pid=$1

    echo "Killing process $pid..."
    kill -15 "$pid"  # SIGTERM

    sleep 2

    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Process still running, force killing..."
        kill -9 "$pid"  # SIGKILL
    fi

    echo "Process $pid terminated"
}

main() {
    case "${1:-list}" in
        list)
            list_ml_processes
            ;;
        kill)
            kill_process "$2"
            ;;
        *)
            echo "Usage: $0 {list|kill PID}"
            exit 1
            ;;
    esac
}

main "$@"
```

---

## Practice Problems

### Problem 1: Disk Space Analyzer

Create a script that:
- Analyzes disk usage in `/data` directory
- Identifies top 10 largest directories
- Finds old files (> 90 days) for cleanup
- Generates cleanup recommendations

### Problem 2: Process Watchdog

Create a script that:
- Monitors specific processes (e.g., ML training)
- Restarts if process crashes
- Logs all events
- Sends alerts when intervention needed

### Problem 3: System Health Check

Create a script that:
- Checks CPU, memory, disk usage
- Verifies critical processes are running
- Monitors GPU health
- Generates daily health reports

---

## Best Practices

### 1. Safe File Operations

```bash
# Always backup before destructive operations
cp important_file{,.backup}

# Use -i for interactive confirmation
rm -i file.txt
mv -i old.txt new.txt

# Test with echo first
echo rm large_file.txt  # See what would be deleted
# Then run the actual command
```

### 2. Process Management

```bash
# Graceful shutdown before force kill
kill -15 PID
sleep 5
if ps -p PID > /dev/null; then
    kill -9 PID
fi

# Use nice for CPU-intensive tasks
nice -n 10 python train.py

# Monitor before killing
ps -p PID -o pid,ppid,cmd,pcpu,pmem
```

### 3. Resource Monitoring

```bash
# Set up alerts
if [[ $cpu_usage -gt $CPU_THRESHOLD ]]; then
    logger "High CPU usage: $cpu_usage%"
    # Send alert
fi

# Regular monitoring
watch -n 5 'ps aux --sort=-%cpu | head -20'

# Log resource usage
while true; do
    date >> resource_log.txt
    free -h >> resource_log.txt
    sleep 300
done
```

---

## Validation

Test your knowledge:

```bash
# Create test environment
mkdir -p /tmp/ml-test/{models,data,logs}

# Practice file operations
cd /tmp/ml-test
touch models/model_{v1,v2,v3}.pkl
echo "test data" > data/dataset.csv

# Practice permissions
chmod 644 models/*.pkl
chown $USER:$USER data/dataset.csv

# Practice process management
python3 -m http.server 8000 &
SERVER_PID=$!
# ... test ...
kill $SERVER_PID

# Practice monitoring
# Monitor system for 30 seconds
timeout 30 top

# Check specific process
ps -p $$ -o pid,ppid,cmd,pcpu,pmem
```

---

## Resources

- [Linux Filesystem Hierarchy](https://www.pathname.com/fhs/)
- [The Linux Command Line Book](http://linuxcommand.org/tlcl.php)
- [Process Management Guide](https://www.kernel.org/doc/html/latest/admin-guide/pm/index.html)
- [/proc Documentation](https://www.kernel.org/doc/html/latest/filesystems/proc.html)
- [NVIDIA Management Tools](https://developer.nvidia.com/nvidia-system-management-interface)

---

## Next Steps

1. **Exercise 03: SSH & Networking** - Remote access and network management
2. Practice file operations daily
3. Monitor your own ML experiments
4. Automate cleanup tasks
5. Build monitoring dashboards

---

**Master your infrastructure from the ground up! 🚀**
