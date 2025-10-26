# Step-by-Step Implementation Guide: Filesystem & Process Management

## Overview

Master Linux filesystem operations and process management by building monitoring and management tools for ML infrastructure. Learn to monitor system resources, manage GPU utilization, analyze disk usage, handle log cleanup, and control ML training processes.

**Time**: 4-5 hours | **Difficulty**: Beginner to Intermediate

---

## Prerequisites

```bash
# Verify system utilities
which ps top htop df du find

# Install additional tools
sudo apt update
sudo apt install -y \
    htop \
    iotop \
    ncdu \
    sysstat \
    jq

# For GPU monitoring (if available)
nvidia-smi --version  # NVIDIA GPUs

# Verify permissions for /proc and /sys
ls -la /proc/cpuinfo
ls -la /sys/class/thermal/
```

---

## Learning Objectives

By completing this exercise, you will be able to:

✅ Navigate the Linux filesystem hierarchy effectively
✅ Monitor system resources (CPU, memory, disk, GPU)
✅ Manage running processes and jobs
✅ Analyze disk space usage and clean up logs
✅ Use /proc filesystem for system information
✅ Implement resource monitoring for ML workloads
✅ Handle process signals and cleanup

---

## Phase 1: Resource Monitoring Script (90 minutes)

### Step 1: Understanding System Monitoring

Linux provides rich information through `/proc` and `/sys`:

```bash
# CPU information
cat /proc/cpuinfo | grep "model name" | head -1
cat /proc/loadavg  # Load averages

# Memory information
cat /proc/meminfo | grep -E "MemTotal|MemAvailable|MemFree"
free -h  # Human-readable memory stats

# Disk I/O
cat /proc/diskstats

# Process information
cat /proc/<PID>/status
cat /proc/<PID>/stat
```

### Step 2: Create monitor_resources.sh

Start with the script structure:

```bash
#!/bin/bash
#
# monitor_resources.sh - Monitor system resources for ML workloads
#
# Description:
#   Monitors CPU, memory, disk, and process usage
#   Generates alerts when thresholds are exceeded
#   Logs metrics for historical analysis
#
# Usage:
#   ./monitor_resources.sh [OPTIONS]
#
# Options:
#   -i, --interval SEC     Monitoring interval (default: 60)
#   -c, --cpu-threshold    CPU threshold percentage (default: 80)
#   -m, --mem-threshold    Memory threshold percentage (default: 85)
#   -d, --disk-threshold   Disk threshold percentage (default: 90)
#   -l, --log-file PATH    Log file path
#   -a, --alert-email      Email for alerts
#   -v, --verbose          Verbose output
#   -h, --help             Show help
#

set -euo pipefail

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
INTERVAL=60
CPU_THRESHOLD=80
MEM_THRESHOLD=85
DISK_THRESHOLD=90
LOG_FILE="/var/log/resource-monitor.log"
ALERT_EMAIL=""
VERBOSE=false
METRICS_DIR="/var/lib/metrics"

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        log "DEBUG" "$@"
    fi
}
```

### Step 3: Implement CPU Monitoring

Add CPU monitoring functions:

```bash
get_cpu_usage() {
    # Read /proc/stat to calculate CPU usage
    local cpu_line=$(grep "^cpu " /proc/stat)
    local idle=$(echo "$cpu_line" | awk '{print $5}')
    local total=0

    for value in $(echo "$cpu_line" | awk '{print $2,$3,$4,$5,$6,$7,$8}'); do
        total=$((total + value))
    done

    # Calculate percentage (simplified)
    local idle_pct=$(echo "scale=2; ($idle / $total) * 100" | bc)
    local usage_pct=$(echo "scale=2; 100 - $idle_pct" | bc)

    echo "$usage_pct"
}

get_cpu_count() {
    grep -c "^processor" /proc/cpuinfo
}

get_load_average() {
    local load=$(cat /proc/loadavg | awk '{print $1, $2, $3}')
    echo "$load"
}

check_cpu_threshold() {
    local usage=$1

    if (( $(echo "$usage > $CPU_THRESHOLD" | bc -l) )); then
        log "WARNING" "CPU usage ($usage%) exceeds threshold ($CPU_THRESHOLD%)"
        send_alert "CPU" "$usage"
        return 1
    fi

    log_verbose "CPU usage: $usage%"
    return 0
}
```

### Step 4: Implement Memory Monitoring

Add memory tracking:

```bash
get_memory_usage() {
    # Parse /proc/meminfo
    local mem_total=$(grep "MemTotal:" /proc/meminfo | awk '{print $2}')
    local mem_available=$(grep "MemAvailable:" /proc/meminfo | awk '{print $2}')

    local mem_used=$((mem_total - mem_available))
    local mem_pct=$(echo "scale=2; ($mem_used / $mem_total) * 100" | bc)

    echo "$mem_pct"
}

get_memory_details() {
    local mem_total=$(grep "MemTotal:" /proc/meminfo | awk '{print $2}')
    local mem_free=$(grep "MemFree:" /proc/meminfo | awk '{print $2}')
    local mem_available=$(grep "MemAvailable:" /proc/meminfo | awk '{print $2}')
    local mem_cached=$(grep "^Cached:" /proc/meminfo | awk '{print $2}')
    local mem_buffers=$(grep "Buffers:" /proc/meminfo | awk '{print $2}')

    # Convert to MB
    mem_total_mb=$((mem_total / 1024))
    mem_free_mb=$((mem_free / 1024))
    mem_available_mb=$((mem_available / 1024))
    mem_cached_mb=$((mem_cached / 1024))
    mem_buffers_mb=$((mem_buffers / 1024))

    cat <<EOF
Total: ${mem_total_mb}MB
Free: ${mem_free_mb}MB
Available: ${mem_available_mb}MB
Cached: ${mem_cached_mb}MB
Buffers: ${mem_buffers_mb}MB
EOF
}

check_memory_threshold() {
    local usage=$1

    if (( $(echo "$usage > $MEM_THRESHOLD" | bc -l) )); then
        log "WARNING" "Memory usage ($usage%) exceeds threshold ($MEM_THRESHOLD%)"
        send_alert "MEMORY" "$usage"
        return 1
    fi

    log_verbose "Memory usage: $usage%"
    return 0
}
```

### Step 5: Implement Disk Monitoring

Track disk space usage:

```bash
get_disk_usage() {
    local mount_point="${1:-/}"

    local usage=$(df -h "$mount_point" | awk 'NR==2 {print $5}' | sed 's/%//')
    echo "$usage"
}

get_disk_details() {
    df -h | awk 'NR==1 || $6 ~ /^\/$|^\/home$|^\/data$|^\/models$/'
}

check_disk_threshold() {
    local mount_point="$1"
    local usage=$(get_disk_usage "$mount_point")

    if [[ $usage -gt $DISK_THRESHOLD ]]; then
        log "WARNING" "Disk usage on $mount_point ($usage%) exceeds threshold ($DISK_THRESHOLD%)"
        send_alert "DISK" "$usage" "$mount_point"
        return 1
    fi

    log_verbose "Disk usage on $mount_point: $usage%"
    return 0
}

find_large_files() {
    local directory="${1:-.}"
    local size_mb="${2:-100}"

    log "INFO" "Finding files larger than ${size_mb}MB in $directory..."

    find "$directory" -type f -size "+${size_mb}M" -exec ls -lh {} \; | \
        awk '{print $9, $5}' | \
        sort -k2 -hr | \
        head -20
}
```

### Step 6: Implement Process Monitoring

Track top resource-consuming processes:

```bash
get_top_cpu_processes() {
    local count="${1:-5}"

    log "INFO" "Top $count CPU-consuming processes:"
    ps aux --sort=-%cpu | head -$((count + 1)) | \
        awk '{printf "  %-10s %5s  %5s  %s\n", $1, $3"%", $4"%", $11}'
}

get_top_memory_processes() {
    local count="${1:-5}"

    log "INFO" "Top $count memory-consuming processes:"
    ps aux --sort=-%mem | head -$((count + 1)) | \
        awk '{printf "  %-10s %5s  %5s  %s\n", $1, $4"%", $6, $11}'
}

get_process_count() {
    local total=$(ps aux | wc -l)
    local running=$(ps aux | awk '$8 ~ /R/ {print}' | wc -l)
    local sleeping=$(ps aux | awk '$8 ~ /S/ {print}' | wc -l)

    echo "Total: $total, Running: $running, Sleeping: $sleeping"
}
```

### Step 7: Add Metrics Collection

Save metrics for historical analysis:

```bash
save_metrics() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local cpu_usage=$(get_cpu_usage)
    local mem_usage=$(get_memory_usage)
    local disk_usage=$(get_disk_usage "/")
    local load_avg=$(get_load_average)

    # Create JSON metrics
    local metrics=$(cat <<EOF
{
  "timestamp": "$timestamp",
  "cpu": {
    "usage_percent": $cpu_usage,
    "load_average": "$load_avg",
    "cores": $(get_cpu_count)
  },
  "memory": {
    "usage_percent": $mem_usage
  },
  "disk": {
    "root_usage_percent": $disk_usage
  }
}
EOF
    )

    # Append to metrics file
    local metrics_file="$METRICS_DIR/metrics-$(date '+%Y-%m-%d').jsonl"
    mkdir -p "$METRICS_DIR"
    echo "$metrics" >> "$metrics_file"

    log_verbose "Metrics saved to $metrics_file"
}
```

### Step 8: Implement Alert System

Add basic alerting:

```bash
send_alert() {
    local resource="$1"
    local value="$2"
    local extra="${3:-}"

    local message="ALERT: $resource usage at ${value}%"
    if [[ -n "$extra" ]]; then
        message="$message ($extra)"
    fi

    log "ALERT" "$message"

    # Send email if configured
    if [[ -n "$ALERT_EMAIL" ]] && command -v mail &> /dev/null; then
        echo "$message" | mail -s "Resource Alert: $resource" "$ALERT_EMAIL"
    fi

    # Could also integrate with:
    # - Slack webhook
    # - PagerDuty API
    # - Prometheus Alertmanager
}
```

### Step 9: Create Main Monitoring Loop

Tie it all together:

```bash
monitor_loop() {
    log "INFO" "Starting resource monitoring (interval: ${INTERVAL}s)"
    log "INFO" "Thresholds - CPU: ${CPU_THRESHOLD}%, Memory: ${MEM_THRESHOLD}%, Disk: ${DISK_THRESHOLD}%"

    while true; do
        log "INFO" "=========================================="
        log "INFO" "Resource Check - $(date)"
        log "INFO" "=========================================="

        # CPU monitoring
        local cpu_usage=$(get_cpu_usage)
        log "INFO" "CPU Usage: ${cpu_usage}%"
        check_cpu_threshold "$cpu_usage"

        local load_avg=$(get_load_average)
        log "INFO" "Load Average: $load_avg"

        # Memory monitoring
        local mem_usage=$(get_memory_usage)
        log "INFO" "Memory Usage: ${mem_usage}%"
        check_memory_threshold "$mem_usage"

        if [[ "$VERBOSE" == true ]]; then
            get_memory_details
        fi

        # Disk monitoring
        check_disk_threshold "/"
        if [[ -d "/data" ]]; then
            check_disk_threshold "/data"
        fi

        if [[ "$VERBOSE" == true ]]; then
            get_disk_details
        fi

        # Process monitoring
        if [[ "$VERBOSE" == true ]]; then
            get_process_count
            get_top_cpu_processes 3
            get_top_memory_processes 3
        fi

        # Save metrics
        save_metrics

        log "INFO" "Next check in ${INTERVAL} seconds"
        sleep "$INTERVAL"
    done
}

main() {
    # Parse arguments (implementation similar to backup_data.sh)
    # ...

    # Start monitoring
    monitor_loop
}

main "$@"
```

### Step 10: Test the Monitor

```bash
# Run in foreground with verbose output
./solutions/monitor_resources.sh --verbose --interval 10

# Run in background
./solutions/monitor_resources.sh --interval 60 --log-file /tmp/monitor.log &

# Check logs
tail -f /var/log/resource-monitor.log

# View saved metrics
cat /var/lib/metrics/metrics-$(date '+%Y-%m-%d').jsonl | jq '.'

# Stress test CPU to trigger alert
stress --cpu 4 --timeout 60  # Install: sudo apt install stress

# Check monitoring detected the spike
grep "WARNING.*CPU" /var/log/resource-monitor.log
```

---

## Phase 2: GPU Utilization Tracking (60 minutes)

### Summary

The `track_gpu_util.sh` script monitors NVIDIA GPU usage for ML training jobs.

**Key Features**:
- Monitor GPU utilization, memory, temperature
- Track per-process GPU usage
- Alert on GPU memory leaks
- Log metrics for analysis

**Core Implementation**:

```bash
get_gpu_utilization() {
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits
}

get_gpu_processes() {
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
}

monitor_gpu_memory_leak() {
    local pid="$1"
    local previous_mem=0
    local leak_threshold=100  # MB increase per minute

    while true; do
        local current_mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | \
                           grep "^$pid" | awk '{print $2}')

        if [[ -n "$current_mem" ]]; then
            local mem_increase=$((current_mem - previous_mem))

            if [[ $mem_increase -gt $leak_threshold ]]; then
                log "WARNING" "Potential GPU memory leak detected for PID $pid (increase: ${mem_increase}MB/min)"
            fi

            previous_mem=$current_mem
        fi

        sleep 60
    done
}
```

---

## Phase 3: Disk Space Analysis (60 minutes)

### Summary

The `analyze_disk_space.sh` script provides detailed disk usage analysis.

**Features**:
- Find large directories and files
- Identify growth trends
- Generate disk usage reports
- Suggest cleanup candidates

**Key Functions**:

```bash
analyze_directory() {
    local directory="$1"
    local depth="${2:-2}"

    log "INFO" "Analyzing disk usage in $directory (depth: $depth)..."

    du -h --max-depth="$depth" "$directory" 2>/dev/null | \
        sort -hr | \
        head -20
}

find_duplicate_files() {
    local directory="$1"

    log "INFO" "Finding duplicate files in $directory..."

    find "$directory" -type f -exec md5sum {} + 2>/dev/null | \
        sort | \
        uniq -w32 -dD
}

find_old_files() {
    local directory="$1"
    local days="${2:-90}"

    log "INFO" "Finding files older than $days days in $directory..."

    find "$directory" -type f -mtime +$days -exec ls -lh {} \; | \
        awk '{print $9, $5, $6, $7, $8}' | \
        sort -k2 -hr | \
        head -20
}

generate_disk_report() {
    local output_file="${1:-disk-report-$(date +%Y%m%d).txt}"

    cat > "$output_file" <<EOF
Disk Usage Report
Generated: $(date)
=====================================

Overall Disk Usage:
$(df -h)

Largest Directories:
$(du -h / --max-depth=2 2>/dev/null | sort -hr | head -20)

Largest Files:
$(find / -type f -size +100M -exec ls -lh {} \; 2>/dev/null | \
  awk '{print $9, $5}' | sort -k2 -hr | head -20)

Disk I/O Statistics:
$(iostat -x 1 2 | tail -20)

EOF

    log "SUCCESS" "Disk report generated: $output_file"
}
```

---

## Phase 4: Log Cleanup Management (60 minutes)

### Summary

The `cleanup_logs.sh` script manages log file growth and rotation.

**Features**:
- Find and compress old logs
- Implement retention policies
- Archive logs before deletion
- Clean up application-specific logs

**Implementation**:

```bash
find_large_logs() {
    local log_dir="${1:-/var/log}"
    local size_mb="${2:-50}"

    find "$log_dir" -type f -name "*.log" -size "+${size_mb}M" -exec ls -lh {} \;
}

compress_old_logs() {
    local log_dir="${1:-/var/log}"
    local days_old="${2:-7}"

    log "INFO" "Compressing logs older than $days_old days in $log_dir..."

    find "$log_dir" -type f -name "*.log" -mtime +$days_old ! -name "*.gz" | \
    while IFS= read -r logfile; do
        log_verbose "Compressing: $logfile"
        gzip "$logfile"
    done
}

cleanup_old_logs() {
    local log_dir="${1:-/var/log}"
    local retention_days="${2:-30}"

    log "INFO" "Removing compressed logs older than $retention_days days..."

    find "$log_dir" -type f -name "*.gz" -mtime +$retention_days -delete
}

rotate_log_file() {
    local log_file="$1"
    local max_size_mb="${2:-100}"

    local file_size_mb=$(du -m "$log_file" | cut -f1)

    if [[ $file_size_mb -gt $max_size_mb ]]; then
        log "INFO" "Rotating $log_file (${file_size_mb}MB > ${max_size_mb}MB)"

        local timestamp=$(date +%Y%m%d_%H%M%S)
        local rotated_file="${log_file}.${timestamp}"

        mv "$log_file" "$rotated_file"
        gzip "$rotated_file"

        # Create new empty log
        touch "$log_file"
        chmod 644 "$log_file"

        log "SUCCESS" "Log rotated to ${rotated_file}.gz"
    fi
}
```

---

## Phase 5: Process Management (60 minutes)

### Summary

The `manage_processes.sh` script controls ML training processes.

**Features**:
- Start/stop training jobs
- Monitor process resources
- Handle graceful shutdowns
- Restart crashed processes

**Key Implementations**:

```bash
find_process_by_name() {
    local process_name="$1"

    pgrep -f "$process_name"
}

get_process_info() {
    local pid="$1"

    if [[ ! -f "/proc/$pid/status" ]]; then
        log "ERROR" "Process $pid not found"
        return 1
    fi

    log "INFO" "Process Information for PID $pid:"
    echo "  Name: $(cat /proc/$pid/comm)"
    echo "  Status: $(grep "State:" /proc/$pid/status | awk '{print $2}')"
    echo "  CPU: $(ps -p $pid -o %cpu= | tr -d ' ')%"
    echo "  Memory: $(ps -p $pid -o %mem= | tr -d ' ')%"
    echo "  Runtime: $(ps -p $pid -o etime= | tr -d ' ')"
    echo "  Command: $(cat /proc/$pid/cmdline | tr '\0' ' ')"
}

kill_process_gracefully() {
    local pid="$1"
    local max_wait="${2:-30}"

    log "INFO" "Sending SIGTERM to process $pid..."
    kill -TERM "$pid"

    # Wait for process to exit
    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
        if [[ $waited -ge $max_wait ]]; then
            log "WARNING" "Process $pid did not exit after ${max_wait}s, sending SIGKILL..."
            kill -KILL "$pid"
            break
        fi

        sleep 1
        ((waited++))
    done

    if ! kill -0 "$pid" 2>/dev/null; then
        log "SUCCESS" "Process $pid terminated successfully"
        return 0
    else
        log "ERROR" "Failed to kill process $pid"
        return 1
    fi
}

monitor_process_resources() {
    local pid="$1"
    local interval="${2:-5}"

    log "INFO" "Monitoring process $pid (interval: ${interval}s)"

    while kill -0 "$pid" 2>/dev/null; do
        local cpu=$(ps -p $pid -o %cpu= | tr -d ' ')
        local mem=$(ps -p $pid -o %mem= | tr -d ' ')
        local vsz=$(ps -p $pid -o vsz= | tr -d ' ')

        log "INFO" "PID $pid - CPU: ${cpu}%, Memory: ${mem}%, VSZ: ${vsz}KB"

        sleep "$interval"
    done

    log "INFO" "Process $pid has exited"
}
```

---

## Testing & Validation

### Test All Scripts

```bash
# Test resource monitor
./solutions/monitor_resources.sh --verbose --interval 5 &
MONITOR_PID=$!

# Generate load and observe
stress --cpu 2 --timeout 30

# Stop monitor
kill $MONITOR_PID

# Test disk analysis
./solutions/analyze_disk_space.sh /var/log

# Test log cleanup (dry-run first)
./solutions/cleanup_logs.sh --dry-run /var/log 30

# Test GPU monitoring (if available)
./solutions/track_gpu_util.sh --interval 10

# Test process management
./solutions/manage_processes.sh monitor bash

# Lint all scripts
shellcheck solutions/*.sh
```

---

## Best Practices Demonstrated

1. **Use /proc for system info**: Direct access to kernel data
2. **Parse output carefully**: Use `awk`, `grep`, `sed` correctly
3. **Handle edge cases**: Check for missing files, empty values
4. **Signal handling**: Graceful shutdowns with SIGTERM before SIGKILL
5. **Resource cleanup**: Remove temporary files and processes
6. **Metrics collection**: Store historical data for analysis

---

## Next Steps

1. Integrate with monitoring systems (Prometheus, Grafana)
2. Add advanced alerting (PagerDuty, Slack)
3. Create dashboards for visualizing metrics
4. Implement predictive analytics for resource planning
5. Add multi-host monitoring capabilities

---

## Resources

- [Linux /proc Filesystem](https://www.kernel.org/doc/html/latest/filesystems/proc.html)
- [ps Command Manual](https://man7.org/linux/man-pages/man1/ps.1.html)
- [NVIDIA SMI Documentation](https://developer.nvidia.com/nvidia-system-management-interface)
- [Bash Process Management](https://www.gnu.org/software/bash/manual/html_node/Job-Control.html)

---

**Congratulations!** You've built production-ready system monitoring and management tools. These skills are essential for maintaining ML infrastructure at scale. 🚀
