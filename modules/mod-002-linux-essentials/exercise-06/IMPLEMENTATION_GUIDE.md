# Implementation Guide: Log File Analysis for ML Systems

## Overview

This guide teaches you to analyze log files from ML training systems, troubleshoot issues, and extract insights from logs. You'll work with application logs, system logs, and training metrics to diagnose problems and monitor ML workloads.

**Estimated Time:** 75-90 minutes
**Difficulty:** Intermediate

## Prerequisites

- Completed Exercises 01-05
- Understanding of Linux command line
- Basic regex knowledge
- Text editor familiarity

## Phase 1: Understanding Log Files (20 minutes)

### Step 1.1: Common Log Locations

```bash
mkdir -p ~/log-analysis-lab
cd ~/log-analysis-lab

# System logs locations
sudo ls -lh /var/log/

# Key system logs
ls -lh /var/log/syslog 2>/dev/null || ls -lh /var/log/messages 2>/dev/null
ls -lh /var/log/auth.log 2>/dev/null
ls -lh /var/log/kern.log 2>/dev/null
```

**Create reference guide:**
```bash
cat > log_locations.md << 'EOF'
# Common Linux Log Locations

## System Logs
/var/log/syslog          # System messages (Debian/Ubuntu)
/var/log/messages        # System messages (RHEL/CentOS)
/var/log/auth.log        # Authentication logs
/var/log/kern.log        # Kernel logs
/var/log/dmesg           # Boot messages

## Application Logs
/var/log/apache2/        # Apache web server
/var/log/nginx/          # Nginx web server
/var/log/mysql/          # MySQL database
~/.local/share/          # User application logs

## ML Application Logs
~/ml-projects/*/logs/    # Project-specific logs
/var/log/ml-services/    # ML services (if configured)
~/.cache/torch/          # PyTorch cache/logs
~/.keras/                # Keras logs
EOF

cat log_locations.md
```

### Step 1.2: Create Sample ML Training Logs

```bash
# Create sample training log
cat > training.log << 'EOF'
2024-01-15 10:00:00,123 INFO Starting model training...
2024-01-15 10:00:00,456 INFO Dataset: fraud_detection_v2.csv
2024-01-15 10:00:00,789 INFO Training samples: 50000, Validation: 10000
2024-01-15 10:00:01,234 INFO Model: RandomForestClassifier(n_estimators=100)
2024-01-15 10:00:05,567 INFO Epoch 1/10 - Loss: 0.6931, Accuracy: 0.5123, Val_Loss: 0.6892, Val_Acc: 0.5234
2024-01-15 10:00:10,890 INFO Epoch 2/10 - Loss: 0.5234, Accuracy: 0.7456, Val_Loss: 0.5123, Val_Acc: 0.7589
2024-01-15 10:00:15,123 WARNING Learning rate decreased to 0.0001
2024-01-15 10:00:16,456 INFO Epoch 3/10 - Loss: 0.4567, Accuracy: 0.7890, Val_Loss: 0.4456, Val_Acc: 0.8012
2024-01-15 10:00:21,789 ERROR OutOfMemoryError: CUDA out of memory
2024-01-15 10:00:21,890 INFO Reducing batch size from 128 to 64
2024-01-15 10:00:22,123 INFO Resuming training...
2024-01-15 10:00:27,456 INFO Epoch 4/10 - Loss: 0.4123, Accuracy: 0.8123, Val_Loss: 0.4089, Val_Acc: 0.8234
2024-01-15 10:00:32,789 INFO Epoch 5/10 - Loss: 0.3890, Accuracy: 0.8345, Val_Loss: 0.3912, Val_Acc: 0.8456
2024-01-15 10:00:38,123 WARNING Validation loss not improving for 2 epochs
2024-01-15 10:00:43,456 INFO Epoch 6/10 - Loss: 0.3678, Accuracy: 0.8478, Val_Loss: 0.3789, Val_Acc: 0.8523
2024-01-15 10:00:48,789 INFO Epoch 7/10 - Loss: 0.3567, Accuracy: 0.8567, Val_Loss: 0.3698, Val_Acc: 0.8589
2024-01-15 10:00:54,123 INFO Epoch 8/10 - Loss: 0.3456, Accuracy: 0.8634, Val_Loss: 0.3623, Val_Acc: 0.8612
2024-01-15 10:00:59,456 INFO Epoch 9/10 - Loss: 0.3389, Accuracy: 0.8678, Val_Loss: 0.3578, Val_Acc: 0.8645
2024-01-15 10:01:04,789 INFO Epoch 10/10 - Loss: 0.3345, Accuracy: 0.8712, Val_Loss: 0.3534, Val_Acc: 0.8678
2024-01-15 10:01:05,123 INFO Training completed in 65.0 seconds
2024-01-15 10:01:05,456 INFO Best validation accuracy: 0.8678
2024-01-15 10:01:05,789 INFO Model saved to: models/fraud_detector_v2.pkl
EOF

# Create system log sample
cat > system.log << 'EOF'
Jan 15 10:00:00 ml-server kernel: [12345.678] nvidia: loading out-of-tree module taints kernel.
Jan 15 10:00:00 ml-server systemd[1]: Started NVIDIA Persistence Daemon.
Jan 15 10:00:05 ml-server sshd[1234]: Accepted publickey for mluser from 192.168.1.100
Jan 15 10:00:10 ml-server docker[5678]: Container ml-training-job started
Jan 15 10:00:21 ml-server kernel: [12366.890] CUDA error: out of memory
Jan 15 10:00:25 ml-server ml-service[9012]: WARNING: GPU memory usage at 95%
Jan 15 10:01:00 ml-server cron[2345]: (root) CMD (run-parts /etc/cron.hourly)
Jan 15 10:01:05 ml-server docker[5678]: Container ml-training-job exited with code 0
EOF

echo "Sample logs created successfully"
ls -lh *.log
```

**Validation:**
- [ ] Created sample training and system logs
- [ ] Understand common log locations

## Phase 2: Basic Log Reading (20 minutes)

### Step 2.1: Read Logs with cat, less, head, tail

```bash
# View entire log
cat training.log

# View with pager (better for large files)
less training.log  # Press q to quit

# First 10 lines
head training.log

# Last 10 lines
tail training.log

# Specific number of lines
head -n 5 training.log
tail -n 5 training.log

# Multiple files
cat training.log system.log
```

### Step 2.2: Real-Time Log Monitoring

```bash
# Follow log in real-time (simulated)
cat > generate_logs.sh << 'EOF'
#!/bin/bash
# Simulate continuous log generation

for i in {1..20}; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') INFO Processing batch $i/20"
    sleep 1
done
EOF

chmod +x generate_logs.sh

# In one terminal, run:
./generate_logs.sh >> live_training.log

# In another terminal (or run in background):
tail -f live_training.log

# Follow last 20 lines
tail -n 20 -f live_training.log

# Follow multiple files
tail -f training.log system.log
```

**Validation:**
- [ ] Can view logs with different tools
- [ ] Can monitor logs in real-time with tail -f

## Phase 3: Log Filtering with grep (30 minutes)

### Step 3.1: Basic grep Patterns

```bash
# Find errors
grep ERROR training.log

# Find warnings
grep WARNING training.log

# Find errors OR warnings
grep -E "ERROR|WARNING" training.log

# Case-insensitive search
grep -i error training.log

# Count occurrences
grep -c ERROR training.log
grep -c WARNING training.log

# Show line numbers
grep -n ERROR training.log
```

### Step 3.2: Context and Inverted Matching

```bash
# Show 2 lines before and after match
grep -C 2 "OutOfMemoryError" training.log

# Show 3 lines after match
grep -A 3 "ERROR" training.log

# Show 2 lines before match
grep -B 2 "Reducing batch size" training.log

# Invert match (exclude lines)
grep -v INFO training.log  # Show non-INFO lines

# Multiple patterns
grep -e ERROR -e WARNING training.log
```

### Step 3.3: Advanced grep with Regex

```bash
# Match accuracy values
grep -oP "Accuracy: \K[0-9.]+" training.log

# Extract epoch numbers
grep -oP "Epoch \K[0-9]+" training.log

# Find lines with specific pattern
grep -E "Epoch [0-9]+/[0-9]+" training.log

# Validate accuracy is improving
grep "Val_Acc:" training.log | grep -oP "Val_Acc: \K[0-9.]+"

# Find timestamps
grep -oP "^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}" training.log
```

### Step 3.4: grep Performance Tips

```bash
# Search recursively in directory
grep -r "ERROR" /var/log/ 2>/dev/null | head

# Search only specific file types
grep -r --include="*.log" "ERROR" .

# Exclude directories
grep -r --exclude-dir="cache" "ERROR" .

# Use fgrep for fixed strings (faster)
fgrep "ERROR" training.log

# Parallel grep for large logs (if ripgrep available)
command -v rg && rg "ERROR" training.log || grep "ERROR" training.log
```

**Validation:**
- [ ] Can filter logs with grep
- [ ] Can use regex patterns
- [ ] Can extract specific values
- [ ] Understand context options (-A, -B, -C)

## Phase 4: Log Analysis with awk (30 minutes)

### Step 4.1: Extract Specific Columns

```bash
# Print specific fields (space-delimited)
awk '{print $1, $2, $4}' training.log | head

# Print last field
awk '{print $NF}' training.log | head

# Print all except first two fields
awk '{$1=$2=""; print $0}' training.log | head
```

### Step 4.2: Filter and Process

```bash
# Print only ERROR lines
awk '/ERROR/ {print}' training.log

# Print lines where accuracy > 0.8
awk '/Accuracy:/ {
    match($0, /Accuracy: ([0-9.]+)/, arr)
    if (arr[1] > 0.8) print
}' training.log

# Count log levels
awk '{print $4}' training.log | sort | uniq -c

# Calculate average accuracy
awk '/Accuracy: [0-9.]+/ {
    match($0, /Accuracy: ([0-9.]+)/, arr)
    sum += arr[1]
    count++
}
END {
    if (count > 0) print "Average Accuracy:", sum/count
}' training.log
```

### Step 4.3: Advanced awk - Extract Metrics

```bash
# Extract all metrics from training
cat > extract_metrics.awk << 'EOF'
/Epoch [0-9]+/ {
    # Extract epoch number
    match($0, /Epoch ([0-9]+)/, epoch_arr)
    epoch = epoch_arr[1]

    # Extract loss
    match($0, /Loss: ([0-9.]+)/, loss_arr)
    loss = loss_arr[1]

    # Extract accuracy
    match($0, /Accuracy: ([0-9.]+)/, acc_arr)
    acc = acc_arr[1]

    # Extract validation accuracy
    match($0, /Val_Acc: ([0-9.]+)/, val_arr)
    val_acc = val_arr[1]

    printf "Epoch %d: Loss=%.4f, Acc=%.4f, Val_Acc=%.4f\n", epoch, loss, acc, val_acc
}
EOF

awk -f extract_metrics.awk training.log
```

### Step 4.4: Generate Summary Statistics

```bash
cat > log_summary.sh << 'EOF'
#!/bin/bash
LOG_FILE="$1"

echo "=== Log Analysis Summary ==="
echo ""

echo "Total lines: $(wc -l < "$LOG_FILE")"
echo ""

echo "Log level distribution:"
awk '{print $4}' "$LOG_FILE" | sort | uniq -c | sort -rn
echo ""

echo "Errors found:"
grep -c ERROR "$LOG_FILE"
grep ERROR "$LOG_FILE"
echo ""

echo "Warnings found:"
grep -c WARNING "$LOG_FILE"
grep WARNING "$LOG_FILE"
echo ""

echo "Training metrics:"
awk -f extract_metrics.awk "$LOG_FILE"
EOF

chmod +x log_summary.sh
./log_summary.sh training.log
```

**Validation:**
- [ ] Can extract specific fields with awk
- [ ] Can perform calculations on log data
- [ ] Can generate summary statistics

## Phase 5: System Logs with journalctl (20 minutes)

### Step 5.1: Basic journalctl Usage

```bash
# View system journal (last 50 lines)
sudo journalctl -n 50

# Follow journal in real-time
sudo journalctl -f

# Show journal for current boot
sudo journalctl -b

# Show journal from specific service
sudo journalctl -u ssh
sudo journalctl -u docker

# Show kernel messages
sudo journalctl -k
```

### Step 5.2: Time-Based Filtering

```bash
# Logs since specific time
sudo journalctl --since "2024-01-15 10:00:00"
sudo journalctl --since "1 hour ago"
sudo journalctl --since "today"

# Logs within time range
sudo journalctl --since "1 hour ago" --until "30 min ago"

# Logs from yesterday
sudo journalctl --since yesterday --until today
```

### Step 5.3: Filtering and Output

```bash
# Filter by priority (error level)
sudo journalctl -p err  # Errors only
sudo journalctl -p warning  # Warnings and above

# Show only specific fields
sudo journalctl -o json-pretty -n 5

# Export to file
sudo journalctl --since today > today_logs.txt

# Search for pattern
sudo journalctl | grep -i "cuda\|nvidia"
```

**Validation:**
- [ ] Can view system logs with journalctl
- [ ] Can filter by time and service
- [ ] Understand log priorities

## Phase 6: Log Analysis Scripts (30 minutes)

### Step 6.1: Training Log Analyzer

```bash
cat > analyze_training.sh << 'EOF'
#!/bin/bash
# Analyze ML training logs

LOG_FILE="$1"

if [ -z "$LOG_FILE" ]; then
    echo "Usage: $0 <log_file>"
    exit 1
fi

echo "=== Training Log Analysis ==="
echo "Log file: $LOG_FILE"
echo "Analysis time: $(date)"
echo ""

# Training duration
START_TIME=$(grep "Starting model training" "$LOG_FILE" | awk '{print $1, $2}')
END_TIME=$(grep "Training completed" "$LOG_FILE" | awk '{print $1, $2}')
echo "Start time: $START_TIME"
echo "End time: $END_TIME"
echo ""

# Epochs completed
EPOCHS=$(grep -c "Epoch [0-9]" "$LOG_FILE")
echo "Epochs completed: $EPOCHS"
echo ""

# Final metrics
echo "Final metrics:"
grep "Epoch.*10/10" "$LOG_FILE" | grep -oP "(Loss|Accuracy|Val_Acc): [0-9.]+" | while read line; do
    echo "  $line"
done
echo ""

# Best validation accuracy
BEST_VAL_ACC=$(grep "Best validation accuracy" "$LOG_FILE" | grep -oP "[0-9.]+")
echo "Best validation accuracy: $BEST_VAL_ACC"
echo ""

# Errors and warnings
echo "Issues encountered:"
echo "  Errors: $(grep -c ERROR "$LOG_FILE")"
echo "  Warnings: $(grep -c WARNING "$LOG_FILE")"

if grep -q ERROR "$LOG_FILE"; then
    echo ""
    echo "Error details:"
    grep ERROR "$LOG_FILE" | sed 's/^/  /'
fi

if grep -q WARNING "$LOG_FILE"; then
    echo ""
    echo "Warning details:"
    grep WARNING "$LOG_FILE" | sed 's/^/  /'
fi
EOF

chmod +x analyze_training.sh
./analyze_training.sh training.log
```

### Step 6.2: Error Pattern Detector

```bash
cat > detect_errors.sh << 'EOF'
#!/bin/bash
# Detect common error patterns in logs

LOG_FILE="$1"

echo "=== Error Pattern Detection ==="
echo ""

# CUDA/GPU errors
if grep -qi "cuda.*error\|out of memory" "$LOG_FILE"; then
    echo "⚠️  GPU/CUDA errors detected:"
    grep -i "cuda.*error\|out of memory" "$LOG_FILE"
    echo ""
fi

# Network errors
if grep -qi "connection.*refused\|timeout\|network.*error" "$LOG_FILE"; then
    echo "⚠️  Network errors detected:"
    grep -i "connection.*refused\|timeout\|network.*error" "$LOG_FILE"
    echo ""
fi

# File I/O errors
if grep -qi "no such file\|permission denied\|disk.*full" "$LOG_FILE"; then
    echo "⚠️  File I/O errors detected:"
    grep -i "no such file\|permission denied\|disk.*full" "$LOG_FILE"
    echo ""
fi

# Memory errors
if grep -qi "out of memory\|memory error\|allocation failed" "$LOG_FILE"; then
    echo "⚠️  Memory errors detected:"
    grep -i "out of memory\|memory error\|allocation failed" "$LOG_FILE"
    echo ""
fi

# Model convergence issues
if grep -qi "nan\|inf\|diverging\|exploding" "$LOG_FILE"; then
    echo "⚠️  Model convergence issues detected:"
    grep -i "nan\|inf\|diverging\|exploding" "$LOG_FILE"
    echo ""
fi

echo "Analysis complete."
EOF

chmod +x detect_errors.sh
./detect_errors.sh training.log
```

### Step 6.3: Log Metrics Dashboard

```bash
cat > metrics_dashboard.sh << 'EOF'
#!/bin/bash
# Generate metrics dashboard from training logs

LOG_FILE="$1"

clear
echo "╔════════════════════════════════════════════╗"
echo "║     ML Training Metrics Dashboard         ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Extract metrics
awk '/Epoch [0-9]+\/[0-9]+/ {
    match($0, /Epoch ([0-9]+)/, e)
    match($0, /Loss: ([0-9.]+)/, l)
    match($0, /Accuracy: ([0-9.]+)/, a)
    match($0, /Val_Acc: ([0-9.]+)/, v)

    epoch = e[1]
    loss = l[1]
    acc = a[1]
    val_acc = v[1]

    printf "Epoch %2d │ Loss: %.4f │ Acc: %.4f │ Val: %.4f", epoch, loss, acc, val_acc

    # Show progress bar for validation accuracy
    bar_length = int(val_acc * 50)
    printf " │ "
    for (i = 0; i < bar_length; i++) printf "█"
    printf "\n"
}' "$LOG_FILE"

echo ""
echo "Status: $(grep -q "Training completed" "$LOG_FILE" && echo "✓ Completed" || echo "⚠ In Progress")"
echo ""
EOF

chmod +x metrics_dashboard.sh
./metrics_dashboard.sh training.log
```

**Validation:**
- [ ] Created training log analyzer
- [ ] Created error pattern detector
- [ ] Created metrics dashboard

## Phase 7: Log Rotation and Retention (15 minutes)

### Step 7.1: Understanding logrotate

```bash
# View logrotate configuration
cat /etc/logrotate.conf

# View application-specific configs
ls -la /etc/logrotate.d/

# Example logrotate config
cat > ml-training.logrotate << 'EOF'
/home/mluser/ml-projects/*/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 mluser mluser
    postrotate
        # Optional: restart service
        # systemctl reload ml-service
    endscript
}
EOF

# Test logrotate config
sudo logrotate -d ml-training.logrotate
```

### Step 7.2: Manual Log Rotation Script

```bash
cat > rotate_logs.sh << 'EOF'
#!/bin/bash
# Manual log rotation script

LOG_DIR="${1:-./logs}"
MAX_AGE_DAYS=7
MAX_SIZE_MB=100

echo "Rotating logs in: $LOG_DIR"

# Find and compress old logs
find "$LOG_DIR" -name "*.log" -mtime +1 -size +${MAX_SIZE_MB}M -exec gzip {} \;

# Delete old compressed logs
find "$LOG_DIR" -name "*.log.gz" -mtime +$MAX_AGE_DAYS -delete

# Count remaining logs
echo "Current logs: $(find "$LOG_DIR" -name "*.log" | wc -l)"
echo "Compressed logs: $(find "$LOG_DIR" -name "*.log.gz" | wc -l)"
EOF

chmod +x rotate_logs.sh
```

**Validation:**
- [ ] Understand logrotate configuration
- [ ] Can create custom rotation rules

## Common Issues and Solutions

### Issue 1: Permission Denied Reading Logs

**Symptoms:**
```
cat: /var/log/syslog: Permission denied
```

**Solution:**
```bash
# Use sudo
sudo cat /var/log/syslog

# Or add user to log group
sudo usermod -a -G adm $USER

# Re-login for group changes to take effect
```

### Issue 2: Log File Too Large for cat

**Symptoms:**
Terminal hangs when opening huge log file

**Solution:**
```bash
# Use less instead of cat
less huge.log

# View specific sections
head -n 1000 huge.log
tail -n 1000 huge.log

# Search without loading entire file
grep "ERROR" huge.log | less

# Get file size first
ls -lh huge.log
```

### Issue 3: Real-Time Monitoring Not Working

**Symptoms:**
`tail -f` not showing new lines

**Solution:**
```bash
# Check if file is being appended
stat logfile.log

# Use tail with retry
tail -F logfile.log  # Capital F follows file even if renamed

# Check if process is writing to deleted file
lsof | grep deleted | grep log
```

### Issue 4: Cannot Find Specific Errors

**Symptoms:**
grep returns no results but errors exist

**Solution:**
```bash
# Case-insensitive search
grep -i error logfile.log

# Search for variations
grep -E "error|ERROR|Error" logfile.log

# Check file encoding
file logfile.log

# Search binary files
strings binary.log | grep error
```

## Best Practices Summary

### Log Reading

✅ Use `less` for large files (not cat)
✅ Use `tail -f` for real-time monitoring
✅ Use `grep` before reading to filter
✅ Check file size before opening: `ls -lh`
✅ Use sudo only when necessary

### Log Filtering

✅ Use `-i` for case-insensitive searches
✅ Use `-C` for context around matches
✅ Pipe through `less` for long outputs
✅ Use regex for pattern matching
✅ Combine grep with awk for complex filters

### Log Analysis

✅ Create reusable analysis scripts
✅ Extract metrics to CSV for visualization
✅ Automate common analysis tasks
✅ Monitor errors and warnings regularly
✅ Set up alerts for critical patterns

### Log Management

✅ Implement log rotation
✅ Compress old logs
✅ Delete logs older than retention period
✅ Monitor disk space used by logs
✅ Centralize logs from multiple services

### ML-Specific

✅ Log training metrics (loss, accuracy, etc.)
✅ Log hyperparameters and model config
✅ Log errors with full stack traces
✅ Include timestamps in all log entries
✅ Separate training logs from system logs

## Completion Checklist

### Understanding
- [ ] Know common log locations
- [ ] Understand log formats
- [ ] Can identify log levels (INFO, WARNING, ERROR)

### Basic Operations
- [ ] Can view logs with cat, less, head, tail
- [ ] Can monitor logs in real-time with tail -f
- [ ] Can read system logs with journalctl

### Filtering
- [ ] Can filter logs with grep
- [ ] Can use regex patterns
- [ ] Can extract context around matches

### Analysis
- [ ] Can parse logs with awk
- [ ] Can extract metrics from logs
- [ ] Can generate summary statistics
- [ ] Created custom analysis scripts

### Management
- [ ] Understand log rotation
- [ ] Can configure logrotate
- [ ] Can manage log retention

### Troubleshooting
- [ ] Can detect error patterns
- [ ] Can analyze training failures
- [ ] Can identify performance issues

## Next Steps

1. **Exercise 07: System Troubleshooting** - Apply log analysis to diagnose system issues
2. **Advanced Topics:**
   - Centralized logging (ELK stack, Grafana Loki)
   - Structured logging (JSON logs)
   - Log aggregation and visualization

3. **ML-Specific:**
   - Experiment tracking (Weights & Biases, MLflow)
   - Performance profiling logs
   - Distributed training logs

## Resources

- [grep Manual](https://www.gnu.org/software/grep/manual/)
- [awk Tutorial](https://www.gnu.org/software/gawk/manual/)
- [journalctl Documentation](https://www.freedesktop.org/software/systemd/man/journalctl.html)
- [logrotate Manual](https://linux.die.net/man/8/logrotate)

Congratulations! You can now effectively analyze logs from ML systems.
