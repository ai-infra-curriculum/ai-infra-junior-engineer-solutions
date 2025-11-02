# Implementation Guide: Real-World Troubleshooting Scenarios

## Overview

This guide provides hands-on practice with realistic ML infrastructure troubleshooting scenarios. Learn systematic approaches to diagnose and fix common issues like disk full errors, permission problems, hung processes, and network connectivity issues.

**Estimated Time:** 90-120 minutes
**Difficulty:** Intermediate to Advanced

## Prerequisites

- Completed Exercises 01-06
- Confidence with Linux command line
- Access to test Linux system

## Troubleshooting Framework

Use this systematic approach for all scenarios:

1. **Identify** - What is the symptom/error?
2. **Gather Information** - Collect relevant logs and metrics
3. **Hypothesize** - What could cause this?
4. **Test** - Verify hypothesis
5. **Fix** - Apply solution
6. **Verify** - Confirm issue is resolved
7. **Document** - Record solution for future

## Scenario 1: Disk Full Error (20 minutes)

### Problem

```
Error: OSError: [Errno 28] No space left on device
Training job failed while saving checkpoint to /var/ml/checkpoints/
```

### Investigation Steps

**Step 1: Check disk usage**
```bash
# Overall disk usage
df -h

# Check specific directory
df -h /var/ml/checkpoints

# Check inodes (sometimes full before disk)
df -i
```

**Step 2: Find large files**
```bash
# Top 10 largest directories
sudo du -h /var/ml | sort -rh | head -10

# Find files larger than 1GB
sudo find /var/ml -type f -size +1G -exec ls -lh {} \;

# Find recently created large files
sudo find /var/ml -type f -mtime -7 -size +100M -exec ls -lh {} \;
```

**Step 3: Identify old checkpoints**
```bash
# List checkpoints by age
ls -lht /var/ml/checkpoints/

# Count checkpoint files
find /var/ml/checkpoints -name "*.pt" | wc -l

# Total size of checkpoints
du -sh /var/ml/checkpoints/
```

### Solution

**Create cleanup script:**
```bash
cat > cleanup_checkpoints.sh << 'EOF'
#!/bin/bash
# Clean old ML checkpoints

CHECKPOINT_DIR="/var/ml/checkpoints"
RETENTION_DAYS=7

echo "Cleaning checkpoints older than $RETENTION_DAYS days..."

# Find and delete old checkpoints
find "$CHECKPOINT_DIR" -name "*.pt" -mtime +$RETENTION_DAYS -delete
find "$CHECKPOINT_DIR" -name "*.ckpt" -mtime +$RETENTION_DAYS -delete

# Keep only last 5 checkpoints per model
for model_dir in "$CHECKPOINT_DIR"/*; do
    if [ -d "$model_dir" ]; then
        ls -t "$model_dir"/checkpoint_*.pt 2>/dev/null | tail -n +6 | xargs -r rm
    fi
done

echo "Cleanup complete. Current usage:"
df -h "$CHECKPOINT_DIR"
du -sh "$CHECKPOINT_DIR"
EOF

chmod +x cleanup_checkpoints.sh
./cleanup_checkpoints.sh
```

**Prevention:**
```bash
# Set up automatic cleanup with cron
(crontab -l 2>/dev/null; echo "0 2 * * * /path/to/cleanup_checkpoints.sh") | crontab -

# Or use logrotate-style approach
# Configure model training to only keep N latest checkpoints
```

**Validation:**
- [ ] Freed sufficient disk space
- [ ] Training can save checkpoints again
- [ ] Cleanup script runs successfully

## Scenario 2: Permission Denied Error (15 minutes)

### Problem

```
PermissionError: [Errno 13] Permission denied: '/data/datasets/training.csv'
```

### Investigation

```bash
# Check file permissions
ls -l /data/datasets/training.csv

# Check directory permissions
ls -ld /data/datasets/

# Check ownership
stat /data/datasets/training.csv

# Check current user
whoami
id

# Check if file exists
[ -f /data/datasets/training.csv ] && echo "File exists" || echo "File not found"
```

### Solution

**Fix permissions:**
```bash
# Make readable by user
chmod 644 /data/datasets/training.csv

# Make readable by group
chmod 664 /data/datasets/training.csv
chgrp mlteam /data/datasets/training.csv

# Fix directory permissions (need execute to enter)
chmod 755 /data/datasets/

# Recursive fix for entire directory
chmod -R 755 /data/datasets/
find /data/datasets -type f -exec chmod 644 {} \;
```

**For shared datasets:**
```bash
# Set up proper group ownership
sudo groupadd mlteam
sudo usermod -a -G mlteam $USER

# Set directory for group collaboration
sudo chown -R :mlteam /data/datasets
sudo chmod -R 775 /data/datasets

# Set default permissions for new files
sudo chmod g+s /data/datasets  # setgid
```

**Validation:**
- [ ] Can read the file
- [ ] Training script runs successfully

## Scenario 3: Hung/Zombie Process (20 minutes)

### Problem

```
Training process appears frozen
htop shows process in 'D' (uninterruptible sleep) state
```

### Investigation

```bash
# Find the process
ps aux | grep python
ps aux | grep train

# Check process state
ps -eo pid,stat,cmd | grep train

# Check what process is doing
sudo strace -p <PID>

# Check open files
sudo lsof -p <PID>

# Check for I/O wait
iostat -x 1

# Check system load
uptime
```

### Solution

**Gentle termination:**
```bash
# Try SIGTERM first (graceful)
kill <PID>

# Wait 10 seconds
sleep 10

# Check if still running
ps -p <PID>

# If still running, SIGKILL
kill -9 <PID>
```

**For stuck I/O:**
```bash
# Check disk I/O
iostat -x 1 5

# Check if disk is full
df -h

# Check for NFS issues
mount | grep nfs
```

**Zombie processes:**
```bash
# Find zombies
ps aux | grep defunct

# Zombies can't be killed directly
# Kill parent process instead
ps -o ppid= -p <zombie_pid>
kill <parent_pid>
```

**Prevention script:**
```bash
cat > monitor_training.sh << 'EOF'
#!/bin/bash
# Monitor training process and restart if hung

PID_FILE="/var/run/training.pid"
TIMEOUT=300  # 5 minutes

while true; do
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")

        # Check if process is in D state
        STATE=$(ps -o state= -p "$PID" 2>/dev/null)

        if [ "$STATE" = "D" ]; then
            echo "Process $PID is stuck. Killing..."
            kill -9 "$PID"
            rm "$PID_FILE"
        fi
    fi

    sleep 60
done
EOF
```

**Validation:**
- [ ] Process terminated successfully
- [ ] System resources freed
- [ ] No zombie processes remain

## Scenario 4: Out of Memory (20 minutes)

### Problem

```
Killed
(Process terminated by OOM killer)
```

### Investigation

```bash
# Check memory usage
free -h

# Check what's using memory
ps aux --sort=-%mem | head -10

# Check OOM killer logs
sudo dmesg | grep -i "out of memory"
sudo journalctl | grep -i "oom"

# Check swap usage
swapon --show
```

### Solution

**Immediate fix:**
```bash
# Clear cache (safe)
sudo sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Check memory again
free -h

# Kill memory-hungry processes
# Find process
ps aux --sort=-%mem | head

# Kill it
kill -9 <PID>
```

**Long-term solutions:**
```bash
# 1. Add swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 2. Reduce batch size in training script
# Edit training config to use smaller batch size

# 3. Monitor memory usage
cat > memory_monitor.sh << 'EOF'
#!/bin/bash
# Alert when memory usage exceeds threshold

THRESHOLD=90

while true; do
    MEM_USAGE=$(free | grep Mem | awk '{print int($3/$2 * 100)}')

    if [ $MEM_USAGE -gt $THRESHOLD ]; then
        echo "WARNING: Memory usage at ${MEM_USAGE}%" | mail -s "Memory Alert" admin@company.com
    fi

    sleep 300
done
EOF
```

**Validation:**
- [ ] System has available memory
- [ ] Training process can run
- [ ] Monitoring in place

## Scenario 5: Network Connectivity Issues (20 minutes)

### Problem

```
ConnectionError: Unable to reach data lake at data.company.com:9000
```

### Investigation

```bash
# Check if host resolves
nslookup data.company.com
dig data.company.com

# Check connectivity
ping -c 4 data.company.com

# Check if port is open
nc -zv data.company.com 9000
telnet data.company.com 9000

# Check routing
traceroute data.company.com

# Check local firewall
sudo ufw status
sudo iptables -L -n | grep 9000

# Check if service is listening on remote
# (if you have SSH access)
ssh remote-server "sudo ss -tln | grep 9000"
```

### Solution

**DNS issues:**
```bash
# Add to /etc/hosts temporarily
echo "192.168.1.100 data.company.com" | sudo tee -a /etc/hosts

# Or fix DNS servers
echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolv.conf
```

**Firewall issues:**
```bash
# Allow outgoing connection
sudo ufw allow out 9000/tcp

# Or temporarily disable (testing only!)
sudo ufw disable
```

**Service not running:**
```bash
# Check service status
systemctl status data-service

# Start service
sudo systemctl start data-service
```

**Validation:**
- [ ] Can resolve hostname
- [ ] Can ping host
- [ ] Can connect to port
- [ ] Application works

## Scenario 6: GPU/CUDA Issues (25 minutes)

### Problem

```
RuntimeError: CUDA error: no kernel image is available for execution
```

### Investigation

```bash
# Check if GPU is detected
lspci | grep -i nvidia

# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"

# Check for driver/CUDA mismatch
nvidia-smi | grep "CUDA Version"
```

### Solution

**CUDA version mismatch:**
```bash
# Reinstall PyTorch with correct CUDA version
# For CUDA 11.8:
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

**Driver issues:**
```bash
# Reinstall NVIDIA driver
sudo apt remove --purge nvidia-*
sudo ubuntu-drivers autoinstall
sudo reboot

# After reboot
nvidia-smi
```

**GPU memory issues:**
```bash
# Check GPU memory
nvidia-smi

# Clear GPU memory in Python
# Add to training script:
import torch
torch.cuda.empty_cache()

# Or reduce batch size
```

**Validation:**
- [ ] nvidia-smi shows GPU
- [ ] PyTorch detects CUDA
- [ ] Training runs on GPU

## Troubleshooting Toolkit

**Create master troubleshooting script:**
```bash
cat > troubleshoot.sh << 'EOF'
#!/bin/bash
# System troubleshooting toolkit

echo "=== System Health Check ==="
echo ""

# Disk space
echo "[ Disk Space ]"
df -h | grep -E "(Filesystem|/dev/)"
echo ""

# Memory
echo "[ Memory Usage ]"
free -h
echo ""

# CPU Load
echo "[ CPU Load ]"
uptime
echo ""

# Top processes by memory
echo "[ Top Memory Consumers ]"
ps aux --sort=-%mem | head -6
echo ""

# Top processes by CPU
echo "[ Top CPU Consumers ]"
ps aux --sort=-%cpu | head -6
echo ""

# Network connectivity
echo "[ Network Status ]"
ip addr show | grep "inet " | grep -v "127.0.0.1"
echo ""

# Listening services
echo "[ Listening Services ]"
sudo ss -tln | grep LISTEN | head -5
echo ""

# Recent errors in syslog
echo "[ Recent Errors ]"
sudo journalctl -p err -n 5 --no-pager
echo ""

# GPU status (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "[ GPU Status ]"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo ""
fi

echo "Health check complete."
EOF

chmod +x troubleshoot.sh
./troubleshoot.sh
```

## Best Practices Summary

### Systematic Approach

✅ Follow consistent troubleshooting framework
✅ Gather information before making changes
✅ Test hypotheses systematically
✅ Document all changes
✅ Verify fixes completely

### Common Mistakes to Avoid

❌ Jumping to conclusions without investigation
❌ Making multiple changes at once
❌ Not checking logs
❌ Restarting without understanding root cause
❌ Not documenting solutions

### Prevention

✅ Set up monitoring and alerting
✅ Implement automated cleanup
✅ Use proper resource limits
✅ Regular health checks
✅ Document system configuration

### Documentation

✅ Keep runbook of common issues
✅ Document non-obvious fixes
✅ Note workarounds and their limitations
✅ Update after each incident
✅ Share knowledge with team

## Completion Checklist

### Skills Demonstrated
- [ ] Can diagnose disk space issues
- [ ] Can fix permission problems
- [ ] Can handle stuck processes
- [ ] Can troubleshoot memory issues
- [ ] Can resolve network connectivity problems
- [ ] Can fix GPU/CUDA issues
- [ ] Created troubleshooting toolkit
- [ ] Can follow systematic approach

### Scenarios Completed
- [ ] Scenario 1: Disk Full
- [ ] Scenario 2: Permission Denied
- [ ] Scenario 3: Hung Process
- [ ] Scenario 4: Out of Memory
- [ ] Scenario 5: Network Issues
- [ ] Scenario 6: GPU/CUDA Problems

## Next Steps

1. **Exercise 08: System Automation** - Automate maintenance to prevent issues
2. **Advanced Troubleshooting:**
   - Performance profiling
   - Distributed system debugging
   - Container troubleshooting

3. **Production Skills:**
   - Incident response procedures
   - Post-mortem analysis
   - Runbook development

## Resources

- [Linux Performance Tools](http://www.brendangregg.com/linuxperf.html)
- [SRE Book - Troubleshooting](https://sre.google/sre-book/effective-troubleshooting/)
- [NVIDIA GPU Troubleshooting](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

Congratulations! You can now troubleshoot common ML infrastructure issues.
