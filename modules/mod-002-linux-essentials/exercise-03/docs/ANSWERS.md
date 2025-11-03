# Exercise 03: Process Management - Reflection Questions

This document provides comprehensive answers to the reflection questions in Exercise 03: Process Management for ML Training Jobs.

---

## Question 1: When would you use kill -9 vs kill -15?

### Short Answer

- **Use `kill -15` (SIGTERM) first**: Allows graceful shutdown with cleanup
- **Use `kill -9` (SIGKILL) as last resort**: Force terminates without cleanup

### Detailed Explanation

#### kill -15 (SIGTERM - Signal 15)

**What it does:**
- Sends a termination request to the process
- Process **can catch and handle** the signal
- Allows cleanup: save checkpoints, close files, release resources
- Process can choose to ignore it (though most don't)

**When to use:**
```bash
# First attempt for ANY process termination
kill -15 <PID>
# or simply
kill <PID>  # SIGTERM is default
```

**ML Training example:**
```bash
# Training process with signal handling
kill -TERM 12345

# Process receives signal:
# 1. Saves current checkpoint
# 2. Writes final metrics
# 3. Closes data files
# 4. Releases GPU memory
# 5. Exits cleanly
```

#### kill -9 (SIGKILL - Signal 9)

**What it does:**
- **Immediately terminates** the process
- Process **cannot catch or ignore** it
- Kernel terminates it directly
- **No cleanup possible**

**When to use:**
```bash
# ONLY use when:
# 1. kill -15 didn't work after waiting
# 2. Process is completely hung
# 3. Process is in uninterruptible sleep
# 4. Emergency situations

# First try graceful:
kill -15 <PID>
sleep 10

# If still running, force kill:
if ps -p <PID> > /dev/null; then
    kill -9 <PID>
fi
```

**Risks of kill -9:**
- **Lost data**: Unsaved checkpoints gone
- **Corrupted files**: Partially written data
- **Resource leaks**: GPU memory not released
- **Zombie processes**: Children may become zombies
- **Lock files**: May leave stale locks

### Decision Flow Chart

```
Process needs to stop
       ↓
Try: kill -15 <PID>
       ↓
Wait 10-30 seconds
       ↓
Still running? ──NO──→ Success! (graceful shutdown)
       ↓
      YES
       ↓
Check if doing cleanup? (logs, etc.)
       ↓
      YES ──→ Wait longer (up to 2 minutes for ML)
       ↓
      NO
       ↓
Try: kill -9 <PID>
       ↓
Process terminated (forcefully)
```

### Real-World ML Examples

#### Scenario 1: Normal Training Stop

```bash
# Training running for 5 hours
$ ps aux | grep train
user  12345  95.2  35.0  python train.py

# Stop gracefully
$ kill -15 12345

# Process saves checkpoint at epoch 487
# Writes metrics to wandb
# Releases GPU memory
# Exits after 3 seconds
```

**Result:** Can resume from epoch 487 later ✓

#### Scenario 2: Hung Training Process

```bash
# Training appears stuck
$ ps aux | grep train
user  12345  0.0  35.0  python train.py  # 0% CPU!

# Try graceful first
$ kill -15 12345
$ sleep 10
$ ps -p 12345  # Still there

# Check state
$ ps aux | grep 12345
user  12345  0.0  35.0  D  # D = uninterruptible sleep

# Force kill (no choice)
$ kill -9 12345
```

**Result:** Lost checkpoint at current epoch ✗

#### Scenario 3: Proper Shutdown Workflow

```bash
# Our training script with signal handling
./manage_training.sh stop

# Internal logic:
# 1. Sends SIGTERM
# 2. Waits up to 10 seconds
# 3. Only uses SIGKILL if necessary
```

### Best Practices Summary

1. **Always try SIGTERM first**
   ```bash
   kill -TERM <PID>
   ```

2. **Wait appropriately**
   - Simple process: 5 seconds
   - ML training: 10-30 seconds
   - Database: 60+ seconds

3. **Check if cleanup is happening**
   ```bash
   # Watch CPU/disk activity
   watch -n 1 "ps -p <PID> -o %cpu,%mem,stat,time"
   ```

4. **Use wrapper scripts**
   ```bash
   # Handles the logic for you
   ./manage_training.sh stop
   ```

5. **SIGKILL only when:**
   - SIGTERM failed after waiting
   - Process completely unresponsive
   - Emergency resource recovery

### Signal Comparison Table

| Signal | Number | Can Catch? | Cleanup? | Use Case |
|--------|--------|------------|----------|----------|
| SIGTERM | 15 | Yes | Yes | Normal shutdown (try first) |
| SIGKILL | 9 | No | No | Force kill (last resort) |
| SIGINT | 2 | Yes | Yes | User interrupt (Ctrl+C) |
| SIGHUP | 1 | Yes | Yes | Hangup / reload config |
| SIGQUIT | 3 | Yes | Yes | Quit with core dump |

---

## Question 2: How can you monitor a training job after disconnecting from SSH?

### Short Answer

Use persistent terminal sessions (**tmux** or **screen**) and background processes with **nohup**. Check status via SSH reconnection or remote monitoring tools.

### Detailed Solutions

#### Solution 1: Tmux (Recommended)

**Why tmux?**
- Modern and actively maintained
- Better features than screen
- Split panes for simultaneous monitoring
- Easy to script
- Works great with SSH

**Basic workflow:**
```bash
# On remote server
ssh user@ml-server

# Start tmux session
tmux new -s training

# Start training
python train.py --epochs 1000

# Detach: Ctrl+b d
# Session continues running!

# Disconnect from SSH
exit

# Later: Reconnect to SSH
ssh user@ml-server

# Reattach to session
tmux attach -t training

# See training still running!
```

**Advanced multi-pane setup:**
```bash
# Create session with monitoring layout
tmux new -s ml_monitor

# Split horizontally
Ctrl+b "

# Navigate to bottom pane
Ctrl+b ↓

# Split bottom pane vertically
Ctrl+b %

# Now you have 3 panes:
# Top (full width): Training output
# Bottom-left: GPU monitoring
# Bottom-right: Resource monitoring

# In top pane:
python train.py

# In bottom-left:
nvidia-smi -l 1

# In bottom-right:
htop

# Detach: Ctrl+b d
# Everything keeps running!
```

**Tmux Commands:**
```bash
# Create named session
tmux new -s experiment_001

# List sessions
tmux ls

# Attach to session
tmux attach -t experiment_001

# Kill session
tmux kill-session -t experiment_001

# Detach (from inside)
Ctrl+b d

# Send commands to detached session
tmux send-keys -t experiment_001 "echo 'status check'" C-m
```

#### Solution 2: Screen

**Alternative to tmux (older but ubiquitous):**
```bash
# Start screen session
screen -S training

# Start training
python train.py

# Detach: Ctrl+a d

# Disconnect SSH
exit

# Reconnect later
ssh user@ml-server
screen -r training
```

**Screen commands:**
```bash
# Create session
screen -S name

# List sessions
screen -ls

# Reattach
screen -r name

# Detach
Ctrl+a d

# Kill session
screen -X -S name quit
```

#### Solution 3: Nohup with Background Processes

**For processes that don't need interactive monitoring:**
```bash
# Start with nohup
nohup python train.py > training.log 2>&1 &

# Note the PID
echo $! > training.pid

# Disconnect
exit

# Later: Check if still running
ssh user@ml-server
cat training.pid
ps -p $(cat training.pid)

# View logs
tail -f training.log
```

#### Solution 4: Systemd Services (Production)

**Best for always-running services:**
```bash
# Create service file
sudo nano /etc/systemd/system/ml-training.service
```

```ini
[Unit]
Description=ML Training Service
After=network.target

[Service]
Type=simple
User=mluser
WorkingDirectory=/opt/ml/training
ExecStart=/usr/bin/python3 train.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable ml-training
sudo systemctl start ml-training

# Check status anytime
sudo systemctl status ml-training

# View logs
journalctl -u ml-training -f
```

### Complete Monitoring Workflow

#### Initial Setup (One Time)

```bash
# On remote server
ssh user@ml-server

# Install monitoring tools
sudo apt install tmux htop

# Create workspace
mkdir -p ~/experiments/exp001
cd ~/experiments/exp001

# Create monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== Training Status ==="
    date
    echo ""
    tail -5 training.log
    echo ""
    echo "=== GPU ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    sleep 10
done
EOF

chmod +x monitor.sh
```

#### Start Training Session

```bash
# Start tmux
tmux new -s exp001

# Split for monitoring
Ctrl+b "  # horizontal split

# Top pane: training
python train.py --config config.yaml | tee training.log

# Bottom pane: monitoring
Ctrl+b ↓
./monitor.sh

# Detach
Ctrl+b d

# Disconnect
exit
```

#### Check Status Later

```bash
# Reconnect
ssh user@ml-server

# Quick check (without attaching)
tmux capture-pane -t exp001 -p | tail -10

# Full reattach
tmux attach -t exp001

# Or just check logs
tail -f ~/experiments/exp001/training.log
```

### Remote Monitoring Tools

#### Solution 5: Web-Based Dashboards

**TensorBoard:**
```bash
# Start TensorBoard
tensorboard --logdir runs/ --port 6006 --bind_all &

# Access from local machine
ssh -L 6006:localhost:6006 user@ml-server

# Open browser: http://localhost:6006
```

**Weights & Biases:**
```python
# In training script
import wandb

wandb.init(project="my-experiment")

for epoch in range(epochs):
    # Training...
    wandb.log({"loss": loss, "accuracy": acc})

# Monitor from anywhere: https://wandb.ai
```

**MLflow:**
```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 &

# Access remotely
ssh -L 5000:localhost:5000 user@ml-server
# Browse: http://localhost:5000
```

#### Solution 6: Log File Monitoring

**Stream logs to local machine:**
```bash
# Real-time log streaming
ssh user@ml-server "tail -f ~/experiments/exp001/training.log"

# With grep for errors
ssh user@ml-server "tail -f ~/experiments/exp001/training.log | grep ERROR"
```

**Automated status checks:**
```bash
# Create local monitoring script
cat > check_training.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    ssh user@ml-server "tail -3 ~/experiments/exp001/training.log"
    echo ""
    sleep 60
done
EOF

chmod +x check_training.sh
./check_training.sh
```

### Comparison of Methods

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Tmux** | Modern, feature-rich, split panes | Requires installation | Interactive monitoring |
| **Screen** | Ubiquitous, simple | Older, fewer features | Basic sessions |
| **Nohup** | Simple, no dependencies | No interactivity | Fire-and-forget |
| **Systemd** | Auto-restart, logging | Requires sudo setup | Production services |
| **TensorBoard** | Visual, real-time metrics | Requires instrumentation | Metrics tracking |
| **W&B/MLflow** | Cloud-based, team sharing | Internet required | Team projects |

### Best Practices

1. **Always use persistent sessions for long training**
   ```bash
   # Good
   tmux new -s training
   python train.py

   # Bad
   python train.py  # Lost if SSH disconnects
   ```

2. **Name sessions meaningfully**
   ```bash
   tmux new -s resnet50_imagenet_run003
   ```

3. **Log everything**
   ```bash
   python train.py 2>&1 | tee training.log
   ```

4. **Set up monitoring before starting**
   ```bash
   # Start monitoring first
   tmux new -s monitor
   watch -n 10 nvidia-smi

   # Then start training in another session
   tmux new -s training
   python train.py
   ```

5. **Use automated status updates**
   ```python
   # Send status emails/slack messages
   import smtplib

   def notify(message):
       # Send email/slack on checkpoints
       pass

   for epoch in range(epochs):
       train_epoch()
       if epoch % 10 == 0:
           notify(f"Epoch {epoch} complete")
   ```

### Our Implementation

**Using our scripts:**
```bash
# Launch in persistent session
./launch_training.sh exp001 train_model.py tmux

# Detach automatically after launch
# Reconnect anytime
tmux attach -t exp001

# Or use wrapper
./manage_training.sh start  # Uses nohup
./manage_training.sh status # Check without attaching
./manage_training.sh log    # Stream logs
```

---

## Question 3: What's the benefit of using systemd for ML services?

### Short Answer

Systemd provides **automatic restart**, **resource limits**, **logging**, **dependency management**, and **lifecycle control** - essential for production ML services that need to run reliably 24/7.

### Key Benefits

#### 1. Automatic Restart on Failure

**Problem without systemd:**
```bash
# ML inference service crashes at 3am
# Stays down until someone manually restarts
# Lost revenue, angry users
```

**Solution with systemd:**
```ini
[Service]
ExecStart=/usr/bin/python3 /opt/ml/serve.py
Restart=on-failure
RestartSec=10
```

**What happens:**
- Service crashes → systemd automatically restarts it in 10 seconds
- Keeps trying indefinitely
- No manual intervention needed

**Example:**
```bash
# Service crashes
$ sudo systemctl status ml-inference
● ml-inference.service - ML Inference API
   Active: failed (Result: exit-code)

# 10 seconds later
$ sudo systemctl status ml-inference
● ml-inference.service - ML Inference API
   Active: active (running) since Mon 2025-10-31 03:00:15
```

#### 2. Resource Limits (Prevent Runaway Processes)

**Problem:**
```python
# Memory leak in inference service
# Gradually consumes all system memory
# Eventually triggers OOM killer
# Takes down other services too
```

**Solution:**
```ini
[Service]
# Limit to 8GB RAM
MemoryLimit=8G

# Limit to 4 CPUs (400%)
CPUQuota=400%

# Limit open files
LimitNOFILE=10000
```

**What happens:**
- Service tries to exceed 8GB → killed and restarted
- Prevents taking down entire system
- OOM killer won't run randomly

**Real-world example:**
```bash
# Before systemd limits
$ free -h
               total        used        free
Mem:            32G         31G          1G  # Service leaked all memory!

# With systemd limits
$ free -h
               total        used        free
Mem:            32G         12G         20G  # Limited to 8G, plenty free

# Service was restarted when it hit limit
$ journalctl -u ml-inference | grep "exceeded"
Nov 01 03:15:00 server systemd[1]: ml-inference.service: Process exceeded memory limit
```

#### 3. Centralized Logging

**Problem without systemd:**
```bash
# Logs scattered everywhere
/var/log/ml_service.log
/tmp/inference.log
/opt/ml/logs/predictions.log
~/debug.log

# Difficult to aggregate
# No standardization
# Logs get lost
```

**Solution with systemd:**
```ini
[Service]
StandardOutput=journal
StandardError=journal
```

**Benefits:**
```bash
# All logs in one place
journalctl -u ml-inference

# Filter by time
journalctl -u ml-inference --since "1 hour ago"

# Follow logs
journalctl -u ml-inference -f

# Combine with system logs
journalctl -u ml-inference -u nginx

# Export to JSON
journalctl -u ml-inference -o json

# Filter by priority
journalctl -u ml-inference -p err
```

**Integrated monitoring:**
```bash
# Check for errors in last hour
journalctl -u ml-inference --since "1 hour ago" -p err | wc -l

# Alert if > 10 errors
if [ $(journalctl -u ml-inference --since "1 hour ago" -p err | wc -l) -gt 10 ]; then
    alert "Too many errors in ml-inference"
fi
```

#### 4. Dependency Management

**Problem:**
```bash
# ML service needs:
# 1. Network available
# 2. Database running
# 3. Redis cache running
# 4. GPU initialized

# If started too early → crashes
# Manual ordering is fragile
```

**Solution:**
```ini
[Unit]
Description=ML Inference Service
After=network.target postgresql.service redis.service nvidia-persistenced.service
Requires=postgresql.service redis.service

[Service]
ExecStart=/usr/bin/python3 /opt/ml/serve.py
```

**What happens:**
- Systemd waits for dependencies before starting
- Automatically stops if dependencies fail
- Proper startup order guaranteed

**Dependency chain example:**
```
network.target
    ↓
postgresql.service
    ↓
redis.service
    ↓
nvidia-persistenced.service
    ↓
ml-inference.service  ← Only starts when all ready
```

#### 5. Lifecycle Management

**Standard control interface:**
```bash
# Start
sudo systemctl start ml-inference

# Stop
sudo systemctl stop ml-inference

# Restart
sudo systemctl restart ml-inference

# Reload config (no downtime)
sudo systemctl reload ml-inference

# Enable at boot
sudo systemctl enable ml-inference

# Disable
sudo systemctl disable ml-inference

# Status
sudo systemctl status ml-inference

# Check if active
systemctl is-active ml-inference
```

**Scripting and automation:**
```bash
# Deploy script
sudo systemctl stop ml-inference
cp new_model.h5 /opt/ml/models/
sudo systemctl start ml-inference

# Health check
if ! systemctl is-active --quiet ml-inference; then
    alert "ML service is down!"
fi
```

#### 6. Boot-Time Startup

**Problem:**
```bash
# Server reboots (power failure, maintenance)
# ML services don't auto-start
# Manual startup needed
# Downtime until someone notices
```

**Solution:**
```bash
sudo systemctl enable ml-inference
```

**What happens:**
- Service starts automatically on boot
- No manual intervention
- Minimal downtime
- Works with cron jobs, scripts, containers

#### 7. Environment Management

**Clean environment setup:**
```ini
[Service]
# Set environment variables
Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="PYTHONUNBUFFERED=1"
Environment="MODEL_PATH=/opt/ml/models/production"
EnvironmentFile=/etc/ml/config.env

# Set working directory
WorkingDirectory=/opt/ml

# Run as specific user
User=mluser
Group=mlgroup
```

**Benefits:**
- Consistent environment
- Secrets management via EnvironmentFile
- Proper user isolation
- No conflicts with other services

### Real-World ML Use Cases

#### Use Case 1: Model Inference API

**Scenario:** FastAPI service serving model predictions

**Without systemd:**
```bash
# Manual management
cd /opt/ml
nohup python serve.py &

# Issues:
# - Crashes at night → stays down
# - No resource limits → memory leaks
# - Logs go to nohup.out → hard to find
# - Doesn't start on reboot
```

**With systemd:**
```ini
[Unit]
Description=ML Model Inference API
After=network.target

[Service]
Type=simple
User=mluser
WorkingDirectory=/opt/ml
ExecStart=/usr/bin/python3 /opt/ml/serve.py
Restart=on-failure
RestartSec=5

# Resource limits
MemoryLimit=8G
CPUQuota=400%

# Logging
StandardOutput=journal
StandardError=journal

# Environment
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="MODEL_PATH=/opt/ml/models/v1.2.3.h5"

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable ml-inference
sudo systemctl start ml-inference

# Benefits:
# ✓ Auto-restarts on crash
# ✓ Starts on boot
# ✓ Resource limited
# ✓ Logs in journalctl
# ✓ Easy to manage
```

#### Use Case 2: Batch Processing Service

**Scenario:** Periodic batch inference every hour

**Without systemd:**
```bash
# Cron job
0 * * * * /opt/ml/batch_process.py

# Issues:
# - No logging
# - Overlapping runs if slow
# - No resource control
# - Hard to monitor
```

**With systemd (timer):**
```ini
# /etc/systemd/system/ml-batch.service
[Unit]
Description=ML Batch Processing

[Service]
Type=oneshot
User=mluser
ExecStart=/usr/bin/python3 /opt/ml/batch_process.py
MemoryLimit=16G

# /etc/systemd/system/ml-batch.timer
[Unit]
Description=ML Batch Processing Timer

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable ml-batch.timer
sudo systemctl start ml-batch.timer

# Benefits:
# ✓ No overlapping runs
# ✓ Centralized logging
# ✓ Resource limits
# ✓ Easy status checks
```

#### Use Case 3: Model Training Pipeline

**Scenario:** Long-running distributed training

```ini
[Unit]
Description=Distributed Training Coordinator
After=network.target

[Service]
Type=simple
User=mluser
WorkingDirectory=/opt/ml/training
ExecStart=/usr/bin/python3 coordinator.py
Restart=no  # Don't auto-restart training

# Limit resources
CPUQuota=1600%  # 16 CPUs
MemoryLimit=128G

# GPU setup
Environment="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"

[Install]
WantedBy=multi-user.target
```

### Systemd vs Alternatives

| Method | Auto-Restart | Resource Limits | Logging | Boot Startup | Dependencies |
|--------|--------------|-----------------|---------|--------------|--------------|
| **systemd** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **supervisord** | ✓ | ✗ | ✓ | Manual | ✗ |
| **docker** | ✓ | ✓ | ✓ | ✓ | ✗ |
| **kubernetes** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **nohup** | ✗ | ✗ | ✗ | ✗ | ✗ |
| **screen/tmux** | ✗ | ✗ | ✗ | ✗ | ✗ |

### Complete Production Example

**File:** `/etc/systemd/system/ml-inference-api.service`

```ini
[Unit]
Description=ML Inference API (ResNet50 Image Classification)
Documentation=https://docs.example.com/ml-api
After=network-online.target postgresql.service redis.service
Wants=network-online.target
Requires=postgresql.service redis.service

[Service]
Type=simple
User=mluser
Group=mlgroup
WorkingDirectory=/opt/ml/inference

# Main process
ExecStart=/opt/ml/venv/bin/python3 /opt/ml/inference/serve.py

# Health check
ExecStartPost=/bin/sleep 5
ExecStartPost=/usr/bin/curl -f http://localhost:8000/health || exit 1

# Graceful shutdown
ExecStop=/bin/kill -TERM $MAINPID
TimeoutStopSec=30

# Restart policy
Restart=on-failure
RestartSec=10
StartLimitInterval=5min
StartLimitBurst=3

# Resource limits
CPUQuota=400%          # 4 CPUs
MemoryLimit=8G         # 8GB RAM
TasksMax=100           # Max 100 threads
LimitNOFILE=10000      # Max open files

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true

# Environment
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="MODEL_PATH=/opt/ml/models/resnet50_v1.2.3.h5"
Environment="LOG_LEVEL=INFO"
EnvironmentFile=-/etc/ml/inference.env

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ml-inference

[Install]
WantedBy=multi-user.target
```

**Usage:**
```bash
# Deploy
sudo cp ml-inference-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ml-inference-api
sudo systemctl start ml-inference-api

# Check status
sudo systemctl status ml-inference-api

# View logs
journalctl -u ml-inference-api -f

# Restart (deployment)
sudo systemctl restart ml-inference-api

# Stop (maintenance)
sudo systemctl stop ml-inference-api
```

### Best Practices Summary

1. **Always use systemd for production ML services**
2. **Set resource limits** to prevent runaway processes
3. **Enable automatic restart** with backoff
4. **Use journal logging** for centralized logs
5. **Declare dependencies** explicitly
6. **Enable services** to auto-start on boot
7. **Set timeouts** appropriately (especially for ML)
8. **Use EnvironmentFile** for secrets
9. **Run as non-root user** for security
10. **Document your service** with Description field

---

## Question 4: How would you limit CPU/memory for a training process?

### Short Answer

Use **systemd resource limits**, **cgroups**, **nice/renice** for CPU priority, **ulimit** for per-user limits, or container resource limits. For active processes, use **cpulimit** or **renice**.

### Method 1: Systemd Resource Limits (Recommended for Services)

#### CPU Limits

**Limit to specific percentage:**
```ini
[Service]
# Limit to 200% (2 CPUs worth)
CPUQuota=200%

# Or limit to specific cores
CPUAffinity=0 1 2 3  # Use cores 0-3 only
```

**Example service:**
```ini
[Unit]
Description=ML Training (Resource Limited)

[Service]
Type=simple
ExecStart=/usr/bin/python3 /opt/ml/train.py

# CPU limits
CPUQuota=400%              # Max 4 CPUs
CPUAffinity=0 1 2 3        # Use cores 0-3

# Priority
Nice=10                     # Lower priority (higher nice value)

[Install]
WantedBy=multi-user.target
```

```bash
# Start training with limits
sudo systemctl start ml-training

# Verify CPU usage stays under limit
top -p $(systemctl show -p MainPID ml-training | cut -d= -f2)
```

#### Memory Limits

**Hard and soft limits:**
```ini
[Service]
# Hard limit: Kill if exceeded
MemoryLimit=8G

# Soft limit with margin
MemoryHigh=7G  # Slow down at 7G
MemoryMax=8G    # Kill at 8G

# Swap limit
MemorySwapMax=2G
```

**Complete example:**
```ini
[Unit]
Description=ML Training with Memory Limits

[Service]
ExecStart=/usr/bin/python3 /opt/ml/train.py

# Memory limits
MemoryHigh=14G    # Start throttling at 14GB
MemoryMax=16G     # Hard kill at 16GB
MemorySwapMax=4G  # Allow 4GB swap

# What happens:
# - < 14GB: Normal operation
# - 14-16GB: Throttled (slow down)
# - > 16GB: Process killed and restarted

[Install]
WantedBy=multi-user.target
```

#### Combined CPU and Memory

**Production training service:**
```ini
[Service]
ExecStart=/usr/bin/python3 train.py --batch-size 32

# CPU limits
CPUQuota=800%              # Max 8 CPUs
CPUAffinity=0-7            # Cores 0-7
Nice=5                      # Slightly lower priority

# Memory limits
MemoryHigh=30G
MemoryMax=32G

# Other limits
TasksMax=200               # Max threads/processes
LimitNOFILE=10000         # Max open files

Restart=on-failure
```

### Method 2: Nice and Renice (CPU Priority)

#### Understanding Nice Values

```
Nice value range: -20 (highest priority) to +19 (lowest priority)
Default: 0

Lower nice = Higher priority = More CPU time
Higher nice = Lower priority = Less CPU time
```

#### Start Process with Nice

```bash
# Start with lower priority (+10)
nice -n 10 python train.py

# Start with higher priority (-10, requires sudo)
sudo nice -n -10 python train.py

# Check nice value
ps -eo pid,ni,cmd | grep train
```

#### Change Priority of Running Process

```bash
# Find PID
PID=$(pgrep -f train.py)

# Lower priority (anyone can do this)
renice +10 $PID

# Higher priority (requires sudo)
sudo renice -10 $PID

# Verify
ps -p $PID -o pid,ni,cmd
```

**Real-world example:**
```bash
# High-priority inference service
$ ps -p $(pgrep inference) -o pid,ni,cmd
  PID  NI CMD
 1234 -10 python inference.py

# Background training (low priority)
$ ps -p $(pgrep train) -o pid,ni,cmd
  PID  NI CMD
 5678  15 python train.py

# Inference gets CPU time first
# Training only uses idle CPU
```

### Method 3: Cpulimit (Active CPU Limiting)

**Install:**
```bash
sudo apt install cpulimit
```

**Limit running process:**
```bash
# Limit PID to 50% of one CPU
cpulimit -p 12345 -l 50

# Limit by name to 200% (2 CPUs)
cpulimit -e python -l 200

# Run command with limit
cpulimit -l 100 -- python train.py
```

**Background limiting:**
```bash
# Start training
python train.py &
TRAIN_PID=$!

# Limit it in background
cpulimit -p $TRAIN_PID -l 200 -b

# Training runs at max 2 CPUs
# cpulimit runs in background monitoring it
```

**Wrapper script:**
```bash
#!/bin/bash
# limited_train.sh

# Start training in background
python train.py &
TRAIN_PID=$!

# Apply CPU limit
cpulimit -p $TRAIN_PID -l 400 -b  # Max 4 CPUs

# Wait for training to complete
wait $TRAIN_PID
```

### Method 4: Cgroups (Direct Control)

**Create cgroup for training:**
```bash
# Create cgroup
sudo cgcreate -g cpu,memory:/ml_training

# Set CPU limit (50% of all CPUs)
sudo cgset -r cpu.cfs_quota_us=500000 ml_training
sudo cgset -r cpu.cfs_period_us=1000000 ml_training

# Set memory limit (16GB)
sudo cgset -r memory.limit_in_bytes=17179869184 ml_training

# Run process in cgroup
sudo cgexec -g cpu,memory:/ml_training python train.py
```

**Verify limits:**
```bash
# Check CPU usage
cat /sys/fs/cgroup/cpu/ml_training/cpu.stat

# Check memory usage
cat /sys/fs/cgroup/memory/ml_training/memory.usage_in_bytes
```

### Method 5: Ulimit (Per-User/Process Limits)

**Set limits for current shell:**
```bash
# Limit memory (in KB)
ulimit -m 16777216  # 16GB

# Limit CPU time (in seconds)
ulimit -t 86400  # 24 hours

# Limit processes
ulimit -u 500

# Limit file size
ulimit -f 10485760  # 10GB

# Show all limits
ulimit -a

# Start training with limits
python train.py
```

**Make permanent (in ~/.bashrc or /etc/security/limits.conf):**
```bash
# /etc/security/limits.conf
mluser  soft  as       16777216  # 16GB virtual memory
mluser  hard  as       17825792  # 17GB hard limit
mluser  soft  cpu      86400     # 24 hours CPU time
mluser  hard  cpu      172800    # 48 hours hard limit
mluser  soft  nproc    500       # Max 500 processes
mluser  hard  nproc    1000
```

### Method 6: Docker/Container Limits

**Run training in container with limits:**
```bash
# CPU limit: 4 CPUs
# Memory limit: 16GB
docker run \
    --cpus=4 \
    --memory=16g \
    --memory-swap=20g \
    ml-training:latest \
    python train.py
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  training:
    image: ml-training:latest
    command: python train.py
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 16G
        reservations:
          cpus: '2.0'
          memory: 8G
```

### Method 7: Kubernetes Resource Limits

**Pod with resource limits:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training
spec:
  containers:
  - name: training
    image: ml-training:latest
    command: ["python", "train.py"]
    resources:
      requests:
        memory: "8Gi"
        cpu: "4"
      limits:
        memory: "16Gi"
        cpu: "8"
```

### Comparison Table

| Method | CPU Control | Memory Control | Ease of Use | Granularity | Production Ready |
|--------|-------------|----------------|-------------|-------------|------------------|
| **systemd** | ✓✓✓ | ✓✓✓ | ✓✓ | High | ✓✓✓ |
| **nice/renice** | ✓✓ (priority) | ✗ | ✓✓✓ | Low | ✓✓ |
| **cpulimit** | ✓✓✓ | ✗ | ✓✓✓ | Medium | ✓✓ |
| **cgroups** | ✓✓✓ | ✓✓✓ | ✓ | High | ✓✓✓ |
| **ulimit** | ✓ | ✓ | ✓✓ | Low | ✓ |
| **docker** | ✓✓✓ | ✓✓✓ | ✓✓ | High | ✓✓✓ |
| **kubernetes** | ✓✓✓ | ✓✓✓ | ✓ | High | ✓✓✓ |

### Real-World Scenarios

#### Scenario 1: Shared GPU Server

**Problem:** Multiple users training on same server, need fair CPU/memory allocation

**Solution:**
```bash
# User 1: High-priority small model (4 CPUs, 8GB)
sudo cgcreate -g cpu,memory:/user1_training
sudo cgset -r cpu.cfs_quota_us=400000 user1_training  # 4 CPUs
sudo cgset -r memory.limit_in_bytes=8589934592 user1_training  # 8GB
sudo cgexec -g cpu,memory:/user1_training su - user1 -c "python train_small.py"

# User 2: Low-priority large model (8 CPUs, 16GB)
sudo cgcreate -g cpu,memory:/user2_training
sudo cgset -r cpu.cfs_quota_us=800000 user2_training  # 8 CPUs
sudo cgset -r memory.limit_in_bytes=17179869184 user2_training  # 16GB
sudo cgset -r cpu.shares=512 user2_training  # Lower priority
sudo cgexec -g cpu,memory:/user2_training su - user2 -c "python train_large.py"
```

#### Scenario 2: Production Inference + Background Training

**Problem:** Inference must be responsive, training can use leftover resources

**Solution:**
```bash
# High-priority inference (4 CPUs, always available)
sudo systemctl start ml-inference.service
```

```ini
# /etc/systemd/system/ml-inference.service
[Service]
ExecStart=/usr/bin/python3 inference.py
CPUQuota=400%
MemoryMax=8G
Nice=-10  # High priority
```

```bash
# Low-priority training (uses idle CPUs only)
nice -n 19 python train.py &
```

**Result:**
- Inference always responsive (high priority)
- Training uses idle CPU (low priority)
- Training yields CPU when inference needs it

#### Scenario 3: Preventing Memory OOM

**Problem:** Training with large batch size causes OOM killer to randomly kill processes

**Solution:**
```ini
[Service]
ExecStart=/usr/bin/python3 train.py --batch-size 32

# Soft limit with warning zone
MemoryHigh=30G    # Throttle at 30GB
MemoryMax=32G     # Kill at 32GB

# What happens:
# - < 30GB: Normal speed
# - 30-32GB: Slowed down (pressure, gives warning)
# - > 32GB: Killed cleanly by systemd (not OOM killer)

Restart=on-failure
RestartSec=10
```

**Benefits:**
- Controlled termination (not random OOM killer)
- Other processes protected
- Can restart with smaller batch size
- Logs show why it was killed

### Monitoring Resource Limits

**Check current usage:**
```bash
# CPU and memory usage
ps -p $PID -o pid,%cpu,%mem,vsz,rss,cmd

# systemd service usage
systemctl status ml-training

# cgroup usage
cat /sys/fs/cgroup/cpu/ml_training/cpu.stat
cat /sys/fs/cgroup/memory/ml_training/memory.usage_in_bytes

# Continuous monitoring
top -p $PID
htop -p $PID
```

**Alerts when approaching limits:**
```bash
#!/bin/bash
# monitor_limits.sh

PID=$1
MEM_LIMIT_GB=16

while true; do
    MEM_MB=$(ps -p $PID -o rss= | awk '{print $1/1024}')
    MEM_GB=$(echo "scale=2; $MEM_MB/1024" | bc)

    if (( $(echo "$MEM_GB > $MEM_LIMIT_GB * 0.9" | bc -l) )); then
        echo "WARNING: Memory usage ${MEM_GB}GB approaching limit ${MEM_LIMIT_GB}GB"
        # Send alert
    fi

    sleep 60
done
```

### Best Practices

1. **Use systemd for production services**
   - Most integrated
   - Best resource control
   - Automatic restart with limits

2. **Combine CPU priority and limits**
   ```ini
   Nice=10          # Lower priority
   CPUQuota=400%    # Hard limit
   ```

3. **Always set memory limits**
   - Prevents OOM killer from randomly killing processes
   - Controlled termination
   - Protection for other services

4. **Monitor resource usage**
   - Alert before hitting limits
   - Adjust batch size dynamically
   - Log resource usage for optimization

5. **Test limits before production**
   - Run with limits in dev/staging
   - Verify training completes
   - Adjust as needed

### Our Implementation

**Using our scripts with limits:**
```bash
# Method 1: Simple nice priority
nice -n 10 ./manage_training.sh start

# Method 2: cpulimit wrapper
./manage_training.sh start
cpulimit -p $(cat training.pid) -l 400 -b

# Method 3: systemd service (best)
sudo cp ml-training.service /etc/systemd/system/
sudo systemctl start ml-training
```

---

## Question 5: What tools would you use to diagnose a hung process?

### Short Answer

Use **ps** to check state, **lsof** for open files/network, **strace** to trace system calls, **/proc** filesystem for detailed info, **gdb** for debugging, and **perf** for performance analysis. Start simple (ps, top) and move to advanced tools (strace, gdb) as needed.

### Diagnostic Process Flow

```
Process hung?
    ↓
1. Check basic state (ps, top)
    ↓
2. Check what it's waiting for (lsof, /proc)
    ↓
3. Trace system calls (strace)
    ↓
4. Debug if code issue (gdb, pdb)
    ↓
5. Analyze performance (perf)
```

### Tool 1: ps - Process State

**First step: Check process state**

```bash
# Find the process
PID=$(pgrep -f train.py)

# Check state
ps -p $PID -o pid,stat,wchan,cmd
```

**Output interpretation:**
```
PID   STAT  WCHAN      CMD
12345 S     poll_sc... python train.py   # Sleeping (normal)
12345 D     io_sche... python train.py   # Uninterruptible sleep (I/O)
12345 R     -          python train.py   # Running (normal)
12345 T     do_signa... python train.py  # Stopped
```

**STAT column meanings:**
- `S`: Sleeping (waiting for event) - **Normal**
- `D`: Uninterruptible sleep - **Stuck on I/O**
- `R`: Running - **Working fine**
- `T`: Stopped (Ctrl+Z) - **User stopped**
- `Z`: Zombie - **Parent didn't reap**
- `R+`: Running in foreground - **Normal**

**WCHAN shows what it's waiting for:**
- `poll_schedule_timeout`: Waiting for network/event
- `io_schedule`: Waiting for disk I/O
- `wait_woken`: Waiting to be woken
- `do_wait`: Waiting for child process
- `-`: Not waiting (actively running)

**Detailed state check:**
```bash
ps -p $PID -o pid,ppid,stat,wchan,%cpu,%mem,etime,cmd
```

### Tool 2: top/htop - Real-time Monitoring

**Check if process is active:**
```bash
# Monitor specific PID
top -p $PID

# Better: htop
htop -p $PID
```

**Key indicators:**
- **CPU 0%**: Not doing anything → hung
- **CPU 100%**: Working (or infinite loop)
- **High %MEM, CPU 0%**: Loaded data, waiting
- **STATE D**: Waiting for I/O

### Tool 3: /proc Filesystem - Detailed Info

**Most detailed low-level information:**

```bash
PID=12345

# Process status
cat /proc/$PID/status

# Stack trace (requires root)
sudo cat /proc/$PID/stack

# Command line
cat /proc/$PID/cmdline | tr '\0' ' '

# Environment
cat /proc/$PID/environ | tr '\0' '\n'

# Current working directory
readlink /proc/$PID/cwd

# Open file descriptors
ls -l /proc/$PID/fd/

# Memory maps
cat /proc/$PID/maps

# I/O statistics
cat /proc/$PID/io
```

**Check if stuck on I/O:**
```bash
# Shows kernel stack
sudo cat /proc/$PID/stack

# Example output if stuck on disk I/O:
[<0>] io_schedule+0x16/0x40
[<0>] wait_on_page_bit+0x120/0x2a0
[<0>] filemap_fault+0x450/0x990
[<0>] __do_fault+0x38/0x150
```

### Tool 4: lsof - Open Files and Network

**What files/sockets is process using?**

```bash
# All open files
lsof -p $PID

# Just network connections
lsof -i -a -p $PID

# Just regular files
lsof -p $PID -a -d ^0,1,2  # Exclude stdin/stdout/stderr

# Check if waiting on specific file
lsof -p $PID | grep dataset.h5
```

**Real examples:**

**Hung on NFS:**
```bash
$ lsof -p 12345 | grep nfs
python  12345 user  10r   REG   0,48  5G  /mnt/nfs/model.h5
```
→ Waiting for NFS mount

**Stuck network connection:**
```bash
$ lsof -i -a -p 12345
python 12345 user  8u  IPv4  ESTABLISHED  192.168.1.10:8080->10.0.0.5:34567
```
→ Waiting for network response

**Open file count:**
```bash
# Too many open files?
lsof -p $PID | wc -l
ulimit -n  # Compare to limit
```

### Tool 5: strace - System Call Tracing

**Most powerful: See exactly what process is doing**

```bash
# Attach to running process
sudo strace -p $PID

# With timestamps
sudo strace -tt -p $PID

# Count system calls
sudo strace -c -p $PID

# Trace specific calls only
sudo strace -e trace=read,write,open,close -p $PID

# Output to file
sudo strace -o trace.log -p $PID
```

**Example outputs:**

**Hung on network:**
```bash
$ sudo strace -p 12345
recvfrom(8, ...                         # Stuck here
^C
# → Waiting for network data
```

**Busy loop (100% CPU but not progressing):**
```bash
$ sudo strace -c -p 12345
# Wait 10 seconds, Ctrl+C
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 99.99    5.234567          10    523456           gettimeofday
# → Calling gettimeofday in tight loop
```

**Disk I/O wait:**
```bash
$ sudo strace -p 12345
read(10, ...                            # Stuck here
# → Waiting for disk read
```

**Deadlock on lock:**
```bash
$ sudo strace -p 12345
futex(0x7f8a4c0021d0, FUTEX_WAIT, ...  # Stuck here
# → Waiting for mutex/lock
```

### Tool 6: gdb - Debugger

**Attach to Python process:**

```bash
# Attach gdb
sudo gdb -p $PID

# Get Python backtrace
(gdb) py-bt

# Get C backtrace
(gdb) bt

# Continue execution
(gdb) continue

# Detach
(gdb) detach
(gdb) quit
```

**Example: Find where Python is stuck:**
```
(gdb) py-bt
Traceback (most recent call first):
  File "train.py", line 245, in load_data
    data = np.load(filename)
  File "train.py", line 367, in train_epoch
    batch = load_data(batch_files)
  File "train.py", line 445, in train
    train_epoch(epoch)
```
→ Stuck loading data file

### Tool 7: pdb - Python Debugger

**For Python specifically:**

```bash
# Send SIGUSR1 to activate debugger (if configured)
kill -SIGUSR1 $PID

# Or inject pdb
python -c "
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')
import gdb
gdb.execute(f'attach {PID}')
gdb.execute('py-bt')
"
```

### Tool 8: perf - Performance Analysis

**CPU profiling:**

```bash
# Profile for 10 seconds
sudo perf record -p $PID sleep 10

# View report
sudo perf report

# Find hot functions
sudo perf top -p $PID
```

**Example output:**
```
54.32%  python  libpython3.so  [.] PyEval_EvalFrameDefault
12.45%  python  libc.so        [.] memcpy
 8.23%  python  numpy.so       [.] PyArray_MatMul
```
→ Shows where CPU time is spent

### Tool 9: iostat - I/O Statistics

**Check if system I/O is the problem:**

```bash
# I/O stats every 2 seconds
iostat -x 2

# High %util → disk bottleneck
Device   r/s   w/s   rkB/s   wkB/s  %util
sda     1234    56   12345    5678     98
# → Disk saturated at 98%
```

### Tool 10: netstat/ss - Network Connections

**Check network status:**

```bash
# All connections for PID
netstat -anp | grep $PID

# Or with ss (faster)
ss -anp | grep $PID

# Check for ESTABLISHED connections
ss -anp | grep $PID | grep ESTABLISHED

# Check for TIME_WAIT (connection closed but lingering)
ss -anp | grep $PID | grep TIME_WAIT
```

**Example hung on network:**
```bash
$ ss -anp | grep 12345
ESTABLISHED  0   0   10.0.0.5:34567  192.168.1.10:8080  users:(("python",pid=12345,fd=8))
# → Connection established but no data transfer
```

### Complete Diagnostic Workflow

#### Step-by-Step Investigation

```bash
#!/bin/bash
# diagnose_hung_process.sh

PID=$1

echo "=== Process Diagnostic ==="
echo ""

# 1. Basic info
echo "[1] Basic Process Info:"
ps -p $PID -o pid,ppid,stat,wchan,%cpu,%mem,etime,cmd
echo ""

# 2. State check
STATE=$(ps -p $PID -o stat --no-headers | tr -d ' ')
echo "[2] Process State: $STATE"
case $STATE in
    *D*) echo "    ⚠ Uninterruptible sleep - likely I/O wait" ;;
    *R*) echo "    ✓ Running normally" ;;
    *S*) echo "    ℹ Sleeping - waiting for event" ;;
    *T*) echo "    ⚠ Stopped (Ctrl+Z?)" ;;
    *Z*) echo "    ✗ Zombie process" ;;
esac
echo ""

# 3. Kernel stack
echo "[3] Kernel Stack (what it's waiting for):"
if [ -r "/proc/$PID/stack" ]; then
    sudo cat "/proc/$PID/stack" || echo "    Permission denied"
else
    echo "    N/A"
fi
echo ""

# 4. Open files
echo "[4] Open Files (first 10):"
lsof -p $PID 2>/dev/null | head -10 || echo "    lsof not available"
echo ""

# 5. Network connections
echo "[5] Network Connections:"
lsof -i -a -p $PID 2>/dev/null || echo "    None or lsof not available"
echo ""

# 6. I/O statistics
echo "[6] I/O Statistics:"
if [ -r "/proc/$PID/io" ]; then
    cat "/proc/$PID/io"
else
    echo "    Permission denied"
fi
echo ""

# 7. System call trace (5 seconds)
echo "[7] System Call Trace (5 sec sample):"
echo "    Running: sudo strace -c -p $PID"
sudo timeout 5 strace -c -p $PID 2>&1 | tail -15 || echo "    strace failed"
echo ""

# 8. Recommendations
echo "=== Recommendations ==="
if echo "$STATE" | grep -q D; then
    echo "• Process stuck on I/O"
    echo "  → Check disk: iostat -x 1"
    echo "  → Check NFS mounts: df -h | grep nfs"
    echo "  → Check network: ss -anp | grep $PID"
elif echo "$STATE" | grep -q S; then
    echo "• Process sleeping (normal for waiting processes)"
    echo "  → Check if it's making progress: watch -n 1 'ls -lh output*'"
    echo "  → Attach strace: sudo strace -p $PID"
else
    echo "• Run full strace: sudo strace -tt -o trace.log -p $PID"
    echo "• Attach debugger: sudo gdb -p $PID"
fi
```

### Real-World Scenarios

#### Scenario 1: Hung on Dataset Loading

**Symptoms:**
- CPU: 0%
- State: D (uninterruptible sleep)
- Not progressing

**Diagnosis:**
```bash
$ ps -p 12345 -o stat,wchan
D    io_schedule

$ sudo cat /proc/12345/stack
[<0>] io_schedule
[<0>] wait_on_page_bit
[<0>] filemap_fault

$ lsof -p 12345 | grep .h5
python 12345 user 10r REG /data/dataset.h5
```

**Root cause:** Slow disk I/O or NFS mount issue

**Solution:**
- Check disk: `iostat -x 1`
- Check NFS: `df -h`
- Consider local SSD
- Prefetch data

#### Scenario 2: Deadlock on Lock

**Symptoms:**
- CPU: 0%
- State: S (sleeping)
- Multiple threads stuck

**Diagnosis:**
```bash
$ sudo strace -p 12345
futex(0x7f8a4c0021d0, FUTEX_WAIT, 2, NULL

$ sudo gdb -p 12345
(gdb) info threads
  Id   Target Id         Frame
* 1    Thread 0x7f8a (LWP 12345) futex_wait
  2    Thread 0x7f8b (LWP 12346) futex_wait

(gdb) py-bt
Waiting for lock at dataset.py:123
```

**Root cause:** Deadlock between threads

**Solution:**
- Fix locking logic
- Use timeout on locks
- Review concurrent access patterns

#### Scenario 3: Infinite Loop (100% CPU, Not Progressing)

**Symptoms:**
- CPU: 100%
- State: R (running)
- No output

**Diagnosis:**
```bash
$ sudo strace -c -p 12345
# After 10 seconds:
% time     calls    syscall
------ --------- ----------------
 99.99   5234567   gettimeofday

$ sudo perf top -p 12345
54.32%  _check_convergence
23.12%  _compute_loss
```

**Root cause:** Infinite loop in convergence check

**Solution:**
- Add max iterations
- Add progress logging
- Fix convergence logic

### Tools Comparison

| Tool | Use Case | Requires Root | Invasive | Output Readability |
|------|----------|---------------|----------|-------------------|
| **ps** | First check | No | No | ✓✓✓ |
| **top** | Real-time | No | No | ✓✓✓ |
| **/proc** | Detailed info | Sometimes | No | ✓✓ |
| **lsof** | File/network | Sometimes | No | ✓✓ |
| **strace** | System calls | Yes | Yes | ✓✓ |
| **gdb** | Debugging | Yes | Yes | ✓ |
| **perf** | Profiling | Yes | Minimal | ✓✓ |
| **iostat** | I/O stats | No | No | ✓✓ |

### Our Implementation

**Using our diagnose script:**
```bash
# Quick diagnosis
./diagnose_process.sh 12345

# Shows:
# - Process state
# - What it's waiting for
# - Open files
# - Network connections
# - Recommendations
```

### Best Practices

1. **Start simple**: ps, top, /proc
2. **Check state first**: D = I/O, S = waiting, R = running
3. **Use strace for system calls**: Most direct view
4. **Check open files**: Often reveals the bottleneck
5. **Profile if CPU is high**: perf, gdb
6. **Monitor over time**: Hung vs slow
7. **Have debugging tools ready**: Install strace, gdb, perf beforehand

---

## Question 6: Why is graceful shutdown important for ML training?

### Short Answer

Graceful shutdown **saves training progress** (checkpoints), **preserves resources** (files, GPU memory), **maintains data integrity**, and enables **training resumption** without data loss. Without it, hours or days of training can be lost.

### The Cost of Ungraceful Shutdown

#### Scenario: Training Without Checkpoints

```python
# BAD: No checkpoint saving
for epoch in range(1000):  # 100 hours of training
    train_epoch()
    # No checkpoint saving

# Server crash at epoch 487
# Result: Lost 48.7 hours of training 💸
```

**Financial impact:**
- 8x V100 GPUs: $24/hour
- Lost 48.7 hours: $24 × 48.7 = **$1,169 lost**

#### Scenario: With Graceful Shutdown

```python
# GOOD: Regular checkpoints + signal handling
import signal

def signal_handler(signum, frame):
    save_checkpoint(current_epoch)  # Save before exit
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

for epoch in range(1000):
    train_epoch()
    if epoch % 10 == 0:
        save_checkpoint(epoch)

# Server crash at epoch 487
# Checkpoint saved at epoch 480
# Resume from epoch 480
# Lost: 7 epochs = 0.7 hours
# Cost: $24 × 0.7 = $16.80 lost ✓
```

### Key Benefits of Graceful Shutdown

#### 1. Save Training Checkpoints

**What gets saved:**
- Model weights
- Optimizer state
- Learning rate schedule
- Random state (reproducibility)
- Current epoch number
- Training metrics history

**Example:**
```python
def graceful_shutdown(signum, frame):
    checkpoint = {
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': best_loss,
        'accuracy': best_accuracy,
        'random_state': torch.get_rng_state(),
        'timestamp': datetime.now()
    }
    torch.save(checkpoint, f'checkpoint_epoch_{current_epoch}.pt')
    print(f"Checkpoint saved at epoch {current_epoch}")
    sys.exit(0)
```

**Resume training:**
```python
# Load checkpoint
checkpoint = torch.load('checkpoint_epoch_480.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
torch.set_rng_state(checkpoint['random_state'])
start_epoch = checkpoint['epoch'] + 1

# Continue training
for epoch in range(start_epoch, total_epochs):
    train_epoch()
```

#### 2. Release GPU Memory Properly

**Without graceful shutdown:**
```bash
# Force kill (SIGKILL)
kill -9 <PID>

# GPU memory NOT released
$ nvidia-smi
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  Tesla V100-SXM2...  On   | 00000000:00:04.0 Off |                    0 |
| GPU Memory Usage   Free: 0MiB / Used: 16384MiB | Total: 16384MiB         |
# → Memory still allocated (zombie allocation)
# → Next job can't use GPU
# → Requires nvidia-smi -r (reset) or reboot
```

**With graceful shutdown:**
```python
def cleanup():
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Close file handles
    for handle in open_files:
        handle.close()

    print("Cleanup complete")

signal.signal(signal.SIGTERM, lambda sig, frame: (cleanup(), sys.exit(0)))
```

```bash
# Graceful termination
kill -15 <PID>

# GPU memory properly released
$ nvidia-smi
| GPU Memory Usage   Free: 16384MiB / Used: 0MiB | Total: 16384MiB         |
# → Memory freed, available for next job ✓
```

#### 3. Maintain Data Integrity

**File corruption risk:**

```python
# Writing checkpoint when killed
f = open('checkpoint.pt', 'wb')
f.write(large_checkpoint_data)  # ← SIGKILL here
# File partially written = corrupted ✗
```

**With graceful shutdown:**
```python
def save_checkpoint_atomic(data, path):
    # Write to temp file first
    temp_path = path + '.tmp'
    torch.save(data, temp_path)

    # Atomic rename (either completes or doesn't)
    os.rename(temp_path, path)
    # ✓ File is always in valid state

def signal_handler(signum, frame):
    save_checkpoint_atomic(checkpoint, 'checkpoint.pt')
    sys.exit(0)
```

#### 4. Log Final Metrics

**Without graceful shutdown:**
```python
for epoch in range(1000):
    loss, acc = train_epoch()
    metrics.append((epoch, loss, acc))

# SIGKILL here
# Metrics in memory → lost ✗
# No record of training progress
```

**With graceful shutdown:**
```python
def signal_handler(signum, frame):
    # Write metrics to file
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Log to wandb/mlflow
    wandb.log({"final_epoch": current_epoch, "reason": "interrupted"})

    sys.exit(0)

# Now have complete training history ✓
```

#### 5. Clean Up Distributed Training

**Multi-GPU/multi-node training:**

```python
import torch.distributed as dist

def graceful_shutdown(signum, frame):
    # Save checkpoint
    if dist.get_rank() == 0:  # Only rank 0 saves
        save_checkpoint()

    # Synchronize all processes
    dist.barrier()

    # Clean up process group
    dist.destroy_process_group()

    sys.exit(0)

# All processes shut down cleanly
# No hanging processes on other nodes
# Process group cleaned up properly
```

### Real-World Impact Examples

#### Example 1: ImageNet Training (8 GPUs, 90 epochs)

**Without graceful shutdown:**
```
Training time: 1 week
Cost: $24/hour × 8 GPUs × 168 hours = $32,256
Power failure at epoch 75
No checkpoint → Start over
Lost: $26,880 (83% of cost)
```

**With graceful shutdown:**
```
Training time: 1 week
Checkpoint every 5 epochs
Power failure at epoch 75
Resume from epoch 75
Lost: $0 ✓
```

#### Example 2: LLM Fine-tuning (4xA100, 100B parameters)

**Without graceful shutdown:**
```
Model: 100B parameters = 200GB checkpoint
Training time: 2 weeks
Cost: $40/hour × 4 GPUs × 336 hours = $53,760
OOM kill at 78% complete
No checkpoint → Start over
Lost: $41,933
```

**With graceful shutdown:**
```
Regular checkpoints (every hour)
OOM detected → SIGTERM sent
Graceful shutdown saves checkpoint at 78%
Resume from 78%
Lost: < $100 (< 1 hour)
```

#### Example 3: Research Experiment (Single GPU)

**Without graceful shutdown:**
```
Grad student training overnight
Morning: check results
Process killed (out of memory)
No checkpoint saved
Result: No data for paper 📉
```

**With graceful shutdown:**
```
Process receives SIGTERM before OOM kill
Checkpoint saved at current state
Partial results available
Can analyze what worked
Can resume with adjusted config
Result: Progress towards paper ✓
```

### Implementation Patterns

#### Pattern 1: Simple Signal Handler

```python
#!/usr/bin/env python3
import signal
import sys

def train():
    for epoch in range(1000):
        train_epoch()
        if epoch % 10 == 0:
            save_checkpoint(epoch)

def graceful_shutdown(signum, frame):
    print(f"\nReceived signal {signum}")
    print(f"Saving checkpoint at epoch {current_epoch}")
    save_checkpoint(current_epoch)
    print("Shutdown complete")
    sys.exit(0)

# Register handlers
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

if __name__ == "__main__":
    train()
```

#### Pattern 2: Context Manager

```python
import atexit
from contextlib import contextmanager

@contextmanager
def training_session(checkpoint_path):
    """Context manager for training with automatic cleanup"""

    def cleanup():
        print("Cleaning up...")
        save_checkpoint(checkpoint_path)
        torch.cuda.empty_cache()

    # Register cleanup
    atexit.register(cleanup)

    # Register signal handlers
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    yield

    # Normal exit cleanup
    cleanup()

# Usage
with training_session('checkpoint.pt'):
    for epoch in range(1000):
        train_epoch()
```

#### Pattern 3: Training Class

```python
class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.current_epoch = 0
        self.shutdown_requested = False

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\nShutdown requested (signal {signum})")
        self.shutdown_requested = True

    def train(self, epochs):
        for epoch in range(epochs):
            if self.shutdown_requested:
                print("Graceful shutdown in progress...")
                self.save_checkpoint()
                break

            self.current_epoch = epoch
            self.train_epoch()

            if epoch % 10 == 0:
                self.save_checkpoint()

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, f'checkpoint_{self.current_epoch}.pt')
        print(f"✓ Checkpoint saved at epoch {self.current_epoch}")
```

### Best Practices

#### 1. Always Handle SIGTERM and SIGINT

```python
signal.signal(signal.SIGTERM, graceful_shutdown)  # kill, systemctl stop
signal.signal(signal.SIGINT, graceful_shutdown)   # Ctrl+C
```

#### 2. Save Checkpoints Atomically

```python
# Write to temp file, then atomic rename
temp_path = checkpoint_path + '.tmp'
torch.save(checkpoint, temp_path)
os.rename(temp_path, checkpoint_path)  # Atomic on POSIX
```

#### 3. Regular Checkpoints + Signal Handler

```python
# Regular checkpoints (every N epochs)
if epoch % 10 == 0:
    save_checkpoint()

# Plus signal handler for unexpected shutdown
signal.signal(signal.SIGTERM, save_and_exit)
```

#### 4. Test Your Shutdown Logic

```bash
# Start training
python train.py &
PID=$!

# Wait a bit
sleep 10

# Send SIGTERM
kill -TERM $PID

# Verify checkpoint was saved
ls -l checkpoint_*.pt
```

#### 5. Log Shutdown Reason

```python
def graceful_shutdown(signum, frame):
    reason = {
        signal.SIGTERM: "SIGTERM received",
        signal.SIGINT: "User interrupt (Ctrl+C)"
    }

    logging.info(f"Shutdown: {reason[signum]} at epoch {current_epoch}")
    save_checkpoint()
    sys.exit(0)
```

### Comparison: Graceful vs Force Kill

| Aspect | Graceful (SIGTERM) | Force (SIGKILL) |
|--------|-------------------|-----------------|
| **Checkpoint saved** | ✓ Yes | ✗ No |
| **GPU memory freed** | ✓ Yes | ✗ Often no |
| **File integrity** | ✓ Guaranteed | ✗ Risk of corruption |
| **Metrics logged** | ✓ Yes | ✗ Lost |
| **Resume training** | ✓ Easy | ✗ Start over |
| **Distributed cleanup** | ✓ Clean | ✗ Hanging processes |
| **Time to shutdown** | Few seconds | Immediate |
| **Data loss** | Minimal | Complete |

### Our Implementation

**Our training simulator demonstrates this:**

```python
# train_model.py
def graceful_shutdown(self, signum, frame):
    print(f"\n[{self.timestamp()}] Received signal {signum}")
    print(f"Saving checkpoint at epoch {self.current_epoch}...")
    self.save_checkpoint(reason="shutdown")
    print("Shutdown complete")
    sys.exit(0)

# Register handlers
signal.signal(signal.SIGTERM, self.graceful_shutdown)
signal.signal(signal.SIGINT, self.graceful_shutdown)
```

**Usage:**
```bash
# Start training
./manage_training.sh start

# Stop gracefully (SIGTERM)
./manage_training.sh stop

# Checkpoint saved ✓
# Can resume later ✓
```

---

## Summary

These six questions cover the essential aspects of process management for ML infrastructure:

1. **Signal handling**: Understanding when to use graceful (SIGTERM) vs force (SIGKILL) termination
2. **Remote monitoring**: Using persistent sessions (tmux/screen) and monitoring tools for long-running jobs
3. **Service management**: Leveraging systemd for production ML services with auto-restart and resource limits
4. **Resource control**: Limiting CPU/memory to prevent resource exhaustion and ensure fair sharing
5. **Diagnostics**: Using proper tools (ps, lsof, strace, gdb) to diagnose hung or problematic processes
6. **Graceful shutdown**: Preserving training progress and resources through proper signal handling

Mastering these concepts enables you to:
- Run reliable, long-running ML training jobs
- Efficiently share GPU servers among teams
- Quickly diagnose and resolve process issues
- Minimize wasted compute resources
- Operate production ML services with high availability

---

**Exercise 03: Process Management - Complete with Comprehensive Answers**
