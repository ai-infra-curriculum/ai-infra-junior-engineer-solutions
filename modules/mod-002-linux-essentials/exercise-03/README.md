# Exercise 03: Process Management for ML Training Jobs - Solution

## Overview

This solution demonstrates comprehensive Linux process management in the context of ML infrastructure. It covers process monitoring, control, GPU management, system services, persistent sessions, and troubleshooting - all essential skills for managing ML training workloads.

## Learning Objectives Covered

- ✅ Monitor running processes with ps, top, and htop
- ✅ Manage process lifecycle (start, stop, pause, resume, kill)
- ✅ Use job control for background and foreground processes
- ✅ Monitor GPU processes with nvidia-smi
- ✅ Manage system services with systemctl
- ✅ Handle stuck or runaway ML training processes
- ✅ Implement resource monitoring for ML workloads
- ✅ Use screen and tmux for persistent training sessions

## Solution Structure

```
exercise-03/
├── README.md                          # This file - solution overview
├── IMPLEMENTATION_GUIDE.md            # Step-by-step implementation guide
├── scripts/                           # Process management scripts
│   ├── train_model.py                 # ML training simulator
│   ├── manage_training.sh             # Training process manager
│   ├── monitor_resources.sh           # Resource usage monitor
│   ├── analyze_resources.py           # Resource analysis tool
│   ├── monitor_gpu.sh                 # GPU monitoring script
│   ├── gpu_process_manager.sh         # GPU process management
│   ├── launch_training.sh             # Launch training in persistent session
│   ├── diagnose_process.sh            # Process diagnostic tool
│   └── validate_exercise.sh           # Exercise validation
├── examples/                          # Example scenarios
│   ├── process_references/            # Command references
│   ├── service_examples/              # Systemd service examples
│   └── troubleshooting_scenarios/     # Troubleshooting guides
└── docs/
    └── ANSWERS.md                     # Reflection question answers
```

## Key Concepts

### Process States

```
R = Running
S = Sleeping (waiting for event)
D = Uninterruptible sleep (usually I/O)
T = Stopped
Z = Zombie (terminated but not reaped)
< = High priority
N = Low priority
```

### Common Process Commands

| Command | Purpose |
|---------|---------|
| `ps aux` | Show all processes |
| `top` | Real-time process monitoring |
| `htop` | Interactive process viewer |
| `kill -15 PID` | Graceful terminate (SIGTERM) |
| `kill -9 PID` | Force kill (SIGKILL) |
| `jobs` | List background jobs |
| `fg %1` | Bring job to foreground |
| `bg %1` | Resume job in background |
| `nohup command &` | Run immune to hangups |

### Signal Reference

| Signal | Number | Description |
|--------|--------|-------------|
| SIGHUP | 1 | Hangup (reload config) |
| SIGINT | 2 | Interrupt (Ctrl+C) |
| SIGQUIT | 3 | Quit (with core dump) |
| SIGKILL | 9 | Force kill (cannot be caught) |
| SIGTERM | 15 | Terminate gracefully (default) |
| SIGCONT | 18 | Continue if stopped |
| SIGSTOP | 19 | Stop process (cannot be caught) |
| SIGTSTP | 20 | Stop (Ctrl+Z) |

### ML Training Process Workflow

```
1. Start training in persistent session (screen/tmux)
2. Monitor resources (CPU, memory, GPU)
3. Detach from session (continue in background)
4. Reattach to check progress
5. Gracefully stop if needed (save checkpoint)
6. Force kill only as last resort
```

## Quick Start

### 1. Set Up Workspace

```bash
cd scripts
chmod +x *.sh *.py
```

### 2. Start ML Training Simulation

```bash
./manage_training.sh start
```

This starts a simulated ML training process with:
- 60 epochs
- Checkpoint every 10 epochs
- Graceful shutdown handling
- PID file tracking

### 3. Monitor Training Process

```bash
# Check status
./manage_training.sh status

# Monitor resources
./monitor_resources.sh

# View logs
./manage_training.sh log
```

### 4. Stop Training Gracefully

```bash
./manage_training.sh stop
```

This sends SIGTERM, allows checkpoint saving, then force kills if needed.

## Process Management Commands

### Viewing Processes

```bash
# All processes
ps aux

# Sort by CPU usage
ps aux --sort=-%cpu | head -10

# Sort by memory usage
ps aux --sort=-%mem | head -10

# Process tree
ps auxf
pstree

# Specific process
ps -p <PID> -o pid,ppid,%cpu,%mem,etime,cmd

# Find Python processes
ps aux | grep python | grep -v grep
pgrep -a python
```

### Monitoring Processes

```bash
# Real-time monitoring
top

# Better alternative
htop

# Batch mode (for scripts)
top -b -n 1 > snapshot.txt

# Monitor specific process
top -p <PID>

# Monitor specific user
top -u username
```

### Process Control

```bash
# Start in background
./script.sh &

# List jobs
jobs
jobs -l  # with PIDs

# Bring to foreground
fg %1

# Suspend (Ctrl+Z)
# Resume in background
bg %1

# Run immune to hangups
nohup ./script.sh &

# Run in persistent session
screen -S training
tmux new -s training
```

### Killing Processes

```bash
# Graceful termination
kill -15 <PID>
kill -TERM <PID>

# Force kill (last resort)
kill -9 <PID>
kill -KILL <PID>

# Kill all matching processes
killall python
pkill -f "train.py"

# Kill by user
pkill -u username
```

### Changing Priority

```bash
# Lower priority (higher nice value)
nice -n 10 ./train.py
renice +10 <PID>

# Higher priority (requires sudo)
sudo renice -10 <PID>

# View nice values
ps -eo pid,ni,cmd
```

## GPU Process Management

### NVIDIA GPU Monitoring

```bash
# Show GPU status
nvidia-smi

# Continuous monitoring
nvidia-smi -l 1

# Process monitoring
nvidia-smi pmon

# Custom query
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv

# Query GPU processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### Kill GPU Process

```bash
# Find GPU processes
nvidia-smi

# Kill gracefully
kill -15 <PID>

# Force kill if needed
kill -9 <PID>
```

### Our GPU Scripts

```bash
# Monitor GPU
./monitor_gpu.sh

# List GPU processes
./gpu_process_manager.sh list

# Kill GPU process
./gpu_process_manager.sh kill <PID>
```

## System Services with systemctl

### Service Management

```bash
# Check service status
systemctl status docker

# Start service
sudo systemctl start docker

# Stop service
sudo systemctl stop docker

# Restart service
sudo systemctl restart docker

# Reload configuration
sudo systemctl reload docker

# Enable at boot
sudo systemctl enable docker

# Disable at boot
sudo systemctl disable docker
```

### Service Information

```bash
# List all units
systemctl list-units

# List failed units
systemctl list-units --failed

# Check if active
systemctl is-active docker

# Check if enabled
systemctl is-enabled docker
```

### Service Logs

```bash
# View service logs
journalctl -u docker

# Follow logs
journalctl -u docker -f

# Last 50 lines
journalctl -u docker -n 50

# Recent logs
journalctl --since "1 hour ago"
journalctl --since today
```

## Persistent Sessions

### Screen Commands

```bash
# Start named session
screen -S training

# Detach (inside screen)
Ctrl+a d

# List sessions
screen -ls

# Reattach
screen -r training

# Kill session
screen -X -S training quit

# Multiple windows
Ctrl+a c    # Create window
Ctrl+a n    # Next window
Ctrl+a p    # Previous window
Ctrl+a "    # List windows
```

### Tmux Commands

```bash
# Start named session
tmux new -s training

# Detach (inside tmux)
Ctrl+b d

# List sessions
tmux ls

# Reattach
tmux attach -t training

# Kill session
tmux kill-session -t training

# Split panes
Ctrl+b %    # Vertical split
Ctrl+b "    # Horizontal split
Ctrl+b arrow # Navigate panes
```

### Launch Training Script

```bash
# Launch in tmux
./launch_training.sh experiment001 train.py tmux

# Launch in screen
./launch_training.sh experiment001 train.py screen

# Reattach
tmux attach -t experiment001
# or
screen -r experiment001
```

## Troubleshooting

### Process Won't Stop

```bash
# 1. Try graceful shutdown
kill -TERM <PID>
sleep 10

# 2. Check if still running
ps -p <PID>

# 3. Force kill
kill -9 <PID>

# 4. Check if zombie
ps aux | grep <PID>
# If Z in STAT column, kill parent:
ps -o ppid= -p <PID>
kill -9 <PPID>
```

### High CPU Usage

```bash
# Find CPU hogs
ps aux --sort=-%cpu | head -10
top -o %CPU

# Lower priority
renice +10 <PID>

# Limit CPU
cpulimit -p <PID> -l 50  # 50% of one CPU

# Kill if needed
kill -15 <PID>
```

### Out of Memory

```bash
# Check for OOM kills
dmesg | grep -i "killed process"
journalctl -k | grep -i "out of memory"

# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Identify memory hog
top -o %MEM
```

### Process Stuck

```bash
# Check state
ps aux | grep <PID>
# Look for D (uninterruptible sleep)

# Check what it's waiting for
sudo cat /proc/<PID>/stack

# Check I/O wait
iostat -x 1

# Diagnose with our tool
./diagnose_process.sh <PID>
```

### Can't Find Process

```bash
# Search by name
pgrep -a python
ps aux | grep train

# Search by port
lsof -i :8080
netstat -tulpn | grep 8080

# Search by file
lsof /path/to/file
```

## Real-World ML Scenarios

### Scenario 1: Long-Running Training Job

**Requirement**: Train model for 3 days without losing progress if SSH disconnects.

**Solution**:
```bash
# Start in tmux
tmux new -s deep-training
python train.py --epochs 1000 --checkpoint-interval 10
# Ctrl+b d to detach

# Check progress later
tmux attach -t deep-training

# Monitor resources from outside
./monitor_resources.sh
```

### Scenario 2: Multiple Experiments

**Requirement**: Run 5 experiments in parallel, each on different GPU.

**Solution**:
```bash
# Launch experiments
for i in {0..4}; do
    tmux new -d -s "exp$i"
    tmux send-keys -t "exp$i" "export CUDA_VISIBLE_DEVICES=$i" C-m
    tmux send-keys -t "exp$i" "python train.py --config exp$i.yaml" C-m
done

# List all experiments
tmux ls

# Monitor specific experiment
tmux attach -t exp0

# Check GPU usage
nvidia-smi
```

### Scenario 3: Training Process Hung

**Requirement**: Training stopped progressing, need to diagnose and restart.

**Solution**:
```bash
# Find training process
pgrep -a python | grep train

# Diagnose
./diagnose_process.sh <PID>

# Check if waiting on I/O
iostat -x 1

# Check GPU
nvidia-smi

# If truly hung, kill and restart
kill -TERM <PID>
sleep 10
kill -9 <PID>  # if needed

# Restart from checkpoint
python train.py --resume checkpoint_epoch_50.h5
```

### Scenario 4: Resource Monitoring

**Requirement**: Track resource usage throughout training for optimization.

**Solution**:
```bash
# Start monitoring before training
./monitor_resources.sh &
MONITOR_PID=$!

# Start training
./manage_training.sh start

# Let it run...
sleep 3600

# Stop monitoring
kill $MONITOR_PID

# Analyze
./analyze_resources.py resource_usage.csv
```

### Scenario 5: Graceful Shutdown

**Requirement**: Stop training but save checkpoint first.

**Solution**:
```bash
# Find training PID
PID=$(cat training.pid)

# Send SIGTERM (triggers checkpoint save in our train_model.py)
kill -TERM $PID

# Wait for graceful shutdown
for i in {1..30}; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "Shutdown complete"
        break
    fi
    sleep 1
done

# Force kill if still running
if ps -p $PID > /dev/null 2>&1; then
    kill -9 $PID
fi

# Verify checkpoint was saved
ls -lh checkpoint_epoch_*.json
```

## Testing the Solution

### 1. Test Training Manager

```bash
cd scripts

# Start training
./manage_training.sh start

# Check status
./manage_training.sh status

# View logs
./manage_training.sh log

# Stop training
./manage_training.sh stop
```

### 2. Test Resource Monitoring

```bash
# Start training
./manage_training.sh start

# Monitor in another terminal
./monitor_resources.sh

# Analyze results
./analyze_resources.py resource_usage.csv
```

### 3. Test GPU Monitoring (if GPU available)

```bash
# Monitor GPU
./monitor_gpu.sh

# List GPU processes
./gpu_process_manager.sh list
```

### 4. Test Persistent Sessions

```bash
# Launch training in tmux
./launch_training.sh test_exp train_model.py tmux

# List sessions
tmux ls

# Attach
tmux attach -t test_exp

# Detach: Ctrl+b d
```

### 5. Test Process Diagnostics

```bash
# Start a process
./manage_training.sh start
PID=$(cat ml_training_sim/training.pid)

# Diagnose
./diagnose_process.sh $PID

# Stop
./manage_training.sh stop
```

### 6. Run Validation

```bash
./validate_exercise.sh
```

## Integration with Previous Exercises

- **Exercise 01 (Navigation)**: Uses navigation skills for directory management
- **Exercise 02 (Permissions)**: Applies proper permissions to scripts and logs
- **Uses file operations**: Creating, monitoring, and managing process artifacts

## Skills Acquired

- ✅ Process monitoring and analysis
- ✅ Process lifecycle management
- ✅ Signal handling and graceful shutdown
- ✅ Job control for background tasks
- ✅ GPU process monitoring
- ✅ System service management
- ✅ Persistent session management
- ✅ Resource monitoring and analysis
- ✅ Process troubleshooting and recovery
- ✅ ML-specific process patterns

## Common Issues and Solutions

### Issue 1: Script Not Executable

**Symptom**: `Permission denied` when running script

**Solution**:
```bash
chmod +x script.sh
```

### Issue 2: Can't Find Training PID

**Symptom**: `training.pid` file not found

**Solution**:
```bash
# Find Python processes
pgrep -a python

# Or search for train.py
ps aux | grep train.py
```

### Issue 3: nvidia-smi Not Found

**Symptom**: GPU scripts fail

**Solution**:
- Skip GPU sections if no NVIDIA GPU available
- Or install NVIDIA drivers

### Issue 4: Process Still Running After Kill

**Symptom**: Process survives `kill -15`

**Solution**:
```bash
# Use force kill
kill -9 <PID>

# Check if zombie
ps aux | grep <PID>
# If zombie, kill parent
```

### Issue 5: Screen/Tmux Not Installed

**Symptom**: Commands not found

**Solution**:
```bash
sudo apt install screen tmux
```

## Performance Considerations

### Process Monitoring Overhead

- `ps` has minimal overhead (one-time snapshot)
- `top` uses ~1-2% CPU (continuous monitoring)
- `htop` similar to top but more user-friendly
- Custom monitoring scripts: depends on interval and metrics

### Best Practices

1. **Use appropriate monitoring intervals**:
   - Fast processes: 1-2 seconds
   - Long-running: 5-10 seconds
   - Historical tracking: 30-60 seconds

2. **Choose the right signal**:
   - Try SIGTERM first (allows cleanup)
   - Use SIGKILL only if SIGTERM fails
   - Never use SIGKILL as first choice for ML training

3. **Persistent sessions**:
   - Use tmux/screen for jobs > 1 hour
   - Always detach, don't close terminal
   - Name sessions meaningfully

4. **Resource monitoring**:
   - Monitor before and during training
   - Track over time for optimization
   - Set alerts for resource exhaustion

## Time to Complete

- **Setup and understanding**: 20 minutes
- **Implementing process monitoring**: 30 minutes
- **Implementing training management**: 40 minutes
- **Testing GPU monitoring**: 20 minutes
- **Testing persistent sessions**: 20 minutes
- **Troubleshooting practice**: 30 minutes
- **Total**: 120-160 minutes

## Next Steps

- Complete Exercise 04: Shell Scripting - Automate process management
- Complete Exercise 05: Package Management - Install ML software stacks
- Learn about advanced monitoring with Prometheus and Grafana

## Resources

- [Linux Process Management](https://www.digitalocean.com/community/tutorials/process-management-in-linux)
- [systemd Essentials](https://www.digitalocean.com/community/tutorials/systemd-essentials-working-with-services-units-and-the-journal)
- [Screen Tutorial](https://linuxize.com/post/how-to-use-linux-screen/)
- [Tmux Guide](https://tmuxcheatsheet.com/)
- [NVIDIA GPU Monitoring](https://developer.nvidia.com/nvidia-system-management-interface)

## Conclusion

This solution provides comprehensive tools and knowledge for managing processes in ML infrastructure environments. The skills learned here are essential for running reliable, long-running ML training workloads.

**Key Achievement**: Complete implementation of process management for ML training jobs with monitoring, control, and troubleshooting capabilities.

---

**Exercise 03: Process Management for ML Training Jobs - ✅ READY FOR IMPLEMENTATION**
