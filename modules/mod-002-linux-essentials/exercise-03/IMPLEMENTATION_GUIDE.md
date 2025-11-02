# Implementation Guide: Process Management and System Monitoring for ML Workloads

## Overview

This comprehensive guide walks you through mastering Linux process management in the context of ML infrastructure. You'll learn to monitor, control, and troubleshoot processes running on production ML systems, with specific focus on training jobs, GPU resources, and long-running workloads.

**Estimated Time:** 2-3 hours
**Difficulty:** Intermediate
**Prerequisites:** Completed Exercises 01-02, Basic Python knowledge

## What You'll Learn

By completing this guide, you will be able to:
- Monitor and analyze running processes with multiple tools
- Control process lifecycle (start, stop, pause, resume, kill)
- Manage background and foreground jobs effectively
- Monitor and control GPU processes for ML workloads
- Use systemd to manage ML services
- Handle stuck or runaway training processes
- Implement persistent training sessions with screen/tmux
- Troubleshoot common process-related issues in production

## Prerequisites Check

Before starting, ensure you have:

```bash
# Check you have basic commands available
which ps top kill jobs
which python3 bash

# Check disk space (need ~500MB)
df -h ~

# Verify you can create directories
mkdir -p ~/ml-process-management
cd ~/ml-process-management
```

**Optional but recommended:**
```bash
# Check for htop (better than top)
which htop

# Check for GPU monitoring (if you have NVIDIA GPU)
which nvidia-smi

# Check for screen/tmux
which screen tmux
```

If any of these are missing, install them:
```bash
sudo apt update
sudo apt install htop screen tmux
```

## Phase 1: Understanding Processes (30 minutes)

### Step 1.1: Exploring Process Basics

First, let's understand what processes are running on your system.

```bash
# Create and enter workspace
mkdir -p ~/ml-process-management
cd ~/ml-process-management

# View your current shell's processes
ps
```

**Expected Output:**
```
  PID TTY          TIME CMD
 1234 pts/0    00:00:00 bash
 5678 pts/0    00:00:00 ps
```

**Understanding the output:**
- **PID**: Process ID (unique identifier for each process)
- **TTY**: Terminal type (pts/0 means pseudo-terminal)
- **TIME**: CPU time consumed by the process
- **CMD**: Command that started the process

**Validation:**
- [ ] You see at least your bash shell and ps command
- [ ] Each process has a unique PID

### Step 1.2: Viewing All Processes

```bash
# View all processes for current user
ps -u $(whoami)

# View ALL processes on the system
ps aux

# Count total processes
ps aux | wc -l
```

**Expected Output:**
```
USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
root         1  0.0  0.1  16856  8324 ?        Ss   10:15   0:02 /sbin/init
root         2  0.0  0.0      0     0 ?        S    10:15   0:00 [kthreadd]
...
```

**Understanding ps aux columns:**

Create a reference file for later use:
```bash
cat > ps_reference.txt << 'EOF'
PS AUX OUTPUT COLUMNS
=====================

USER    = User who owns the process
PID     = Process ID (unique identifier)
%CPU    = CPU usage percentage
%MEM    = Memory usage percentage
VSZ     = Virtual memory size (KB)
RSS     = Resident set size - physical memory (KB)
TTY     = Terminal type (? = no terminal/daemon)
STAT    = Process state (see below)
START   = Start time
TIME    = Total CPU time consumed
COMMAND = Command with arguments

PROCESS STATES (STAT):
======================
R = Running or runnable (on run queue)
S = Sleeping (waiting for an event)
D = Uninterruptible sleep (usually I/O)
T = Stopped (by job control signal or tracing)
Z = Zombie (terminated but not reaped by parent)
< = High priority (not nice to other users)
N = Low priority (nice to other users)
L = Has pages locked into memory
s = Session leader
l = Multi-threaded
+ = In foreground process group

Examples:
---------
Ss  = Sleeping session leader
R+  = Running in foreground
S<  = Sleeping with high priority
Ssl = Sleeping session leader, multi-threaded, locked memory
D   = Uninterruptible sleep (potentially stuck on I/O)
Z   = Zombie process (dead but not cleaned up)
EOF

cat ps_reference.txt
```

**Validation:**
- [ ] You can see processes from different users
- [ ] You understand what each column means
- [ ] You've saved the reference file

### Step 1.3: Finding Specific Processes

```bash
# Find all Python processes
ps aux | grep python | grep -v grep

# Better way using pgrep
pgrep -a python

# Find processes sorted by CPU usage
ps aux --sort=-%cpu | head -10

# Find processes sorted by memory usage
ps aux --sort=-%mem | head -10

# View process tree (shows parent-child relationships)
ps auxf
# or
pstree
```

**Practical Exercise:**

Find the highest CPU and memory consumers on your system:

```bash
echo "=== Top 5 CPU Users ==="
ps aux --sort=-%cpu | head -6

echo ""
echo "=== Top 5 Memory Users ==="
ps aux --sort=-%mem | head -6
```

**Expected Output:**
```
=== Top 5 CPU Users ===
USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
root      1234  5.2  2.1 123456 87654 ?        Ssl  10:15   1:23 /usr/bin/some-service
...

=== Top 5 Memory Users ===
USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
user      5678  0.5 15.3 456789 234567 ?       Sl   11:20   0:45 python train.py
...
```

**Validation:**
- [ ] You can find Python processes (if any running)
- [ ] You can identify CPU-intensive processes
- [ ] You can identify memory-intensive processes
- [ ] You understand process tree hierarchy

### Step 1.4: Real-Time Process Monitoring with top

```bash
# Start top (press 'q' to quit)
top
```

Create a reference for top commands:
```bash
cat > top_reference.txt << 'EOF'
TOP COMMAND REFERENCE
=====================

Starting top:
-------------
top              # Normal mode
top -u username  # Show specific user only
top -p PID       # Monitor specific PID
top -d 2         # Update every 2 seconds (default is 3)
top -b -n 1      # Batch mode, 1 iteration (for scripts)

Interactive Commands (while top is running):
-------------------------------------------
P    Sort by CPU usage (default)
M    Sort by memory usage
T    Sort by running time
k    Kill a process (prompts for PID and signal)
r    Renice a process (change priority)
u    Filter by username
c    Toggle command line display (full path vs name)
V    Forest/tree view (show process hierarchy)
1    Toggle individual CPU cores display
h    Help screen
q    Quit

Understanding the Display:
--------------------------
TOP SECTION (System Summary):
  Line 1: Uptime, users, load average
  Line 2: Tasks (total, running, sleeping, stopped, zombie)
  Line 3: CPU usage
    us = user processes
    sy = system/kernel processes
    ni = nice (low priority) processes
    id = idle (higher is better)
    wa = I/O wait (high means disk bottleneck)
    hi = hardware interrupts
    si = software interrupts
    st = steal time (virtualization)
  Line 4-5: Memory and swap usage

PROCESS SECTION:
  PID    = Process ID
  USER   = Owner
  PR     = Priority (20 is normal)
  NI     = Nice value (-20 to 19, lower = higher priority)
  VIRT   = Virtual memory total (KB)
  RES    = Resident memory (actual RAM used, KB)
  SHR    = Shared memory (KB)
  S      = Status (R/S/D/T/Z)
  %CPU   = CPU usage percentage
  %MEM   = Memory usage percentage
  TIME+  = Total CPU time consumed
  COMMAND= Process name/command

Tips:
-----
- Press '1' to see per-CPU usage (useful on multi-core systems)
- Press 'M' to sort by memory when investigating memory leaks
- Press 'c' to see full command paths for better identification
- Watch 'wa' (I/O wait) - high values indicate disk bottleneck
- Watch load average - should be < number of CPU cores
EOF

cat top_reference.txt
```

**Practical Exercise:**

Run top in batch mode to capture a snapshot:
```bash
# Capture top snapshot
top -b -n 1 > top_snapshot.txt

# View the snapshot
head -30 top_snapshot.txt

# Save it for comparison later
cp top_snapshot.txt top_snapshot_baseline.txt
```

**Validation:**
- [ ] You understand top's display layout
- [ ] You can sort by CPU and memory
- [ ] You captured a process snapshot
- [ ] You understand load average and CPU metrics

### Step 1.5: Advanced Monitoring with htop (if available)

If htop is installed, try it:

```bash
# Start htop (more user-friendly than top)
htop
```

**htop advantages:**
- Color-coded display
- Mouse support
- Visual CPU and memory bars
- Tree view by default
- Easier process management (F9 to kill, F7/F8 to nice)

**Validation:**
- [ ] You've explored htop (if available)
- [ ] You understand the visual representation
- [ ] You can navigate with keyboard and mouse

## Phase 2: Process Control and Job Management (45 minutes)

### Step 2.1: Understanding Foreground and Background Jobs

```bash
cd ~/ml-process-management

# Create a simple long-running script
cat > long_task.sh << 'EOF'
#!/bin/bash
# Simulate long-running ML task

echo "Starting long task (PID: $$)..."
for i in {1..30}; do
    echo "Processing step $i/30..."
    sleep 2
done
echo "Task complete!"
EOF

chmod +x long_task.sh
```

**Run in foreground** (blocks your terminal):
```bash
# Start the task
./long_task.sh

# This blocks until complete (60 seconds)
# Press Ctrl+C to interrupt if needed
```

**Expected Output:**
```
Starting long task (PID: 12345)...
Processing step 1/30...
Processing step 2/30...
...
```

**Validation:**
- [ ] Task runs and blocks your terminal
- [ ] You can interrupt with Ctrl+C
- [ ] You understand foreground execution

### Step 2.2: Running Jobs in Background

```bash
# Run in background with &
./long_task.sh &

# Note the job number [1] and PID
```

**Expected Output:**
```
[1] 12345
Starting long task (PID: 12345)...
Processing step 1/30...
```

**Key observations:**
- `[1]` is the job number
- `12345` is the process ID (PID)
- Terminal is not blocked - you can continue working

```bash
# List background jobs
jobs

# List with PIDs
jobs -l
```

**Expected Output:**
```
[1]+  Running                 ./long_task.sh &
```

**Validation:**
- [ ] Task runs in background
- [ ] Terminal is not blocked
- [ ] You can see the job with `jobs` command
- [ ] You understand job numbers vs PIDs

### Step 2.3: Job Control - Foreground/Background Switching

Create a demo script:
```bash
cat > job_control_demo.sh << 'EOF'
#!/bin/bash
# Demonstrate job control

echo "=== Job Control Demo ==="
echo ""

# Start first background job
echo "Starting Job 1..."
(for i in {1..20}; do echo "Job 1: Step $i"; sleep 2; done) &
JOB1_PID=$!
echo "Job 1 PID: $JOB1_PID"

# Start second background job
echo "Starting Job 2..."
(for i in {1..20}; do echo "Job 2: Step $i"; sleep 2; done) &
JOB2_PID=$!
echo "Job 2 PID: $JOB2_PID"

echo ""
echo "Current jobs:"
jobs -l

echo ""
echo "Both jobs running in background. They will complete in 40 seconds."
echo "You can:"
echo "  - List jobs: jobs"
echo "  - Bring to foreground: fg %1"
echo "  - Send to background: bg %1 (after Ctrl+Z)"
echo "  - Kill job: kill %1"

# Wait for all jobs
wait
echo ""
echo "All jobs completed!"
EOF

chmod +x job_control_demo.sh
./job_control_demo.sh
```

**Interactive practice:**

```bash
# Start a long task
./long_task.sh

# Suspend it with Ctrl+Z
# (Press Ctrl+Z now)

# Job is now stopped
jobs

# Resume in background
bg %1

# Bring back to foreground
fg %1

# Suspend again with Ctrl+Z
# Kill it
kill %1
```

**Job notation reference:**
```bash
cat > job_notation.txt << 'EOF'
JOB CONTROL NOTATION
====================

Job References:
---------------
%1      = Job number 1
%2      = Job number 2
%+      = Current job (most recently started/stopped)
%-      = Previous job
%%      = Current job (same as %+)
%?str   = Job whose command contains 'str'

Examples:
---------
fg %1          # Bring job 1 to foreground
bg %2          # Resume job 2 in background
kill %1        # Kill job 1
kill %-        # Kill previous job
kill %?train   # Kill job with 'train' in command

Signals:
--------
Ctrl+C         # Send SIGINT (interrupt/terminate)
Ctrl+Z         # Send SIGTSTP (suspend/stop)
Ctrl+D         # Send EOF (end of input)

Commands:
---------
jobs           # List all jobs
jobs -l        # List with PIDs
fg             # Foreground last job
fg %n          # Foreground job n
bg             # Background last job
bg %n          # Background job n
kill %n        # Kill job n
disown %n      # Remove job from shell's job table
EOF

cat job_notation.txt
```

**Validation:**
- [ ] You can start jobs in background
- [ ] You can suspend jobs with Ctrl+Z
- [ ] You can resume jobs with bg/fg
- [ ] You understand job notation (%1, %+, etc.)

### Step 2.4: Process Signals and Termination

Create a signal handling test:

```bash
cat > signal_test.py << 'EOF'
#!/usr/bin/env python3
"""Test process signal handling"""

import signal
import sys
import time
import os

def signal_handler(signum, frame):
    """Handle signals"""
    signal_names = {
        signal.SIGTERM: "SIGTERM (15)",
        signal.SIGINT: "SIGINT (2)",
        signal.SIGHUP: "SIGHUP (1)"
    }
    sig_name = signal_names.get(signum, f"Signal {signum}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Received {sig_name}")

    if signum == signal.SIGTERM:
        print("Gracefully shutting down...")
        print("Saving state...")
        time.sleep(1)
        print("Cleanup complete")
        sys.exit(0)
    elif signum == signal.SIGINT:
        print("Interrupted by user (Ctrl+C)")
        sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

print(f"Process started (PID: {os.getpid()})")
print("Press Ctrl+C to send SIGINT")
print("Or run: kill -TERM <PID> to send SIGTERM")
print("Running indefinitely...")
print("-" * 50)

try:
    counter = 0
    while True:
        counter += 1
        print(f"[{time.strftime('%H:%M:%S')}] Working... iteration {counter}")
        time.sleep(3)
except KeyboardInterrupt:
    print("\nKeyboardInterrupt caught")
    sys.exit(0)
EOF

chmod +x signal_test.py
```

**Testing signals:**

```bash
# Start the test process in background
./signal_test.py &
TEST_PID=$!
echo "Test process PID: $TEST_PID"

# Watch it run for a few seconds
sleep 5

# Send SIGTERM (graceful shutdown)
kill -TERM $TEST_PID

# Verify it shut down
sleep 2
jobs
```

**Expected behavior:**
- Process receives SIGTERM
- Runs shutdown handler
- Exits cleanly

**Try force kill:**
```bash
# Start again
./signal_test.py &
TEST_PID=$!

# Force kill (cannot be caught)
kill -9 $TEST_PID

# Verify termination
jobs
```

**Signal reference:**
```bash
cat > signal_reference.txt << 'EOF'
LINUX SIGNALS REFERENCE
=======================

Common Signals:
---------------
Signal   Number  Description                     Can Catch?
------   ------  -----------                     ----------
SIGHUP     1     Hangup (terminal closed)        Yes
SIGINT     2     Interrupt (Ctrl+C)              Yes
SIGQUIT    3     Quit (Ctrl+\)                   Yes
SIGKILL    9     Kill immediately                NO
SIGTERM   15     Terminate gracefully            Yes
SIGCONT   18     Continue if stopped             Yes
SIGSTOP   19     Stop process                    NO
SIGTSTP   20     Stop from terminal (Ctrl+Z)     Yes

Usage:
------
kill -l                List all signals
kill PID               Send SIGTERM (15)
kill -15 PID           Send SIGTERM explicitly
kill -TERM PID         Send SIGTERM by name
kill -9 PID            Send SIGKILL (force kill)
kill -KILL PID         Send SIGKILL by name
killall name           Kill all processes named 'name'
pkill pattern          Kill processes matching pattern

Best Practices:
---------------
1. Always try SIGTERM first (kill -15)
   - Allows process to cleanup
   - Saves state/checkpoints
   - Closes files properly

2. Wait a few seconds

3. If process doesn't exit, use SIGKILL (kill -9)
   - Cannot be caught or ignored
   - Immediate termination
   - No cleanup possible

For ML Training:
----------------
GOOD:  kill -TERM <pid>  # Saves checkpoint
BAD:   kill -9 <pid>     # Loses progress

Only use kill -9 if:
- Process is truly hung
- SIGTERM didn't work after 10-30 seconds
- Process is in D state (uninterruptible sleep)
EOF

cat signal_reference.txt
```

**Validation:**
- [ ] You understand different signals
- [ ] You can send SIGTERM for graceful shutdown
- [ ] You know when to use SIGKILL
- [ ] You understand the difference between catchable and uncatchable signals

### Step 2.5: Using nohup for Persistent Processes

```bash
# nohup = no hangup (process continues after logout)

# Create a long-running task
cat > persistent_task.sh << 'EOF'
#!/bin/bash
for i in {1..100}; do
    echo "$(date): Iteration $i"
    sleep 5
done
EOF

chmod +x persistent_task.sh

# Run with nohup
nohup ./persistent_task.sh > persistent.log 2>&1 &
NOHUP_PID=$!

echo "Process started: $NOHUP_PID"
echo "Log file: persistent.log"

# Check it's running
ps -p $NOHUP_PID

# View output
tail -f persistent.log
# Press Ctrl+C to stop viewing (process continues)

# Kill when done
kill $NOHUP_PID
```

**Validation:**
- [ ] Process starts with nohup
- [ ] Output goes to log file
- [ ] Process would survive terminal disconnection
- [ ] You can monitor the log file

## Phase 3: ML Training Process Management (45 minutes)

### Step 3.1: Create ML Training Simulator

Now let's create a realistic ML training simulator with proper signal handling:

```bash
cd ~/ml-process-management
mkdir ml_training
cd ml_training

cat > train_model.py << 'EOF'
#!/usr/bin/env python3
"""ML Training Simulator with Signal Handling"""

import time
import sys
import signal
import json
import os
from datetime import datetime

class TrainingSimulator:
    """Simulates ML training with checkpointing"""

    def __init__(self, epochs=60, checkpoint_interval=10):
        self.epochs = epochs
        self.checkpoint_interval = checkpoint_interval
        self.current_epoch = 0
        self.running = True

        # Register signal handlers
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        signal.signal(signal.SIGINT, self.graceful_shutdown)

        print(f"Training Simulator - PID: {os.getpid()}")
        print(f"Epochs: {epochs}, Checkpoint every {checkpoint_interval}")
        print("=" * 60)

    def graceful_shutdown(self, signum, frame):
        """Handle shutdown gracefully"""
        print(f"\n[{datetime.now()}] Received signal {signum}")
        print(f"Saving checkpoint at epoch {self.current_epoch}...")
        self.running = False
        self.save_checkpoint()
        print("Shutdown complete")
        sys.exit(0)

    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'timestamp': str(datetime.now()),
            'status': 'checkpoint',
            'loss': 1.0 / max(self.current_epoch, 1),
            'accuracy': 1 - (1.0 / max(self.current_epoch, 1))
        }
        filename = f'checkpoint_epoch_{self.current_epoch:04d}.json'
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"Checkpoint saved: {filename}")

    def train(self):
        """Run training simulation"""
        print(f"Starting training for {self.epochs} epochs")
        print("-" * 60)

        for epoch in range(1, self.epochs + 1):
            if not self.running:
                break

            self.current_epoch = epoch

            # Simulate training
            loss = 1.0 / epoch
            accuracy = 1 - (1.0 / epoch)

            # Progress bar
            progress = int((epoch / self.epochs) * 40)
            bar = "#" * progress + "-" * (40 - progress)

            print(f"Epoch {epoch:3d}/{self.epochs} [{bar}] "
                  f"Loss: {loss:.4f} Acc: {accuracy:.4f}")

            time.sleep(1)  # Simulate epoch time

            # Periodic checkpoint
            if epoch % self.checkpoint_interval == 0:
                self.save_checkpoint()

        if self.running:
            print("-" * 60)
            print("Training complete!")

            # Save final model
            final_model = {
                'epochs': self.epochs,
                'final_accuracy': 1 - (1.0 / self.epochs),
                'timestamp': str(datetime.now())
            }
            with open('final_model.json', 'w') as f:
                json.dump(final_model, f, indent=2)
            print("Final model saved: final_model.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    args = parser.parse_args()

    trainer = TrainingSimulator(args.epochs, args.checkpoint_interval)
    trainer.train()
EOF

chmod +x train_model.py
```

**Test the simulator:**

```bash
# Run for 20 epochs (20 seconds)
python3 train_model.py --epochs 20 --checkpoint-interval 5
```

**Expected Output:**
```
Training Simulator - PID: 12345
Epochs: 20, Checkpoint every 5
============================================================
Starting training for 20 epochs
------------------------------------------------------------
Epoch   1/20 [##--------------------------------------] Loss: 1.0000 Acc: 0.0000
Epoch   2/20 [####------------------------------------] Loss: 0.5000 Acc: 0.5000
...
Epoch   5/20 [##########------------------------------] Loss: 0.2000 Acc: 0.8000
Checkpoint saved: checkpoint_epoch_0005.json
...
```

**Validation:**
- [ ] Training simulator runs
- [ ] Checkpoints save at intervals
- [ ] Progress is displayed
- [ ] Final model is saved

### Step 3.2: Create Training Process Manager

Create a script to manage the training process:

```bash
cat > manage_training.sh << 'EOF'
#!/bin/bash
# Training process management wrapper

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/training.pid"

mkdir -p "$LOG_DIR"

start_training() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Training already running (PID: $PID)"
            return 1
        fi
        rm "$PID_FILE"
    fi

    echo "Starting training..."
    nohup python3 "$SCRIPT_DIR/train_model.py" --epochs 60 --checkpoint-interval 10 \
        > "$LOG_DIR/training.log" 2>&1 &

    PID=$!
    echo $PID > "$PID_FILE"
    echo "Training started (PID: $PID)"
    echo "Log: $LOG_DIR/training.log"
}

stop_training() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No training process found"
        return 1
    fi

    PID=$(cat "$PID_FILE")
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "Training process not running"
        rm "$PID_FILE"
        return 1
    fi

    echo "Stopping training (PID: $PID)..."
    kill -TERM $PID

    # Wait for graceful shutdown
    for i in {1..10}; do
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "Training stopped gracefully"
            rm "$PID_FILE"
            return 0
        fi
        sleep 1
    done

    # Force kill if still running
    echo "Force stopping..."
    kill -9 $PID 2>/dev/null
    rm "$PID_FILE"
    echo "Training stopped (forced)"
}

status_training() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Status: Not running"
        return 1
    fi

    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Status: Running (PID: $PID)"
        echo ""
        ps -p $PID -o pid,ppid,%cpu,%mem,etime,cmd
        echo ""
        if [ -f "$LOG_DIR/training.log" ]; then
            echo "Latest log entries:"
            tail -5 "$LOG_DIR/training.log"
        fi
    else
        echo "Status: Stopped (stale PID file)"
        rm "$PID_FILE"
    fi
}

tail_log() {
    if [ -f "$LOG_DIR/training.log" ]; then
        tail -f "$LOG_DIR/training.log"
    else
        echo "No log file found"
    fi
}

case "$1" in
    start)
        start_training
        ;;
    stop)
        stop_training
        ;;
    restart)
        stop_training
        sleep 2
        start_training
        ;;
    status)
        status_training
        ;;
    log)
        tail_log
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|log}"
        exit 1
        ;;
esac
EOF

chmod +x manage_training.sh
```

**Test the manager:**

```bash
# Start training
./manage_training.sh start

# Check status
./manage_training.sh status

# View logs
./manage_training.sh log
# Press Ctrl+C to stop viewing

# Stop training
./manage_training.sh stop
```

**Expected Output:**
```
$ ./manage_training.sh start
Starting training...
Training started (PID: 12345)
Log: /home/user/ml-process-management/ml_training/logs/training.log

$ ./manage_training.sh status
Status: Running (PID: 12345)

  PID  PPID %CPU %MEM     ELAPSED CMD
12345     1  0.5  0.2       00:15 python3 train_model.py --epochs 60

Latest log entries:
Epoch  15/60 [##########--------------------------] Loss: 0.0667 Acc: 0.9333
```

**Validation:**
- [ ] Can start training process
- [ ] Can check status with PID
- [ ] Can view logs
- [ ] Can stop gracefully with SIGTERM
- [ ] PID file is managed correctly

### Step 3.3: Resource Monitoring

Create a resource monitoring script:

```bash
cat > monitor_resources.sh << 'EOF'
#!/bin/bash
# Monitor training process resources

PID_FILE="training.pid"
OUTPUT_FILE="resource_usage.csv"

if [ ! -f "$PID_FILE" ]; then
    echo "No training process found"
    echo "Start training first: ./manage_training.sh start"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p $PID > /dev/null 2>&1; then
    echo "Process not running"
    exit 1
fi

echo "Monitoring PID: $PID"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Create CSV header
echo "timestamp,cpu_percent,mem_percent,mem_rss_mb,mem_vsz_mb" > "$OUTPUT_FILE"

# Display header
printf "%-20s %8s %8s %12s %12s\n" "TIME" "CPU%" "MEM%" "RSS(MB)" "VSZ(MB)"
printf "%-20s %8s %8s %12s %12s\n" "----" "----" "----" "-------" "-------"

while ps -p $PID > /dev/null 2>&1; do
    # Get process stats
    STATS=$(ps -p $PID -o %cpu,%mem,rss,vsz --no-headers)
    CPU=$(echo $STATS | awk '{print $1}')
    MEM=$(echo $STATS | awk '{print $2}')
    RSS=$(echo $STATS | awk '{print $3/1024}')  # Convert to MB
    VSZ=$(echo $STATS | awk '{print $4/1024}')  # Convert to MB

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Display
    printf "%-20s %7.1f%% %7.1f%% %11.1f %11.1f\n" \
        "$TIMESTAMP" "$CPU" "$MEM" "$RSS" "$VSZ"

    # Log to CSV
    echo "$TIMESTAMP,$CPU,$MEM,$RSS,$VSZ" >> "$OUTPUT_FILE"

    sleep 2
done

echo ""
echo "Process ended"
echo "Resource usage logged to: $OUTPUT_FILE"
EOF

chmod +x monitor_resources.sh
```

**Test resource monitoring:**

```bash
# In one terminal, start training
./manage_training.sh start

# In another terminal (or background), monitor
./monitor_resources.sh
```

**Expected Output:**
```
Monitoring PID: 12345
Press Ctrl+C to stop monitoring

TIME                   CPU%     MEM%      RSS(MB)      VSZ(MB)
----                   ----     ----      -------      -------
2025-11-01 10:30:15     1.2%     0.5%        45.2        234.5
2025-11-01 10:30:17     1.5%     0.5%        45.3        234.5
2025-11-01 10:30:19     1.3%     0.5%        45.3        234.5
...
```

**Validation:**
- [ ] Monitoring script tracks CPU and memory
- [ ] Data is logged to CSV file
- [ ] Monitoring continues until process ends
- [ ] Can stop monitoring with Ctrl+C

### Step 3.4: Resource Analysis

Create analysis script:

```bash
cat > analyze_resources.py << 'EOF'
#!/usr/bin/env python3
"""Analyze resource usage from monitoring data"""

import csv
import sys
import os

def analyze_resource_usage(csv_file):
    """Analyze and display resource usage statistics"""

    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    cpu_values = []
    mem_values = []
    rss_values = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cpu_values.append(float(row['cpu_percent']))
                mem_values.append(float(row['mem_percent']))
                rss_values.append(float(row['mem_rss_mb']))
            except (ValueError, KeyError):
                continue

    if not cpu_values:
        print("No data to analyze")
        return

    print("Resource Usage Analysis")
    print("=" * 60)
    print(f"Duration: {len(cpu_values) * 2} seconds ({len(cpu_values)} samples)")
    print()

    print("CPU Usage:")
    print(f"  Average: {sum(cpu_values)/len(cpu_values):6.2f}%")
    print(f"  Maximum: {max(cpu_values):6.2f}%")
    print(f"  Minimum: {min(cpu_values):6.2f}%")
    print()

    print("Memory Usage (%):")
    print(f"  Average: {sum(mem_values)/len(mem_values):6.2f}%")
    print(f"  Maximum: {max(mem_values):6.2f}%")
    print(f"  Minimum: {min(mem_values):6.2f}%")
    print()

    print("Memory (RSS in MB):")
    print(f"  Average: {sum(rss_values)/len(rss_values):8.2f} MB")
    print(f"  Maximum: {max(rss_values):8.2f} MB")
    print(f"  Minimum: {min(rss_values):8.2f} MB")
    print()

    # Simple recommendations
    print("Recommendations:")
    avg_cpu = sum(cpu_values)/len(cpu_values)
    avg_mem = sum(rss_values)/len(rss_values)

    if avg_cpu > 80:
        print("  - High CPU usage detected. Consider optimization or multi-processing")
    elif avg_cpu < 20:
        print("  - Low CPU usage. Process may be I/O bound or underutilized")

    if avg_mem > 1000:
        print(f"  - Memory usage: {avg_mem:.0f}MB. Monitor for memory leaks")

    print("=" * 60)

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "resource_usage.csv"
    analyze_resource_usage(csv_file)
EOF

chmod +x analyze_resources.py
```

**Test analysis:**

```bash
# After some monitoring data is collected
python3 analyze_resources.py resource_usage.csv
```

**Expected Output:**
```
Resource Usage Analysis
============================================================
Duration: 60 seconds (30 samples)

CPU Usage:
  Average:   1.35%
  Maximum:   2.10%
  Minimum:   0.80%

Memory Usage (%):
  Average:   0.52%
  Maximum:   0.55%
  Minimum:   0.50%

Memory (RSS in MB):
  Average:    45.30 MB
  Maximum:    45.80 MB
  Minimum:    45.00 MB

Recommendations:
  - Low CPU usage. Process may be I/O bound or underutilized
============================================================
```

**Validation:**
- [ ] Analysis script runs on CSV data
- [ ] Statistics are calculated correctly
- [ ] Recommendations are provided
- [ ] Output is well-formatted

## Phase 4: GPU Process Management (30 minutes)

**Note:** This phase requires an NVIDIA GPU. If you don't have one, read through for understanding and skip to Phase 5.

### Step 4.1: GPU Monitoring Basics

```bash
cd ~/ml-process-management
mkdir gpu_monitoring
cd gpu_monitoring

# Create GPU reference
cat > gpu_reference.txt << 'EOF'
NVIDIA GPU MONITORING REFERENCE
================================

Basic Commands:
---------------
nvidia-smi                    # Show current GPU status
nvidia-smi -l 1               # Continuous monitoring (1 sec updates)
nvidia-smi -l 5               # Update every 5 seconds
nvidia-smi --help-query-gpu   # Show all queryable GPU properties

Process Monitoring:
-------------------
nvidia-smi pmon               # Process monitoring mode
nvidia-smi pmon -c 10         # Monitor for 10 iterations
nvidia-smi pmon -s u          # Show utilization only

Query Specific Information:
---------------------------
# GPU information
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv

# Process information
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Detailed GPU stats
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.total,memory.used,memory.free --format=csv,noheader

Logging:
--------
# Log GPU stats to file
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used \
  --format=csv -l 5 > gpu_log.csv

Control:
--------
# Set persistence mode (keeps GPU initialized)
sudo nvidia-smi -pm 1

# Reset GPU (if hung)
sudo nvidia-smi --gpu-reset -i 0

Finding GPU Processes:
----------------------
# Find what's using GPU
nvidia-smi

# Get PIDs of GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader

# Find process using most GPU memory
nvidia-smi --query-compute-apps=pid,used_memory \
  --format=csv,noheader,nounits | sort -k2 -rn | head -1

Killing GPU Processes:
----------------------
# Find PID from nvidia-smi
nvidia-smi

# Kill gracefully
kill -TERM <PID>

# Force kill if needed
kill -9 <PID>

Multi-GPU:
----------
# Monitor specific GPU
nvidia-smi -i 0                # GPU 0 only
nvidia-smi -i 0,1              # GPUs 0 and 1

# Set which GPU to use
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 only
export CUDA_VISIBLE_DEVICES=1,2  # Use GPUs 1 and 2
export CUDA_VISIBLE_DEVICES=-1   # No GPU (CPU only)
EOF

cat gpu_reference.txt
```

**Test GPU monitoring (if GPU available):**

```bash
# Check GPU status
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi

# Query specific information
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
```

**Expected Output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.05    Driver Version: 525.85.05    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-...     Off  | 00000000:00:04.0 Off |                    0 |
| N/A   30C    P0    43W / 250W |      0MiB / 40960MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### Step 4.2: GPU Process Management Script

```bash
cat > gpu_manager.sh << 'EOF'
#!/bin/bash
# GPU Process Manager

check_nvidia_smi() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "nvidia-smi not found. GPU monitoring not available."
        echo "This script requires an NVIDIA GPU with drivers installed."
        return 1
    fi
    return 0
}

list_gpu_processes() {
    if ! check_nvidia_smi; then
        return 1
    fi

    echo "=== GPU Processes ==="
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
}

show_gpu_status() {
    if ! check_nvidia_smi; then
        return 1
    fi

    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
}

monitor_gpu() {
    if ! check_nvidia_smi; then
        return 1
    fi

    echo "=== GPU Monitoring ==="
    echo "Press Ctrl+C to stop"
    nvidia-smi -l 2
}

kill_gpu_process() {
    local pid=$1

    if [ -z "$pid" ]; then
        echo "Usage: $0 kill <PID>"
        return 1
    fi

    if ! check_nvidia_smi; then
        return 1
    fi

    # Verify process is using GPU
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q "^$pid$"; then
        echo "Stopping GPU process $pid..."
        kill -TERM $pid
        sleep 2

        if ps -p $pid > /dev/null 2>&1; then
            echo "Process still running, force killing..."
            kill -9 $pid
        else
            echo "Process stopped"
        fi
    else
        echo "PID $pid not found in GPU processes"
        return 1
    fi
}

case "$1" in
    list)
        list_gpu_processes
        ;;
    status)
        show_gpu_status
        ;;
    monitor)
        monitor_gpu
        ;;
    kill)
        kill_gpu_process "$2"
        ;;
    *)
        echo "GPU Process Manager"
        echo ""
        echo "Usage: $0 {list|status|monitor|kill <PID>}"
        echo ""
        echo "Commands:"
        echo "  list    - List processes using GPU"
        echo "  status  - Show GPU status"
        echo "  monitor - Continuous GPU monitoring"
        echo "  kill    - Kill GPU process by PID"
        exit 1
        ;;
esac
EOF

chmod +x gpu_manager.sh
```

**Test GPU manager:**

```bash
# Check GPU status
./gpu_manager.sh status

# List GPU processes
./gpu_manager.sh list

# Monitor (press Ctrl+C to stop)
./gpu_manager.sh monitor
```

**Validation (if GPU available):**
- [ ] Can check GPU status
- [ ] Can list GPU processes
- [ ] Can monitor GPU continuously
- [ ] Understand how to kill GPU processes

## Phase 5: System Services with systemd (30 minutes)

### Step 5.1: Understanding systemd

```bash
cd ~/ml-process-management
mkdir systemd_services
cd systemd_services

# Create systemd reference
cat > systemd_reference.txt << 'EOF'
SYSTEMD SERVICE MANAGEMENT
==========================

Basic Commands:
---------------
systemctl status <service>     # Check service status
systemctl start <service>      # Start service
systemctl stop <service>       # Stop service
systemctl restart <service>    # Restart service
systemctl reload <service>     # Reload configuration
systemctl enable <service>     # Enable at boot
systemctl disable <service>    # Disable at boot

System State:
-------------
systemctl list-units           # List all units
systemctl list-units --type=service  # List services only
systemctl list-units --failed  # List failed units
systemctl is-active <service>  # Check if active
systemctl is-enabled <service> # Check if enabled
systemctl is-failed <service>  # Check if failed

Service Logs (journalctl):
--------------------------
journalctl -u <service>        # View service logs
journalctl -u <service> -f     # Follow logs (like tail -f)
journalctl -u <service> -n 50  # Last 50 lines
journalctl -u <service> --since "1 hour ago"
journalctl -u <service> --since today
journalctl -u <service> --since "2025-11-01 10:00:00"

Common Services:
----------------
docker          # Docker daemon
ssh / sshd      # SSH server
cron            # Scheduled tasks
nginx           # Web server
postgresql      # Database

Examples:
---------
# Check Docker status
systemctl status docker

# Start Docker
sudo systemctl start docker

# Enable Docker at boot
sudo systemctl enable docker

# View Docker logs
journalctl -u docker -n 100

# Restart SSH
sudo systemctl restart sshd

Service Unit File:
------------------
Location: /etc/systemd/system/<service>.service

[Unit]
Description=My Service
After=network.target

[Service]
Type=simple
User=username
ExecStart=/path/to/command
Restart=on-failure

[Install]
WantedBy=multi-user.target

After creating/modifying:
sudo systemctl daemon-reload
sudo systemctl enable <service>
sudo systemctl start <service>
EOF

cat systemd_reference.txt
```

### Step 5.2: Check System Services

```bash
# List all services
systemctl list-units --type=service

# List only running services
systemctl list-units --type=service --state=running

# List failed services
systemctl list-units --failed

# Check specific service status
systemctl status ssh
# or
systemctl status sshd

# Check if Docker is available
systemctl status docker
```

**Expected Output:**
```
● ssh.service - OpenBSD Secure Shell server
   Loaded: loaded (/lib/systemd/system/ssh.service; enabled; vendor preset: enabled)
   Active: active (running) since Fri 2025-11-01 10:15:23 UTC; 2h ago
   ...
```

### Step 5.3: Create Example ML Service Unit File

```bash
cat > ml-training.service << 'EOF'
[Unit]
Description=ML Training Service
Documentation=https://github.com/ai-infra-curriculum
After=network.target

[Service]
Type=simple
User=mluser
Group=mluser
WorkingDirectory=/opt/ml/training

# Main command
ExecStart=/usr/bin/python3 /opt/ml/training/train.py --config production.yaml

# Restart policy
Restart=on-failure
RestartSec=10
StartLimitBurst=3
StartLimitIntervalSec=60

# Resource limits
CPUQuota=200%
MemoryLimit=8G
TasksMax=100

# Environment variables
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PYTHONUNBUFFERED=1"
Environment="TF_CPP_MIN_LOG_LEVEL=1"

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ml-training

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

cat > systemd_service_guide.md << 'EOF'
# Creating Systemd Services for ML Training

## Service Unit File Structure

### [Unit] Section
```ini
Description=Human-readable description
Documentation=URL to documentation
After=service.target      # Start after this target
Requires=service.target   # Dependency (must be running)
Wants=service.target      # Soft dependency
```

### [Service] Section
```ini
Type=simple              # Service type (simple, forking, oneshot)
User=username            # Run as this user
Group=groupname          # Run as this group
WorkingDirectory=/path   # Working directory

ExecStart=/path/to/cmd   # Main command (required)
ExecStartPre=/pre/cmd    # Run before main command
ExecStartPost=/post/cmd  # Run after main command
ExecStop=/stop/cmd       # Custom stop command
ExecReload=/reload/cmd   # Reload command

Restart=on-failure       # Restart policy
RestartSec=10            # Wait before restart
StartLimitBurst=3        # Max restart attempts
StartLimitIntervalSec=60 # Time window for restarts

# Resource Limits
CPUQuota=200%            # Max 2 CPUs (200%)
MemoryLimit=8G           # Max 8GB RAM
TasksMax=100             # Max processes/threads

# Environment
Environment="VAR=value"
EnvironmentFile=/path/to/envfile

# Logging
StandardOutput=journal   # Log stdout to journal
StandardError=journal    # Log stderr to journal
SyslogIdentifier=name    # Log identifier
```

### [Install] Section
```ini
WantedBy=multi-user.target   # Install target
RequiredBy=service.target    # Required by target
```

## Installation Steps

1. **Create service file:**
   ```bash
   sudo nano /etc/systemd/system/ml-training.service
   ```

2. **Reload systemd:**
   ```bash
   sudo systemctl daemon-reload
   ```

3. **Enable service:**
   ```bash
   sudo systemctl enable ml-training
   ```

4. **Start service:**
   ```bash
   sudo systemctl start ml-training
   ```

5. **Check status:**
   ```bash
   systemctl status ml-training
   ```

6. **View logs:**
   ```bash
   journalctl -u ml-training -f
   ```

## Common Service Types

**Type=simple** (default)
- Service is considered started immediately
- ExecStart is the main process
- Good for: Most applications

**Type=forking**
- Service forks and parent exits
- Good for: Traditional daemons

**Type=oneshot**
- Process must exit before systemd continues
- Good for: Scripts, one-time tasks

**Type=notify**
- Service notifies systemd when ready
- Good for: Services with sd_notify support

## Best Practices

1. **Always run as non-root user** unless necessary
2. **Set resource limits** to prevent runaway processes
3. **Configure restart policy** for resilience
4. **Use journal logging** for centralized logs
5. **Enable security features** (NoNewPrivileges, PrivateTmp)
6. **Test before enabling** at boot
7. **Document your service** in Description and Documentation

## Example ML Training Service

```ini
[Unit]
Description=Production ML Model Training
After=network.target docker.service

[Service]
Type=simple
User=mlops
WorkingDirectory=/opt/ml/experiments

ExecStart=/usr/bin/python3 train.py --config prod.yaml
Restart=on-failure
RestartSec=30

CPUQuota=400%
MemoryLimit=16G

Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="PYTHONUNBUFFERED=1"

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

**Service fails to start:**
```bash
systemctl status ml-training
journalctl -u ml-training -n 50
```

**Check if enabled:**
```bash
systemctl is-enabled ml-training
```

**Manually test command:**
```bash
sudo -u mluser /path/to/command
```

**Reload after changes:**
```bash
sudo systemctl daemon-reload
sudo systemctl restart ml-training
```
EOF

cat systemd_service_guide.md
```

**Validation:**
- [ ] You understand systemd service structure
- [ ] You can check service status
- [ ] You can view service logs
- [ ] You know how to create service unit files

### Step 5.4: Working with journalctl

```bash
# Create journalctl reference
cat > journalctl_reference.txt << 'EOF'
JOURNALCTL REFERENCE
====================

Basic Usage:
------------
journalctl                     # All logs
journalctl -n 50               # Last 50 lines
journalctl -f                  # Follow (like tail -f)
journalctl -r                  # Reverse order (newest first)

Filter by Service:
------------------
journalctl -u docker           # Docker service logs
journalctl -u docker -f        # Follow Docker logs
journalctl -u docker -n 100    # Last 100 lines

Filter by Time:
---------------
journalctl --since "1 hour ago"
journalctl --since "2 hours ago"
journalctl --since today
journalctl --since yesterday
journalctl --since "2025-11-01 10:00:00"
journalctl --until "2025-11-01 11:00:00"
journalctl --since "10:00" --until "11:00"

Filter by Priority:
-------------------
journalctl -p err              # Errors only
journalctl -p warning          # Warnings and above
journalctl -p info             # Info and above
# Priorities: emerg, alert, crit, err, warning, notice, info, debug

Filter by Process:
------------------
journalctl _PID=1234           # Specific PID
journalctl _COMM=python        # Specific command

Output Formats:
---------------
journalctl -o short            # Default format
journalctl -o verbose          # All fields
journalctl -o json             # JSON output
journalctl -o json-pretty      # Pretty JSON
journalctl -o cat              # Just messages

Disk Usage:
-----------
journalctl --disk-usage        # Show journal disk usage
sudo journalctl --vacuum-time=7d   # Keep 7 days
sudo journalctl --vacuum-size=1G   # Keep 1GB

Kernel Messages:
----------------
journalctl -k                  # Kernel messages
journalctl -k -b               # Kernel messages this boot

Boot Messages:
--------------
journalctl -b                  # Current boot
journalctl -b -1               # Previous boot
journalctl --list-boots        # List all boots

Examples:
---------
# Docker errors in last hour
journalctl -u docker -p err --since "1 hour ago"

# Follow SSH login attempts
journalctl -u ssh -f | grep "Accepted\|Failed"

# Check for OOM kills
journalctl -k | grep -i "out of memory"

# All logs for specific PID
journalctl _PID=12345

# Logs between times
journalctl --since "2025-11-01 09:00" --until "2025-11-01 10:00"
EOF

cat journalctl_reference.txt
```

**Practice with journalctl:**

```bash
# View recent system logs
journalctl -n 20

# View logs from last hour
journalctl --since "1 hour ago"

# View SSH logs
journalctl -u ssh -n 50
# or
journalctl -u sshd -n 50

# Follow system logs
journalctl -f
# Press Ctrl+C to stop
```

**Validation:**
- [ ] You can view service logs
- [ ] You can filter by time
- [ ] You can follow logs in real-time
- [ ] You understand log priorities

## Phase 6: Persistent Sessions (30 minutes)

### Step 6.1: Screen Basics

```bash
cd ~/ml-process-management
mkdir persistent_sessions
cd persistent_sessions

# Create screen reference
cat > screen_reference.txt << 'EOF'
SCREEN REFERENCE
================

Screen allows terminal sessions that persist after disconnection.

Starting Screen:
----------------
screen                  # Start new unnamed session
screen -S name          # Start named session
screen -ls              # List all sessions
screen -r               # Reattach to detached session
screen -r name          # Reattach to named session
screen -d name          # Detach session (from outside)
screen -X -S name quit  # Kill named session

Inside Screen Session:
----------------------
Ctrl+a d                # Detach from session
Ctrl+a c                # Create new window
Ctrl+a n                # Next window
Ctrl+a p                # Previous window
Ctrl+a "                # List all windows
Ctrl+a 0-9              # Switch to window 0-9
Ctrl+a k                # Kill current window
Ctrl+a [                # Enter copy mode (scroll back)
Ctrl+a ]                # Paste buffer
Ctrl+a ?                # Help

ML Training Workflow:
---------------------
# Start training session
screen -S training
python train.py --epochs 100
# Ctrl+a d to detach

# Continue working, disconnect SSH, etc.

# Later, reattach
screen -r training

# Check running sessions
screen -ls

Examples:
---------
# Multiple experiments
screen -S exp001
python train.py --config exp001.yaml
# Ctrl+a d

screen -S exp002
python train.py --config exp002.yaml
# Ctrl+a d

# List experiments
screen -ls

# Reattach to specific
screen -r exp001

# Kill session
screen -X -S exp001 quit
EOF

cat screen_reference.txt
```

**Practice with screen:**

```bash
# Start named session
screen -S test_session

# Inside screen: run a long task
for i in {1..20}; do echo "Iteration $i"; sleep 2; done

# Detach: Press Ctrl+a then d

# List sessions
screen -ls

# Reattach
screen -r test_session

# Kill session (from inside): Ctrl+a then k, then y
# Or from outside:
screen -X -S test_session quit
```

**Validation:**
- [ ] You can start named screen session
- [ ] You can detach with Ctrl+a d
- [ ] You can list sessions
- [ ] You can reattach to session
- [ ] You can kill session

### Step 6.2: Tmux Basics (Modern Alternative)

```bash
cat > tmux_reference.txt << 'EOF'
TMUX REFERENCE
==============

Tmux is a modern screen alternative with more features.

Starting Tmux:
--------------
tmux                    # Start new unnamed session
tmux new -s name        # Start named session
tmux ls                 # List all sessions
tmux attach             # Attach to last session
tmux attach -t name     # Attach to named session
tmux kill-session -t name  # Kill session

Inside Tmux (Prefix: Ctrl+b):
------------------------------
Ctrl+b d                # Detach from session
Ctrl+b c                # Create new window
Ctrl+b n                # Next window
Ctrl+b p                # Previous window
Ctrl+b w                # List windows
Ctrl+b 0-9              # Switch to window 0-9
Ctrl+b &                # Kill window (confirms)
Ctrl+b %                # Split pane vertically
Ctrl+b "                # Split pane horizontally
Ctrl+b arrow            # Navigate between panes
Ctrl+b x                # Kill pane
Ctrl+b [                # Enter copy mode (scroll)
Ctrl+b ?                # Help

ML Training Workflow:
---------------------
# Start training session
tmux new -s training
python train.py --epochs 100
# Ctrl+b d to detach

# Reattach later
tmux attach -t training

# Create monitoring session with splits
tmux new -s monitor

# Split vertically: Ctrl+b %
# Left pane: python train.py
# Right pane: nvidia-smi -l 1

# Split right pane horizontally: Ctrl+b "
# Top-right: nvidia-smi -l 1
# Bottom-right: htop

# Navigate panes: Ctrl+b arrow keys

Examples:
---------
# Multiple experiments
tmux new -s exp001
tmux new -s exp002
tmux new -s exp003

# List all
tmux ls

# Attach to specific
tmux attach -t exp002

# Kill specific session
tmux kill-session -t exp001

# Kill all sessions
tmux kill-server

# Rename session (inside tmux)
Ctrl+b $
EOF

cat tmux_reference.txt
```

**Practice with tmux:**

```bash
# Start named session
tmux new -s test_tmux

# Run a task
for i in {1..20}; do echo "Iteration $i"; sleep 2; done

# Detach: Ctrl+b then d

# List sessions
tmux ls

# Reattach
tmux attach -t test_tmux

# Try pane splitting
# Ctrl+b %  (vertical split)
# Ctrl+b "  (horizontal split)
# Ctrl+b arrow  (navigate)

# Kill session
tmux kill-session -t test_tmux
```

**Validation:**
- [ ] You can start named tmux session
- [ ] You can detach with Ctrl+b d
- [ ] You can list sessions
- [ ] You can reattach to session
- [ ] You can split panes
- [ ] You can navigate panes
- [ ] You can kill session

### Step 6.3: Launch Training Script

Create a universal launcher:

```bash
cat > launch_training.sh << 'EOF'
#!/bin/bash
# Launch training in persistent session

EXPERIMENT_NAME="${1:-experiment}"
TRAINING_SCRIPT="${2:-train.py}"
SESSION_TYPE="${3:-tmux}"

if [ "$SESSION_TYPE" = "tmux" ]; then
    if ! command -v tmux &> /dev/null; then
        echo "tmux not installed. Using screen."
        SESSION_TYPE="screen"
    fi
fi

echo "Launching training session"
echo "  Name: $EXPERIMENT_NAME"
echo "  Script: $TRAINING_SCRIPT"
echo "  Type: $SESSION_TYPE"
echo ""

if [ "$SESSION_TYPE" = "tmux" ]; then
    # Launch with tmux
    tmux new-session -d -s "$EXPERIMENT_NAME"
    tmux send-keys -t "$EXPERIMENT_NAME" "cd $(pwd)" C-m
    tmux send-keys -t "$EXPERIMENT_NAME" "python3 $TRAINING_SCRIPT" C-m

    echo "Training started in tmux session: $EXPERIMENT_NAME"
    echo "  Attach: tmux attach -t $EXPERIMENT_NAME"
    echo "  Detach: Ctrl+b d"
else
    # Launch with screen
    screen -dmS "$EXPERIMENT_NAME" bash -c "cd $(pwd) && python3 $TRAINING_SCRIPT"

    echo "Training started in screen session: $EXPERIMENT_NAME"
    echo "  Attach: screen -r $EXPERIMENT_NAME"
    echo "  Detach: Ctrl+a d"
fi

echo ""
echo "List sessions:"
if [ "$SESSION_TYPE" = "tmux" ]; then
    tmux ls
else
    screen -ls
fi
EOF

chmod +x launch_training.sh
```

**Test launcher:**

```bash
# Make sure train_model.py exists
cd ~/ml-process-management/ml_training

# Launch with tmux
../persistent_sessions/launch_training.sh test_exp train_model.py tmux

# Check it's running
tmux ls

# Attach to see progress
tmux attach -t test_exp

# Detach: Ctrl+b d

# Kill when done
tmux kill-session -t test_exp
```

**Validation:**
- [ ] Launcher creates session
- [ ] Training starts in session
- [ ] Can attach to see progress
- [ ] Can detach without stopping
- [ ] Session persists after detach

## Phase 7: Troubleshooting and Recovery (30 minutes)

### Step 7.1: Process Diagnostic Tool

```bash
cd ~/ml-process-management
mkdir troubleshooting
cd troubleshooting

cat > diagnose_process.sh << 'EOF'
#!/bin/bash
# Comprehensive process diagnostic tool

PID=$1

if [ -z "$PID" ]; then
    echo "Usage: $0 <PID>"
    echo ""
    echo "Examples:"
    echo "  $0 1234          # Diagnose process 1234"
    echo "  $0 \$(cat file.pid)  # Diagnose from PID file"
    exit 1
fi

if ! ps -p $PID > /dev/null 2>&1; then
    echo "ERROR: Process $PID not found"
    exit 1
fi

echo "=== Process Diagnostics for PID: $PID ==="
echo ""

echo "Basic Information:"
echo "------------------"
ps -p $PID -o pid,ppid,user,%cpu,%mem,vsz,rss,stat,start,time,cmd
echo ""

echo "Process State:"
echo "--------------"
STAT=$(ps -p $PID -o stat --no-headers)
echo "State: $STAT"
case $STAT in
    R*) echo "  Running or runnable" ;;
    S*) echo "  Sleeping (interruptible)" ;;
    D*) echo "  Uninterruptible sleep (likely I/O wait)" ;;
    T*) echo "  Stopped (by signal or debugger)" ;;
    Z*) echo "  Zombie (terminated, not reaped)" ;;
esac
echo ""

echo "Process Tree:"
echo "-------------"
pstree -p $PID 2>/dev/null || echo "pstree not available"
echo ""

echo "Open Files (first 20):"
echo "----------------------"
lsof -p $PID 2>/dev/null | head -20 || echo "lsof not available or permission denied"
echo ""

echo "Network Connections:"
echo "--------------------"
lsof -i -a -p $PID 2>/dev/null || echo "No network connections or lsof not available"
echo ""

if [ -r /proc/$PID/limits ]; then
    echo "Resource Limits:"
    echo "----------------"
    cat /proc/$PID/limits
    echo ""
fi

if [ -r /proc/$PID/status ]; then
    echo "Memory Info:"
    echo "------------"
    grep -E "^Vm|^Rss" /proc/$PID/status
    echo ""
fi

if [ -r /proc/$PID/cmdline ]; then
    echo "Command Line:"
    echo "-------------"
    cat /proc/$PID/cmdline | tr '\0' ' '
    echo ""
    echo ""
fi

if [ -r /proc/$PID/cwd ]; then
    echo "Working Directory:"
    echo "------------------"
    readlink /proc/$PID/cwd
    echo ""
fi

echo "=== Diagnostic Complete ==="
EOF

chmod +x diagnose_process.sh
```

**Test diagnostic tool:**

```bash
# Start a test process
cd ~/ml-process-management/ml_training
./manage_training.sh start
TEST_PID=$(cat training.pid)

# Diagnose it
cd ~/ml-process-management/troubleshooting
./diagnose_process.sh $TEST_PID

# Stop test process
cd ~/ml-process-management/ml_training
./manage_training.sh stop
```

**Expected Output:**
```
=== Process Diagnostics for PID: 12345 ===

Basic Information:
------------------
  PID  PPID USER     %CPU %MEM    VSZ   RSS STAT  START   TIME CMD
12345     1 user      1.5  0.5 234567 45678 S     10:30   0:15 python3 train_model.py

Process State:
--------------
State: S
  Sleeping (interruptible)

Process Tree:
-------------
python3(12345)

...
```

**Validation:**
- [ ] Diagnostic tool runs
- [ ] Shows process information
- [ ] Shows open files
- [ ] Shows network connections
- [ ] Shows resource limits

### Step 7.2: Common Problem Scenarios

Create troubleshooting guide:

```bash
cat > troubleshooting_guide.md << 'EOF'
# Process Troubleshooting Guide

## Scenario 1: Process Won't Stop

### Symptoms
- `kill -15 PID` doesn't terminate process
- Process shows as running but unresponsive

### Diagnosis
```bash
# Check process state
ps aux | grep <PID>

# Look at STAT column
# D = uninterruptible sleep (can't be killed with SIGTERM)
# Z = zombie (already dead, waiting for parent to reap)
```

### Solution

**If process is in D state (uninterruptible sleep):**
```bash
# Usually waiting on I/O
# Check what it's waiting for
sudo cat /proc/<PID>/stack

# Check disk I/O
iostat -x 1

# May need to fix underlying issue (disk problem, NFS mount, etc.)
# If truly stuck, only SIGKILL might work
kill -9 <PID>
```

**If process is zombie (Z state):**
```bash
# Find parent process
PPID=$(ps -o ppid= -p <PID>)

# Kill parent to force reaping
kill -15 $PPID

# If parent won't die
kill -9 $PPID
```

**Standard approach:**
```bash
# 1. Try graceful shutdown
kill -TERM <PID>

# 2. Wait 10-30 seconds
sleep 10

# 3. Check if still running
ps -p <PID>

# 4. Force kill if needed
kill -9 <PID>
```

## Scenario 2: Process Consuming 100% CPU

### Diagnosis
```bash
# Find CPU hogs
top -o %CPU

# Or with ps
ps aux --sort=-%cpu | head -10

# Check if it's expected (ML training should use CPU)
# Check if it's stuck in infinite loop
```

### Solution

**If legitimate workload:**
```bash
# Lower priority to not starve other processes
renice +10 <PID>

# Or start with lower priority
nice -n 10 ./command
```

**If runaway process:**
```bash
# Kill it
kill -TERM <PID>

# Check code for infinite loops
```

**If need to limit CPU:**
```bash
# Install cpulimit if available
sudo apt install cpulimit

# Limit to 50% of one CPU
cpulimit -p <PID> -l 50

# Limit to 200% (2 CPUs)
cpulimit -p <PID> -l 200
```

## Scenario 3: Out of Memory (OOM)

### Symptoms
- Process killed unexpectedly
- "Killed" message in logs
- System becomes slow/unresponsive

### Diagnosis
```bash
# Check for OOM kills in kernel log
dmesg | grep -i "killed process"
dmesg | grep -i "out of memory"

# Or with journalctl
journalctl -k | grep -i "out of memory"

# Check current memory usage
free -h

# Find memory hogs
ps aux --sort=-%mem | head -10

# Check swap usage
swapon -s
```

### Solution

**Immediate:**
```bash
# Kill memory-hungry process
kill -15 <PID>

# Free cache (doesn't help with actual OOM)
sudo sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

**Long-term:**
```bash
# For ML training:
# 1. Reduce batch size
# 2. Use gradient checkpointing
# 3. Add more RAM or swap
# 4. Use mixed precision training
# 5. Split data across workers

# Add swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Monitor memory before training
free -h
```

## Scenario 4: Training Process Stuck

### Symptoms
- No progress for long time
- No CPU/GPU usage
- Process in D state

### Diagnosis
```bash
# Check process state
ps aux | grep <PID>

# If D state, check what it's waiting for
sudo cat /proc/<PID>/stack

# Check I/O wait
iostat -x 1

# Check disk space
df -h

# Check if waiting on network
netstat -anp | grep <PID>
lsof -i -a -p <PID>

# Use diagnostic tool
./diagnose_process.sh <PID>
```

### Solution

**If I/O bound:**
```bash
# Check disk space
df -h

# Check disk performance
iostat -x 1

# Check for disk errors
dmesg | grep -i error

# May need to fix disk issue or move to faster storage
```

**If network bound:**
```bash
# Check network connectivity
ping <remote_host>

# Check if remote service is up
telnet <remote_host> <port>

# May need to fix network/firewall/remote service
```

**If truly stuck:**
```bash
# Last resort: force kill and restart
kill -9 <PID>

# Restart from last checkpoint
python train.py --resume checkpoint_epoch_50.pth
```

## Scenario 5: Too Many Processes

### Symptoms
- "Cannot fork" errors
- System becomes unresponsive
- Can't start new processes

### Diagnosis
```bash
# Count processes per user
ps aux | awk '{print $1}' | sort | uniq -c | sort -rn

# Check user process limit
ulimit -u

# Check system-wide limits
cat /proc/sys/kernel/pid_max
```

### Solution

```bash
# Kill unnecessary processes
pkill -u username process_name

# Increase user limit (temporary)
ulimit -u 4096

# Increase permanently (edit /etc/security/limits.conf)
# username soft nproc 4096
# username hard nproc 8192

# Find and kill runaway fork bombs
pkill -9 -u username
```

## Scenario 6: Can't Find Process

### Solution

```bash
# Search by name
pgrep -a python
ps aux | grep train

# Search by port
lsof -i :8080
netstat -tulpn | grep 8080
ss -tulpn | grep 8080

# Search by file
lsof /path/to/file

# Search by user
ps -u username

# Search in command line
ps aux | grep "train.py"
```

## Scenario 7: GPU Process Issues

### Symptoms
- "Out of memory" errors from CUDA
- GPU not being used
- Multiple processes fighting for GPU

### Diagnosis
```bash
# Check GPU usage
nvidia-smi

# Check GPU processes
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
```

### Solution

```bash
# Kill GPU process
nvidia-smi  # Find PID
kill -15 <PID>

# Set specific GPU
export CUDA_VISIBLE_DEVICES=0

# Disable GPU (use CPU)
export CUDA_VISIBLE_DEVICES=-1

# Clear GPU memory (kill all GPU processes)
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I {} kill -9 {}

# Reset GPU (if driver hung)
sudo nvidia-smi --gpu-reset
```

## Quick Reference

### Process States
- R: Running
- S: Sleeping (interruptible)
- D: Uninterruptible sleep (I/O wait, can't be killed with SIGTERM)
- T: Stopped
- Z: Zombie (dead but not reaped)

### Kill Hierarchy
1. Try SIGTERM (15): `kill -15 <PID>`
2. Wait 10-30 seconds
3. Try SIGKILL (9): `kill -9 <PID>`

### Memory Check
```bash
free -h                              # Overall memory
ps aux --sort=-%mem | head -10       # Top memory users
dmesg | grep -i "out of memory"      # OOM kills
```

### CPU Check
```bash
top                                  # Interactive
ps aux --sort=-%cpu | head -10       # Top CPU users
htop                                 # Better interactive
```

### Useful Commands
```bash
pgrep -a <name>                      # Find PID by name
pkill <pattern>                      # Kill by pattern
lsof -i :<port>                      # Find process on port
./diagnose_process.sh <PID>          # Full diagnostic
```
EOF

cat troubleshooting_guide.md
```

**Validation:**
- [ ] You understand common issues
- [ ] You know how to diagnose problems
- [ ] You have solutions for each scenario
- [ ] You know when to use force kill

## Phase 8: Validation and Testing (20 minutes)

### Step 8.1: Create Validation Script

```bash
cd ~/ml-process-management

cat > validate_exercise.sh << 'EOF'
#!/bin/bash
# Validate Exercise 03 completion

echo "=== Exercise 03 Validation ==="
echo ""

PASS=0
FAIL=0

# Test 1: Check directories exist
echo "Test 1: Directory structure"
for dir in ml_training gpu_monitoring systemd_services persistent_sessions troubleshooting; do
    if [ -d "$dir" ]; then
        echo "  ✓ Directory exists: $dir"
        ((PASS++))
    else
        echo "  ✗ Missing directory: $dir"
        ((FAIL++))
    fi
done
echo ""

# Test 2: Check scripts are executable
echo "Test 2: Executable scripts"
for script in ml_training/manage_training.sh ml_training/monitor_resources.sh troubleshooting/diagnose_process.sh; do
    if [ -x "$script" ]; then
        echo "  ✓ Script executable: $script"
        ((PASS++))
    else
        echo "  ✗ Script not executable: $script"
        ((FAIL++))
    fi
done
echo ""

# Test 3: Check reference files exist
echo "Test 3: Reference files"
for file in ps_reference.txt top_reference.txt signal_reference.txt; do
    if [ -f "$file" ]; then
        echo "  ✓ Reference file exists: $file"
        ((PASS++))
    else
        echo "  ✗ Missing reference file: $file"
        ((FAIL++))
    fi
done
echo ""

# Test 4: Test process monitoring
echo "Test 4: Process monitoring commands"
if command -v ps &> /dev/null; then
    echo "  ✓ ps command available"
    ((PASS++))
else
    echo "  ✗ ps command not found"
    ((FAIL++))
fi

if command -v top &> /dev/null; then
    echo "  ✓ top command available"
    ((PASS++))
else
    echo "  ✗ top command not found"
    ((FAIL++))
fi
echo ""

# Test 5: Test Python availability
echo "Test 5: Python environment"
if command -v python3 &> /dev/null; then
    echo "  ✓ Python3 available"
    ((PASS++))
    PYTHON_VERSION=$(python3 --version)
    echo "    Version: $PYTHON_VERSION"
else
    echo "  ✗ Python3 not found"
    ((FAIL++))
fi
echo ""

# Test 6: Optional tools
echo "Test 6: Optional tools (won't affect pass/fail)"
if command -v htop &> /dev/null; then
    echo "  ✓ htop available (recommended)"
else
    echo "  ℹ htop not installed (optional but recommended)"
fi

if command -v screen &> /dev/null; then
    echo "  ✓ screen available"
else
    echo "  ℹ screen not installed (install: sudo apt install screen)"
fi

if command -v tmux &> /dev/null; then
    echo "  ✓ tmux available"
else
    echo "  ℹ tmux not installed (install: sudo apt install tmux)"
fi

if command -v nvidia-smi &> /dev/null; then
    echo "  ✓ nvidia-smi available (GPU monitoring enabled)"
else
    echo "  ℹ nvidia-smi not available (skip GPU sections)"
fi
echo ""

# Summary
echo "=== Validation Summary ==="
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "✓ All validations passed!"
    echo ""
    echo "You have successfully completed Exercise 03!"
    echo ""
    echo "Knowledge check:"
    echo "1. Explain the difference between kill -15 and kill -9"
    echo "2. What does the STAT column 'D' mean in ps output?"
    echo "3. How do you detach from a screen session?"
    echo "4. What's the best way to ensure a training job continues after SSH disconnects?"
    echo "5. How would you find which process is using a specific port?"
    exit 0
else
    echo "✗ Some validations failed. Review the errors above."
    exit 1
fi
EOF

chmod +x validate_exercise.sh
```

**Run validation:**

```bash
./validate_exercise.sh
```

**Expected Output:**
```
=== Exercise 03 Validation ===

Test 1: Directory structure
  ✓ Directory exists: ml_training
  ✓ Directory exists: gpu_monitoring
  ...

Test 2: Executable scripts
  ✓ Script executable: ml_training/manage_training.sh
  ...

...

=== Validation Summary ===
Passed: 12
Failed: 0

✓ All validations passed!
```

**Validation:**
- [ ] All tests pass
- [ ] All directories created
- [ ] All scripts executable
- [ ] Reference files created

### Step 8.2: Test End-to-End Workflow

```bash
# Complete workflow test
cd ~/ml-process-management

# 1. Start training
cd ml_training
./manage_training.sh start
echo "Training started, waiting 10 seconds..."
sleep 10

# 2. Check status
./manage_training.sh status

# 3. Monitor resources
cd ~/ml-process-management
./ml_training/monitor_resources.sh &
MONITOR_PID=$!
echo "Monitoring started, PID: $MONITOR_PID"
sleep 10

# 4. Stop monitoring
kill $MONITOR_PID
echo "Monitoring stopped"

# 5. Check logs
tail -10 ml_training/logs/training.log

# 6. Stop training gracefully
cd ml_training
./manage_training.sh stop
echo "Training stopped"

# 7. Verify checkpoints saved
ls -lh checkpoint_epoch_*.json

# 8. Analyze resources
cd ~/ml-process-management
python3 ml_training/analyze_resources.py ml_training/resource_usage.csv

echo ""
echo "End-to-end test complete!"
```

**Validation:**
- [ ] Training starts successfully
- [ ] Status shows running process
- [ ] Monitoring captures data
- [ ] Training stops gracefully
- [ ] Checkpoints are saved
- [ ] Resource analysis works

## Common Issues and Solutions

### Issue 1: Permission Denied on Scripts

**Problem:** `./script.sh: Permission denied`

**Solution:**
```bash
chmod +x script.sh
```

### Issue 2: Command Not Found

**Problem:** `python3: command not found`

**Solution:**
```bash
# Check Python installation
which python3

# Install if needed
sudo apt update
sudo apt install python3
```

### Issue 3: PID File Stale

**Problem:** Process stopped but PID file remains

**Solution:**
```bash
# Check if process is really running
ps -p $(cat training.pid)

# If not running, remove PID file
rm training.pid
```

### Issue 4: Can't Kill Process

**Problem:** Process won't stop with kill -15

**Solution:**
```bash
# Force kill
kill -9 <PID>

# If zombie, kill parent
ps -o ppid= -p <PID>
kill <PPID>
```

### Issue 5: Out of Disk Space

**Problem:** Training fails with disk errors

**Solution:**
```bash
# Check disk space
df -h

# Clean up old checkpoints
rm checkpoint_epoch_00*.json

# Clean up logs
rm logs/*.log
```

## Best Practices Summary

### Process Management
1. **Always try SIGTERM before SIGKILL**
   - Allows graceful shutdown
   - Saves checkpoints
   - Cleans up resources

2. **Use PID files for long-running processes**
   - Easy to check if running
   - Easy to send signals
   - Easy to manage

3. **Monitor resource usage**
   - Identify bottlenecks
   - Prevent OOM kills
   - Optimize performance

### ML Training
1. **Use persistent sessions (screen/tmux)**
   - Survives disconnections
   - Can reattach anytime
   - Essential for remote work

2. **Implement checkpoint saving**
   - Regular intervals
   - On shutdown signals
   - Allows recovery

3. **Proper signal handling**
   - Catch SIGTERM
   - Save state before exit
   - Clean up resources

### Monitoring
1. **Regular monitoring during training**
   - CPU and memory usage
   - GPU utilization (if applicable)
   - Disk I/O

2. **Log everything**
   - Training progress
   - Resource usage
   - Errors and warnings

3. **Use the right tools**
   - ps for snapshots
   - top/htop for interactive
   - Custom scripts for logging

## Completion Checklist

### Phase 1: Understanding Processes
- [ ] Can use ps to view processes
- [ ] Can sort processes by CPU/memory
- [ ] Can use top for real-time monitoring
- [ ] Understand process states (R, S, D, T, Z)
- [ ] Can find specific processes

### Phase 2: Process Control
- [ ] Can run jobs in background
- [ ] Can use job control (fg, bg, jobs)
- [ ] Understand signals (SIGTERM, SIGKILL, etc.)
- [ ] Can kill processes gracefully
- [ ] Can use nohup for persistent processes

### Phase 3: ML Training Management
- [ ] Created ML training simulator
- [ ] Created process manager script
- [ ] Can monitor resource usage
- [ ] Can analyze resource data
- [ ] Understand checkpointing

### Phase 4: GPU Management
- [ ] Understand nvidia-smi (if GPU available)
- [ ] Can list GPU processes
- [ ] Can kill GPU processes
- [ ] Know how to set CUDA_VISIBLE_DEVICES

### Phase 5: System Services
- [ ] Understand systemd service structure
- [ ] Can check service status
- [ ] Can view service logs with journalctl
- [ ] Created example service unit file
- [ ] Know how to install services

### Phase 6: Persistent Sessions
- [ ] Can use screen for persistent sessions
- [ ] Can use tmux for persistent sessions
- [ ] Can detach and reattach
- [ ] Created training launcher script
- [ ] Understand when to use persistent sessions

### Phase 7: Troubleshooting
- [ ] Created diagnostic script
- [ ] Understand common issues
- [ ] Know how to diagnose hung processes
- [ ] Know how to handle OOM situations
- [ ] Can find and kill processes

### Phase 8: Validation
- [ ] All tests pass
- [ ] End-to-end workflow works
- [ ] Can answer knowledge check questions
- [ ] Understand all concepts

## Knowledge Check Answers

**1. Difference between kill -15 and kill -9?**
- kill -15 (SIGTERM): Graceful termination, can be caught by process, allows cleanup
- kill -9 (SIGKILL): Force kill, cannot be caught, immediate termination, no cleanup

**2. What does STAT 'D' mean?**
- Uninterruptible sleep, usually waiting on I/O
- Cannot be interrupted by signals (including SIGTERM)
- May indicate disk/network issues or hung I/O

**3. How to detach from screen?**
- Press Ctrl+a then d

**4. Best way to ensure training continues after SSH disconnect?**
- Use screen or tmux persistent sessions
- Or use nohup with background job (&)
- Or create systemd service

**5. How to find which process uses a specific port?**
```bash
lsof -i :8080
netstat -tulpn | grep 8080
ss -tulpn | grep 8080
```

## Next Steps

After completing this exercise, you should:

1. **Practice regularly**
   - Monitor real training jobs
   - Use persistent sessions for all long-running tasks
   - Implement proper signal handling in your scripts

2. **Move to next exercise**
   - Exercise 04: Shell Scripting - Automate these tasks
   - Exercise 05: Package Management - Install ML software
   - Exercise 06: Networking - Manage ML services

3. **Explore advanced topics**
   - cgroups for resource limiting
   - systemd timers for scheduled tasks
   - Advanced GPU management with nvidia-docker
   - Distributed process management with parallel/GNU parallel

4. **Apply to real ML workflows**
   - Set up production training pipelines
   - Implement auto-restart on failure
   - Create monitoring dashboards
   - Build CI/CD for ML models

## Additional Resources

- [Linux Process Management](https://www.digitalocean.com/community/tutorials/process-management-in-linux)
- [systemd Guide](https://www.digitalocean.com/community/tutorials/systemd-essentials-working-with-services-units-and-the-journal)
- [Screen Tutorial](https://linuxize.com/post/how-to-use-linux-screen/)
- [Tmux Cheat Sheet](https://tmuxcheatsheet.com/)
- [NVIDIA SMI Documentation](https://developer.nvidia.com/nvidia-system-management-interface)
- [Signal Handling in Python](https://docs.python.org/3/library/signal.html)

---

**Congratulations!** You have completed the Process Management and System Monitoring implementation guide. You now have the skills to effectively manage ML training workloads in production Linux environments.

**Total Time:** 2-3 hours
**Skills Acquired:** 10+ essential process management skills
**Scripts Created:** 8+ production-ready tools
**Ready for:** Exercise 04 - Shell Scripting Automation
