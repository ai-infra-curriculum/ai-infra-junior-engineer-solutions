# Exercise 07: Troubleshooting - Reflection Questions

## Overview Questions

### 1. What makes troubleshooting in production ML systems different from general software troubleshooting?

**Answer:**

ML systems introduce unique troubleshooting challenges:

**Resource Intensity:**
- **Large Data Volumes:** ML workloads process massive datasets, making disk space management critical. A single checkpoint can be gigabytes, and forgetting to clean old ones quickly fills disks.
- **Memory Requirements:** Models with millions/billions of parameters require substantial RAM/VRAM. OOM errors are common and often require careful memory profiling.
- **GPU Dependencies:** Deep learning relies on GPUs, adding layers of complexity (drivers, CUDA versions, library compatibility).

**Long-Running Processes:**
- Training jobs can run for hours/days/weeks
- Hung processes waste expensive GPU time
- Checkpointing is critical for recovery
- Process state (R, S, D, Z, T) matters more than in typical applications

**Complex Dependencies:**
- Tight coupling between driver versions, CUDA toolkit, and ML frameworks
- PyTorch/TensorFlow versions must match CUDA versions
- Incompatibilities often manifest as runtime errors, not build failures

**Data Pipeline Issues:**
- Network connectivity crucial for downloading models/datasets
- Distributed training adds inter-node communication debugging
- Data loading bottlenecks can appear as hung processes

**Reproducibility Challenges:**
- Non-deterministic behavior in distributed training
- GPU memory fragmentation issues
- Race conditions in multi-process data loading

### 2. Why is a systematic troubleshooting approach important?

**Answer:**

A systematic approach is essential for effective troubleshooting:

**Efficiency:**
- **Saves Time:** Following a methodical process (investigate → identify → fix → verify) prevents random trial-and-error
- **Reduces Downtime:** Faster diagnosis means less disruption to training pipelines
- **Example:** Instead of randomly restarting processes, checking `df -h` immediately identifies disk full errors

**Completeness:**
- **Catches Root Causes:** Surface symptoms often mask deeper issues
- **Prevents Recurrence:** Understanding the root cause enables prevention strategies
- **Example:** Fixing permissions with `chmod 777` (symptom treatment) vs. implementing proper group-based access (root cause fix)

**Documentation:**
- **Knowledge Transfer:** Systematic logs help team members solve similar issues
- **Pattern Recognition:** Repeated issues become apparent when documented consistently
- **Audit Trail:** Critical for post-incident reviews

**Risk Mitigation:**
- **Prevents New Problems:** Hasty fixes can break other things
- **Validates Solutions:** The "verify" step confirms the fix actually worked
- **Example:** Using `--dry-run` before disk cleanup prevents accidental data deletion

**Learning:**
- **Builds Expertise:** Understanding *why* a fix works improves future troubleshooting
- **Creates Playbooks:** Systematic approaches can be codified into runbooks
- **Enables Automation:** Well-documented processes can be scripted

**The Troubleshooting Workflow:**
1. **Investigate:** Gather data systematically (logs, metrics, state)
2. **Identify:** Analyze data to find root cause
3. **Fix:** Apply targeted solution
4. **Verify:** Confirm issue is resolved
5. **Document:** Record findings for future reference
6. **Prevent:** Implement measures to avoid recurrence

### 3. What are the most critical skills for effective troubleshooting?

**Answer:**

**Technical Skills:**

1. **System Analysis:**
   - Reading logs effectively (dmesg, journalctl, application logs)
   - Understanding process states and signals
   - Interpreting resource metrics (CPU, memory, disk, network)
   - Using diagnostic tools (strace, lsof, netstat, nvidia-smi)

2. **Pattern Recognition:**
   - Recognizing common error signatures
   - Identifying cascading failures
   - Spotting resource exhaustion patterns
   - Example: "Killed" message + high memory usage = OOM killer

3. **Tool Proficiency:**
   - Command-line expertise (grep, awk, find)
   - Monitoring tools (top, htop, vmstat)
   - Network debugging (ping, curl, nslookup)
   - GPU tools (nvidia-smi, nvcc)

**Analytical Skills:**

4. **Root Cause Analysis:**
   - Distinguishing symptoms from causes
   - Using "5 Whys" technique
   - Understanding system dependencies
   - Example: "Training slow" → "High I/O wait" → "Disk full" → "No checkpoint cleanup" → "Missing cron job"

5. **Hypothesis Testing:**
   - Forming testable hypotheses
   - Designing experiments to validate theories
   - Using controlled changes to isolate variables
   - Example: Testing DNS by switching from domain name to IP address

**Soft Skills:**

6. **Clear Communication:**
   - Documenting findings clearly
   - Explaining technical issues to non-technical stakeholders
   - Writing effective incident reports
   - Creating actionable runbooks

7. **Patience and Persistence:**
   - Not jumping to conclusions
   - Following through entire diagnostic process
   - Staying calm under pressure
   - Learning from each incident

8. **Risk Assessment:**
   - Understanding impact of potential fixes
   - Weighing urgency vs. safety
   - Knowing when to escalate
   - Using safety nets (backups, dry-runs, staging environments)

**Learning Skills:**

9. **Continuous Learning:**
   - Reading man pages and documentation
   - Understanding underlying systems (Linux kernel, networking stack)
   - Staying current with ML framework changes
   - Building mental models of system behavior

10. **Meta-Skills:**
    - Knowing what you don't know
    - Asking good questions
    - Seeking help effectively
    - Learning from post-mortems

## Scenario-Specific Questions

### Scenario 1: Disk Full

**Q: Why is disk space management particularly critical for ML workloads?**

**A:**

1. **Large File Sizes:**
   - Model checkpoints can be 100MB - 10GB+ each
   - Datasets often exceed available RAM, stored on disk
   - Intermediate training artifacts accumulate quickly

2. **Continuous Generation:**
   - Checkpoints saved every epoch/N steps
   - Logs and metrics files grow continuously
   - TensorBoard logs can consume significant space

3. **Cascading Failures:**
   - Full disk prevents checkpoint saves → potential data loss
   - Cannot write logs → harder to debug other issues
   - System instability when root filesystem fills

4. **Prevention Strategies:**
   - Implement retention policies (keep last N checkpoints)
   - Compress old checkpoints (gzip reduces size 5-10x)
   - Use separate partition for ML data
   - Monitor with alerts at 80% usage
   - Automated cleanup cron jobs

**Q: What's the difference between deleting files and compressing them?**

**A:**

**Deletion:**
- Permanently removes files
- Frees disk space immediately
- Irreversible (unless backed up elsewhere)
- Use for: truly obsolete data, duplicates, temp files

**Compression (gzip):**
- Reduces file size (typically 80-90% for model checkpoints)
- Data remains accessible (can be decompressed)
- Good for: old checkpoints you might need for analysis
- Trade-off: CPU time to compress/decompress

**Best Practice:**
- Keep last N checkpoints uncompressed (fast access)
- Compress checkpoints older than X days
- Delete checkpoints older than Y days (after archiving to S3/GCS)
- Example: Keep last 5 uncompressed, compress 5-30 days old, archive >30 days

### Scenario 2: Permission Denied

**Q: Explain the difference between user, group, and other permissions.**

**A:**

**Linux Permission Model:**

```
-rw-r--r-- 1 alice mlteam 1024 Jan 01 12:00 model.pth
│││ │ │      │     │
│││ │ └──────┼─────┼── Other permissions (everyone else)
│││ └────────┼─────── Group permissions (mlteam)
││└──────────────── User permissions (alice, the owner)
│└───────────────── File type (- = regular file, d = directory)
└────────────────── Special bits (setuid/setgid/sticky)
```

**Permission Bits (rwx):**
- **r (4):** Read - view file contents or list directory
- **w (2):** Write - modify file or create/delete files in directory
- **x (1):** Execute - run file as program or access (cd into) directory

**Three Permission Groups:**

1. **User (Owner):**
   - Applies only to the file owner
   - Usually the creator of the file
   - Change with: `chown newuser file`

2. **Group:**
   - Applies to users in the file's group
   - Enables team access without broad permissions
   - Change with: `chown :newgroup file` or `chgrp newgroup file`

3. **Other (World):**
   - Applies to everyone else
   - Typically most restrictive
   - Security risk if too permissive

**Numeric Notation:**
```
chmod 755 file
      │││
      ││└── Other: r-x (4+1=5)
      │└─── Group: r-x (4+1=5)
      └──── User:  rwx (4+2+1=7)
```

**Best Practices for ML Teams:**
- Use group permissions for team collaboration
- Create dedicated groups (mlteam, mldevs)
- Set umask 002 for automatic group write access
- Use setgid bit on directories: `chmod g+s /data/shared/`
- Regular permission audits

**Q: When should you use chmod vs. chown vs. adding users to groups?**

**A:**

**Use `chmod` when:**
- File has correct ownership but wrong permission bits
- Need to add/remove specific permissions
- Fixing individual file access issues
- Examples:
  ```bash
  chmod u+rw file.txt        # Give owner read/write
  chmod g+rx /shared/dir     # Give group read/execute
  chmod 644 data.csv         # Standard file permissions
  ```

**Use `chown` when:**
- File owned by wrong user/group
- Transferring ownership after copying files as root
- Setting up new resources for team
- Examples:
  ```bash
  chown alice:mlteam model.pth      # Change owner and group
  chown alice file.txt              # Change owner only
  chown :mlteam /data/shared        # Change group only
  chown -R mluser:mlteam /data/     # Recursive ownership
  ```

**Use `usermod -aG group user` when:**
- Multiple users need access to shared resources
- Implementing team-based access control
- Better than giving "other" permissions (more secure)
- Requires logout/login to take effect
- Examples:
  ```bash
  sudo usermod -aG mlteam bob       # Add bob to mlteam group
  sudo usermod -aG docker alice     # Allow alice to use Docker
  sudo usermod -aG video mluser     # Allow GPU device access
  ```

**Decision Tree:**
1. Is the file owned by the right user/group?
   - No → Use `chown`
   - Yes → Continue

2. Do multiple users need access?
   - Yes → Add users to group with `usermod -aG`
   - No → Continue

3. Are permission bits incorrect?
   - Yes → Use `chmod`

**Example Scenario:**
```bash
# Problem: Alice can't access Bob's model file
ls -la model.pth
# -rw------- 1 bob bob 1024 Jan 01 12:00 model.pth

# Solution 1: Change ownership to shared group
sudo chown bob:mlteam model.pth
sudo chmod g+rw model.pth
sudo usermod -aG mlteam alice
# Alice logs out/in, can now access

# Solution 2: Transfer ownership
sudo chown alice:alice model.pth
# Alice now owns it, simple but doesn't solve team access

# Solution 3: Use ACLs (more advanced)
setfacl -m u:alice:rw model.pth
# Grants Alice specific access without changing ownership
```

### Scenario 3: Hung Process

**Q: What are the different process states and what do they indicate?**

**A:**

**Process States (from ps stat column):**

1. **R - Running/Runnable:**
   - Currently executing on CPU or waiting in run queue
   - Normal state for active processes
   - If stuck in R with high CPU but no progress → infinite loop
   - Example: Training actively computing gradients

2. **S - Interruptible Sleep:**
   - Waiting for an event (I/O, lock, timer)
   - Can be interrupted by signals
   - Most common state for idle processes
   - Example: Process waiting for user input or network data

3. **D - Uninterruptible Sleep:**
   - Waiting for I/O or kernel operation
   - **Cannot be killed with regular signals!**
   - Usually brief, but can indicate:
     - Disk I/O problems (dying drive, full filesystem)
     - NFS mount issues (unreachable server)
     - Kernel bugs
   - Example: Process stuck reading from NFS mount
   - **Solution:** Fix underlying I/O issue or reboot

4. **Z - Zombie (Defunct):**
   - Process terminated but parent hasn't reaped it
   - Takes no resources except PID table entry
   - Cosmetic issue unless many zombies accumulate
   - Cannot be killed (already dead!)
   - Example: Child process exited but parent in infinite loop
   - **Solution:** Kill parent process to clean up zombie

5. **T - Stopped:**
   - Suspended by job control signal (Ctrl+Z) or debugger
   - Can be resumed with SIGCONT or `fg`
   - Example: User accidentally pressed Ctrl+Z
   - **Solution:** `kill -CONT PID` or `fg`

6. **X - Dead (rare):**
   - Process terminating, very brief state
   - Shouldn't see this in ps output

**Additional State Modifiers:**
- **< :** High priority (nice < 0)
- **N :** Low priority (nice > 0)
- **L :** Has pages locked in memory
- **s :** Session leader
- **l :** Multi-threaded
- **+ :** In foreground process group

**Troubleshooting by State:**

```bash
# Find processes in D state (may indicate problems)
ps aux | awk '$8 ~ /D/ {print $0}'

# Find zombie processes
ps aux | awk '$8 ~ /Z/ {print $0}'

# Check what process is waiting on (D state)
cat /proc/PID/wchan  # Shows kernel function process is waiting in

# Get stack trace (D state diagnosis)
sudo cat /proc/PID/stack
```

**Real-World Example:**
```
USER  PID  STAT  TIME COMMAND
alice 1234 R     99:30 python train.py    # Running, high CPU
bob   5678 S     0:02  python server.py   # Sleeping, normal
carol 9012 D     1:45  rsync backup/      # Stuck on I/O - problem!
dave  3456 Z     0:00  [python] <defunct> # Zombie, harmless
eve   7890 T     5:20  python debug.py    # Stopped, resume with fg
```

**Q: Explain the signal escalation strategy.**

**A:**

**Signal Escalation Approach:**

Signals are how the OS communicates with processes. Different signals offer different levels of "politeness" for termination:

**Level 1: SIGTERM (15) - Polite Request**
```bash
kill -TERM PID  # or just: kill PID
```
- Default signal
- Asks process to terminate gracefully
- Process can catch signal and clean up:
  - Close files
  - Save state
  - Release locks
  - Flush buffers
  - Send shutdown notifications
- Best practice: Always try SIGTERM first
- Wait: 10-30 seconds

**Level 2: SIGINT (2) - Interrupt**
```bash
kill -INT PID   # Same as Ctrl+C
```
- Interrupt signal (like keyboard Ctrl+C)
- Slightly more forceful than SIGTERM
- Still allows cleanup
- Some processes handle INT but not TERM
- Wait: 5-10 seconds

**Level 3: SIGQUIT (3) - Quit with Core Dump**
```bash
kill -QUIT PID  # Same as Ctrl+\
```
- Causes process to dump core for debugging
- Useful if you need to analyze why process hung
- Creates core dump file (if enabled)
- Wait: 5 seconds

**Level 4: SIGKILL (9) - Forced Termination**
```bash
kill -9 PID     # Force kill
```
- **Cannot be caught or ignored**
- Kernel immediately terminates process
- No cleanup possible:
  - Files may be left open
  - Locks may not be released
  - Data may be corrupted
  - Temporary files may remain
- Last resort only!
- Use when:
  - Process doesn't respond to other signals
  - Immediate termination required
  - Process is consuming critical resources

**Exception: SIGSTOP/SIGCONT (Pause/Resume)**
```bash
kill -STOP PID   # Pause process (cannot be caught)
kill -CONT PID   # Resume process
```
- SIGSTOP pauses without terminating
- SIGCONT resumes execution
- Useful for temporarily freezing a process

**Cannot Kill With Signals:**
- **D state processes:** Waiting on kernel/I/O, must fix underlying issue
- **Zombie processes:** Already dead, parent must reap them

**Implementation in kill.sh:**
```bash
# Try SIGTERM
kill -TERM $PID
wait 10 seconds
if still alive:
    # Try SIGINT
    kill -INT $PID
    wait 5 seconds
    if still alive:
        # Try SIGQUIT
        kill -QUIT $PID
        wait 5 seconds
        if still alive:
            # Force with SIGKILL
            kill -9 $PID
```

**Real-World Wisdom:**
- Production services should handle SIGTERM gracefully
- Always log which signal you're using
- Document why you needed to escalate
- Investigate why SIGTERM didn't work (indicates bug)
- Never script automatic SIGKILL without human approval

### Scenario 4: Out of Memory

**Q: How does the Linux OOM killer decide which process to kill?**

**A:**

**OOM Killer Mechanism:**

When system runs critically low on memory and cannot free any more, the Linux kernel's **Out-Of-Memory (OOM) killer** selects and terminates processes to prevent system crash.

**Trigger Conditions:**
- Physical RAM exhausted
- Swap space full (if configured)
- Memory allocation request fails
- Kernel cannot evict more pages from cache

**Selection Process:**

1. **Calculate OOM Score for Each Process:**
   ```bash
   # View OOM scores
   cat /proc/*/oom_score | sort -n

   # View for specific process
   cat /proc/PID/oom_score
   ```

   Score based on:
   - **Memory Usage:** Larger processes score higher
   - **Runtime:** Younger processes score higher (older = more work invested)
   - **Importance:** Root processes score lower
   - **Nice Value:** Lower priority processes score higher
   - **OOM Adjustment:** Manual tuning (see below)

2. **OOM Score = (Total VM Used) × 1000 / (Total RAM)**
   - Normalized to 0-1000
   - Higher score = more likely to be killed
   - Process using 50% RAM gets ~500 score

3. **Manual Adjustment (oom_score_adj):**
   ```bash
   # View current adjustment
   cat /proc/PID/oom_score_adj

   # Values: -1000 to +1000
   # -1000 = Never kill (disable OOM for this process)
   # 0 = Default behavior
   # +1000 = Always kill first

   # Protect critical process
   echo -1000 > /proc/PID/oom_score_adj

   # Make process sacrificial
   echo 1000 > /proc/PID/oom_score_adj
   ```

4. **Kill Highest Scoring Process:**
   - Send SIGKILL (cannot be caught)
   - Process immediately terminated
   - Memory freed
   - Logged to dmesg/kernel log

**Checking for OOM Events:**
```bash
# Recent OOM kills
dmesg -T | grep -i "out of memory"
dmesg -T | grep -i "killed process"

# Identify victim
dmesg -T | grep "Out of memory: Killed process"
# Example output:
# Out of memory: Killed process 1234 (python) total-vm:8GB, anon-rss:6GB

# Journalctl view
journalctl -k --since "24 hours ago" | grep -i "oom"
```

**Why ML Training is Vulnerable:**
- Large model parameters consume gigabytes
- Batch data loaded into memory
- Gradients stored for backpropagation
- Multiple processes (data loaders) compete for memory
- Sudden spikes during forward pass

**Prevention Strategies:**

1. **Reserve Memory for System:**
   ```bash
   # Set minimum free memory
   sysctl vm.min_free_kbytes=1048576  # 1GB
   ```

2. **Protect Critical Processes:**
   ```bash
   # Database, monitoring, SSH
   echo -1000 > /proc/$(pidof sshd)/oom_score_adj
   ```

3. **Make Training Jobs Sacrificial:**
   ```bash
   # Training can be restarted from checkpoint
   echo 500 > /proc/$(pidof python)/oom_score_adj
   ```

4. **Monitor Memory Usage:**
   ```bash
   # Alert at 80% usage
   watch -n 10 'free -m | awk "NR==2 {print \$3/\$2*100}"'
   ```

5. **Configure cgroups:**
   ```bash
   # Limit memory for specific process
   systemd-run --scope -p MemoryLimit=8G python train.py
   ```

**Q: What are the trade-offs of using swap space?**

**A:**

**Swap Space:**
Disk space used as virtual RAM when physical memory is exhausted.

**Advantages:**

1. **Prevents OOM Kills:**
   - Provides emergency buffer
   - Process slows instead of dies
   - Opportunity to intervene

2. **Enables Larger Workloads:**
   - Can run memory-intensive jobs that exceed physical RAM
   - Good for: data processing, compilation, batch jobs
   - Not ideal for: real-time, low-latency applications

3. **Memory Pressure Relief:**
   - Swaps out idle pages (e.g., SSH daemon loaded at boot but rarely used)
   - Frees physical RAM for active processes
   - Kernel decides what to swap based on usage patterns

**Disadvantages:**

1. **Performance Degradation:**
   - **Disk is 1000x+ slower than RAM**
   - SSD: ~500MB/s vs RAM: ~50GB/s
   - HDD: ~100MB/s vs RAM: ~50GB/s
   - Swap thrashing: system spends all time swapping, no useful work

2. **Training Slowdown:**
   - ML training requires fast memory access
   - Swapped training data → extreme slowdown
   - GPU sits idle waiting for data from slow swap
   - May make training impractical

3. **Disk Wear:**
   - SSD have limited write cycles
   - Heavy swapping degrades SSD lifespan
   - Less concern for HDDs

4. **Masking Problems:**
   - Swap allows oversized workloads to "work"
   - Hiding the fact that you need more RAM
   - Better to fix root cause (reduce memory usage)

**Best Practices for ML:**

1. **Add Swap as Emergency Buffer (not primary solution):**
   ```bash
   # Create 4GB swap file
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile

   # Make permanent
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

2. **Tune Swappiness (how aggressively to swap):**
   ```bash
   # Default: 60 (fairly aggressive)
   # ML recommendation: 10 (swap only when really needed)
   sudo sysctl vm.swappiness=10

   # Make permanent
   echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
   ```

   **Swappiness values:**
   - `0`: Swap only to avoid OOM
   - `10`: Swap conservatively (good for servers, ML)
   - `60`: Default (balanced)
   - `100`: Aggressive swapping

3. **Monitor Swap Usage:**
   ```bash
   # Check swap usage
   free -h
   swapon --show

   # Watch for swap activity
   vmstat 1
   # si/so columns: swap in/out
   # High si/so = thrashing (bad!)
   ```

4. **Sizing:**
   - **Desktops:** 1-2x physical RAM
   - **Servers:** 0.5x RAM or 4GB minimum
   - **ML Training:** 0.25-0.5x RAM (emergency buffer)
   - **Production ML Inference:** Often no swap (predictable latency)

**When to Use Swap:**
- ✓ Emergency buffer to prevent OOM
- ✓ Occasional memory spikes
- ✓ Swap out idle processes
- ✗ Regular training workload (get more RAM instead)
- ✗ Latency-sensitive production inference
- ✗ As primary memory solution

**Real-World Scenario:**
```
System: 16GB RAM, 0GB swap
Training job needs 18GB → OOM killed immediately

System: 16GB RAM, 8GB swap, swappiness=10
Training job needs 18GB → Uses 2GB swap, slows down 20%, completes successfully
You get results but know you need more RAM

System: 16GB RAM, 8GB swap, swappiness=60
Training job needs 14GB → Swaps aggressively, thrashes, takes 10x longer
Poor performance despite fitting in RAM due to aggressive swapping
```

### Scenario 5: CUDA/GPU

**Q: What are the common causes of CUDA version mismatches?**

**A:**

CUDA version mismatches are one of the most frustrating ML infrastructure issues. Understanding the layers involved is key:

**The CUDA Stack:**
```
┌────────────────────────────────┐
│ Application (PyTorch, TF)      │  ← Framework built for specific CUDA
├────────────────────────────────┤
│ CUDA Toolkit (nvcc, cuDNN)     │  ← Development libraries
├────────────────────────────────┤
│ CUDA Runtime (cudart)          │  ← Runtime libraries
├────────────────────────────────┤
│ NVIDIA Driver                   │  ← Kernel module, supports CUDA versions
├────────────────────────────────┤
│ GPU Hardware                    │  ← Physical device
└────────────────────────────────┘
```

**Common Mismatch Scenarios:**

1. **Driver Too Old for CUDA Toolkit:**
   ```bash
   nvidia-smi  # Shows: CUDA Version: 11.4
   nvcc --version  # Shows: release 12.0

   # Problem: Driver supports max CUDA 11.4, but toolkit is 12.0
   # Solution: Update driver OR downgrade CUDA toolkit
   ```

2. **PyTorch Built for Wrong CUDA Version:**
   ```bash
   nvidia-smi  # Driver supports CUDA 12.0
   python -c "import torch; print(torch.version.cuda)"  # Shows: 11.7

   # Problem: PyTorch built for CUDA 11.7, system has 12.0
   # Solution: Install PyTorch for correct CUDA version:
   pip install torch --index-url https://download.pytorch.org/whl/cu120
   ```

3. **Multiple CUDA Installations:**
   ```bash
   which nvcc  # /usr/local/cuda-11.8/bin/nvcc
   echo $LD_LIBRARY_PATH  # Includes /usr/local/cuda-12.0/lib64

   # Problem: nvcc from 11.8 but libraries from 12.0
   # Solution: Ensure CUDA_HOME, PATH, LD_LIBRARY_PATH all point to same version
   ```

4. **Conda vs. System CUDA Conflict:**
   ```bash
   # System CUDA: 12.0
   # Conda environment has its own CUDA: 11.7

   # Problem: Environment variables point to system, conda packages expect conda's CUDA
   # Solution: Let conda manage CUDA entirely:
   conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch
   ```

5. **Missing cuDNN:**
   ```bash
   # TensorFlow/PyTorch require cuDNN (deep learning primitives)
   python train.py
   # Error: cannot find libcudnn.so.8

   # Solution: Install cuDNN matching CUDA version
   ```

**How to Check Versions:**

```bash
# 1. Driver version and max supported CUDA
nvidia-smi
# Look for: "CUDA Version: 12.0" (this is MAX supported)

# 2. Installed CUDA toolkit
nvcc --version
ls /usr/local/cuda*/version.txt

# 3. PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available())"

# 4. TensorFlow CUDA version
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 5. Runtime library version
ldconfig -p | grep libcudart
```

**Version Compatibility Matrix:**

| Driver Version | Max CUDA Support | PyTorch CUDA Versions | TensorFlow Versions |
|----------------|------------------|----------------------|---------------------|
| ≥525.60        | 12.0             | cu118, cu121         | 2.15+ |
| ≥515.43        | 11.7             | cu116, cu117         | 2.11-2.14 |
| ≥470.57        | 11.4             | cu113, cu115         | 2.8-2.10 |
| ≥450.80        | 11.0             | cu110, cu111         | 2.4-2.7 |

**Key Principle:**
- **Driver is backward compatible:** Driver 525 (supports CUDA 12.0) can run applications built for CUDA 11.x
- **Applications are forward-locked:** App built for CUDA 12.0 cannot run on driver that only supports 11.x

**Prevention Strategies:**

1. **Document Required Versions:**
   ```yaml
   # requirements.txt or environment.yml
   # NVIDIA Driver: >=515.43
   # CUDA Toolkit: 11.7
   torch==2.0.0+cu117
   ```

2. **Use Environment Modules:**
   ```bash
   # Load specific CUDA version
   module load cuda/11.7
   ```

3. **Virtual Environments:**
   ```bash
   # Isolated Python + CUDA environment
   conda create -n ml_cu117 python=3.10
   conda activate ml_cu117
   conda install pytorch pytorch-cuda=11.7 -c pytorch
   ```

4. **Container Images:**
   ```dockerfile
   # Pin exact versions
   FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
   ```

5. **Automated Checks:**
   ```bash
   # Pre-training verification script
   python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
   ```

**Q: How do you verify GPU is accessible from Python?**

**A:**

**Complete GPU Verification Checklist:**

**1. Hardware Level:**
```bash
# Check GPU is detected by system
lspci | grep -i nvidia
# Should show something like: "NVIDIA Corporation GA102 [GeForce RTX 3090]"

# Check device files exist
ls -l /dev/nvidia*
# Should show: /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-uvm

# Check current user can access device
[ -r /dev/nvidia0 ] && [ -w /dev/nvidia0 ] && echo "Access OK" || echo "No access"
```

**2. Driver Level:**
```bash
# Check driver loaded
nvidia-smi
# Should show GPU information, driver version, CUDA version

# Check kernel modules
lsmod | grep nvidia
# Should show: nvidia, nvidia_uvm, nvidia_modeset, etc.
```

**3. CUDA Toolkit Level:**
```bash
# Check CUDA installed
which nvcc
nvcc --version

# Check environment variables
echo $CUDA_HOME        # Should point to /usr/local/cuda-XX.X
echo $LD_LIBRARY_PATH  # Should include $CUDA_HOME/lib64
```

**4. Python Level:**

**PyTorch Verification:**
```python
import torch

# Basic checks
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # Should be True

if torch.cuda.is_available():
    # Detailed information
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")

    # Memory info
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    # Actual computation test
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print("✓ GPU computation successful")
        print(f"Result shape: {z.shape}")
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
else:
    print("✗ CUDA not available")
    print("\nTroubleshooting steps:")
    print("1. Check nvidia-smi works")
    print("2. Verify PyTorch CUDA version matches system CUDA")
    print("3. Reinstall PyTorch with CUDA support")
```

**TensorFlow Verification:**
```python
import tensorflow as tf

# Basic checks
print(f"TensorFlow version: {tf.__version__}")

# GPU devices
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu}")

        # Get GPU details
        details = tf.config.experimental.get_device_details(gpu)
        print(f"  Compute capability: {details.get('compute_capability')}")

    # Test computation
    try:
        with tf.device('/GPU:0'):
            x = tf.random.normal([1000, 1000])
            y = tf.random.normal([1000, 1000])
            z = tf.matmul(x, y)
        print("✓ GPU computation successful")
        print(f"Result shape: {z.shape}")
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
else:
    print("✗ No GPUs available")
    print("\nTroubleshooting steps:")
    print("1. Check nvidia-smi works")
    print("2. Verify TensorFlow-GPU is installed (not just TensorFlow)")
    print("3. Check CUDA and cuDNN versions match TensorFlow requirements")
```

**5. Comprehensive Test Script:**
```python
#!/usr/bin/env python3
"""
Comprehensive GPU verification script
"""
import subprocess
import sys

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)

def check_hardware():
    """Check GPU hardware"""
    print("="*60)
    print("1. Hardware Check")
    print("="*60)

    success, output = run_command("lspci | grep -i nvidia")
    if success and output:
        print(f"✓ NVIDIA GPU detected:\n{output}")
        return True
    else:
        print("✗ No NVIDIA GPU detected")
        return False

def check_driver():
    """Check NVIDIA driver"""
    print("\n" + "="*60)
    print("2. Driver Check")
    print("="*60)

    success, output = run_command("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
    if success and output:
        print(f"✓ Driver working:\n{output}")
        return True
    else:
        print("✗ nvidia-smi failed - driver not loaded")
        return False

def check_cuda():
    """Check CUDA toolkit"""
    print("\n" + "="*60)
    print("3. CUDA Toolkit Check")
    print("="*60)

    success, output = run_command("nvcc --version | grep release")
    if success and output:
        print(f"✓ CUDA toolkit installed:\n{output}")
        return True
    else:
        print("⚠ nvcc not found (toolkit may not be installed or not in PATH)")
        return False

def check_pytorch():
    """Check PyTorch GPU support"""
    print("\n" + "="*60)
    print("4. PyTorch Check")
    print("="*60)

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✓ CUDA available in PyTorch")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU 0: {torch.cuda.get_device_name(0)}")

            # Test computation
            x = torch.randn(100, 100, device='cuda')
            y = x @ x.T
            print(f"✓ GPU computation works")
            return True
        else:
            print("✗ CUDA not available in PyTorch")
            return False
    except ImportError:
        print("⚠ PyTorch not installed")
        return False
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
        return False

def check_tensorflow():
    """Check TensorFlow GPU support"""
    print("\n" + "="*60)
    print("5. TensorFlow Check")
    print("="*60)

    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ {len(gpus)} GPU(s) available in TensorFlow")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")

            # Test computation
            with tf.device('/GPU:0'):
                x = tf.random.normal([100, 100])
                y = tf.matmul(x, x, transpose_b=True)
            print(f"✓ GPU computation works")
            return True
        else:
            print("✗ No GPUs available in TensorFlow")
            return False
    except ImportError:
        print("⚠ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"✗ TensorFlow error: {e}")
        return False

def main():
    """Run all checks"""
    print("\nGPU Verification Script")
    print("="*60)

    results = {
        'Hardware': check_hardware(),
        'Driver': check_driver(),
        'CUDA': check_cuda(),
        'PyTorch': check_pytorch(),
        'TensorFlow': check_tensorflow()
    }

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:15} {status}")

    all_critical = results['Hardware'] and results['Driver']
    if all_critical:
        print("\n✓ GPU is accessible!")
    else:
        print("\n✗ GPU is NOT accessible - check failed items above")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

**Save as `verify_gpu.py` and run:**
```bash
python verify_gpu.py
```

This script provides a complete diagnostic of GPU accessibility at all levels, making troubleshooting straightforward.

### Scenario 6: Network Connectivity

**Q: What's the difference between DNS issues and routing issues?**

**A:**

**DNS (Domain Name System) Issues:**

**What DNS Does:**
- Translates human-readable names (`huggingface.co`) to IP addresses (`18.65.165.10`)
- Like a phone book for the internet
- Client sends name → DNS server responds with IP

**Symptoms of DNS Problems:**
```bash
# This fails (uses DNS)
ping huggingface.co
# ping: huggingface.co: Name or service not known

# But this works (uses IP directly, no DNS needed)
ping 8.8.8.8
# PING 8.8.8.8: 64 bytes from 8.8.8.8: time=10ms
```

**Diagnostic Commands:**
```bash
# Test DNS resolution
nslookup huggingface.co
# If fails: DNS not working

# Check DNS servers
cat /etc/resolv.conf
# Should show: nameserver 8.8.8.8 (or your DNS servers)

# Test specific DNS server
nslookup huggingface.co 8.8.8.8
# If this works but regular nslookup fails: your configured DNS is broken
```

**Common Causes:**
1. **No DNS servers configured:** `/etc/resolv.conf` empty or wrong
2. **DNS servers unreachable:** Firewall blocking port 53
3. **DNS servers down:** Corporate DNS server offline
4. **Wrong DNS servers:** Typo in configuration

**Solution:**
```bash
# Use public DNS (Google, Cloudflare)
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
echo "nameserver 1.1.1.1" | sudo tee -a /etc/resolv.conf

# Or via systemd-resolved
echo "DNS=8.8.8.8 1.1.1.1" | sudo tee /etc/systemd/resolved.conf.d/dns.conf
sudo systemctl restart systemd-resolved
```

---

**Routing Issues:**

**What Routing Does:**
- Determines path packets take through the network
- Decides which network interface to use
- Directs packets to next hop toward destination

**Symptoms of Routing Problems:**
```bash
# DNS works (can resolve name)
nslookup huggingface.co
# Server: 8.8.8.8
# Address: 18.65.165.10

# But cannot reach destination
ping 18.65.165.10
# No response or "Network is unreachable"
```

**Diagnostic Commands:**
```bash
# Check routing table
ip route show
# Should show:
# default via 192.168.1.1 dev eth0  ← Default gateway
# 192.168.1.0/24 dev eth0           ← Local network

# No default route = routing problem!

# Trace route to destination
traceroute huggingface.co
# Shows each hop packets take
# Where it stops = routing problem location

# Test gateway reachable
ping $(ip route | grep default | awk '{print $3}')
# If fails: can't reach gateway (local routing problem)
```

**Common Causes:**
1. **No default gateway:** `ip route` shows no `default via...`
2. **Wrong gateway:** Gateway IP is incorrect
3. **Gateway down:** Router/gateway machine is offline
4. **Network interface down:** `ip link show` shows interface DOWN

**Solution:**
```bash
# Add default route
sudo ip route add default via 192.168.1.1 dev eth0

# Bring interface up
sudo ip link set eth0 up

# Restart networking
sudo systemctl restart networking
```

---

**Visual Comparison:**

**DNS Issue:**
```
Your Computer
    │
    │ "What's the IP of huggingface.co?"
    ├──────────> DNS Server (broken/unreachable)
    │                ✗ No response or wrong answer
    └──────────> Can't even start trying to connect

Solution: Fix DNS configuration
```

**Routing Issue:**
```
Your Computer
    │
    │ "What's the IP of huggingface.co?"
    ├──────────> DNS Server ✓ (works fine)
    │ <─────────── "It's 18.65.165.10"
    │
    │ "How do I reach 18.65.165.10?"
    ├──────────> Routing Table (broken)
    └──────────> ✗ "No route to host" or packet dies

Solution: Fix routing configuration
```

**Combined Test:**
```bash
#!/bin/bash
# Test DNS vs Routing

echo "Testing DNS..."
if nslookup google.com > /dev/null 2>&1; then
    echo "✓ DNS working"
else
    echo "✗ DNS broken - fix /etc/resolv.conf"
    exit 1
fi

echo "Testing routing..."
if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
    echo "✓ Routing working"
else
    echo "✗ Routing broken - check 'ip route'"
    exit 1
fi

echo "Testing combined (DNS + routing)..."
if ping -c 1 google.com > /dev/null 2>&1; then
    echo "✓ Both working!"
else
    echo "✗ Still broken despite DNS and routing OK"
    echo "  Check: firewall, proxy, network interface"
fi
```

**Q: How do proxy settings affect ML workflows?**

**A:**

Proxies are intermediaries that sit between your machine and the internet, common in corporate/institutional environments for security and monitoring.

**How Proxies Work:**
```
Your Code
    │
    │ HTTP/HTTPS request
    ├──────────> Proxy Server
    │                │
    │                │ Forwards request
    │                ├──────────> Internet (HuggingFace, PyPI, etc.)
    │                │ <───────── Response
    │ <─────────── Forwards response
```

**Impact on ML Workflows:**

**1. Model/Dataset Downloads:**
```python
# Without proxy awareness:
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
# ✗ Error: ConnectionError: HTTPSConnectionPool

# With proxy:
import os
os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
os.environ['HTTPS_PROXY'] = 'http://proxy.company.com:8080'
model = AutoModel.from_pretrained("bert-base-uncased")
# ✓ Works
```

**2. Package Installation:**
```bash
# pip
pip install torch
# ✗ Error: Could not fetch URL

# With proxy:
pip install --proxy http://proxy:8080 torch
# Or set environment variable:
export HTTP_PROXY=http://proxy:8080
export HTTPS_PROXY=http://proxy:8080
pip install torch

# Configure permanently:
pip config set global.proxy http://proxy:8080
```

**3. Git Operations:**
```bash
# Clone without proxy awareness:
git clone https://github.com/huggingface/transformers.git
# ✗ Error: Failed to connect

# Configure git proxy:
git config --global http.proxy http://proxy:8080
git config --global https.proxy http://proxy:8080
```

**4. Docker Image Pulls:**
```bash
# Configure Docker daemon
# /etc/docker/daemon.json:
{
  "proxies": {
    "default": {
      "httpProxy": "http://proxy:8080",
      "httpsProxy": "http://proxy:8080",
      "noProxy": "localhost,127.0.0.1"
    }
  }
}

sudo systemctl restart docker
```

**5. Application-Specific Proxy:**
```python
# Python requests library
import requests

proxies = {
    'http': 'http://proxy:8080',
    'https': 'http://proxy:8080'
}

response = requests.get('https://huggingface.co/models', proxies=proxies)

# PyTorch download
import torch
torch.hub.set_dir('/path/to/cache')  # Use cached models to avoid proxy issues

# TensorFlow datasets
import tensorflow_datasets as tfds
tfds.core.download.urllib_request.install_proxy({'http': 'http://proxy:8080'})
```

**Common Proxy Configurations:**

**Environment Variables (most common):**
```bash
# In ~/.bashrc or ~/.profile
export HTTP_PROXY="http://proxy.company.com:8080"
export HTTPS_PROXY="http://proxy.company.com:8080"
export NO_PROXY="localhost,127.0.0.1,192.168.0.0/16"
export http_proxy="$HTTP_PROXY"  # Lowercase for compatibility
export https_proxy="$HTTPS_PROXY"
export no_proxy="$NO_PROXY"
```

**Authenticated Proxies:**
```bash
# With username/password
export HTTP_PROXY="http://username:password@proxy:8080"
export HTTPS_PROXY="http://username:password@proxy:8080"

# URL-encode special characters in password:
# @ → %40, : → %3A, / → %2F
```

**NO_PROXY (bypass list):**
```bash
# Hosts that should NOT go through proxy
export NO_PROXY="localhost,127.0.0.1,*.internal.company.com,192.168.0.0/16"
```

**Troubleshooting Proxy Issues:**

**1. Test Proxy Connection:**
```bash
# Test proxy is reachable
nc -zv proxy.company.com 8080

# Test HTTP through proxy
curl -x http://proxy:8080 http://google.com

# Test HTTPS through proxy
curl -x http://proxy:8080 https://google.com
```

**2. Common Errors:**
```python
# Error: "407 Proxy Authentication Required"
# Solution: Add credentials to proxy URL

# Error: "SSL: CERTIFICATE_VERIFY_FAILED"
# Cause: Proxy does SSL inspection, breaks certificate chain
# Solution (NOT RECOMMENDED for production):
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Better solution: Install proxy's CA certificate
```

**3. Debug Proxy Settings:**
```python
import os
import urllib.request

# Check if proxy is configured
proxy = os.environ.get('HTTP_PROXY', 'Not set')
print(f"HTTP_PROXY: {proxy}")

# Test with urllib
proxy_handler = urllib.request.ProxyHandler({
    'http': proxy,
    'https': proxy
})
opener = urllib.request.build_opener(proxy_handler)
try:
    response = opener.open('http://google.com', timeout=5)
    print(f"✓ Proxy working: {response.code}")
except Exception as e:
    print(f"✗ Proxy error: {e}")
```

**Best Practices:**

1. **Environment Variables:** Set globally in shell RC file
2. **NO_PROXY:** Always exclude localhost and internal IPs
3. **Security:** Don't hardcode credentials in code, use environment variables
4. **Documentation:** Document proxy requirements in README
5. **Fallback:** Cache models/datasets locally to reduce proxy dependency
6. **Testing:** Always test proxy configuration before production use

**Complete Setup Script:**
```bash
#!/bin/bash
# setup_proxy.sh

PROXY_HOST="proxy.company.com"
PROXY_PORT="8080"
PROXY_URL="http://${PROXY_HOST}:${PROXY_PORT}"

# Create proxy environment file
cat > ~/.proxy_env << EOF
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export NO_PROXY="localhost,127.0.0.1,*.internal,192.168.0.0/16"
export no_proxy="\${NO_PROXY}"
EOF

# Source in shell RC files
for rc in ~/.bashrc ~/.zshrc; do
    if [ -f "$rc" ]; then
        echo "source ~/.proxy_env" >> "$rc"
    fi
done

# Configure tools
pip config set global.proxy "${PROXY_URL}"
git config --global http.proxy "${PROXY_URL}"
git config --global https.proxy "${PROXY_URL}"

# Configure APT (Debian/Ubuntu)
echo "Acquire::http::Proxy \"${PROXY_URL}\";" | sudo tee /etc/apt/apt.conf.d/proxy.conf

echo "✓ Proxy configured. Run 'source ~/.bashrc' to apply."
```

## General Reflection

**Q: What did you learn from implementing these troubleshooting scenarios?**

**A:**

**1. Systematic Approach is Essential:**
   - Random fixes waste time and can make things worse
   - Investigation → Identification → Fix → Verification workflow works
   - Documentation prevents solving the same problem repeatedly

**2. Safety First:**
   - Always use `--dry-run` before destructive operations
   - Confirm before deleting files or killing processes
   - Backup configuration files before modifying
   - Test in non-production first

**3. Context Matters:**
   - Same symptom can have multiple causes (e.g., "slow training" could be disk, memory, GPU, or network)
   - Need to understand the full system (hardware, OS, drivers, frameworks, application)
   - ML infrastructure adds complexity beyond typical software engineering

**4. Automation is Powerful:**
   - Scripts codify knowledge and make troubleshooting repeatable
   - But scripts need safety features (confirmations, dry-runs, validation)
   - Well-written scripts serve as documentation

**5. User Experience in Scripts:**
   - Colored output dramatically improves readability
   - Clear help messages reduce support burden
   - Comprehensive error messages guide users toward solutions
   - Examples in help text are invaluable

**6. Prevention > Cure:**
   - Monitoring and alerting catch problems early
   - Automation (cron jobs for cleanup) prevents issues
   - Good documentation prevents mistakes
   - Learning from incidents improves system design

**7. ML-Specific Challenges:**
   - Resource management (disk, memory, GPU) is critical
   - Long-running processes need special handling
   - Complex dependency chains (driver → CUDA → framework)
   - Network reliability crucial for distributed training and data loading

**8. Communication:**
   - Scripts should explain what they're doing and why
   - Analysis summaries help non-experts understand issues
   - Next steps guidance empowers users to self-serve
   - Good documentation reduces interrupt-driven work

**Q: How would you prioritize these troubleshooting skills in a learning path?**

**A:**

**Foundation (Master First):**

1. **Basic Linux CLI:**
   - Navigation, file operations, permissions
   - Process management (ps, kill, signals)
   - Resource monitoring (top, df, free)
   - **Why first:** Everything else builds on this

2. **Log Analysis:**
   - Reading system logs (dmesg, journalctl)
   - Application logs
   - Pattern matching (grep, awk)
   - **Why:** Logs are your primary diagnostic tool

3. **Disk Management:**
   - Understanding filesystems
   - Space monitoring and cleanup
   - Compression and archiving
   - **Why:** Disk full is common and prevents all work

**Intermediate (Build On Foundation):**

4. **Process Troubleshooting:**
   - Process states and signals
   - Using strace, lsof
   - Graceful vs. forced termination
   - **Why:** Hung processes waste resources

5. **Memory Management:**
   - Understanding RAM vs. swap
   - Memory profiling
   - OOM killer
   - **Why:** OOM kills destroy training runs

6. **Permissions and Security:**
   - User/group/other model
   - chmod, chown, ACLs
   - Troubleshooting access issues
   - **Why:** Common in team environments

**Advanced (ML-Specific):**

7. **Network Debugging:**
   - DNS, routing, proxies
   - Firewall rules
   - SSL/TLS issues
   - **Why:** Critical for distributed training and data loading

8. **GPU/CUDA Troubleshooting:**
   - Driver installation and management
   - CUDA version compatibility
   - Environment configuration
   - **Why:** Essential for deep learning, complex dependency chain

**Meta-Skills (Develop Throughout):**

9. **Systematic Methodology:**
   - Hypothesis-driven debugging
   - Root cause analysis
   - Documentation habits
   - **Why:** Multiplies effectiveness of technical skills

10. **Scripting and Automation:**
    - Bash scripting for repetitive tasks
    - Safety features in automation
    - Building reusable tools
    - **Why:** Scales your impact

**Learning Path:**

**Week 1-2: Foundation**
- Basic Linux commands and navigation
- Log file reading and grep
- Disk space management

**Week 3-4: Intermediate I**
- Process management and signals
- Memory monitoring and optimization
- File permissions and access control

**Week 5-6: Intermediate II**
- Network basics and troubleshooting
- Scripting fundamentals
- Safety practices

**Week 7-8: Advanced**
- GPU/CUDA setup and debugging
- Complex dependency management
- System-level optimization

**Ongoing:**
- Practice on real systems
- Document solutions
- Build personal troubleshooting toolkit
- Learn from incidents and post-mortems

**Key Principle:** Each level builds on the previous. Don't skip foundations even if GPUs seem more interesting. You can't debug CUDA if you can't read logs or understand processes.

---

## Conclusion

Troubleshooting is a core skill for AI infrastructure engineers. The combination of systematic methodology, technical knowledge, and practical experience enables effective problem-solving in production ML environments.

These scenarios cover the most common issues, but real-world troubleshooting requires continuous learning and adaptation. Every incident is an opportunity to improve systems, documentation, and skills.

**Remember:**
- **Investigate thoroughly** before making changes
- **Document everything** for yourself and teammates
- **Automate repetitive tasks** to scale your impact
- **Learn from failures** to prevent recurrence
- **Help others learn** by sharing knowledge

Building expertise in troubleshooting transforms you from someone who can follow instructions to someone who can diagnose and solve novel problems independently—a critical skill for any infrastructure engineer.
