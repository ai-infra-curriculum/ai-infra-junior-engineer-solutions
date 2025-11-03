# Exercise 07: Real-World Troubleshooting Scenarios

Production-ready troubleshooting solutions for common ML infrastructure issues encountered by junior AI infrastructure engineers.

## Overview

This exercise provides comprehensive investigation and fix scripts for six critical scenarios that frequently occur in ML infrastructure environments. Each scenario includes detailed diagnostic tools and automated remediation scripts.

## Scenarios

### Scenario 1: Disk Full Error

**Problem:** Training jobs fail with "No space left on device" errors
- **Location:** `/var/ml/checkpoints/` filling up with old model checkpoints
- **Impact:** Cannot save model checkpoints, training fails

**Tools:**
- `scenario1/investigate.sh` - Comprehensive disk space analysis
- `scenario1/cleanup.sh` - Automated cleanup with safety features

**Usage:**
```bash
cd scenario1

# Investigate disk usage
./investigate.sh

# Preview cleanup (dry-run)
./cleanup.sh --dry-run

# Perform cleanup
./cleanup.sh

# Aggressive cleanup
./cleanup.sh --aggressive
```

**Key Features:**
- Identifies space-consuming directories and files
- Keeps last N checkpoints automatically
- Compresses old files
- Cleans package caches, Docker resources, logs
- Shows before/after disk usage
- Prevention recommendations

### Scenario 2: Permission Denied

**Problem:** Cannot access model files or data directories
- **Error:** `PermissionError: [Errno 13] Permission denied`
- **Impact:** Training cannot read/write files

**Tools:**
- `scenario2/investigate.sh` - Comprehensive permission analysis
- `scenario2/fix.sh` - Automated permission fixes

**Usage:**
```bash
cd scenario2

# Investigate permission issues
./investigate.sh /data/models/checkpoint.pth

# Auto-detect and fix
./fix.sh /data/models/checkpoint.pth

# Change ownership
./fix.sh /data/models --method owner --user mluser

# Fix parent directory permissions
./fix.sh /data/models --method parent

# Preview changes
./fix.sh /data/models --dry-run
```

**Key Features:**
- Analyzes file/directory ownership and permissions
- Checks parent directory traversal permissions
- Tests ACLs and SELinux contexts
- Multiple fix methods (owner, group, chmod, parent)
- Recursive operations for directories
- Best practices for team access

### Scenario 3: Hung Process

**Problem:** Training process appears frozen or unresponsive
- **Symptoms:** High CPU but no progress, or stuck in D state
- **Impact:** Resources wasted, training not progressing

**Tools:**
- `scenario3/investigate.sh` - Process state analysis
- `scenario3/kill.sh` - Graceful termination with escalation

**Usage:**
```bash
cd scenario3

# Find hung processes
ps aux | grep python | grep train

# Investigate process (replace PID)
./investigate.sh 12345

# Graceful termination
./kill.sh 12345

# Force termination
./kill.sh 12345 --force

# Kill process tree
./kill.sh 12345 --children
```

**Key Features:**
- Process state analysis (R, S, D, Z, T states)
- System call tracing with strace
- File descriptor analysis
- Deadlock detection
- Graceful termination with signal escalation (TERM → INT → QUIT → KILL)
- Handles uninterruptible sleep (D state)

### Scenario 4: Out of Memory (OOM)

**Problem:** Training killed by OOM killer
- **Error:** `OSError: [Errno 28]` or process simply "Killed"
- **Impact:** Training fails, data loss

**Tools:**
- `scenario4/investigate.sh` - Memory usage analysis
- `scenario4/optimize.sh` - Memory optimization and swap setup

**Usage:**
```bash
cd scenario4

# Investigate memory issues
./investigate.sh

# Show recommendations only
./optimize.sh --show-recommendations

# Add 8GB swap space
./optimize.sh --add-swap 8G

# Tune VM parameters for ML
./optimize.sh --tune-vm

# Full optimization
./optimize.sh --add-swap 4G --tune-vm
```

**Key Features:**
- OOM killer event detection
- Memory consumption analysis
- Swap space management
- VM parameter tuning (swappiness, cache pressure)
- Application-level recommendations:
  - Batch size reduction
  - Data generator usage
  - Gradient checkpointing
  - Mixed precision training
  - GPU memory management

### Scenario 5: CUDA/GPU Not Available

**Problem:** GPU not detected or CUDA errors
- **Error:** `RuntimeError: CUDA not available`
- **Impact:** Cannot use GPU acceleration

**Tools:**
- `scenario5/investigate.sh` - GPU and CUDA diagnostics
- `scenario5/fix.sh` - Driver and environment setup

**Usage:**
```bash
cd scenario5

# Investigate GPU availability
./investigate.sh

# Verify GPU is working
./fix.sh --verify

# Setup CUDA environment variables
./fix.sh --setup-env

# Install NVIDIA driver (CAUTION!)
./fix.sh --install-driver
```

**Key Features:**
- GPU hardware detection
- Driver version checking
- CUDA toolkit verification
- Environment variable validation
- Library path checking
- PyTorch/TensorFlow GPU support testing
- Permission verification
- Nouveau driver conflict detection
- Version compatibility checking

### Scenario 6: Network Connectivity

**Problem:** Cannot download models, API timeouts, DNS failures
- **Symptoms:** Timeouts, connection refused, name resolution failures
- **Impact:** Cannot download data, models, or packages

**Tools:**
- `scenario6/investigate.sh` - Network diagnostics
- `scenario6/fix.sh` - Network configuration fixes

**Usage:**
```bash
cd scenario6

# Investigate connectivity (tests huggingface.co by default)
./investigate.sh

# Test specific host
./investigate.sh pytorch.org

# Test connectivity to common services
./fix.sh --test-connectivity

# Fix DNS issues
./fix.sh --fix-dns

# Configure proxy
./fix.sh --configure-proxy http://proxy.company.com:8080
```

**Key Features:**
- Interface and gateway checking
- DNS resolution testing (nslookup, dig, getent)
- External connectivity verification
- Proxy detection and configuration
- Firewall rule analysis
- Port connectivity testing
- SSL/TLS certificate verification
- System time synchronization check
- VPN/tunnel detection

## Script Features

### Common Features Across All Scripts

1. **Colored Output**
   - ✓ Green for success
   - ⚠ Yellow for warnings
   - ✗ Red for errors
   - ℹ Blue for information

2. **Dry-Run Mode**
   ```bash
   ./cleanup.sh --dry-run
   ./fix.sh --dry-run
   ```
   - Preview changes before execution
   - Safe exploration of potential fixes

3. **Comprehensive Help**
   ```bash
   ./script.sh --help
   ```
   - Detailed usage information
   - Examples for common scenarios

4. **Safety Features**
   - Confirmation prompts for dangerous operations
   - Backup of configuration files
   - Validation checks before modifications
   - Clear warning messages

## Troubleshooting Workflow

### General Approach

1. **Investigate**
   ```bash
   cd scenarioN
   ./investigate.sh [args]
   ```
   - Gather diagnostic information
   - Identify root cause
   - Review analysis summary

2. **Plan Fix**
   - Review investigation output
   - Determine appropriate solution
   - Consider using --dry-run first

3. **Apply Fix**
   ```bash
   ./fix.sh [options]
   # or
   ./cleanup.sh [options]
   # or
   ./kill.sh [options]
   # or
   ./optimize.sh [options]
   ```

4. **Verify**
   - Test that issue is resolved
   - Check for side effects
   - Document solution

### Quick Reference

| Issue | Quick Check | Quick Fix |
|-------|-------------|-----------|
| Disk full | `df -h` | `./scenario1/cleanup.sh` |
| Permission denied | `ls -la /path/to/file` | `./scenario2/fix.sh /path/to/file` |
| Hung process | `ps aux \| grep process` | `./scenario3/kill.sh PID` |
| Out of memory | `free -h` | `./scenario4/optimize.sh --add-swap 4G` |
| GPU not found | `nvidia-smi` | `./scenario5/fix.sh --setup-env` |
| Network down | `ping 8.8.8.8` | `./scenario6/fix.sh --fix-dns` |

## Directory Structure

```
exercise-07/
├── README.md                   # This file
├── docs/
│   └── ANSWERS.md             # Reflection questions
├── scenario1/                  # Disk Full
│   ├── investigate.sh
│   └── cleanup.sh
├── scenario2/                  # Permission Denied
│   ├── investigate.sh
│   └── fix.sh
├── scenario3/                  # Hung Process
│   ├── investigate.sh
│   └── kill.sh
├── scenario4/                  # Out of Memory
│   ├── investigate.sh
│   └── optimize.sh
├── scenario5/                  # CUDA/GPU
│   ├── investigate.sh
│   └── fix.sh
└── scenario6/                  # Network Connectivity
    ├── investigate.sh
    └── fix.sh
```

## Prerequisites

### Required Packages

Most scripts will work with standard Linux utilities, but some scenarios benefit from additional tools:

```bash
# Debian/Ubuntu
sudo apt install \
    iproute2 \
    net-tools \
    dnsutils \
    curl \
    wget \
    netcat \
    strace \
    lsof \
    iotop \
    sysstat \
    acl

# Optional for GPU scenarios
sudo apt install \
    nvidia-driver-525 \
    nvidia-cuda-toolkit
```

### Python Packages (for GPU verification)

```bash
pip install torch torchvision  # PyTorch
# or
pip install tensorflow         # TensorFlow
```

## Best Practices

### Prevention

1. **Disk Space**
   - Set up automatic cleanup cron jobs
   - Implement retention policies in training code
   - Monitor disk usage with alerts
   - Use separate partitions for ML data

2. **Permissions**
   - Use dedicated groups for team access
   - Set appropriate umask (002 for shared environments)
   - Use ACLs for fine-grained control
   - Regular permission audits

3. **Process Management**
   - Add timeouts to long-running operations
   - Implement health checks and auto-restart
   - Monitor for deadlocks
   - Use proper signal handling in code

4. **Memory Management**
   - Profile memory usage during development
   - Use data generators instead of loading all data
   - Implement gradient checkpointing
   - Monitor with alerts at 80% usage
   - Configure swap space as buffer

5. **GPU Management**
   - Verify GPU before training
   - Use environment modules for CUDA
   - Document version requirements
   - Test after driver updates

6. **Network**
   - Document proxy requirements
   - Handle timeouts gracefully in code
   - Use retry logic with exponential backoff
   - Cache downloads when possible

### Monitoring

Set up monitoring for:
- Disk space (alert at 80%)
- Memory usage (alert at 80%)
- Process states (alert on D state)
- Network connectivity (alert on failures)
- GPU availability (alert if not detected)

### Documentation

Always document:
- What went wrong
- How you diagnosed it
- What fixed it
- How to prevent it
- Who to contact for help

## Common Commands

### Disk Management
```bash
df -h                                    # Check disk usage
du -sh /path/*                          # Directory sizes
find /path -type f -size +100M          # Large files
ncdu /path                              # Interactive disk usage
```

### Process Management
```bash
ps aux | grep process                    # Find process
ps -p PID -o stat,wchan,cmd             # Process details
lsof -p PID                             # Open files
strace -p PID                           # System calls
kill -TERM PID                          # Graceful kill
```

### Memory Management
```bash
free -h                                  # Memory usage
vmstat 1                                # Virtual memory stats
ps aux --sort=-%mem | head              # Top memory users
cat /proc/PID/status | grep Vm          # Process memory
```

### Network Debugging
```bash
ip addr                                  # Interface info
ip route                                # Routing table
ping 8.8.8.8                            # Test connectivity
nslookup google.com                     # DNS test
curl -I https://site.com                # HTTP test
nc -zv host 443                         # Port test
```

### GPU Management
```bash
lspci | grep -i nvidia                  # GPU hardware
nvidia-smi                              # GPU status
nvcc --version                          # CUDA version
python -c "import torch; print(torch.cuda.is_available())"
```

## Support and Resources

### Internal Resources
- See `docs/ANSWERS.md` for reflection questions
- Check learning repository for detailed explanations
- Review module lecture notes for theory

### External Resources
- Linux man pages: `man command`
- NVIDIA documentation: https://docs.nvidia.com/
- PyTorch troubleshooting: https://pytorch.org/docs/stable/
- TensorFlow GPU guide: https://www.tensorflow.org/install/gpu

## Contributing

When adding new scenarios or improving scripts:
1. Follow existing script structure
2. Include comprehensive help text
3. Add safety features (dry-run, confirmations)
4. Test thoroughly before committing
5. Update this README

## License

Part of AI Infrastructure Junior Engineer Learning curriculum.

---

**Note:** These scripts are educational tools. Always test in non-production environments first and understand what each script does before running with elevated privileges.
