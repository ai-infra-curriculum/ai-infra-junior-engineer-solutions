# Step-by-Step Implementation Guide: ML Fundamentals for Infrastructure

## Overview

Master the fundamentals of ML infrastructure by setting up production-ready ML environments, monitoring GPU resources, and benchmarking performance. Learn to configure PyTorch, TensorFlow, manage CUDA installations, and build tools for GPU monitoring and environment setup automation.

**Time**: 3-4 hours | **Difficulty**: Beginner to Intermediate

---

## Prerequisites

```bash
# Check system
uname -a
lsb_release -a

# Check if NVIDIA GPU is available
lspci | grep -i nvidia

# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version  # If CUDA toolkit installed

# Install Python
python3 --version  # Should be 3.8+
pip3 --version
```

---

## Learning Objectives

By completing this exercise, you will be able to:

✅ Set up isolated Python environments for ML projects
✅ Install and configure PyTorch and TensorFlow with GPU support
✅ Verify CUDA installation and GPU availability
✅ Monitor GPU utilization in real-time
✅ Benchmark GPU performance for ML workloads
✅ Automate ML environment setup with scripts
✅ Troubleshoot common ML infrastructure issues

---

## Phase 1: Environment Setup Script (60 minutes)

### Step 1: Understanding ML Environment Management

**Why virtual environments matter**:
- Isolate dependencies between projects
- Prevent version conflicts
- Enable reproducible setups
- Simplify deployment

**Options**:
1. **venv** - Built-in Python virtual environments
2. **conda** - Package and environment manager
3. **Docker** - Containerized environments (covered in mod-005)

### Step 2: Create setup_ml_environment.sh

This script automates ML environment setup:

```bash
#!/bin/bash
#
# setup_ml_environment.sh - Automated ML environment setup
#
# Description:
#   Sets up complete ML development environment with PyTorch/TensorFlow
#   Installs CUDA-enabled versions when GPU is available
#   Configures environment variables and validates installation
#
# Usage:
#   ./setup_ml_environment.sh [OPTIONS]
#
# Options:
#   --framework FRAMEWORK    Install specific framework (pytorch|tensorflow|both)
#   --python-version VERSION Python version (default: 3.10)
#   --cuda-version VERSION   CUDA version (default: auto-detect)
#   --env-name NAME          Environment name (default: ml-env)
#   --use-conda              Use conda instead of venv
#   -v, --verbose            Verbose output
#   -h, --help               Show help
#

set -euo pipefail

# Configuration
FRAMEWORK="both"
PYTHON_VERSION="3.10"
CUDA_VERSION=""
ENV_NAME="ml-env"
USE_CONDA=false
VERBOSE=false

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_verbose() {
    if [[ "$VERBOSE" == true ]]; then
        log "DEBUG: $*"
    fi
}

# Detect CUDA version
detect_cuda() {
    log "INFO: Detecting CUDA installation..."

    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        log "SUCCESS: CUDA $CUDA_VERSION detected"
        return 0
    elif command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        log "INFO: CUDA driver version: $CUDA_VERSION (toolkit not installed)"
        return 0
    else
        log "WARNING: No CUDA installation detected, will install CPU versions"
        CUDA_VERSION="cpu"
        return 1
    fi
}

# Create virtual environment
create_venv() {
    local env_dir="$1"

    log "INFO: Creating Python virtual environment: $env_dir"

    if [[ -d "$env_dir" ]]; then
        log "WARNING: Environment already exists: $env_dir"
        read -p "Remove and recreate? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$env_dir"
        else
            log "INFO: Using existing environment"
            return 0
        fi
    fi

    python${PYTHON_VERSION} -m venv "$env_dir"
    log "SUCCESS: Virtual environment created"

    # Activate and upgrade pip
    source "$env_dir/bin/activate"
    pip install --upgrade pip setuptools wheel
}

# Create conda environment
create_conda_env() {
    local env_name="$1"

    log "INFO: Creating conda environment: $env_name"

    if conda env list | grep -q "^${env_name} "; then
        log "WARNING: Conda environment already exists: $env_name"
        read -p "Remove and recreate? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n "$env_name"
        else
            log "INFO: Using existing environment"
            conda activate "$env_name"
            return 0
        fi
    fi

    conda create -n "$env_name" python="$PYTHON_VERSION" -y
    conda activate "$env_name"
    log "SUCCESS: Conda environment created"
}

# Install PyTorch
install_pytorch() {
    log "INFO: Installing PyTorch..."

    if [[ "$CUDA_VERSION" == "cpu" ]]; then
        log "INFO: Installing CPU-only PyTorch"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    elif [[ "$CUDA_VERSION" =~ ^11\. ]]; then
        log "INFO: Installing PyTorch with CUDA 11.8"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$CUDA_VERSION" =~ ^12\. ]]; then
        log "INFO: Installing PyTorch with CUDA 12.1"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        log "WARNING: Unknown CUDA version, installing CPU version"
        pip install torch torchvision torchaudio
    fi

    # Verify installation
    python -c "import torch; print(f'PyTorch {torch.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}')"
    log "SUCCESS: PyTorch installed"
}

# Install TensorFlow
install_tensorflow() {
    log "INFO: Installing TensorFlow..."

    # TensorFlow 2.x auto-detects CUDA
    pip install tensorflow

    # Verify installation
    python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed'); print(f'GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"
    log "SUCCESS: TensorFlow installed"
}

# Install common ML packages
install_common_packages() {
    log "INFO: Installing common ML packages..."

    pip install \
        numpy \
        pandas \
        scikit-learn \
        matplotlib \
        seaborn \
        jupyter \
        jupyterlab \
        tensorboard \
        mlflow \
        wandb \
        tqdm \
        requests

    log "SUCCESS: Common packages installed"
}

# Verify GPU setup
verify_gpu() {
    log "INFO: Verifying GPU setup..."

    python <<EOF
import sys

# Check PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.version.cuda}")
        print(f"  ✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  ✓ GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  ✗ CUDA not available (CPU-only)")
except ImportError:
    print("✗ PyTorch not installed")

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  ✓ GPU count: {len(gpus)}")
        for gpu in gpus:
            print(f"  ✓ {gpu.name}")
    else:
        print("  ✗ No GPUs available (CPU-only)")
except ImportError:
    print("✗ TensorFlow not installed")
EOF

    log "SUCCESS: GPU verification complete"
}

# Create activation script
create_activation_script() {
    local env_dir="$1"
    local script_path="activate-ml-env.sh"

    cat > "$script_path" <<EOF
#!/bin/bash
# Activate ML environment

if [[ "$USE_CONDA" == true ]]; then
    conda activate $ENV_NAME
else
    source $env_dir/bin/activate
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow verbosity

echo "ML Environment activated!"
echo "Python: \$(python --version)"
echo "Location: \$(which python)"
EOF

    chmod +x "$script_path"
    log "SUCCESS: Created activation script: $script_path"
}

# Main function
main() {
    log "INFO: =========================================="
    log "INFO: ML Environment Setup"
    log "INFO: =========================================="

    # Detect CUDA
    detect_cuda

    # Create environment
    if [[ "$USE_CONDA" == true ]]; then
        create_conda_env "$ENV_NAME"
    else
        create_venv "$ENV_NAME"
    fi

    # Install frameworks
    case "$FRAMEWORK" in
        pytorch)
            install_pytorch
            ;;
        tensorflow)
            install_tensorflow
            ;;
        both)
            install_pytorch
            install_tensorflow
            ;;
    esac

    # Install common packages
    install_common_packages

    # Verify setup
    verify_gpu

    # Create activation script
    if [[ "$USE_CONDA" != true ]]; then
        create_activation_script "$ENV_NAME"
    fi

    log "INFO: =========================================="
    log "SUCCESS: ML Environment setup complete!"
    log "INFO: =========================================="
    log "INFO: To activate:"
    if [[ "$USE_CONDA" == true ]]; then
        log "INFO:   conda activate $ENV_NAME"
    else
        log "INFO:   source $ENV_NAME/bin/activate"
        log "INFO:   # or"
        log "INFO:   source activate-ml-env.sh"
    fi
}

# Parse arguments and run
# (Implementation similar to previous scripts)

main "$@"
```

### Step 3: Test Environment Setup

```bash
# Make script executable
chmod +x solutions/setup_ml_environment.sh

# Run setup (PyTorch only)
./solutions/setup_ml_environment.sh --framework pytorch --env-name pytorch-env

# Run setup (both frameworks)
./solutions/setup_ml_environment.sh --framework both

# Activate and test
source ml-env/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Phase 2: GPU Monitoring Script (60 minutes)

### Summary

The `monitor_gpu.py` script provides real-time GPU monitoring for ML workloads.

**Features**:
- Real-time GPU utilization tracking
- Memory usage monitoring
- Per-process GPU usage
- Temperature and power monitoring
- Historical data logging
- Alert on thresholds

**Core Implementation**:

```python
#!/usr/bin/env python3
"""
monitor_gpu.py - Real-time GPU monitoring for ML workloads

Usage:
    python monitor_gpu.py [--interval 5] [--log-file gpu.log]
"""

import subprocess
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List

def get_gpu_stats() -> List[Dict]:
    """Get GPU statistics using nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)

        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            gpus.append({
                'index': int(parts[0]),
                'name': parts[1],
                'gpu_util': float(parts[2]),
                'mem_util': float(parts[3]),
                'mem_used_mb': int(parts[4]),
                'mem_total_mb': int(parts[5]),
                'temp_c': float(parts[6]),
                'power_w': float(parts[7]),
                'power_limit_w': float(parts[8])
            })

        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: nvidia-smi not found or failed")
        return []

def get_gpu_processes() -> List[Dict]:
    """Get processes using GPU"""
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)

        processes = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                processes.append({
                    'pid': int(parts[0]),
                    'gpu_mem_mb': int(parts[1])
                })

        return processes
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

def display_stats(gpus: List[Dict], processes: List[Dict]):
    """Display formatted GPU statistics"""
    print("\n" + "="*80)
    print(f"GPU Monitoring - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    for gpu in gpus:
        print(f"\nGPU {gpu['index']}: {gpu['name']}")
        print(f"  Utilization:  {gpu['gpu_util']:>5.1f}%")
        print(f"  Memory:       {gpu['mem_used_mb']:>6} MB / {gpu['mem_total_mb']} MB ({gpu['mem_util']:>5.1f}%)")
        print(f"  Temperature:  {gpu['temp_c']:>5.1f}°C")
        print(f"  Power:        {gpu['power_w']:>6.1f} W / {gpu['power_limit_w']:.1f} W")

    if processes:
        print(f"\nRunning Processes: {len(processes)}")
        for proc in processes:
            print(f"  PID {proc['pid']}: {proc['gpu_mem_mb']} MB")

def monitor_loop(interval: int = 5, log_file: str = None):
    """Main monitoring loop"""
    print(f"Starting GPU monitor (interval: {interval}s)")
    print("Press Ctrl+C to stop")

    while True:
        try:
            gpus = get_gpu_stats()
            processes = get_gpu_processes()

            display_stats(gpus, processes)

            # Log to file
            if log_file:
                with open(log_file, 'a') as f:
                    data = {
                        'timestamp': datetime.now().isoformat(),
                        'gpus': gpus,
                        'processes': processes
                    }
                    f.write(json.dumps(data) + '\n')

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor GPU usage')
    parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')
    parser.add_argument('--log-file', type=str, help='Log file path')

    args = parser.parse_args()
    monitor_loop(args.interval, args.log_file)
```

---

## Phase 3: GPU Benchmark Script (60 minutes)

### Summary

The `benchmark_gpu.py` script benchmarks GPU performance for ML workloads.

**Tests**:
- Matrix multiplication performance
- Convolution operations
- Data transfer speeds (CPU ↔ GPU)
- Mixed precision performance
- Multi-GPU scaling

**Example Implementation**:

```python
import torch
import time
import numpy as np

def benchmark_matmul(size: int = 8192, iterations: int = 100):
    """Benchmark matrix multiplication"""
    print(f"\nBenchmarking Matrix Multiplication ({size}x{size})...")

    # CPU
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)

    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start

    # GPU
    if torch.cuda.is_available():
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(iterations):
            _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        speedup = cpu_time / gpu_time
        print(f"  CPU Time: {cpu_time:.3f}s")
        print(f"  GPU Time: {gpu_time:.3f}s")
        print(f"  Speedup:  {speedup:.2f}x")
    else:
        print("  GPU not available")

# Run benchmarks
benchmark_matmul(size=4096, iterations=50)
```

---

## Best Practices

1. **Always use virtual environments**: Prevent dependency conflicts
2. **Verify GPU setup**: Check CUDA compatibility before training
3. **Monitor resources**: Track GPU usage during training
4. **Pin PyTorch/TensorFlow versions**: Ensure reproducibility
5. **Test on small dataset first**: Verify setup before large jobs

---

## Troubleshooting

### CUDA Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA toolkit
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory Errors
```python
# Reduce batch size
batch_size = 16  # Instead of 64

# Clear CUDA cache
torch.cuda.empty_cache()

# Use gradient checkpointing
model.gradient_checkpointing_enable()
```

---

## Next Steps

1. Set up Jupyter Lab for interactive development
2. Create training monitoring dashboards
3. Implement automated GPU allocation
4. Build multi-GPU training pipelines

---

**Congratulations!** You've set up a production-ready ML infrastructure environment! 🚀
