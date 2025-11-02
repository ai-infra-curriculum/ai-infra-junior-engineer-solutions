# Implementation Guide: GPU Computing Fundamentals

## Overview

This comprehensive guide provides production-ready implementations for GPU computing in machine learning infrastructure. It covers everything from basic GPU detection to advanced memory optimization techniques, multi-GPU setups, and production-grade monitoring utilities.

**Target Audience:** Junior AI Infrastructure Engineers
**Prerequisites:** Python, PyTorch basics, Linux command line
**Estimated Time:** 4-6 hours

## Table of Contents

1. [Environment Setup and GPU Detection](#1-environment-setup-and-gpu-detection)
2. [PyTorch GPU Operations](#2-pytorch-gpu-operations)
3. [GPU Memory Monitoring and Optimization](#3-gpu-memory-monitoring-and-optimization)
4. [Multi-GPU Basics](#4-multi-gpu-basics)
5. [Performance Profiling](#5-performance-profiling)
6. [Common GPU Errors and Debugging](#6-common-gpu-errors-and-debugging)
7. [Production GPU Management Utilities](#7-production-gpu-management-utilities)

---

## 1. Environment Setup and GPU Detection

### 1.1 System Requirements Check

Before starting, verify your system has the necessary components:

```bash
# Check for NVIDIA GPU
lspci | grep -i nvidia

# Check NVIDIA driver installation
nvidia-smi

# Check CUDA installation
nvcc --version

# Expected output from nvidia-smi:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA RTX 3060     Off  | 00000000:01:00.0 Off |                  N/A |
# | 30%   45C    P8    12W / 170W |      0MiB / 12288MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

### 1.2 PyTorch Installation with CUDA Support

```bash
# Check Python version (requires 3.8+)
python3 --version

# Create virtual environment
python3 -m venv gpu-ml-env
source gpu-ml-env/bin/activate

# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional dependencies
pip install transformers datasets accelerate
pip install psutil gpustat py3nvml
pip install matplotlib pandas tensorboard

# Verify installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 1.3 Production-Grade GPU Detection Module

Create `gpu_detector.py`:

```python
#!/usr/bin/env python3
"""
Production GPU Detection and Information Module

This module provides comprehensive GPU detection, validation,
and information gathering for production ML infrastructure.
"""

import sys
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import torch
import subprocess


@dataclass
class GPUInfo:
    """Data class for GPU information."""
    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory_gb: float
    allocated_memory_gb: float
    reserved_memory_gb: float
    free_memory_gb: float
    multi_processor_count: int
    is_available: bool
    temperature: Optional[float] = None
    power_usage_w: Optional[float] = None
    utilization_percent: Optional[float] = None


class GPUDetector:
    """GPU detection and monitoring utility."""

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0

    def get_system_info(self) -> Dict:
        """Get comprehensive system information."""
        info = {
            "cuda_available": self.cuda_available,
            "device_count": self.device_count,
            "cuda_version": torch.version.cuda if self.cuda_available else None,
            "cudnn_version": torch.backends.cudnn.version() if self.cuda_available else None,
            "cudnn_enabled": torch.backends.cudnn.enabled if self.cuda_available else False,
            "pytorch_version": torch.__version__,
            "python_version": sys.version.split()[0],
        }
        return info

    def get_gpu_info(self, device_id: int = 0) -> Optional[GPUInfo]:
        """Get detailed information for a specific GPU."""
        if not self.cuda_available or device_id >= self.device_count:
            return None

        props = torch.cuda.get_device_properties(device_id)

        gpu_info = GPUInfo(
            device_id=device_id,
            name=torch.cuda.get_device_name(device_id),
            compute_capability=torch.cuda.get_device_capability(device_id),
            total_memory_gb=props.total_memory / 1024**3,
            allocated_memory_gb=torch.cuda.memory_allocated(device_id) / 1024**3,
            reserved_memory_gb=torch.cuda.memory_reserved(device_id) / 1024**3,
            free_memory_gb=(props.total_memory - torch.cuda.memory_allocated(device_id)) / 1024**3,
            multi_processor_count=props.multi_processor_count,
            is_available=True
        )

        # Try to get additional metrics from nvidia-smi
        try:
            gpu_info.temperature = self._get_gpu_temperature(device_id)
            gpu_info.power_usage_w = self._get_gpu_power(device_id)
            gpu_info.utilization_percent = self._get_gpu_utilization(device_id)
        except Exception:
            pass  # nvidia-smi not available or failed

        return gpu_info

    def get_all_gpus_info(self) -> List[GPUInfo]:
        """Get information for all available GPUs."""
        return [
            self.get_gpu_info(i)
            for i in range(self.device_count)
        ]

    def _get_gpu_temperature(self, device_id: int) -> Optional[float]:
        """Get GPU temperature in Celsius."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu',
                 '--format=csv,noheader,nounits', f'--id={device_id}'],
                capture_output=True, text=True, timeout=5
            )
            return float(result.stdout.strip())
        except Exception:
            return None

    def _get_gpu_power(self, device_id: int) -> Optional[float]:
        """Get GPU power usage in watts."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw',
                 '--format=csv,noheader,nounits', f'--id={device_id}'],
                capture_output=True, text=True, timeout=5
            )
            return float(result.stdout.strip())
        except Exception:
            return None

    def _get_gpu_utilization(self, device_id: int) -> Optional[float]:
        """Get GPU utilization percentage."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu',
                 '--format=csv,noheader,nounits', f'--id={device_id}'],
                capture_output=True, text=True, timeout=5
            )
            return float(result.stdout.strip())
        except Exception:
            return None

    def validate_cuda_setup(self) -> Tuple[bool, List[str]]:
        """
        Validate CUDA setup and return validation status with messages.

        Returns:
            Tuple of (is_valid, list of messages)
        """
        messages = []
        is_valid = True

        # Check CUDA availability
        if not self.cuda_available:
            is_valid = False
            messages.append("❌ CUDA is not available")
            messages.append("   - Check if NVIDIA drivers are installed: nvidia-smi")
            messages.append("   - Verify CUDA toolkit installation: nvcc --version")
            messages.append("   - Ensure PyTorch was installed with CUDA support")
            return is_valid, messages

        messages.append("✅ CUDA is available")

        # Check device count
        if self.device_count == 0:
            is_valid = False
            messages.append("❌ No CUDA devices detected")
            return is_valid, messages

        messages.append(f"✅ {self.device_count} GPU(s) detected")

        # Check compute capability (minimum 3.5 for modern PyTorch)
        for i in range(self.device_count):
            capability = torch.cuda.get_device_capability(i)
            if capability[0] < 3 or (capability[0] == 3 and capability[1] < 5):
                is_valid = False
                messages.append(
                    f"❌ GPU {i}: Compute capability {capability} is too low "
                    f"(minimum 3.5 required)"
                )
            else:
                messages.append(
                    f"✅ GPU {i}: Compute capability {capability} is supported"
                )

        # Check cuDNN
        if torch.backends.cudnn.enabled:
            messages.append(f"✅ cuDNN is enabled (version {torch.backends.cudnn.version()})")
        else:
            messages.append("⚠️  cuDNN is not enabled (performance may be reduced)")

        return is_valid, messages

    def print_detailed_report(self):
        """Print a comprehensive GPU report."""
        print("=" * 70)
        print("GPU DETECTION AND VALIDATION REPORT")
        print("=" * 70)

        # System info
        system_info = self.get_system_info()
        print("\n📊 System Information:")
        print(f"  PyTorch Version: {system_info['pytorch_version']}")
        print(f"  Python Version:  {system_info['python_version']}")
        print(f"  CUDA Available:  {system_info['cuda_available']}")

        if system_info['cuda_available']:
            print(f"  CUDA Version:    {system_info['cuda_version']}")
            print(f"  cuDNN Version:   {system_info['cudnn_version']}")
            print(f"  cuDNN Enabled:   {system_info['cudnn_enabled']}")
            print(f"  GPU Count:       {system_info['device_count']}")

        # Validation
        print("\n🔍 CUDA Setup Validation:")
        is_valid, messages = self.validate_cuda_setup()
        for msg in messages:
            print(f"  {msg}")

        # GPU details
        if self.cuda_available and self.device_count > 0:
            print("\n💻 GPU Details:")
            for gpu_info in self.get_all_gpus_info():
                print(f"\n  GPU {gpu_info.device_id}: {gpu_info.name}")
                print(f"    Compute Capability:  {gpu_info.compute_capability}")
                print(f"    Total Memory:        {gpu_info.total_memory_gb:.2f} GB")
                print(f"    Allocated Memory:    {gpu_info.allocated_memory_gb:.2f} GB")
                print(f"    Free Memory:         {gpu_info.free_memory_gb:.2f} GB")
                print(f"    Multi-Processors:    {gpu_info.multi_processor_count}")

                if gpu_info.temperature is not None:
                    print(f"    Temperature:         {gpu_info.temperature}°C")
                if gpu_info.power_usage_w is not None:
                    print(f"    Power Usage:         {gpu_info.power_usage_w:.1f} W")
                if gpu_info.utilization_percent is not None:
                    print(f"    GPU Utilization:     {gpu_info.utilization_percent}%")

        # Test tensor creation
        print("\n🧪 Device Testing:")
        test_results = self._test_tensor_creation()
        for result in test_results:
            print(f"  {result}")

        print("\n" + "=" * 70)
        if is_valid:
            print("✅ GPU setup is valid and ready for ML workloads")
        else:
            print("❌ GPU setup has issues - please review messages above")
        print("=" * 70)

    def _test_tensor_creation(self) -> List[str]:
        """Test tensor creation on available devices."""
        results = []

        # Test CPU
        try:
            cpu_tensor = torch.randn(1000, 1000)
            results.append(f"✅ CPU tensor created: {cpu_tensor.shape} on {cpu_tensor.device}")
        except Exception as e:
            results.append(f"❌ CPU tensor creation failed: {e}")

        # Test GPU
        if self.cuda_available:
            for i in range(self.device_count):
                try:
                    gpu_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
                    results.append(
                        f"✅ GPU {i} tensor created: {gpu_tensor.shape} on {gpu_tensor.device}"
                    )
                except Exception as e:
                    results.append(f"❌ GPU {i} tensor creation failed: {e}")

        return results

    def export_json(self, filepath: str):
        """Export GPU information to JSON file."""
        data = {
            "system_info": self.get_system_info(),
            "gpus": [asdict(gpu) for gpu in self.get_all_gpus_info()],
            "validation": {
                "is_valid": self.validate_cuda_setup()[0],
                "messages": self.validate_cuda_setup()[1]
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"GPU information exported to {filepath}")


def main():
    """Main entry point."""
    detector = GPUDetector()
    detector.print_detailed_report()

    # Optionally export to JSON
    # detector.export_json("gpu_info.json")


if __name__ == "__main__":
    main()
```

**Usage:**

```bash
python3 gpu_detector.py

# Export to JSON for programmatic access
python3 -c "from gpu_detector import GPUDetector; GPUDetector().export_json('gpu_info.json')"
```

---

## 2. PyTorch GPU Operations

### 2.1 Device Management Best Practices

Create `device_manager.py`:

```python
#!/usr/bin/env python3
"""
Production Device Management Module

Handles device selection, tensor movement, and device-aware
model operations for production ML infrastructure.
"""

import torch
import time
from typing import Optional, Union, List, Dict
from contextlib import contextmanager


class DeviceManager:
    """Manages device selection and tensor operations across CPU/GPU."""

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """
        Initialize device manager.

        Args:
            device: Specific device to use, or None for auto-selection
        """
        if device is None:
            self.device = self._auto_select_device()
        else:
            self.device = torch.device(device)

        self.device_type = self.device.type
        self.device_index = self.device.index if self.device.type == 'cuda' else None

        print(f"DeviceManager initialized on: {self.device}")

    def _auto_select_device(self) -> torch.device:
        """Automatically select the best available device."""
        if not torch.cuda.is_available():
            return torch.device('cpu')

        # If multiple GPUs, select the one with most free memory
        if torch.cuda.device_count() > 1:
            free_memory = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                free = props.total_memory - allocated
                free_memory.append((i, free))

            # Sort by free memory and select the one with most
            best_gpu = max(free_memory, key=lambda x: x[1])[0]
            return torch.device(f'cuda:{best_gpu}')

        return torch.device('cuda:0')

    def to_device(self,
                  obj: Union[torch.Tensor, torch.nn.Module, Dict, List],
                  non_blocking: bool = False) -> Union[torch.Tensor, torch.nn.Module, Dict, List]:
        """
        Move object to the managed device.

        Args:
            obj: Tensor, model, dict, or list to move
            non_blocking: Use non-blocking transfer (async)

        Returns:
            Object on the target device
        """
        if isinstance(obj, (torch.Tensor, torch.nn.Module)):
            return obj.to(self.device, non_blocking=non_blocking)
        elif isinstance(obj, dict):
            return {k: self.to_device(v, non_blocking) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.to_device(item, non_blocking) for item in obj]
        else:
            return obj

    def benchmark_transfer(self, size: int = 1000) -> Dict[str, float]:
        """
        Benchmark data transfer speeds.

        Args:
            size: Size of square tensor to transfer

        Returns:
            Dictionary with transfer timing results
        """
        if self.device_type != 'cuda':
            return {"error": "GPU not available for transfer benchmark"}

        # Create tensor on CPU
        cpu_tensor = torch.randn(size, size)
        tensor_size_mb = (cpu_tensor.element_size() * cpu_tensor.nelement()) / 1024**2

        results = {
            "tensor_size_mb": tensor_size_mb,
            "tensor_shape": f"{size}x{size}",
        }

        # Benchmark CPU -> GPU (blocking)
        start = time.time()
        gpu_tensor = cpu_tensor.to(self.device)
        torch.cuda.synchronize()
        results["cpu_to_gpu_ms"] = (time.time() - start) * 1000

        # Benchmark CPU -> GPU (non-blocking)
        start = time.time()
        gpu_tensor_nb = cpu_tensor.to(self.device, non_blocking=True)
        torch.cuda.synchronize()
        results["cpu_to_gpu_nonblocking_ms"] = (time.time() - start) * 1000

        # Benchmark GPU -> CPU
        start = time.time()
        cpu_tensor_back = gpu_tensor.to('cpu')
        torch.cuda.synchronize()
        results["gpu_to_cpu_ms"] = (time.time() - start) * 1000

        # Calculate throughput
        results["throughput_gb_per_sec"] = (
            tensor_size_mb / 1024
        ) / (results["cpu_to_gpu_ms"] / 1000)

        return results

    @contextmanager
    def temporary_device(self, device: Union[str, torch.device]):
        """
        Context manager for temporarily using a different device.

        Usage:
            with device_manager.temporary_device('cuda:1'):
                # Operations on cuda:1
                pass
            # Back to original device
        """
        original_device = self.device
        self.device = torch.device(device)
        try:
            yield self
        finally:
            self.device = original_device

    def get_memory_info(self) -> Dict[str, float]:
        """Get current device memory information."""
        if self.device_type != 'cuda':
            return {"device": "cpu", "memory_tracking": "not_available"}

        device_id = self.device_index if self.device_index is not None else 0
        props = torch.cuda.get_device_properties(device_id)

        return {
            "device": str(self.device),
            "total_gb": props.total_memory / 1024**3,
            "allocated_gb": torch.cuda.memory_allocated(device_id) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(device_id) / 1024**3,
            "free_gb": (props.total_memory - torch.cuda.memory_allocated(device_id)) / 1024**3,
        }

    def clear_memory(self):
        """Clear GPU cache."""
        if self.device_type == 'cuda':
            torch.cuda.empty_cache()
            if self.device_index is not None:
                torch.cuda.synchronize(self.device_index)

    def print_info(self):
        """Print device information."""
        print(f"\n{'='*60}")
        print(f"Device Manager Information")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Type: {self.device_type}")

        if self.device_type == 'cuda':
            device_id = self.device_index if self.device_index is not None else 0
            print(f"GPU Name: {torch.cuda.get_device_name(device_id)}")

            mem_info = self.get_memory_info()
            print(f"\nMemory Information:")
            print(f"  Total: {mem_info['total_gb']:.2f} GB")
            print(f"  Allocated: {mem_info['allocated_gb']:.2f} GB")
            print(f"  Free: {mem_info['free_gb']:.2f} GB")


def demonstrate_device_management():
    """Demonstrate device management capabilities."""
    print("=" * 70)
    print("DEVICE MANAGEMENT DEMONSTRATION")
    print("=" * 70)

    # Initialize device manager
    dm = DeviceManager()
    dm.print_info()

    # Test tensor movement
    print(f"\n{'='*60}")
    print("Tensor Movement")
    print(f"{'='*60}")

    cpu_tensor = torch.randn(1000, 1000)
    print(f"Created tensor on CPU: {cpu_tensor.device}")

    gpu_tensor = dm.to_device(cpu_tensor)
    print(f"Moved to device: {gpu_tensor.device}")

    # Test dict movement
    data_dict = {
        'input': torch.randn(100, 100),
        'target': torch.randn(100, 10),
        'mask': torch.ones(100, dtype=torch.bool)
    }

    print(f"\nMoving dictionary of tensors...")
    data_dict_gpu = dm.to_device(data_dict)
    print(f"Input tensor device: {data_dict_gpu['input'].device}")

    # Benchmark transfers
    if dm.device_type == 'cuda':
        print(f"\n{'='*60}")
        print("Transfer Benchmark")
        print(f"{'='*60}")

        results = dm.benchmark_transfer(size=2000)
        print(f"Tensor size: {results['tensor_size_mb']:.2f} MB")
        print(f"CPU → GPU: {results['cpu_to_gpu_ms']:.2f} ms")
        print(f"CPU → GPU (non-blocking): {results['cpu_to_gpu_nonblocking_ms']:.2f} ms")
        print(f"GPU → CPU: {results['gpu_to_cpu_ms']:.2f} ms")
        print(f"Throughput: {results['throughput_gb_per_sec']:.2f} GB/s")

    # Cleanup
    dm.clear_memory()
    print(f"\n{'='*60}")
    print("Demonstration complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    demonstrate_device_management()
```

### 2.2 Efficient Data Loading with GPU

Create `gpu_dataloader.py`:

```python
#!/usr/bin/env python3
"""
GPU-Optimized Data Loading

Demonstrates efficient data loading patterns for GPU training,
including pinned memory and prefetching.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import time
from typing import Tuple


class DummyDataset(Dataset):
    """Simple dataset for demonstration."""

    def __init__(self, size: int = 1000, input_dim: int = 100, output_dim: int = 10):
        self.size = size
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(self.input_dim)
        y = torch.randn(self.output_dim)
        return x, y


def benchmark_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
    num_batches: int = 100
) -> float:
    """
    Benchmark dataloader configuration.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to use pinned memory
        device: Target device
        num_batches: Number of batches to process

    Returns:
        Total time in seconds
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    start_time = time.time()

    for i, (x, y) in enumerate(dataloader):
        if i >= num_batches:
            break

        # Move to device
        x = x.to(device, non_blocking=pin_memory)
        y = y.to(device, non_blocking=pin_memory)

        # Simulate some work
        _ = torch.matmul(x, x.T)

        if device.type == 'cuda':
            torch.cuda.synchronize()

    return time.time() - start_time


def demonstrate_dataloader_optimization():
    """Compare different dataloader configurations."""
    print("=" * 70)
    print("DATALOADER OPTIMIZATION FOR GPU")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")

    if device.type == 'cpu':
        print("⚠️  GPU not available - demonstration limited to CPU")
        print("   With GPU, you would see significant benefits from:")
        print("   - pinned memory")
        print("   - non-blocking transfers")
        print("   - multiple workers\n")

    dataset = DummyDataset(size=10000, input_dim=100, output_dim=10)
    batch_size = 32
    num_batches = 100

    configurations = [
        {"num_workers": 0, "pin_memory": False, "name": "Baseline (single worker, no pinning)"},
        {"num_workers": 2, "pin_memory": False, "name": "Multi-worker (no pinning)"},
        {"num_workers": 0, "pin_memory": True, "name": "Single worker (pinned memory)"},
        {"num_workers": 2, "pin_memory": True, "name": "Multi-worker + pinned memory"},
        {"num_workers": 4, "pin_memory": True, "name": "More workers (4) + pinned memory"},
    ]

    results = []

    for config in configurations:
        print(f"Testing: {config['name']}")

        elapsed = benchmark_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'] and device.type == 'cuda',
            device=device,
            num_batches=num_batches
        )

        results.append({
            "config": config['name'],
            "time": elapsed,
            "throughput": num_batches / elapsed
        })

        print(f"  Time: {elapsed:.3f}s | Throughput: {num_batches/elapsed:.1f} batches/s\n")

    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline_time = results[0]['time']

    for result in results:
        speedup = baseline_time / result['time']
        print(f"{result['config']}")
        print(f"  Time: {result['time']:.3f}s | Speedup: {speedup:.2f}x")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("1. Always use pin_memory=True when loading to GPU")
    print("2. Use non_blocking=True in .to() calls")
    print("3. Optimal num_workers: 2-4 per GPU (experiment with your setup)")
    print("4. Monitor CPU usage - too many workers can slow things down")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_dataloader_optimization()
```

---

## 3. GPU Memory Monitoring and Optimization

### 3.1 Advanced Memory Profiling

Create `memory_profiler.py`:

```python
#!/usr/bin/env python3
"""
Advanced GPU Memory Profiling and Optimization

Production-grade memory monitoring, profiling, and optimization
utilities for ML infrastructure.
"""

import torch
import time
import functools
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory at a point in time."""
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    total_gb: float
    timestamp: float
    label: str = ""


class MemoryProfiler:
    """GPU memory profiler with detailed tracking."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.snapshots: List[MemorySnapshot] = []
        self.enabled = torch.cuda.is_available()

        if not self.enabled:
            print("⚠️  GPU not available - memory profiling disabled")

    def snapshot(self, label: str = "") -> Optional[MemorySnapshot]:
        """Take a memory snapshot."""
        if not self.enabled:
            return None

        props = torch.cuda.get_device_properties(self.device_id)
        allocated = torch.cuda.memory_allocated(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        total = props.total_memory

        snap = MemorySnapshot(
            allocated_gb=allocated / 1024**3,
            reserved_gb=reserved / 1024**3,
            free_gb=(total - allocated) / 1024**3,
            total_gb=total / 1024**3,
            timestamp=time.time(),
            label=label
        )

        self.snapshots.append(snap)
        return snap

    def print_snapshot(self, snapshot: MemorySnapshot):
        """Print a single memory snapshot."""
        if snapshot.label:
            print(f"\n{snapshot.label}")
        print(f"  Allocated: {snapshot.allocated_gb:6.2f} GB")
        print(f"  Reserved:  {snapshot.reserved_gb:6.2f} GB")
        print(f"  Free:      {snapshot.free_gb:6.2f} GB")
        print(f"  Total:     {snapshot.total_gb:6.2f} GB")
        print(f"  Utilization: {(snapshot.allocated_gb/snapshot.total_gb)*100:.1f}%")

    def compare_snapshots(self, snap1: MemorySnapshot, snap2: MemorySnapshot):
        """Compare two memory snapshots."""
        delta_allocated = snap2.allocated_gb - snap1.allocated_gb
        delta_reserved = snap2.reserved_gb - snap1.reserved_gb

        print(f"\nMemory Delta: {snap1.label} → {snap2.label}")
        print(f"  Allocated: {delta_allocated:+.3f} GB")
        print(f"  Reserved:  {delta_reserved:+.3f} GB")

    @contextmanager
    def track_memory(self, label: str = ""):
        """
        Context manager to track memory usage of a code block.

        Usage:
            with profiler.track_memory("Model Forward"):
                output = model(input)
        """
        if not self.enabled:
            yield
            return

        before = self.snapshot(f"{label} - Before")
        try:
            yield
        finally:
            after = self.snapshot(f"{label} - After")
            self.compare_snapshots(before, after)

    def profile_function(self, func: Callable) -> Callable:
        """
        Decorator to profile memory usage of a function.

        Usage:
            @profiler.profile_function
            def my_function():
                ...
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            with self.track_memory(f"Function: {func.__name__}"):
                result = func(*args, **kwargs)
            return result

        return wrapper

    def print_summary(self):
        """Print summary of all snapshots."""
        if not self.snapshots:
            print("No snapshots recorded")
            return

        print("\n" + "=" * 70)
        print("MEMORY PROFILING SUMMARY")
        print("=" * 70)

        for i, snap in enumerate(self.snapshots):
            print(f"\nSnapshot {i}: {snap.label}")
            print(f"  Allocated: {snap.allocated_gb:.3f} GB")
            print(f"  Reserved:  {snap.reserved_gb:.3f} GB")
            print(f"  Free:      {snap.free_gb:.3f} GB")

            if i > 0:
                delta = snap.allocated_gb - self.snapshots[i-1].allocated_gb
                print(f"  Delta:     {delta:+.3f} GB")

        # Peak usage
        peak = max(self.snapshots, key=lambda s: s.allocated_gb)
        print(f"\n📊 Peak Memory Usage: {peak.allocated_gb:.3f} GB ({peak.label})")
        print("=" * 70)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.enabled:
            return {"error": "GPU not available"}

        return {
            "current": {
                "allocated_gb": torch.cuda.memory_allocated(self.device_id) / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved(self.device_id) / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated(self.device_id) / 1024**3,
                "max_reserved_gb": torch.cuda.max_memory_reserved(self.device_id) / 1024**3,
            },
            "device": {
                "total_memory_gb": torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3,
                "name": torch.cuda.get_device_name(self.device_id),
            },
            "snapshots_count": len(self.snapshots)
        }

    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device_id)
            torch.cuda.reset_accumulated_memory_stats(self.device_id)


class MemoryOptimizer:
    """Memory optimization utilities."""

    @staticmethod
    def clear_cache():
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def get_fragmentation_info(device_id: int = 0) -> Dict[str, float]:
        """Analyze memory fragmentation."""
        if not torch.cuda.is_available():
            return {"error": "GPU not available"}

        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)

        # Fragmentation is when reserved > allocated
        fragmentation = reserved - allocated
        fragmentation_percent = (fragmentation / reserved * 100) if reserved > 0 else 0

        return {
            "allocated_gb": allocated / 1024**3,
            "reserved_gb": reserved / 1024**3,
            "fragmentation_gb": fragmentation / 1024**3,
            "fragmentation_percent": fragmentation_percent
        }

    @staticmethod
    def optimize_model_memory(model: torch.nn.Module) -> Dict[str, Any]:
        """
        Analyze and optimize model memory usage.

        Args:
            model: PyTorch model

        Returns:
            Dictionary with optimization recommendations
        """
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2

        recommendations = []

        # Check for float64 parameters
        float64_params = sum(1 for p in model.parameters() if p.dtype == torch.float64)
        if float64_params > 0:
            recommendations.append(
                f"Convert {float64_params} float64 parameters to float32 (save ~50% memory)"
            )

        # Check for non-trainable parameters
        frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
        if frozen_params > 0:
            recommendations.append(
                f"Consider model.eval() and torch.no_grad() for {frozen_params} frozen parameters"
            )

        return {
            "parameter_memory_mb": param_memory,
            "buffer_memory_mb": buffer_memory,
            "total_memory_mb": param_memory + buffer_memory,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "recommendations": recommendations
        }


def demonstrate_memory_profiling():
    """Demonstrate memory profiling capabilities."""
    print("=" * 70)
    print("GPU MEMORY PROFILING DEMONSTRATION")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\n⚠️  GPU not available - skipping demonstration")
        return

    profiler = MemoryProfiler()

    # Initial state
    profiler.snapshot("Initial State")
    profiler.print_snapshot(profiler.snapshots[-1])

    # Allocate tensors with tracking
    print("\n" + "=" * 70)
    print("Allocating Tensors")
    print("=" * 70)

    with profiler.track_memory("Small Tensor Allocation"):
        tensor1 = torch.randn(1000, 1000, device='cuda')

    with profiler.track_memory("Large Tensor Allocation"):
        tensor2 = torch.randn(5000, 5000, device='cuda')

    # Function profiling
    @profiler.profile_function
    def matrix_operations():
        result = torch.matmul(tensor1, tensor1)
        return result

    print("\n" + "=" * 70)
    print("Profiling Function")
    print("=" * 70)

    result = matrix_operations()

    # Check fragmentation
    print("\n" + "=" * 70)
    print("Memory Fragmentation Analysis")
    print("=" * 70)

    frag = MemoryOptimizer.get_fragmentation_info()
    print(f"Allocated: {frag['allocated_gb']:.3f} GB")
    print(f"Reserved:  {frag['reserved_gb']:.3f} GB")
    print(f"Fragmentation: {frag['fragmentation_gb']:.3f} GB ({frag['fragmentation_percent']:.1f}%)")

    # Cleanup
    print("\n" + "=" * 70)
    print("Cleanup")
    print("=" * 70)

    with profiler.track_memory("Delete Tensors"):
        del tensor1, tensor2, result
        MemoryOptimizer.clear_cache()

    # Summary
    profiler.print_summary()

    # Stats
    print("\n" + "=" * 70)
    print("Memory Statistics")
    print("=" * 70)
    stats = profiler.get_memory_stats()
    print(f"Device: {stats['device']['name']}")
    print(f"Total Memory: {stats['device']['total_memory_gb']:.2f} GB")
    print(f"Current Allocated: {stats['current']['allocated_gb']:.3f} GB")
    print(f"Peak Allocated: {stats['current']['max_allocated_gb']:.3f} GB")


if __name__ == "__main__":
    demonstrate_memory_profiling()
```

### 3.2 Memory Optimization Techniques

Create `memory_optimization_techniques.py`:

```python
#!/usr/bin/env python3
"""
GPU Memory Optimization Techniques

Practical examples of memory optimization strategies for
production ML workloads.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class MemoryEfficientModel(nn.Module):
    """Example model with memory optimization techniques."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        if self.use_checkpointing and self.training:
            # Use gradient checkpointing to save memory during training
            x = torch.utils.checkpoint.checkpoint(self._forward_block1, x)
            x = torch.utils.checkpoint.checkpoint(self._forward_block2, x)
        else:
            x = self._forward_block1(x)
            x = self._forward_block2(x)

        x = self.layer4(x)
        return x

    def _forward_block1(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        return x

    def _forward_block2(self, x):
        x = self.layer3(x)
        x = self.activation(x)
        return x


def demonstrate_mixed_precision():
    """Demonstrate mixed precision training for memory savings."""
    print("=" * 70)
    print("MIXED PRECISION TRAINING")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\n⚠️  GPU not available")
        return

    device = torch.device('cuda')
    model = MemoryEfficientModel(1024, 2048, 512).to(device)

    # Create dummy data
    x = torch.randn(32, 1024, device=device)
    y = torch.randn(32, 512, device=device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # FP32 (Full Precision)
    print("\n1. Full Precision (FP32):")
    torch.cuda.reset_peak_memory_stats()

    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    fp32_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"   Peak Memory: {fp32_memory:.2f} MB")

    # FP16 (Mixed Precision)
    print("\n2. Mixed Precision (FP16 with AMP):")
    torch.cuda.reset_peak_memory_stats()

    scaler = torch.cuda.amp.GradScaler()

    with torch.cuda.amp.autocast():
        output = model(x)
        loss = criterion(output, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    fp16_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"   Peak Memory: {fp16_memory:.2f} MB")
    print(f"   Memory Savings: {fp32_memory - fp16_memory:.2f} MB ({((fp32_memory - fp16_memory)/fp32_memory * 100):.1f}%)")

    print("\n💡 Tips:")
    print("   - AMP reduces memory by ~50% for large models")
    print("   - Use with torch.cuda.amp.autocast() context")
    print("   - Requires GradScaler to prevent underflow")
    print("   - Supported on GPUs with Tensor Cores (Volta+)")


def demonstrate_gradient_checkpointing():
    """Demonstrate gradient checkpointing for memory-intensive models."""
    print("\n" + "=" * 70)
    print("GRADIENT CHECKPOINTING")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\n⚠️  GPU not available")
        return

    device = torch.device('cuda')
    batch_size = 16
    x = torch.randn(batch_size, 1024, device=device)
    y = torch.randn(batch_size, 512, device=device)

    # Without checkpointing
    print("\n1. Without Gradient Checkpointing:")
    torch.cuda.reset_peak_memory_stats()

    model = MemoryEfficientModel(1024, 4096, 512, use_checkpointing=False).to(device)
    criterion = nn.MSELoss()

    output = model(x)
    loss = criterion(output, y)
    loss.backward()

    no_checkpoint_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"   Peak Memory: {no_checkpoint_memory:.2f} MB")

    # With checkpointing
    print("\n2. With Gradient Checkpointing:")
    torch.cuda.reset_peak_memory_stats()

    model = MemoryEfficientModel(1024, 4096, 512, use_checkpointing=True).to(device)
    model.train()

    output = model(x)
    loss = criterion(output, y)
    loss.backward()

    checkpoint_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"   Peak Memory: {checkpoint_memory:.2f} MB")
    print(f"   Memory Savings: {no_checkpoint_memory - checkpoint_memory:.2f} MB ({((no_checkpoint_memory - checkpoint_memory)/no_checkpoint_memory * 100):.1f}%)")

    print("\n💡 Tips:")
    print("   - Trades compute for memory (recomputes activations during backward)")
    print("   - Useful for very deep models or large batch sizes")
    print("   - Can save 50-80% memory on deep networks")
    print("   - Increases training time by ~20-30%")


def demonstrate_inplace_operations():
    """Demonstrate memory savings with inplace operations."""
    print("\n" + "=" * 70)
    print("INPLACE OPERATIONS")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\n⚠️  GPU not available")
        return

    device = torch.device('cuda')

    # Regular operations
    print("\n1. Regular Operations:")
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)

    z = x + y  # Creates new tensor
    z = torch.relu(z)  # Creates new tensor

    regular_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"   Peak Memory: {regular_memory:.2f} MB")

    # Inplace operations
    print("\n2. Inplace Operations:")
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(5000, 5000, device=device)
    y = torch.randn(5000, 5000, device=device)

    x.add_(y)  # Inplace addition
    x.relu_()  # Inplace ReLU

    inplace_memory = torch.cuda.max_memory_allocated() / 1024**2
    print(f"   Peak Memory: {inplace_memory:.2f} MB")
    print(f"   Memory Savings: {regular_memory - inplace_memory:.2f} MB ({((regular_memory - inplace_memory)/regular_memory * 100):.1f}%)")

    print("\n⚠️  Warning:")
    print("   - Inplace ops can break autograd - use carefully")
    print("   - Don't use on tensors that need gradients")
    print("   - Useful for data preprocessing and inference")


def demonstrate_batch_size_optimization():
    """Find optimal batch size for GPU."""
    print("\n" + "=" * 70)
    print("BATCH SIZE OPTIMIZATION")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\n⚠️  GPU not available")
        return

    device = torch.device('cuda')
    model = MemoryEfficientModel(512, 1024, 256).to(device)

    print("\nTesting different batch sizes:\n")
    print(f"{'Batch Size':<12} {'Memory (MB)':<15} {'Status':<20}")
    print("-" * 50)

    batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    optimal_batch_size = 8

    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            x = torch.randn(batch_size, 512, device=device)
            output = model(x)
            loss = output.sum()
            loss.backward()

            memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"{batch_size:<12} {memory_mb:<15.2f} ✓ Success")
            optimal_batch_size = batch_size

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{batch_size:<12} {'N/A':<15} ✗ OOM")
                break
            else:
                raise

    print(f"\n💡 Recommended batch size: {optimal_batch_size}")
    print("   - Start with power of 2 (16, 32, 64, ...)")
    print("   - Increase until OOM, then reduce by 20-30%")
    print("   - Balance memory usage vs throughput")


def main():
    """Run all memory optimization demonstrations."""
    demonstrate_mixed_precision()
    demonstrate_gradient_checkpointing()
    demonstrate_inplace_operations()
    demonstrate_batch_size_optimization()

    print("\n" + "=" * 70)
    print("MEMORY OPTIMIZATION SUMMARY")
    print("=" * 70)
    print("""
Key Techniques:
1. Mixed Precision (AMP) - 40-50% memory reduction, minimal accuracy impact
2. Gradient Checkpointing - 50-80% memory reduction, 20-30% slower training
3. Inplace Operations - 10-30% memory reduction, careful with autograd
4. Optimal Batch Size - Maximize GPU utilization without OOM
5. Clear Cache - Free unused memory between operations
6. Model Sharding - Split large models across multiple GPUs

Production Recommendations:
- Always use AMP for large models (FP16/BF16)
- Use gradient checkpointing for models > 1B parameters
- Monitor memory usage in production
- Set up OOM alerts and automatic recovery
- Test memory limits before deploying
    """)


if __name__ == "__main__":
    main()
```

---

## 4. Multi-GPU Basics

### 4.1 DataParallel Implementation

Create `multi_gpu_training.py`:

```python
#!/usr/bin/env python3
"""
Multi-GPU Training with DataParallel

Production examples of multi-GPU training strategies,
including DataParallel and basic DistributedDataParallel.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from typing import Dict, List, Tuple


class SimpleModel(nn.Module):
    """Simple model for multi-GPU demonstration."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 2048, output_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class DummyDataset(Dataset):
    """Dummy dataset for training."""

    def __init__(self, size: int = 10000, input_dim: int = 1024, output_dim: int = 512):
        self.size = size
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(self.input_dim)
        y = torch.randn(self.output_dim)
        return x, y


def train_single_gpu(model: nn.Module,
                     dataloader: DataLoader,
                     device: torch.device,
                     epochs: int = 1) -> Dict[str, float]:
    """Train on single GPU."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    elapsed = time.time() - start_time

    return {
        "time": elapsed,
        "avg_loss": total_loss / len(dataloader),
        "throughput": len(dataloader) * epochs / elapsed
    }


def train_data_parallel(model: nn.Module,
                        dataloader: DataLoader,
                        device_ids: List[int],
                        epochs: int = 1) -> Dict[str, float]:
    """Train with DataParallel across multiple GPUs."""
    # Wrap model in DataParallel
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(f'cuda:{device_ids[0]}')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(f'cuda:{device_ids[0]}')
            y = y.to(f'cuda:{device_ids[0]}')

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    elapsed = time.time() - start_time

    return {
        "time": elapsed,
        "avg_loss": total_loss / len(dataloader),
        "throughput": len(dataloader) * epochs / elapsed
    }


def demonstrate_multi_gpu():
    """Demonstrate single vs multi-GPU training."""
    print("=" * 70)
    print("MULTI-GPU TRAINING DEMONSTRATION")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\n⚠️  No GPU available")
        return

    gpu_count = torch.cuda.device_count()
    print(f"\nAvailable GPUs: {gpu_count}")

    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if gpu_count < 2:
        print("\n⚠️  Multi-GPU training requires at least 2 GPUs")
        print("   Demonstrating single GPU training only")

    # Prepare data
    dataset = DummyDataset(size=5000, input_dim=1024, output_dim=512)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = SimpleModel(input_dim=1024, hidden_dim=2048, output_dim=512)

    # Single GPU training
    print("\n" + "=" * 70)
    print("Single GPU Training")
    print("=" * 70)

    single_gpu_model = SimpleModel(input_dim=1024, hidden_dim=2048, output_dim=512)
    single_results = train_single_gpu(
        single_gpu_model,
        dataloader,
        torch.device('cuda:0'),
        epochs=2
    )

    print(f"Time: {single_results['time']:.2f}s")
    print(f"Throughput: {single_results['throughput']:.1f} batches/s")
    print(f"Average Loss: {single_results['avg_loss']:.4f}")

    # Multi-GPU training (if available)
    if gpu_count >= 2:
        print("\n" + "=" * 70)
        print(f"Multi-GPU Training ({gpu_count} GPUs)")
        print("=" * 70)

        multi_gpu_model = SimpleModel(input_dim=1024, hidden_dim=2048, output_dim=512)

        # Increase batch size for multi-GPU (can handle larger batches)
        multi_gpu_batch_size = batch_size * gpu_count
        multi_dataloader = DataLoader(
            dataset,
            batch_size=multi_gpu_batch_size,
            shuffle=True,
            num_workers=4
        )

        multi_results = train_data_parallel(
            multi_gpu_model,
            multi_dataloader,
            device_ids=list(range(gpu_count)),
            epochs=2
        )

        print(f"Time: {multi_results['time']:.2f}s")
        print(f"Throughput: {multi_results['throughput']:.1f} batches/s")
        print(f"Average Loss: {multi_results['avg_loss']:.4f}")

        # Comparison
        print("\n" + "=" * 70)
        print("Comparison")
        print("=" * 70)

        speedup = single_results['time'] / multi_results['time']
        throughput_increase = multi_results['throughput'] / single_results['throughput']

        print(f"Speedup: {speedup:.2f}x")
        print(f"Throughput Increase: {throughput_increase:.2f}x")
        print(f"\nEfficiency: {(speedup / gpu_count) * 100:.1f}%")
        print("(100% = perfect linear scaling)")

    print("\n" + "=" * 70)
    print("Multi-GPU Best Practices")
    print("=" * 70)
    print("""
1. DataParallel:
   - Simple to use: just wrap model with nn.DataParallel()
   - Good for single-node multi-GPU
   - Limited scalability (GIL bottleneck)
   - Uneven GPU utilization (GPU 0 does more work)

2. DistributedDataParallel (DDP):
   - Recommended for production
   - Better performance and scalability
   - Supports multi-node training
   - Requires more setup (process per GPU)

3. Key Considerations:
   - Scale batch size with number of GPUs
   - Increase num_workers for data loading
   - Monitor GPU utilization (nvidia-smi)
   - Watch for GPU 0 bottleneck with DataParallel
   - Consider gradient accumulation for large models
    """)


def monitor_gpu_usage():
    """Monitor GPU usage during training."""
    print("\n" + "=" * 70)
    print("GPU UTILIZATION MONITORING")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\n⚠️  No GPU available")
        return

    print("\nRun this command in another terminal to monitor GPUs:")
    print("  watch -n 0.5 nvidia-smi")
    print("\nOr use:")
    print("  gpustat -cp -i 0.5")
    print("\nKey metrics to watch:")
    print("  - GPU Utilization % (should be high, 80-100%)")
    print("  - Memory Usage (should be consistent, not growing)")
    print("  - Temperature (< 85°C is good)")
    print("  - Power Usage (higher = more utilized)")


if __name__ == "__main__":
    demonstrate_multi_gpu()
    monitor_gpu_usage()
```

---

## 5. Performance Profiling

### 5.1 PyTorch Profiler

Create `gpu_profiler.py`:

```python
#!/usr/bin/env python3
"""
GPU Performance Profiling with PyTorch Profiler

Production-grade profiling utilities for identifying
bottlenecks and optimizing GPU performance.
"""

import torch
import torch.nn as nn
import torch.profiler
from typing import Optional
import time


class ProfilingModel(nn.Module):
    """Model for profiling demonstration."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def profile_model_with_pytorch_profiler(model: nn.Module,
                                        input_tensor: torch.Tensor,
                                        device: torch.device):
    """
    Profile model using PyTorch Profiler.

    Generates detailed performance metrics and timeline.
    """
    print("=" * 70)
    print("PYTORCH PROFILER")
    print("=" * 70)

    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Warmup
    for _ in range(10):
        _ = model(input_tensor)

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.profiler.record_function("model_forward"):
            output = model(input_tensor)

    # Print summary
    print("\n📊 Profiling Summary:\n")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))

    # Export trace for visualization
    trace_file = "/tmp/pytorch_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"\n💾 Chrome trace exported to: {trace_file}")
    print("   Open in Chrome at: chrome://tracing/")

    # Memory summary
    print("\n💾 Memory Profiling:\n")
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10
    ))


def profile_with_autograd_profiler(model: nn.Module,
                                   input_tensor: torch.Tensor,
                                   device: torch.device):
    """Profile using autograd profiler (legacy but useful)."""
    print("\n" + "=" * 70)
    print("AUTOGRAD PROFILER")
    print("=" * 70)

    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Warmup
    for _ in range(10):
        _ = model(input_tensor)

    # Profile
    with torch.autograd.profiler.profile(
        use_cuda=True,
        record_shapes=True,
        profile_memory=True
    ) as prof:
        output = model(input_tensor)

    # Print results
    print("\n📊 Operations Summary:\n")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total",
        row_limit=15
    ))


def benchmark_kernel_performance():
    """Benchmark specific CUDA kernels."""
    print("\n" + "=" * 70)
    print("KERNEL PERFORMANCE BENCHMARK")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\n⚠️  GPU not available")
        return

    device = torch.device('cuda')
    sizes = [1000, 2000, 5000, 10000]

    operations = {
        "matmul": lambda x, y: torch.matmul(x, y),
        "conv2d": lambda x, w: torch.nn.functional.conv2d(x, w),
        "relu": lambda x: torch.relu(x),
        "softmax": lambda x: torch.softmax(x, dim=-1),
    }

    print(f"\n{'Operation':<15} {'Size':<10} {'Time (ms)':<12} {'GFLOPS':<10}")
    print("-" * 55)

    for op_name, op_func in operations.items():
        for size in [1000, 5000]:
            if op_name == "matmul":
                x = torch.randn(size, size, device=device)
                y = torch.randn(size, size, device=device)

                # Warmup
                for _ in range(10):
                    _ = op_func(x, y)

                torch.cuda.synchronize()
                start = time.time()

                for _ in range(100):
                    _ = op_func(x, y)

                torch.cuda.synchronize()
                elapsed_ms = (time.time() - start) / 100 * 1000

                # Calculate GFLOPS (approximate)
                flops = 2 * size * size * size  # Matrix multiply FLOPs
                gflops = (flops / (elapsed_ms / 1000)) / 1e9

                print(f"{op_name:<15} {size}x{size:<5} {elapsed_ms:>10.3f}  {gflops:>8.2f}")


def analyze_gpu_bottlenecks():
    """Identify common GPU bottlenecks."""
    print("\n" + "=" * 70)
    print("GPU BOTTLENECK ANALYSIS")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\n⚠️  GPU not available")
        return

    device = torch.device('cuda')

    print("\n1. Memory Bandwidth Bound Test:")
    print("   Testing large data movement vs computation...")

    # Memory bandwidth test
    size = 10000
    x = torch.randn(size, size, device=device)

    # Just memory copy
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y = x.clone()
    torch.cuda.synchronize()
    copy_time = (time.time() - start) / 100

    # Compute-heavy operation
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y = torch.matmul(x, x)
    torch.cuda.synchronize()
    compute_time = (time.time() - start) / 100

    print(f"   Memory Copy Time: {copy_time*1000:.3f} ms")
    print(f"   Compute Time: {compute_time*1000:.3f} ms")

    if copy_time > compute_time * 0.5:
        print("   ⚠️  Memory bandwidth may be bottleneck")
        print("       → Use larger batch sizes")
        print("       → Fuse operations to reduce memory traffic")
    else:
        print("   ✓ Compute bound (good GPU utilization)")

    print("\n2. CPU-GPU Transfer Bottleneck Test:")

    # Test transfer overhead
    cpu_tensor = torch.randn(1000, 1000)

    start = time.time()
    for _ in range(100):
        gpu_tensor = cpu_tensor.to(device)
        torch.cuda.synchronize()
    transfer_time = (time.time() - start) / 100

    # Test computation time
    gpu_tensor = cpu_tensor.to(device)
    start = time.time()
    for _ in range(100):
        result = torch.matmul(gpu_tensor, gpu_tensor)
        torch.cuda.synchronize()
    compute_time = (time.time() - start) / 100

    print(f"   Transfer Time: {transfer_time*1000:.3f} ms")
    print(f"   Compute Time: {compute_time*1000:.3f} ms")

    if transfer_time > compute_time:
        print("   ⚠️  CPU-GPU transfer is bottleneck")
        print("       → Keep data on GPU as long as possible")
        print("       → Use pinned memory and non_blocking transfers")
        print("       → Batch operations to amortize transfer cost")
    else:
        print("   ✓ Transfer overhead is acceptable")


def main():
    """Run all profiling demonstrations."""
    if not torch.cuda.is_available():
        print("⚠️  GPU not available - profiling requires CUDA")
        return

    device = torch.device('cuda')
    model = ProfilingModel()
    input_tensor = torch.randn(8, 3, 32, 32)

    # PyTorch Profiler
    profile_model_with_pytorch_profiler(model, input_tensor, device)

    # Autograd Profiler
    profile_with_autograd_profiler(model, input_tensor, device)

    # Kernel benchmarks
    benchmark_kernel_performance()

    # Bottleneck analysis
    analyze_gpu_bottlenecks()

    print("\n" + "=" * 70)
    print("PROFILING RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. Use PyTorch Profiler for detailed analysis:
   - Identifies slow operations
   - Shows memory usage
   - Generates visual timeline

2. Monitor key metrics:
   - GPU utilization (should be > 80%)
   - Memory bandwidth utilization
   - Kernel execution time
   - CPU-GPU transfer overhead

3. Optimization strategies:
   - Increase batch size (if memory allows)
   - Fuse operations to reduce kernel launches
   - Use mixed precision (AMP)
   - Optimize data pipeline (num_workers, pin_memory)
   - Profile regularly during development

4. Tools:
   - PyTorch Profiler: Built-in, easy to use
   - NVIDIA Nsight: Advanced GPU profiling
   - tensorboard --logdir=<profiler_logs>: Visualize profiles
   - nvidia-smi: Real-time monitoring
    """)


if __name__ == "__main__":
    main()
```

---

## 6. Common GPU Errors and Debugging

Create `gpu_debugging_guide.py`:

```python
#!/usr/bin/env python3
"""
GPU Error Handling and Debugging Guide

Comprehensive guide to common GPU errors, their causes,
and production-ready solutions.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import traceback


class GPUErrorHandler:
    """Production error handling for GPU operations."""

    @staticmethod
    def handle_oom_error(func, *args, **kwargs):
        """
        Handle Out of Memory errors with automatic recovery.

        Tries the operation with progressively smaller batch sizes.
        """
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️  OOM Error detected: {e}")
                print("   Attempting recovery...")

                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # If batch operation, try reducing batch size
                if 'batch_size' in kwargs:
                    original_batch_size = kwargs['batch_size']
                    new_batch_size = original_batch_size // 2

                    if new_batch_size > 0:
                        print(f"   Retrying with batch_size={new_batch_size}")
                        kwargs['batch_size'] = new_batch_size
                        return GPUErrorHandler.handle_oom_error(func, *args, **kwargs)
                    else:
                        print("   ❌ Cannot reduce batch size further")
                        raise
                else:
                    print("   ❌ Cannot auto-recover from OOM")
                    raise
            else:
                raise

    @staticmethod
    def validate_tensors_same_device(tensors: list, operation_name: str = "operation"):
        """
        Validate that all tensors are on the same device.

        Raises informative error if devices don't match.
        """
        if not tensors:
            return

        devices = [t.device for t in tensors if isinstance(t, torch.Tensor)]

        if len(set(devices)) > 1:
            device_info = "\n".join([f"  Tensor {i}: {d}" for i, d in enumerate(devices)])
            raise RuntimeError(
                f"Device mismatch in {operation_name}:\n{device_info}\n"
                f"All tensors must be on the same device."
            )

    @staticmethod
    def safe_to_device(obj, device, non_blocking=False):
        """Safely move object to device with error handling."""
        try:
            if isinstance(obj, torch.Tensor):
                return obj.to(device, non_blocking=non_blocking)
            elif isinstance(obj, nn.Module):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: GPUErrorHandler.safe_to_device(v, device, non_blocking)
                        for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                result = [GPUErrorHandler.safe_to_device(item, device, non_blocking)
                         for item in obj]
                return type(obj)(result)
            else:
                return obj
        except RuntimeError as e:
            print(f"❌ Error moving to device {device}: {e}")
            print(f"   Object type: {type(obj)}")
            if isinstance(obj, torch.Tensor):
                print(f"   Tensor shape: {obj.shape}")
                print(f"   Tensor dtype: {obj.dtype}")
            raise


def demonstrate_common_errors():
    """Demonstrate and solve common GPU errors."""
    print("=" * 70)
    print("COMMON GPU ERRORS AND SOLUTIONS")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\n⚠️  GPU not available - showing conceptual examples")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # Error 1: Device Mismatch
    print("\n1. DEVICE MISMATCH ERROR")
    print("-" * 70)
    print("Problem: Tensors on different devices")

    try:
        cpu_tensor = torch.randn(100, 100)
        if torch.cuda.is_available():
            gpu_tensor = torch.randn(100, 100, device='cuda')
            # This will fail
            result = cpu_tensor + gpu_tensor
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        print("\n✓ Solution:")
        print("   # Move tensors to same device")
        print("   cpu_tensor = cpu_tensor.to(device)")
        print("   result = cpu_tensor + gpu_tensor")

        if torch.cuda.is_available():
            cpu_tensor = cpu_tensor.to(device)
            result = cpu_tensor + gpu_tensor
            print(f"   ✓ Success! Result device: {result.device}")

    # Error 2: Out of Memory
    print("\n2. OUT OF MEMORY (OOM) ERROR")
    print("-" * 70)
    print("Problem: Allocating more memory than available")

    if torch.cuda.is_available():
        try:
            # Try to allocate huge tensor
            huge_tensor = torch.randn(50000, 50000, device='cuda')
        except RuntimeError as e:
            print(f"❌ Error: {str(e)[:100]}...")
            print("\n✓ Solutions:")
            print("   1. Reduce batch size")
            print("   2. Use gradient checkpointing")
            print("   3. Use mixed precision (FP16)")
            print("   4. Clear cache: torch.cuda.empty_cache()")
            print("   5. Use gradient accumulation")

            # Demonstrate fix
            print("\n   Demonstrating fix:")
            print("   # Clear cache")
            torch.cuda.empty_cache()
            print("   # Use smaller size")
            smaller_tensor = torch.randn(1000, 1000, device='cuda')
            print(f"   ✓ Success! Shape: {smaller_tensor.shape}")
            del smaller_tensor
            torch.cuda.empty_cache()

    # Error 3: CUDA Initialization Failed
    print("\n3. CUDA INITIALIZATION ERRORS")
    print("-" * 70)
    print("Problem: CUDA not properly initialized")
    print("\n❌ Common error messages:")
    print("   - 'CUDA driver version is insufficient'")
    print("   - 'CUDA out of memory' at startup")
    print("   - 'no CUDA-capable device is detected'")
    print("\n✓ Solutions:")
    print("   1. Check NVIDIA driver: nvidia-smi")
    print("   2. Verify CUDA installation: nvcc --version")
    print("   3. Check PyTorch CUDA: torch.cuda.is_available()")
    print("   4. Reinstall PyTorch with correct CUDA version")
    print("   5. Check GPU is not being used by another process")

    # Error 4: Synchronization Issues
    print("\n4. CUDA SYNCHRONIZATION ERRORS")
    print("-" * 70)
    print("Problem: Timing issues with async GPU operations")

    if torch.cuda.is_available():
        print("\n❌ Wrong way:")
        print("   start = time.time()")
        print("   result = model(input)  # Async on GPU")
        print("   elapsed = time.time() - start  # Wrong! Didn't wait for GPU")

        print("\n✓ Correct way:")
        print("   start = time.time()")
        print("   result = model(input)")
        print("   torch.cuda.synchronize()  # Wait for GPU")
        print("   elapsed = time.time() - start  # Correct!")

    # Error 5: Mixed Dtype
    print("\n5. DTYPE MISMATCH ERRORS")
    print("-" * 70)
    print("Problem: Operations between different dtypes")

    try:
        float_tensor = torch.randn(10, 10, dtype=torch.float32, device=device)
        int_tensor = torch.randint(0, 10, (10, 10), dtype=torch.int32, device=device)
        # This might fail depending on operation
        # result = float_tensor + int_tensor
        print("❌ Error: Cannot operate on float32 and int32 tensors")
        print("\n✓ Solution:")
        print("   int_tensor = int_tensor.float()")
        int_tensor = int_tensor.float()
        result = float_tensor + int_tensor
        print(f"   ✓ Success! Result dtype: {result.dtype}")
    except Exception as e:
        print(f"Error: {e}")


def create_debugging_utilities():
    """Production debugging utilities."""
    print("\n" + "=" * 70)
    print("DEBUGGING UTILITIES")
    print("=" * 70)

    class GPUDebugger:
        """Utility class for GPU debugging."""

        @staticmethod
        def print_tensor_info(tensor: torch.Tensor, name: str = "tensor"):
            """Print comprehensive tensor information."""
            print(f"\n{name}:")
            print(f"  Shape: {tensor.shape}")
            print(f"  Dtype: {tensor.dtype}")
            print(f"  Device: {tensor.device}")
            print(f"  Requires Grad: {tensor.requires_grad}")
            print(f"  Is Contiguous: {tensor.is_contiguous()}")
            print(f"  Memory (MB): {tensor.element_size() * tensor.nelement() / 1024**2:.2f}")
            if tensor.requires_grad and tensor.grad is not None:
                print(f"  Has Gradient: True")
                print(f"  Gradient Shape: {tensor.grad.shape}")

        @staticmethod
        def check_model_device(model: nn.Module) -> Dict[str, Any]:
            """Check which devices model parameters are on."""
            devices = {}
            for name, param in model.named_parameters():
                device = str(param.device)
                if device not in devices:
                    devices[device] = []
                devices[device].append(name)

            return devices

        @staticmethod
        def memory_summary(device_id: int = 0):
            """Print GPU memory summary."""
            if not torch.cuda.is_available():
                print("GPU not available")
                return

            print(f"\nGPU {device_id} Memory Summary:")
            print(f"  Allocated: {torch.cuda.memory_allocated(device_id) / 1024**3:.2f} GB")
            print(f"  Reserved:  {torch.cuda.memory_reserved(device_id) / 1024**3:.2f} GB")
            print(f"  Max Allocated: {torch.cuda.max_memory_allocated(device_id) / 1024**3:.2f} GB")
            print(f"  Max Reserved:  {torch.cuda.max_memory_reserved(device_id) / 1024**3:.2f} GB")

    # Demonstrate utilities
    if torch.cuda.is_available():
        device = torch.device('cuda')
        tensor = torch.randn(1000, 1000, device=device, requires_grad=True)

        print("\nTensor Info Utility:")
        GPUDebugger.print_tensor_info(tensor, "demo_tensor")

        print("\n" + "-" * 70)
        GPUDebugger.memory_summary()

        print("\n" + "-" * 70)
        print("Model Device Check Utility:")
        model = nn.Linear(100, 10).to(device)
        devices = GPUDebugger.check_model_device(model)
        for device_name, params in devices.items():
            print(f"  Device {device_name}: {len(params)} parameters")


def main():
    """Run all debugging demonstrations."""
    demonstrate_common_errors()
    create_debugging_utilities()

    print("\n" + "=" * 70)
    print("DEBUGGING CHECKLIST")
    print("=" * 70)
    print("""
When encountering GPU errors:

1. Check CUDA availability:
   - torch.cuda.is_available()
   - nvidia-smi

2. Verify device placement:
   - Print tensor.device for all tensors
   - Ensure model and data are on same device

3. Monitor memory:
   - torch.cuda.memory_allocated()
   - nvidia-smi for real-time monitoring
   - Clear cache when needed

4. Synchronize for timing:
   - Always use torch.cuda.synchronize() before time.time()

5. Check dtypes:
   - Ensure compatible dtypes in operations
   - Use tensor.dtype to verify

6. Enable debugging:
   - CUDA_LAUNCH_BLOCKING=1 for better error messages
   - torch.autograd.set_detect_anomaly(True) for NaN detection

7. Production error handling:
   - Implement OOM recovery
   - Log GPU errors with context
   - Set up monitoring alerts
    """)


if __name__ == "__main__":
    main()
```

---

## 7. Production GPU Management Utilities

Create `production_gpu_manager.py`:

```python
#!/usr/bin/env python3
"""
Production GPU Management Utilities

Enterprise-grade GPU monitoring, management, and health
checking utilities for ML infrastructure.
"""

import torch
import subprocess
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import threading


@dataclass
class GPUMetrics:
    """GPU metrics at a point in time."""
    timestamp: str
    gpu_id: int
    name: str
    temperature: float
    power_usage_w: float
    power_limit_w: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    gpu_util_percent: float
    compute_mode: str


class ProductionGPUManager:
    """Production GPU manager with monitoring and health checks."""

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        self.monitoring_active = False
        self.metrics_history: List[GPUMetrics] = []

    def get_gpu_metrics(self, gpu_id: int = 0) -> Optional[GPUMetrics]:
        """Get current metrics for a GPU."""
        if not self.cuda_available:
            return None

        try:
            # Use nvidia-smi to get detailed metrics
            cmd = [
                'nvidia-smi',
                f'--id={gpu_id}',
                '--query-gpu=timestamp,name,temperature.gpu,power.draw,power.limit,'
                'memory.used,memory.total,utilization.gpu,compute_mode',
                '--format=csv,noheader,nounits'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                values = result.stdout.strip().split(', ')

                memory_used = float(values[5])
                memory_total = float(values[6])

                return GPUMetrics(
                    timestamp=values[0],
                    gpu_id=gpu_id,
                    name=values[1],
                    temperature=float(values[2]),
                    power_usage_w=float(values[3]),
                    power_limit_w=float(values[4]),
                    memory_used_mb=memory_used,
                    memory_total_mb=memory_total,
                    memory_percent=(memory_used / memory_total * 100),
                    gpu_util_percent=float(values[7]),
                    compute_mode=values[8]
                )
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
            return None

    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive GPU health check.

        Returns health status and any issues found.
        """
        health_status = {
            "healthy": True,
            "issues": [],
            "warnings": [],
            "timestamp": datetime.now().isoformat(),
            "gpus": []
        }

        if not self.cuda_available:
            health_status["healthy"] = False
            health_status["issues"].append("CUDA not available")
            return health_status

        for gpu_id in range(self.device_count):
            metrics = self.get_gpu_metrics(gpu_id)

            if metrics is None:
                health_status["healthy"] = False
                health_status["issues"].append(f"GPU {gpu_id}: Unable to get metrics")
                continue

            gpu_health = {
                "gpu_id": gpu_id,
                "healthy": True,
                "issues": [],
                "warnings": []
            }

            # Temperature check
            if metrics.temperature > 85:
                gpu_health["healthy"] = False
                gpu_health["issues"].append(
                    f"Temperature critical: {metrics.temperature}°C (> 85°C)"
                )
            elif metrics.temperature > 75:
                gpu_health["warnings"].append(
                    f"Temperature high: {metrics.temperature}°C (> 75°C)"
                )

            # Power check
            if metrics.power_usage_w > metrics.power_limit_w * 0.95:
                gpu_health["warnings"].append(
                    f"Power usage near limit: {metrics.power_usage_w}W / {metrics.power_limit_w}W"
                )

            # Memory check
            if metrics.memory_percent > 95:
                gpu_health["healthy"] = False
                gpu_health["issues"].append(
                    f"Memory critical: {metrics.memory_percent:.1f}% used"
                )
            elif metrics.memory_percent > 85:
                gpu_health["warnings"].append(
                    f"Memory high: {metrics.memory_percent:.1f}% used"
                )

            # Check PyTorch memory
            torch_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
            torch_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2

            if torch_reserved > torch_allocated * 1.5:
                gpu_health["warnings"].append(
                    f"High memory fragmentation: {torch_reserved:.0f}MB reserved vs "
                    f"{torch_allocated:.0f}MB allocated"
                )

            if not gpu_health["healthy"]:
                health_status["healthy"] = False

            health_status["gpus"].append(gpu_health)

        return health_status

    def print_health_report(self):
        """Print formatted health report."""
        health = self.health_check()

        print("=" * 70)
        print("GPU HEALTH REPORT")
        print("=" * 70)
        print(f"Timestamp: {health['timestamp']}")
        print(f"Overall Status: {'✓ HEALTHY' if health['healthy'] else '✗ UNHEALTHY'}")

        if health['issues']:
            print(f"\n❌ ISSUES ({len(health['issues'])}):")
            for issue in health['issues']:
                print(f"  - {issue}")

        if health['warnings']:
            print(f"\n⚠️  WARNINGS ({len(health['warnings'])}):")
            for warning in health['warnings']:
                print(f"  - {warning}")

        for gpu_health in health['gpus']:
            print(f"\nGPU {gpu_health['gpu_id']}:")
            print(f"  Status: {'✓ Healthy' if gpu_health['healthy'] else '✗ Unhealthy'}")

            if gpu_health['issues']:
                print("  Issues:")
                for issue in gpu_health['issues']:
                    print(f"    - {issue}")

            if gpu_health['warnings']:
                print("  Warnings:")
                for warning in gpu_health['warnings']:
                    print(f"    - {warning}")

        print("=" * 70)

    def monitor_continuous(self, interval_seconds: int = 5, duration_seconds: int = 60):
        """
        Continuously monitor GPUs for a specified duration.

        Args:
            interval_seconds: Time between measurements
            duration_seconds: Total monitoring duration
        """
        print(f"Starting GPU monitoring (interval={interval_seconds}s, duration={duration_seconds}s)")
        print("Press Ctrl+C to stop early\n")

        start_time = time.time()
        self.metrics_history = []

        try:
            while time.time() - start_time < duration_seconds:
                for gpu_id in range(self.device_count):
                    metrics = self.get_gpu_metrics(gpu_id)
                    if metrics:
                        self.metrics_history.append(metrics)
                        print(f"GPU {gpu_id}: {metrics.temperature}°C, "
                              f"{metrics.gpu_util_percent}% util, "
                              f"{metrics.memory_percent:.1f}% mem")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")

        self.print_monitoring_summary()

    def print_monitoring_summary(self):
        """Print summary of collected metrics."""
        if not self.metrics_history:
            print("No metrics collected")
            return

        print("\n" + "=" * 70)
        print("MONITORING SUMMARY")
        print("=" * 70)

        for gpu_id in range(self.device_count):
            gpu_metrics = [m for m in self.metrics_history if m.gpu_id == gpu_id]

            if not gpu_metrics:
                continue

            temps = [m.temperature for m in gpu_metrics]
            utils = [m.gpu_util_percent for m in gpu_metrics]
            mem_percents = [m.memory_percent for m in gpu_metrics]
            powers = [m.power_usage_w for m in gpu_metrics]

            print(f"\nGPU {gpu_id} ({gpu_metrics[0].name}):")
            print(f"  Temperature: avg={sum(temps)/len(temps):.1f}°C, "
                  f"max={max(temps):.1f}°C, min={min(temps):.1f}°C")
            print(f"  Utilization: avg={sum(utils)/len(utils):.1f}%, "
                  f"max={max(utils):.1f}%, min={min(utils):.1f}%")
            print(f"  Memory:      avg={sum(mem_percents)/len(mem_percents):.1f}%, "
                  f"max={max(mem_percents):.1f}%, min={min(mem_percents):.1f}%")
            print(f"  Power:       avg={sum(powers)/len(powers):.1f}W, "
                  f"max={max(powers):.1f}W, min={min(powers):.1f}W")

        print("=" * 70)

    def export_metrics(self, filepath: str):
        """Export collected metrics to JSON file."""
        data = [asdict(m) for m in self.metrics_history]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Metrics exported to {filepath}")

    def get_optimal_device(self) -> int:
        """
        Select optimal GPU based on current utilization.

        Returns GPU ID with lowest memory usage and utilization.
        """
        if not self.cuda_available or self.device_count == 0:
            return -1  # CPU

        if self.device_count == 1:
            return 0

        best_gpu = 0
        best_score = float('inf')

        for gpu_id in range(self.device_count):
            metrics = self.get_gpu_metrics(gpu_id)

            if metrics:
                # Score based on memory and utilization
                # Lower is better
                score = metrics.memory_percent * 0.7 + metrics.gpu_util_percent * 0.3

                if score < best_score:
                    best_score = score
                    best_gpu = gpu_id

        return best_gpu

    def clear_all_memory(self):
        """Clear GPU memory on all devices."""
        if self.cuda_available:
            for gpu_id in range(self.device_count):
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            print(f"Cleared memory on {self.device_count} GPU(s)")


def demonstrate_production_management():
    """Demonstrate production GPU management."""
    print("=" * 70)
    print("PRODUCTION GPU MANAGEMENT")
    print("=" * 70)

    manager = ProductionGPUManager()

    if not manager.cuda_available:
        print("\n⚠️  No GPU available")
        return

    # Health check
    print("\n1. Running health check...")
    manager.print_health_report()

    # Get optimal device
    print("\n2. Finding optimal GPU...")
    optimal_gpu = manager.get_optimal_device()
    print(f"   Optimal GPU: {optimal_gpu}")

    # Show metrics for each GPU
    print("\n3. Current GPU metrics:")
    for gpu_id in range(manager.device_count):
        metrics = manager.get_gpu_metrics(gpu_id)
        if metrics:
            print(f"\n   GPU {gpu_id}:")
            print(f"     Name: {metrics.name}")
            print(f"     Temperature: {metrics.temperature}°C")
            print(f"     Utilization: {metrics.gpu_util_percent}%")
            print(f"     Memory: {metrics.memory_used_mb:.0f}MB / {metrics.memory_total_mb:.0f}MB "
                  f"({metrics.memory_percent:.1f}%)")
            print(f"     Power: {metrics.power_usage_w:.1f}W / {metrics.power_limit_w:.1f}W")

    # Continuous monitoring (short demo)
    print("\n4. Starting continuous monitoring (10 seconds)...")
    manager.monitor_continuous(interval_seconds=2, duration_seconds=10)

    # Export metrics
    # manager.export_metrics("/tmp/gpu_metrics.json")

    print("\n" + "=" * 70)
    print("PRODUCTION BEST PRACTICES")
    print("=" * 70)
    print("""
1. Monitoring:
   - Set up continuous GPU monitoring
   - Alert on temperature > 80°C
   - Alert on memory > 90%
   - Track utilization trends

2. Health Checks:
   - Run health checks before training jobs
   - Implement automatic recovery from failures
   - Log all GPU errors with context

3. Resource Management:
   - Select optimal GPU automatically
   - Clear memory between jobs
   - Monitor for memory leaks
   - Implement job queuing for multi-user systems

4. Production Deployment:
   - Use CUDA_VISIBLE_DEVICES to isolate GPUs
   - Set up GPU quotas per user/job
   - Implement graceful degradation to CPU
   - Monitor power consumption for cost optimization

5. Maintenance:
   - Regular driver updates
   - Monitor for hardware failures
   - Track GPU performance degradation
   - Plan for GPU replacement cycles
    """)


if __name__ == "__main__":
    demonstrate_production_management()
```

---

## Summary and Next Steps

This implementation guide has covered:

1. **Environment Setup**: Production-grade GPU detection and validation
2. **PyTorch GPU Operations**: Efficient device management and data loading
3. **Memory Optimization**: Profiling, mixed precision, gradient checkpointing
4. **Multi-GPU**: DataParallel and scaling strategies
5. **Performance Profiling**: PyTorch Profiler and bottleneck analysis
6. **Error Handling**: Common errors and debugging utilities
7. **Production Management**: Monitoring, health checks, and best practices

### Testing Your Implementation

```bash
# 1. Verify GPU setup
python3 gpu_detector.py

# 2. Test device management
python3 device_manager.py

# 3. Profile memory usage
python3 memory_profiler.py

# 4. Run benchmarks
python3 cpu_vs_gpu_benchmark.py

# 5. Test multi-GPU (if available)
python3 multi_gpu_training.py

# 6. Profile performance
python3 gpu_profiler.py

# 7. Test production manager
python3 production_gpu_manager.py
```

### Production Deployment Checklist

- [ ] GPU drivers and CUDA toolkit installed
- [ ] PyTorch installed with correct CUDA version
- [ ] Health checks passing
- [ ] Monitoring system configured
- [ ] Error handling and recovery implemented
- [ ] Memory optimization techniques applied
- [ ] Performance profiled and optimized
- [ ] Multi-GPU scaling tested (if applicable)
- [ ] Documentation and runbooks created
- [ ] Team trained on GPU best practices

### Additional Resources

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/)
- [Mixed Precision Training Guide](https://pytorch.org/docs/stable/amp.html)
- [Distributed Training Tutorial](https://pytorch.org/tutorials/beginner/dist_overview.html)

---

**End of Implementation Guide**

For questions or issues, consult the debugging section or refer to the production GPU management utilities for troubleshooting.
