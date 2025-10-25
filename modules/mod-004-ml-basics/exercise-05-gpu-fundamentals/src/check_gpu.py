#!/usr/bin/env python3
"""
GPU Detection and Information Module

This module provides functions to detect GPU availability and report
detailed information about available CUDA devices.
"""

import sys
from typing import Dict, List, Optional
import torch


def check_gpu_availability() -> bool:
    """
    Check if CUDA-capable GPU is available.

    Returns:
        bool: True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()


def get_gpu_info() -> Dict[str, any]:
    """
    Get detailed information about available GPUs.

    Returns:
        Dict containing GPU information including:
        - cuda_available: bool
        - gpu_count: int
        - gpus: List of GPU details
        - cuda_version: str
        - pytorch_version: str
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": 0,
        "gpus": [],
        "cuda_version": None,
        "cudnn_version": None,
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0]
    }

    if info["cuda_available"]:
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()

        for i in range(info["gpu_count"]):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "compute_capability": torch.cuda.get_device_capability(i),
                "total_memory_gb": gpu_props.total_memory / 1024**3,
                "allocated_memory_gb": torch.cuda.memory_allocated(i) / 1024**3,
                "reserved_memory_gb": torch.cuda.memory_reserved(i) / 1024**3,
                "multi_processor_count": gpu_props.multi_processor_count,
            }
            info["gpus"].append(gpu_info)

    return info


def test_device_creation() -> Dict[str, bool]:
    """
    Test creating tensors on different devices.

    Returns:
        Dict with test results for CPU and GPU tensor creation
    """
    results = {
        "cpu_tensor_created": False,
        "gpu_tensor_created": False,
        "cpu_device": None,
        "gpu_device": None,
    }

    try:
        # Test CPU tensor creation
        cpu_tensor = torch.randn(1000, 1000)
        results["cpu_tensor_created"] = True
        results["cpu_device"] = str(cpu_tensor.device)
    except Exception as e:
        print(f"Error creating CPU tensor: {e}")

    if torch.cuda.is_available():
        try:
            # Test GPU tensor creation
            gpu_tensor = torch.randn(1000, 1000, device='cuda')
            results["gpu_tensor_created"] = True
            results["gpu_device"] = str(gpu_tensor.device)
        except Exception as e:
            print(f"Error creating GPU tensor: {e}")

    return results


def print_gpu_report() -> None:
    """
    Print a formatted report of GPU information.
    """
    print("=" * 60)
    print("GPU Detection and Information")
    print("=" * 60)

    cuda_available = check_gpu_availability()
    print(f"\nCUDA Available: {cuda_available}")

    if cuda_available:
        info = get_gpu_info()
        print(f"Number of GPUs: {info['gpu_count']}")

        for gpu in info["gpus"]:
            print(f"\n--- GPU {gpu['id']} ---")
            print(f"Name: {gpu['name']}")
            print(f"Compute Capability: {gpu['compute_capability']}")
            print(f"Total Memory: {gpu['total_memory_gb']:.2f} GB")
            print(f"Currently Allocated: {gpu['allocated_memory_gb']:.2f} GB")
            print(f"Currently Reserved: {gpu['reserved_memory_gb']:.2f} GB")
            print(f"Multi-Processor Count: {gpu['multi_processor_count']}")

        print(f"\nCUDA Version: {info['cuda_version']}")
        print(f"cuDNN Version: {info['cudnn_version']}")
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    else:
        print("\nNo CUDA-capable GPU detected.")
        print("This is fine for learning! We can still run everything on CPU.")
        print("\nTo use GPU in the future:")
        print("1. Get a machine with NVIDIA GPU")
        print("2. Install NVIDIA drivers")
        print("3. Install CUDA toolkit")
        print("4. Install PyTorch with CUDA support")

    # PyTorch configuration
    print("\n" + "=" * 60)
    print("PyTorch Configuration")
    print("=" * 60)
    info = get_gpu_info()
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"Python Version: {info['python_version']}")

    # Test device creation
    print("\n" + "=" * 60)
    print("Device Testing")
    print("=" * 60)

    test_results = test_device_creation()

    if test_results["cpu_tensor_created"]:
        print(f"✓ CPU tensor created successfully")
        print(f"  Device: {test_results['cpu_device']}")
    else:
        print("✗ Failed to create CPU tensor")

    if cuda_available:
        if test_results["gpu_tensor_created"]:
            print(f"✓ GPU tensor created successfully")
            print(f"  Device: {test_results['gpu_device']}")
        else:
            print("✗ Failed to create GPU tensor")
    else:
        print("⚠ GPU tensors not available (CPU only mode)")


def get_device() -> torch.device:
    """
    Get the best available device (GPU if available, otherwise CPU).

    Returns:
        torch.device: The selected device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    """Main function to run GPU detection report."""
    print_gpu_report()


if __name__ == "__main__":
    main()
