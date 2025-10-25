#!/usr/bin/env python3
"""
GPU Memory Management Module

This module demonstrates GPU memory tracking, allocation, and cleanup
best practices for ML workloads.
"""

import time
from typing import Dict, Optional
import torch


def get_gpu_memory_info(device_id: int = 0) -> Optional[Dict[str, float]]:
    """
    Get current GPU memory usage information.

    Args:
        device_id: GPU device ID (default: 0)

    Returns:
        Dictionary with memory info in GB, or None if GPU not available
    """
    if not torch.cuda.is_available():
        return None

    return {
        "allocated_gb": torch.cuda.memory_allocated(device_id) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(device_id) / 1024**3,
        "total_gb": torch.cuda.get_device_properties(device_id).total_memory / 1024**3,
        "free_gb": (
            torch.cuda.get_device_properties(device_id).total_memory -
            torch.cuda.memory_allocated(device_id)
        ) / 1024**3
    }


def print_gpu_memory(label: str = "", device_id: int = 0) -> None:
    """
    Print current GPU memory usage with a label.

    Args:
        label: Description label for this memory checkpoint
        device_id: GPU device ID
    """
    if not torch.cuda.is_available():
        print("GPU not available")
        return

    info = get_gpu_memory_info(device_id)

    if label:
        print(f"\n{label}")

    print(f"  Allocated: {info['allocated_gb']:.2f} GB")
    print(f"  Reserved:  {info['reserved_gb']:.2f} GB")
    print(f"  Total:     {info['total_gb']:.2f} GB")
    print(f"  Free:      {info['free_gb']:.2f} GB")


def track_gpu_memory(
    operation_name: str,
    operation_func,
    *args,
    **kwargs
) -> tuple:
    """
    Track GPU memory usage before and after an operation.

    Args:
        operation_name: Name of the operation for logging
        operation_func: Function to execute
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Tuple of (result, memory_delta_gb)
    """
    if not torch.cuda.is_available():
        result = operation_func(*args, **kwargs)
        return result, 0.0

    # Memory before operation
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / 1024**3

    # Execute operation
    result = operation_func(*args, **kwargs)

    # Memory after operation
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated() / 1024**3

    memory_delta = mem_after - mem_before

    print(f"\n{operation_name}:")
    print(f"  Memory before: {mem_before:.2f} GB")
    print(f"  Memory after:  {mem_after:.2f} GB")
    print(f"  Memory delta:  {memory_delta:+.2f} GB")

    return result, memory_delta


def cleanup_gpu_memory() -> None:
    """
    Clean up GPU memory by emptying the cache.
    """
    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def demonstrate_memory_lifecycle() -> None:
    """
    Demonstrate GPU memory allocation, usage, and cleanup lifecycle.
    """
    if not torch.cuda.is_available():
        print("GPU not available - skipping memory management demonstration")
        return

    print("=" * 60)
    print("GPU Memory Management Demonstration")
    print("=" * 60)

    # Initial state
    print_gpu_memory("1. Initial State")

    # Allocate small tensor
    print("\n" + "=" * 60)
    print("2. Allocating 1000x1000 tensor")
    print("=" * 60)
    tensor1 = torch.randn(1000, 1000, device='cuda')
    print_gpu_memory("After allocation")

    # Allocate larger tensor
    print("\n" + "=" * 60)
    print("3. Allocating 2000x2000 tensor")
    print("=" * 60)
    tensor2 = torch.randn(2000, 2000, device='cuda')
    print_gpu_memory("After second allocation")

    # Perform operation
    print("\n" + "=" * 60)
    print("4. Performing matrix multiplication")
    print("=" * 60)
    result = torch.matmul(tensor1[:1000, :1000], tensor1[:1000, :1000])
    print_gpu_memory("After computation")

    # Delete first tensor
    print("\n" + "=" * 60)
    print("5. Deleting first tensor")
    print("=" * 60)
    del tensor1
    print_gpu_memory("After deletion (memory still reserved by cache)")

    # Empty cache
    print("\n" + "=" * 60)
    print("6. Emptying cache")
    print("=" * 60)
    torch.cuda.empty_cache()
    print_gpu_memory("After emptying cache (memory returned to GPU)")

    # Cleanup remaining
    print("\n" + "=" * 60)
    print("7. Final cleanup")
    print("=" * 60)
    del tensor2, result
    torch.cuda.empty_cache()
    print_gpu_memory("After final cleanup")


def demonstrate_oom_handling() -> None:
    """
    Demonstrate handling of Out of Memory (OOM) errors.
    """
    if not torch.cuda.is_available():
        print("\nGPU not available - skipping OOM demonstration")
        return

    print("\n" + "=" * 60)
    print("Out of Memory (OOM) Error Handling")
    print("=" * 60)

    try:
        # Try to allocate a very large tensor
        print("\nAttempting to allocate massive tensor (expected to fail)...")
        huge_tensor = torch.randn(50000, 50000, device='cuda')
        print("✓ Allocation succeeded (you have a lot of GPU memory!)")
        del huge_tensor
        torch.cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("✗ Out of Memory Error (expected):")
            print(f"   {str(e)[:100]}...")
            print("\nThis is expected! GPU memory is limited.")
            print("\nIn production, you would:")
            print("  1. Use smaller batch sizes")
            print("  2. Use gradient checkpointing")
            print("  3. Use model parallelism across multiple GPUs")
            print("  4. Use mixed precision (FP16/BF16)")
            print("  5. Clear cache and retry with smaller allocation")

            # Clean up any partial allocations
            torch.cuda.empty_cache()
        else:
            raise


def benchmark_memory_transfer() -> None:
    """
    Benchmark CPU to GPU memory transfer speeds.
    """
    if not torch.cuda.is_available():
        print("\nGPU not available - skipping transfer benchmark")
        return

    print("\n" + "=" * 60)
    print("Memory Transfer Benchmark")
    print("=" * 60)

    sizes = [100, 1000, 5000]

    print(f"\n{'Size':<10} {'CPU→GPU (ms)':<15} {'GPU→CPU (ms)':<15} {'GB/s':<10}")
    print("-" * 55)

    for size in sizes:
        # Create tensor on CPU
        cpu_tensor = torch.randn(size, size)
        tensor_size_gb = (cpu_tensor.element_size() * cpu_tensor.nelement()) / 1024**3

        # CPU to GPU transfer
        start = time.time()
        gpu_tensor = cpu_tensor.to('cuda')
        torch.cuda.synchronize()
        cpu_to_gpu_time = time.time() - start

        # GPU to CPU transfer
        start = time.time()
        cpu_tensor_back = gpu_tensor.to('cpu')
        torch.cuda.synchronize()
        gpu_to_cpu_time = time.time() - start

        # Calculate throughput
        throughput = tensor_size_gb / cpu_to_gpu_time

        print(
            f"{size}x{size:<5} {cpu_to_gpu_time*1000:>12.2f}  "
            f"{gpu_to_cpu_time*1000:>12.2f}  {throughput:>8.2f}"
        )

        # Cleanup
        del cpu_tensor, gpu_tensor, cpu_tensor_back

    torch.cuda.empty_cache()


def monitor_memory_during_operation() -> None:
    """
    Monitor memory usage during a series of operations.
    """
    if not torch.cuda.is_available():
        print("\nGPU not available - skipping memory monitoring")
        return

    print("\n" + "=" * 60)
    print("Memory Monitoring During Operations")
    print("=" * 60)

    # Start with clean slate
    torch.cuda.empty_cache()

    operations = []

    # Operation 1: Create tensors
    def create_tensors():
        return [torch.randn(1000, 1000, device='cuda') for _ in range(5)]

    tensors, mem_delta = track_gpu_memory(
        "Creating 5 tensors (1000x1000)",
        create_tensors
    )
    operations.append(("Create tensors", mem_delta))

    # Operation 2: Matrix multiplication
    def matmul_operation():
        return torch.matmul(tensors[0], tensors[1])

    result, mem_delta = track_gpu_memory(
        "Matrix multiplication",
        matmul_operation
    )
    operations.append(("Matrix multiply", mem_delta))

    # Operation 3: Delete some tensors
    def delete_tensors():
        for t in tensors[:3]:
            del t
        torch.cuda.empty_cache()

    _, mem_delta = track_gpu_memory(
        "Deleting 3 tensors",
        delete_tensors
    )
    operations.append(("Delete tensors", mem_delta))

    # Summary
    print("\n" + "=" * 60)
    print("Memory Usage Summary")
    print("=" * 60)
    print(f"\n{'Operation':<25} {'Memory Change (GB)':<20}")
    print("-" * 45)
    for op_name, delta in operations:
        print(f"{op_name:<25} {delta:>+18.3f}")

    # Final cleanup
    del result
    torch.cuda.empty_cache()


def main():
    """Main function to run memory management demonstrations."""
    print("=" * 60)
    print("GPU Memory Management Examples")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\nNo GPU available. These examples require a CUDA-capable GPU.")
        print("Install PyTorch with CUDA support and run on a GPU-enabled system.")
        return

    # Run demonstrations
    demonstrate_memory_lifecycle()
    demonstrate_oom_handling()
    benchmark_memory_transfer()
    monitor_memory_during_operation()

    print("\n" + "=" * 60)
    print("Memory Management Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
