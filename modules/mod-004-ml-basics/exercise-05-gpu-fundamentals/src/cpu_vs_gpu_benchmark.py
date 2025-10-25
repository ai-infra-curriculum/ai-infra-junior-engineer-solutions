#!/usr/bin/env python3
"""
CPU vs GPU Performance Benchmark Module

This module benchmarks matrix multiplication operations on CPU and GPU
to demonstrate the performance advantages of GPU acceleration.
"""

import time
from typing import Dict, List, Tuple
import torch


def benchmark_matmul(
    size: int,
    device: torch.device,
    iterations: int = 100,
    warmup: int = 10
) -> float:
    """
    Benchmark matrix multiplication on specified device.

    Args:
        size: Size of square matrices to multiply (size x size)
        device: Device to run benchmark on (CPU or CUDA)
        iterations: Number of iterations for benchmark
        warmup: Number of warmup iterations

    Returns:
        Average time per operation in milliseconds
    """
    # Create random matrices on specified device
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warmup iterations (important for GPU!)
    for _ in range(warmup):
        _ = torch.matmul(a, b)

    # Synchronize GPU before timing (GPU operations are async)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark iterations
    start = time.time()
    for _ in range(iterations):
        c = torch.matmul(a, b)

    # Synchronize again to ensure all operations complete
    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_time = elapsed / iterations

    return avg_time * 1000  # Convert to milliseconds


def compare_cpu_gpu(
    sizes: List[int],
    iterations: int = 100
) -> List[Dict[str, any]]:
    """
    Compare CPU and GPU performance across multiple matrix sizes.

    Args:
        sizes: List of matrix sizes to test
        iterations: Number of iterations per test

    Returns:
        List of dictionaries containing benchmark results
    """
    results = []

    for size in sizes:
        result = {
            "size": size,
            "cpu_time_ms": None,
            "gpu_time_ms": None,
            "speedup": None,
            "gpu_available": torch.cuda.is_available()
        }

        # CPU benchmark
        print(f"\nBenchmarking matrix size {size}x{size} on CPU...")
        result["cpu_time_ms"] = benchmark_matmul(
            size,
            torch.device('cpu'),
            iterations
        )

        # GPU benchmark (if available)
        if torch.cuda.is_available():
            print(f"Benchmarking matrix size {size}x{size} on GPU...")
            result["gpu_time_ms"] = benchmark_matmul(
                size,
                torch.device('cuda'),
                iterations
            )
            result["speedup"] = result["cpu_time_ms"] / result["gpu_time_ms"]
        else:
            print("GPU not available, skipping GPU benchmark")

        results.append(result)

    return results


def print_benchmark_results(results: List[Dict[str, any]]) -> None:
    """
    Print formatted benchmark results.

    Args:
        results: List of benchmark result dictionaries
    """
    print("\n" + "=" * 60)
    print("Matrix Multiplication Benchmark Results")
    print("=" * 60)

    for result in results:
        print(f"\nMatrix size: {result['size']}x{result['size']}")
        print(f"  CPU: {result['cpu_time_ms']:.2f} ms")

        if result['gpu_time_ms'] is not None:
            print(f"  GPU: {result['gpu_time_ms']:.2f} ms")
            print(f"  Speedup: {result['speedup']:.1f}x", end="")

            if result['speedup'] > 1:
                print(" (GPU faster)")
            else:
                print(" (CPU faster - overhead costs dominate)")
        else:
            print("  GPU: Not available")

    # Summary
    print("\n" + "=" * 60)
    print("Key Observations:")
    print("=" * 60)

    if any(r['gpu_time_ms'] is not None for r in results):
        print("1. GPU has overhead costs for small matrices")
        print("2. GPU advantage grows with matrix size")
        print("3. For ML workloads (large matrices), GPU is significantly faster")
        print("4. Typical ML speedups: 10-100x for training, 5-50x for inference")
    else:
        print("1. GPU not available - running in CPU-only mode")
        print("2. With a GPU, you would see 10-100x speedup for large matrices")
        print("3. ML training and inference are much faster with GPU acceleration")


def benchmark_elementwise_operations(
    size: int,
    device: torch.device,
    iterations: int = 100
) -> Dict[str, float]:
    """
    Benchmark various element-wise operations.

    Args:
        size: Size of tensors to test
        device: Device to run on
        iterations: Number of iterations

    Returns:
        Dictionary with timing results for different operations
    """
    tensor = torch.randn(size, size, device=device)
    results = {}

    operations = {
        "relu": lambda x: torch.relu(x),
        "sigmoid": lambda x: torch.sigmoid(x),
        "tanh": lambda x: torch.tanh(x),
        "addition": lambda x: x + x,
        "multiplication": lambda x: x * x,
    }

    for op_name, op_func in operations.items():
        # Warmup
        for _ in range(10):
            _ = op_func(tensor)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(iterations):
            _ = op_func(tensor)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.time() - start
        results[op_name] = (elapsed / iterations) * 1000

    return results


def run_all_benchmarks() -> None:
    """
    Run comprehensive benchmark suite.
    """
    print("=" * 60)
    print("GPU Performance Benchmarking Suite")
    print("=" * 60)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("\nNo GPU detected - running CPU-only benchmarks")

    # Matrix multiplication benchmarks
    print("\n" + "=" * 60)
    print("Part 1: Matrix Multiplication Benchmarks")
    print("=" * 60)

    sizes = [100, 500, 1000, 2000]
    results = compare_cpu_gpu(sizes, iterations=100)
    print_benchmark_results(results)

    # Element-wise operations (if GPU available)
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Part 2: Element-wise Operation Comparison")
        print("=" * 60)

        size = 1000
        print(f"\nTesting {size}x{size} tensors:")

        cpu_ops = benchmark_elementwise_operations(
            size,
            torch.device('cpu')
        )
        gpu_ops = benchmark_elementwise_operations(
            size,
            torch.device('cuda')
        )

        print(f"\n{'Operation':<15} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
        print("-" * 55)

        for op_name in cpu_ops.keys():
            cpu_time = cpu_ops[op_name]
            gpu_time = gpu_ops[op_name]
            speedup = cpu_time / gpu_time

            print(
                f"{op_name:<15} {cpu_time:>10.3f}  {gpu_time:>10.3f}  "
                f"{speedup:>8.1f}x"
            )

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


def main():
    """Main function to run benchmarks."""
    run_all_benchmarks()


if __name__ == "__main__":
    main()
