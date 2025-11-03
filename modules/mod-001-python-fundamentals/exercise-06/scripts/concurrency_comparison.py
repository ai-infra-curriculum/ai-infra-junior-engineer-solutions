#!/usr/bin/env python3
"""
Concurrency Comparison: Async vs Threading vs Multiprocessing

Demonstrates when to use each concurrency model with performance benchmarks.
"""

import asyncio
import time
import threading
import multiprocessing
from typing import List, Dict
import math


# ============================================================================
# CPU-Bound Task (compute-heavy)
# ============================================================================

def cpu_intensive_task(n: int) -> float:
    """
    CPU-intensive computation.

    Args:
        n: Input value

    Returns:
        Computed result
    """
    result = 0.0
    for i in range(1000000):
        result += math.sqrt(i * n)
    return result


# ============================================================================
# I/O-Bound Task (network/file operations)
# ============================================================================

def io_bound_task(task_id: int) -> Dict:
    """
    I/O-bound operation (simulated with sleep).

    Args:
        task_id: Task identifier

    Returns:
        Result dictionary
    """
    time.sleep(0.5)  # Simulate I/O wait
    return {"task_id": task_id, "result": task_id * 2}


async def io_bound_task_async(task_id: int) -> Dict:
    """
    Async I/O-bound operation.

    Args:
        task_id: Task identifier

    Returns:
        Result dictionary
    """
    await asyncio.sleep(0.5)  # Simulate async I/O
    return {"task_id": task_id, "result": task_id * 2}


# ============================================================================
# Sequential Execution
# ============================================================================

def run_cpu_sequential(tasks: int) -> float:
    """Run CPU tasks sequentially."""
    start = time.time()
    results = [cpu_intensive_task(i) for i in range(tasks)]
    elapsed = time.time() - start
    return elapsed


def run_io_sequential(tasks: int) -> float:
    """Run I/O tasks sequentially."""
    start = time.time()
    results = [io_bound_task(i) for i in range(tasks)]
    elapsed = time.time() - start
    return elapsed


# ============================================================================
# Threading Approach
# ============================================================================

def run_cpu_threading(tasks: int) -> float:
    """Run CPU tasks with threading."""
    start = time.time()

    threads = []
    results = []

    def worker(n: int):
        result = cpu_intensive_task(n)
        results.append(result)

    for i in range(tasks):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    elapsed = time.time() - start
    return elapsed


def run_io_threading(tasks: int) -> float:
    """Run I/O tasks with threading."""
    start = time.time()

    threads = []
    results = []

    def worker(task_id: int):
        result = io_bound_task(task_id)
        results.append(result)

    for i in range(tasks):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    elapsed = time.time() - start
    return elapsed


# ============================================================================
# Multiprocessing Approach
# ============================================================================

def run_cpu_multiprocessing(tasks: int) -> float:
    """Run CPU tasks with multiprocessing."""
    start = time.time()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(cpu_intensive_task, range(tasks))

    elapsed = time.time() - start
    return elapsed


def run_io_multiprocessing(tasks: int) -> float:
    """Run I/O tasks with multiprocessing."""
    start = time.time()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(io_bound_task, range(tasks))

    elapsed = time.time() - start
    return elapsed


# ============================================================================
# Async Approach
# ============================================================================

async def run_io_async(tasks: int) -> float:
    """Run I/O tasks with asyncio."""
    start = time.time()

    task_list = [io_bound_task_async(i) for i in range(tasks)]
    results = await asyncio.gather(*task_list)

    elapsed = time.time() - start
    return elapsed


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_cpu_bound():
    """Benchmark CPU-bound task with different approaches."""
    print("=" * 70)
    print("CPU-Bound Task Benchmark")
    print("=" * 70)
    print()
    print("Task: Mathematical computation (sqrt in tight loop)")
    print("Number of tasks: 4")
    print()

    num_tasks = 4

    # Sequential
    print("1. Sequential execution...")
    time_seq = run_cpu_sequential(num_tasks)
    print(f"   Time: {time_seq:.2f}s")
    print()

    # Threading (GIL-limited)
    print("2. Threading...")
    time_thread = run_cpu_threading(num_tasks)
    print(f"   Time: {time_thread:.2f}s")
    speedup_thread = time_seq / time_thread
    print(f"   Speedup: {speedup_thread:.2f}x")
    print()

    # Multiprocessing (true parallelism)
    print("3. Multiprocessing...")
    time_mp = run_cpu_multiprocessing(num_tasks)
    print(f"   Time: {time_mp:.2f}s")
    speedup_mp = time_seq / time_mp
    print(f"   Speedup: {speedup_mp:.2f}x")
    print()

    # Summary
    print("Summary:")
    print(f"  • Sequential:      {time_seq:.2f}s (baseline)")
    print(f"  • Threading:       {time_thread:.2f}s ({speedup_thread:.2f}x)")
    print(f"  • Multiprocessing: {time_mp:.2f}s ({speedup_mp:.2f}x)")
    print()
    print("Key Insight:")
    print("  For CPU-bound tasks, multiprocessing provides true parallelism")
    print("  by bypassing Python's GIL. Threading shows minimal improvement")
    print("  due to GIL contention.")
    print()


async def benchmark_io_bound():
    """Benchmark I/O-bound task with different approaches."""
    print("=" * 70)
    print("I/O-Bound Task Benchmark")
    print("=" * 70)
    print()
    print("Task: I/O operation (0.5s sleep simulating network/disk)")
    print("Number of tasks: 10")
    print()

    num_tasks = 10

    # Sequential
    print("1. Sequential execution...")
    time_seq = run_io_sequential(num_tasks)
    print(f"   Time: {time_seq:.2f}s")
    print()

    # Threading
    print("2. Threading...")
    time_thread = run_io_threading(num_tasks)
    print(f"   Time: {time_thread:.2f}s")
    speedup_thread = time_seq / time_thread
    print(f"   Speedup: {speedup_thread:.2f}x")
    print()

    # Async
    print("3. Async (asyncio)...")
    time_async = await run_io_async(num_tasks)
    print(f"   Time: {time_async:.2f}s")
    speedup_async = time_seq / time_async
    print(f"   Speedup: {speedup_async:.2f}x")
    print()

    # Multiprocessing
    print("4. Multiprocessing...")
    time_mp = run_io_multiprocessing(num_tasks)
    print(f"   Time: {time_mp:.2f}s")
    speedup_mp = time_seq / time_mp
    print(f"   Speedup: {speedup_mp:.2f}x")
    print()

    # Summary
    print("Summary:")
    print(f"  • Sequential:      {time_seq:.2f}s (baseline)")
    print(f"  • Threading:       {time_thread:.2f}s ({speedup_thread:.2f}x)")
    print(f"  • Async:           {time_async:.2f}s ({speedup_async:.2f}x)")
    print(f"  • Multiprocessing: {time_mp:.2f}s ({speedup_mp:.2f}x)")
    print()
    print("Key Insight:")
    print("  For I/O-bound tasks, async provides the best performance with")
    print("  lowest overhead. Threading is also effective. Multiprocessing")
    print("  has higher overhead but still provides parallelism.")
    print()


def explain_when_to_use():
    """Explain when to use each approach."""
    print("=" * 70)
    print("When to Use Each Approach")
    print("=" * 70)
    print()

    print("✓ Use ASYNC when:")
    print("  • Making many API calls or HTTP requests")
    print("  • Reading/writing multiple files")
    print("  • Database queries with connection pooling")
    print("  • WebSocket connections")
    print("  • Any I/O-bound operation that waits")
    print()
    print("  Advantages:")
    print("    - Lowest overhead (single thread)")
    print("    - Best for thousands of concurrent operations")
    print("    - Clean syntax with async/await")
    print("    - Easy error handling")
    print()
    print("  ML Examples:")
    print("    - Downloading multiple model files")
    print("    - Batch inference API calls")
    print("    - Loading training data from distributed storage")
    print("    - Concurrent model serving requests")
    print()

    print("✓ Use THREADING when:")
    print("  • I/O-bound tasks (but async not available)")
    print("  • Working with blocking libraries")
    print("  • Need shared memory between tasks")
    print("  • Simple parallelism for I/O operations")
    print()
    print("  Advantages:")
    print("    - Works with any Python library")
    print("    - Shared memory (easier state management)")
    print("    - Lower overhead than multiprocessing")
    print()
    print("  ML Examples:")
    print("    - Real-time data preprocessing pipeline")
    print("    - Background model loading")
    print("    - Concurrent logging and monitoring")
    print()

    print("✓ Use MULTIPROCESSING when:")
    print("  • CPU-bound computations")
    print("  • Heavy mathematical operations")
    print("  • Need true parallelism (bypass GIL)")
    print("  • Processing independent data chunks")
    print()
    print("  Advantages:")
    print("    - True parallelism (uses multiple CPU cores)")
    print("    - Not limited by GIL")
    print("    - Best for computation-heavy tasks")
    print()
    print("  ML Examples:")
    print("    - Training multiple models in parallel")
    print("    - Data augmentation (image transforms)")
    print("    - Feature engineering on large datasets")
    print("    - Hyperparameter search (parallel trials)")
    print("    - Batch prediction on CPU")
    print()

    print("✓ Use SEQUENTIAL when:")
    print("  • Task requires less than 1 second total")
    print("  • Tasks must execute in specific order")
    print("  • Debugging and development")
    print("  • Overhead of concurrency not worth it")
    print()


def explain_gil():
    """Explain Python's Global Interpreter Lock."""
    print("=" * 70)
    print("Understanding Python's GIL (Global Interpreter Lock)")
    print("=" * 70)
    print()

    print("What is the GIL?")
    print("  Python's GIL is a mutex that protects access to Python objects,")
    print("  preventing multiple threads from executing Python bytecode at once.")
    print()

    print("Impact on Threading:")
    print("  • Only ONE thread can execute Python code at a time")
    print("  • CPU-bound tasks don't benefit from threading")
    print("  • I/O-bound tasks CAN benefit (thread releases GIL during I/O)")
    print()

    print("Why doesn't async have this problem?")
    print("  • Async uses a single thread (cooperative multitasking)")
    print("  • Tasks voluntarily yield control (await points)")
    print("  • No GIL contention since only one task runs at a time")
    print("  • Perfect for I/O-bound operations")
    print()

    print("How does multiprocessing bypass the GIL?")
    print("  • Each process has its own Python interpreter")
    print("  • Each process has its own GIL")
    print("  • True parallelism on multiple CPU cores")
    print("  • Trade-off: higher memory overhead and IPC cost")
    print()


def show_performance_chart():
    """Show performance characteristics chart."""
    print("=" * 70)
    print("Performance Characteristics Comparison")
    print("=" * 70)
    print()

    print(f"{'Characteristic':<20} {'Async':<15} {'Threading':<15} {'Multiprocessing':<15}")
    print("-" * 70)
    print(f"{'CPU-bound tasks':<20} {'Poor':<15} {'Poor':<15} {'Excellent':<15}")
    print(f"{'I/O-bound tasks':<20} {'Excellent':<15} {'Good':<15} {'Good':<15}")
    print(f"{'Memory overhead':<20} {'Very Low':<15} {'Low':<15} {'High':<15}")
    print(f"{'Startup cost':<20} {'Very Low':<15} {'Low':<15} {'High':<15}")
    print(f"{'Max concurrency':<20} {'Thousands':<15} {'Hundreds':<15} {'CPU cores':<15}")
    print(f"{'Shared memory':<20} {'Yes (easy)':<15} {'Yes (easy)':<15} {'No (IPC)':<15}")
    print(f"{'Bypasses GIL':<20} {'N/A':<15} {'No':<15} {'Yes':<15}")
    print(f"{'Error handling':<20} {'Excellent':<15} {'Good':<15} {'Complex':<15}")
    print(f"{'Debugging':<20} {'Good':<15} {'Hard':<15} {'Very Hard':<15}")
    print()


async def main():
    """Run all demonstrations and benchmarks."""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  Concurrency Comparison".center(68) + "█")
    print("█" + "  Async vs Threading vs Multiprocessing".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")

    # Explanation
    explain_when_to_use()

    # GIL explanation
    explain_gil()

    # Performance chart
    show_performance_chart()

    # CPU-bound benchmark
    benchmark_cpu_bound()

    # I/O-bound benchmark
    await benchmark_io_bound()

    # Summary
    print("=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("1. Async: Best for I/O-bound tasks (API calls, file I/O)")
    print("2. Threading: Good for I/O when async not available")
    print("3. Multiprocessing: Required for CPU-bound parallelism")
    print("4. Sequential: Fine for short tasks or when order matters")
    print("5. Python's GIL prevents true threading parallelism")
    print("6. Choose based on task type, not assumption")
    print("7. Async has lowest overhead, multiprocessing highest")
    print("=" * 70)
    print()

    print("Real-World ML Recommendations:")
    print("  • Data loading: Async (reading many files)")
    print("  • Data preprocessing: Multiprocessing (CPU-heavy)")
    print("  • API inference: Async (many requests)")
    print("  • Batch inference: Multiprocessing (CPU-bound)")
    print("  • Model training: Multiprocessing (parallel experiments)")
    print("  • Monitoring/logging: Threading or Async")
    print()


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    multiprocessing.set_start_method('fork', force=True)
    asyncio.run(main())
