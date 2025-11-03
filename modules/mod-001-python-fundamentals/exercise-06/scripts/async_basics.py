#!/usr/bin/env python3
"""
Async Programming Basics

Demonstrates fundamental async/await concepts and concurrent execution patterns.
"""

import asyncio
import time
from typing import Dict, List


async def download_model(model_name: str) -> Dict:
    """
    Simulate async model download.

    Args:
        model_name: Name of model to download

    Returns:
        Download result dictionary
    """
    print(f"  → Starting download: {model_name}")
    await asyncio.sleep(2)  # Simulate network delay
    print(f"  ✓ Completed download: {model_name}")
    return {"name": model_name, "size_mb": 150, "status": "downloaded"}


async def load_dataset(dataset_name: str) -> Dict:
    """
    Simulate async dataset loading.

    Args:
        dataset_name: Name of dataset to load

    Returns:
        Dataset dictionary
    """
    print(f"  → Loading dataset: {dataset_name}")
    await asyncio.sleep(1)  # Simulate I/O
    print(f"  ✓ Loaded dataset: {dataset_name}")
    return {"name": dataset_name, "samples": 10000, "features": 50}


async def preprocess_data(data: Dict) -> Dict:
    """
    Simulate async data preprocessing.

    Args:
        data: Dataset to preprocess

    Returns:
        Preprocessed dataset
    """
    print(f"  → Preprocessing: {data['name']}")
    await asyncio.sleep(1.5)  # Simulate processing
    print(f"  ✓ Preprocessed: {data['name']}")
    return {**data, "preprocessed": True, "normalized": True}


async def sequential_execution():
    """Execute tasks sequentially (one after another)"""
    print("=" * 70)
    print("Sequential Execution")
    print("=" * 70)
    print("Tasks run one at a time, waiting for each to complete\n")

    start = time.time()

    # Each task waits for previous to complete
    model = await download_model("resnet50")
    data = await load_dataset("imagenet")
    processed = await preprocess_data(data)

    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Model: {model['name']}")
    print(f"  Dataset: {data['name']} ({data['samples']} samples)")
    print(f"  Preprocessed: {processed['preprocessed']}")
    print(f"\n⏱  Sequential time: {elapsed:.2f}s")
    print()


async def concurrent_execution():
    """Execute tasks concurrently (simultaneously)"""
    print("=" * 70)
    print("Concurrent Execution")
    print("=" * 70)
    print("Tasks run simultaneously, waiting only for the slowest\n")

    start = time.time()

    # Create tasks but don't await yet
    model_task = download_model("resnet50")
    data_task = load_dataset("imagenet")

    # Wait for both to complete concurrently
    model, data = await asyncio.gather(model_task, data_task)

    # Now preprocess (depends on data)
    processed = await preprocess_data(data)

    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Model: {model['name']}")
    print(f"  Dataset: {data['name']} ({data['samples']} samples)")
    print(f"  Preprocessed: {processed['preprocessed']}")
    print(f"\n⏱  Concurrent time: {elapsed:.2f}s")
    print()


async def demonstrate_gather():
    """Demonstrate asyncio.gather() with multiple tasks"""
    print("=" * 70)
    print("asyncio.gather() - Running Multiple Tasks")
    print("=" * 70)
    print()

    async def task(name: str, duration: float) -> str:
        print(f"  → Task {name} started")
        await asyncio.sleep(duration)
        print(f"  ✓ Task {name} completed")
        return f"{name}_result"

    start = time.time()

    # Run 5 tasks concurrently
    results = await asyncio.gather(
        task("A", 1.0),
        task("B", 0.5),
        task("C", 1.5),
        task("D", 0.8),
        task("E", 1.2)
    )

    elapsed = time.time() - start

    print(f"\nResults: {results}")
    print(f"⏱  Total time: {elapsed:.2f}s (max of individual times)")
    print(f"   Sequential would take: {1.0 + 0.5 + 1.5 + 0.8 + 1.2:.1f}s")
    print()


async def demonstrate_task_creation():
    """Demonstrate explicit task creation with create_task()"""
    print("=" * 70)
    print("asyncio.create_task() - Background Tasks")
    print("=" * 70)
    print()

    async def background_work(task_id: int) -> str:
        print(f"  → Background task {task_id} started")
        await asyncio.sleep(1)
        print(f"  ✓ Background task {task_id} completed")
        return f"task_{task_id}_done"

    # Create tasks immediately (they start running in background)
    task1 = asyncio.create_task(background_work(1))
    task2 = asyncio.create_task(background_work(2))
    task3 = asyncio.create_task(background_work(3))

    print("Tasks created and running in background...")
    print("Doing other work while tasks run...")
    await asyncio.sleep(0.5)
    print("Still waiting for tasks to complete...")

    # Wait for all tasks
    results = await asyncio.gather(task1, task2, task3)

    print(f"\nAll tasks completed: {results}")
    print()


async def compare_sleep():
    """Compare asyncio.sleep() vs time.sleep()"""
    print("=" * 70)
    print("asyncio.sleep() vs time.sleep()")
    print("=" * 70)
    print()

    print("Using asyncio.sleep() (non-blocking):")
    start = time.time()

    # These can run concurrently
    await asyncio.gather(
        asyncio.sleep(1),
        asyncio.sleep(1),
        asyncio.sleep(1)
    )

    async_elapsed = time.time() - start
    print(f"  ⏱  Time: {async_elapsed:.2f}s (concurrent)\n")

    print("Using time.sleep() would block:")
    print("  ✗ time.sleep(1) + time.sleep(1) + time.sleep(1) = ~3.0s")
    print("  ✗ Blocks event loop, no concurrency possible")
    print()


def main():
    """Run all demonstrations"""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  Async Programming Basics".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")

    # Sequential execution
    asyncio.run(sequential_execution())

    # Concurrent execution
    asyncio.run(concurrent_execution())

    # Calculate speedup
    # Sequential: 2s + 1s + 1.5s = 4.5s
    # Concurrent: max(2s, 1s) + 1.5s = 3.5s
    speedup = 4.5 / 3.5
    print("=" * 70)
    print(f"Speedup: {speedup:.2f}x faster with concurrent execution")
    print("=" * 70)
    print()

    # Demonstrate gather
    asyncio.run(demonstrate_gather())

    # Demonstrate create_task
    asyncio.run(demonstrate_task_creation())

    # Compare sleep methods
    asyncio.run(compare_sleep())

    print("=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("1. Use 'async def' to define coroutines")
    print("2. Use 'await' to call async functions")
    print("3. Use asyncio.gather() to run tasks concurrently")
    print("4. Use asyncio.sleep() (not time.sleep()) in async code")
    print("5. Concurrent execution can provide significant speedups for I/O")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
