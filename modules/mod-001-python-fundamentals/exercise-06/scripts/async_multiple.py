#!/usr/bin/env python3
"""
Async Programming with Multiple Tasks

Demonstrates processing multiple items concurrently with asyncio.
"""

import asyncio
import random
import time
from typing import List, Dict


async def process_sample(sample_id: int) -> Dict:
    """
    Process a single sample asynchronously.

    Args:
        sample_id: Unique sample identifier

    Returns:
        Processing result dictionary
    """
    # Simulate variable processing time
    delay = random.uniform(0.1, 0.5)
    await asyncio.sleep(delay)

    return {
        "sample_id": sample_id,
        "processed": True,
        "processing_time": delay,
        "result": sample_id * 2
    }


async def process_batch_async(batch: List[int]) -> List[Dict]:
    """
    Process entire batch concurrently.

    Args:
        batch: List of sample IDs to process

    Returns:
        List of processing results
    """
    # Create all tasks
    tasks = [process_sample(sample_id) for sample_id in batch]

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    return results


async def download_model(name: str) -> tuple:
    """
    Download a single model.

    Args:
        name: Model name

    Returns:
        Tuple of (name, metadata)
    """
    # Simulate variable download time
    delay = random.uniform(0.5, 2.0)
    await asyncio.sleep(delay)

    return name, {
        "name": name,
        "downloaded": True,
        "size_mb": random.randint(50, 500),
        "download_time": delay
    }


async def download_multiple_models(model_names: List[str]) -> Dict[str, Dict]:
    """
    Download multiple models concurrently.

    Args:
        model_names: List of model names to download

    Returns:
        Dictionary mapping model names to metadata
    """
    tasks = [download_model(name) for name in model_names]
    results = await asyncio.gather(*tasks)

    # Convert list of tuples to dictionary
    return dict(results)


async def fetch_with_progress(items: List[int], batch_size: int = 10) -> List[Dict]:
    """
    Process items in batches with progress reporting.

    Args:
        items: Items to process
        batch_size: Number of items per batch

    Returns:
        List of all results
    """
    all_results = []

    # Split into batches
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

    for batch_num, batch in enumerate(batches, 1):
        print(f"  Processing batch {batch_num}/{len(batches)} ({len(batch)} items)...")

        batch_results = await process_batch_async(batch)
        all_results.extend(batch_results)

        print(f"  ✓ Batch {batch_num} complete")

    return all_results


async def race_tasks():
    """Demonstrate asyncio.wait() with FIRST_COMPLETED"""
    print("=" * 70)
    print("Task Racing - First to Complete Wins")
    print("=" * 70)
    print()

    async def racer(name: str, speed: float) -> str:
        print(f"  🏁 {name} started")
        await asyncio.sleep(speed)
        print(f"  ✓ {name} finished!")
        return name

    # Create racing tasks
    tasks = {
        asyncio.create_task(racer("Lightning", 0.5)),
        asyncio.create_task(racer("Thunder", 0.8)),
        asyncio.create_task(racer("Storm", 1.2))
    }

    # Wait for first to complete
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    winner = list(done)[0].result()
    print(f"\n🏆 Winner: {winner}")

    # Cancel remaining tasks
    for task in pending:
        task.cancel()

    print(f"   Cancelled {len(pending)} remaining tasks")
    print()


async def process_with_timeout():
    """Demonstrate timeout handling"""
    print("=" * 70)
    print("Task Timeout Handling")
    print("=" * 70)
    print()

    async def slow_operation(duration: float) -> str:
        print(f"  → Starting operation (will take {duration}s)")
        await asyncio.sleep(duration)
        return "completed"

    # Task that completes in time
    print("Test 1: Operation within timeout")
    try:
        result = await asyncio.wait_for(slow_operation(1.0), timeout=2.0)
        print(f"  ✓ {result}\n")
    except asyncio.TimeoutError:
        print("  ✗ Timed out\n")

    # Task that times out
    print("Test 2: Operation exceeds timeout")
    try:
        result = await asyncio.wait_for(slow_operation(3.0), timeout=1.0)
        print(f"  ✓ {result}\n")
    except asyncio.TimeoutError:
        print("  ✗ Timed out after 1.0s\n")


async def main():
    """Run all demonstrations"""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  Async Programming - Multiple Tasks".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")

    # Example 1: Process batch async
    print("=" * 70)
    print("Example 1: Processing Batch Asynchronously")
    print("=" * 70)
    print()

    batch = list(range(20))
    print(f"Processing {len(batch)} samples concurrently...")

    start = time.time()
    results = await process_batch_async(batch)
    elapsed = time.time() - start

    print(f"\n✓ Processed {len(results)} samples in {elapsed:.2f}s")
    print(f"  Average time per sample: {elapsed/len(results):.3f}s")

    # Calculate what sequential would take
    total_time = sum(r['processing_time'] for r in results)
    print(f"  Sequential would take: ~{total_time:.2f}s")
    print(f"  Speedup: {total_time/elapsed:.1f}x")
    print()

    # Example 2: Download multiple models
    print("=" * 70)
    print("Example 2: Downloading Multiple Models")
    print("=" * 70)
    print()

    models = ["resnet50", "vgg16", "mobilenet", "efficientnet", "bert-base"]
    print(f"Downloading {len(models)} models concurrently...")

    start = time.time()
    downloaded = await download_multiple_models(models)
    elapsed = time.time() - start

    print(f"\n✓ Downloaded {len(downloaded)} models in {elapsed:.2f}s")
    print(f"\nModels:")
    for name, metadata in downloaded.items():
        print(f"  • {name}: {metadata['size_mb']}MB "
              f"(took {metadata['download_time']:.2f}s)")

    # Calculate sequential time
    total_time = sum(m['download_time'] for m in downloaded.values())
    print(f"\n  Sequential would take: ~{total_time:.2f}s")
    print(f"  Speedup: {total_time/elapsed:.1f}x")
    print()

    # Example 3: Batched processing with progress
    print("=" * 70)
    print("Example 3: Batched Processing with Progress")
    print("=" * 70)
    print()

    items = list(range(50))
    print(f"Processing {len(items)} items in batches of 10...\n")

    start = time.time()
    results = await fetch_with_progress(items, batch_size=10)
    elapsed = time.time() - start

    print(f"\n✓ All {len(results)} items processed in {elapsed:.2f}s")
    print()

    # Example 4: Task racing
    await race_tasks()

    # Example 5: Timeout handling
    await process_with_timeout()

    # Summary
    print("=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("1. Use list comprehension to create multiple tasks")
    print("2. asyncio.gather(*tasks) waits for all tasks")
    print("3. asyncio.wait() provides more control (FIRST_COMPLETED, etc.)")
    print("4. asyncio.wait_for() adds timeout to any coroutine")
    print("5. Process in batches for large datasets")
    print("6. Concurrent execution provides significant speedups")
    print("=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
