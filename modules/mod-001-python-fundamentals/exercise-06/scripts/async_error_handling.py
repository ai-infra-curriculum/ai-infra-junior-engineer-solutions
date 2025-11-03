#!/usr/bin/env python3
"""
Async Error Handling

Demonstrates error handling patterns in asynchronous code.
"""

import asyncio
import random
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def risky_operation(task_id: int, failure_rate: float = 0.3) -> Dict:
    """
    Operation that might fail.

    Args:
        task_id: Task identifier
        failure_rate: Probability of failure

    Returns:
        Result dictionary

    Raises:
        ValueError: If operation fails
    """
    await asyncio.sleep(0.1)

    if random.random() < failure_rate:
        raise ValueError(f"Task {task_id} failed randomly")

    return {"task_id": task_id, "result": "success", "value": task_id * 2}


async def safe_risky_operation(task_id: int, failure_rate: float = 0.3) -> Dict:
    """
    Wrap risky operation with error handling.

    Args:
        task_id: Task identifier
        failure_rate: Failure probability

    Returns:
        Result dictionary (success or failure)
    """
    try:
        result = await risky_operation(task_id, failure_rate)
        return result
    except ValueError as e:
        logger.warning(f"Task {task_id} failed: {e}")
        return {"task_id": task_id, "result": "failed", "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in task {task_id}: {e}")
        return {"task_id": task_id, "result": "error", "error": str(e)}


async def run_tasks_with_error_handling(num_tasks: int, failure_rate: float = 0.3) -> Dict[str, int]:
    """
    Run multiple tasks with error handling.

    Args:
        num_tasks: Number of tasks to run
        failure_rate: Probability of failure per task

    Returns:
        Statistics dictionary
    """
    tasks = [safe_risky_operation(i, failure_rate) for i in range(num_tasks)]
    results = await asyncio.gather(*tasks)

    # Count outcomes
    successful = sum(1 for r in results if r["result"] == "success")
    failed = sum(1 for r in results if r["result"] == "failed")
    errors = sum(1 for r in results if r["result"] == "error")

    return {
        "total": num_tasks,
        "successful": successful,
        "failed": failed,
        "errors": errors,
        "success_rate": successful / num_tasks if num_tasks > 0 else 0
    }


async def retry_async(func, *args, max_retries: int = 3, delay: float = 0.5, **kwargs):
    """
    Retry async function on failure.

    Args:
        func: Async function to retry
        *args: Positional arguments
        max_retries: Maximum retry attempts
        delay: Delay between retries
        **kwargs: Keyword arguments

    Returns:
        Function result

    Raises:
        Exception: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} retry attempts failed")
                raise
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
            await asyncio.sleep(delay * (attempt + 1))  # Exponential backoff


async def gather_with_exceptions():
    """Demonstrate return_exceptions parameter"""
    print("=" * 70)
    print("asyncio.gather() with return_exceptions")
    print("=" * 70 + "\n")

    async def task_may_fail(task_id: int) -> str:
        await asyncio.sleep(0.1)
        if task_id % 3 == 0:
            raise ValueError(f"Task {task_id} designed to fail")
        return f"Task {task_id} success"

    print("Without return_exceptions (stops on first error):")
    try:
        results = await asyncio.gather(
            task_may_fail(1),
            task_may_fail(2),
            task_may_fail(3),  # Will raise
            task_may_fail(4)
        )
    except ValueError as e:
        print(f"  ✗ Stopped on error: {e}\n")

    print("With return_exceptions=True (continues despite errors):")
    results = await asyncio.gather(
        task_may_fail(1),
        task_may_fail(2),
        task_may_fail(3),  # Will raise but caught
        task_may_fail(4),
        return_exceptions=True
    )

    print("  Results:")
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"    Task {i}: ✗ {result}")
        else:
            print(f"    Task {i}: ✓ {result}")
    print()


async def main():
    """Run all demonstrations"""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  Async Error Handling".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")

    # Example 1: Error handling with gather
    await gather_with_exceptions()

    # Example 2: Safe error handling
    print("=" * 70)
    print("Error Handling with Individual Task Wrapping")
    print("=" * 70 + "\n")

    stats = await run_tasks_with_error_handling(20, failure_rate=0.3)
    print(f"Results:")
    print(f"  Total tasks: {stats['total']}")
    print(f"  Successful: {stats['successful']} ({stats['success_rate']:.0%})")
    print(f"  Failed: {stats['failed']}")
    print(f"  Errors: {stats['errors']}")
    print()

    # Example 3: Retry logic
    print("=" * 70)
    print("Retry Logic")
    print("=" * 70 + "\n")

    async def flaky_task():
        if random.random() < 0.7:  # 70% failure rate
            raise ValueError("Flaky task failed")
        return "success"

    try:
        result = await retry_async(flaky_task, max_retries=5, delay=0.2)
        print(f"✓ Retry succeeded: {result}")
    except Exception as e:
        print(f"✗ All retries failed: {e}")
    print()

    # Summary
    print("=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("1. Use return_exceptions=True to collect errors without stopping")
    print("2. Wrap individual tasks with try-except for granular handling")
    print("3. Implement retry logic for transient failures")
    print("4. Log errors for debugging and monitoring")
    print("5. Return structured results (success/failed/error)")
    print("6. Don't let one failure stop the entire pipeline")
    print("=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(main())
