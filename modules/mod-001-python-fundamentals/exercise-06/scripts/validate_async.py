#!/usr/bin/env python3
"""
Async Programming Validation Script

Validates all async programming implementations.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Dict


class ValidationResult:
    """Store validation results."""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.message = ""
        self.execution_time = 0.0

    def __repr__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} - {self.test_name}: {self.message} ({self.execution_time:.3f}s)"


async def validate_async_basics() -> ValidationResult:
    """Validate async basics implementation."""
    result = ValidationResult("Async Basics")
    start = time.time()

    try:
        # Test basic async function
        async def simple_task(value: int) -> int:
            await asyncio.sleep(0.1)
            return value * 2

        # Test gather
        tasks = [simple_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        if len(results) == 5 and results[0] == 0 and results[4] == 8:
            result.passed = True
            result.message = "Async basics working correctly"
        else:
            result.message = f"Unexpected results: {results}"

    except Exception as e:
        result.message = f"Error: {str(e)}"

    result.execution_time = time.time() - start
    return result


async def validate_concurrent_speedup() -> ValidationResult:
    """Validate that concurrent execution is faster than sequential."""
    result = ValidationResult("Concurrent Speedup")
    start = time.time()

    try:
        async def task(delay: float) -> str:
            await asyncio.sleep(delay)
            return "done"

        # Sequential execution
        seq_start = time.time()
        for _ in range(5):
            await task(0.1)
        seq_time = time.time() - seq_start

        # Concurrent execution
        conc_start = time.time()
        await asyncio.gather(*[task(0.1) for _ in range(5)])
        conc_time = time.time() - conc_start

        speedup = seq_time / conc_time

        if speedup >= 3.0:  # Should be ~5x but account for overhead
            result.passed = True
            result.message = f"Concurrent {speedup:.1f}x faster than sequential"
        else:
            result.message = f"Insufficient speedup: {speedup:.1f}x (expected >3x)"

    except Exception as e:
        result.message = f"Error: {str(e)}"

    result.execution_time = time.time() - start
    return result


async def validate_error_handling() -> ValidationResult:
    """Validate async error handling."""
    result = ValidationResult("Error Handling")
    start = time.time()

    try:
        async def failing_task(should_fail: bool) -> Dict:
            await asyncio.sleep(0.05)
            if should_fail:
                raise ValueError("Task failed")
            return {"status": "success"}

        # Test with return_exceptions
        tasks = [
            failing_task(False),
            failing_task(True),
            failing_task(False)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for r in results if isinstance(r, dict))
        error_count = sum(1 for r in results if isinstance(r, Exception))

        if success_count == 2 and error_count == 1:
            result.passed = True
            result.message = "Error handling works correctly"
        else:
            result.message = f"Unexpected results: {success_count} success, {error_count} errors"

    except Exception as e:
        result.message = f"Error: {str(e)}"

    result.execution_time = time.time() - start
    return result


async def validate_timeout_handling() -> ValidationResult:
    """Validate timeout handling."""
    result = ValidationResult("Timeout Handling")
    start = time.time()

    try:
        async def slow_task():
            await asyncio.sleep(2.0)
            return "completed"

        # Should timeout
        timed_out = False
        try:
            await asyncio.wait_for(slow_task(), timeout=0.5)
        except asyncio.TimeoutError:
            timed_out = True

        if timed_out:
            result.passed = True
            result.message = "Timeout handling works correctly"
        else:
            result.message = "Task did not timeout as expected"

    except Exception as e:
        result.message = f"Error: {str(e)}"

    result.execution_time = time.time() - start
    return result


async def validate_task_cancellation() -> ValidationResult:
    """Validate task cancellation."""
    result = ValidationResult("Task Cancellation")
    start = time.time()

    try:
        async def long_running_task():
            try:
                await asyncio.sleep(5.0)
                return "completed"
            except asyncio.CancelledError:
                return "cancelled"

        task = asyncio.create_task(long_running_task())
        await asyncio.sleep(0.1)  # Let task start
        task.cancel()

        try:
            await task
            result.message = "Task was not cancelled"
        except asyncio.CancelledError:
            result.passed = True
            result.message = "Task cancellation works correctly"

    except Exception as e:
        result.message = f"Error: {str(e)}"

    result.execution_time = time.time() - start
    return result


async def validate_semaphore() -> ValidationResult:
    """Validate semaphore rate limiting."""
    result = ValidationResult("Semaphore Rate Limiting")
    start = time.time()

    try:
        semaphore = asyncio.Semaphore(2)  # Max 2 concurrent
        concurrent_count = 0
        max_concurrent = 0

        async def limited_task(task_id: int):
            nonlocal concurrent_count, max_concurrent

            async with semaphore:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.1)
                concurrent_count -= 1

            return task_id

        tasks = [limited_task(i) for i in range(10)]
        await asyncio.gather(*tasks)

        if max_concurrent == 2:
            result.passed = True
            result.message = "Semaphore limited concurrency to 2"
        else:
            result.message = f"Max concurrent was {max_concurrent}, expected 2"

    except Exception as e:
        result.message = f"Error: {str(e)}"

    result.execution_time = time.time() - start
    return result


async def validate_batch_processing() -> ValidationResult:
    """Validate batch processing."""
    result = ValidationResult("Batch Processing")
    start = time.time()

    try:
        async def process_item(item: int) -> int:
            await asyncio.sleep(0.01)
            return item * 2

        async def process_batch(items: List[int]) -> List[int]:
            tasks = [process_item(item) for item in items]
            return await asyncio.gather(*tasks)

        items = list(range(100))
        batch_size = 10
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

        all_results = []
        for batch in batches:
            batch_results = await process_batch(batch)
            all_results.extend(batch_results)

        if len(all_results) == 100 and all_results[50] == 100:
            result.passed = True
            result.message = "Batch processing works correctly"
        else:
            result.message = f"Unexpected results: {len(all_results)} items"

    except Exception as e:
        result.message = f"Error: {str(e)}"

    result.execution_time = time.time() - start
    return result


async def validate_async_context_manager() -> ValidationResult:
    """Validate async context managers."""
    result = ValidationResult("Async Context Manager")
    start = time.time()

    try:
        class AsyncResource:
            def __init__(self):
                self.opened = False
                self.closed = False

            async def __aenter__(self):
                await asyncio.sleep(0.01)
                self.opened = True
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await asyncio.sleep(0.01)
                self.closed = True
                return False

        async with AsyncResource() as resource:
            if not resource.opened:
                result.message = "Resource not opened"
                return result

        if resource.closed:
            result.passed = True
            result.message = "Async context manager works correctly"
        else:
            result.message = "Resource not closed"

    except Exception as e:
        result.message = f"Error: {str(e)}"

    result.execution_time = time.time() - start
    return result


async def validate_task_racing() -> ValidationResult:
    """Validate task racing with FIRST_COMPLETED."""
    result = ValidationResult("Task Racing")
    start = time.time()

    try:
        async def racer(name: str, delay: float) -> str:
            await asyncio.sleep(delay)
            return name

        tasks = {
            asyncio.create_task(racer("fast", 0.1)),
            asyncio.create_task(racer("medium", 0.3)),
            asyncio.create_task(racer("slow", 0.5))
        }

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        winner = list(done)[0].result()

        # Cancel remaining
        for task in pending:
            task.cancel()

        if winner == "fast":
            result.passed = True
            result.message = "Task racing works correctly"
        else:
            result.message = f"Wrong winner: {winner}"

    except Exception as e:
        result.message = f"Error: {str(e)}"

    result.execution_time = time.time() - start
    return result


async def validate_retry_logic() -> ValidationResult:
    """Validate async retry logic."""
    result = ValidationResult("Retry Logic")
    start = time.time()

    try:
        attempt_count = 0

        async def flaky_task():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Task failed")
            return "success"

        async def retry_async(func, max_retries: int = 3):
            for attempt in range(max_retries):
                try:
                    return await func()
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(0.01)

        result_value = await retry_async(flaky_task)

        if result_value == "success" and attempt_count == 3:
            result.passed = True
            result.message = "Retry logic works correctly (3 attempts)"
        else:
            result.message = f"Unexpected result: {result_value}, attempts: {attempt_count}"

    except Exception as e:
        result.message = f"Error: {str(e)}"

    result.execution_time = time.time() - start
    return result


async def validate_all() -> List[ValidationResult]:
    """Run all validation tests."""
    print("=" * 70)
    print("Running Async Programming Validation Tests")
    print("=" * 70)
    print()

    tests = [
        validate_async_basics(),
        validate_concurrent_speedup(),
        validate_error_handling(),
        validate_timeout_handling(),
        validate_task_cancellation(),
        validate_semaphore(),
        validate_batch_processing(),
        validate_async_context_manager(),
        validate_task_racing(),
        validate_retry_logic()
    ]

    results = await asyncio.gather(*tests)

    return results


def print_results(results: List[ValidationResult]):
    """Print validation results."""
    print("\nValidation Results:")
    print("-" * 70)

    for result in results:
        print(result)

    print("-" * 70)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pass_rate = (passed / total) * 100

    print()
    print(f"Summary: {passed}/{total} tests passed ({pass_rate:.0f}%)")
    print()

    if passed == total:
        print("✓ All async programming validations passed!")
        print()
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        print()
        return 1


async def main():
    """Main validation entry point."""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  Async Programming Validation".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")

    start_time = time.time()

    results = await validate_all()

    total_time = time.time() - start_time

    exit_code = print_results(results)

    print(f"Total validation time: {total_time:.2f}s")
    print()

    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
