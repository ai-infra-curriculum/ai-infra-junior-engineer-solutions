"""
Tests for async programming basics.
"""

import pytest
import asyncio
import time
from typing import List, Dict


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Basic Async/Await Tests
# ============================================================================

@pytest.mark.asyncio
async def test_simple_async_function():
    """Test basic async function execution."""
    async def simple_task() -> str:
        await asyncio.sleep(0.01)
        return "completed"

    result = await simple_task()
    assert result == "completed"


@pytest.mark.asyncio
async def test_async_with_parameters():
    """Test async function with parameters."""
    async def multiply(a: int, b: int) -> int:
        await asyncio.sleep(0.01)
        return a * b

    result = await multiply(5, 7)
    assert result == 35


@pytest.mark.asyncio
async def test_async_return_types():
    """Test async functions with different return types."""
    async def return_dict() -> Dict:
        await asyncio.sleep(0.01)
        return {"status": "success", "value": 42}

    async def return_list() -> List:
        await asyncio.sleep(0.01)
        return [1, 2, 3, 4, 5]

    dict_result = await return_dict()
    list_result = await return_list()

    assert dict_result["value"] == 42
    assert len(list_result) == 5


# ============================================================================
# asyncio.gather() Tests
# ============================================================================

@pytest.mark.asyncio
async def test_gather_multiple_tasks():
    """Test gathering multiple async tasks."""
    async def task(value: int) -> int:
        await asyncio.sleep(0.01)
        return value * 2

    results = await asyncio.gather(
        task(1),
        task(2),
        task(3)
    )

    assert results == [2, 4, 6]


@pytest.mark.asyncio
async def test_gather_with_list_comprehension():
    """Test gather with list comprehension."""
    async def square(n: int) -> int:
        await asyncio.sleep(0.01)
        return n ** 2

    tasks = [square(i) for i in range(5)]
    results = await asyncio.gather(*tasks)

    assert results == [0, 1, 4, 9, 16]


@pytest.mark.asyncio
async def test_gather_preserves_order():
    """Test that gather preserves task order."""
    async def delayed_task(value: int, delay: float) -> int:
        await asyncio.sleep(delay)
        return value

    # Tasks complete in different order, but results should be ordered
    results = await asyncio.gather(
        delayed_task(1, 0.03),
        delayed_task(2, 0.01),
        delayed_task(3, 0.02)
    )

    assert results == [1, 2, 3]  # Ordered by task creation, not completion


# ============================================================================
# Concurrent Execution Tests
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_speedup():
    """Test that concurrent execution is faster than sequential."""
    async def sleep_task(duration: float) -> str:
        await asyncio.sleep(duration)
        return "done"

    # Sequential
    start = time.time()
    await sleep_task(0.1)
    await sleep_task(0.1)
    await sleep_task(0.1)
    sequential_time = time.time() - start

    # Concurrent
    start = time.time()
    await asyncio.gather(
        sleep_task(0.1),
        sleep_task(0.1),
        sleep_task(0.1)
    )
    concurrent_time = time.time() - start

    # Concurrent should be at least 2x faster
    assert sequential_time / concurrent_time >= 2.0


@pytest.mark.asyncio
async def test_many_concurrent_tasks():
    """Test handling many concurrent tasks."""
    async def small_task(value: int) -> int:
        await asyncio.sleep(0.001)
        return value + 1

    num_tasks = 100
    tasks = [small_task(i) for i in range(num_tasks)]
    results = await asyncio.gather(*tasks)

    assert len(results) == num_tasks
    assert results[0] == 1
    assert results[-1] == num_tasks


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_gather_exception_without_return_exceptions():
    """Test that gather stops on first exception by default."""
    async def failing_task():
        await asyncio.sleep(0.01)
        raise ValueError("Task failed")

    async def success_task():
        await asyncio.sleep(0.01)
        return "success"

    with pytest.raises(ValueError, match="Task failed"):
        await asyncio.gather(
            success_task(),
            failing_task(),
            success_task()
        )


@pytest.mark.asyncio
async def test_gather_with_return_exceptions():
    """Test gather with return_exceptions=True."""
    async def maybe_fail(should_fail: bool):
        await asyncio.sleep(0.01)
        if should_fail:
            raise ValueError("Failed")
        return "success"

    results = await asyncio.gather(
        maybe_fail(False),
        maybe_fail(True),
        maybe_fail(False),
        return_exceptions=True
    )

    assert len(results) == 3
    assert results[0] == "success"
    assert isinstance(results[1], ValueError)
    assert results[2] == "success"


@pytest.mark.asyncio
async def test_handle_exceptions_per_task():
    """Test handling exceptions individually per task."""
    async def safe_task(value: int, should_fail: bool) -> Dict:
        try:
            await asyncio.sleep(0.01)
            if should_fail:
                raise ValueError("Task error")
            return {"value": value, "success": True}
        except ValueError as e:
            return {"value": value, "success": False, "error": str(e)}

    results = await asyncio.gather(
        safe_task(1, False),
        safe_task(2, True),
        safe_task(3, False)
    )

    assert results[0]["success"] is True
    assert results[1]["success"] is False
    assert results[2]["success"] is True


# ============================================================================
# asyncio.create_task() Tests
# ============================================================================

@pytest.mark.asyncio
async def test_create_task():
    """Test creating tasks with create_task()."""
    async def background_work(value: int) -> int:
        await asyncio.sleep(0.05)
        return value * 10

    # Create task (starts running immediately)
    task = asyncio.create_task(background_work(5))

    # Do other work
    await asyncio.sleep(0.01)

    # Wait for task to complete
    result = await task

    assert result == 50


@pytest.mark.asyncio
async def test_multiple_create_tasks():
    """Test creating multiple tasks."""
    async def worker(worker_id: int) -> Dict:
        await asyncio.sleep(0.01)
        return {"worker_id": worker_id, "status": "done"}

    tasks = [asyncio.create_task(worker(i)) for i in range(5)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 5
    assert all(r["status"] == "done" for r in results)


# ============================================================================
# Timeout Tests
# ============================================================================

@pytest.mark.asyncio
async def test_wait_for_timeout():
    """Test wait_for with timeout."""
    async def slow_task():
        await asyncio.sleep(1.0)
        return "completed"

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_task(), timeout=0.1)


@pytest.mark.asyncio
async def test_wait_for_completes_in_time():
    """Test wait_for when task completes in time."""
    async def fast_task():
        await asyncio.sleep(0.05)
        return "completed"

    result = await asyncio.wait_for(fast_task(), timeout=1.0)
    assert result == "completed"


# ============================================================================
# Task Cancellation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_task_cancellation():
    """Test cancelling a task."""
    async def long_task():
        try:
            await asyncio.sleep(10.0)
            return "completed"
        except asyncio.CancelledError:
            return "cancelled"

    task = asyncio.create_task(long_task())
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancel_multiple_tasks():
    """Test cancelling multiple tasks."""
    async def worker():
        await asyncio.sleep(5.0)

    tasks = [asyncio.create_task(worker()) for _ in range(3)]

    # Cancel all tasks
    for task in tasks:
        task.cancel()

    # Gather with return_exceptions to collect cancellations
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(r, asyncio.CancelledError) for r in results)


# ============================================================================
# asyncio.wait() Tests
# ============================================================================

@pytest.mark.asyncio
async def test_wait_first_completed():
    """Test wait with FIRST_COMPLETED."""
    async def racer(name: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return name

    tasks = {
        asyncio.create_task(racer("fast", 0.05)),
        asyncio.create_task(racer("slow", 0.5))
    }

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    assert len(done) == 1
    assert len(pending) == 1

    winner = list(done)[0].result()
    assert winner == "fast"

    # Cancel pending
    for task in pending:
        task.cancel()


@pytest.mark.asyncio
async def test_wait_all_completed():
    """Test wait with ALL_COMPLETED (default)."""
    async def task(value: int) -> int:
        await asyncio.sleep(0.01)
        return value

    tasks = {asyncio.create_task(task(i)) for i in range(3)}

    done, pending = await asyncio.wait(tasks)

    assert len(done) == 3
    assert len(pending) == 0


# ============================================================================
# Async Context Manager Tests
# ============================================================================

@pytest.mark.asyncio
async def test_async_context_manager():
    """Test async context manager."""
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
        assert resource.opened is True
        assert resource.closed is False

    assert resource.closed is True


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_async_performance_benefit():
    """Test that async provides performance benefit for I/O operations."""
    num_operations = 20
    operation_time = 0.01

    # Sequential
    start = time.time()
    for _ in range(num_operations):
        await asyncio.sleep(operation_time)
    sequential_time = time.time() - start

    # Concurrent
    start = time.time()
    await asyncio.gather(*[asyncio.sleep(operation_time) for _ in range(num_operations)])
    concurrent_time = time.time() - start

    # Concurrent should be significantly faster
    speedup = sequential_time / concurrent_time
    assert speedup >= 10.0  # Should be ~20x but account for overhead
