# Implementation Guide - Exercise 06: Async Programming

Step-by-step guide for implementing asynchronous programming for ML workflows.

## Prerequisites

- Python 3.11+ installed
- Completed Exercises 01-05
- Install required packages: `pip install aiofiles aiohttp pytest-asyncio`
- Understanding of synchronous vs asynchronous execution

## Time Estimate

90-120 minutes total

## Implementation Steps

### Part 1: Async Basics (25 minutes)

**Step 1: Create async_basics.py**
- Implement async functions with `async def`
- Use `await` to call async functions
- Demonstrate `asyncio.sleep()` vs `time.sleep()`
- Compare sequential vs concurrent execution
- Use `asyncio.gather()` for concurrent tasks
- Run: `python scripts/async_basics.py`

**Key concepts:**
- `async def` declares coroutine function
- `await` suspends execution until result ready
- `asyncio.run()` starts event loop
- `asyncio.gather()` runs tasks concurrently
- Event loop schedules and executes coroutines

### Part 2: Multiple Concurrent Tasks (20 minutes)

**Step 2: Create async_multiple.py**
- Implement `process_sample()` for single item
- Implement `process_batch_async()` with gather
- Implement `download_multiple_models()` with dict results
- Demonstrate variable processing times
- Run: `python scripts/async_multiple.py`

**Key concepts:**
- List comprehension to create tasks: `[func(x) for x in items]`
- Unpacking tasks: `asyncio.gather(*tasks)`
- Collecting results from multiple tasks
- Concurrent execution speedup

### Part 3: Async File I/O (25 minutes)

**Step 3: Create async_file_io.py**
- Import `aiofiles` for async file operations
- Implement `read_file_async()` with aiofiles.open
- Implement `write_file_async()` with aiofiles.open
- Implement `read_multiple_files()` with error handling
- Implement `process_csv_async()` for CSV files
- Implement `save_predictions_async()` for output
- Run: `python scripts/async_file_io.py`

**Key concepts:**
- `async with aiofiles.open()` for async file I/O
- `await f.read()` and `await f.write()`
- `return_exceptions=True` to handle errors gracefully
- Processing multiple files concurrently

### Part 4: Async API Calls (20 minutes)

**Step 4: Create async_api_calls.py**
- Import `aiohttp` for async HTTP
- Implement `fetch_model_metadata()` with session
- Implement `fetch_multiple_models()` with gather
- Implement `batch_inference_api()` with batching
- Add timeout handling
- Add error handling per request
- Run: `python scripts/async_api_calls.py`

**Key concepts:**
- `async with aiohttp.ClientSession()` for connection pooling
- `async with session.get(url)` for requests
- `await response.json()` to parse response
- `aiohttp.ClientTimeout()` for timeouts
- Reusing session across requests

### Part 5: Async ML Pipeline (30 minutes)

**Step 5: Create async_ml_pipeline.py**
- Create `Sample` dataclass
- Implement `AsyncMLPipeline` class
  - `load_data()` - async data loading
  - `preprocess_sample()` - single sample preprocessing
  - `preprocess_batch()` - concurrent batch preprocessing
  - `predict_sample()` - single sample inference
  - `predict_batch()` - concurrent batch inference
  - `run_pipeline()` - orchestrate complete pipeline
- Add timing metrics
- Calculate throughput
- Run: `python scripts/async_ml_pipeline.py`

**Key concepts:**
- Dataclass for structured data
- Pipeline stages with async operations
- Batch processing for efficiency
- Performance metrics collection
- Real-world async ML workflow

### Part 6: Error Handling (20 minutes)

**Step 6: Create async_error_handling.py**
- Implement `risky_operation()` that may fail
- Implement `safe_risky_operation()` with try-except
- Implement `run_tasks_with_error_handling()`
- Implement `retry_async()` decorator pattern
- Collect and categorize errors
- Run: `python scripts/async_error_handling.py`

**Key concepts:**
- Try-except in async functions
- `return_exceptions=True` in gather()
- Categorizing results (success/failed/error)
- Async retry logic
- Logging async errors

### Part 7: Concurrency Comparison (25 minutes)

**Step 7: Create concurrency_comparison.py**
- Implement `cpu_bound_task()` for computation
- Implement `io_bound_task()` (synchronous)
- Implement `io_bound_task_async()` (async)
- Implement `sync_io_tasks()` baseline
- Implement `async_io_tasks()` with gather
- Implement `threaded_io_tasks()` with threading
- Compare execution times
- Provide usage guidelines
- Run: `python scripts/concurrency_comparison.py`

**Key concepts:**
- Async best for I/O-bound tasks
- Threading good for I/O with sync libraries
- Multiprocessing best for CPU-bound tasks
- Measuring and comparing performance
- Decision matrix for concurrency

### Part 8: Validation (15 minutes)

**Step 8: Create validate_async.py**
- Test async function execution
- Test asyncio.gather() functionality
- Test error handling in async code
- Test concurrent task execution
- Verify speedup from concurrency
- Run: `python scripts/validate_async.py`

### Part 9: Testing (20 minutes)

**Step 9: Create pytest tests**
- Install pytest-asyncio: `pip install pytest-asyncio`
- Create `tests/test_async_basics.py`
  - Test async function execution
  - Test concurrent vs sequential timing
  - Test asyncio.gather()
- Create `tests/test_async_pipeline.py`
  - Test pipeline stages
  - Test complete pipeline execution
  - Test error handling

**Run tests:**
```bash
pytest tests/ -v
pytest tests/ --cov=scripts --cov-report=term-missing
```

## Quick Validation

```bash
# Install dependencies
pip install aiofiles aiohttp pytest-asyncio

# Run all implementation scripts
python scripts/async_basics.py
python scripts/async_multiple.py
python scripts/async_file_io.py
python scripts/async_api_calls.py
python scripts/async_ml_pipeline.py
python scripts/async_error_handling.py
python scripts/concurrency_comparison.py

# Validate everything
python scripts/validate_async.py

# Run pytest
pytest tests/ -v
```

## Key Concepts Checklist

- [ ] `async def` for coroutine functions
- [ ] `await` to call async functions
- [ ] `asyncio.run()` to start event loop
- [ ] `asyncio.gather()` for concurrent tasks
- [ ] `asyncio.sleep()` for non-blocking delays
- [ ] `async with` for async context managers
- [ ] `aiofiles` for async file I/O
- [ ] `aiohttp` for async HTTP requests
- [ ] Connection pooling with ClientSession
- [ ] Timeout handling
- [ ] Error handling with return_exceptions
- [ ] Retry logic for transient failures
- [ ] Choosing between async/threading/multiprocessing
- [ ] Performance measurement and optimization

## Common Issues

**Issue:** RuntimeError: Event loop is closed
**Solution:** Use `asyncio.run()` or manage loop properly:
```python
# ✓ GOOD
asyncio.run(main())

# ✗ BAD
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
asyncio.run(another_func())  # Error: loop is closed
```

**Issue:** Coroutine never awaited warning
**Solution:** Always await async functions:
```python
# ✗ BAD
result = async_function()  # Returns coroutine, doesn't execute

# ✓ GOOD
result = await async_function()  # Executes and gets result
```

**Issue:** Blocking the event loop
**Solution:** Use async alternatives or run_in_executor:
```python
# ✗ BAD
async def bad():
    time.sleep(1)  # Blocks entire event loop!

# ✓ GOOD
async def good():
    await asyncio.sleep(1)  # Non-blocking

# ✓ GOOD for unavoidable blocking code
async def with_blocking():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, blocking_function)
```

**Issue:** Too many concurrent connections
**Solution:** Use semaphore to limit concurrency:
```python
semaphore = asyncio.Semaphore(100)

async def limited_fetch(url):
    async with semaphore:
        return await fetch(url)
```

**Issue:** Mixing async and sync code
**Solution:** Keep them separate or use run_in_executor:
```python
# ✗ BAD
async def process():
    data = sync_load()  # Blocks!
    return await async_process(data)

# ✓ GOOD
async def process():
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, sync_load)
    return await async_process(data)
```

## Best Practices

1. **Always use async with for resources:**
   ```python
   async with aiohttp.ClientSession() as session:
       async with session.get(url) as response:
           data = await response.json()
   ```

2. **Set timeouts for all operations:**
   ```python
   try:
       result = await asyncio.wait_for(operation(), timeout=10.0)
   except asyncio.TimeoutError:
       handle_timeout()
   ```

3. **Handle errors per task:**
   ```python
   results = await asyncio.gather(*tasks, return_exceptions=True)
   for result in results:
       if isinstance(result, Exception):
           log_error(result)
   ```

4. **Limit concurrency with semaphores:**
   ```python
   semaphore = asyncio.Semaphore(50)
   async with semaphore:
       await expensive_operation()
   ```

5. **Batch operations for efficiency:**
   ```python
   batch_size = 100
   batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
   for batch in batches:
       await process_batch(batch)
   ```

## Performance Tips

1. **I/O-bound tasks:** Use async (10-20x speedup typical)
2. **CPU-bound tasks:** Use multiprocessing (not async)
3. **Batch size:** 50-100 concurrent tasks is often optimal
4. **Connection pooling:** Reuse sessions/connections
5. **Timeouts:** Always set reasonable timeouts
6. **Memory:** Monitor memory with many concurrent tasks

## When to Use Async

**Use async for:**
- Network requests (API calls, web scraping)
- File I/O (reading/writing many files)
- Database queries (with async drivers)
- Concurrent data processing
- Real-time data streams

**Don't use async for:**
- CPU-bound computation (use multiprocessing)
- Single sequential operation
- Already fast operations (< 1ms)
- Code that's simpler synchronously

## Next Steps

After completing this exercise:

1. **Exercise 07: Testing** - Test async code with pytest-asyncio
2. **Apply to projects:**
   - Build async data ingestion pipeline
   - Create concurrent model serving system
   - Implement async experiment tracking
3. **Advanced topics:**
   - AsyncIO streams
   - Async generators
   - Async context managers
   - Custom event loop policies

## Resources

- [Python Asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Real Python Async Tutorial](https://realpython.com/async-io-python/)
- [aiohttp Documentation](https://docs.aiohttp.org/)
- [aiofiles Documentation](https://github.com/Tinche/aiofiles)
- [AsyncIO Cheatsheet](https://cheat.readthedocs.io/en/latest/python/asyncio.html)

---

**Last Updated:** 2025-10-30
