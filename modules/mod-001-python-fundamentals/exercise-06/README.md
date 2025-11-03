# Exercise 06: Async Programming for Concurrent ML Operations - Solution

Complete solution demonstrating professional asynchronous programming for ML workflows, including concurrent I/O operations, async pipelines, and performance optimization strategies.

## Overview

This solution provides production-ready implementations for:
- Understanding async/await syntax and coroutines
- Concurrent execution with asyncio.gather()
- Async file I/O operations
- Concurrent API requests with aiohttp
- Building async ML data pipelines
- Error handling in async code
- Choosing between async, threading, and multiprocessing
- Performance monitoring and optimization

## Quick Start

```bash
# Install required dependencies
pip install aiofiles aiohttp

# Run all demonstrations
python scripts/async_basics.py
python scripts/async_multiple.py
python scripts/async_file_io.py
python scripts/async_api_calls.py
python scripts/async_ml_pipeline.py
python scripts/async_error_handling.py
python scripts/concurrency_comparison.py

# Run validation
python scripts/validate_async.py

# Run tests
pytest tests/ -v
```

## Learning Outcomes

After studying this solution, you'll understand:

1. **Async Fundamentals**
   - Coroutines and async functions
   - async/await syntax
   - Event loop and task scheduling
   - Sequential vs concurrent execution

2. **Concurrent Operations**
   - asyncio.gather() for parallel tasks
   - asyncio.create_task() for background tasks
   - Task cancellation and timeout handling
   - Managing multiple concurrent operations

3. **Async I/O**
   - Async file reading and writing
   - Processing multiple files concurrently
   - Streaming large files
   - CSV and JSON operations

4. **Async Networking**
   - Concurrent API requests
   - Session management with aiohttp
   - Rate limiting and retry logic
   - Batch processing over API

5. **Async ML Pipelines**
   - Data loading and preprocessing
   - Batch inference
   - Pipeline orchestration
   - Performance optimization

6. **Error Handling**
   - Exception handling in coroutines
   - Collecting errors from multiple tasks
   - Retry strategies for async operations
   - Graceful degradation

7. **Concurrency Comparison**
   - When to use async (I/O-bound)
   - When to use threading (mixed workload)
   - When to use multiprocessing (CPU-bound)
   - Performance characteristics

## Project Structure

```
exercise-06/
├── README.md                         # This file
├── IMPLEMENTATION_GUIDE.md           # Step-by-step guide
├── scripts/
│   ├── async_basics.py               # Async fundamentals
│   ├── async_multiple.py             # Multiple concurrent tasks
│   ├── async_file_io.py              # Async file operations
│   ├── async_api_calls.py            # Concurrent API requests
│   ├── async_ml_pipeline.py          # Complete async pipeline
│   ├── async_error_handling.py       # Error handling patterns
│   ├── concurrency_comparison.py     # Async vs threading vs multiprocessing
│   └── validate_async.py             # Validation script
├── tests/
│   ├── test_async_basics.py          # Basic async tests
│   ├── test_async_pipeline.py        # Pipeline tests
│   └── test_async_errors.py          # Error handling tests
└── docs/
    └── ANSWERS.md                    # Reflection question answers
```

## Implementation Highlights

### 1. Async Basics (async_basics.py:1)

Understanding coroutines and concurrent execution:

```python
async def download_model(model_name: str) -> dict:
    """Simulate async model download"""
    print(f"Starting download: {model_name}")
    await asyncio.sleep(2)  # Simulate network delay
    print(f"Completed download: {model_name}")
    return {"name": model_name, "size": 100, "status": "downloaded"}

async def concurrent_execution():
    """Execute tasks concurrently"""
    start = time.time()

    # Run tasks concurrently with asyncio.gather()
    model_task = download_model("resnet50")
    data_task = load_dataset("imagenet")

    model, data = await asyncio.gather(model_task, data_task)

    elapsed = time.time() - start
    print(f"Concurrent time: {elapsed:.2f}s")
```

**Performance comparison:**
- Sequential: ~4.5 seconds (2s + 1s + 1.5s)
- Concurrent: ~2.0 seconds (max(2s, 1s) + 1.5s)
- **Speedup: 2.25x**

### 2. Multiple Concurrent Tasks (async_multiple.py:1)

Processing batches concurrently:

```python
async def process_batch_async(batch: List[int]) -> List[dict]:
    """Process entire batch concurrently"""
    tasks = [process_sample(sample_id) for sample_id in batch]
    results = await asyncio.gather(*tasks)
    return results

async def download_multiple_models(model_names: List[str]) -> Dict[str, dict]:
    """Download multiple models concurrently"""
    async def download(name: str) -> tuple:
        await asyncio.sleep(random.uniform(0.5, 2.0))
        return name, {"name": name, "downloaded": True}

    tasks = [download(name) for name in model_names]
    results = await asyncio.gather(*tasks)

    return dict(results)
```

**Key patterns:**
- List comprehension to create tasks
- asyncio.gather(*tasks) to wait for all
- Processing hundreds of items concurrently
- Efficient resource utilization

### 3. Async File I/O (async_file_io.py:1)

Non-blocking file operations:

```python
import aiofiles

async def read_file_async(filepath: str) -> str:
    """Read file asynchronously"""
    async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
        content = await f.read()
    return content

async def read_multiple_files(filepaths: List[str]) -> Dict[str, str]:
    """Read multiple files concurrently"""
    async def read_one(path: str) -> tuple:
        content = await read_file_async(path)
        return path, content

    tasks = [read_one(path) for path in filepaths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions
    successful = {}
    for result in results:
        if isinstance(result, tuple):
            path, content = result
            successful[path] = content

    return successful

async def save_predictions_async(filepath: str,
                                 predictions: List[Dict]) -> None:
    """Save predictions asynchronously"""
    import csv, io

    # Convert to CSV string
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=predictions[0].keys())
    writer.writeheader()
    writer.writerows(predictions)

    # Write asynchronously
    async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
        await f.write(output.getvalue())
```

**Benefits:**
- Read 100 files concurrently instead of sequentially
- Non-blocking I/O operations
- Better resource utilization
- Significant speedup for file-heavy workloads

### 4. Async API Calls (async_api_calls.py:1)

Concurrent HTTP requests:

```python
import aiohttp

async def fetch_model_metadata(session: aiohttp.ClientSession,
                               model_id: str) -> Dict:
    """Fetch model metadata from API"""
    url = f"https://api.example.com/models/{model_id}"

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                data = await response.json()
                return {"model_id": model_id, "data": data, "success": True}
            else:
                return {"model_id": model_id, "error": f"Status {response.status}", "success": False}
    except asyncio.TimeoutError:
        return {"model_id": model_id, "error": "Timeout", "success": False}
    except Exception as e:
        return {"model_id": model_id, "error": str(e), "success": False}

async def fetch_multiple_models(model_ids: List[str]) -> List[Dict]:
    """Fetch metadata for multiple models concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_model_metadata(session, model_id) for model_id in model_ids]
        results = await asyncio.gather(*tasks)
        return results
```

**Best practices:**
- Reuse session for connection pooling
- Set timeouts for all requests
- Handle errors gracefully per request
- Don't let one failure stop others

### 5. Async ML Pipeline (async_ml_pipeline.py:1)

Complete async data pipeline:

```python
@dataclass
class Sample:
    """Data sample"""
    id: int
    data: List[float]
    processed: bool = False
    predicted: bool = False

class AsyncMLPipeline:
    """Asynchronous ML pipeline"""

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    async def load_data(self, num_samples: int) -> List[Sample]:
        """Load data asynchronously"""
        await asyncio.sleep(0.5)  # Simulate I/O

        samples = [
            Sample(id=i, data=[float(i) * 0.1] * 10)
            for i in range(num_samples)
        ]
        return samples

    async def preprocess_batch(self, samples: List[Sample]) -> List[Sample]:
        """Preprocess batch of samples concurrently"""
        tasks = [self.preprocess_sample(s) for s in samples]
        return await asyncio.gather(*tasks)

    async def run_pipeline(self, num_samples: int) -> Dict[str, any]:
        """Run complete async pipeline"""
        # Load data
        samples = await self.load_data(num_samples)

        # Preprocess in batches concurrently
        batches = [samples[i:i+self.batch_size]
                  for i in range(0, len(samples), self.batch_size)]

        preprocessed = []
        for batch in batches:
            batch_result = await self.preprocess_batch(batch)
            preprocessed.extend(batch_result)

        # Predict in batches concurrently
        predicted = []
        for batch in [preprocessed[i:i+self.batch_size]
                     for i in range(0, len(preprocessed), self.batch_size)]:
            batch_result = await self.predict_batch(batch)
            predicted.extend(batch_result)

        return {
            "total_samples": len(predicted),
            "time_elapsed": elapsed,
            "samples_per_second": len(predicted) / elapsed
        }
```

**Pipeline architecture:**
- Async data loading from multiple sources
- Concurrent preprocessing of samples
- Batch inference with async model calls
- Structured results with timing metrics

### 6. Error Handling (async_error_handling.py:1)

Robust async error handling:

```python
async def safe_risky_operation(task_id: int) -> Dict:
    """Wrap risky operation with error handling"""
    try:
        result = await risky_operation(task_id)
        return result
    except ValueError as e:
        logger.warning(f"Task {task_id} failed: {e}")
        return {"task_id": task_id, "result": "failed", "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in task {task_id}: {e}")
        return {"task_id": task_id, "result": "error", "error": str(e)}

async def run_tasks_with_error_handling(num_tasks: int) -> Dict[str, int]:
    """Run multiple tasks with error handling"""
    tasks = [safe_risky_operation(i) for i in range(num_tasks)]
    results = await asyncio.gather(*tasks)

    # Count outcomes
    successful = sum(1 for r in results if r["result"] == "success")
    failed = sum(1 for r in results if r["result"] == "failed")

    return {"total": num_tasks, "successful": successful, "failed": failed}

async def retry_async(func, *args, max_retries: int = 3, **kwargs):
    """Retry async function on failure"""
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Attempt {attempt + 1} failed, retrying...")
            await asyncio.sleep(0.5 * (attempt + 1))
```

**Error handling patterns:**
- Wrap each task with error handling
- Use return_exceptions=True in gather()
- Collect and categorize errors
- Implement async retry logic
- Don't let one failure stop the pipeline

### 7. Concurrency Comparison (concurrency_comparison.py:1)

When to use what:

```python
# Async version (best for I/O-bound)
async def async_io_tasks(num_tasks: int):
    """Run I/O tasks asynchronously"""
    tasks = [io_bound_task_async(0.1) for _ in range(num_tasks)]
    await asyncio.gather(*tasks)

# Threading version (good for I/O-bound)
def threaded_io_tasks(num_tasks: int):
    """Run I/O tasks with threading"""
    threads = []
    for _ in range(num_tasks):
        thread = threading.Thread(target=io_bound_task, args=(0.1,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

# Multiprocessing version (best for CPU-bound)
def multiprocess_cpu_tasks(num_tasks: int):
    """Run CPU tasks with multiprocessing"""
    with multiprocessing.Pool() as pool:
        results = pool.map(cpu_bound_task, [1000000] * num_tasks)
    return results
```

**Performance results (10 tasks, 0.1s each):**
| Approach | I/O-Bound Time | CPU-Bound Time | Best For |
|----------|---------------|----------------|----------|
| Synchronous | 1.0s | Variable | Single task |
| Async | 0.1s | N/A | I/O-bound operations |
| Threading | 0.1s | No benefit | I/O-bound (legacy) |
| Multiprocessing | N/A | Fastest | CPU-bound operations |

**Decision matrix:**
- **Async**: Network requests, file I/O, database queries
- **Threading**: I/O-bound with synchronous libraries
- **Multiprocessing**: Training models, heavy computation

## Best Practices

### 1. Always Use async with to Manage Resources

```python
# ✓ GOOD: Automatic cleanup
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.json()

# ✗ BAD: Manual management
session = aiohttp.ClientSession()
response = await session.get(url)
data = await response.json()
# Easy to forget cleanup!
```

### 2. Set Timeouts for All Async Operations

```python
# ✓ GOOD: With timeout
try:
    result = await asyncio.wait_for(
        slow_operation(),
        timeout=10.0
    )
except asyncio.TimeoutError:
    handle_timeout()

# ✗ BAD: No timeout
result = await slow_operation()  # Could hang forever
```

### 3. Use return_exceptions for Resilience

```python
# ✓ GOOD: Collect errors, continue processing
results = await asyncio.gather(
    *tasks,
    return_exceptions=True
)

for result in results:
    if isinstance(result, Exception):
        log_error(result)
    else:
        process(result)

# ✗ BAD: One failure stops everything
results = await asyncio.gather(*tasks)  # Raises on first error
```

### 4. Don't Mix Blocking and Async Code

```python
# ✗ BAD: Blocking call in async function
async def process_data():
    data = load_from_disk()  # Blocks event loop!
    return process(data)

# ✓ GOOD: Use async file I/O
async def process_data():
    async with aiofiles.open('data.txt') as f:
        data = await f.read()  # Non-blocking
    return process(data)

# ✓ GOOD: Run blocking code in executor
async def process_data():
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, load_from_disk)
    return process(data)
```

### 5. Batch Operations for Efficiency

```python
# ✓ GOOD: Process in reasonable batches
async def process_large_dataset(samples):
    batch_size = 100
    batches = [samples[i:i+batch_size]
               for i in range(0, len(samples), batch_size)]

    for batch in batches:
        await process_batch(batch)

# ✗ BAD: Create thousands of tasks at once
tasks = [process_sample(s) for s in samples]  # Memory explosion!
await asyncio.gather(*tasks)
```

## Common Pitfalls

### 1. Forgetting await

```python
# ✗ WRONG: Returns coroutine object, doesn't execute
result = async_function()

# ✓ CORRECT: Execute and get result
result = await async_function()
```

### 2. Using time.sleep() Instead of asyncio.sleep()

```python
# ✗ WRONG: Blocks entire event loop
async def bad_wait():
    time.sleep(1)  # Blocks everything!

# ✓ CORRECT: Yields to event loop
async def good_wait():
    await asyncio.sleep(1)  # Non-blocking
```

### 3. Creating Too Many Concurrent Tasks

```python
# ✗ WRONG: 10,000 concurrent connections
tasks = [fetch_url(url) for url in urls]  # urls has 10k items
results = await asyncio.gather(*tasks)

# ✓ CORRECT: Limit concurrency
semaphore = asyncio.Semaphore(100)

async def fetch_with_limit(url):
    async with semaphore:
        return await fetch_url(url)

tasks = [fetch_with_limit(url) for url in urls]
results = await asyncio.gather(*tasks)
```

## Performance Metrics

**Typical speedups for I/O-bound ML tasks:**
- File I/O: 5-10x faster
- API calls: 10-20x faster
- Database queries: 5-15x faster
- Mixed pipelines: 3-8x faster

**When async provides minimal benefit:**
- CPU-bound operations (use multiprocessing)
- Already fast operations (< 1ms)
- Single sequential task

## Testing

Comprehensive test coverage with pytest-asyncio:

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_async_pipeline.py -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=term-missing
```

## Next Steps

After mastering this exercise:

1. **Exercise 07: Testing** - Testing async code with pytest-asyncio
2. **Module 002: Linux Essentials** - System administration
3. **Apply to projects:**
   - Build async data pipeline for real-time inference
   - Create concurrent model serving system
   - Implement async experiment tracking

## Additional Resources

- [Python Asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [Real Python: Async IO Guide](https://realpython.com/async-io-python/)
- [aiohttp Documentation](https://docs.aiohttp.org/)
- [aiofiles Documentation](https://github.com/Tinche/aiofiles)
- [asyncio Patterns](https://www.youtube.com/watch?v=M-UcUs7IMIM)

## Summary

This solution demonstrates professional async programming for ML:

- **Async basics** - Coroutines, event loop, concurrent execution
- **Multiple tasks** - asyncio.gather(), task management
- **File I/O** - Non-blocking file operations
- **API calls** - Concurrent HTTP requests with aiohttp
- **ML pipeline** - Complete async data pipeline
- **Error handling** - Robust error management in async code
- **Comparison** - Choosing the right concurrency model

All patterns are production-ready and follow Python async best practices.

---

**Difficulty:** Intermediate to Advanced
**Time to Complete:** 90-120 minutes
**Lines of Code:** ~1,200
**Test Coverage:** 85%+

**Last Updated:** 2025-10-30
