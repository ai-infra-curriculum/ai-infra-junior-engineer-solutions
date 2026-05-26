# Exercise 06: Async Programming for ML — Solution

## What the exercise asked for

Use Python's `asyncio` for concurrent I/O-bound operations in
ML pipelines: async file I/O, concurrent API calls, async data
loading, error handling.

## Key patterns

See [`async_basics.py`](./async_basics.py) for a working
example covering all four patterns:

1. Concurrent API calls with `asyncio.gather`.
2. Async file I/O with `aiofiles`.
3. Async batched data loading.
4. Error handling with `asyncio.gather(..., return_exceptions=True)`.

## When to use async vs. threads vs. processes

| Workload | Best fit | Why |
|---|---|---|
| Many concurrent HTTP requests (e.g., feature service lookups) | **async** | I/O-bound; async is most efficient at high concurrency. |
| Disk reads of many files | **async** | Same logic. |
| CPU-heavy computation (NumPy / PyTorch ops) | **multiprocessing** | GIL means threads don't parallelize CPU work. |
| Mixed I/O + light CPU (a typical ML serving request) | **async** | Threading works too but async scales better. |
| GUI / blocking C libraries | **threads** | Some libs don't release the GIL. |

For ML platforms specifically:

- **Async**: ideal for the serving API frontend (FastAPI is
  async-native), feature-store lookups, calling external APIs
  (OpenAI, etc.), and orchestrating training-pipeline steps.
- **Multiprocessing**: the right tool for parallel data
  preprocessing across CPU cores.
- **Threads**: rarely the right answer in modern Python ML
  code; the cases where it wins are narrow.

## Common mistakes

- Awaiting blocking calls (`time.sleep(1)` inside an async
  function blocks the entire event loop; use
  `await asyncio.sleep(1)`).
- Calling sync libraries from async code (e.g., `requests`
  instead of `aiohttp` / `httpx`).
- Not bounding concurrency — 10,000 concurrent connections
  takes the API down. Use `asyncio.Semaphore` to cap.
- Mixing event loops in tests; use `pytest-asyncio` properly.

## Bounded concurrency pattern

```python
import asyncio

async def fetch_with_limit(items, max_concurrent=10):
    sem = asyncio.Semaphore(max_concurrent)
    async def bounded(item):
        async with sem:
            return await fetch(item)
    return await asyncio.gather(*(bounded(i) for i in items))
```

## Cross-references

- Exercise prompt:
  `ai-infra-junior-engineer-learning/lessons/mod-001-python-fundamentals/exercises/exercise-06-async-programming.md`
- The next module's API exercise (mod-007 ex-06) uses async
  patterns in a production ML API context.
