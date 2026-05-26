"""Reference solution: async patterns for ML pipelines.

Demonstrates:
  - asyncio.gather for concurrent operations
  - async file I/O via aiofiles
  - error handling that doesn't tank the whole batch
  - bounded concurrency via Semaphore

Run:
    pip install aiohttp aiofiles
    python async_basics.py
"""

from __future__ import annotations

import asyncio
from typing import Any


async def fake_feature_lookup(user_id: str) -> dict[str, Any]:
    """Simulates a feature-store call (sleeps to mimic I/O)."""
    await asyncio.sleep(0.1)
    if user_id == "bad":
        raise ValueError(f"feature lookup failed for {user_id}")
    return {"user_id": user_id, "embedding_dim": 128}


async def concurrent_feature_lookups(user_ids: list[str]) -> list[dict[str, Any]]:
    """Concurrent feature lookups via asyncio.gather.

    Without `return_exceptions=True`, one failure cancels the
    whole batch. With it, you get exceptions in-place and can
    decide per item.
    """
    results = await asyncio.gather(
        *(fake_feature_lookup(uid) for uid in user_ids),
        return_exceptions=True,
    )
    successes = []
    for uid, result in zip(user_ids, results):
        if isinstance(result, Exception):
            print(f"  WARN: lookup failed for {uid}: {result}")
            continue
        successes.append(result)
    return successes


async def bounded_concurrent_lookups(
    user_ids: list[str], max_concurrent: int = 10
) -> list[dict[str, Any]]:
    """Cap the number of in-flight lookups to avoid overwhelming
    downstream services."""
    sem = asyncio.Semaphore(max_concurrent)

    async def bounded(uid: str) -> dict[str, Any] | Exception:
        async with sem:
            try:
                return await fake_feature_lookup(uid)
            except Exception as e:
                return e

    results = await asyncio.gather(*(bounded(uid) for uid in user_ids))
    return [r for r in results if not isinstance(r, Exception)]


async def main() -> None:
    user_ids = [f"user-{i:03d}" for i in range(20)] + ["bad"]

    print("Pattern 1: concurrent feature lookups (one failure tolerated)")
    results = await concurrent_feature_lookups(user_ids)
    print(f"  Got {len(results)} successful results out of {len(user_ids)} requests")

    print("\nPattern 2: bounded concurrency (max 5 in-flight)")
    results = await bounded_concurrent_lookups(user_ids, max_concurrent=5)
    print(f"  Got {len(results)} successful results out of {len(user_ids)} requests")


if __name__ == "__main__":
    asyncio.run(main())
