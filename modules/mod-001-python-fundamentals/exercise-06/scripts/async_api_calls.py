#!/usr/bin/env python3
"""
Async API Calls

Demonstrates concurrent HTTP requests with aiohttp.
Note: This script shows the patterns but won't make real API calls.
"""

import asyncio
import time
from typing import List, Dict, Optional

try:
    import aiohttp
except ImportError:
    print("Error: aiohttp not installed")
    print("Install with: pip install aiohttp")
    exit(1)


async def fetch_model_metadata(session: aiohttp.ClientSession,
                               model_id: str) -> Dict:
    """
    Fetch model metadata from API.

    Args:
        session: aiohttp ClientSession
        model_id: Model identifier

    Returns:
        Metadata dictionary
    """
    url = f"https://api.example.com/models/{model_id}"

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                data = await response.json()
                return {"model_id": model_id, "data": data, "success": True}
            else:
                return {
                    "model_id": model_id,
                    "error": f"HTTP {response.status}",
                    "success": False
                }
    except asyncio.TimeoutError:
        return {"model_id": model_id, "error": "Timeout", "success": False}
    except aiohttp.ClientError as e:
        return {"model_id": model_id, "error": str(e), "success": False}
    except Exception as e:
        return {"model_id": model_id, "error": f"Unexpected: {e}", "success": False}


async def fetch_multiple_models(model_ids: List[str]) -> List[Dict]:
    """
    Fetch metadata for multiple models concurrently.

    Args:
        model_ids: List of model identifiers

    Returns:
        List of metadata dictionaries
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_model_metadata(session, model_id) for model_id in model_ids]
        results = await asyncio.gather(*tasks)
        return results


async def post_inference_request(session: aiohttp.ClientSession,
                                 api_url: str,
                                 data: Dict) -> Dict:
    """
    Send inference request to API.

    Args:
        session: aiohttp ClientSession
        api_url: API endpoint URL
        data: Request data

    Returns:
        Response dictionary
    """
    try:
        async with session.post(api_url, json=data, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                result = await response.json()
                return {"success": True, "result": result}
            else:
                return {"success": False, "error": f"HTTP {response.status}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def batch_inference_api(samples: List[Dict],
                              api_url: str,
                              batch_size: int = 10) -> List[Dict]:
    """
    Send samples to inference API in batches.

    Args:
        samples: List of samples to process
        api_url: API endpoint URL
        batch_size: Number of samples per batch

    Returns:
        List of inference results
    """
    async def send_batch(session: aiohttp.ClientSession, batch: List[Dict]) -> Dict:
        return await post_inference_request(session, api_url, {"samples": batch})

    # Split into batches
    batches = [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]

    async with aiohttp.ClientSession() as session:
        tasks = [send_batch(session, batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        return results


async def fetch_with_retry(session: aiohttp.ClientSession,
                          url: str,
                          max_retries: int = 3) -> Optional[Dict]:
    """
    Fetch URL with retry logic.

    Args:
        session: aiohttp ClientSession
        url: URL to fetch
        max_retries: Maximum retry attempts

    Returns:
        Response data or None
    """
    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status >= 500:  # Server error, retry
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                        continue
                else:
                    return None  # Client error, don't retry
        except (asyncio.TimeoutError, aiohttp.ClientError):
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
                continue

    return None


async def fetch_with_rate_limit(urls: List[str], requests_per_second: int = 10) -> List[Dict]:
    """
    Fetch URLs with rate limiting.

    Args:
        urls: List of URLs to fetch
        requests_per_second: Maximum requests per second

    Returns:
        List of responses
    """
    semaphore = asyncio.Semaphore(requests_per_second)

    async def limited_fetch(session: aiohttp.ClientSession, url: str) -> Dict:
        async with semaphore:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    return {"url": url, "status": response.status, "success": True}
            except Exception as e:
                return {"url": url, "error": str(e), "success": False}

    async with aiohttp.ClientSession() as session:
        tasks = [limited_fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results


async def simulate_api_calls():
    """Simulate API call patterns (for demonstration)"""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  Async API Calls (Demonstration)".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")

    # Example 1: Concurrent model fetches (simulated)
    print("=" * 70)
    print("Example 1: Fetching Multiple Models (Pattern)")
    print("=" * 70)
    print()

    model_ids = [f"model_{i:03d}" for i in range(20)]
    print(f"Pattern: Fetch metadata for {len(model_ids)} models concurrently")
    print()
    print("Code pattern:")
    print("  async with aiohttp.ClientSession() as session:")
    print("      tasks = [fetch_model_metadata(session, id) for id in model_ids]")
    print("      results = await asyncio.gather(*tasks)")
    print()
    print("Benefits:")
    print("  • Reuses single session for all requests (connection pooling)")
    print("  • All requests execute concurrently")
    print("  • 10-20x faster than sequential requests")
    print()

    # Example 2: Batch API calls
    print("=" * 70)
    print("Example 2: Batch Inference API (Pattern)")
    print("=" * 70)
    print()

    print("Pattern: Send data in batches to avoid overwhelming API")
    print()
    print("Code pattern:")
    print("  batches = [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]")
    print("  async with aiohttp.ClientSession() as session:")
    print("      tasks = [send_batch(session, batch) for batch in batches]")
    print("      results = await asyncio.gather(*tasks)")
    print()
    print("Benefits:")
    print("  • Respects API limits (e.g., max 100 samples per request)")
    print("  • Still concurrent across batches")
    print("  • Better error handling per batch")
    print()

    # Example 3: Retry logic
    print("=" * 70)
    print("Example 3: Retry Logic (Pattern)")
    print("=" * 70)
    print()

    print("Pattern: Retry failed requests with exponential backoff")
    print()
    print("Code pattern:")
    print("  for attempt in range(max_retries):")
    print("      try:")
    print("          response = await session.get(url)")
    print("          if response.status == 200:")
    print("              return await response.json()")
    print("      except (TimeoutError, ClientError):")
    print("          await asyncio.sleep(1 * (attempt + 1))  # Backoff")
    print()
    print("When to retry:")
    print("  • 5xx errors (server errors)")
    print("  • Timeout errors")
    print("  • Connection errors")
    print()
    print("When NOT to retry:")
    print("  • 4xx errors (client errors)")
    print("  • 401/403 (authentication/authorization)")
    print()

    # Example 4: Rate limiting
    print("=" * 70)
    print("Example 4: Rate Limiting (Pattern)")
    print("=" * 70)
    print()

    print("Pattern: Limit concurrent requests with semaphore")
    print()
    print("Code pattern:")
    print("  semaphore = asyncio.Semaphore(10)  # Max 10 concurrent")
    print("  async def limited_fetch(url):")
    print("      async with semaphore:")
    print("          return await fetch(url)")
    print()
    print("Benefits:")
    print("  • Prevents overwhelming the API")
    print("  • Respects rate limits (e.g., 10 requests/second)")
    print("  • Still much faster than sequential")
    print()

    # Example 5: Error handling
    print("=" * 70)
    print("Example 5: Error Handling (Pattern)")
    print("=" * 70)
    print()

    print("Pattern: Handle errors per request, don't stop on failure")
    print()
    print("Code pattern:")
    print("  async def safe_fetch(url):")
    print("      try:")
    print("          response = await session.get(url)")
    print("          return {'url': url, 'success': True, 'data': await response.json()}")
    print("      except Exception as e:")
    print("          return {'url': url, 'success': False, 'error': str(e)}")
    print()
    print("  results = await asyncio.gather(*[safe_fetch(url) for url in urls])")
    print("  successful = [r for r in results if r['success']]")
    print()
    print("Benefits:")
    print("  • One failure doesn't stop others")
    print("  • Can retry failed requests separately")
    print("  • Clear success/failure tracking")
    print()

    # Summary
    print("=" * 70)
    print("Best Practices for Async API Calls")
    print("=" * 70)
    print()
    print("✓ Always use ClientSession for connection pooling")
    print("✓ Set timeouts for all requests")
    print("✓ Handle errors per request")
    print("✓ Use retry logic for transient failures")
    print("✓ Rate limit with semaphores")
    print("✓ Batch requests when possible")
    print("✓ Monitor and log all failures")
    print()
    print("⚠ Common Pitfalls:")
    print("  • Creating new session per request (slow!)")
    print("  • No timeout (hangs forever)")
    print("  • Not handling errors (one failure stops all)")
    print("  • Too many concurrent requests (overwhelms API)")
    print()

    # Performance example
    print("=" * 70)
    print("Performance Comparison")
    print("=" * 70)
    print()
    print("Scenario: Fetch metadata for 100 models")
    print()
    print("Sequential (one at a time):")
    print("  100 requests × 200ms each = 20 seconds")
    print()
    print("Async concurrent (all at once):")
    print("  100 requests concurrent = ~200ms total")
    print("  Speedup: 100x faster!")
    print()
    print("Async with rate limit (10 concurrent):")
    print("  100 requests / 10 concurrent = 10 batches")
    print("  10 batches × 200ms = 2 seconds")
    print("  Speedup: 10x faster, respects API limits")
    print()

    print("=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("1. Use aiohttp.ClientSession for async HTTP requests")
    print("2. Reuse session for connection pooling")
    print("3. Set timeouts to prevent hanging")
    print("4. Handle errors gracefully per request")
    print("5. Use retry logic for transient failures")
    print("6. Rate limit with asyncio.Semaphore")
    print("7. Async provides 10-100x speedup for API calls")
    print("=" * 70)
    print()


if __name__ == "__main__":
    asyncio.run(simulate_api_calls())
