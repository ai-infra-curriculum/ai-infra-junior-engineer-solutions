"""
Rate Limiting Module

Implements request rate limiting for API endpoints.
"""

import time
from collections import deque
from typing import Optional


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 100,
        burst_size: int = 20
    ):
        """Initialize rate limiter."""
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.requests = deque()

    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_update = now

    def allow_request(self) -> bool:
        """Check if request is allowed."""
        self._refill_tokens()
        if self.tokens >= 1:
            self.tokens -= 1
            self.requests.append(time.time())
            return True
        return False

    def get_wait_time(self) -> float:
        """Get time to wait before next request allowed."""
        self._refill_tokens()
        if self.tokens >= 1:
            return 0.0
        return (1.0 - self.tokens) / (self.requests_per_minute / 60.0)

    def reset(self):
        """Reset rate limiter state."""
        self.tokens = self.burst_size
        self.last_update = time.time()
        self.requests.clear()
