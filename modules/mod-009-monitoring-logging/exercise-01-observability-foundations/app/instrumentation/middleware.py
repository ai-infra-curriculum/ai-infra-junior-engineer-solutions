"""
Observability middleware for FastAPI.

Automatically captures metrics, logs, and traces for all HTTP requests.
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.instrumentation.logging import get_logger, LogContext
from app.instrumentation.metrics import (
    record_request,
    record_error,
    increment_active_requests,
    decrement_active_requests,
)

logger = get_logger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds observability to all HTTP requests.

    Captures:
    - Request/response metrics
    - Structured logs with correlation IDs
    - Request timing
    - Error tracking
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process HTTP request with observability.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            HTTP response
        """
        # Generate request ID for correlation
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Get request details
        method = request.method
        path = request.url.path

        # Skip metrics endpoint to avoid recursion
        if path == "/metrics":
            return await call_next(request)

        # Increment active requests
        increment_active_requests(path)

        # Start timing
        start_time = time.time()

        try:
            # Set up logging context for this request
            with LogContext(
                request_id=request_id,
                method=method,
                endpoint=path,
            ):
                logger.info(
                    "Request started",
                    client_host=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                )

                # Process request
                response = await call_next(request)

                # Calculate duration
                duration = time.time() - start_time
                duration_ms = duration * 1000

                # Record metrics
                record_request(
                    method=method,
                    endpoint=path,
                    duration=duration,
                    status_code=response.status_code
                )

                # Log completion
                logger.info(
                    "Request completed",
                    status_code=response.status_code,
                    duration_ms=round(duration_ms, 2),
                )

                # Add custom headers
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

                return response

        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            duration_ms = duration * 1000

            # Record error metrics
            record_error(
                method=method,
                endpoint=path,
                exception_type=type(e).__name__
            )

            # Log error
            logger.error(
                "Request failed",
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=round(duration_ms, 2),
                exc_info=True
            )

            # Re-raise to let FastAPI handle it
            raise

        finally:
            # Decrement active requests
            decrement_active_requests(path)
