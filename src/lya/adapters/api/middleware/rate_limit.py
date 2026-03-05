"""Rate limiting middleware."""

from __future__ import annotations

import time
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.

    Tracks requests per IP address with a sliding window.
    """

    def __init__(
        self,
        app,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Check rate limit and process request."""
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit
        if not self._is_allowed(client_ip):
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                path=request.url.path,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": self.window_seconds,
                },
            )

        # Add rate limit headers
        response = await call_next(request)
        remaining = self.max_requests - len(self._requests.get(client_ip, []))
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Window"] = str(self.window_seconds)

        return response

    def _is_allowed(self, client_ip: str) -> bool:
        """Check if request is within rate limit."""
        current_time = time.time()
        window_start = current_time - self.window_seconds

        # Get or create request history
        if client_ip not in self._requests:
            self._requests[client_ip] = []

        # Remove old requests outside window
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > window_start
        ]

        # Check if under limit
        if len(self._requests[client_ip]) >= self.max_requests:
            return False

        # Record this request
        self._requests[client_ip].append(current_time)
        return True
