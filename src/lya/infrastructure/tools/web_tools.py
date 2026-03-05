"""Web Tools for Lya.

HTTP requests and web operations for autonomous agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WebResponse:
    """Web request response."""

    success: bool
    status_code: int | None
    url: str
    content: str
    headers: dict[str, str]
    json_data: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "status_code": self.status_code,
            "url": self.url,
            "content": self.content[:5000] if len(self.content) > 5000 else self.content,
            "headers": self.headers,
            "json": self.json_data,
            "error": self.error,
        }


class WebTools:
    """
    Web tools for HTTP operations.

    Features:
    - HTTP requests (GET, POST, PUT, DELETE, PATCH)
    - JSON fetching and parsing
    - Status checking
    - URL validation
    """

    DEFAULT_USER_AGENT = "Lya-Agent/0.1"
    DEFAULT_TIMEOUT = 30.0

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"User-Agent": self.DEFAULT_USER_AGENT},
                follow_redirects=True,
            )
        return self._client

    async def request(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        data: str | dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> WebResponse:
        """
        Make HTTP request.

        Args:
            url: Target URL
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            headers: Additional headers
            data: Request body data
            json_data: JSON data to send
            timeout: Request timeout

        Returns:
            WebResponse with result
        """
        try:
            client = await self._get_client()

            # Build request
            request_headers = headers or {}

            # Prepare content
            content = None
            if json_data is not None:
                content = json.dumps(json_data)
                request_headers.setdefault("Content-Type", "application/json")
            elif data is not None:
                content = data if isinstance(data, str) else json.dumps(data)

            logger.debug(
                "Making HTTP request",
                method=method.upper(),
                url=url,
                has_data=content is not None,
            )

            response = await client.request(
                method=method.upper(),
                url=url,
                content=content,
                headers=request_headers,
                timeout=timeout or self.timeout,
            )

            # Parse response
            response_text = response.text
            response_json = None

            try:
                if "application/json" in response.headers.get("content-type", ""):
                    response_json = response.json()
            except (json.JSONDecodeError, ValueError):
                pass

            logger.debug(
                "HTTP request completed",
                status_code=response.status_code,
                url=str(response.url),
                content_length=len(response_text),
            )

            return WebResponse(
                success=response.status_code < 400,
                status_code=response.status_code,
                url=str(response.url),
                content=response_text,
                headers=dict(response.headers),
                json_data=response_json,
            )

        except httpx.HTTPStatusError as e:
            logger.warning(
                "HTTP error",
                status_code=e.response.status_code,
                url=url,
            )
            return WebResponse(
                success=False,
                status_code=e.response.status_code,
                url=url,
                content=e.response.text,
                headers=dict(e.response.headers),
                error=f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
            )

        except httpx.TimeoutException:
            logger.error("HTTP request timeout", url=url, timeout=timeout or self.timeout)
            return WebResponse(
                success=False,
                status_code=None,
                url=url,
                content="",
                headers={},
                error=f"Request timed out after {timeout or self.timeout}s",
            )

        except Exception as e:
            logger.error("HTTP request failed", url=url, error=str(e))
            return WebResponse(
                success=False,
                status_code=None,
                url=url,
                content="",
                headers={},
                error=str(e),
            )

    async def fetch_json(
        self,
        url: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Fetch and parse JSON from URL.

        Args:
            url: JSON URL
            timeout: Request timeout

        Returns:
            Parsed JSON data or error dict
        """
        response = await self.request(url, method="GET", timeout=timeout)

        if not response.success:
            return {
                "success": False,
                "error": response.error,
                "status_code": response.status_code,
            }

        if response.json_data is not None:
            return {
                "success": True,
                "data": response.json_data,
                "status_code": response.status_code,
            }

        # Try to parse manually
        try:
            data = json.loads(response.content)
            return {
                "success": True,
                "data": data,
                "status_code": response.status_code,
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON: {str(e)}",
                "content_preview": response.content[:200],
                "status_code": response.status_code,
            }

    async def check_status(
        self,
        url: str,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """
        Quick status check (HEAD request).

        Args:
            url: URL to check
            timeout: Request timeout

        Returns:
            Status check result
        """
        try:
            client = await self._get_client()

            response = await client.head(
                url=url,
                timeout=timeout,
                follow_redirects=True,
            )

            return {
                "success": response.status_code < 400,
                "up": response.status_code < 400,
                "status_code": response.status_code,
                "url": str(response.url),
            }

        except Exception as e:
            return {
                "success": False,
                "up": False,
                "error": str(e),
                "url": url,
            }

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> WebResponse:
        """Convenience method for GET requests."""
        return await self.request(url, method="GET", headers=headers, timeout=timeout)

    async def post(
        self,
        url: str,
        data: str | dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> WebResponse:
        """Convenience method for POST requests."""
        return await self.request(
            url, method="POST", data=data, json_data=json_data,
            headers=headers, timeout=timeout
        )

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def normalize_url(self, base: str, path: str) -> str:
        """Normalize URL with base and path."""
        return urljoin(base, path)

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> WebTools:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# Global instance
_web_tools: WebTools | None = None


def get_web_tools() -> WebTools:
    """Get global web tools instance."""
    global _web_tools
    if _web_tools is None:
        _web_tools = WebTools()
    return _web_tools
