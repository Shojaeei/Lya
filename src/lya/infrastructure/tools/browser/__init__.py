"""Browser automation tools."""

from .playwright_adapter import (
    BrowserSession,
    BrowserAction,
    ExtractedData,
    ScreenshotResult,
    PlaywrightBrowserAdapter,
    get_browser_capability_code,
)

__all__ = [
    "BrowserSession",
    "BrowserAction",
    "ExtractedData",
    "ScreenshotResult",
    "PlaywrightBrowserAdapter",
    "get_browser_capability_code",
]
