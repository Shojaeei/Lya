"""Playwright Browser Adapter.

Provides browser automation capabilities using Playwright.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


@dataclass
class BrowserAction:
    """Represents a browser action."""
    action: Literal["navigate", "click", "fill", "select", "scroll", "screenshot", "extract"]
    selector: str | None = None
    value: str | None = None
    options: dict[str, Any] | None = None


@dataclass
class ExtractedData:
    """Extracted data from webpage."""
    text: str
    links: list[dict[str, str]]
    images: list[dict[str, str]]
    title: str
    meta: dict[str, str]
    structured: dict[str, Any] | None = None


@dataclass
class ScreenshotResult:
    """Screenshot capture result."""
    path: Path | None
    base64: str | None
    width: int
    height: int


class BrowserSession:
    """Manages a browser session with Playwright."""

    def __init__(
        self,
        headless: bool = True,
        browser_type: Literal["chromium", "firefox", "webkit"] = "chromium",
        user_agent: str | None = None,
        viewport: dict[str, int] | None = None,
    ):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright not installed. Run: pip install playwright\n"
                "Then install browsers: playwright install chromium"
            )

        self.headless = headless
        self.browser_type = browser_type
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        self.viewport = viewport or {"width": 1920, "height": 1080}

        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def __aenter__(self) -> "BrowserSession":
        """Start browser session."""
        self._playwright = await async_playwright().start()

        # Launch browser
        browser_launcher = getattr(self._playwright, self.browser_type)
        self._browser = await browser_launcher.launch(headless=self.headless)

        # Create context
        self._context = await self._browser.new_context(
            user_agent=self.user_agent,
            viewport=self.viewport,
            accept_downloads=True,
        )

        # Create page
        self._page = await self._context.new_page()

        # Set default timeout
        self._page.set_default_timeout(30000)

        logger.info("Browser session started", browser=self.browser_type)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup browser session."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

        logger.info("Browser session closed")

    @property
    def page(self) -> Page:
        """Get current page."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        return self._page

    async def navigate(self, url: str, wait_for: str | None = "networkidle") -> None:
        """Navigate to URL."""
        logger.info("Navigating to URL", url=url)

        try:
            response = await self.page.goto(url, wait_until=wait_for)

            if response:
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}: {response.status_text}")

                logger.info(
                    "Page loaded",
                    url=url,
                    status=response.status,
                    title=await self.page.title(),
                )

        except Exception as e:
            logger.error("Navigation failed", url=url, error=str(e))
            raise

    async def click(self, selector: str, wait_for_navigation: bool = False) -> None:
        """Click an element."""
        logger.debug("Clicking element", selector=selector)

        try:
            if wait_for_navigation:
                async with self.page.expect_navigation():
                    await self.page.click(selector)
            else:
                await self.page.click(selector)

        except Exception as e:
            logger.error("Click failed", selector=selector, error=str(e))
            raise

    async def fill(self, selector: str, value: str) -> None:
        """Fill a form field."""
        logger.debug("Filling field", selector=selector)

        try:
            await self.page.fill(selector, value)
        except Exception as e:
            logger.error("Fill failed", selector=selector, error=str(e))
            raise

    async def select(self, selector: str, value: str) -> None:
        """Select from dropdown."""
        logger.debug("Selecting option", selector=selector, value=value)

        try:
            await self.page.select_option(selector, value)
        except Exception as e:
            logger.error("Select failed", selector=selector, error=str(e))
            raise

    async def get_text(self, selector: str | None = None) -> str:
        """Get text content."""
        if selector:
            element = await self.page.query_selector(selector)
            if not element:
                raise ValueError(f"Element not found: {selector}")
            return await element.text_content() or ""
        else:
            return await self.page.content()

    async def extract_all(self) -> ExtractedData:
        """Extract all data from current page."""
        logger.debug("Extracting page data")

        # Get basic info
        title = await self.page.title()
        content = await self.page.content()

        # Extract links
        links = await self.page.eval_on_selector_all(
            "a[href]",
            """elements => elements.map(e => ({
                text: e.textContent?.trim() || '',
                href: e.href,
                title: e.title || ''
            }))""",
        )

        # Filter valid links
        valid_links = [
            link for link in links
            if link["href"] and not link["href"].startswith(("javascript:", "mailto:", "tel:"))
        ]

        # Extract images
        images = await self.page.eval_on_selector_all(
            "img[src]",
            """elements => elements.map(e => ({
                src: e.src,
                alt: e.alt || '',
                width: e.naturalWidth,
                height: e.naturalHeight
            }))""",
        )

        # Extract meta tags
        meta = await self.page.eval_on_selector_all(
            "meta",
            """elements => {
                const data = {};
                elements.forEach(e => {
                    const name = e.name || e.property || e.getAttribute('property');
                    if (name) data[name] = e.content;
                });
                return data;
            }""",
        )

        # Get visible text only (cleaner)
        visible_text = await self.page.evaluate(
            """() => {
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    null,
                    false
                );
                const textNodes = [];
                let node;
                while (node = walker.nextNode()) {
                    const text = node.textContent.trim();
                    if (text && text.length > 0) {
                        textNodes.push(text);
                    }
                }
                return textNodes.join('\\n');
            }"""
        )

        return ExtractedData(
            text=visible_text[:10000],  # Limit text
            links=valid_links[:50],  # Limit links
            images=images[:20],  # Limit images
            title=title,
            meta=meta,
        )

    async def screenshot(
        self,
        path: Path | str | None = None,
        full_page: bool = True,
        selector: str | None = None,
    ) -> ScreenshotResult:
        """Take screenshot."""
        logger.debug("Taking screenshot")

        screenshot_bytes = await self.page.screenshot(
            path=str(path) if path else None,
            full_page=full_page and not selector,
            type="png",
        )

        # Get dimensions
        if selector:
            element = await self.page.query_selector(selector)
            if element:
                box = await element.bounding_box()
                width, height = int(box["width"]), int(box["height"])
            else:
                width, height = 0, 0
        else:
            viewport = await self.page.viewport_size()
            width, height = viewport["width"], viewport["height"]

        base64_str = base64.b64encode(screenshot_bytes).decode() if screenshot_bytes else None

        return ScreenshotResult(
            path=Path(path) if path else None,
            base64=base64_str,
            width=width,
            height=height,
        )

    async def scroll(self, direction: Literal["up", "down", "bottom"] = "down") -> None:
        """Scroll the page."""
        if direction == "bottom":
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        elif direction == "down":
            await self.page.evaluate("window.scrollBy(0, window.innerHeight)")
        elif direction == "up":
            await self.page.evaluate("window.scrollBy(0, -window.innerHeight)")

    async def wait_for_selector(self, selector: str, timeout: int = 10000) -> None:
        """Wait for element to appear."""
        await self.page.wait_for_selector(selector, timeout=timeout)


class PlaywrightBrowserAdapter:
    """
    High-level browser automation adapter.

    Provides safe, rate-limited browser automation with
    built-in safety checks and logging.
    """

    # Safety: Blocked domains
    BLOCKED_DOMAINS = [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "192.168.",
        "10.0.",
        "172.16.",
    ]

    # Safety: Rate limiting
    REQUEST_DELAY = 1.0  # seconds between requests
    MAX_REQUESTS_PER_SESSION = 100

    def __init__(self):
        self._session: BrowserSession | None = None
        self._request_count = 0
        self._last_request_time = 0.0

    def _is_blocked_url(self, url: str) -> bool:
        """Check if URL is blocked for safety."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        for blocked in self.BLOCKED_DOMAINS:
            if blocked in domain:
                return True

        return False

    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting."""
        import time

        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            await asyncio.sleep(self.REQUEST_DELAY - elapsed)

        self._last_request_time = time.time()
        self._request_count += 1

        if self._request_count > self.MAX_REQUESTS_PER_SESSION:
            raise RuntimeError("Maximum requests per session exceeded")

    async def browse(
        self,
        url: str,
        actions: list[BrowserAction] | None = None,
        extract_data: bool = True,
    ) -> dict[str, Any]:
        """
        Browse a website and perform actions.

        Args:
            url: URL to browse
            actions: List of actions to perform
            extract_data: Whether to extract page data

        Returns:
            Results including extracted data and screenshots
        """
        # Safety check
        if self._is_blocked_url(url):
            raise ValueError(f"URL blocked for safety: {url}")

        await self._enforce_rate_limit()

        results = {
            "url": url,
            "actions_performed": [],
            "data": None,
            "screenshots": [],
        }

        async with BrowserSession() as session:
            # Navigate
            await session.navigate(url)

            # Perform actions
            if actions:
                for action in actions:
                    try:
                        if action.action == "click":
                            await session.click(action.selector)
                            results["actions_performed"].append(f"click:{action.selector}")

                        elif action.action == "fill":
                            await session.fill(action.selector, action.value or "")
                            results["actions_performed"].append(f"fill:{action.selector}")

                        elif action.action == "select":
                            await session.select(action.selector, action.value or "")
                            results["actions_performed"].append(f"select:{action.selector}")

                        elif action.action == "scroll":
                            await session.scroll(action.value or "down")  # type: ignore
                            results["actions_performed"].append("scroll")

                        elif action.action == "screenshot":
                            screenshot = await session.screenshot()
                            results["screenshots"].append(screenshot.base64)
                            results["actions_performed"].append("screenshot")

                        elif action.action == "wait":
                            if action.selector:
                                await session.wait_for_selector(action.selector)
                            else:
                                await asyncio.sleep(float(action.value or 1))
                            results["actions_performed"].append("wait")

                    except Exception as e:
                        logger.error("Action failed", action=action.action, error=str(e))
                        results["actions_performed"].append(f"error:{action.action}")

            # Extract data
            if extract_data:
                data = await session.extract_all()
                results["data"] = {
                    "title": data.title,
                    "text": data.text[:5000],  # Limit text
                    "links": data.links[:20],  # Limit links
                    "meta": data.meta,
                }

        return results

    async def scrape(
        self,
        url: str,
        selector: str | None = None,
    ) -> str:
        """
        Quick scrape of webpage text.

        Args:
            url: URL to scrape
            selector: Optional CSS selector to extract specific element

        Returns:
            Extracted text
        """
        if self._is_blocked_url(url):
            raise ValueError(f"URL blocked for safety: {url}")

        await self._enforce_rate_limit()

        async with BrowserSession() as session:
            await session.navigate(url, wait_for="domcontentloaded")

            if selector:
                return await session.get_text(selector)
            else:
                data = await session.extract_all()
                return data.text

    async def fill_form(
        self,
        url: str,
        fields: dict[str, str],
        submit_selector: str | None = None,
    ) -> dict[str, Any]:
        """
        Fill out a web form.

        Args:
            url: Form page URL
            fields: Dict of {selector: value}
            submit_selector: Optional submit button selector

        Returns:
            Results including final page data
        """
        if self._is_blocked_url(url):
            raise ValueError(f"URL blocked for safety: {url}")

        await self._enforce_rate_limit()

        async with BrowserSession() as session:
            await session.navigate(url)

            # Fill fields
            for selector, value in fields.items():
                await session.fill(selector, value)

            # Submit if specified
            if submit_selector:
                await session.click(submit_selector, wait_for_navigation=True)
            else:
                # Try common submit selectors
                for submit in ["button[type='submit']", "input[type='submit']", "button:has-text('Submit')"]:
                    try:
                        await session.click(submit, wait_for_navigation=True)
                        break
                    except:
                        continue

            # Return final state
            return {
                "url": session.page.url,
                "title": await session.page.title(),
                "text": await session.get_text(),
            }

    async def screenshot_page(
        self,
        url: str,
        full_page: bool = True,
    ) -> str:
        """
        Take screenshot of webpage.

        Args:
            url: URL to screenshot
            full_page: Whether to capture full page

        Returns:
            Base64-encoded PNG
        """
        if self._is_blocked_url(url):
            raise ValueError(f"URL blocked for safety: {url}")

        await self._enforce_rate_limit()

        async with BrowserSession() as session:
            await session.navigate(url)
            result = await session.screenshot(full_page=full_page)
            return result.base64 or ""

    async def search_and_click(
        self,
        url: str,
        link_text: str,
    ) -> dict[str, Any]:
        """
        Navigate to page, find link by text, click it.

        Args:
            url: Starting URL
            link_text: Text to search for in links

        Returns:
            Results from clicked page
        """
        if self._is_blocked_url(url):
            raise ValueError(f"URL blocked for safety: {url}")

        await self._enforce_rate_limit()

        async with BrowserSession() as session:
            await session.navigate(url)

            # Find link
            links = await session.extract_all()
            target_link = None

            for link in links.links:
                if link_text.lower() in link["text"].lower():
                    target_link = link["href"]
                    break

            if not target_link:
                raise ValueError(f"Link not found: {link_text}")

            # Click it
            await session.navigate(target_link)

            return {
                "url": session.page.url,
                "title": await session.page.title(),
                "text": await session.get_text(),
            }

    async def close(self) -> None:
        """Cleanup."""
        pass  # Sessions are context managers


# Convenience functions for capability generation
def get_browser_capability_code() -> str:
    """Generate the browser automation capability code."""
    return '''"""Browser Automation Capability.

Provides web browsing, form filling, and data extraction.
"""

from typing import Any
import asyncio
from dataclasses import dataclass

@dataclass
class BrowseResult:
    url: str
    title: str
    text: str
    links: list[dict]

async def browse_website(url: str, wait_seconds: int = 3) -> BrowseResult:
    """
    Browse a website and extract content.

    Args:
        url: Website URL to browse
        wait_seconds: Time to wait for page load

    Returns:
        BrowseResult with title, text, and links
    """
    # Import here to allow capability to be loaded without deps
    try:
        from lya.infrastructure.tools.browser.playwright_adapter import PlaywrightBrowserAdapter
    except ImportError:
        raise ImportError("Playwright not installed. Run: pip install playwright")

    adapter = PlaywrightBrowserAdapter()

    try:
        text = await adapter.scrape(url)
        # Parse basic info from text
        lines = text.split("\\n")
        title = lines[0] if lines else ""

        return BrowseResult(
            url=url,
            title=title,
            text=text,
            links=[]  # Simplified for now
        )
    finally:
        await adapter.close()

async def fill_form(url: str, fields: dict[str, str]) -> dict[str, Any]:
    """
    Fill out a web form.

    Args:
        url: Form page URL
        fields: Dictionary of {field_selector: value}

    Returns:
        Result after form submission
    """
    from lya.infrastructure.tools.browser.playwright_adapter import PlaywrightBrowserAdapter

    adapter = PlaywrightBrowserAdapter()

    try:
        return await adapter.fill_form(url, fields)
    finally:
        await adapter.close()

async def take_screenshot(url: str, full_page: bool = True) -> str:
    """
    Take screenshot of webpage.

    Args:
        url: Page URL
        full_page: Capture full page or viewport

    Returns:
        Base64-encoded PNG image
    """
    from lya.infrastructure.tools.browser.playwright_adapter import PlaywrightBrowserAdapter

    adapter = PlaywrightBrowserAdapter()

    try:
        return await adapter.screenshot_page(url, full_page)
    finally:
        await adapter.close()
'''
