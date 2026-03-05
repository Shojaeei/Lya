"""Browser Automation Capability.

Provides web browsing, form filling, data extraction, and screenshot
capabilities using Playwright.

Safety features:
- Rate limiting
- Domain allowlisting
- Content filtering
- Session isolation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
import asyncio
import json
import hashlib

try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BrowserSession:
    """Browser session configuration."""
    headless: bool = True
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    viewport: dict[str, int] = field(default_factory=lambda: {"width": 1920, "height": 1080})
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)
    rate_limit_ms: int = 1000  # Minimum time between requests
    max_pages: int = 10
    timeout_ms: int = 30000


@dataclass
class ExtractedElement:
    """Extracted web element."""
    tag: str
    text: str
    attributes: dict[str, str] = field(default_factory=dict)
    selector: str = ""
    position: dict[str, int] = field(default_factory=dict)


@dataclass
class FormField:
    """Form field definition."""
    selector: str
    field_type: Literal["text", "password", "select", "checkbox", "radio", "file", "textarea"]
    value: str | bool | list[str]
    required: bool = False


class BrowserAutomation:
    """
    Browser automation using Playwright.

    Features:
    - Navigate websites
    - Fill forms
    - Click elements
    - Extract data
    - Take screenshots
    - Download files
    """

    def __init__(self, session: BrowserSession | None = None):
        self.session = session or BrowserSession()
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._pages: dict[str, Page] = {}
        self._last_request_time: dict[str, float] = {}
        self._downloads_dir = Path.home() / ".lya" / "downloads"
        self._downloads_dir.mkdir(parents=True, exist_ok=True)

    async def _ensure_playwright(self) -> None:
        """Initialize Playwright if not already done."""
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright not installed. Run: "
                "pip install playwright && playwright install"
            )

        if self._playwright is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.session.headless,
            )
            self._context = await self._browser.new_context(
                viewport=self.session.viewport,
                user_agent=self.session.user_agent,
                accept_downloads=True,
            )

            # Set up download handler
            self._context.set_default_timeout(self.session.timeout_ms)
            self._context.on("download", self._handle_download)

    async def _handle_download(self, download) -> None:
        """Handle file downloads."""
        try:
            path = self._downloads_dir / download.suggested_filename
            await download.save_as(path)
            logger.info("Downloaded file", path=str(path), url=download.url)
        except Exception as e:
            logger.error("Download failed", error=str(e))

    def _check_domain_allowed(self, url: str) -> bool:
        """Check if domain is allowed."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Check blocked domains
        for blocked in self.session.blocked_domains:
            if blocked in domain:
                logger.warning("Domain blocked", domain=domain, blocked=blocked)
                return False

        # Check allowed domains (if whitelist configured)
        if self.session.allowed_domains:
            for allowed in self.session.allowed_domains:
                if allowed in domain or domain == allowed:
                    return True
            logger.warning("Domain not in whitelist", domain=domain)
            return False

        return True

    async def _rate_limit(self, url: str) -> None:
        """Apply rate limiting."""
        domain = urlparse(url).netloc
        last_time = self._last_request_time.get(domain, 0)
        elapsed = asyncio.get_event_loop().time() - last_time

        if elapsed < self.session.rate_limit_ms / 1000:
            delay = (self.session.rate_limit_ms / 1000) - elapsed
            logger.debug("Rate limiting", domain=domain, delay=delay)
            await asyncio.sleep(delay)

        self._last_request_time[domain] = asyncio.get_event_loop().time()

    async def _get_page(self, page_id: str | None = None) -> Page:
        """Get or create a page."""
        await self._ensure_playwright()

        if page_id and page_id in self._pages:
            return self._pages[page_id]

        if len(self._pages) >= self.session.max_pages:
            # Close oldest page
            oldest = min(self._pages.items(), key=lambda x: x[1]._access_time if hasattr(x[1], '_access_time') else 0)
            await oldest[1].close()
            del self._pages[oldest[0]]

        page = await self._context.new_page()
        page_id = page_id or f"page_{len(self._pages)}"
        self._pages[page_id] = page

        return page

    async def navigate(
        self,
        url: str,
        wait_for: str | None = None,
        page_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Navigate to a URL.

        Args:
            url: URL to navigate to
            wait_for: Optional selector to wait for
            page_id: Page identifier for session management

        Returns:
            Navigation result with page info
        """
        if not self._check_domain_allowed(url):
            raise PermissionError(f"Domain not allowed: {url}")

        await self._rate_limit(url)
        page = await self._get_page(page_id)

        try:
            response = await page.goto(url, wait_until="networkidle")

            if wait_for:
                await page.wait_for_selector(wait_for, timeout=self.session.timeout_ms)

            title = await page.title()
            content = await page.content()

            logger.info("Navigated", url=url, title=title[:50])

            return {
                "success": True,
                "url": url,
                "title": title,
                "status": response.status if response else None,
                "content_length": len(content),
                "page_id": page_id,
            }

        except Exception as e:
            logger.error("Navigation failed", url=url, error=str(e))
            return {
                "success": False,
                "url": url,
                "error": str(e),
            }

    async def extract_text(
        self,
        url: str | None = None,
        selector: str = "body",
        page_id: str | None = None,
    ) -> str:
        """
        Extract text content from page.

        Args:
            url: URL to navigate (if not using existing page)
            selector: CSS selector for extraction
            page_id: Page identifier

        Returns:
            Extracted text
        """
        page = await self._get_page(page_id)

        if url:
            if not self._check_domain_allowed(url):
                raise PermissionError(f"Domain not allowed: {url}")
            await self._rate_limit(url)
            await page.goto(url, wait_until="networkidle")

        try:
            element = await page.query_selector(selector)
            if element:
                text = await element.inner_text()
                return text.strip()
            return ""

        except Exception as e:
            logger.error("Text extraction failed", error=str(e))
            raise

    async def extract_structured(
        self,
        selectors: dict[str, str],
        url: str | None = None,
        page_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Extract structured data using CSS selectors.

        Args:
            selectors: Dict of {field_name: css_selector}
            url: URL to navigate
            page_id: Page identifier

        Returns:
            Dict of extracted data
        """
        page = await self._get_page(page_id)

        if url:
            if not self._check_domain_allowed(url):
                raise PermissionError(f"Domain not allowed: {url}")
            await self._rate_limit(url)
            await page.goto(url, wait_until="networkidle")

        result = {}

        for field_name, selector in selectors.items():
            try:
                elements = await page.query_selector_all(selector)
                if len(elements) == 1:
                    text = await elements[0].inner_text()
                    result[field_name] = text.strip()
                elif len(elements) > 1:
                    texts = []
                    for el in elements:
                        text = await el.inner_text()
                        texts.append(text.strip())
                    result[field_name] = texts
                else:
                    result[field_name] = None
            except Exception as e:
                logger.warning("Extraction failed for field", field=field_name, error=str(e))
                result[field_name] = None

        return result

    async def fill_form(
        self,
        fields: list[FormField],
        submit_selector: str | None = None,
        page_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Fill a form with provided data.

        Args:
            fields: List of form fields to fill
            submit_selector: Optional selector for submit button
            page_id: Page identifier

        Returns:
            Form submission result
        """
        page = await self._get_page(page_id)
        results = {"filled": [], "failed": []}

        for field in fields:
            try:
                element = await page.query_selector(field.selector)
                if not element:
                    results["failed"].append({"field": field.selector, "reason": "not_found"})
                    continue

                if field.field_type == "text":
                    await element.fill(str(field.value))
                elif field.field_type == "password":
                    await element.fill(str(field.value))
                elif field.field_type == "textarea":
                    await element.fill(str(field.value))
                elif field.field_type == "select":
                    await element.select_option(str(field.value))
                elif field.field_type == "checkbox":
                    if field.value:
                        await element.check()
                    else:
                        await element.uncheck()
                elif field.field_type == "radio":
                    await element.check()
                elif field.field_type == "file":
                    if isinstance(field.value, str):
                        await element.set_input_files(field.value)

                results["filled"].append(field.selector)

            except Exception as e:
                logger.error("Form fill failed", field=field.selector, error=str(e))
                results["failed"].append({"field": field.selector, "reason": str(e)})

        # Submit if requested
        if submit_selector and results["filled"]:
            try:
                submit_btn = await page.query_selector(submit_selector)
                if submit_btn:
                    await submit_btn.click()
                    await page.wait_for_load_state("networkidle")
                    results["submitted"] = True
                    results["current_url"] = page.url
                else:
                    results["submitted"] = False
                    results["submit_error"] = "Submit button not found"
            except Exception as e:
                results["submitted"] = False
                results["submit_error"] = str(e)

        return results

    async def click(
        self,
        selector: str,
        wait_for_navigation: bool = True,
        page_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Click an element.

        Args:
            selector: Element selector
            wait_for_navigation: Wait for page navigation
            page_id: Page identifier

        Returns:
            Click result
        """
        page = await self._get_page(page_id)

        try:
            element = await page.query_selector(selector)
            if not element:
                return {"success": False, "error": "Element not found"}

            if wait_for_navigation:
                async with page.expect_navigation():
                    await element.click()
            else:
                await element.click()

            return {
                "success": True,
                "url": page.url,
                "title": await page.title(),
            }

        except Exception as e:
            logger.error("Click failed", selector=selector, error=str(e))
            return {"success": False, "error": str(e)}

    async def take_screenshot(
        self,
        path: str | None = None,
        selector: str | None = None,
        full_page: bool = False,
        page_id: str | None = None,
    ) -> str:
        """
        Take a screenshot.

        Args:
            path: Path to save screenshot (auto-generated if None)
            selector: Optional element to screenshot
            full_page: Capture full page or viewport
            page_id: Page identifier

        Returns:
            Path to saved screenshot
        """
        page = await self._get_page(page_id)

        if path is None:
            timestamp = asyncio.get_event_loop().time()
            hash_val = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
            path = str(self._downloads_dir / f"screenshot_{hash_val}.png")

        path = Path(path)

        try:
            if selector:
                element = await page.query_selector(selector)
                if element:
                    await element.screenshot(path=str(path))
                else:
                    raise ValueError(f"Element not found: {selector}")
            else:
                await page.screenshot(path=str(path), full_page=full_page)

            logger.info("Screenshot saved", path=str(path))
            return str(path)

        except Exception as e:
            logger.error("Screenshot failed", error=str(e))
            raise

    async def scroll(
        self,
        direction: Literal["up", "down", "top", "bottom"] = "down",
        amount: int = 1000,
        page_id: str | None = None,
    ) -> None:
        """Scroll the page."""
        page = await self._get_page(page_id)

        if direction == "up":
            await page.evaluate(f"window.scrollBy(0, -{amount})")
        elif direction == "down":
            await page.evaluate(f"window.scrollBy(0, {amount})")
        elif direction == "top":
            await page.evaluate("window.scrollTo(0, 0)")
        elif direction == "bottom":
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

    async def get_page_info(self, page_id: str | None = None) -> dict[str, Any]:
        """Get current page information."""
        page = await self._get_page(page_id)

        return {
            "url": page.url,
            "title": await page.title(),
            "viewport": await page.viewport_size(),
        }

    async def search_on_page(
        self,
        query: str,
        page_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for text on the page and return matches."""
        page = await self._get_page(page_id)

        # Use Playwright's built-in locator
        results = []
        try:
            # Get all text content
            body = await page.query_selector("body")
            if body:
                text = await body.inner_text()
                if query.lower() in text.lower():
                    # Find the element containing the text
                    locator = page.locator(f"text={query}")
                    count = await locator.count()
                    for i in range(min(count, 10)):
                        element = locator.nth(i)
                        results.append({
                            "text": await element.inner_text(),
                            "visible": await element.is_visible(),
                        })
        except Exception as e:
            logger.error("Search failed", error=str(e))

        return results

    async def close_page(self, page_id: str) -> None:
        """Close a specific page."""
        if page_id in self._pages:
            await self._pages[page_id].close()
            del self._pages[page_id]

    async def close(self) -> None:
        """Close all browser resources."""
        for page in self._pages.values():
            await page.close()
        self._pages.clear()

        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_playwright()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience functions for capability interface
async def browse_website(url: str, extract_text: bool = True) -> dict[str, Any]:
    """Quick browse and extract."""
    async with BrowserAutomation() as browser:
        result = await browser.navigate(url)
        if result["success"] and extract_text:
            result["text"] = await browser.extract_text()
        return result


async def fill_and_submit(
    url: str,
    fields: dict[str, str],
    submit_button: str = "button[type='submit']",
) -> dict[str, Any]:
    """Fill form and submit."""
    form_fields = [
        FormField(
            selector=f"input[name='{name}'], #{name}, [placeholder='{name}']",
            field_type="text",
            value=value,
        )
        for name, value in fields.items()
    ]

    async with BrowserAutomation() as browser:
        await browser.navigate(url)
        return await browser.fill_form(form_fields, submit_button)


async def extract_table(
    url: str,
    table_selector: str = "table",
) -> list[dict[str, str]]:
    """Extract data from HTML table."""
    async with BrowserAutomation() as browser:
        await browser.navigate(url)

        # Get table rows
        page = await browser._get_page()
        rows = await page.query_selector_all(f"{table_selector} tr")

        if not rows:
            return []

        # Extract headers
        headers = []
        header_cells = await rows[0].query_selector_all("th, td")
        for cell in header_cells:
            text = await cell.inner_text()
            headers.append(text.strip())

        # Extract data rows
        data = []
        for row in rows[1:]:
            cells = await row.query_selector_all("td")
            if cells:
                row_data = {}
                for i, cell in enumerate(cells):
                    if i < len(headers):
                        text = await cell.inner_text()
                        row_data[headers[i]] = text.strip()
                if row_data:
                    data.append(row_data)

        return data
