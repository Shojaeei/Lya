"""Document Loader — pure Python 3.14.

Loads text content from PDF, TXT, and URL sources.
Uses PyMuPDF for PDF if available, falls back to basic extraction.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Load documents from various formats."""

    @staticmethod
    async def load(path: Path) -> str:
        """Load document content based on file extension."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return DocumentLoader.load_pdf(path)
        elif suffix in (".txt", ".md", ".csv", ".log", ".json", ".py", ".js"):
            return DocumentLoader.load_text(path)
        else:
            return DocumentLoader.load_text(path)

    @staticmethod
    def load_text(path: Path) -> str:
        """Load plain text file."""
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def load_pdf(path: Path) -> str:
        """Load PDF using PyMuPDF (fitz) if available."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            logger.warning("pymupdf_not_installed", message="Install pymupdf for PDF support")
            # Fallback: basic text extraction
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            # Strip binary garbage
            text = re.sub(r"[^\x20-\x7E\n\r\t]", "", text)
            return text

    @staticmethod
    async def load_url(url: str) -> str:
        """Load content from a URL."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            html = response.text

            if "text/html" in content_type:
                return DocumentLoader._strip_html(html)
            return html

    @staticmethod
    def _strip_html(html: str) -> str:
        """Basic HTML to text conversion."""
        # Remove script and style tags
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.I)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.I)
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode entities
        text = text.replace("&amp;", "&").replace("&lt;", "<")
        text = text.replace("&gt;", ">").replace("&nbsp;", " ")
        text = text.replace("&quot;", '"')
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()
