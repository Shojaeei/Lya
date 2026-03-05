"""Text Chunker — pure Python 3.14.

Splits text into overlapping chunks with smart boundary detection.
"""

from __future__ import annotations

import re


class TextChunker:
    """Split text into overlapping chunks at paragraph/sentence boundaries."""

    @staticmethod
    def chunk(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[str]:
        """Split text into overlapping chunks.

        Tries to split at paragraph boundaries, then sentence boundaries,
        then word boundaries.

        Args:
            text: Input text
            chunk_size: Target size of each chunk in characters
            overlap: Number of overlap characters between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        chunks: list[str] = []
        pos = 0

        while pos < len(text):
            end = pos + chunk_size

            if end >= len(text):
                chunk = text[pos:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            segment = text[pos:end]

            # Try to split at paragraph boundary
            split_at = segment.rfind("\n\n")
            if split_at > chunk_size // 3:
                split_at += 2  # Include the newlines
            else:
                # Try sentence boundary
                for sep in (". ", "! ", "? ", ".\n"):
                    idx = segment.rfind(sep)
                    if idx > chunk_size // 3:
                        split_at = idx + len(sep)
                        break
                else:
                    # Try word boundary
                    space_idx = segment.rfind(" ")
                    if space_idx > chunk_size // 3:
                        split_at = space_idx + 1
                    else:
                        split_at = chunk_size

            chunk = text[pos:pos + split_at].strip()
            if chunk:
                chunks.append(chunk)

            # Move position with overlap
            pos = pos + split_at - overlap
            if pos <= 0:
                pos = split_at

        return chunks
