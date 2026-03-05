"""Outgoing port for LLM operations."""

from typing import Any, AsyncIterator, Protocol


class LLMPort(Protocol):
    """
    Outgoing port for LLM interactions.

    This port abstracts LLM provider details from the application layer.
    Infrastructure adapters implement this protocol.
    """

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text using LLM."""
        ...

    async def generate_structured(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate structured output using LLM."""
        ...

    async def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream generated text."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the configured model."""
        ...
