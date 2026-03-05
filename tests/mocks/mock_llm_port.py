"""Mock LLM Port implementation."""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

from lya.application.ports.outgoing.llm_port import LLMPort


class MockLLMPort(LLMPort):
    """Mock implementation of LLM Port for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        """
        Initialize mock with predefined responses.

        Args:
            responses: Dict mapping prompts to responses
        """
        self.responses = responses or {}
        self.generate_call_count = 0
        self.chat_call_count = 0
        self.embed_call_count = 0
        self.last_prompt: str | None = None
        self.last_messages: list[dict[str, str]] | None = None

        # Create async mocks for tracking
        self.generate = AsyncMock(side_effect=self._generate_impl)
        self.chat = AsyncMock(side_effect=self._chat_impl)
        self.embed = AsyncMock(side_effect=self._embed_impl)
        self.embed_batch = AsyncMock(side_effect=self._embed_batch_impl)
        self.generate_structured = AsyncMock(side_effect=self._generate_structured_impl)

    async def _generate_impl(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Internal implementation of generate."""
        self.generate_call_count += 1
        self.last_prompt = prompt

        # Return predefined response or default
        return self.responses.get(prompt, "Default mock response")

    async def _chat_impl(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Internal implementation of chat."""
        self.chat_call_count += 1
        self.last_messages = messages

        # Get last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return self.responses.get(msg.get("content", ""), "Mock chat response")

        return "Mock chat response"

    async def _embed_impl(self, text: str) -> list[float]:
        """Internal implementation of embed."""
        self.embed_call_count += 1

        # Return deterministic embedding
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        import random
        random.seed(int.from_bytes(hash_bytes[:4], "big"))
        embedding = [random.uniform(-1, 1) for _ in range(384)]
        norm = sum(x * x for x in embedding) ** 0.5
        return [x / norm for x in embedding]

    async def _embed_batch_impl(self, texts: list[str]) -> list[list[float]]:
        """Internal implementation of embed_batch."""
        return [await self._embed_impl(text) for text in texts]

    async def _generate_structured_impl(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Internal implementation of generate_structured."""
        # Return a mock response based on schema
        return {key: f"mock_{key}" for key in output_schema.get("properties", {}).keys()}

    async def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream mock response."""
        response = await self.generate(prompt, temperature)
        # Yield word by word
        for word in response.split():
            yield word + " "

    def get_model_info(self) -> dict[str, Any]:
        """Get mock model info."""
        return {
            "provider": "mock",
            "model": "mock-model",
        }

    async def list_models(self) -> list[dict[str, Any]]:
        """List mock models."""
        return [
            {"id": "mock-model", "name": "Mock Model"},
        ]

    async def close(self) -> None:
        """Close mock."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
