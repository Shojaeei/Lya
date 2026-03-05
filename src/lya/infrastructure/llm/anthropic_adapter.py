"""Anthropic Claude LLM Adapter.

Implements LLM Port for Anthropic Claude API.
"""

from __future__ import annotations

from typing import Any, AsyncIterator
import json

from lya.infrastructure.config.settings import settings
from lya.infrastructure.config.logging import get_logger
from lya.application.ports.outgoing.llm_port import LLMPort

logger = get_logger(__name__)


class AnthropicAdapter(LLMPort):
    """
    Adapter for Anthropic Claude API.

    Supports:
    - Text generation (messages API)
    - Streaming responses
    - Structured JSON output
    """

    DEFAULT_MODEL = "claude-3-haiku-20240307"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or self.DEFAULT_MODEL
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required. Run: pip install anthropic"
                )
        return self._client

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text using Anthropic Claude."""
        client = self._get_client()

        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 1024,
                temperature=temperature or 0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        except Exception as e:
            logger.error("Anthropic generation failed", error=str(e))
            raise RuntimeError(f"Anthropic generation failed: {e}") from e

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Chat completion using Anthropic API."""
        client = self._get_client()

        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Anthropic only uses 'user' and 'assistant'
            if role in ["user", "assistant"]:
                anthropic_messages.append({"role": role, "content": content})

        # Extract system message if present
        system = None
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content")
                break

        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=max_tokens or 1024,
                temperature=temperature or 0.7,
                system=system,
                messages=anthropic_messages,
            )
            return response.content[0].text

        except Exception as e:
            logger.error("Anthropic chat failed", error=str(e))
            raise RuntimeError(f"Anthropic chat failed: {e}") from e

    async def generate_structured(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate structured JSON output."""
        # Add schema to prompt
        schema_prompt = f"""{prompt}

You must respond with valid JSON that matches this schema:
{json.dumps(output_schema, indent=2)}

Respond ONLY with the JSON object, no other text."""

        response = await self.generate(
            prompt=schema_prompt,
            temperature=temperature or 0.1,
        )

        try:
            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nResponse: {response}") from e

    async def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream generated text tokens."""
        client = self._get_client()

        try:
            async with client.messages.stream(
                model=self.model,
                max_tokens=1024,
                temperature=temperature or 0.7,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error("Anthropic streaming failed", error=str(e))
            raise RuntimeError(f"Streaming failed: {e}") from e

    async def embed(self, text: str) -> list[float]:
        """Anthropic doesn't provide embeddings natively."""
        raise NotImplementedError(
            "Anthropic does not provide embeddings. "
            "Use OpenAI or Ollama for embeddings."
        )

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Anthropic doesn't provide embeddings natively."""
        raise NotImplementedError(
            "Anthropic does not provide embeddings. "
            "Use OpenAI or Ollama for embeddings."
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the configured model."""
        return {
            "provider": "anthropic",
            "model": self.model,
            "api_key_configured": bool(self.api_key),
        }

    async def list_models(self) -> list[dict[str, Any]]:
        """List available Anthropic models."""
        return [
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
            {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet"},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
        ]

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.close()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
