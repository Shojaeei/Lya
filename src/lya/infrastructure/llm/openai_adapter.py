"""OpenAI LLM Adapter.

Implements LLM Port for OpenAI API.
"""

from __future__ import annotations

from typing import Any, AsyncIterator
import json

from lya.infrastructure.config.settings import settings
from lya.infrastructure.config.logging import get_logger
from lya.application.ports.outgoing.llm_port import LLMPort

logger = get_logger(__name__)


class OpenAIAdapter(LLMPort):
    """
    Adapter for OpenAI API.

    Supports:
    - Text generation (chat completions)
    - Streaming responses
    - Structured JSON output
    - Embeddings
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.openai_api_key
        self.model = model or self.DEFAULT_MODEL
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required. Run: pip install openai"
                )
        return self._client

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text using OpenAI chat completions."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Chat completion using OpenAI API."""
        client = self._get_client()

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or 0.7,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error("OpenAI chat failed", error=str(e))
            raise RuntimeError(f"OpenAI chat failed: {e}") from e

    async def generate_structured(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate structured JSON output using function calling."""
        client = self._get_client()

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature or 0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            return json.loads(content)

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}") from e
        except Exception as e:
            logger.error("OpenAI structured generation failed", error=str(e))
            raise

    async def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream generated text tokens."""
        client = self._get_client()

        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature or 0.7,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("OpenAI streaming failed", error=str(e))
            raise RuntimeError(f"Streaming failed: {e}") from e

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using OpenAI."""
        client = self._get_client()

        try:
            response = await client.embeddings.create(
                model=self.DEFAULT_EMBEDDING_MODEL,
                input=text,
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error("OpenAI embedding failed", error=str(e))
            raise RuntimeError(f"Embedding failed: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        client = self._get_client()

        try:
            response = await client.embeddings.create(
                model=self.DEFAULT_EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error("OpenAI batch embedding failed", error=str(e))
            raise RuntimeError(f"Batch embedding failed: {e}") from e

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the configured model."""
        return {
            "provider": "openai",
            "model": self.model,
            "api_key_configured": bool(self.api_key),
        }

    async def list_models(self) -> list[dict[str, Any]]:
        """List available OpenAI models."""
        # OpenAI has a fixed list of models
        return [
            {"id": "gpt-4o", "name": "GPT-4o"},
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
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
