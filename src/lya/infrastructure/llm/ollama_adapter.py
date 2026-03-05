"""Ollama LLM Adapter.

Implements the LLM Port for Ollama local LLM inference.
"""

from typing import Any, AsyncIterator
import httpx
import json
from dataclasses import dataclass

from lya.infrastructure.config.settings import settings
from lya.infrastructure.config.logging import get_logger
from lya.application.ports.outgoing.llm_port import LLMPort

logger = get_logger(__name__)


@dataclass
class OllamaMessage:
    """Ollama chat message format."""
    role: str
    content: str


class OllamaAdapter(LLMPort):
    """
    Adapter for Ollama local LLM API.

    Supports:
    - Text generation (generate)
    - Chat completion (chat)
    - Embeddings (embed)
    - Streaming responses
    """

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None) -> None:
        self.base_url = base_url or settings.llm.base_url
        self.model = model or settings.llm.model
        self.timeout = settings.llm.timeout
        self.api_key = api_key or settings.llm.ollama_api_key
        self._client: httpx.AsyncClient | None = None
        logger.info(
            "OllamaAdapter initialized",
            base_url=self.base_url,
            model=self.model,
            has_api_key=bool(self.api_key),
        )

        logger.info(
            "OllamaAdapter initialized",
            base_url=self.base_url,
            model=self.model,
            has_api_key=bool(self.api_key),
            timeout=self.timeout,
        )

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authorization if API key is set."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            logger.debug("Using API key authentication")
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate text using Ollama generate endpoint.

        Args:
            prompt: The prompt to generate from
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt

        Returns:
            Generated text
        """
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {},
        }

        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if system_prompt:
            payload["system"] = system_prompt

        try:
            logger.debug("Sending request to Ollama", model=self.model)
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers=self._get_headers(),
            )
            response.raise_for_status()

            data = response.json()
            result = data.get("response", "").strip()
            logger.debug("Received response from Ollama", length=len(result))
            return result

        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Error: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.text}") from e

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> str:
        """
        Chat completion using Ollama chat endpoint.

        Args:
            messages: List of messages with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            images: Optional list of base64-encoded images for vision models

        Returns:
            Assistant's response
        """
        client = await self._get_client()

        # If images provided, add them to the last user message
        chat_messages = list(messages)
        if images and chat_messages:
            for i in range(len(chat_messages) - 1, -1, -1):
                if chat_messages[i].get("role") == "user":
                    chat_messages[i] = {
                        **chat_messages[i],
                        "images": images,
                    }
                    break

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "stream": False,
            "options": {},
        }

        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        try:
            logger.debug("Sending chat request", model=self.model, message_count=len(messages))
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers=self._get_headers(),
            )
            response.raise_for_status()

            data = response.json()
            result = data.get("message", {}).get("content", "").strip()
            logger.debug("Chat response received", response_length=len(result))
            return result

        except httpx.ConnectError as e:
            logger.error("Ollama connection failed for chat", error=str(e))
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Error: {e}"
            ) from e
        except httpx.HTTPStatusError as e:
            logger.error("Ollama chat API error", status=e.response.status_code, error=e.response.text)
            raise RuntimeError(f"Ollama API error: {e.response.text}") from e
        except Exception as e:
            logger.error("Chat request failed", error=str(e))
            raise RuntimeError(f"Chat failed: {e}") from e

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream chat completion tokens.

        Args:
            messages: List of messages with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Yields:
            Token chunks as they're generated
        """
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {},
        }

        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                headers=self._get_headers(),
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            msg = data.get("message", {})
                            if chunk := msg.get("content"):
                                yield chunk
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            raise RuntimeError(f"Chat streaming failed: {e}") from e

    def with_model(self, model_name: str) -> "OllamaAdapter":
        """Create a new adapter with a different model but same connection."""
        return OllamaAdapter(
            base_url=self.base_url,
            model=model_name,
            api_key=self.api_key,
        )

    async def generate_structured(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Generate structured JSON output.

        Args:
            prompt: The prompt
            output_schema: JSON schema for expected output
            temperature: Sampling temperature

        Returns:
            Parsed JSON response
        """
        # Add schema to prompt
        schema_prompt = f"""{prompt}

You must respond with valid JSON that matches this schema:
{json.dumps(output_schema, indent=2)}

Respond ONLY with the JSON object, no other text."""

        response = await self.generate(
            prompt=schema_prompt,
            temperature=temperature or 0.1,  # Low temp for structured output
        )

        try:
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response}") from e

    async def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream generated text tokens.

        Args:
            prompt: The prompt
            temperature: Sampling temperature

        Yields:
            Token chunks as they're generated
        """
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {},
        }

        if temperature is not None:
            payload["options"]["temperature"] = temperature

        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
                headers=self._get_headers(),
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if chunk := data.get("response"):
                                yield chunk
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            raise RuntimeError(f"Streaming failed: {e}") from e

    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        client = await self._get_client()

        # Use nomic-embed-text or similar embedding model
        embed_model = settings.memory.embedding_model

        try:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": embed_model,
                    "prompt": text,
                },
                headers=self._get_headers(),
            )
            response.raise_for_status()

            data = response.json()
            return data.get("embedding", [])

        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = await self.embed(text)
            embeddings.append(embedding)
        return embeddings

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the configured model."""
        return {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "embedding_model": settings.memory.embedding_model,
            "has_api_key": bool(self.api_key),
        }

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models from Ollama."""
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.base_url}/api/tags",
                headers=self._get_headers(),
            )
            response.raise_for_status()

            data = response.json()
            return data.get("models", [])

        except Exception as e:
            raise ConnectionError(f"Failed to list models: {e}") from e

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
