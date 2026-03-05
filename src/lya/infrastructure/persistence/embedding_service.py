"""Embedding service for generating text embeddings."""

from __future__ import annotations

from typing import Any, Protocol

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """
        Generate embeddings for text.

        Args:
            text: Text or list of texts to embed

        Returns:
            Embedding vector(s)
        """
        ...

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        ...


class OllamaEmbeddingProvider:
    """Embedding provider using Ollama."""

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_DIMENSION = 768

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.base_url = base_url or settings.llm.base_url
        self.model = model or settings.memory.embedding_model
        self._dimension = self.DEFAULT_DIMENSION

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Generate embeddings using Ollama API."""
        import httpx

        texts = [text] if isinstance(text, str) else text

        async with httpx.AsyncClient(timeout=30.0) as client:
            embeddings = []

            for t in texts:
                headers = {}
                api_key = getattr(settings.llm, "ollama_api_key", None)
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": t,
                    },
                    headers=headers,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"Ollama embedding failed: {response.status_code}")

                result = response.json()
                embeddings.append(result["embedding"])

            return embeddings[0] if isinstance(text, str) else embeddings


class SentenceTransformerProvider:
    """Embedding provider using sentence-transformers."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384

    def __init__(self, model: str | None = None):
        self.model_name = model or getattr(settings.memory, "embedding_model", self.DEFAULT_MODEL)
        self._model = None
        self._dimension = self.DEFAULT_DIMENSION

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info("Loaded embedding model", model=self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Run: pip install sentence-transformers"
                )

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Generate embeddings using sentence-transformers."""
        import asyncio

        self._load_model()

        # Run in thread pool since sentence-transformers is CPU-bound
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(text, convert_to_numpy=True).tolist(),
        )

        return embeddings if isinstance(text, str) else embeddings


class OpenAIEmbeddingProvider:
    """Embedding provider using OpenAI API."""

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSION = 1536

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or getattr(settings.llm, "openai_api_key", None)
        self.model = model or self.DEFAULT_MODEL
        self._dimension = self.DEFAULT_DIMENSION

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai required. Run: pip install openai")

        client = openai.AsyncOpenAI(api_key=self.api_key)

        texts = [text] if isinstance(text, str) else text

        response = await client.embeddings.create(
            model=self.model,
            input=texts,
        )

        embeddings = [item.embedding for item in response.data]
        return embeddings[0] if isinstance(text, str) else embeddings


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self, provider: EmbeddingProvider | None = None):
        self._provider = provider

    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        if self._provider is None:
            raise RuntimeError("No embedding provider configured")

        return await self._provider.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        if self._provider is None:
            raise RuntimeError("No embedding provider configured")

        return await self._provider.embed(texts)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._provider is None:
            return 384  # Default dimension
        return self._provider.dimension

    @classmethod
    def create(cls, provider_type: str = "auto") -> EmbeddingService:
        """Create embedding service with appropriate provider."""
        provider: EmbeddingProvider | None = None

        if provider_type == "ollama":
            provider = OllamaEmbeddingProvider()
        elif provider_type == "sentence-transformers":
            provider = SentenceTransformerProvider()
        elif provider_type == "openai":
            provider = OpenAIEmbeddingProvider()
        elif provider_type == "auto":
            # Try providers in order
            try:
                provider = OllamaEmbeddingProvider()
                logger.info("Using Ollama embedding provider")
            except Exception:
                try:
                    provider = SentenceTransformerProvider()
                    logger.info("Using sentence-transformers embedding provider")
                except Exception:
                    logger.warning("No embedding provider available")
                    provider = None

        return cls(provider)


# Global embedding service instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service."""
    global _embedding_service
    if _embedding_service is None:
        # Use ollama as the embedding provider
        _embedding_service = EmbeddingService.create("ollama")
    return _embedding_service


def set_embedding_service(service: EmbeddingService) -> None:
    """Set the global embedding service."""
    global _embedding_service
    _embedding_service = service
