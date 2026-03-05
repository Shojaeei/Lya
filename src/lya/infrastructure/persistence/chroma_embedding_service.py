"""ChromaDB embedding service.

Provides text embedding generation using various embedding models.
Supports local models via sentence-transformers and API-based models.
"""

from __future__ import annotations

from typing import Any

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


class ChromaEmbeddingService:
    """
    Service for generating text embeddings.

    Supports multiple embedding providers:
    - sentence-transformers (local)
    - OpenAI API
    - Ollama embeddings
    """

    # Default embedding model
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None, provider: str | None = None):
        """
        Initialize embedding service.

        Args:
            model_name: Name of the embedding model
            provider: Embedding provider (sentence_transformers, openai, ollama)
        """
        self.model_name = model_name or settings.get("embedding_model", self.DEFAULT_MODEL)
        self.provider = provider or settings.get("embedding_provider", "sentence_transformers")
        self._embedding_model: Any = None
        self._ollama_client = None
        self._openai_client = None
        self._dimension = 384  # Default for all-MiniLM-L6-v2

    async def __aenter__(self) -> ChromaEmbeddingService:
        """Async context manager entry."""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        pass  # No cleanup needed

    async def _initialize(self) -> None:
        """Initialize the embedding model."""
        try:
            if self.provider == "sentence_transformers":
                await self._init_sentence_transformers()
            elif self.provider == "openai":
                await self._init_openai()
            elif self.provider == "ollama":
                await self._init_ollama()
            else:
                raise ValueError(f"Unknown embedding provider: {self.provider}")

            logger.info(
                "Embedding service initialized",
                provider=self.provider,
                model=self.model_name,
            )
        except Exception as e:
            logger.error(
                "Failed to initialize embedding service",
                error=str(e),
                provider=self.provider,
            )
            raise

    async def _init_sentence_transformers(self) -> None:
        """Initialize sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.model_name)
            self._dimension = self._embedding_model.get_sentence_embedding_dimension()

            logger.info(
                "SentenceTransformer model loaded",
                model=self.model_name,
                dimension=self._dimension,
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Run: pip install sentence-transformers"
            )

    async def _init_openai(self) -> None:
        """Initialize OpenAI embedding client."""
        try:
            import openai

            api_key = settings.get("openai_api_key")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")

            self._openai_client = openai.AsyncOpenAI(api_key=api_key)

            # Map model names
            model_map = {
                "all-MiniLM-L6-v2": "text-embedding-3-small",
                "large": "text-embedding-3-large",
            }
            self.model_name = model_map.get(self.model_name, self.model_name)
            self._dimension = 1536 if "3-small" in self.model_name else 3072

            logger.info(
                "OpenAI embedding client initialized",
                model=self.model_name,
            )
        except ImportError:
            raise ImportError("openai required. Run: pip install openai")

    async def _init_ollama(self) -> None:
        """Initialize Ollama embedding client."""
        try:
            import httpx

            self._ollama_client = httpx.AsyncClient(
                base_url=settings.get("ollama_base_url", "http://localhost:11434"),
                timeout=60.0,
            )

            # Ollama uses different embedding models
            if "nomic" not in self.model_name:
                self.model_name = "nomic-embed-text"

            self._dimension = 768  # nomic-embed-text dimension

            logger.info(
                "Ollama embedding client initialized",
                base_url=settings.get("ollama_base_url", "http://localhost:11434"),
            )
        except ImportError:
            raise ImportError("httpx required")

    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not text.strip():
            return [0.0] * self._dimension

        try:
            if self.provider == "sentence_transformers":
                return await self._embed_sentence_transformers(text)
            elif self.provider == "openai":
                return await self._embed_openai(text)
            elif self.provider == "ollama":
                return await self._embed_ollama(text)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e), text_preview=text[:50])
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            if self.provider == "sentence_transformers":
                return await self._embed_batch_sentence_transformers(texts)
            elif self.provider == "openai":
                return await self._embed_batch_openai(texts)
            elif self.provider == "ollama":
                return await self._embed_batch_ollama(texts)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        except Exception as e:
            logger.error("Batch embedding failed", error=str(e), count=len(texts))
            raise

    async def _embed_sentence_transformers(self, text: str) -> list[float]:
        """Embed using sentence-transformers."""
        import numpy as np

        # Run in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._embedding_model.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()

    async def _embed_batch_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        """Batch embed using sentence-transformers."""
        import numpy as np
        import asyncio

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._embedding_model.encode(texts, convert_to_numpy=True)
        )
        return [emb.tolist() for emb in embeddings]

    async def _embed_openai(self, text: str) -> list[float]:
        """Embed using OpenAI API."""
        response = await self._openai_client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding

    async def _embed_batch_openai(self, texts: list[str]) -> list[list[float]]:
        """Batch embed using OpenAI API."""
        response = await self._openai_client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [data.embedding for data in response.data]

    async def _embed_ollama(self, text: str) -> list[float]:
        """Embed using Ollama."""
        response = await self._ollama_client.post(
            "/api/embeddings",
            json={
                "model": self.model_name,
                "prompt": text,
            },
        )
        response.raise_for_status()
        result = response.json()
        return result["embedding"]

    async def _embed_batch_ollama(self, texts: list[str]) -> list[list[float]]:
        """Batch embed using Ollama."""
        # Ollama doesn't support batch natively, so do sequentially
        embeddings = []
        for text in texts:
            emb = await self._embed_ollama(text)
            embeddings.append(emb)
        return embeddings

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            a: First embedding
            b: Second embedding

        Returns:
            Similarity score (-1 to 1)
        """
        import math

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
