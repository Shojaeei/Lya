"""Infrastructure persistence package."""

from .embedding_service import (
    EmbeddingService,
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    SentenceTransformerProvider,
    OpenAIEmbeddingProvider,
    get_embedding_service,
    set_embedding_service,
)

# ChromaDB imports - wrapped for compatibility
try:
    from .chroma_memory_repo import ChromaMemoryRepository, ChromaMemoryRepositoryFactory
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False
    ChromaMemoryRepository = None  # type: ignore
    ChromaMemoryRepositoryFactory = None  # type: ignore

__all__ = [
    "ChromaMemoryRepository",
    "ChromaMemoryRepositoryFactory",
    "EmbeddingService",
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
    "SentenceTransformerProvider",
    "OpenAIEmbeddingProvider",
    "get_embedding_service",
    "set_embedding_service",
]
