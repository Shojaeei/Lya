"""Infrastructure layer."""

from .config import settings, get_logger, configure_logging

# Persistence imports - wrapped for compatibility
try:
    from .persistence import ChromaMemoryRepository, EmbeddingService
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False
    ChromaMemoryRepository = None  # type: ignore
    EmbeddingService = None  # type: ignore

# LLM imports - wrapped for compatibility
try:
    from .llm import OllamaAdapter, OpenAIAdapter, AnthropicAdapter
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False
    OllamaAdapter = None  # type: ignore
    OpenAIAdapter = None  # type: ignore
    AnthropicAdapter = None  # type: ignore

__all__ = [
    "settings",
    "get_logger",
    "configure_logging",
    "ChromaMemoryRepository",
    "EmbeddingService",
    "OllamaAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
]
