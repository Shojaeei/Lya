"""LLM Provider Factory."""

from __future__ import annotations

from typing import Literal

from lya.infrastructure.config.settings import settings
from lya.infrastructure.config.logging import get_logger
from lya.application.ports.outgoing.llm_port import LLMPort

logger = get_logger(__name__)

ProviderType = Literal["ollama", "openai", "anthropic"]


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _providers: dict[str, type[LLMPort]] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[LLMPort]) -> None:
        """Register a provider class."""
        cls._providers[name] = provider_class
        logger.debug("Registered LLM provider", provider=name)

    @classmethod
    def create(
        cls,
        provider_type: ProviderType | None = None,
        **kwargs,
    ) -> LLMPort:
        """
        Create an LLM provider instance.

        Args:
            provider_type: Type of provider (ollama, openai, anthropic)
            **kwargs: Provider-specific configuration

        Returns:
            LLM provider instance
        """
        # Use configured provider if not specified
        provider_type = provider_type or settings.llm.provider

        # Lazy imports to avoid circular dependencies
        if provider_type == "ollama":
            from .ollama_adapter import OllamaAdapter
            return OllamaAdapter(**kwargs)
        elif provider_type == "openai":
            from .openai_adapter import OpenAIAdapter
            return OpenAIAdapter(**kwargs)
        elif provider_type == "anthropic":
            from .anthropic_adapter import AnthropicAdapter
            return AnthropicAdapter(**kwargs)
        else:
            raise ValueError(f"Unknown LLM provider: {provider_type}")

    @classmethod
    def available_providers(cls) -> list[str]:
        """Get list of available provider types."""
        return ["ollama", "openai", "anthropic"]

    @classmethod
    def get_default_provider(cls) -> str:
        """Get the default provider from settings."""
        return settings.llm.provider


# Convenience function
def create_llm_provider(
    provider_type: ProviderType | None = None,
    **kwargs,
) -> LLMPort:
    """Create an LLM provider instance."""
    return LLMProviderFactory.create(provider_type, **kwargs)
