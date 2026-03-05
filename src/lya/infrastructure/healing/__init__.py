"""Self-healing system for Lya.

This package provides automatic error detection and recovery capabilities.
"""

from lya.infrastructure.healing.self_healing import (
    SelfHealing,
    HealingStrategy,
    RetryStrategy,
    FallbackStrategy,
    CircuitBreakerStrategy,
    CircuitBreaker,
    CircuitState,
    HealingStrategyConfig,
    OllamaHealingStrategy,
    MemoryHealingStrategy,
    DiskHealingStrategy,
    DatabaseHealingStrategy,
    CacheHealingStrategy,
)

from lya.infrastructure.healing.healing_system import (
    HealingSystem,
    HealingStrategy as SystemHealingStrategy,
    HealingPolicy,
    HealingMetrics,
    OllamaRestartStrategy,
    MemoryCleanupStrategy,
    DiskCleanupStrategy,
    TaskRestartStrategy,
    CacheResetStrategy,
    DatabaseReconnectStrategy,
)

from lya.infrastructure.healing.self_healing_system import (
    SelfHealingSystem,
    CircuitBreakerState,
    CircuitBreakerConfig,
    RetryConfig,
)

__all__ = [
    # Main systems
    "SelfHealing",
    "HealingSystem",
    "SelfHealingSystem",
    # Base classes
    "HealingStrategy",
    "SystemHealingStrategy",
    # Strategy implementations
    "RetryStrategy",
    "FallbackStrategy",
    "CircuitBreakerStrategy",
    "OllamaHealingStrategy",
    "MemoryHealingStrategy",
    "DiskHealingStrategy",
    "DatabaseHealingStrategy",
    "CacheHealingStrategy",
    "OllamaRestartStrategy",
    "MemoryCleanupStrategy",
    "DiskCleanupStrategy",
    "TaskRestartStrategy",
    "CacheResetStrategy",
    "DatabaseReconnectStrategy",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerState",
    "CircuitState",
    # Configuration
    "HealingStrategyConfig",
    "CircuitBreakerConfig",
    "RetryConfig",
    "HealingPolicy",
    "HealingMetrics",
]
