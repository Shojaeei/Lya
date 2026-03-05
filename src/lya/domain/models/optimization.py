"""Performance and Optimization Domain Models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any
from uuid import UUID, uuid4


class OptimizationType(Enum):
    """Types of optimizations."""
    MEMORY = auto()
    CPU = auto()
    CACHE = auto()
    NETWORK = auto()
    DATABASE = auto()


class OptimizationStatus(Enum):
    """Status of optimization."""
    PENDING = auto()
    APPLIED = auto()
    SUCCESS = auto()
    FAILED = auto()
    ROLLED_BACK = auto()


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    metric_id: UUID
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    tokens_per_second: float
    response_time_ms: float
    cache_hit_rate: float = 0.0
    cache_size: int = 0
    active_tasks: int = 0
    queue_depth: int = 0

    def __post_init__(self):
        if not isinstance(self.metric_id, UUID):
            self.metric_id = UUID(self.metric_id) if isinstance(self.metric_id, str) else uuid4()

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": str(self.metric_id),
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "tokens_per_second": self.tokens_per_second,
            "response_time_ms": self.response_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_size": self.cache_size,
            "active_tasks": self.active_tasks,
            "queue_depth": self.queue_depth,
        }


@dataclass
class Optimization:
    """An optimization action applied."""
    optimization_id: UUID
    optimization_type: OptimizationType
    status: OptimizationStatus
    description: str
    timestamp: datetime
    metrics_before: dict[str, Any] = field(default_factory=dict)
    metrics_after: dict[str, Any] | None = None
    improvement_percent: float = 0.0

    def __post_init__(self):
        if not isinstance(self.optimization_id, UUID):
            self.optimization_id = UUID(self.optimization_id) if isinstance(self.optimization_id, str) else uuid4()

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimization_id": str(self.optimization_id),
            "optimization_type": self.optimization_type.name,
            "status": self.status.name,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "improvement_percent": self.improvement_percent,
        }


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    access_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
