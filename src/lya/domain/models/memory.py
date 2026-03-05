"""Memory domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any
from uuid import UUID, uuid4


class MemoryType(Enum):
    """Types of memories."""
    EPISODIC = auto()      # Specific experiences/events
    SEMANTIC = auto()      # Facts and knowledge
    PROCEDURAL = auto()    # How to do things
    REFLECTIVE = auto()    # Insights and learnings
    CONVERSATION = auto()  # Dialogue history
    OBSERVATION = auto()   # Observed facts


class MemoryImportance(Enum):
    """Importance levels for memories."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1


@dataclass
class MemoryContext:
    """Context information for a memory."""
    goal_id: UUID | None = None
    task_id: UUID | None = None
    source: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": str(self.goal_id) if self.goal_id else None,
            "task_id": str(self.task_id) if self.task_id else None,
            "source": self.source,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryContext:
        return cls(
            goal_id=UUID(data["goal_id"]) if data.get("goal_id") else None,
            task_id=UUID(data["task_id"]) if data.get("task_id") else None,
            source=data.get("source"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


class Memory:
    """
    Represents a stored memory for the agent.

    Memories are the foundation of learning and experience retention.
    """

    def __init__(
        self,
        memory_id: UUID | None = None,
        agent_id: UUID | None = None,
        content: str = "",
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        embedding: list[float] | None = None,
        context: MemoryContext | None = None,
    ):
        self._id = memory_id or uuid4()
        self._agent_id = agent_id
        self._content = content
        self._type = memory_type
        self._importance = importance
        self._embedding = embedding
        self._context = context or MemoryContext()
        self._created_at = datetime.now(timezone.utc)
        self._accessed_at: datetime | None = None
        self._access_count = 0
        self._decay_rate = 0.01  # Memory decay rate
        self._is_consolidated = False

    # ═══════════════════════════════════════════════════════════════
    # Properties
    # ═══════════════════════════════════════════════════════════════

    @property
    def id(self) -> UUID:
        """Memory unique identifier."""
        return self._id

    @property
    def agent_id(self) -> UUID | None:
        """Owning agent ID."""
        return self._agent_id

    @property
    def content(self) -> str:
        """Memory content."""
        return self._content

    @property
    def type(self) -> MemoryType:
        """Memory type."""
        return self._type

    @property
    def importance(self) -> MemoryImportance:
        """Memory importance."""
        return self._importance

    @property
    def importance_value(self) -> int:
        """Numeric importance value."""
        return self._importance.value

    @property
    def embedding(self) -> list[float] | None:
        """Vector embedding for semantic search."""
        return self._embedding

    @property
    def context(self) -> MemoryContext:
        """Memory context."""
        return self._context

    @property
    def created_at(self) -> datetime:
        """Creation timestamp."""
        return self._created_at

    @property
    def accessed_at(self) -> datetime | None:
        """Last access timestamp."""
        return self._accessed_at

    @property
    def access_count(self) -> int:
        """Number of times memory was accessed."""
        return self._access_count

    @property
    def is_consolidated(self) -> bool:
        """Whether memory has been consolidated."""
        return self._is_consolidated

    @property
    def age_hours(self) -> float:
        """Age of memory in hours."""
        return (datetime.now(timezone.utc) - self._created_at).total_seconds() / 3600

    # ═══════════════════════════════════════════════════════════════
    # Memory Operations
    # ═══════════════════════════════════════════════════════════════

    def access(self) -> None:
        """Record access to this memory."""
        self._accessed_at = datetime.now(timezone.utc)
        self._access_count += 1

    def update_content(self, new_content: str) -> None:
        """Update memory content."""
        self._content = new_content
        # Reset embedding since content changed
        self._embedding = None

    def set_embedding(self, embedding: list[float]) -> None:
        """Set vector embedding."""
        self._embedding = embedding

    def mark_consolidated(self) -> None:
        """Mark memory as consolidated."""
        self._is_consolidated = True

    def calculate_relevance_score(self, query_embedding: list[float]) -> float:
        """
        Calculate relevance score for a query.

        Uses cosine similarity weighted by importance and recency.
        """
        if not self._embedding or not query_embedding:
            return 0.0

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(self._embedding, query_embedding))
        norm_a = sum(a * a for a in self._embedding) ** 0.5
        norm_b = sum(b * b for b in query_embedding) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)

        # Apply importance weight
        importance_weight = self.importance_value / 5.0

        # Apply recency decay
        age_factor = max(0, 1 - (self.age_hours * self._decay_rate))

        # Combine factors
        score = similarity * (0.5 + 0.5 * importance_weight) * age_factor

        return max(0, min(1, score))

    def should_forget(self, threshold_days: int = 30) -> bool:
        """
        Determine if memory should be forgotten based on:
        - Low importance
        - No recent access
        - High age
        """
        if self._importance == MemoryImportance.CRITICAL:
            return False

        days_old = self.age_hours / 24

        if days_old < threshold_days:
            return False

        # Calculate forget score
        importance_factor = 1 / self.importance_value
        access_factor = 1 / (1 + self._access_count)
        age_factor = days_old / threshold_days

        forget_score = (importance_factor + access_factor + age_factor) / 3

        return forget_score > 0.7

    # ═══════════════════════════════════════════════════════════════
    # Summary Generation
    # ═══════════════════════════════════════════════════════════════

    def generate_summary(self, max_length: int = 100) -> str:
        """Generate a brief summary of the memory."""
        content = self._content[:max_length]
        if len(self._content) > max_length:
            content += "..."
        return f"[{self._type.name}] {content}"

    # ═══════════════════════════════════════════════════════════════
    # Serialization
    # ═══════════════════════════════════════════════════════════════

    def to_dict(self) -> dict[str, Any]:
        """Convert memory to dictionary."""
        return {
            "id": str(self._id),
            "agent_id": str(self._agent_id) if self._agent_id else None,
            "content": self._content,
            "type": self._type.name,
            "importance": self._importance.name,
            "embedding": self._embedding,
            "context": self._context.to_dict(),
            "created_at": self._created_at.isoformat(),
            "accessed_at": self._accessed_at.isoformat() if self._accessed_at else None,
            "access_count": self._access_count,
            "decay_rate": self._decay_rate,
            "is_consolidated": self._is_consolidated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """Create memory from dictionary."""
        memory = cls(
            memory_id=UUID(data["id"]),
            agent_id=UUID(data["agent_id"]) if data.get("agent_id") else None,
            content=data["content"],
            memory_type=MemoryType[data.get("type", "EPISODIC")],
            importance=MemoryImportance[data.get("importance", "MEDIUM")],
            embedding=data.get("embedding"),
            context=MemoryContext.from_dict(data.get("context", {})),
        )
        memory._created_at = datetime.fromisoformat(data["created_at"])
        if data.get("accessed_at"):
            memory._accessed_at = datetime.fromisoformat(data["accessed_at"])
        memory._access_count = data.get("access_count", 0)
        memory._decay_rate = data.get("decay_rate", 0.01)
        memory._is_consolidated = data.get("is_consolidated", False)
        return memory
