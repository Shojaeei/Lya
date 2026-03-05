"""Outgoing port for memory operations."""

from typing import Any, Protocol
from uuid import UUID

from lya.domain.models.memory import Memory, MemoryContext, MemoryType, MemoryImportance


class MemoryPort(Protocol):
    """
    Outgoing port for memory operations.

    This port abstracts memory storage operations from the application layer.
    Infrastructure adapters implement this protocol.
    """

    async def store(
        self,
        content: str,
        memory_type: MemoryType,
        importance: MemoryImportance,
        agent_id: UUID,
        context: MemoryContext | None = None,
    ) -> Memory:
        """Store a new memory."""
        ...

    async def recall(
        self,
        query: str,
        agent_id: UUID | None = None,
        limit: int = 10,
        memory_type: MemoryType | None = None,
    ) -> list[Memory]:
        """Recall memories by semantic similarity."""
        ...

    async def get_by_goal(
        self,
        goal_id: UUID,
        agent_id: UUID | None = None,
    ) -> list[Memory]:
        """Get memories associated with a goal."""
        ...

    async def consolidate_memories(
        self,
        agent_id: UUID,
        memory_ids: list[UUID],
    ) -> Memory:
        """Consolidate multiple memories into one."""
        ...

    async def forget_old_memories(
        self,
        agent_id: UUID,
        threshold_days: int = 30,
    ) -> int:
        """Remove stale memories."""
        ...

    async def get_stats(self, agent_id: UUID) -> dict[str, Any]:
        """Get memory statistics."""
        ...
