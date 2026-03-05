"""Memory repository interface."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from lya.domain.models.memory import Memory


class MemoryRepository(ABC):
    """
    Repository interface for Memory aggregate.

    This interface defines the contract for memory persistence.
    Implementations can use any storage backend.
    """

    @abstractmethod
    async def get(self, memory_id: UUID) -> Memory | None:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: Unique identifier of the memory

        Returns:
            Memory if found, None otherwise
        """
        pass

    @abstractmethod
    async def save(self, memory: Memory) -> None:
        """
        Save a memory.

        Args:
            memory: Memory to save
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: UUID) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
        memory_type: str | None = None,
        agent_id: UUID | None = None,
    ) -> list[tuple[Memory, float]]:
        """
        Search memories by semantic similarity.

        Args:
            query_embedding: Query vector embedding
            limit: Maximum results to return
            threshold: Minimum similarity score (0-1)
            memory_type: Filter by memory type
            agent_id: Filter by agent ID

        Returns:
            List of (memory, score) tuples ordered by relevance
        """
        pass

    @abstractmethod
    async def get_by_agent(
        self,
        agent_id: UUID,
        memory_type: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """
        Get memories for a specific agent.

        Args:
            agent_id: Agent ID
            memory_type: Optional filter by type
            limit: Maximum results

        Returns:
            List of memories
        """
        pass

    @abstractmethod
    async def get_by_goal(
        self,
        goal_id: UUID,
        limit: int = 100,
    ) -> list[Memory]:
        """
        Get memories associated with a goal.

        Args:
            goal_id: Goal ID
            limit: Maximum results

        Returns:
            List of memories
        """
        pass

    @abstractmethod
    async def delete_by_agent(self, agent_id: UUID) -> int:
        """
        Delete all memories for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Number of memories deleted
        """
        pass

    @abstractmethod
    async def get_stale_memories(
        self,
        threshold_days: int = 30,
        limit: int = 100,
    ) -> list[Memory]:
        """
        Get memories that should be forgotten.

        Args:
            threshold_days: Age threshold for staleness
            limit: Maximum results

        Returns:
            List of stale memories
        """
        pass

    @abstractmethod
    async def count(self, agent_id: UUID | None = None) -> int:
        """
        Count memories.

        Args:
            agent_id: Optional agent filter

        Returns:
            Total count
        """
        pass
