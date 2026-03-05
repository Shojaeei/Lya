"""Goal repository interface."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from lya.domain.models.goal import Goal, GoalStatus


class GoalRepository(ABC):
    """
    Repository interface for Goal aggregate.

    This interface defines the contract for goal persistence.
    Implementations can use any storage backend.
    """

    @abstractmethod
    async def get(self, goal_id: UUID) -> Goal | None:
        """
        Retrieve a goal by ID.

        Args:
            goal_id: Unique identifier of the goal

        Returns:
            Goal if found, None otherwise
        """
        pass

    @abstractmethod
    async def save(self, goal: Goal) -> None:
        """
        Save a goal.

        Args:
            goal: Goal to save
        """
        pass

    @abstractmethod
    async def delete(self, goal_id: UUID) -> bool:
        """
        Delete a goal.

        Args:
            goal_id: ID of goal to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def get_by_agent(
        self,
        agent_id: UUID,
        status: GoalStatus | None = None,
        limit: int = 100,
    ) -> list[Goal]:
        """
        Get goals for a specific agent.

        Args:
            agent_id: Agent ID
            status: Optional status filter
            limit: Maximum results

        Returns:
            List of goals
        """
        pass

    @abstractmethod
    async def get_active_goals(self, agent_id: UUID) -> list[Goal]:
        """
        Get active (non-terminal) goals for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of active goals ordered by priority
        """
        pass

    @abstractmethod
    async def get_pending_goals(
        self,
        agent_id: UUID,
        limit: int = 10,
    ) -> list[Goal]:
        """
        Get pending goals ordered by priority.

        Args:
            agent_id: Agent ID
            limit: Maximum results

        Returns:
            List of pending goals
        """
        pass

    @abstractmethod
    async def get_sub_goals(self, parent_goal_id: UUID) -> list[Goal]:
        """
        Get sub-goals of a parent goal.

        Args:
            parent_goal_id: Parent goal ID

        Returns:
            List of sub-goals
        """
        pass

    @abstractmethod
    async def update_status(
        self,
        goal_id: UUID,
        status: GoalStatus,
    ) -> bool:
        """
        Update goal status.

        Args:
            goal_id: Goal ID
            status: New status

        Returns:
            True if updated, False if not found
        """
        pass

    @abstractmethod
    async def count(
        self,
        agent_id: UUID | None = None,
        status: GoalStatus | None = None,
    ) -> int:
        """
        Count goals.

        Args:
            agent_id: Optional agent filter
            status: Optional status filter

        Returns:
            Total count
        """
        pass
