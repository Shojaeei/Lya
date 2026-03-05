"""Incoming ports for user queries."""

from typing import Any, Protocol
from uuid import UUID


class GetAgentStatusQuery(Protocol):
    """Query to get agent status."""

    async def execute(self, agent_id: UUID) -> dict[str, Any]:
        """Get agent status information."""
        ...


class GetGoalStatusQuery(Protocol):
    """Query to get goal status."""

    async def execute(self, goal_id: UUID) -> dict[str, Any]:
        """Get goal status and progress."""
        ...


class ListGoalsQuery(Protocol):
    """Query to list goals."""

    async def execute(
        self,
        agent_id: UUID,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List goals for an agent."""
        ...


class SearchMemoriesQuery(Protocol):
    """Query to search memories."""

    async def execute(
        self,
        query: str,
        agent_id: UUID | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memories."""
        ...


class GetAgentMetricsQuery(Protocol):
    """Query to get agent metrics."""

    async def execute(
        self,
        agent_id: UUID,
        timeframe: str = "all",
    ) -> dict[str, Any]:
        """Get agent performance metrics."""
        ...
