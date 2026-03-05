"""Incoming ports for agent commands."""

from typing import Any, Protocol
from uuid import UUID

from lya.domain.models.agent import AgentConfig


class CreateAgentCommand(Protocol):
    """Command to create a new agent."""

    async def execute(
        self,
        name: str,
        config: AgentConfig | None = None,
    ) -> UUID:
        """Create and return agent ID."""
        ...


class StartAgentCommand(Protocol):
    """Command to start an agent."""

    async def execute(self, agent_id: UUID) -> None:
        """Start the agent."""
        ...


class PauseAgentCommand(Protocol):
    """Command to pause an agent."""

    async def execute(self, agent_id: UUID) -> None:
        """Pause the agent."""
        ...


class ShutdownAgentCommand(Protocol):
    """Command to shutdown an agent."""

    async def execute(
        self,
        agent_id: UUID,
        reason: str | None = None,
    ) -> None:
        """Shutdown the agent."""
        ...


class AddGoalCommand(Protocol):
    """Command to add a goal to an agent."""

    async def execute(
        self,
        agent_id: UUID,
        description: str,
        priority: int = 3,
        parent_goal_id: UUID | None = None,
    ) -> UUID:
        """Add goal and return goal ID."""
        ...


class CancelGoalCommand(Protocol):
    """Command to cancel a goal."""

    async def execute(
        self,
        goal_id: UUID,
        reason: str = "",
    ) -> None:
        """Cancel the goal."""
        ...


class ExecuteTaskCommand(Protocol):
    """Command to execute a specific task."""

    async def execute(self, task_id: UUID) -> dict[str, Any]:
        """Execute the task and return result."""
        ...


class SelfImproveCommand(Protocol):
    """Command to trigger self-improvement."""

    async def execute(
        self,
        agent_id: UUID,
        improvement_type: str = "general",
    ) -> dict[str, Any]:
        """Trigger self-improvement and return results."""
        ...
