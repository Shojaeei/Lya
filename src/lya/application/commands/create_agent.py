"""Application layer command handlers."""

from typing import TYPE_CHECKING, Any
from uuid import UUID

from lya.domain.models.agent import Agent, AgentConfig
from lya.domain.models.goal import Goal, GoalPriority
from lya.domain.repositories.goal_repo import GoalRepository
from lya.domain.repositories.task_repo import TaskRepository

if TYPE_CHECKING:
    from lya.application.ports.outgoing import MemoryPort, LLMPort, ToolPort


class CreateAgentHandler:
    """Handler for creating agents."""

    def __init__(self) -> None:
        self._agents: dict[UUID, Agent] = {}

    async def execute(
        self,
        name: str,
        config: AgentConfig | None = None,
    ) -> UUID:
        """Create a new agent."""
        agent_config = config or AgentConfig()
        agent_config.name = name

        agent = Agent(config=agent_config)
        self._agents[agent.id] = agent

        return agent.id

    def get_agent(self, agent_id: UUID) -> Agent | None:
        """Get agent by ID."""
        return self._agents.get(agent_id)


class AddGoalHandler:
    """Handler for adding goals."""

    def __init__(
        self,
        goal_repo: GoalRepository,
        agent_handler: CreateAgentHandler,
    ) -> None:
        self._goal_repo = goal_repo
        self._agent_handler = agent_handler

    async def execute(
        self,
        agent_id: UUID,
        description: str,
        priority: int = 3,
        parent_goal_id: UUID | None = None,
    ) -> UUID:
        """Add a goal to an agent."""
        agent = self._agent_handler.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        goal = Goal(
            agent_id=agent_id,
            description=description,
            priority=GoalPriority(priority),
            parent_goal_id=parent_goal_id,
        )

        await self._goal_repo.save(goal)
        agent.add_goal(goal.id)

        return goal.id


class CancelGoalHandler:
    """Handler for cancelling goals."""

    def __init__(self, goal_repo: GoalRepository) -> None:
        self._goal_repo = goal_repo

    async def execute(
        self,
        goal_id: UUID,
        reason: str = "",
    ) -> None:
        """Cancel a goal."""
        goal = await self._goal_repo.get(goal_id)
        if not goal:
            raise ValueError(f"Goal {goal_id} not found")

        goal.cancel(reason)
        await self._goal_repo.save(goal)
