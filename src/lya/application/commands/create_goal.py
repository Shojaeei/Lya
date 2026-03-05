"""Create Goal Command Handler.

Handles the creation of new goals, including validation,
persistence, and event publishing.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from lya.domain.models.goal import Goal, GoalPriority, GoalStatus
from lya.domain.models.events import (
    GoalCreated,
    EventPublisher,
)
from lya.domain.repositories.goal_repo import GoalRepository
from lya.domain.services.planning_service import PlanningService
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CreateGoalRequest:
    """Request to create a goal."""
    description: str
    agent_id: UUID
    priority: int = 3  # 1-5, lower is higher priority
    parent_goal_id: UUID | None = None
    context: str | None = None  # Additional context for planning


@dataclass
class CreateGoalResult:
    """Result of creating a goal."""
    goal_id: UUID
    success: bool
    message: str
    plan_generated: bool = False
    sub_goals_created: int = 0


class CreateGoalHandler:
    """
    Command handler for creating goals.

    Responsibilities:
    - Validate input
    - Create Goal entity
    - Generate initial plan (if enabled)
    - Persist goal
    - Publish GoalCreated event
    - Optionally create sub-goals
    """

    def __init__(
        self,
        goal_repository: GoalRepository,
        event_publisher: EventPublisher,
        planning_service: PlanningService | None = None,
    ) -> None:
        self._repo = goal_repository
        self._events = event_publisher
        self._planning = planning_service

    async def execute(self, request: CreateGoalRequest) -> CreateGoalResult:
        """
        Execute the create goal command.

        Args:
            request: Goal creation request

        Returns:
            CreateGoalResult with goal ID and status
        """
        try:
            logger.info(
                "Creating goal",
                description=request.description,
                agent_id=str(request.agent_id),
                priority=request.priority,
            )

            # Validate input
            if not request.description or len(request.description.strip()) < 3:
                return CreateGoalResult(
                    goal_id=UUID(int=0),
                    success=False,
                    message="Description must be at least 3 characters",
                )

            if not 1 <= request.priority <= 5:
                return CreateGoalResult(
                    goal_id=UUID(int=0),
                    success=False,
                    message="Priority must be between 1 (highest) and 5 (lowest)",
                )

            # Create priority enum
            priority_map = {
                1: GoalPriority.CRITICAL,
                2: GoalPriority.HIGH,
                3: GoalPriority.MEDIUM,
                4: GoalPriority.LOW,
                5: GoalPriority.TRIVIAL,
            }
            priority = priority_map.get(request.priority, GoalPriority.MEDIUM)

            # Create goal entity
            goal = Goal(
                agent_id=request.agent_id,
                description=request.description.strip(),
                priority=priority,
                parent_goal_id=request.parent_goal_id,
            )

            # Generate plan if planning service available
            plan_generated = False
            if self._planning:
                try:
                    plan = await self._planning.create_plan(
                        goal=goal,
                        available_tools=[],  # TODO: Get from tool registry
                        context=request.context,
                    )
                    goal.set_plan(plan)
                    plan_generated = True
                    logger.info("Plan generated for goal", goal_id=str(goal.id))
                except Exception as e:
                    logger.warning(
                        "Failed to generate plan",
                        goal_id=str(goal.id),
                        error=str(e),
                    )

            # Persist goal
            await self._repo.save(goal)
            logger.info("Goal saved to repository", goal_id=str(goal.id))

            # Create sub-goals if complex goal (optional)
            sub_goals_created = 0
            if self._planning and request.context and "complex" in request.context.lower():
                try:
                    sub_goals = await self._planning.decompose_goal(goal, max_depth=2)
                    for sub_goal in sub_goals:
                        await self._repo.save(sub_goal)
                        goal.add_sub_goal(sub_goal.id)
                        sub_goals_created += 1

                    # Update parent goal with sub-goals
                    await self._repo.save(goal)
                    logger.info(
                        "Created sub-goals",
                        parent_goal_id=str(goal.id),
                        count=sub_goals_created,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to decompose goal",
                        goal_id=str(goal.id),
                        error=str(e),
                    )

            # Publish event
            await self._events.publish(
                GoalCreated(
                    goal_id=str(goal.id),
                    agent_id=str(goal.agent_id),
                    description=goal.description,
                    priority=goal.priority_value,
                )
            )

            logger.info(
                "Goal created successfully",
                goal_id=str(goal.id),
                description=goal.description,
            )

            return CreateGoalResult(
                goal_id=goal.id,
                success=True,
                message=f"Goal created with {len(goal.plan.steps) if goal.plan else 0} steps" if plan_generated else "Goal created",
                plan_generated=plan_generated,
                sub_goals_created=sub_goals_created,
            )

        except Exception as e:
            logger.error(
                "Failed to create goal",
                error=str(e),
                description=request.description,
            )
            return CreateGoalResult(
                goal_id=UUID(int=0),
                success=False,
                message=f"Internal error: {str(e)}",
            )


# Convenience function for direct use
async def create_goal(
    description: str,
    agent_id: UUID,
    priority: int = 3,
    parent_goal_id: UUID | None = None,
    goal_repo: GoalRepository | None = None,
    event_bus: EventPublisher | None = None,
    planning_service: PlanningService | None = None,
) -> CreateGoalResult:
    """
    Convenience function to create a goal.

    Args:
        description: Goal description
        agent_id: Owning agent ID
        priority: Priority 1-5
        parent_goal_id: Optional parent goal
        goal_repo: Repository (injected or None for default)
        event_bus: Event publisher (injected or None)
        planning_service: Optional planning service

    Returns:
        CreateGoalResult
    """
    request = CreateGoalRequest(
        description=description,
        agent_id=agent_id,
        priority=priority,
        parent_goal_id=parent_goal_id,
    )

    # Note: In real usage, dependencies would be injected
    handler = CreateGoalHandler(
        goal_repository=goal_repo,  # type: ignore
        event_publisher=event_bus,  # type: ignore
        planning_service=planning_service,
    )

    return await handler.execute(request)
