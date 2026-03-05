"""Planning service for goal decomposition and strategy generation."""

from abc import ABC, abstractmethod
from typing import Any, Protocol
from uuid import UUID

from lya.domain.models.goal import Goal, Plan
from lya.domain.models.task import Task


class LLMPort(Protocol):
    """Port for LLM interactions."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using LLM."""
        ...

    async def generate_structured(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        **kwargs: Any
    ) -> dict[str, Any]:
        """Generate structured output using LLM."""
        ...


class PlanningService:
    """
    Domain service for planning and goal decomposition.

    This service handles the high-level planning logic that doesn't
    belong to any single entity but is a cross-cutting concern.
    """

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def create_plan(
        self,
        goal: Goal,
        available_tools: list[dict[str, Any]],
        context: str | None = None,
    ) -> Plan:
        """
        Create a plan for achieving a goal.

        Args:
            goal: The goal to plan for
            available_tools: List of available tool definitions
            context: Optional context from memory

        Returns:
            A plan with steps to achieve the goal
        """
        prompt = self._build_planning_prompt(goal, available_tools, context)

        schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "tool": {"type": ["string", "null"]},
                            "parameters": {"type": "object"},
                        },
                        "required": ["description"],
                    },
                },
                "estimated_duration_seconds": {"type": "integer"},
            },
            "required": ["steps"],
        }

        try:
            response = await self._llm.generate_structured(prompt, schema)
            plan = Plan()

            for step_data in response.get("steps", []):
                plan.add_step(
                    description=step_data["description"],
                    tool=step_data.get("tool"),
                    parameters=step_data.get("parameters"),
                )

            plan.estimated_duration = response.get("estimated_duration_seconds", 0)
            return plan

        except Exception:
            # Fallback to simple single-step plan
            plan = Plan()
            plan.add_step(description=goal.description)
            return plan

    async def decompose_goal(
        self,
        goal: Goal,
        max_depth: int = 3,
    ) -> list[Goal]:
        """
        Decompose a goal into sub-goals.

        Args:
            goal: The goal to decompose
            max_depth: Maximum decomposition depth

        Returns:
            List of sub-goals
        """
        if max_depth <= 0:
            return []

        prompt = f"""Decompose this goal into 2-4 sub-goals:

Goal: {goal.description}

Each sub-goal should be:
- More specific than the parent
- Achievable independently
- Ordered by dependency

Output format: JSON array of sub-goal descriptions."""

        try:
            schema = {
                "type": "object",
                "properties": {
                    "sub_goals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 4,
                    },
                },
                "required": ["sub_goals"],
            }

            response = await self._llm.generate_structured(prompt, schema)

            sub_goals = []
            for i, description in enumerate(response.get("sub_goals", [])):
                from lya.domain.models.goal import GoalPriority

                sub_goal = Goal(
                    agent_id=goal.agent_id,
                    description=description,
                    priority=GoalPriority(min(goal.priority_value + 1, 5)),
                    parent_goal_id=goal.id,
                )
                sub_goals.append(sub_goal)

            return sub_goals

        except Exception:
            return []

    async def refine_plan(
        self,
        goal: Goal,
        failed_task: Task,
        error_message: str,
    ) -> Plan:
        """
        Refine a plan after a task failure.

        Args:
            goal: The goal being pursued
            failed_task: The task that failed
            error_message: The error that occurred

        Returns:
            A refined plan
        """
        current_plan = goal.plan
        if not current_plan:
            return Plan()

        prompt = f"""Refine this plan after a task failure:

Goal: {goal.description}

Current Plan:
{self._format_plan(current_plan)}

Failed Task: {failed_task.description}
Error: {error_message}

Provide a refined plan that addresses the failure.
"""

        # For now, return the same plan with modified failed step
        # In production, this would use LLM to generate alternatives
        return current_plan

    def _build_planning_prompt(
        self,
        goal: Goal,
        tools: list[dict[str, Any]],
        context: str | None,
    ) -> str:
        """Build the planning prompt."""
        tools_str = "\n".join(
            f"- {t.get('name', 'unknown')}: {t.get('description', '')}"
            for t in tools
        )

        context_str = f"\nContext:\n{context}\n" if context else ""

        return f"""Create a step-by-step plan to achieve this goal.

Goal: {goal.description}

Available Tools:
{tools_str}
{context_str}

Provide a plan with specific steps. Each step should optionally use a tool."""

    def _format_plan(self, plan: Plan) -> str:
        """Format plan for display."""
        return "\n".join(
            f"{i+1}. {step.get('description', '')}"
            for i, step in enumerate(plan.steps)
        )
