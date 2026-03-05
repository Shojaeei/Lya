"""
Lya - Professional Autonomous AGI Agent

A production-grade autonomous agent system built with Clean Architecture principles.

Example Usage:
    # Create and start an agent
    from lya.domain.models.agent import Agent, AgentConfig
    from lya.application.commands.create_agent import CreateAgentHandler

    config = AgentConfig(name="MyAgent", autonomous=True)
    agent = await CreateAgentHandler().execute(name="MyAgent", config=config)

    # Add a goal
    from lya.application.commands.add_goal import AddGoalHandler
    goal_id = await AddGoalHandler().execute(
        agent_id=agent.id,
        description="Research Python best practices"
    )

Architecture:
    - Domain Layer: Core business logic, entities, and domain services
    - Application Layer: Use cases, commands, queries, and event handlers
    - Infrastructure Layer: External adapters (LLM, DB, tools, etc.)
    - Adapters Layer: CLI, API, and other interfaces

For more information, see the documentation at docs/
"""

__version__ = "0.2.0"
__author__ = "Shoji"
__license__ = "MIT"

# Domain exports
from lya.domain.models.agent import Agent, AgentState, AgentConfig
from lya.domain.models.goal import Goal, GoalStatus, GoalPriority, Plan
from lya.domain.models.task import Task, TaskStatus, TaskResult
from lya.domain.models.memory import Memory, MemoryType, MemoryImportance, MemoryContext
from lya.domain.models.events import DomainEvent, EventPublisher
from lya.domain.exceptions import DomainException, ValidationError, BusinessRuleError

__all__ = [
    # Version
    "__version__",
    # Agent
    "Agent",
    "AgentState",
    "AgentConfig",
    # Goal
    "Goal",
    "GoalStatus",
    "GoalPriority",
    "Plan",
    # Task
    "Task",
    "TaskStatus",
    "TaskResult",
    # Memory
    "Memory",
    "MemoryType",
    "MemoryImportance",
    "MemoryContext",
    # Events
    "DomainEvent",
    "EventPublisher",
    # Exceptions
    "DomainException",
    "ValidationError",
    "BusinessRuleError",
]
