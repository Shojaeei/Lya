"""
Lya Domain Package

This package contains the core business logic of Lya.
It is independent of any external frameworks or infrastructure.

Domain layer principles:
- Contains business entities, value objects, and domain services
- No dependencies on infrastructure or application layers
- Expresses business rules through rich domain models
- Uses domain events for loose coupling
"""

from .models.agent import Agent, AgentState, AgentConfig
from .models.goal import Goal, GoalStatus, GoalPriority
from .models.task import Task, TaskStatus, TaskResult
from .models.memory import Memory, MemoryType, MemoryImportance
from .models.events import DomainEvent, EventPublisher
from .exceptions import DomainException, ValidationError, BusinessRuleError

__all__ = [
    # Agent
    "Agent",
    "AgentState",
    "AgentConfig",
    # Goal
    "Goal",
    "GoalStatus",
    "GoalPriority",
    # Task
    "Task",
    "TaskStatus",
    "TaskResult",
    # Memory
    "Memory",
    "MemoryType",
    "MemoryImportance",
    # Events
    "DomainEvent",
    "EventPublisher",
    # Exceptions
    "DomainException",
    "ValidationError",
    "BusinessRuleError",
]
