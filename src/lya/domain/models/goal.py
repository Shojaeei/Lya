"""Goal domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any
from uuid import UUID, uuid4


class GoalStatus(Enum):
    """Goal lifecycle states."""
    PENDING = auto()
    PLANNING = auto()
    READY = auto()
    IN_PROGRESS = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class GoalPriority(Enum):
    """Goal priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    TRIVIAL = 5


@dataclass
class Plan:
    """A plan for achieving a goal."""
    steps: list[dict[str, Any]] = field(default_factory=list)
    estimated_duration: int = 0  # seconds
    dependencies: list[UUID] = field(default_factory=list)

    def add_step(self, description: str, tool: str | None = None,
                 parameters: dict[str, Any] | None = None) -> None:
        """Add a step to the plan."""
        self.steps.append({
            "order": len(self.steps) + 1,
            "description": description,
            "tool": tool,
            "parameters": parameters or {},
            "completed": False,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "estimated_duration": self.estimated_duration,
            "dependencies": [str(d) for d in self.dependencies],
        }


class Goal:
    """
    Represents an objective the agent wants to achieve.

    Goals are high-level objectives that are decomposed into tasks.
    """

    def __init__(
        self,
        goal_id: UUID | None = None,
        agent_id: UUID | None = None,
        description: str = "",
        priority: GoalPriority = GoalPriority.MEDIUM,
        parent_goal_id: UUID | None = None,
    ):
        self._id = goal_id or uuid4()
        self._agent_id = agent_id
        self._description = description
        self._priority = priority
        self._parent_goal_id = parent_goal_id
        self._status = GoalStatus.PENDING
        self._plan: Plan | None = None
        self._tasks: list[UUID] = []
        self._sub_goals: list[UUID] = []
        self._created_at = datetime.now(timezone.utc)
        self._started_at: datetime | None = None
        self._completed_at: datetime | None = None
        self._result: Any = None
        self._error_message: str | None = None
        self._retry_count = 0
        self._max_retries = 3
        self._metadata: dict[str, Any] = {}

    # ═══════════════════════════════════════════════════════════════
    # Properties
    # ═══════════════════════════════════════════════════════════════

    @property
    def id(self) -> UUID:
        """Goal unique identifier."""
        return self._id

    @property
    def agent_id(self) -> UUID | None:
        """ID of the agent owning this goal."""
        return self._agent_id

    @property
    def description(self) -> str:
        """Goal description."""
        return self._description

    @property
    def priority(self) -> GoalPriority:
        """Goal priority."""
        return self._priority

    @property
    def priority_value(self) -> int:
        """Numeric priority value (lower is higher priority)."""
        return self._priority.value

    @property
    def status(self) -> GoalStatus:
        """Current goal status."""
        return self._status

    @property
    def parent_goal_id(self) -> UUID | None:
        """Parent goal ID if this is a sub-goal."""
        return self._parent_goal_id

    @property
    def plan(self) -> Plan | None:
        """The plan for achieving this goal."""
        return self._plan

    @property
    def tasks(self) -> list[UUID]:
        """List of task IDs."""
        return self._tasks.copy()

    @property
    def sub_goals(self) -> list[UUID]:
        """List of sub-goal IDs."""
        return self._sub_goals.copy()

    @property
    def created_at(self) -> datetime:
        """Creation timestamp."""
        return self._created_at

    @property
    def started_at(self) -> datetime | None:
        """Start timestamp."""
        return self._started_at

    @property
    def completed_at(self) -> datetime | None:
        """Completion timestamp."""
        return self._completed_at

    @property
    def result(self) -> Any:
        """Goal result if completed."""
        return self._result

    @property
    def error_message(self) -> str | None:
        """Error message if failed."""
        return self._error_message

    @property
    def retry_count(self) -> int:
        """Current retry count."""
        return self._retry_count

    @property
    def can_retry(self) -> bool:
        """Whether goal can be retried."""
        return self._retry_count < self._max_retries

    @property
    def is_active(self) -> bool:
        """Whether goal is currently active."""
        return self._status in {
            GoalStatus.IN_PROGRESS,
            GoalStatus.PLANNING,
            GoalStatus.READY,
        }

    @property
    def is_terminal(self) -> bool:
        """Whether goal has reached terminal state."""
        return self._status in {
            GoalStatus.COMPLETED,
            GoalStatus.FAILED,
            GoalStatus.CANCELLED,
        }

    # ═══════════════════════════════════════════════════════════════
    # Status Transitions
    # ═══════════════════════════════════════════════════════════════

    def start_planning(self) -> None:
        """Start planning phase."""
        if self._status != GoalStatus.PENDING:
            raise ValueError(f"Cannot plan goal in {self._status.name} state")
        self._status = GoalStatus.PLANNING

    def set_plan(self, plan: Plan) -> None:
        """Set the plan for this goal."""
        if self._status != GoalStatus.PLANNING:
            raise ValueError(f"Cannot set plan in {self._status.name} state")
        self._plan = plan
        self._status = GoalStatus.READY

    def start_execution(self) -> None:
        """Start executing the goal."""
        if self._status not in (GoalStatus.READY, GoalStatus.PAUSED):
            raise ValueError(f"Cannot start goal in {self._status.name} state")
        self._status = GoalStatus.IN_PROGRESS
        if self._started_at is None:
            self._started_at = datetime.now(timezone.utc)

    def pause(self) -> None:
        """Pause goal execution."""
        if self._status != GoalStatus.IN_PROGRESS:
            raise ValueError(f"Cannot pause goal in {self._status.name} state")
        self._status = GoalStatus.PAUSED

    def complete(self, result: Any) -> None:
        """Mark goal as completed."""
        if self._status not in (GoalStatus.IN_PROGRESS, GoalStatus.READY):
            raise ValueError(f"Cannot complete goal in {self._status.name} state")
        self._status = GoalStatus.COMPLETED
        self._completed_at = datetime.now(timezone.utc)
        self._result = result

    def fail(self, error_message: str) -> None:
        """Mark goal as failed."""
        self._status = GoalStatus.FAILED
        self._completed_at = datetime.now(timezone.utc)
        self._error_message = error_message

    def cancel(self, reason: str = "") -> None:
        """Cancel the goal."""
        if self.is_terminal:
            return
        self._status = GoalStatus.CANCELLED
        self._completed_at = datetime.now(timezone.utc)
        self._error_message = reason

    def retry(self) -> None:
        """Retry a failed goal."""
        if self._status != GoalStatus.FAILED:
            raise ValueError(f"Cannot retry goal in {self._status.name} state")
        if not self.can_retry:
            raise ValueError("Maximum retries exceeded")
        self._retry_count += 1
        self._status = GoalStatus.PENDING
        self._error_message = None

    # ═══════════════════════════════════════════════════════════════
    # Task Management
    # ═══════════════════════════════════════════════════════════════

    def add_task(self, task_id: UUID) -> None:
        """Add a task to this goal."""
        if task_id not in self._tasks:
            self._tasks.append(task_id)

    def remove_task(self, task_id: UUID) -> None:
        """Remove a task from this goal."""
        if task_id in self._tasks:
            self._tasks.remove(task_id)

    # ═══════════════════════════════════════════════════════════════
    # Sub-goal Management
    # ═══════════════════════════════════════════════════════════════

    def add_sub_goal(self, goal_id: UUID) -> None:
        """Add a sub-goal."""
        if goal_id not in self._sub_goals:
            self._sub_goals.append(goal_id)

    # ═══════════════════════════════════════════════════════════════
    # Metadata
    # ═══════════════════════════════════════════════════════════════

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata key-value pair."""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self._metadata.get(key, default)

    # ═══════════════════════════════════════════════════════════════
    # Serialization
    # ═══════════════════════════════════════════════════════════════

    def to_dict(self) -> dict[str, Any]:
        """Convert goal to dictionary."""
        return {
            "id": str(self._id),
            "agent_id": str(self._agent_id) if self._agent_id else None,
            "description": self._description,
            "priority": self._priority.name,
            "priority_value": self.priority_value,
            "parent_goal_id": str(self._parent_goal_id) if self._parent_goal_id else None,
            "status": self._status.name,
            "plan": self._plan.to_dict() if self._plan else None,
            "tasks": [str(t) for t in self._tasks],
            "sub_goals": [str(g) for g in self._sub_goals],
            "created_at": self._created_at.isoformat(),
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "completed_at": self._completed_at.isoformat() if self._completed_at else None,
            "result": self._result,
            "error_message": self._error_message,
            "retry_count": self._retry_count,
            "max_retries": self._max_retries,
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Goal:
        """Create goal from dictionary."""
        goal = cls(
            goal_id=UUID(data["id"]),
            agent_id=UUID(data["agent_id"]) if data.get("agent_id") else None,
            description=data["description"],
            priority=GoalPriority[data.get("priority", "MEDIUM")],
            parent_goal_id=UUID(data["parent_goal_id"]) if data.get("parent_goal_id") else None,
        )
        goal._status = GoalStatus[data.get("status", "PENDING")]
        goal._created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            goal._started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            goal._completed_at = datetime.fromisoformat(data["completed_at"])
        goal._result = data.get("result")
        goal._error_message = data.get("error_message")
        goal._retry_count = data.get("retry_count", 0)
        goal._max_retries = data.get("max_retries", 3)
        goal._metadata = data.get("metadata", {})
        return goal
