"""Domain events for event-driven architecture."""

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable
from uuid import UUID, uuid4


@dataclass(frozen=True)
class DomainEvent(ABC):
    """Base class for all domain events."""

    event_id: UUID = field(default_factory=uuid4, init=False)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc), init=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.__class__.__name__,
            "timestamp": self.timestamp.isoformat(),
            **{k: str(v) if isinstance(v, UUID) else v
               for k, v in self.__dict__.items()
               if k not in ("event_id", "timestamp")}
        }


# ═══════════════════════════════════════════════════════════════
# Agent Events
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AgentCreated(DomainEvent):
    """Emitted when a new agent is created."""
    agent_id: str
    name: str
    config: dict[str, Any]


@dataclass(frozen=True)
class AgentStateChanged(DomainEvent):
    """Emitted when agent state changes."""
    agent_id: str
    previous_state: str
    new_state: str
    reason: str | None = None


@dataclass(frozen=True)
class AgentShutdown(DomainEvent):
    """Emitted when agent shuts down."""
    agent_id: str
    reason: str | None = None


# ═══════════════════════════════════════════════════════════════
# Goal Events
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GoalCreated(DomainEvent):
    """Emitted when a goal is created."""
    goal_id: str
    agent_id: str
    description: str
    priority: int


@dataclass(frozen=True)
class GoalStarted(DomainEvent):
    """Emitted when goal execution starts."""
    goal_id: str
    agent_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class GoalCompleted(DomainEvent):
    """Emitted when a goal is completed."""
    goal_id: str
    agent_id: str
    result_summary: str
    duration_seconds: float


@dataclass(frozen=True)
class GoalFailed(DomainEvent):
    """Emitted when a goal fails."""
    goal_id: str
    agent_id: str
    error_message: str
    can_retry: bool = True


@dataclass(frozen=True)
class GoalCancelled(DomainEvent):
    """Emitted when a goal is cancelled."""
    goal_id: str
    agent_id: str
    reason: str


# ═══════════════════════════════════════════════════════════════
# Task Events
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TaskCreated(DomainEvent):
    """Emitted when a task is created."""
    task_id: str
    goal_id: str
    description: str
    tool_name: str | None = None


@dataclass(frozen=True)
class TaskStarted(DomainEvent):
    """Emitted when task execution starts."""
    task_id: str
    goal_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class TaskCompleted(DomainEvent):
    """Emitted when a task is completed."""
    task_id: str
    goal_id: str
    result: Any
    execution_time_ms: int


@dataclass(frozen=True)
class TaskFailed(DomainEvent):
    """Emitted when a task fails."""
    task_id: str
    goal_id: str
    error: str
    retry_count: int = 0


# ═══════════════════════════════════════════════════════════════
# Memory Events
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MemoryStored(DomainEvent):
    """Emitted when a memory is stored."""
    memory_id: str
    agent_id: str
    memory_type: str
    importance: int


@dataclass(frozen=True)
class MemoryRecalled(DomainEvent):
    """Emitted when memories are recalled."""
    query: str
    agent_id: str
    results_count: int


# ═══════════════════════════════════════════════════════════════
# Tool Events
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ToolExecuted(DomainEvent):
    """Emitted when a tool is executed."""
    tool_name: str
    agent_id: str
    success: bool
    execution_time_ms: int


@dataclass(frozen=True)
class ToolGenerated(DomainEvent):
    """Emitted when a new tool is generated."""
    tool_name: str
    agent_id: str
    generated_from: str


# ═══════════════════════════════════════════════════════════════
# Capability Events (Self-Improvement)
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CapabilityGenerated(DomainEvent):
    """Emitted when a new capability is generated."""
    capability_id: str
    name: str
    description: str


@dataclass(frozen=True)
class CapabilityValidated(DomainEvent):
    """Emitted when a capability is validated."""
    capability_id: str
    passed: bool
    errors: list[str]


@dataclass(frozen=True)
class CapabilityRegistered(DomainEvent):
    """Emitted when a capability is registered."""
    capability_id: str
    name: str
    functions: list[str]


@dataclass(frozen=True)
class CapabilityImproved(DomainEvent):
    """Emitted when a capability is improved."""
    capability_id: str
    parent_capability_id: str
    improvement_reason: str


# ═══════════════════════════════════════════════════════════════
# Event Publisher Protocol
# ═══════════════════════════════════════════════════════════════

@runtime_checkable
class EventPublisher(Protocol):
    """Protocol for event publishing."""

    async def publish(self, event: DomainEvent) -> None:
        """Publish an event."""
        ...

    def subscribe(self, event_type: type[DomainEvent], handler: callable) -> None:
        """Subscribe to an event type."""
        ...
