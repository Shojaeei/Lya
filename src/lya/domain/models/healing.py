"""Self-Healing Domain Models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any
from uuid import UUID, uuid4


class IssueSeverity(Enum):
    """Severity levels for issues."""
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    INFO = auto()


class IssueType(Enum):
    """Types of issues that can be healed."""
    LLM_NOT_RESPONDING = auto()
    HIGH_MEMORY_USAGE = auto()
    DISK_FULL = auto()
    DATABASE_CONNECTION_LOST = auto()
    NETWORK_ERROR = auto()
    CACHE_CORRUPTED = auto()
    TASK_STUCK = auto()
    AGENT_CRASHED = auto()


class HealingStatus(Enum):
    """Status of healing action."""
    DETECTED = auto()
    HEALING = auto()
    SUCCESS = auto()
    FAILED = auto()
    DEFERRED = auto()
    IGNORED = auto()


@dataclass
class HealthIssue:
    """A health issue that needs healing."""
    issue_id: UUID
    issue_type: IssueType
    severity: IssueSeverity
    description: str
    detected_at: datetime
    source: str
    context: dict[str, Any] = field(default_factory=dict)
    status: HealingStatus = HealingStatus.DETECTED
    healing_attempts: int = 0

    def __post_init__(self):
        if not isinstance(self.issue_id, UUID):
            self.issue_id = UUID(self.issue_id) if isinstance(self.issue_id, str) else uuid4()

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": str(self.issue_id),
            "issue_type": self.issue_type.name,
            "severity": self.severity.name,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "source": self.source,
            "context": self.context,
            "status": self.status.name,
            "healing_attempts": self.healing_attempts,
        }


@dataclass
class HealingAction:
    """A healing action taken."""
    action_id: UUID
    issue_id: UUID
    action_type: str
    description: str
    status: HealingStatus
    started_at: datetime
    completed_at: datetime | None = None
    result: Any = None
    error_message: str | None = None

    def __post_init__(self):
        if not isinstance(self.action_id, UUID):
            self.action_id = UUID(self.action_id) if isinstance(self.action_id, str) else uuid4()
        if not isinstance(self.issue_id, UUID):
            self.issue_id = UUID(self.issue_id) if isinstance(self.issue_id, str) else uuid4()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": str(self.action_id),
            "issue_id": str(self.issue_id),
            "action_type": self.action_type,
            "description": self.description,
            "status": self.status.name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error_message": self.error_message,
        }


@dataclass
class HealthCheck:
    """Result of a health check."""
    check_id: UUID
    name: str
    component: str
    status: str  # healthy, warning, critical
    message: str
    timestamp: datetime
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.check_id, UUID):
            self.check_id = UUID(self.check_id) if isinstance(self.check_id, str) else uuid4()

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": str(self.check_id),
            "name": self.name,
            "component": self.component,
            "status": self.status,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
        }
