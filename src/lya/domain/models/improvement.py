"""Self-improvement domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any
from uuid import UUID, uuid4


class ImprovementStatus(Enum):
    """Status of an improvement attempt."""

    PENDING = auto()
    GENERATING = auto()
    VALIDATING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ROLLED_BACK = auto()


class ImprovementType(Enum):
    """Type of improvement."""

    NEW_TOOL = auto()
    TOOL_ENHANCEMENT = auto()
    CAPABILITY = auto()
    OPTIMIZATION = auto()
    BUG_FIX = auto()


@dataclass
class CodeChange:
    """
    Represents a code change.

    Attributes:
        id: Unique identifier
        file_path: Path to the file changed
        original_code: Original code (for rollback)
        new_code: New code
        description: What was changed
        timestamp: When the change was made
    """

    file_path: str
    new_code: str
    id: UUID = field(default_factory=uuid4)
    original_code: str | None = None
    description: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "file_path": self.file_path,
            "original_code": self.original_code,
            "new_code": self.new_code[:500] + "..." if len(self.new_code) > 500 else self.new_code,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Improvement:
    """
    Represents a self-improvement attempt.

    Attributes:
        id: Unique identifier
        goal: What the improvement aims to achieve
        improvement_type: Type of improvement
        status: Current status
        description: Detailed description
        test_cases: Test cases to validate the improvement
        changes: Code changes made
        created_at: When created
        completed_at: When completed
        error: Error message if failed
        metadata: Additional data
    """

    goal: str
    improvement_type: ImprovementType
    id: UUID = field(default_factory=uuid4)
    status: ImprovementStatus = ImprovementStatus.PENDING
    description: str = ""
    test_cases: list[str] = field(default_factory=list)
    changes: list[CodeChange] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_generating(self) -> None:
        """Mark as generating code."""
        self.status = ImprovementStatus.GENERATING

    def mark_validating(self) -> None:
        """Mark as validating code."""
        self.status = ImprovementStatus.VALIDATING

    def mark_completed(self) -> None:
        """Mark as completed."""
        self.status = ImprovementStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)

    def mark_failed(self, error: str) -> None:
        """Mark as failed."""
        self.status = ImprovementStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(timezone.utc)

    def add_change(self, change: CodeChange) -> None:
        """Add a code change."""
        self.changes.append(change)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "goal": self.goal,
            "improvement_type": self.improvement_type.name,
            "status": self.status.name,
            "description": self.description,
            "test_cases": self.test_cases,
            "changes": [c.to_dict() for c in self.changes],
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ToolDefinition:
    """
    Definition of a generated tool.

    Attributes:
        name: Tool name
        description: What the tool does
        code: Python code for the tool
        parameters: Parameter definitions
        test_cases: Test cases
        category: Tool category
        validated: Whether code passed validation
    """

    name: str
    description: str
    code: str
    id: UUID = field(default_factory=uuid4)
    parameters: dict[str, Any] = field(default_factory=dict)
    test_cases: list[str] = field(default_factory=list)
    category: str = "generated"
    validated: bool = False
    validation_errors: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "code": self.code[:1000] + "..." if len(self.code) > 1000 else self.code,
            "parameters": self.parameters,
            "test_cases": self.test_cases,
            "category": self.category,
            "validated": self.validated,
            "validation_errors": self.validation_errors,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ImprovementStats:
    """Statistics for self-improvement."""

    total_improvements: int = 0
    successful: int = 0
    failed: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    recent_improvements: list[dict[str, Any]] = field(default_factory=list)

    def record(self, improvement: Improvement) -> None:
        """Record an improvement."""
        self.total_improvements += 1
        if improvement.status == ImprovementStatus.COMPLETED:
            self.successful += 1
        elif improvement.status == ImprovementStatus.FAILED:
            self.failed += 1

        type_name = improvement.improvement_type.name
        self.by_type[type_name] = self.by_type.get(type_name, 0) + 1

        self.recent_improvements.append(improvement.to_dict())
        if len(self.recent_improvements) > 100:
            self.recent_improvements = self.recent_improvements[-100:]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_improvements": self.total_improvements,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": self.successful / max(self.total_improvements, 1),
            "by_type": self.by_type,
            "recent_count": len(self.recent_improvements),
        }
