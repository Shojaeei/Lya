"""Distributed Computing Domain Models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any
from uuid import UUID, uuid4


class NodeStatus(Enum):
    """Status of a distributed node."""
    ONLINE = auto()
    OFFLINE = auto()
    BUSY = auto()
    DEGRADED = auto()
    MAINTENANCE = auto()


class TaskDistributionStrategy(Enum):
    """Strategies for distributing tasks."""
    ROUND_ROBIN = auto()
    LEAST_LOADED = auto()
    CAPABILITY_MATCH = auto()
    HASH_BASED = auto()
    PRIORITY_BASED = auto()


@dataclass
class Node:
    """A node in the distributed cluster."""
    node_id: UUID
    name: str
    host: str
    port: int
    status: NodeStatus
    capabilities: list[str] = field(default_factory=list)
    current_load: float = 0.0  # 0-1
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.node_id, UUID):
            self.node_id = UUID(self.node_id) if isinstance(self.node_id, str) else uuid4()

    def is_healthy(self) -> bool:
        return self.status == NodeStatus.ONLINE and self.current_load < 0.9

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": str(self.node_id),
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "status": self.status.name,
            "capabilities": self.capabilities,
            "current_load": self.current_load,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "registered_at": self.registered_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class DistributedTask:
    """A task to be distributed across nodes."""
    task_id: UUID
    task_type: str
    payload: dict[str, Any]
    priority: int = 5  # 1-10
    node_id: UUID | None = None
    status: str = "pending"  # pending, assigned, running, completed, failed
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assigned_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None

    def __post_init__(self):
        if not isinstance(self.task_id, UUID):
            self.task_id = UUID(self.task_id) if isinstance(self.task_id, str) else uuid4()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority,
            "node_id": str(self.node_id) if self.node_id else None,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class ClusterState:
    """Current state of the cluster."""
    cluster_id: UUID
    coordinator_id: UUID | None
    nodes: list[Node] = field(default_factory=list)
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_load: float = 0.0
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not isinstance(self.cluster_id, UUID):
            self.cluster_id = UUID(self.cluster_id) if isinstance(self.cluster_id, str) else uuid4()

    def get_online_nodes(self) -> list[Node]:
        return [n for n in self.nodes if n.status == NodeStatus.ONLINE]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": str(self.cluster_id),
            "coordinator_id": str(self.coordinator_id) if self.coordinator_id else None,
            "nodes": [n.to_dict() for n in self.nodes],
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "average_load": self.average_load,
            "updated_at": self.updated_at.isoformat(),
        }
