"""Deployment Domain Models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


class DeploymentTarget(Enum):
    """Deployment targets."""
    DOCKER = auto()
    DOCKER_COMPOSE = auto()
    KUBERNETES = auto()
    VPS = auto()
    AWS = auto()
    AZURE = auto()
    GCP = auto()
    LOCAL = auto()


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = auto()
    BUILDING = auto()
    DEPLOYING = auto()
    RUNNING = auto()
    FAILED = auto()
    STOPPED = auto()


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    config_id: UUID
    target: DeploymentTarget
    name: str
    version: str
    environment: dict[str, str] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    networking: dict[str, Any] = field(default_factory=dict)
    volumes: list[dict[str, Any]] = field(default_factory=list)
    health_check: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.config_id, UUID):
            self.config_id = UUID(self.config_id) if isinstance(self.config_id, str) else uuid4()

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": str(self.config_id),
            "target": self.target.name,
            "name": self.name,
            "version": self.version,
            "environment": self.environment,
            "resources": self.resources,
            "networking": self.networking,
            "volumes": self.volumes,
            "health_check": self.health_check,
        }


@dataclass
class Deployment:
    """A deployment instance."""
    deployment_id: UUID
    config: DeploymentConfig
    status: DeploymentStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    endpoint: str | None = None
    container_id: str | None = None
    logs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.deployment_id, UUID):
            self.deployment_id = UUID(self.deployment_id) if isinstance(self.deployment_id, str) else uuid4()

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": str(self.deployment_id),
            "config": self.config.to_dict(),
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "endpoint": self.endpoint,
            "container_id": self.container_id,
            "logs": self.logs,
            "metrics": self.metrics,
        }


@dataclass
class BuildArtifact:
    """Build artifact information."""
    artifact_id: UUID
    name: str
    version: str
    path: Path
    size_bytes: int
    created_at: datetime
    checksum: str
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.artifact_id, UUID):
            self.artifact_id = UUID(self.artifact_id) if isinstance(self.artifact_id, str) else uuid4()
