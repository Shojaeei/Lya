"""Agent domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any
from uuid import UUID, uuid4


class AgentState(Enum):
    """Agent lifecycle states."""
    INITIALIZING = auto()
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    LEARNING = auto()
    REFLECTING = auto()
    PAUSED = auto()
    SHUTTING_DOWN = auto()
    ERROR = auto()


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str = "Lya"
    workspace_path: str = "~/.lya"

    # LLM settings
    llm_provider: str = "ollama"
    llm_model: str = "kimi-k2.5:cloud"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    # Behavior settings
    autonomous: bool = True
    max_concurrent_goals: int = 3
    planning_depth: int = 3
    reflection_interval: int = 300  # seconds

    # Self-improvement
    self_improvement_enabled: bool = True
    code_generation_enabled: bool = False

    # Security
    sandbox_enabled: bool = True
    allow_file_write: bool = False

    # Monitoring
    metrics_enabled: bool = True
    log_level: str = "INFO"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "workspace_path": self.workspace_path,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "autonomous": self.autonomous,
            "max_concurrent_goals": self.max_concurrent_goals,
            "planning_depth": self.planning_depth,
            "reflection_interval": self.reflection_interval,
            "self_improvement_enabled": self.self_improvement_enabled,
            "code_generation_enabled": self.code_generation_enabled,
            "sandbox_enabled": self.sandbox_enabled,
            "allow_file_write": self.allow_file_write,
            "metrics_enabled": self.metrics_enabled,
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Agent:
    """
    Core Agent entity representing an autonomous AI agent.

    This is the root aggregate of the Agent domain.
    """

    def __init__(
        self,
        agent_id: UUID | None = None,
        config: AgentConfig | None = None,
    ):
        self._id = agent_id or uuid4()
        self._config = config or AgentConfig()
        self._state = AgentState.INITIALIZING
        self._created_at = datetime.now(timezone.utc)
        self._updated_at = self._created_at
        self._goals: list[UUID] = []
        self._metrics: dict[str, Any] = {
            "goals_completed": 0,
            "goals_failed": 0,
            "tasks_executed": 0,
            "memories_created": 0,
        }
        self._is_running = False

    # ═══════════════════════════════════════════════════════════════
    # Properties
    # ═══════════════════════════════════════════════════════════════

    @property
    def id(self) -> UUID:
        """Agent unique identifier."""
        return self._id

    @property
    def name(self) -> str:
        """Agent name."""
        return self._config.name

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        return self._state

    @property
    def config(self) -> AgentConfig:
        """Agent configuration."""
        return self._config

    @property
    def created_at(self) -> datetime:
        """Creation timestamp."""
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        """Last update timestamp."""
        return self._updated_at

    @property
    def goals(self) -> list[UUID]:
        """List of goal IDs."""
        return self._goals.copy()

    @property
    def metrics(self) -> dict[str, Any]:
        """Agent metrics."""
        return self._metrics.copy()

    @property
    def is_running(self) -> bool:
        """Whether agent is currently running."""
        return self._is_running

    @property
    def is_autonomous(self) -> bool:
        """Whether agent operates autonomously."""
        return self._config.autonomous

    # ═══════════════════════════════════════════════════════════════
    # State Management
    # ═══════════════════════════════════════════════════════════════

    def transition_to(self, new_state: AgentState, reason: str | None = None) -> None:
        """
        Transition agent to a new state.

        Args:
            new_state: Target state
            reason: Optional reason for transition

        Raises:
            StateTransitionError: If transition is invalid
        """
        from lya.domain.exceptions import StateTransitionError

        valid_transitions = {
            AgentState.INITIALIZING: [AgentState.IDLE, AgentState.ERROR],
            AgentState.IDLE: [AgentState.PLANNING, AgentState.EXECUTING,
                           AgentState.LEARNING, AgentState.REFLECTING,
                           AgentState.PAUSED, AgentState.SHUTTING_DOWN],
            AgentState.PLANNING: [AgentState.IDLE, AgentState.EXECUTING, AgentState.ERROR],
            AgentState.EXECUTING: [AgentState.IDLE, AgentState.LEARNING,
                                  AgentState.ERROR, AgentState.PAUSED],
            AgentState.LEARNING: [AgentState.IDLE, AgentState.ERROR],
            AgentState.REFLECTING: [AgentState.IDLE, AgentState.ERROR],
            AgentState.PAUSED: [AgentState.IDLE, AgentState.SHUTTING_DOWN],
            AgentState.ERROR: [AgentState.IDLE, AgentState.SHUTTING_DOWN],
            AgentState.SHUTTING_DOWN: [],
        }

        if new_state not in valid_transitions.get(self._state, []):
            raise StateTransitionError(
                "Agent", self._state.name, new_state.name
            )

        old_state = self._state
        self._state = new_state
        self._updated_at = datetime.now(timezone.utc)

        # Emit event
        # Note: Actual event emission happens in application layer

    def start(self) -> None:
        """Start the agent."""
        if self._is_running:
            return

        self._is_running = True
        self.transition_to(AgentState.IDLE, "Agent started")

    def pause(self) -> None:
        """Pause the agent."""
        if not self._is_running:
            return

        self.transition_to(AgentState.PAUSED, "Paused by user")

    def resume(self) -> None:
        """Resume the agent from pause."""
        if self._state != AgentState.PAUSED:
            return

        self.transition_to(AgentState.IDLE, "Resumed by user")

    def shutdown(self, reason: str | None = None) -> None:
        """Shutdown the agent gracefully."""
        self.transition_to(AgentState.SHUTTING_DOWN, reason or "Shutdown requested")
        self._is_running = False

    # ═══════════════════════════════════════════════════════════════
    # Goal Management
    # ═══════════════════════════════════════════════════════════════

    def add_goal(self, goal_id: UUID) -> None:
        """Add a goal to the agent."""
        if goal_id not in self._goals:
            self._goals.append(goal_id)
            self._updated_at = datetime.now(timezone.utc)

    def remove_goal(self, goal_id: UUID) -> None:
        """Remove a goal from the agent."""
        if goal_id in self._goals:
            self._goals.remove(goal_id)
            self._updated_at = datetime.now(timezone.utc)

    # ═══════════════════════════════════════════════════════════════
    # Metrics
    # ═══════════════════════════════════════════════════════════════

    def record_goal_completed(self) -> None:
        """Record a completed goal."""
        self._metrics["goals_completed"] += 1

    def record_goal_failed(self) -> None:
        """Record a failed goal."""
        self._metrics["goals_failed"] += 1

    def record_task_executed(self) -> None:
        """Record an executed task."""
        self._metrics["tasks_executed"] += 1

    def record_memory_created(self) -> None:
        """Record a created memory."""
        self._metrics["memories_created"] += 1

    # ═══════════════════════════════════════════════════════════════
    # Serialization
    # ═══════════════════════════════════════════════════════════════

    def to_dict(self) -> dict[str, Any]:
        """Convert agent to dictionary."""
        return {
            "id": str(self._id),
            "name": self.name,
            "state": self._state.name,
            "config": self._config.to_dict(),
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "goals": [str(g) for g in self._goals],
            "metrics": self._metrics,
            "is_running": self._is_running,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Create agent from dictionary."""
        agent = cls(
            agent_id=UUID(data["id"]),
            config=AgentConfig.from_dict(data.get("config", {}))
        )
        agent._state = AgentState[data["state"]]
        agent._created_at = datetime.fromisoformat(data["created_at"])
        agent._updated_at = datetime.fromisoformat(data["updated_at"])
        agent._goals = [UUID(g) for g in data.get("goals", [])]
        agent._metrics = data.get("metrics", agent._metrics)
        agent._is_running = data.get("is_running", False)
        return agent
