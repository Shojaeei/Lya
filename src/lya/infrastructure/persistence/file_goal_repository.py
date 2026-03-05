"""File-based goal repository implementation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from lya.domain.models.goal import Goal, GoalPriority, GoalStatus
from lya.domain.repositories.goal_repo import GoalRepository
from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


class FileGoalRepository(GoalRepository):
    """File-based goal repository using JSON storage."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize the repository.

        Args:
            storage_path: Path to store goal files. Defaults to workspace/goals.
        """
        self.storage_path = storage_path or (settings.workspace_path / "goals")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._goals: dict[UUID, Goal] = {}
        self._loaded = False

    async def _load(self) -> None:
        """Load goals from disk."""
        if self._loaded:
            return

        self._goals = {}
        for goal_file in self.storage_path.glob("*.json"):
            try:
                data = json.loads(goal_file.read_text())
                goal = self._deserialize_goal(data)
                self._goals[goal.id] = goal
            except Exception as e:
                logger.error("Failed to load goal", file=goal_file.name, error=str(e))

        self._loaded = True
        logger.info("Loaded goals from storage", count=len(self._goals))

    async def _save(self, goal: Goal) -> None:
        """Save a goal to disk."""
        goal_file = self.storage_path / f"{goal.id}.json"
        data = self._serialize_goal(goal)
        goal_file.write_text(json.dumps(data, indent=2))

    def _serialize_goal(self, goal: Goal) -> dict[str, Any]:
        """Convert goal to JSON-serializable dict."""
        return {
            "id": str(goal.id),
            "description": goal.description,
            "status": goal.status.name,
            "priority": goal.priority.name,
            "created_at": goal.created_at.isoformat(),
            "completed_at": goal.completed_at.isoformat() if goal.completed_at else None,
            "parent_id": str(goal.parent_goal_id) if goal.parent_goal_id else None,
            "sub_goal_ids": [str(gid) for gid in goal.sub_goals],
            "_metadata": goal.get_metadata("_raw", {}),
        }

    def _deserialize_goal(self, data: dict[str, Any]) -> Goal:
        """Convert dict to Goal."""
        goal = Goal(
            goal_id=UUID(data["id"]),
            description=data["description"],
            priority=GoalPriority[data["priority"]],
            parent_goal_id=UUID(data["parent_id"]) if data.get("parent_id") else None,
        )
        goal._status = GoalStatus[data["status"]]
        goal._created_at = datetime.fromisoformat(data["created_at"])
        if data.get("completed_at"):
            goal._completed_at = datetime.fromisoformat(data["completed_at"])
        for gid in data.get("sub_goal_ids", []):
            goal.add_sub_goal(UUID(gid))
        for key, value in data.get("_metadata", {}).items():
            goal.set_metadata(key, value)
        return goal

    async def get(self, goal_id: UUID) -> Goal | None:
        """Retrieve a goal by ID."""
        await self._load()
        return self._goals.get(goal_id)

    async def save(self, goal: Goal) -> None:
        """Save a goal."""
        await self._load()
        self._goals[goal.id] = goal
        await self._save(goal)
        logger.debug("Goal saved", goal_id=str(goal.id))

    async def delete(self, goal_id: UUID) -> bool:
        """Delete a goal."""
        await self._load()
        if goal_id in self._goals:
            del self._goals[goal_id]
            goal_file = self.storage_path / f"{goal_id}.json"
            if goal_file.exists():
                goal_file.unlink()
            logger.debug("Goal deleted", goal_id=str(goal_id))
            return True
        return False

    async def get_by_agent(
        self,
        agent_id: UUID,
        status: GoalStatus | None = None,
        limit: int = 100,
    ) -> list[Goal]:
        """Get goals for a specific agent."""
        await self._load()
        goals = list(self._goals.values())
        if status:
            goals = [g for g in goals if g.status == status]
        return goals[:limit]

    async def get_active_goals(self, agent_id: UUID) -> list[Goal]:
        """Get active (non-terminal) goals for an agent."""
        await self._load()
        active_statuses = {GoalStatus.PENDING, GoalStatus.IN_PROGRESS}
        goals = [g for g in self._goals.values() if g.status in active_statuses]
        return sorted(goals, key=lambda g: g.priority.value)

    async def get_pending_goals(
        self,
        agent_id: UUID,
        limit: int = 10,
    ) -> list[Goal]:
        """Get pending goals ordered by priority."""
        await self._load()
        goals = [g for g in self._goals.values() if g.status == GoalStatus.PENDING]
        return sorted(goals, key=lambda g: g.priority.value)[:limit]

    async def get_sub_goals(self, parent_goal_id: UUID) -> list[Goal]:
        """Get sub-goals of a parent goal."""
        await self._load()
        return [g for g in self._goals.values() if g.parent_goal_id == parent_goal_id]

    async def update_status(
        self,
        goal_id: UUID,
        status: GoalStatus,
    ) -> bool:
        """Update goal status."""
        await self._load()
        goal = self._goals.get(goal_id)
        if goal:
            if status == GoalStatus.IN_PROGRESS:
                goal.start_execution()
            elif status == GoalStatus.COMPLETED:
                goal.complete(result={"manual_update": True})
            elif status == GoalStatus.FAILED:
                goal.fail("Manual status update to failed")
            elif status == GoalStatus.CANCELLED:
                goal.cancel("Manual status update to cancelled")
            else:
                goal._status = status
            await self._save(goal)
            return True
        return False

    async def count(
        self,
        agent_id: UUID | None = None,
        status: GoalStatus | None = None,
    ) -> int:
        """Count goals."""
        await self._load()
        goals = list(self._goals.values())
        if agent_id:
            goals = [g for g in goals if g.agent_id == agent_id]
        if status:
            goals = [g for g in goals if g.status == status]
        return len(goals)
