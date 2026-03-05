"""File-based task repository implementation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from lya.domain.models.task import Task, TaskPriority, TaskStatus
from lya.domain.repositories.task_repo import TaskRepository
from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


class FileTaskRepository(TaskRepository):
    """File-based task repository using JSON storage."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize the repository.

        Args:
            storage_path: Path to store task files. Defaults to workspace/tasks.
        """
        self.storage_path = storage_path or (settings.workspace_path / "tasks")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[UUID, Task] = {}
        self._loaded = False

    async def _load(self) -> None:
        """Load tasks from disk."""
        if self._loaded:
            return

        self._tasks = {}
        for task_file in self.storage_path.glob("*.json"):
            try:
                data = json.loads(task_file.read_text())
                task = self._deserialize_task(data)
                self._tasks[task.id] = task
            except Exception as e:
                logger.error("Failed to load task", file=task_file.name, error=str(e))

        self._loaded = True
        logger.info("Loaded tasks from storage", count=len(self._tasks))

    async def _save(self, task: Task) -> None:
        """Save a task to disk."""
        task_file = self.storage_path / f"{task.id}.json"
        data = self._serialize_task(task)
        task_file.write_text(json.dumps(data, indent=2))

    def _serialize_task(self, task: Task) -> dict[str, Any]:
        """Convert task to JSON-serializable dict."""
        return {
            "id": str(task.id),
            "description": task.description,
            "status": task.status.name,
            "priority": task.priority.name,
            "goal_id": str(task.goal_id) if task.goal_id else None,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "_metadata": task._metadata,
        }

    def _deserialize_task(self, data: dict[str, Any]) -> Task:
        """Convert dict to Task."""
        task = Task(
            task_id=UUID(data["id"]),
            description=data["description"],
            priority=TaskPriority[data["priority"]],
            goal_id=UUID(data["goal_id"]) if data.get("goal_id") else None,
        )
        task._status = TaskStatus[data["status"]]
        task._created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            task._started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            task._completed_at = datetime.fromisoformat(data["completed_at"])
        task._metadata = data.get("_metadata", {})
        return task

    async def get(self, task_id: UUID) -> Task | None:
        """Retrieve a task by ID."""
        await self._load()
        return self._tasks.get(task_id)

    async def save(self, task: Task) -> None:
        """Save a task."""
        await self._load()
        self._tasks[task.id] = task
        await self._save(task)
        logger.debug("Task saved", task_id=str(task.id))

    async def save_many(self, tasks: list[Task]) -> None:
        """Save multiple tasks."""
        await self._load()
        for task in tasks:
            self._tasks[task.id] = task
            await self._save(task)
        logger.debug("Tasks saved", count=len(tasks))

    async def delete(self, task_id: UUID) -> bool:
        """Delete a task."""
        await self._load()
        if task_id in self._tasks:
            del self._tasks[task_id]
            task_file = self.storage_path / f"{task_id}.json"
            if task_file.exists():
                task_file.unlink()
            logger.debug("Task deleted", task_id=str(task_id))
            return True
        return False

    async def get_by_goal(
        self,
        goal_id: UUID,
        status: TaskStatus | None = None,
    ) -> list[Task]:
        """Get tasks for a specific goal."""
        await self._load()
        tasks = [t for t in self._tasks.values() if t.goal_id == goal_id]
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.priority.value)

    async def get_pending_tasks(self, goal_id: UUID) -> list[Task]:
        """Get pending tasks for a goal."""
        return await self.get_by_goal(goal_id, TaskStatus.PENDING)

    async def get_executable_tasks(self, goal_id: UUID) -> list[Task]:
        """Get tasks ready for execution."""
        await self._load()
        tasks = [
            t for t in self._tasks.values()
            if t.goal_id == goal_id and t.status == TaskStatus.PENDING
        ]
        return sorted(tasks, key=lambda t: t.priority.value)

    async def get_completed_tasks(self, goal_id: UUID) -> list[Task]:
        """Get completed tasks for a goal."""
        return await self.get_by_goal(goal_id, TaskStatus.COMPLETED)

    async def delete_by_goal(self, goal_id: UUID) -> int:
        """Delete all tasks for a goal."""
        await self._load()
        to_delete = [tid for tid, t in self._tasks.items() if t.goal_id == goal_id]
        for task_id in to_delete:
            del self._tasks[task_id]
            task_file = self.storage_path / f"{task_id}.json"
            if task_file.exists():
                task_file.unlink()
        logger.debug("Tasks deleted for goal", goal_id=str(goal_id), count=len(to_delete))
        return len(to_delete)

    async def count(
        self,
        goal_id: UUID | None = None,
        status: TaskStatus | None = None,
    ) -> int:
        """Count tasks."""
        await self._load()
        tasks = list(self._tasks.values())
        if goal_id:
            tasks = [t for t in tasks if t.goal_id == goal_id]
        if status:
            tasks = [t for t in tasks if t.status == status]
        return len(tasks)
