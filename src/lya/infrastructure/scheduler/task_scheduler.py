"""Task Scheduler — pure Python asyncio-based task scheduling.

Features:
- Persistent JSON storage (survives restarts)
- One-time and recurring tasks (hourly, daily, weekly)
- Background asyncio loop checking every 30s
- Callback-based execution
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable
from uuid import uuid4

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScheduledTask:
    """A scheduled task."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    chat_id: int = 0
    description: str = ""
    trigger_time: str = ""  # ISO format
    recurring: str | None = None  # "hourly", "daily", "weekly", or None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed: bool = False

    def is_due(self) -> bool:
        """Check if the task is due for execution."""
        if self.completed:
            return False
        try:
            trigger = datetime.fromisoformat(self.trigger_time)
            if trigger.tzinfo is None:
                trigger = trigger.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) >= trigger
        except (ValueError, TypeError):
            return False

    def reschedule(self) -> bool:
        """Reschedule recurring task. Returns False if not recurring."""
        if not self.recurring:
            return False

        try:
            trigger = datetime.fromisoformat(self.trigger_time)
            if trigger.tzinfo is None:
                trigger = trigger.replace(tzinfo=timezone.utc)

            deltas = {
                "hourly": timedelta(hours=1),
                "daily": timedelta(days=1),
                "weekly": timedelta(weeks=1),
            }
            delta = deltas.get(self.recurring)
            if not delta:
                return False

            # Move to next occurrence
            now = datetime.now(timezone.utc)
            while trigger <= now:
                trigger += delta

            self.trigger_time = trigger.isoformat()
            self.completed = False
            return True

        except (ValueError, TypeError):
            return False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduledTask:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Type for the callback
TaskCallback = Callable[[ScheduledTask], Awaitable[None]]


class TaskScheduler:
    """Asyncio-based task scheduler with JSON persistence."""

    def __init__(self, workspace: Path):
        self._workspace = workspace / "scheduler"
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._db_file = self._workspace / "tasks.json"
        self._tasks: list[ScheduledTask] = []
        self._callback: TaskCallback | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._load()

    def _load(self) -> None:
        """Load tasks from disk."""
        if self._db_file.exists():
            try:
                data = json.loads(self._db_file.read_text(encoding="utf-8"))
                self._tasks = [
                    ScheduledTask.from_dict(t) for t in data.get("tasks", [])
                ]
                logger.info("scheduler_loaded", task_count=len(self._tasks))
            except (json.JSONDecodeError, OSError) as e:
                logger.error("scheduler_load_failed", error=str(e))
                self._tasks = []
        else:
            self._tasks = []

    def _save(self) -> None:
        """Persist tasks to disk."""
        try:
            data = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "tasks": [t.to_dict() for t in self._tasks],
            }
            tmp = self._db_file.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            tmp.replace(self._db_file)
        except OSError as e:
            logger.error("scheduler_save_failed", error=str(e))

    def set_callback(self, callback: TaskCallback) -> None:
        """Set the callback for when tasks are due."""
        self._callback = callback

    async def start(self) -> None:
        """Start the background scheduler loop."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("scheduler_started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._save()
        logger.info("scheduler_stopped")

    async def _loop(self) -> None:
        """Background loop: check for due tasks every 30 seconds."""
        while self._running:
            try:
                await self._check_due_tasks()
            except Exception as e:
                logger.error("scheduler_loop_error", error=str(e))
            await asyncio.sleep(30)

    async def _check_due_tasks(self) -> None:
        """Find and execute due tasks."""
        due_tasks = [t for t in self._tasks if t.is_due()]
        for task in due_tasks:
            try:
                if self._callback:
                    await self._callback(task)
                    logger.info("task_executed", task_id=task.id, desc=task.description[:50])

                if task.recurring:
                    task.reschedule()
                else:
                    task.completed = True

                self._save()
            except Exception as e:
                logger.error("task_execution_failed", task_id=task.id, error=str(e))

    async def schedule(
        self,
        chat_id: int,
        description: str,
        trigger_time: datetime,
        recurring: str | None = None,
    ) -> ScheduledTask:
        """Schedule a new task."""
        if trigger_time.tzinfo is None:
            trigger_time = trigger_time.replace(tzinfo=timezone.utc)

        task = ScheduledTask(
            chat_id=chat_id,
            description=description,
            trigger_time=trigger_time.isoformat(),
            recurring=recurring,
        )
        self._tasks.append(task)
        self._save()
        logger.info(
            "task_scheduled",
            task_id=task.id,
            trigger=task.trigger_time,
            recurring=recurring,
        )
        return task

    async def cancel(self, task_id: str) -> bool:
        """Cancel a task by ID."""
        for task in self._tasks:
            if task.id == task_id and not task.completed:
                task.completed = True
                self._save()
                return True
        return False

    async def list_tasks(self, chat_id: int | None = None) -> list[ScheduledTask]:
        """List pending (non-completed) tasks, optionally filtered by chat_id."""
        tasks = [t for t in self._tasks if not t.completed]
        if chat_id is not None:
            tasks = [t for t in tasks if t.chat_id == chat_id]
        return tasks

    async def cleanup_completed(self) -> int:
        """Remove completed non-recurring tasks older than 24h."""
        now = datetime.now(timezone.utc)
        before = len(self._tasks)
        self._tasks = [
            t for t in self._tasks
            if not t.completed or (
                t.completed and t.created_at and
                (now - datetime.fromisoformat(t.created_at)).total_seconds() < 86400
            )
        ]
        removed = before - len(self._tasks)
        if removed > 0:
            self._save()
        return removed
