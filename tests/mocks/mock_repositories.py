"""Mock repository implementations."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from lya.domain.repositories.goal_repo import GoalRepository
from lya.domain.repositories.memory_repo import MemoryRepository
from lya.domain.repositories.task_repo import TaskRepository
from lya.domain.models.goal import Goal
from lya.domain.models.memory import Memory
from lya.domain.models.task import Task


class MockMemoryRepository(MemoryRepository):
    """In-memory mock implementation of MemoryRepository."""

    def __init__(self):
        self._memories: dict[str, Memory] = {}
        self.call_history: list[dict[str, Any]] = []

    def _log_call(self, method: str, **kwargs) -> None:
        """Log method call for verification."""
        self.call_history.append({"method": method, "args": kwargs})

    async def get(self, memory_id: UUID) -> Memory | None:
        """Get memory by ID."""
        self._log_call("get", memory_id=memory_id)
        return self._memories.get(str(memory_id))

    async def save(self, memory: Memory) -> None:
        """Save memory."""
        self._log_call("save", memory_id=memory.id)
        self._memories[str(memory.id)] = memory

    async def delete(self, memory_id: UUID) -> bool:
        """Delete memory."""
        self._log_call("delete", memory_id=memory_id)
        key = str(memory_id)
        if key in self._memories:
            del self._memories[key]
            return True
        return False

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
        memory_type: str | None = None,
        agent_id: UUID | None = None,
    ) -> list[tuple[Memory, float]]:
        """Search memories."""
        self._log_call("search", limit=limit, threshold=threshold)

        results = []
        for memory in self._memories.values():
            # Filter by type
            if memory_type and memory.type.name != memory_type:
                continue
            # Filter by agent
            if agent_id and memory.agent_id != agent_id:
                continue

            # Calculate mock similarity
            score = 0.5  # Mock score
            if score >= threshold:
                results.append((memory, score))

        # Sort by score and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def get_by_agent(
        self,
        agent_id: UUID,
        memory_type: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories by agent."""
        self._log_call("get_by_agent", agent_id=agent_id)

        results = [
            m for m in self._memories.values()
            if m.agent_id == agent_id
            and (not memory_type or m.type.name == memory_type)
        ]
        return results[:limit]

    async def get_by_goal(self, goal_id: UUID, limit: int = 100) -> list[Memory]:
        """Get memories by goal."""
        self._log_call("get_by_goal", goal_id=goal_id)

        results = [
            m for m in self._memories.values()
            if m.context and m.context.goal_id == goal_id
        ]
        return results[:limit]

    async def delete_by_agent(self, agent_id: UUID) -> int:
        """Delete all agent memories."""
        self._log_call("delete_by_agent", agent_id=agent_id)

        to_delete = [
            key for key, m in self._memories.items()
            if m.agent_id == agent_id
        ]
        for key in to_delete:
            del self._memories[key]
        return len(to_delete)

    async def get_stale_memories(
        self,
        threshold_days: int = 30,
        limit: int = 100,
    ) -> list[Memory]:
        """Get stale memories."""
        self._log_call("get_stale_memories", threshold_days=threshold_days)

        results = [
            m for m in self._memories.values()
            if m.should_forget(threshold_days)
        ]
        return results[:limit]

    async def count(self, agent_id: UUID | None = None) -> int:
        """Count memories."""
        self._log_call("count", agent_id=agent_id)

        if agent_id:
            return len([m for m in self._memories.values() if m.agent_id == agent_id])
        return len(self._memories)


class MockGoalRepository(GoalRepository):
    """In-memory mock implementation of GoalRepository."""

    def __init__(self):
        self._goals: dict[str, Goal] = {}
        self.call_history: list[dict[str, Any]] = []

    def _log_call(self, method: str, **kwargs) -> None:
        self.call_history.append({"method": method, "args": kwargs})

    async def get(self, goal_id: UUID) -> Goal | None:
        """Get goal by ID."""
        self._log_call("get", goal_id=goal_id)
        return self._goals.get(str(goal_id))

    async def save(self, goal: Goal) -> None:
        """Save goal."""
        self._log_call("save", goal_id=goal.id)
        self._goals[str(goal.id)] = goal

    async def delete(self, goal_id: UUID) -> bool:
        """Delete goal."""
        self._log_call("delete", goal_id=goal_id)
        key = str(goal_id)
        if key in self._goals:
            del self._goals[key]
            return True
        return False

    async def get_active(self, limit: int = 100) -> list[Goal]:
        """Get active goals."""
        self._log_call("get_active")

        from lya.domain.models.goal import GoalStatus
        results = [
            g for g in self._goals.values()
            if g.status == GoalStatus.PENDING or g.status == GoalStatus.IN_PROGRESS
        ]
        return results[:limit]

    async def get_by_agent(self, agent_id: UUID, limit: int = 100) -> list[Goal]:
        """Get goals by agent."""
        self._log_call("get_by_agent", agent_id=agent_id)

        results = [
            g for g in self._goals.values()
            if g._metadata.get("agent_id") == str(agent_id)
        ]
        return results[:limit]

    async def get_with_tasks(self, goal_id: UUID) -> Goal | None:
        """Get goal with tasks."""
        self._log_call("get_with_tasks", goal_id=goal_id)
        return await self.get(goal_id)


class MockTaskRepository(TaskRepository):
    """In-memory mock implementation of TaskRepository."""

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self.call_history: list[dict[str, Any]] = []

    def _log_call(self, method: str, **kwargs) -> None:
        self.call_history.append({"method": method, "args": kwargs})

    async def get(self, task_id: UUID) -> Task | None:
        """Get task by ID."""
        self._log_call("get", task_id=task_id)
        return self._tasks.get(str(task_id))

    async def save(self, task: Task) -> None:
        """Save task."""
        self._log_call("save", task_id=task.id)
        self._tasks[str(task.id)] = task

    async def delete(self, task_id: UUID) -> bool:
        """Delete task."""
        self._log_call("delete", task_id=task_id)
        key = str(task_id)
        if key in self._tasks:
            del self._tasks[key]
            return True
        return False

    async def get_by_goal(self, goal_id: UUID, status: str | None = None) -> list[Task]:
        """Get tasks by goal."""
        self._log_call("get_by_goal", goal_id=goal_id, status=status)

        results = [
            t for t in self._tasks.values()
            if t.goal_id == goal_id
            and (not status or t.status.name == status)
        ]
        return results

    async def get_active(self, limit: int = 100) -> list[Task]:
        """Get active tasks."""
        self._log_call("get_active")

        from lya.domain.models.task import TaskStatus
        results = [
            t for t in self._tasks.values()
            if t.status == TaskStatus.PENDING or t.status == TaskStatus.IN_PROGRESS
        ]
        return results[:limit]

    async def get_by_agent(self, agent_id: UUID, limit: int = 100) -> list[Task]:
        """Get tasks by agent."""
        self._log_call("get_by_agent", agent_id=agent_id)

        results = [
            t for t in self._tasks.values()
            if t._metadata.get("agent_id") == str(agent_id)
        ]
        return results[:limit]

    async def get_by_goal_tree(self, goal_id: UUID) -> list[Task]:
        """Get tasks by goal tree."""
        self._log_call("get_by_goal_tree", goal_id=goal_id)
        return await self.get_by_goal(goal_id)
