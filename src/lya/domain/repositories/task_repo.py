"""Task repository interface."""

from abc import ABC, abstractmethod
from uuid import UUID

from lya.domain.models.task import Task, TaskStatus


class TaskRepository(ABC):
    """
    Repository interface for Task aggregate.

    This interface defines the contract for task persistence.
    Implementations can use any storage backend.
    """

    @abstractmethod
    async def get(self, task_id: UUID) -> Task | None:
        """
        Retrieve a task by ID.

        Args:
            task_id: Unique identifier of the task

        Returns:
            Task if found, None otherwise
        """
        pass

    @abstractmethod
    async def save(self, task: Task) -> None:
        """
        Save a task.

        Args:
            task: Task to save
        """
        pass

    @abstractmethod
    async def save_many(self, tasks: list[Task]) -> None:
        """
        Save multiple tasks.

        Args:
            tasks: List of tasks to save
        """
        pass

    @abstractmethod
    async def delete(self, task_id: UUID) -> bool:
        """
        Delete a task.

        Args:
            task_id: ID of task to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def get_by_goal(
        self,
        goal_id: UUID,
        status: TaskStatus | None = None,
    ) -> list[Task]:
        """
        Get tasks for a specific goal.

        Args:
            goal_id: Goal ID
            status: Optional status filter

        Returns:
            List of tasks ordered by execution order
        """
        pass

    @abstractmethod
    async def get_pending_tasks(self, goal_id: UUID) -> list[Task]:
        """
        Get pending tasks for a goal.

        Args:
            goal_id: Goal ID

        Returns:
            List of pending tasks
        """
        pass

    @abstractmethod
    async def get_executable_tasks(self, goal_id: UUID) -> list[Task]:
        """
        Get tasks ready for execution (pending with satisfied dependencies).

        Args:
            goal_id: Goal ID

        Returns:
            List of executable tasks
        """
        pass

    @abstractmethod
    async def get_completed_tasks(self, goal_id: UUID) -> list[Task]:
        """
        Get completed tasks for a goal.

        Args:
            goal_id: Goal ID

        Returns:
            List of completed tasks
        """
        pass

    @abstractmethod
    async def delete_by_goal(self, goal_id: UUID) -> int:
        """
        Delete all tasks for a goal.

        Args:
            goal_id: Goal ID

        Returns:
            Number of tasks deleted
        """
        pass

    @abstractmethod
    async def count(
        self,
        goal_id: UUID | None = None,
        status: TaskStatus | None = None,
    ) -> int:
        """
        Count tasks.

        Args:
            goal_id: Optional goal filter
            status: Optional status filter

        Returns:
            Total count
        """
        pass
