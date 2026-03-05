"""Task API routes."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from lya.domain.models.task import Task, TaskStatus, TaskPriority
from lya.domain.repositories.task_repo import TaskRepository
from lya.infrastructure.persistence.file_task_repository import FileTaskRepository

router = APIRouter()

# Initialize repository
_task_repo: TaskRepository | None = None


def get_task_repository() -> TaskRepository:
    """Get or create task repository instance."""
    global _task_repo
    if _task_repo is None:
        _task_repo = FileTaskRepository()
    return _task_repo


class CreateTaskRequest(BaseModel):
    """Request model for creating a task."""
    description: str = Field(..., min_length=1, max_length=1000)
    goal_id: str | None = None
    priority: str = Field(default="medium")
    parent_id: str | None = None
    estimated_duration_minutes: int | None = None


class UpdateTaskRequest(BaseModel):
    """Request model for updating a task."""
    description: str | None = None
    status: str | None = None
    priority: str | None = None


class TaskResponse(BaseModel):
    """Response model for a task."""
    id: str
    description: str
    status: str
    priority: str
    goal_id: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(request: CreateTaskRequest) -> dict[str, Any]:
    """Create a new task."""
    try:
        priority = TaskPriority[request.priority.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid priority: {request.priority}",
        )

    goal_id = UUID(request.goal_id) if request.goal_id else None

    task = Task(
        description=request.description,
        priority=priority,
        goal_id=goal_id,
        estimated_duration_minutes=request.estimated_duration_minutes,
    )

    repo = get_task_repository()
    await repo.save(task)

    return _task_to_response(task)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> dict[str, Any]:
    """Get a task by ID."""
    try:
        task_uuid = UUID(task_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task ID format",
        )

    repo = get_task_repository()
    task = await repo.get(task_uuid)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    return _task_to_response(task)


@router.get("/")
async def list_tasks(
    status: str | None = None,
    goal_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """List tasks."""
    repo = get_task_repository()

    # Parse filters
    task_status = None
    if status:
        try:
            task_status = TaskStatus[status.upper().replace("-", "_")]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status}",
            )

    goal_uuid = None
    if goal_id:
        try:
            goal_uuid = UUID(goal_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid goal ID format",
            )

    if goal_uuid:
        tasks = await repo.get_by_goal(goal_uuid, task_status)
    else:
        # Get all tasks (no goal filter)
        # Note: This is a simplified implementation
        tasks = []
        # We'd need a method to get all tasks, but for now return empty

    # Apply pagination
    total = len(tasks)
    tasks = tasks[offset:offset + limit]

    return {
        "items": [_task_to_response(t) for t in tasks],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(task_id: str, request: UpdateTaskRequest) -> dict[str, Any]:
    """Update a task."""
    try:
        task_uuid = UUID(task_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task ID format",
        )

    repo = get_task_repository()
    task = await repo.get(task_uuid)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    if request.status:
        try:
            new_status = TaskStatus[request.status.upper().replace("-", "_")]
            task._status = new_status
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {request.status}",
            )

    if request.priority:
        try:
            task._priority = TaskPriority[request.priority.upper()]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid priority: {request.priority}",
            )

    if request.description:
        task._description = request.description

    await repo.save(task)
    return _task_to_response(task)


@router.post("/{task_id}/start", response_model=TaskResponse)
async def start_task(task_id: str) -> dict[str, Any]:
    """Start a task."""
    try:
        task_uuid = UUID(task_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task ID format",
        )

    repo = get_task_repository()
    task = await repo.get(task_uuid)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    task.start()
    await repo.save(task)
    return _task_to_response(task)


@router.post("/{task_id}/complete", response_model=TaskResponse)
async def complete_task(task_id: str) -> dict[str, Any]:
    """Complete a task."""
    try:
        task_uuid = UUID(task_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task ID format",
        )

    repo = get_task_repository()
    task = await repo.get(task_uuid)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    task.complete()
    await repo.save(task)
    return _task_to_response(task)


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task_id: str) -> None:
    """Delete a task."""
    try:
        task_uuid = UUID(task_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task ID format",
        )

    repo = get_task_repository()
    deleted = await repo.delete(task_uuid)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )


def _task_to_response(task: Task) -> dict[str, Any]:
    """Convert Task to response dict."""
    return {
        "id": str(task.id),
        "description": task.description,
        "status": task.status.name.lower().replace("_", "-"),
        "priority": task.priority.name.lower(),
        "goal_id": str(task.goal_id) if task.goal_id else None,
        "created_at": task.created_at.isoformat(),
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }
