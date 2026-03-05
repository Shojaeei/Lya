"""Goal API routes."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from lya.domain.models.goal import Goal, GoalPriority, GoalStatus
from lya.domain.repositories.goal_repo import GoalRepository
from lya.infrastructure.persistence.file_goal_repository import FileGoalRepository

router = APIRouter()

# Initialize repository
_goal_repo: GoalRepository | None = None


def get_goal_repository() -> GoalRepository:
    """Get or create goal repository instance."""
    global _goal_repo
    if _goal_repo is None:
        _goal_repo = FileGoalRepository()
    return _goal_repo


# ═══════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════

class CreateGoalRequest(BaseModel):
    """Request model for creating a goal."""
    description: str = Field(..., min_length=1, max_length=1000)
    priority: str = Field(default="medium")
    parent_id: str | None = None
    deadline: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateGoalRequest(BaseModel):
    """Request model for updating a goal."""
    description: str | None = None
    status: str | None = None
    priority: str | None = None
    metadata: dict[str, Any] | None = None


class GoalResponse(BaseModel):
    """Response model for a goal."""
    id: str
    description: str
    status: str
    priority: str
    created_at: str
    updated_at: str | None = None
    completed_at: str | None = None
    parent_id: str | None = None
    sub_goal_ids: list[str]
    metadata: dict[str, Any]


# ═══════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════

@router.post("/", response_model=GoalResponse, status_code=status.HTTP_201_CREATED)
async def create_goal(request: CreateGoalRequest) -> dict[str, Any]:
    """Create a new goal."""
    try:
        priority = GoalPriority[request.priority.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid priority: {request.priority}",
        )

    parent_id = UUID(request.parent_id) if request.parent_id else None

    goal = Goal(
        description=request.description,
        priority=priority,
        parent_id=parent_id,
    )

    repo = get_goal_repository()
    await repo.save(goal)

    return _goal_to_response(goal)


@router.get("/{goal_id}", response_model=GoalResponse)
async def get_goal(goal_id: str) -> dict[str, Any]:
    """Get a goal by ID."""
    try:
        goal_uuid = UUID(goal_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid goal ID format",
        )

    repo = get_goal_repository()
    goal = await repo.get(goal_uuid)

    if goal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Goal {goal_id} not found",
        )

    return _goal_to_response(goal)


@router.get("/")
async def list_goals(
    status: str | None = None,
    priority: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """List goals with optional filtering."""
    repo = get_goal_repository()

    # Get all goals (for now, agent_id is not used)
    goal_status = None
    if status:
        try:
            goal_status = GoalStatus[status.upper()]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status}",
            )

    goals = await repo.get_by_agent(
        agent_id=UUID(int=0),  # Placeholder - get all goals
        status=goal_status,
        limit=limit,
    )

    # Apply offset
    goals = goals[offset:offset + limit]

    total = await repo.count(status=goal_status)

    return {
        "items": [_goal_to_response(g) for g in goals],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.patch("/{goal_id}", response_model=GoalResponse)
async def update_goal(goal_id: str, request: UpdateGoalRequest) -> dict[str, Any]:
    """Update a goal."""
    try:
        goal_uuid = UUID(goal_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid goal ID format",
        )

    repo = get_goal_repository()
    goal = await repo.get(goal_uuid)

    if goal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Goal {goal_id} not found",
        )

    if request.status:
        try:
            goal_status = GoalStatus[request.status.upper()]
            if goal_status == GoalStatus.IN_PROGRESS:
                goal.start()
            elif goal_status == GoalStatus.COMPLETED:
                goal.complete()
            else:
                goal._status = goal_status
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {request.status}",
            )

    if request.priority:
        try:
            goal._priority = GoalPriority[request.priority.upper()]
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid priority: {request.priority}",
            )

    await repo.save(goal)
    return _goal_to_response(goal)


@router.delete("/{goal_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_goal(goal_id: str) -> None:
    """Delete a goal."""
    try:
        goal_uuid = UUID(goal_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid goal ID format",
        )

    repo = get_goal_repository()
    deleted = await repo.delete(goal_uuid)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Goal {goal_id} not found",
        )


@router.post("/{goal_id}/subgoals", response_model=GoalResponse, status_code=status.HTTP_201_CREATED)
async def create_sub_goal(goal_id: str, request: CreateGoalRequest) -> dict[str, Any]:
    """Create a sub-goal."""
    try:
        parent_uuid = UUID(goal_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid goal ID format",
        )

    try:
        priority = GoalPriority[request.priority.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid priority: {request.priority}",
        )

    repo = get_goal_repository()
    parent = await repo.get(parent_uuid)

    if parent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Parent goal {goal_id} not found",
        )

    sub_goal = parent.create_sub_goal(request.description, priority)
    await repo.save(sub_goal)
    await repo.save(parent)  # Save parent to update sub_goal_ids

    return _goal_to_response(sub_goal)


# ═══════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════

def _goal_to_response(goal: Goal) -> dict[str, Any]:
    """Convert Goal to response dict."""
    return {
        "id": str(goal.id),
        "description": goal.description,
        "status": goal.status.name.lower(),
        "priority": goal.priority.name.lower(),
        "created_at": goal.created_at.isoformat(),
        "updated_at": goal.updated_at.isoformat() if goal.updated_at else None,
        "completed_at": goal.completed_at.isoformat() if goal.completed_at else None,
        "parent_id": str(goal.parent_id) if goal.parent_id else None,
        "sub_goal_ids": [str(gid) for gid in goal.sub_goal_ids],
        "metadata": goal._metadata,
    }
