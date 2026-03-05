"""Memory API routes."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from lya.domain.models.memory import Memory, MemoryType, MemoryImportance, MemoryContext
from lya.domain.repositories.memory_repo import MemoryRepository
from lya.infrastructure.config.settings import settings
from lya.infrastructure.persistence.chroma_memory_repository import ChromaMemoryRepository
from lya.infrastructure.persistence.embedding_service import get_embedding_service

router = APIRouter()

# Initialize repository
_memory_repo: MemoryRepository | None = None


def get_memory_repository() -> MemoryRepository:
    """Get or create memory repository instance."""
    global _memory_repo
    if _memory_repo is None:
        # Create embedding service for semantic search
        embedding_service = get_embedding_service()

        # Create repository with persistence directory from settings
        _memory_repo = ChromaMemoryRepository(
            collection_name="memories",
            persist_directory=str(settings.memory.db_path),
            embedding_function=embedding_service,
        )
    return _memory_repo


class CreateMemoryRequest(BaseModel):
    """Request model for creating a memory."""
    content: str = Field(..., min_length=1, max_length=10000)
    memory_type: str = Field(default="episodic")
    importance: str = Field(default="medium")
    agent_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    """Response model for a memory."""
    id: str
    content: str
    type: str
    importance: str
    agent_id: str | None = None
    created_at: str
    accessed_at: str | None = None
    access_count: int


class SearchMemoryRequest(BaseModel):
    """Request model for searching memories."""
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0, le=1)
    memory_type: str | None = None


class SearchMemoryResponse(BaseModel):
    """Response model for memory search."""
    results: list[dict[str, Any]]
    total: int


@router.post("/", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(request: CreateMemoryRequest) -> dict[str, Any]:
    """Create a new memory."""
    try:
        memory_type = MemoryType[request.memory_type.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid memory type: {request.memory_type}",
        )

    try:
        importance = MemoryImportance[request.importance.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid importance: {request.importance}",
        )

    agent_id = UUID(request.agent_id) if request.agent_id else None

    memory = Memory(
        agent_id=agent_id,
        content=request.content,
        memory_type=memory_type,
        importance=importance,
        context=MemoryContext(tags=request.tags, metadata=request.metadata),
    )

    # Generate embedding for semantic search
    embedding_service = get_embedding_service()
    embedding = await embedding_service.embed(request.content)
    memory.set_embedding(embedding)

    # Save to repository
    repo = get_memory_repository()
    await repo.save(memory)

    return _memory_to_response(memory)


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str) -> dict[str, Any]:
    """Get a memory by ID."""
    try:
        memory_uuid = UUID(memory_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid memory ID format",
        )

    repo = get_memory_repository()
    memory = await repo.get(memory_uuid)

    if memory is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found",
        )

    return _memory_to_response(memory)


@router.post("/search", response_model=SearchMemoryResponse)
async def search_memories(request: SearchMemoryRequest) -> dict[str, Any]:
    """Search memories by semantic similarity."""
    repo = get_memory_repository()
    embedding_service = get_embedding_service()

    # Generate embedding for the query
    query_embedding = await embedding_service.embed(request.query)

    # Parse memory type filter if provided
    memory_type_filter = None
    if request.memory_type:
        try:
            memory_type_filter = MemoryType[request.memory_type.upper()].name
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid memory type: {request.memory_type}",
            )

    # Search using semantic similarity
    results = await repo.search_similar(
        query_embedding=query_embedding,
        limit=request.limit,
        min_relevance=request.threshold,
    )

    # Filter by memory type if specified
    if memory_type_filter:
        results = [(memory, score) for memory, score in results if memory.type.name == memory_type_filter]

    return {
        "results": [
            {
                **(_memory_to_response(memory)),
                "similarity_score": round(score, 4),
            }
            for memory, score in results
        ],
        "total": len(results),
    }


@router.get("/agent/{agent_id}")
async def get_agent_memories(
    agent_id: str,
    memory_type: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Get memories for an agent."""
    try:
        agent_uuid = UUID(agent_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid agent ID format",
        )

    repo = get_memory_repository()

    # Parse memory type filter if provided
    memory_type_filter = None
    if memory_type:
        try:
            memory_type_filter = MemoryType[memory_type.upper()].name
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid memory type: {memory_type}",
            )

    # Get memories for agent
    memories = await repo.get_by_agent(
        agent_id=agent_uuid,
        memory_type=memory_type_filter,
        limit=limit,
    )

    return {
        "items": [_memory_to_response(m) for m in memories],
        "total": len(memories),
    }


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(memory_id: str) -> None:
    """Delete a memory."""
    try:
        memory_uuid = UUID(memory_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid memory ID format",
        )

    repo = get_memory_repository()
    deleted = await repo.delete(memory_uuid)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found",
        )


def _memory_to_response(memory: Memory) -> dict[str, Any]:
    """Convert Memory to response dict."""
    return {
        "id": str(memory.id),
        "content": memory.content[:200] + "..." if len(memory.content) > 200 else memory.content,
        "type": memory.type.name.lower(),
        "importance": memory.importance.name.lower(),
        "agent_id": str(memory.agent_id) if memory.agent_id else None,
        "created_at": memory.created_at.isoformat(),
        "accessed_at": memory.accessed_at.isoformat() if memory.accessed_at else None,
        "access_count": memory.access_count,
    }
