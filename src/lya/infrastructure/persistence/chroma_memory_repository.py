"""ChromaDB memory repository implementation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
except (ImportError, Exception):
    # chromadb may fail on Python 3.14+ due to pydantic v1 incompatibility
    HAS_CHROMADB = False

from lya.domain.models.memory import Memory, MemoryContext, MemoryImportance, MemoryType
from lya.domain.repositories.memory_repo import MemoryRepository
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class ChromaMemoryRepository(MemoryRepository):
    """
    ChromaDB-based memory repository.

    Provides semantic search capabilities through vector embeddings
    and efficient storage/retrieval of memories.
    """

    def __init__(
        self,
        collection_name: str = "memories",
        persist_directory: str | None = None,
        embedding_function: Any | None = None,
    ):
        if not HAS_CHROMADB:
            raise ImportError(
                "ChromaDB is required. Install with: pip install chromadb"
            )

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self._client: chromadb.Client | None = None
        self._collection: chromadb.Collection | None = None

    async def _get_client(self) -> chromadb.Client:
        """Get or create ChromaDB client."""
        if self._client is None:
            settings = ChromaSettings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
            )
            self._client = chromadb.Client(settings)
        return self._client

    async def _get_collection(self) -> chromadb.Collection:
        """Get or create memory collection."""
        if self._collection is None:
            client = await self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )
        return self._collection

    async def initialize(self) -> None:
        """Initialize the repository."""
        await self._get_collection()
        logger.info(
            "ChromaDB memory repository initialized",
            collection=self.collection_name,
            persist_directory=self.persist_directory,
        )

    async def close(self) -> None:
        """Close the repository."""
        if self._client and self.persist_directory:
            # ChromaDB persists automatically
            pass
        self._collection = None
        self._client = None

    # ═══════════════════════════════════════════════════════════════
    # CRUD Operations
    # ═══════════════════════════════════════════════════════════════

    async def save(self, memory: Memory) -> None:
        """Save a memory to the repository."""
        collection = await self._get_collection()

        # Convert memory to document format
        document = memory.content
        metadata = {
            "agent_id": str(memory.agent_id) if memory.agent_id else None,
            "type": memory.type.name,
            "importance": memory.importance.name,
            "importance_value": memory.importance_value,
            "created_at": memory.created_at.isoformat(),
            "accessed_at": memory.accessed_at.isoformat() if memory.accessed_at else None,
            "access_count": memory.access_count,
            "is_consolidated": memory.is_consolidated,
            "goal_id": str(memory.context.goal_id) if memory.context.goal_id else None,
            "task_id": str(memory.context.task_id) if memory.context.task_id else None,
            "source": memory.context.source,
            "tags": ",".join(memory.context.tags),
            **{f"meta_{k}": str(v) for k, v in memory.context.metadata.items()},
        }

        # Add to collection
        collection.add(
            ids=[str(memory.id)],
            documents=[document],
            metadatas=[metadata],
            embeddings=[memory.embedding] if memory.embedding else None,
        )

        logger.debug("Memory saved", memory_id=str(memory.id), type=memory.type.name)

    async def get(self, memory_id: UUID) -> Memory | None:
        """Get a memory by ID."""
        collection = await self._get_collection()

        try:
            result = collection.get(
                ids=[str(memory_id)],
                include=["documents", "metadatas", "embeddings"],
            )

            if not result["ids"]:
                return None

            return self._convert_to_memory(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["embeddings"][0] if result["embeddings"] else None,
            )

        except Exception as e:
            logger.error("Failed to get memory", memory_id=str(memory_id), error=str(e))
            return None

    async def update(self, memory: Memory) -> None:
        """Update an existing memory."""
        collection = await self._get_collection()

        # Update the document
        metadata = {
            "agent_id": str(memory.agent_id) if memory.agent_id else None,
            "type": memory.type.name,
            "importance": memory.importance.name,
            "importance_value": memory.importance_value,
            "created_at": memory.created_at.isoformat(),
            "accessed_at": memory.accessed_at.isoformat() if memory.accessed_at else None,
            "access_count": memory.access_count,
            "is_consolidated": memory.is_consolidated,
            "goal_id": str(memory.context.goal_id) if memory.context.goal_id else None,
            "task_id": str(memory.context.task_id) if memory.context.task_id else None,
            "source": memory.context.source,
            "tags": ",".join(memory.context.tags),
            **{f"meta_{k}": str(v) for k, v in memory.context.metadata.items()},
        }

        collection.update(
            ids=[str(memory.id)],
            documents=[memory.content],
            metadatas=[metadata],
            embeddings=[memory.embedding] if memory.embedding else None,
        )

        logger.debug("Memory updated", memory_id=str(memory.id))

    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory by ID."""
        collection = await self._get_collection()

        try:
            collection.delete(ids=[str(memory_id)])
            logger.debug("Memory deleted", memory_id=str(memory_id))
            return True
        except Exception as e:
            logger.error("Failed to delete memory", memory_id=str(memory_id), error=str(e))
            return False

    # ═══════════════════════════════════════════════════════════════
    # Query Operations
    # ═══════════════════════════════════════════════════════════════

    async def get_by_agent(self, agent_id: UUID, limit: int = 100) -> list[Memory]:
        """Get all memories for an agent."""
        collection = await self._get_collection()

        result = collection.get(
            where={"agent_id": str(agent_id)},
            limit=limit,
            include=["documents", "metadatas", "embeddings"],
        )

        return [
            self._convert_to_memory(
                result["ids"][i],
                result["documents"][i],
                result["metadatas"][i],
                result["embeddings"][i] if result["embeddings"] else None,
            )
            for i in range(len(result["ids"]))
        ]

    async def search_similar(
        self,
        query_embedding: list[float],
        agent_id: UUID | None = None,
        limit: int = 10,
        min_relevance: float = 0.7,
    ) -> list[tuple[Memory, float]]:
        """Search for similar memories using vector similarity."""
        collection = await self._get_collection()

        # Build filter
        where_filter = None
        if agent_id:
            where_filter = {"agent_id": str(agent_id)}

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter,
            include=["documents", "metadatas", "embeddings", "distances"],
        )

        memories_with_scores = []
        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                # ChromaDB returns cosine distance, convert to similarity
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Convert distance to similarity

                if similarity >= min_relevance:
                    memory = self._convert_to_memory(
                        memory_id,
                        results["documents"][0][i],
                        results["metadatas"][0][i],
                        results["embeddings"][0][i] if results["embeddings"] else None,
                    )
                    memories_with_scores.append((memory, similarity))

        return memories_with_scores

    async def search_by_content(
        self,
        query: str,
        agent_id: UUID | None = None,
        limit: int = 10,
    ) -> list[Memory]:
        """Search memories by text content."""
        collection = await self._get_collection()

        where_filter = None
        if agent_id:
            where_filter = {"agent_id": str(agent_id)}

        results = collection.query(
            query_texts=[query],
            n_results=limit,
            where=where_filter,
            include=["documents", "metadatas", "embeddings"],
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                memory = self._convert_to_memory(
                    memory_id,
                    results["documents"][0][i],
                    results["metadatas"][0][i],
                    results["embeddings"][0][i] if results["embeddings"] else None,
                )
                memories.append(memory)

        return memories

    async def get_by_type(
        self,
        memory_type: MemoryType,
        agent_id: UUID | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories by type."""
        collection = await self._get_collection()

        where_filter: dict[str, Any] = {"type": memory_type.name}
        if agent_id:
            where_filter["agent_id"] = str(agent_id)

        results = collection.get(
            where=where_filter,
            limit=limit,
            include=["documents", "metadatas", "embeddings"],
        )

        return [
            self._convert_to_memory(
                results["ids"][i],
                results["documents"][i],
                results["metadatas"][i],
                results["embeddings"][i] if results["embeddings"] else None,
            )
            for i in range(len(results["ids"]))
        ]

    async def get_by_importance(
        self,
        min_importance: MemoryImportance,
        agent_id: UUID | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories by minimum importance level."""
        collection = await self._get_collection()

        # Get all memories and filter by importance
        where_filter = None
        if agent_id:
            where_filter = {"agent_id": str(agent_id)}

        results = collection.get(
            where=where_filter,
            limit=limit * 2,  # Get more to filter
            include=["documents", "metadatas", "embeddings"],
        )

        memories = []
        for i in range(len(results["ids"])):
            metadata = results["metadatas"][i]
            importance_value = MemoryImportance[metadata["importance"]].value

            if importance_value >= min_importance.value:
                memory = self._convert_to_memory(
                    results["ids"][i],
                    results["documents"][i],
                    metadata,
                    results["embeddings"][i] if results["embeddings"] else None,
                )
                memories.append(memory)

        return memories[:limit]

    async def get_recent(
        self,
        hours: int = 24,
        agent_id: UUID | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories from recent time period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()

        collection = await self._get_collection()

        where_filter: dict[str, Any] = {"created_at": {"$gte": cutoff_str}}
        if agent_id:
            where_filter["agent_id"] = str(agent_id)

        results = collection.get(
            where=where_filter,
            limit=limit,
            include=["documents", "metadatas", "embeddings"],
        )

        return [
            self._convert_to_memory(
                results["ids"][i],
                results["documents"][i],
                results["metadatas"][i],
                results["embeddings"][i] if results["embeddings"] else None,
            )
            for i in range(len(results["ids"]))
        ]

    async def get_by_tags(
        self,
        tags: list[str],
        agent_id: UUID | None = None,
        match_all: bool = False,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories by tags."""
        # ChromaDB doesn't support array contains, so we search manually
        collection = await self._get_collection()

        where_filter = None
        if agent_id:
            where_filter = {"agent_id": str(agent_id)}

        results = collection.get(
            where=where_filter,
            limit=limit * 2,
            include=["documents", "metadatas", "embeddings"],
        )

        memories = []
        for i in range(len(results["ids"])):
            metadata = results["metadatas"][i]
            memory_tags = metadata.get("tags", "").split(",") if metadata.get("tags") else []

            if match_all:
                # All tags must be present
                if all(tag in memory_tags for tag in tags):
                    memory = self._convert_to_memory(
                        results["ids"][i],
                        results["documents"][i],
                        metadata,
                        results["embeddings"][i] if results["embeddings"] else None,
                    )
                    memories.append(memory)
            else:
                # Any tag can match
                if any(tag in memory_tags for tag in tags):
                    memory = self._convert_to_memory(
                        results["ids"][i],
                        results["documents"][i],
                        metadata,
                        results["embeddings"][i] if results["embeddings"] else None,
                    )
                    memories.append(memory)

        return memories[:limit]

    # ═══════════════════════════════════════════════════════════════
    # Maintenance Operations
    # ═══════════════════════════════════════════════════════════════

    async def get_forgettable_memories(
        self,
        threshold_days: int = 30,
        agent_id: UUID | None = None,
    ) -> list[Memory]:
        """Get memories that should be forgotten."""
        collection = await self._get_collection()

        where_filter: dict[str, Any] = {
            "importance": {"$ne": MemoryImportance.CRITICAL.name},
        }
        if agent_id:
            where_filter["agent_id"] = str(agent_id)

        results = collection.get(
            where=where_filter,
            include=["documents", "metadatas", "embeddings"],
        )

        forgettable = []
        for i in range(len(results["ids"])):
            memory = self._convert_to_memory(
                results["ids"][i],
                results["documents"][i],
                results["metadatas"][i],
                results["embeddings"][i] if results["embeddings"] else None,
            )

            if memory.should_forget(threshold_days):
                forgettable.append(memory)

        return forgettable

    async def consolidate_memories(self, agent_id: UUID | None = None) -> int:
        """Mark old memories as consolidated."""
        collection = await self._get_collection()

        where_filter: dict[str, Any] = {"is_consolidated": False}
        if agent_id:
            where_filter["agent_id"] = str(agent_id)

        results = collection.get(
            where=where_filter,
            include=["metadatas"],
        )

        consolidated_count = 0
        for i, memory_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            created_at = datetime.fromisoformat(metadata["created_at"])

            # Mark memories older than 24 hours as consolidated
            if datetime.now(timezone.utc) - created_at > timedelta(hours=24):
                collection.update(
                    ids=[memory_id],
                    metadatas=[{**metadata, "is_consolidated": True}],
                )
                consolidated_count += 1

        logger.info(
            "Memory consolidation complete",
            consolidated_count=consolidated_count,
        )

        return consolidated_count

    async def count(self, agent_id: UUID | None = None) -> int:
        """Count total memories."""
        collection = await self._get_collection()

        if agent_id:
            results = collection.get(where={"agent_id": str(agent_id)})
        else:
            results = collection.get()

        return len(results["ids"])

    async def clear(self, agent_id: UUID | None = None) -> int:
        """Clear all memories (optionally for specific agent)."""
        collection = await self._get_collection()

        if agent_id:
            results = collection.get(where={"agent_id": str(agent_id)})
            if results["ids"]:
                collection.delete(ids=results["ids"])
            return len(results["ids"])
        else:
            # Delete entire collection and recreate
            client = await self._get_client()
            client.delete_collection(self.collection_name)
            self._collection = None
            return -1  # Unknown count

    # ═══════════════════════════════════════════════════════════════
    # Helper Methods
    # ═══════════════════════════════════════════════════════════════

    def _convert_to_memory(
        self,
        memory_id: str,
        document: str,
        metadata: dict[str, Any],
        embedding: list[float] | None,
    ) -> Memory:
        """Convert ChromaDB result to Memory object."""
        # Build context
        context = MemoryContext(
            goal_id=UUID(metadata["goal_id"]) if metadata.get("goal_id") else None,
            task_id=UUID(metadata["task_id"]) if metadata.get("task_id") else None,
            source=metadata.get("source"),
            tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
            metadata={
                k.replace("meta_", ""): v
                for k, v in metadata.items()
                if k.startswith("meta_")
            },
        )

        # Create memory
        memory = Memory(
            memory_id=UUID(memory_id),
            agent_id=UUID(metadata["agent_id"]) if metadata.get("agent_id") else None,
            content=document,
            memory_type=MemoryType[metadata.get("type", "EPISODIC")],
            importance=MemoryImportance[metadata.get("importance", "MEDIUM")],
            embedding=embedding,
            context=context,
        )

        # Restore timestamps
        memory._created_at = datetime.fromisoformat(metadata["created_at"])
        if metadata.get("accessed_at"):
            memory._accessed_at = datetime.fromisoformat(metadata["accessed_at"])
        memory._access_count = metadata.get("access_count", 0)
        memory._is_consolidated = metadata.get("is_consolidated", False)

        return memory


class InMemoryMemoryRepository(MemoryRepository):
    """In-memory implementation for testing and development."""

    def __init__(self):
        self._memories: dict[UUID, Memory] = {}

    async def initialize(self) -> None:
        """Initialize the repository."""
        logger.info("In-memory memory repository initialized")

    async def close(self) -> None:
        """Close the repository."""
        self._memories.clear()

    async def save(self, memory: Memory) -> None:
        """Save a memory."""
        self._memories[memory.id] = memory

    async def get(self, memory_id: UUID) -> Memory | None:
        """Get a memory by ID."""
        return self._memories.get(memory_id)

    async def update(self, memory: Memory) -> None:
        """Update an existing memory."""
        self._memories[memory.id] = memory

    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory by ID."""
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False

    async def get_by_agent(self, agent_id: UUID, limit: int = 100) -> list[Memory]:
        """Get all memories for an agent."""
        memories = [
            m for m in self._memories.values()
            if m.agent_id == agent_id
        ]
        return memories[:limit]

    async def search_similar(
        self,
        query_embedding: list[float],
        agent_id: UUID | None = None,
        limit: int = 10,
        min_relevance: float = 0.7,
    ) -> list[tuple[Memory, float]]:
        """Search for similar memories."""
        memories = [
            m for m in self._memories.values()
            if agent_id is None or m.agent_id == agent_id
        ]

        # Calculate relevance scores
        scored = [
            (m, m.calculate_relevance_score(query_embedding))
            for m in memories
        ]

        # Filter and sort
        filtered = [
            (m, score) for m, score in scored
            if score >= min_relevance
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)

        return filtered[:limit]

    async def search_by_content(
        self,
        query: str,
        agent_id: UUID | None = None,
        limit: int = 10,
    ) -> list[Memory]:
        """Search memories by text content."""
        memories = [
            m for m in self._memories.values()
            if agent_id is None or m.agent_id == agent_id
        ]

        # Simple text search
        scored = [
            (m, self._text_similarity(m.content, query))
            for m in memories
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in scored[:limit]]

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    async def get_by_type(
        self,
        memory_type: MemoryType,
        agent_id: UUID | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories by type."""
        memories = [
            m for m in self._memories.values()
            if m.type == memory_type
            and (agent_id is None or m.agent_id == agent_id)
        ]
        return memories[:limit]

    async def get_by_importance(
        self,
        min_importance: MemoryImportance,
        agent_id: UUID | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories by minimum importance."""
        memories = [
            m for m in self._memories.values()
            if m.importance_value >= min_importance.value
            and (agent_id is None or m.agent_id == agent_id)
        ]
        return memories[:limit]

    async def get_recent(
        self,
        hours: int = 24,
        agent_id: UUID | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Get recent memories."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        memories = [
            m for m in self._memories.values()
            if m.created_at >= cutoff
            and (agent_id is None or m.agent_id == agent_id)
        ]

        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    async def get_by_tags(
        self,
        tags: list[str],
        agent_id: UUID | None = None,
        match_all: bool = False,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories by tags."""
        memories = [
            m for m in self._memories.values()
            if agent_id is None or m.agent_id == agent_id
        ]

        if match_all:
            filtered = [
                m for m in memories
                if all(tag in m.context.tags for tag in tags)
            ]
        else:
            filtered = [
                m for m in memories
                if any(tag in m.context.tags for tag in tags)
            ]

        return filtered[:limit]

    async def get_forgettable_memories(
        self,
        threshold_days: int = 30,
        agent_id: UUID | None = None,
    ) -> list[Memory]:
        """Get forgettable memories."""
        memories = [
            m for m in self._memories.values()
            if m.should_forget(threshold_days)
            and (agent_id is None or m.agent_id == agent_id)
        ]
        return memories

    async def consolidate_memories(self, agent_id: UUID | None = None) -> int:
        """Mark old memories as consolidated."""
        count = 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        for memory in self._memories.values():
            if memory.created_at < cutoff and not memory.is_consolidated:
                memory.mark_consolidated()
                count += 1

        return count

    async def count(self, agent_id: UUID | None = None) -> int:
        """Count memories."""
        if agent_id:
            return len([
                m for m in self._memories.values()
                if m.agent_id == agent_id
            ])
        return len(self._memories)

    async def clear(self, agent_id: UUID | None = None) -> int:
        """Clear memories."""
        if agent_id:
            to_delete = [
                mid for mid, m in self._memories.items()
                if m.agent_id == agent_id
            ]
            for mid in to_delete:
                del self._memories[mid]
            return len(to_delete)
        else:
            count = len(self._memories)
            self._memories.clear()
            return count
