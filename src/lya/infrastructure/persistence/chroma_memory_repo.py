"""ChromaDB implementation of MemoryRepository."""

from __future__ import annotations

from typing import Any
from uuid import UUID

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

from lya.domain.models.memory import Memory, MemoryType, MemoryImportance, MemoryContext
from lya.domain.repositories.memory_repo import MemoryRepository
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class ChromaMemoryRepository(MemoryRepository):
    """
    ChromaDB implementation of MemoryRepository.

    Uses ChromaDB for vector storage and semantic search.
    Supports embeddings for similarity-based retrieval.
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "memories",
        embedding_function: Any = None,
    ):
        if not HAS_CHROMADB:
            raise ImportError(
                "chromadb required. Run: pip install chromadb"
            )

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._embedding_function = embedding_function
        self._client: chromadb.Client | None = None
        self._collection: chromadb.Collection | None = None

    async def connect(self) -> None:
        """Initialize ChromaDB connection."""
        logger.info(
            "Connecting to ChromaDB",
            persist_directory=self.persist_directory,
        )

        self._client = chromadb.Client(
            Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
            )
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_function,
        )

        logger.info(
            "ChromaDB connected",
            collection=self.collection_name,
            count=await self.count(),
        )

    async def disconnect(self) -> None:
        """Close ChromaDB connection."""
        if self._client:
            self._client.persist()
            self._client = None
            self._collection = None
            logger.info("ChromaDB disconnected")

    def _ensure_connected(self) -> chromadb.Collection:
        """Ensure we have a valid collection connection."""
        if self._collection is None:
            raise RuntimeError("ChromaDB not connected. Call connect() first.")
        return self._collection

    def _memory_to_document(self, memory: Memory) -> dict[str, Any]:
        """Convert memory to ChromaDB document format."""
        return {
            "id": str(memory.id),
            "content": memory.content,
            "agent_id": str(memory.agent_id) if memory.agent_id else None,
            "type": memory.type.name,
            "importance": memory.importance.name,
            "created_at": memory.created_at.isoformat(),
            "accessed_at": memory.accessed_at.isoformat() if memory.accessed_at else None,
            "access_count": memory.access_count,
            "decay_rate": memory._decay_rate,
            "is_consolidated": memory.is_consolidated,
            "context": memory.context.to_dict() if memory.context else {},
        }

    def _document_to_memory(self, doc: dict[str, Any], embedding: list[float] | None = None) -> Memory:
        """Convert ChromaDB document to Memory."""
        memory = Memory(
            memory_id=UUID(doc["id"]),
            agent_id=UUID(doc["agent_id"]) if doc.get("agent_id") else None,
            content=doc["content"],
            memory_type=MemoryType[doc.get("type", "EPISODIC")],
            importance=MemoryImportance[doc.get("importance", "MEDIUM")],
            embedding=embedding,
            context=MemoryContext.from_dict(doc.get("context", {})),
        )

        # Restore metadata
        from datetime import datetime
        memory._created_at = datetime.fromisoformat(doc["created_at"])
        if doc.get("accessed_at"):
            memory._accessed_at = datetime.fromisoformat(doc["accessed_at"])
        memory._access_count = doc.get("access_count", 0)
        memory._decay_rate = doc.get("decay_rate", 0.01)
        memory._is_consolidated = doc.get("is_consolidated", False)

        return memory

    # ═══════════════════════════════════════════════════════════════
    # Repository Interface Implementation
    # ═══════════════════════════════════════════════════════════════

    async def get(self, memory_id: UUID) -> Memory | None:
        """Retrieve a memory by ID."""
        try:
            collection = self._ensure_connected()
            result = collection.get(
                ids=[str(memory_id)],
                include=["documents", "embeddings", "metadatas"],
            )

            if not result["ids"]:
                return None

            return self._document_to_memory(
                result["metadatas"][0],
                result["embeddings"][0] if result["embeddings"] else None,
            )
        except Exception as e:
            logger.error("Failed to get memory", memory_id=str(memory_id), error=str(e))
            return None

    async def save(self, memory: Memory) -> None:
        """Save a memory."""
        try:
            collection = self._ensure_connected()

            # Get embedding or generate from content
            embedding = memory.embedding
            if embedding is None:
                # ChromaDB will generate embedding automatically
                pass

            doc = self._memory_to_document(memory)

            collection.upsert(
                ids=[str(memory.id)],
                documents=[memory.content],
                embeddings=[embedding] if embedding else None,
                metadatas=[doc],
            )

            logger.debug("Memory saved", memory_id=str(memory.id))

        except Exception as e:
            logger.error("Failed to save memory", memory_id=str(memory.id), error=str(e))
            raise

    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory."""
        try:
            collection = self._ensure_connected()

            # Check if exists first
            existing = await self.get(memory_id)
            if not existing:
                return False

            collection.delete(ids=[str(memory_id)])
            logger.debug("Memory deleted", memory_id=str(memory_id))
            return True

        except Exception as e:
            logger.error("Failed to delete memory", memory_id=str(memory_id), error=str(e))
            return False

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
        memory_type: str | None = None,
        agent_id: UUID | None = None,
    ) -> list[tuple[Memory, float]]:
        """Search memories by semantic similarity."""
        try:
            collection = self._ensure_connected()

            # Build where clause
            where_clause: dict[str, Any] = {}
            if memory_type:
                where_clause["type"] = memory_type
            if agent_id:
                where_clause["agent_id"] = str(agent_id)

            # Perform query
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit * 2,  # Get more to filter by threshold
                where=where_clause if where_clause else None,
                include=["documents", "embeddings", "metadatas", "distances"],
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            # Convert distances to similarity scores
            memories_with_scores = []
            for i, memory_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # Convert cosine distance to similarity
                similarity = 1 - distance

                if similarity >= threshold:
                    memory = self._document_to_memory(
                        results["metadatas"][0][i],
                        results["embeddings"][0][i] if results["embeddings"] else None,
                    )
                    memories_with_scores.append((memory, similarity))

            # Sort by score descending
            memories_with_scores.sort(key=lambda x: x[1], reverse=True)

            # Return limited results
            return memories_with_scores[:limit]

        except Exception as e:
            logger.error("Failed to search memories", error=str(e))
            return []

    async def get_by_agent(
        self,
        agent_id: UUID,
        memory_type: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories for a specific agent."""
        try:
            collection = self._ensure_connected()

            where_clause = {"agent_id": str(agent_id)}
            if memory_type:
                where_clause["type"] = memory_type

            results = collection.get(
                where=where_clause,
                limit=limit,
                include=["documents", "embeddings", "metadatas"],
            )

            if not results["ids"]:
                return []

            memories = []
            for i, _ in enumerate(results["ids"]):
                memory = self._document_to_memory(
                    results["metadatas"][i],
                    results["embeddings"][i] if results["embeddings"] else None,
                )
                memories.append(memory)

            # Sort by creation time descending
            memories.sort(key=lambda m: m.created_at, reverse=True)
            return memories

        except Exception as e:
            logger.error("Failed to get memories by agent", agent_id=str(agent_id), error=str(e))
            return []

    async def get_by_goal(self, goal_id: UUID, limit: int = 100) -> list[Memory]:
        """Get memories associated with a goal."""
        try:
            collection = self._ensure_connected()

            # This uses a contains query on the context metadata
            # Note: ChromaDB doesn't support nested queries directly
            # We use a workaround by querying all and filtering
            results = collection.get(
                limit=limit * 5,  # Get more to filter
                include=["documents", "embeddings", "metadatas"],
            )

            if not results["ids"]:
                return []

            memories = []
            goal_id_str = str(goal_id)

            for i, _ in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                context = metadata.get("context", {})

                if context.get("goal_id") == goal_id_str:
                    memory = self._document_to_memory(
                        metadata,
                        results["embeddings"][i] if results["embeddings"] else None,
                    )
                    memories.append(memory)

            # Sort by creation time descending and limit
            memories.sort(key=lambda m: m.created_at, reverse=True)
            return memories[:limit]

        except Exception as e:
            logger.error("Failed to get memories by goal", goal_id=str(goal_id), error=str(e))
            return []

    async def delete_by_agent(self, agent_id: UUID) -> int:
        """Delete all memories for an agent."""
        try:
            collection = self._ensure_connected()

            # Get all memories for agent
            results = collection.get(
                where={"agent_id": str(agent_id)},
                include=[],
            )

            if not results["ids"]:
                return 0

            # Delete them
            collection.delete(ids=results["ids"])

            logger.info(
                "Deleted agent memories",
                agent_id=str(agent_id),
                count=len(results["ids"]),
            )

            return len(results["ids"])

        except Exception as e:
            logger.error("Failed to delete agent memories", agent_id=str(agent_id), error=str(e))
            return 0

    async def get_stale_memories(
        self,
        threshold_days: int = 30,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories that should be forgotten."""
        try:
            collection = self._ensure_connected()

            # Get all memories (this could be optimized with a date filter)
            results = collection.get(
                limit=limit * 5,
                include=["documents", "embeddings", "metadatas"],
            )

            if not results["ids"]:
                return []

            stale_memories = []

            for i, _ in enumerate(results["ids"]):
                memory = self._document_to_memory(
                    results["metadatas"][i],
                    results["embeddings"][i] if results["embeddings"] else None,
                )

                if memory.should_forget(threshold_days):
                    stale_memories.append(memory)

            # Sort by age and limit
            stale_memories.sort(key=lambda m: m.age_hours, reverse=True)
            return stale_memories[:limit]

        except Exception as e:
            logger.error("Failed to get stale memories", error=str(e))
            return []

    async def count(self, agent_id: UUID | None = None) -> int:
        """Count memories."""
        try:
            collection = self._ensure_connected()

            if agent_id:
                results = collection.get(
                    where={"agent_id": str(agent_id)},
                    include=[],
                )
                return len(results["ids"])
            else:
                return collection.count()

        except Exception as e:
            logger.error("Failed to count memories", error=str(e))
            return 0

    # ═══════════════════════════════════════════════════════════════
    # Additional Methods
    # ═══════════════════════════════════════════════════════════════

    async def persist(self) -> None:
        """Persist the database to disk."""
        if self._client:
            self._client.persist()
            logger.debug("ChromaDB persisted to disk")

    async def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        try:
            collection = self._ensure_connected()
            count = collection.count()

            return {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "total_memories": count,
                "embedding_function": self._embedding_function.__class__.__name__ if self._embedding_function else "default",
            }
        except Exception as e:
            logger.error("Failed to get stats", error=str(e))
            return {}


class ChromaMemoryRepositoryFactory:
    """Factory for creating ChromaMemoryRepository instances."""

    @staticmethod
    def create(
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "memories",
    ) -> ChromaMemoryRepository:
        """Create a ChromaMemoryRepository instance."""
        return ChromaMemoryRepository(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
