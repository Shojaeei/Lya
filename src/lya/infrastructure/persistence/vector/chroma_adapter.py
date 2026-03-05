"""ChromaDB Memory Repository Adapter.

Implements the Memory Repository interface using ChromaDB for vector storage.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID
from pathlib import Path
import json

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.api.types import IncludeEnum
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from lya.domain.models.memory import Memory, MemoryType, MemoryImportance, MemoryContext
from lya.domain.repositories.memory_repo import MemoryRepository
from lya.infrastructure.config.settings import settings
from lya.infrastructure.llm.ollama_adapter import OllamaAdapter


class ChromaMemoryRepository(MemoryRepository):
    """
    ChromaDB implementation of Memory Repository.

    Features:
    - Semantic search with vector similarity
    - Metadata filtering (agent_id, memory_type, etc.)
    - Automatic embedding generation
    - Persistence to disk
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        collection_name: str = "lya_memories",
        embedding_function: Any | None = None,
    ) -> None:
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB not installed. Run: pip install chromadb"
            )

        self.db_path = Path(db_path or settings.memory.db_path).expanduser()
        self.collection_name = collection_name
        self._client: chromadb.AsyncClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._embedding_adapter = embedding_function or OllamaAdapter()

    async def _get_client(self) -> chromadb.AsyncClientAPI:
        """Get or create ChromaDB client."""
        if self._client is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self._client = await chromadb.AsyncHttpClient(
                host="localhost",
                port=8000,
            ) if settings.memory.vector_db == "chroma" and hasattr(settings.memory, 'chroma_host') else chromadb.AsyncClient(
                ChromaSettings(
                    persist_directory=str(self.db_path),
                    anonymized_telemetry=False,
                )
            )
        return self._client

    async def _get_collection(self) -> chromadb.Collection:
        """Get or create memories collection."""
        if self._collection is None:
            client = await self._get_client()

            try:
                self._collection = await client.get_collection(
                    name=self.collection_name,
                )
            except Exception:
                # Collection doesn't exist, create it
                self._collection = await client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
        return self._collection

    def _memory_to_dict(self, memory: Memory) -> dict[str, Any]:
        """Convert memory to ChromaDB document format."""
        return {
            "id": str(memory.id),
            "content": memory.content,
            "agent_id": str(memory.agent_id) if memory.agent_id else None,
            "type": memory.type.name,
            "importance": memory.importance.name,
            "importance_value": memory.importance_value,
            "created_at": memory.created_at.isoformat(),
            "accessed_at": memory.accessed_at.isoformat() if memory.accessed_at else None,
            "access_count": memory.access_count,
            "is_consolidated": memory.is_consolidated,
            "context": memory.context.to_dict() if memory.context else {},
        }

    def _dict_to_memory(self, data: dict[str, Any]) -> Memory:
        """Convert ChromaDB document to Memory entity."""
        return Memory.from_dict(data)

    async def get(self, memory_id: UUID) -> Memory | None:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: Unique identifier

        Returns:
            Memory if found, None otherwise
        """
        collection = await self._get_collection()

        try:
            result = await collection.get(
                ids=[str(memory_id)],
                include=[IncludeEnum.documents, IncludeEnum.metadatas],
            )

            if not result["ids"]:
                return None

            # Reconstruct memory from metadata
            metadata = result["metadatas"][0]
            metadata["id"] = result["ids"][0]
            metadata["content"] = result["documents"][0]

            return self._dict_to_memory(metadata)

        except Exception as e:
            raise RuntimeError(f"Failed to get memory: {e}") from e

    async def save(self, memory: Memory) -> None:
        """
        Save a memory to ChromaDB.

        Args:
            memory: Memory to save
        """
        collection = await self._get_collection()

        try:
            # Generate embedding if not present
            if memory.embedding is None:
                memory._embedding = await self._embedding_adapter.embed(memory.content)

            # Convert to ChromaDB format
            doc_data = self._memory_to_dict(memory)
            content = doc_data.pop("content")  # Content is stored separately

            await collection.upsert(
                ids=[str(memory.id)],
                documents=[content],
                metadatas=[doc_data],
                embeddings=[memory.embedding],
            )

        except Exception as e:
            raise RuntimeError(f"Failed to save memory: {e}") from e

    async def delete(self, memory_id: UUID) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if deleted, False if not found
        """
        collection = await self._get_collection()

        try:
            await collection.delete(ids=[str(memory_id)])
            return True
        except Exception:
            return False

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
        memory_type: str | None = None,
        agent_id: UUID | None = None,
    ) -> list[tuple[Memory, float]]:
        """
        Search memories by semantic similarity.

        Args:
            query_embedding: Query vector
            limit: Maximum results
            threshold: Minimum similarity score (0-1)
            memory_type: Filter by type
            agent_id: Filter by agent

        Returns:
            List of (memory, score) tuples
        """
        collection = await self._get_collection()

        try:
            # Build where clause for filtering
            where: dict[str, Any] = {}
            if memory_type:
                where["type"] = memory_type
            if agent_id:
                where["agent_id"] = str(agent_id)

            results = await collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where if where else None,
                include=[IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.distances],
            )

            memories_with_scores = []
            if results["ids"] and results["ids"][0]:
                for i, memory_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    metadata["id"] = memory_id
                    metadata["content"] = results["documents"][0][i]

                    # Convert distance to similarity score (cosine distance)
                    distance = results["distances"][0][i]
                    similarity = 1 - distance  # Convert distance to similarity

                    if similarity >= threshold:
                        memory = self._dict_to_memory(metadata)
                        memories_with_scores.append((memory, similarity))

            return memories_with_scores

        except Exception as e:
            raise RuntimeError(f"Search failed: {e}") from e

    async def get_by_agent(
        self,
        agent_id: UUID,
        memory_type: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories for a specific agent."""
        collection = await self._get_collection()

        try:
            where: dict[str, Any] = {"agent_id": str(agent_id)}
            if memory_type:
                where["type"] = memory_type

            results = await collection.get(
                where=where,
                limit=limit,
                include=[IncludeEnum.documents, IncludeEnum.metadatas],
            )

            memories = []
            if results["ids"]:
                for i, memory_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    metadata["id"] = memory_id
                    metadata["content"] = results["documents"][i]
                    memories.append(self._dict_to_memory(metadata))

            return memories

        except Exception as e:
            raise RuntimeError(f"Failed to get agent memories: {e}") from e

    async def get_by_goal(self, goal_id: UUID, limit: int = 100) -> list[Memory]:
        """Get memories associated with a goal."""
        collection = await self._get_collection()

        try:
            # Query for memories where context.goal_id matches
            results = await collection.get(
                where={"context.goal_id": str(goal_id)},
                limit=limit,
                include=[IncludeEnum.documents, IncludeEnum.metadatas],
            )

            memories = []
            if results["ids"]:
                for i, memory_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    metadata["id"] = memory_id
                    metadata["content"] = results["documents"][i]
                    memories.append(self._dict_to_memory(metadata))

            return memories

        except Exception as e:
            raise RuntimeError(f"Failed to get goal memories: {e}") from e

    async def delete_by_agent(self, agent_id: UUID) -> int:
        """Delete all memories for an agent."""
        collection = await self._get_collection()

        try:
            # First get all memory IDs for this agent
            results = await collection.get(
                where={"agent_id": str(agent_id)},
                include=[],
            )

            count = len(results["ids"]) if results["ids"] else 0

            if count > 0:
                await collection.delete(ids=results["ids"])

            return count

        except Exception as e:
            raise RuntimeError(f"Failed to delete agent memories: {e}") from e

    async def get_stale_memories(self, threshold_days: int = 30, limit: int = 100) -> list[Memory]:
        """Get memories that should be forgotten based on age and access."""
        # ChromaDB doesn't support complex queries, so we fetch and filter
        collection = await self._get_collection()

        try:
            results = await collection.get(
                limit=limit * 2,  # Fetch more to filter
                include=[IncludeEnum.documents, IncludeEnum.metadatas],
            )

            stale_memories = []
            if results["ids"]:
                for i, memory_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    metadata["id"] = memory_id
                    metadata["content"] = results["documents"][i]

                    memory = self._dict_to_memory(metadata)
                    if memory.should_forget(threshold_days):
                        stale_memories.append(memory)

                    if len(stale_memories) >= limit:
                        break

            return stale_memories

        except Exception as e:
            raise RuntimeError(f"Failed to get stale memories: {e}") from e

    async def count(self, agent_id: UUID | None = None) -> int:
        """Count memories, optionally filtered by agent."""
        collection = await self._get_collection()

        try:
            where = {"agent_id": str(agent_id)} if agent_id else None
            results = await collection.get(
                where=where,
                include=[],
            )
            return len(results["ids"]) if results["ids"] else 0

        except Exception as e:
            raise RuntimeError(f"Failed to count memories: {e}") from e

    async def close(self) -> None:
        """Close the ChromaDB client."""
        # ChromaDB handles persistence automatically
        self._client = None
        self._collection = None
