"""Domain service for memory management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from lya.domain.models.memory import Memory, MemoryType, MemoryImportance, MemoryContext
from lya.domain.repositories.memory_repo import MemoryRepository
from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.persistence import get_embedding_service

import numpy as np
from datetime import timedelta

logger = get_logger(__name__)


class MemoryService:
    """
    Domain service for memory management.

    Handles:
    - Memory creation with automatic embedding
    - Semantic search
    - Memory consolidation and forgetting
    - Context-based retrieval
    """

    def __init__(self, repository: MemoryRepository):
        self._repo = repository
        self._embedding_service = get_embedding_service()

    async def create_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        agent_id: UUID | None = None,
        context: MemoryContext | None = None,
    ) -> Memory:
        """
        Create and store a new memory.

        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance level
            agent_id: Owning agent
            context: Memory context

        Returns:
            Created memory
        """
        # Generate embedding
        try:
            embedding = await self._embedding_service.embed(content)
        except Exception as e:
            logger.warning("Failed to generate embedding", error=str(e))
            embedding = None

        # Create memory
        memory = Memory(
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            embedding=embedding,
            context=context or MemoryContext(),
        )

        # Save
        await self._repo.save(memory)

        logger.info(
            "Memory created",
            memory_id=str(memory.id),
            memory_type=memory_type.name,
            importance=importance.name,
        )

        return memory

    async def recall(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.7,
        memory_type: MemoryType | None = None,
        agent_id: UUID | None = None,
    ) -> list[tuple[Memory, float]]:
        """
        Recall memories based on semantic similarity.

        Args:
            query: Search query
            limit: Maximum results
            threshold: Minimum similarity
            memory_type: Filter by type
            agent_id: Filter by agent

        Returns:
            List of (memory, score) tuples
        """
        # Generate query embedding
        try:
            query_embedding = await self._embedding_service.embed(query)
        except Exception as e:
            logger.error("Failed to embed query", error=str(e))
            return []

        # Search
        results = await self._repo.search(
            query_embedding=query_embedding,
            limit=limit,
            threshold=threshold,
            memory_type=memory_type.name if memory_type else None,
            agent_id=agent_id,
        )

        # Record access
        for memory, score in results:
            memory.access()
            await self._repo.save(memory)

        logger.debug(
            "Memory recall",
            query=query[:50],
            results=len(results),
        )

        return results

    async def get_agent_memories(
        self,
        agent_id: UUID,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Get all memories for an agent."""
        return await self._repo.get_by_agent(
            agent_id=agent_id,
            memory_type=memory_type.name if memory_type else None,
            limit=limit,
        )

    async def get_goal_memories(
        self,
        goal_id: UUID,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories related to a goal."""
        return await self._repo.get_by_goal(goal_id, limit)

    async def consolidate_memories(
        self,
        agent_id: UUID,
        memory_type: MemoryType = MemoryType.EPISODIC,
    ) -> Memory | None:
        """
        Consolidate multiple related memories into a summary.

        This simulates the human process of memory consolidation
        where multiple similar experiences are merged into a
        general understanding.

        Args:
            agent_id: Agent to consolidate memories for
            memory_type: Type of memories to consolidate

        Returns:
            Consolidated memory or None
        """
        # Get memories of this type
        memories = await self._repo.get_by_agent(
            agent_id=agent_id,
            memory_type=memory_type.name,
            limit=20,
        )

        if len(memories) < 3:
            return None

        # Filter unconsolidated memories
        unconsolidated = [m for m in memories if not m.is_consolidated]

        if len(unconsolidated) < 3:
            return None

        # Group by similarity (simplified - in practice would use clustering)
        # For now, just create a summary of the unconsolidated memories
        summary_parts = [f"- {m.content[:100]}" for m in unconsolidated[:10]]
        summary_content = f"Consolidated {len(unconsolidated)} memories:\n" + "\n".join(summary_parts)

        # Create consolidated memory
        consolidated = await self.create_memory(
            content=summary_content,
            memory_type=MemoryType.REFLECTIVE,
            importance=MemoryImportance.HIGH,
            agent_id=agent_id,
            context=MemoryContext(
                tags=["consolidated", memory_type.name.lower()],
                metadata={
                    "source_memories": [str(m.id) for m in unconsolidated],
                    "consolidation_count": len(unconsolidated),
                },
            ),
        )

        # Mark source memories as consolidated
        for memory in unconsolidated:
            memory.mark_consolidated()
            await self._repo.save(memory)

        logger.info(
            "Memories consolidated",
            agent_id=str(agent_id),
            count=len(unconsolidated),
            consolidated_id=str(consolidated.id),
        )

        return consolidated

    async def forget_stale_memories(
        self,
        threshold_days: int = 30,
        dry_run: bool = False,
    ) -> int:
        """
        Remove old, unimportant memories.

        Args:
            threshold_days: Age threshold for forgetting
            dry_run: If True, just count without deleting

        Returns:
            Number of memories forgotten
        """
        stale = await self._repo.get_stale_memories(threshold_days)

        if dry_run:
            logger.info(
                "Memory cleanup dry run",
                count=len(stale),
                threshold_days=threshold_days,
            )
            return len(stale)

        deleted = 0
        for memory in stale:
            if await self._repo.delete(memory.id):
                deleted += 1

        logger.info(
            "Stale memories forgotten",
            count=deleted,
            threshold_days=threshold_days,
        )

        return deleted

    async def get_memory_stats(self, agent_id: UUID | None = None) -> dict[str, Any]:
        """Get memory statistics."""
        total = await self._repo.count(agent_id)

        stats = {
            "total_memories": total,
            "agent_id": str(agent_id) if agent_id else None,
        }

        if hasattr(self._repo, "get_stats"):
            stats.update(await self._repo.get_stats())

        return stats

    async def export_memories(
        self,
        agent_id: UUID | None = None,
        memory_type: MemoryType | None = None,
    ) -> list[dict[str, Any]]:
        """Export memories to a serializable format."""
        if agent_id:
            memories = await self._repo.get_by_agent(
                agent_id=agent_id,
                memory_type=memory_type.name if memory_type else None,
            )
        else:
            # Get all memories
            memories = []
            # This would need a method to get all memories from repo

        return [m.to_dict() for m in memories]

    async def import_memories(
        self,
        data: list[dict[str, Any]],
        agent_id: UUID | None = None,
    ) -> int:
        """Import memories from exported data."""
        imported = 0

        for item in data:
            try:
                memory = Memory.from_dict(item)

                # Override agent_id if specified
                if agent_id:
                    memory._agent_id = agent_id

                await self._repo.save(memory)
                imported += 1

            except Exception as e:
                logger.error("Failed to import memory", error=str(e), item=item)

        logger.info("Memories imported", count=imported)
        return imported

    # ═══════════════════════════════════════════════════════════════
    # Vector Memory Features (ported from old version)
    # ═══════════════════════════════════════════════════════════════

    async def cluster_memories(
        self,
        agent_id: UUID | None = None,
        n_clusters: int = 5,
        memory_type: MemoryType | None = None,
    ) -> dict[str, list[Memory]]:
        """
        Cluster memories by semantic similarity using k-means style clustering.

        Args:
            agent_id: Optional agent filter
            n_clusters: Number of clusters
            memory_type: Optional memory type filter

        Returns:
            Dictionary of cluster_name -> memories
        """
        # Get memories
        memories = await self._repo.get_by_agent(
            agent_id=agent_id,
            memory_type=memory_type.name if memory_type else None,
            limit=500,
        )

        if len(memories) < n_clusters:
            return {"all": memories}

        # Get embeddings for memories
        memory_embeddings = []
        valid_memories = []

        for memory in memories:
            if memory.embedding:
                memory_embeddings.append(memory.embedding)
                valid_memories.append(memory)

        if len(valid_memories) < n_clusters:
            return {"all": valid_memories}

        # Convert to numpy arrays
        embeddings = np.array(memory_embeddings)

        # K-means style clustering (simplified)
        np.random.seed(42)
        indices = np.random.choice(len(embeddings), n_clusters, replace=False)
        centroids = embeddings[indices]

        # Assign to clusters
        clusters: dict[int, list[Memory]] = {i: [] for i in range(n_clusters)}

        for memory in valid_memories:
            # Find closest centroid
            memory_vec = np.array(memory.embedding)
            similarities = [
                np.dot(memory_vec, centroid) /
                (np.linalg.norm(memory_vec) * np.linalg.norm(centroid))
                for centroid in centroids
            ]
            cluster_id = int(np.argmax(similarities))
            clusters[cluster_id].append(memory)

        # Name clusters by most common type
        named_clusters = {}
        for cluster_id, cluster_memories in clusters.items():
            if cluster_memories:
                # Get most common type
                types = [m.type.name for m in cluster_memories]
                most_common = max(set(types), key=types.count)
                name = f"{most_common.lower()}_cluster_{cluster_id}"
                named_clusters[name] = cluster_memories

        logger.info(
            "Memories clustered",
            total=len(valid_memories),
            clusters=n_clusters,
        )

        return named_clusters

    async def search_with_importance(
        self,
        query: str,
        limit: int = 10,
        importance_weight: float = 0.3,
    ) -> list[tuple[Memory, float]]:
        """
        Search memories weighted by importance.

        Args:
            query: Search query
            limit: Maximum results
            importance_weight: How much to weight importance (0-1)

        Returns:
            List of (memory, weighted_score) tuples
        """
        # Get base search results
        results = await self.recall(query, limit=limit * 2, threshold=0.5)

        # Weight by importance
        weighted_results = []
        for memory, similarity in results:
            importance_factor = memory.importance_value / 5.0  # Normalize 1-5 to 0.2-1.0
            weighted_score = similarity * (1 - importance_weight) + importance_factor * importance_weight
            weighted_results.append((memory, weighted_score))

        # Sort by weighted score
        weighted_results.sort(key=lambda x: x[1], reverse=True)

        return weighted_results[:limit]

    async def forget_old_memories(
        self,
        days: int = 30,
        importance_threshold: float = 0.5,
        memory_type: MemoryType | None = None,
    ) -> int:
        """
        Remove old, unimportant memories.

        Args:
            days: Age threshold
            importance_threshold: Importance below which memories can be forgotten
            memory_type: Optional type filter

        Returns:
            Number of memories forgotten
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Get all memories
        memories = await self._repo.get_by_agent(
            agent_id=None,
            memory_type=memory_type.name if memory_type else None,
            limit=10000,
        )

        to_remove = []
        for memory in memories:
            # Check age and importance
            is_old = memory.created_at < cutoff
            is_unimportant = memory.importance_value / 5.0 < importance_threshold

            if is_old and is_unimportant and not memory.is_consolidated:
                to_remove.append(memory)

        # Delete memories
        deleted = 0
        for memory in to_remove:
            if await self._repo.delete(memory.id):
                deleted += 1

        logger.info(
            "Old memories forgotten",
            total=len(memories),
            deleted=deleted,
            days=days,
        )

        return deleted

    def _simple_hash_embedding(self, text: str, dimension: int = 384) -> list[float]:
        """
        Simple hash-based embedding for testing/fallback.
        Uses SHA256 hash to create deterministic embedding.

        Args:
            text: Text to embed
            dimension: Embedding dimension

        Returns:
            Normalized embedding vector
        """
        import hashlib

        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to float array
        embedding = [
            int(hash_bytes[i % len(hash_bytes)]) / 255.0
            for i in range(dimension)
        ]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    async def create_memory_with_fallback(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        agent_id: UUID | None = None,
        context: MemoryContext | None = None,
    ) -> Memory:
        """
        Create memory with fallback embedding if service fails.

        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance level
            agent_id: Owning agent
            context: Memory context

        Returns:
            Created memory
        """
        # Try to generate embedding
        try:
            embedding = await self._embedding_service.embed(content)
        except Exception as e:
            logger.warning(
                "Embedding service failed, using fallback",
                error=str(e),
            )
            # Use simple hash-based embedding
            embedding = self._simple_hash_embedding(content)

        # Create memory
        memory = Memory(
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            embedding=embedding,
            context=context or MemoryContext(),
        )

        # Save
        await self._repo.save(memory)

        logger.info(
            "Memory created (with fallback)",
            memory_id=str(memory.id),
            memory_type=memory_type.name,
        )

        return memory

    async def get_memory_clusters_summary(
        self,
        agent_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Get summary of memory clusters."""
        clusters = await self.cluster_memories(agent_id=agent_id)

        summary = {
            "total_clusters": len(clusters),
            "clusters": {},
        }

        for name, memories in clusters.items():
            types = [m.type.name for m in memories]
            summary["clusters"][name] = {
                "count": len(memories),
                "dominant_type": max(set(types), key=types.count) if types else "unknown",
                "avg_importance": sum(m.importance_value for m in memories) / len(memories) if memories else 0,
                "oldest": min(m.created_at for m in memories).isoformat() if memories else None,
                "newest": max(m.created_at for m in memories).isoformat() if memories else None,
            }

        return summary

    # ═══════════════════════════════════════════════════════════════
    # Clustering (from old vector_memory.py)
    # ═══════════════════════════════════════════════════════════════

    async def cluster_memories(
        self,
        agent_id: UUID,
        n_clusters: int = 5,
        memory_type: MemoryType | None = None,
    ) -> dict[str, list[Memory]]:
        """
        Cluster memories by semantic similarity.

        Groups similar memories together using embeddings.

        Args:
            agent_id: Agent to cluster memories for
            n_clusters: Number of clusters to create
            memory_type: Optional filter by memory type

        Returns:
            Dictionary of cluster_name -> memories
        """
        # Get memories with embeddings
        memories = await self._repo.get_by_agent(
            agent_id=agent_id,
            memory_type=memory_type.name if memory_type else None,
            limit=1000,
        )

        # Filter memories with embeddings
        memories_with_embeddings = [
            m for m in memories if m.embedding is not None
        ]

        if len(memories_with_embeddings) < n_clusters:
            return {"all": memories_with_embeddings}

        try:
            # Create embeddings matrix
            embeddings = np.array([m.embedding for m in memories_with_embeddings])

            # Simple k-means clustering
            np.random.seed(42)
            # Randomly initialize centroids
            indices = np.random.choice(
                len(embeddings),
                n_clusters,
                replace=False,
            )
            centroids = embeddings[indices]

            # Assign to clusters (single iteration for simplicity)
            clusters: dict[int, list[Memory]] = {i: [] for i in range(n_clusters)}

            for memory in memories_with_embeddings:
                # Calculate similarity to each centroid
                similarities = [
                    np.dot(memory.embedding, c)
                    for c in centroids
                ]
                cluster_id = int(np.argmax(similarities))
                clusters[cluster_id].append(memory)

            # Name clusters by most common memory type
            named_clusters = {}
            for cluster_id, cluster_memories in clusters.items():
                if cluster_memories:
                    # Find most common type
                    type_counts = {}
                    for m in cluster_memories:
                        type_counts[m.type] = type_counts.get(m.type, 0) + 1
                    most_common_type = max(type_counts, key=type_counts.get)
                    name = f"{most_common_type.name.lower()}_cluster_{cluster_id}"
                    named_clusters[name] = cluster_memories

            logger.info(
                "Memories clustered",
                agent_id=str(agent_id),
                clusters=len(named_clusters),
                memories=len(memories_with_embeddings),
            )

            return named_clusters

        except Exception as e:
            logger.error("Clustering failed", error=str(e))
            return {"all": memories_with_embeddings}

    async def forget_old_memories(
        self,
        agent_id: UUID,
        days: int = 30,
        importance_threshold: float = 0.5,
        dry_run: bool = False,
    ) -> int:
        """
        Remove old, unimportant memories.

        Args:
            agent_id: Agent to forget memories for
            days: Age threshold
            importance_threshold: Importance level (1-5 scale)
            dry_run: If True, just count without deleting

        Returns:
            Number of memories forgotten
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Get all memories for agent
        memories = await self._repo.get_by_agent(agent_id=agent_id, limit=10000)

        to_remove = []
        for memory in memories:
            # Check if old and unimportant
            is_old = memory.created_at < cutoff
            # Convert importance enum to float (1-5 scale)
            importance_value = memory.importance_value
            is_unimportant = importance_value < (importance_threshold * 5)

            if is_old and is_unimportant and not memory.is_consolidated:
                to_remove.append(memory)

        if dry_run:
            logger.info(
                "Forget dry run",
                count=len(to_remove),
                days=days,
                importance_threshold=importance_threshold,
            )
            return len(to_remove)

        # Delete memories
        deleted = 0
        for memory in to_remove:
            if await self._repo.delete(memory.id):
                deleted += 1

        logger.info(
            "Old memories forgotten",
            count=deleted,
            agent_id=str(agent_id),
        )

        return deleted

    async def recall_with_importance_weighting(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.5,
        importance_weight: float = 0.3,
    ) -> list[tuple[Memory, float]]:
        """
        Recall memories with importance-weighted similarity.

        Args:
            query: Search query
            limit: Maximum results
            threshold: Minimum similarity
            importance_weight: How much to weight importance (0-1)

        Returns:
            List of (memory, weighted_score) tuples
        """
        # Get base results
        results = await self.recall(
            query=query,
            limit=limit * 2,  # Get more for re-ranking
            threshold=threshold,
        )

        if not results:
            return []

        # Apply importance weighting
        weighted_results = []
        for memory, similarity in results:
            # Normalize importance to 0-1
            importance_normalized = memory.importance_value / 5.0

            # Weighted score: (1-w) * similarity + w * importance
            weighted_score = (
                (1 - importance_weight) * similarity +
                importance_weight * importance_normalized
            )

            weighted_results.append((memory, weighted_score))

        # Sort by weighted score
        weighted_results.sort(key=lambda x: x[1], reverse=True)

        return weighted_results[:limit]

    async def get_memory_clusters_summary(
        self,
        agent_id: UUID,
    ) -> list[dict[str, Any]]:
        """Get summary of memory clusters."""
        clusters = await self.cluster_memories(agent_id=agent_id)

        summary = []
        for name, memories in clusters.items():
            if memories:
                # Calculate cluster stats
                avg_importance = sum(m.importance_value for m in memories) / len(memories)
                types = {}
                for m in memories:
                    types[m.type.name] = types.get(m.type.name, 0) + 1

                summary.append({
                    "name": name,
                    "count": len(memories),
                    "avg_importance": avg_importance,
                    "types": types,
                    "sample_memories": [m.content[:100] for m in memories[:3]],
                })

        return sorted(summary, key=lambda x: x["count"], reverse=True)
