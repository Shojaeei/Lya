"""Memory Adapter for Telegram Bot.

JSON-file backed long-term memory with hash-based similarity search.
Works on Python 3.14+ without ChromaDB (which is broken on 3.14).

Features:
- Persistent JSON storage (survives restarts)
- Hash-based embeddings for rough similarity matching
- Text keyword search as fallback
- Automatic importance-based pruning
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from lya.domain.models.memory import (
    Memory,
    MemoryContext,
    MemoryImportance,
    MemoryType,
)
from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)

MAX_MEMORIES = 5000

# Map string importance to enum
_IMPORTANCE_MAP = {
    "CRITICAL": MemoryImportance.CRITICAL,
    "HIGH": MemoryImportance.HIGH,
    "MEDIUM": MemoryImportance.MEDIUM,
    "LOW": MemoryImportance.LOW,
    "TRIVIAL": MemoryImportance.TRIVIAL,
}

# Map string memory type to enum
_TYPE_MAP = {
    "EPISODIC": MemoryType.EPISODIC,
    "SEMANTIC": MemoryType.SEMANTIC,
    "PROCEDURAL": MemoryType.PROCEDURAL,
    "REFLECTIVE": MemoryType.REFLECTIVE,
    "CONVERSATION": MemoryType.CONVERSATION,
    "OBSERVATION": MemoryType.OBSERVATION,
}


def _hash_embedding(text: str, dimension: int = 128) -> list[float]:
    """Generate a deterministic hash-based embedding.

    Uses multiple SHA-256 rounds to produce a fixed-dimension vector.
    Not true semantic — but deterministic and fast.
    """
    # Normalize text
    text = text.lower().strip()

    # Generate multiple hashes for diversity
    vectors: list[float] = []
    for seed in range(0, dimension, 32):
        h = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
        vectors.extend(b / 255.0 for b in h)

    vectors = vectors[:dimension]

    # Normalize to unit vector
    norm = math.sqrt(sum(x * x for x in vectors))
    if norm > 0:
        vectors = [x / norm for x in vectors]

    return vectors


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_score(query: str, content: str) -> float:
    """Simple keyword overlap score for text matching."""
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    if not query_words:
        return 0.0
    overlap = query_words & content_words
    return len(overlap) / len(query_words)


class LyaMemoryAdapter:
    """
    JSON-file backed long-term memory adapter.

    Implements the MemoryService Protocol expected by TelegramBot:
        store_memory() - Store memory with embedding
        recall_memories() - Search by similarity + keywords
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        embedding_service: Any | None = None,
    ):
        self._persist_dir = Path(
            persist_directory
            or str(settings.memory.db_path / "json_memory")
        )
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._db_file = self._persist_dir / "memories.json"
        self._memories: list[dict[str, Any]] = []
        self._initialized = False
        self._dirty = False
        self._embedding_service = embedding_service

    async def initialize(self) -> None:
        """Load memories from disk."""
        try:
            if self._db_file.exists():
                data = json.loads(
                    self._db_file.read_text(encoding="utf-8")
                )
                self._memories = data.get("memories", [])
            else:
                self._memories = []

            self._initialized = True
            logger.info(
                "memory_adapter_initialized",
                backend="json",
                persist_dir=str(self._persist_dir),
                existing_memories=len(self._memories),
            )

        except Exception as e:
            logger.error("memory_adapter_init_failed", error=str(e))
            self._memories = []
            self._initialized = True  # still work, just empty

    async def store_memory(
        self,
        content: str,
        agent_id: UUID | None = None,
        memory_type: str = "CONVERSATION",
        importance: str = "LOW",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a memory.

        Implements the MemoryService Protocol expected by TelegramBot.
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Generate embedding — real if available, hash fallback
            embedding = await self._get_embedding(content)

            entry: dict[str, Any] = {
                "id": str(uuid4()),
                "content": content,
                "agent_id": str(agent_id) if agent_id else "",
                "type": memory_type,
                "importance": importance,
                "tags": tags or [],
                "metadata": metadata or {},
                "embedding": embedding,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "access_count": 0,
            }

            self._memories.append(entry)
            self._dirty = True

            # Prune if over limit — remove oldest LOW importance
            if len(self._memories) > MAX_MEMORIES:
                self._prune()

            # Auto-persist every 10 writes
            if len(self._memories) % 10 == 0:
                await self.persist()

            logger.debug(
                "memory_stored",
                type=memory_type,
                importance=importance,
                total=len(self._memories),
            )

        except Exception as e:
            logger.error("memory_store_failed", error=str(e))

    async def recall_memories(
        self,
        query: str,
        agent_id: UUID | None = None,
        limit: int = 5,
    ) -> list[Memory]:
        """Recall memories by similarity + keyword matching.

        Uses a combined score: hash cosine similarity + keyword overlap.
        Returns list of Memory objects.
        """
        if not self._initialized:
            await self.initialize()

        if not self._memories:
            return []

        try:
            query_embedding = await self._get_embedding(query)

            # Use real embeddings if available — weight cosine higher
            has_real_embeddings = self._embedding_service is not None

            # Score each memory
            scored: list[tuple[dict[str, Any], float]] = []
            for entry in self._memories:
                # Cosine similarity
                emb = entry.get("embedding", [])
                sim = _cosine_similarity(query_embedding, emb) if emb else 0.0

                # Keyword overlap
                kw_score = _keyword_score(query, entry["content"])

                # Boost by importance
                imp_name = entry.get("importance", "LOW")
                imp_boost = {
                    "CRITICAL": 0.15, "HIGH": 0.10,
                    "MEDIUM": 0.05, "LOW": 0.0, "TRIVIAL": -0.05,
                }.get(imp_name, 0.0)

                # With real embeddings: 70% cosine + 20% keyword + 10% importance
                # With hash: 40% cosine + 50% keyword + 10% importance
                if has_real_embeddings:
                    combined = 0.7 * sim + 0.2 * kw_score + imp_boost
                else:
                    combined = 0.4 * sim + 0.5 * kw_score + imp_boost

                if combined > 0.05:
                    scored.append((entry, combined))

            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)

            # Build Memory objects
            memories: list[Memory] = []
            for entry, score in scored[:limit]:
                mem_type = _TYPE_MAP.get(
                    entry.get("type", "CONVERSATION"),
                    MemoryType.CONVERSATION,
                )
                mem_importance = _IMPORTANCE_MAP.get(
                    entry.get("importance", "LOW"),
                    MemoryImportance.LOW,
                )

                memory = Memory(
                    agent_id=UUID(entry["agent_id"]) if entry.get("agent_id") else None,
                    content=entry["content"],
                    memory_type=mem_type,
                    importance=mem_importance,
                    context=MemoryContext(
                        tags=entry.get("tags", []),
                        metadata={"similarity": round(score, 3)},
                    ),
                )
                memories.append(memory)

                # Update access count
                entry["access_count"] = entry.get("access_count", 0) + 1

            if memories:
                self._dirty = True

            logger.debug(
                "memory_recalled",
                query=query[:50],
                results=len(memories),
            )
            return memories

        except Exception as e:
            logger.error("memory_recall_failed", error=str(e))
            return []

    def _prune(self) -> None:
        """Remove oldest LOW/TRIVIAL memories when over capacity."""
        # Sort by importance (ascending) then by date (ascending = oldest first)
        imp_order = {"TRIVIAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

        self._memories.sort(
            key=lambda m: (
                imp_order.get(m.get("importance", "LOW"), 1),
                m.get("created_at", ""),
            )
        )

        # Remove from the front (lowest importance, oldest)
        remove_count = len(self._memories) - MAX_MEMORIES
        if remove_count > 0:
            self._memories = self._memories[remove_count:]
            logger.info("memory_pruned", removed=remove_count)

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding using real service if available, else hash fallback."""
        if self._embedding_service:
            try:
                return await self._embedding_service.embed(text)
            except Exception:
                pass
        return _hash_embedding(text)

    async def get_memory_count(self) -> int:
        """Get total number of stored memories."""
        return len(self._memories)

    async def persist(self) -> None:
        """Save memories to disk."""
        if not self._dirty:
            return
        try:
            data = {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "count": len(self._memories),
                "memories": self._memories,
            }
            # Write atomically: write to temp file then rename
            tmp_file = self._db_file.with_suffix(".tmp")
            tmp_file.write_text(
                json.dumps(data, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            tmp_file.replace(self._db_file)
            self._dirty = False
            logger.debug("memory_persisted", count=len(self._memories))
        except Exception as e:
            logger.error("memory_persist_failed", error=str(e))


async def create_memory_adapter(
    persist_directory: str | None = None,
    embedding_service: Any | None = None,
) -> LyaMemoryAdapter:
    """Factory: create and initialize a LyaMemoryAdapter."""
    adapter = LyaMemoryAdapter(
        persist_directory=persist_directory,
        embedding_service=embedding_service,
    )
    await adapter.initialize()
    return adapter
