"""Working Memory Buffer for Lya.

Short-term memory system with automatic pruning and importance scoring.
Pure Python 3.14+ compatible.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


@dataclass
class MemoryItem:
    """Single item in working memory."""

    id: str
    content: str
    importance: float  # 0.0 to 1.0
    timestamp: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    ttl_seconds: float | None = None  # Time to live

    def is_expired(self) -> bool:
        """Check if item has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.timestamp > self.ttl_seconds

    def get_score(self) -> float:
        """Calculate memory score based on importance and recency."""
        age_hours = (time.time() - self.timestamp) / 3600
        recency_factor = max(0.1, 1.0 - (age_hours / 24))  # Decay over 24 hours

        access_factor = min(1.0, self.access_count / 10)  # Cap at 10 accesses

        return self.importance * 0.5 + recency_factor * 0.3 + access_factor * 0.2


@dataclass
class MemorySummary:
    """Summary of working memory state."""

    total_items: int
    expired_items: int
    total_importance: float
    oldest_item_age_hours: float
    newest_item_age_hours: float
    top_sources: list[tuple[str, int]]


class WorkingMemoryBuffer:
    """
    Short-term working memory for Lya.

    Features:
    - Automatic pruning of low-importance items
    - Time-based expiration (TTL)
    - Importance-based retention
    - Fast lookup by content similarity
    - Context window management
    """

    def __init__(
        self,
        max_items: int = 100,
        default_ttl_seconds: float = 3600,
        prune_threshold: float = 0.3,
        persist_path: str | Path | None = None,
    ) -> None:
        """Initialize working memory buffer.

        Args:
            max_items: Maximum items to store
            default_ttl_seconds: Default time-to-live for items
            prune_threshold: Score threshold below which items are pruned
            persist_path: Path to persist memory (None for no persistence)
        """
        self.max_items = max_items
        self.default_ttl = default_ttl_seconds
        self.prune_threshold = prune_threshold
        self.persist_path = Path(persist_path) if persist_path else None

        self._items: dict[str, MemoryItem] = {}
        self._access_order: list[str] = []  # LRU tracking

        # Load persisted state if available
        if self.persist_path and self.persist_path.exists():
            self._load()

    def add(
        self,
        content: str,
        importance: float = 0.5,
        source: str = "user",
        metadata: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
        item_id: str | None = None,
    ) -> str:
        """Add item to working memory.

        Args:
            content: Memory content
            importance: Importance score (0.0-1.0)
            source: Source of the memory
            metadata: Additional metadata
            ttl_seconds: Custom TTL (uses default if None)
            item_id: Custom ID (generates UUID if None)

        Returns:
            Item ID
        """
        import uuid

        item_id = item_id or str(uuid.uuid4())

        item = MemoryItem(
            id=item_id,
            content=content,
            importance=max(0.0, min(1.0, importance)),
            timestamp=time.time(),
            source=source,
            metadata=metadata or {},
            ttl_seconds=ttl_seconds if ttl_seconds is not None else self.default_ttl,
        )

        self._items[item_id] = item
        self._update_access_order(item_id)

        # Prune if over capacity
        if len(self._items) > self.max_items:
            self._prune_lowest_score(1)

        # Auto-save if persistence enabled
        if self.persist_path:
            self._save()

        return item_id

    def get(self, item_id: str) -> MemoryItem | None:
        """Get item by ID.

        Args:
            item_id: Item ID

        Returns:
            MemoryItem or None if not found
        """
        item = self._items.get(item_id)

        if item:
            if item.is_expired():
                self.delete(item_id)
                return None

            item.access_count += 1
            item.last_accessed = time.time()
            self._update_access_order(item_id)

        return item

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_importance: float = 0.0,
    ) -> list[tuple[MemoryItem, float]]:
        """Query memory by content similarity (simple keyword matching).

        Args:
            query_text: Query text
            top_k: Maximum results
            min_importance: Minimum importance threshold

        Returns:
            List of (item, score) tuples
        """
        query_lower = query_text.lower()
        query_words = set(query_lower.split())

        results: list[tuple[MemoryItem, float]] = []

        for item in self._items.values():
            if item.is_expired():
                continue

            if item.importance < min_importance:
                continue

            # Calculate similarity score
            content_lower = item.content.lower()
            content_words = set(content_lower.split())

            # Simple Jaccard similarity
            intersection = len(query_words & content_words)
            union = len(query_words | content_words)

            if union == 0:
                continue

            similarity = intersection / union

            # Boost by item score
            final_score = similarity * 0.7 + item.get_score() * 0.3

            if similarity > 0:
                results.append((item, final_score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def search_by_source(
        self,
        source: str,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Search items by source.

        Args:
            source: Source to search
            limit: Maximum results

        Returns:
            List of matching items
        """
        results = [
            item for item in self._items.values()
            if item.source == source and not item.is_expired()
        ]

        # Sort by importance desc, then timestamp desc
        results.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)

        return results[:limit]

    def get_context_window(
        self,
        max_tokens: int = 4000,
        include_metadata: bool = False,
    ) -> str:
        """Get context window for LLM.

        Args:
            max_tokens: Approximate token limit
            include_metadata: Include source/timestamp info

        Returns:
            Formatted context string
        """
        # Sort by importance and recency
        items = sorted(
            self._items.values(),
            key=lambda x: x.get_score(),
            reverse=True,
        )

        context_parts = []
        current_length = 0
        max_chars = max_tokens * 4  # Rough approximation

        for item in items:
            if item.is_expired():
                continue

            if include_metadata:
                part = f"[{item.source}] {item.content}"
            else:
                part = item.content

            if current_length + len(part) > max_chars:
                break

            context_parts.append(part)
            current_length += len(part)

        return "\n\n".join(context_parts)

    def delete(self, item_id: str) -> bool:
        """Delete item from memory.

        Args:
            item_id: Item ID to delete

        Returns:
            True if deleted, False if not found
        """
        if item_id in self._items:
            del self._items[item_id]
            if item_id in self._access_order:
                self._access_order.remove(item_id)

            if self.persist_path:
                self._save()

            return True

        return False

    def clear(self, source: str | None = None) -> int:
        """Clear memory items.

        Args:
            source: Only clear items from this source (None for all)

        Returns:
            Number of items cleared
        """
        if source is None:
            count = len(self._items)
            self._items.clear()
            self._access_order.clear()
        else:
            to_delete = [
                item_id for item_id, item in self._items.items()
                if item.source == source
            ]
            count = len(to_delete)
            for item_id in to_delete:
                del self._items[item_id]
                if item_id in self._access_order:
                    self._access_order.remove(item_id)

        if self.persist_path:
            self._save()

        return count

    def prune_expired(self) -> int:
        """Remove all expired items.

        Returns:
            Number of items pruned
        """
        expired = [
            item_id for item_id, item in self._items.items()
            if item.is_expired()
        ]

        for item_id in expired:
            self.delete(item_id)

        return len(expired)

    def prune_low_importance(self, threshold: float | None = None) -> int:
        """Prune items below importance threshold.

        Args:
            threshold: Importance threshold (uses prune_threshold if None)

        Returns:
            Number of items pruned
        """
        threshold = threshold or self.prune_threshold

        to_prune = [
            item_id for item_id, item in self._items.items()
            if item.get_score() < threshold
        ]

        for item_id in to_prune:
            self.delete(item_id)

        return len(to_prune)

    def get_summary(self) -> MemorySummary:
        """Get summary of working memory state."""
        now = time.time()

        if not self._items:
            return MemorySummary(
                total_items=0,
                expired_items=0,
                total_importance=0.0,
                oldest_item_age_hours=0.0,
                newest_item_age_hours=0.0,
                top_sources=[],
            )

        timestamps = [item.timestamp for item in self._items.values()]
        expired_count = sum(1 for item in self._items.values() if item.is_expired())
        total_importance = sum(item.importance for item in self._items.values())

        # Count by source
        source_counts: dict[str, int] = {}
        for item in self._items.values():
            source_counts[item.source] = source_counts.get(item.source, 0) + 1

        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return MemorySummary(
            total_items=len(self._items),
            expired_items=expired_count,
            total_importance=total_importance,
            oldest_item_age_hours=(now - min(timestamps)) / 3600,
            newest_item_age_hours=(now - max(timestamps)) / 3600,
            top_sources=top_sources,
        )

    def list_all(
        self,
        include_expired: bool = False,
        sort_by: str = "importance",
    ) -> list[MemoryItem]:
        """List all memory items.

        Args:
            include_expired: Include expired items
            sort_by: Sort field (importance, timestamp, score)

        Returns:
            List of items
        """
        items = list(self._items.values())

        if not include_expired:
            items = [item for item in items if not item.is_expired()]

        if sort_by == "importance":
            items.sort(key=lambda x: x.importance, reverse=True)
        elif sort_by == "timestamp":
            items.sort(key=lambda x: x.timestamp, reverse=True)
        elif sort_by == "score":
            items.sort(key=lambda x: x.get_score(), reverse=True)
        elif sort_by == "access":
            items.sort(key=lambda x: x.access_count, reverse=True)

        return items

    def _update_access_order(self, item_id: str) -> None:
        """Update LRU access order."""
        if item_id in self._access_order:
            self._access_order.remove(item_id)
        self._access_order.append(item_id)

    def _prune_lowest_score(self, count: int) -> None:
        """Prune items with lowest scores."""
        items_with_scores = [
            (item_id, self._items[item_id].get_score())
            for item_id in self._access_order
        ]
        items_with_scores.sort(key=lambda x: x[1])

        for item_id, _ in items_with_scores[:count]:
            self.delete(item_id)

    def _save(self) -> None:
        """Save working memory to disk."""
        if not self.persist_path:
            return

        data = {
            "max_items": self.max_items,
            "default_ttl": self.default_ttl,
            "prune_threshold": self.prune_threshold,
            "items": [
                {
                    "id": item.id,
                    "content": item.content,
                    "importance": item.importance,
                    "timestamp": item.timestamp,
                    "source": item.source,
                    "metadata": item.metadata,
                    "access_count": item.access_count,
                    "last_accessed": item.last_accessed,
                    "ttl_seconds": item.ttl_seconds,
                }
                for item in self._items.values()
            ],
        }

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load working memory from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            data = json.loads(self.persist_path.read_text())

            self.max_items = data.get("max_items", self.max_items)
            self.default_ttl = data.get("default_ttl", self.default_ttl)
            self.prune_threshold = data.get("prune_threshold", self.prune_threshold)

            for item_data in data.get("items", []):
                item = MemoryItem(**item_data)
                self._items[item.id] = item
                self._access_order.append(item.id)

        except Exception as e:
            # Reset on error
            self._items.clear()
            self._access_order.clear()


class ContextManager:
    """
    Manages conversation context and working memory together.

    Provides a unified interface for context management.
    """

    def __init__(
        self,
        working_memory: WorkingMemoryBuffer | None = None,
        context_window_size: int = 4000,
    ) -> None:
        """Initialize context manager.

        Args:
            working_memory: Working memory buffer
            context_window_size: Default context window size
        """
        self.working_memory = working_memory or WorkingMemoryBuffer()
        self.context_window_size = context_window_size

        self._conversation_history: list[dict[str, Any]] = []
        self._max_history = 50

    def add_user_message(self, content: str, importance: float = 0.6) -> str:
        """Add user message to context.

        Args:
            content: Message content
            importance: Importance score

        Returns:
            Memory item ID
        """
        self._conversation_history.append({
            "role": "user",
            "content": content,
            "timestamp": time.time(),
        })
        self._trim_history()

        return self.working_memory.add(
            content=content,
            importance=importance,
            source="user",
            metadata={"type": "message", "role": "user"},
        )

    def add_assistant_message(self, content: str, importance: float = 0.5) -> str:
        """Add assistant message to context.

        Args:
            content: Message content
            importance: Importance score

        Returns:
            Memory item ID
        """
        self._conversation_history.append({
            "role": "assistant",
            "content": content,
            "timestamp": time.time(),
        })
        self._trim_history()

        return self.working_memory.add(
            content=content,
            importance=importance,
            source="assistant",
            metadata={"type": "message", "role": "assistant"},
        )

    def add_tool_result(
        self,
        tool_name: str,
        result: dict[str, Any],
        importance: float = 0.4,
    ) -> str:
        """Add tool execution result.

        Args:
            tool_name: Tool that was executed
            result: Tool result
            importance: Importance score

        Returns:
            Memory item ID
        """
        content = f"Tool {tool_name}: {json.dumps(result, default=str)[:500]}"

        return self.working_memory.add(
            content=content,
            importance=importance,
            source="tool",
            metadata={"type": "tool_result", "tool": tool_name, "result": result},
        )

    def add_observation(
        self,
        observation: str,
        importance: float = 0.3,
    ) -> str:
        """Add an observation.

        Args:
            observation: Observation text
            importance: Importance score

        Returns:
            Memory item ID
        """
        return self.working_memory.add(
            content=observation,
            importance=importance,
            source="observation",
            metadata={"type": "observation"},
        )

    def get_context_for_llm(
        self,
        include_history: bool = True,
        include_working_memory: bool = True,
    ) -> str:
        """Get formatted context for LLM.

        Args:
            include_history: Include conversation history
            include_working_memory: Include working memory

        Returns:
            Formatted context string
        """
        parts = []

        if include_working_memory:
            working_ctx = self.working_memory.get_context_window(
                max_tokens=self.context_window_size // 2,
                include_metadata=True,
            )
            if working_ctx:
                parts.append("## Working Memory\n" + working_ctx)

        if include_history:
            history_ctx = self._format_history()
            if history_ctx:
                parts.append("## Conversation History\n" + history_ctx)

        return "\n\n".join(parts)

    def _format_history(self) -> str:
        """Format conversation history."""
        lines = []
        for msg in self._conversation_history[-10:]:  # Last 10 messages
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content'][:200]}")
        return "\n".join(lines)

    def _trim_history(self) -> None:
        """Trim conversation history if too long."""
        if len(self._conversation_history) > self._max_history:
            # Move oldest to long-term memory
            oldest = self._conversation_history.pop(0)
            self.working_memory.add(
                content=f"[{oldest['role']}] {oldest['content']}",
                importance=0.3,
                source="conversation_archive",
                ttl_seconds=86400,  # 24 hours
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[MemoryItem, float]]:
        """Search working memory.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Search results
        """
        return self.working_memory.query(query, top_k=top_k)

    def clear(self) -> None:
        """Clear all context."""
        self.working_memory.clear()
        self._conversation_history.clear()


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example usage
    buffer = WorkingMemoryBuffer(
        max_items=50,
        persist_path="./working_memory.json",
    )

    # Add items
    buffer.add(
        "User asked about Python async patterns",
        importance=0.8,
        source="conversation",
    )

    buffer.add(
        "System temperature is normal",
        importance=0.3,
        source="system",
    )

    # Query
    results = buffer.query("async python", top_k=3)
    print(f"Found {len(results)} relevant items")

    # Get context window
    context = buffer.get_context_window(max_tokens=1000)
    print(f"\nContext window:\n{context}")

    # Summary
    summary = buffer.get_summary()
    print(f"\nMemory summary: {summary}")

    # Context manager example
    print("\n--- Context Manager Demo ---")
    ctx = ContextManager(working_memory=buffer)

    ctx.add_user_message("How do I handle errors in Python?")
    ctx.add_assistant_message("You can use try/except blocks...")
    ctx.add_tool_result("web_search", {"results": ["Python exceptions guide"]})

    llm_context = ctx.get_context_for_llm()
    print(f"\nLLM Context:\n{llm_context}")
