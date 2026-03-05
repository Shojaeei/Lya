"""Working Memory Buffer for Lya.

Short-term memory system for active context and reasoning.
Pure Python 3.14+ compatible.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any
from collections.abc import Iterator, Sequence


@dataclass
class MemoryItem:
    """Single item in working memory."""
    content: str
    importance: float  # 0.0 to 1.0
    category: str
    source: str
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "importance": self.importance,
            "category": self.category,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryItem:
        """Create from dictionary."""
        return cls(
            content=data["content"],
            importance=data["importance"],
            category=data["category"],
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data.get("last_accessed", data["timestamp"])),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
        )

    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()

    def relevance_score(self) -> float:
        """
        Calculate relevance score.

        Based on:
        - Importance (higher = better)
        - Recency (newer = better)
        - Access frequency (more = better)
        """
        # Base importance
        score = self.importance

        # Recency decay (decays over 1 hour)
        age_hours = self.age_seconds() / 3600
        recency_factor = max(0.1, 1.0 - (age_hours / 24))  # Decay over 24 hours
        score *= recency_factor

        # Access boost (logarithmic to prevent runaway)
        if self.access_count > 0:
            access_factor = 1.0 + (0.1 * min(self.access_count, 10))
            score *= access_factor

        return score


@dataclass
class ContextWindow:
    """Context window for LLM prompts."""
    items: list[MemoryItem]
    total_tokens: int = 0
    token_limit: int = 8000

    def to_prompt_context(self) -> str:
        """Convert to string for LLM prompt."""
        parts = []
        for item in self.items:
            parts.append(f"[{item.category.upper()}] {item.content}")
        return "\n".join(parts)


class WorkingMemoryBuffer:
    """
    Working memory buffer for short-term context.

    Features:
    - FIFO eviction with importance weighting
    - Automatic decay of old items
    - Relevance-based retrieval
    - Context window management
    """

    def __init__(
        self,
        max_items: int = 100,
        default_importance: float = 0.5,
        decay_interval_seconds: int = 300,  # 5 minutes
    ) -> None:
        """
        Initialize working memory.

        Args:
            max_items: Maximum items to store
            default_importance: Default importance for new items
            decay_interval_seconds: How often to decay old items
        """
        self.max_items = max_items
        self.default_importance = default_importance
        self.decay_interval = decay_interval_seconds

        self._buffer: deque[MemoryItem] = deque(maxlen=max_items)
        self._by_category: dict[str, list[MemoryItem]] = {}
        self._last_decay = time.time()

    # ═══════════════════════════════════════════════════════════════════
    # CORE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════

    def add(
        self,
        content: str,
        category: str = "general",
        importance: float | None = None,
        source: str = "user",
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> MemoryItem:
        """
        Add item to working memory.

        Args:
            content: Content to store
            category: Category for organization
            importance: Importance score (0-1)
            source: Source of the information
            metadata: Additional metadata
            embedding: Optional vector embedding

        Returns:
            Created memory item
        """
        # Check decay
        self._check_decay()

        item = MemoryItem(
            content=content,
            importance=importance or self.default_importance,
            category=category,
            source=source,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
            embedding=embedding,
        )

        # Add to main buffer
        self._buffer.append(item)

        # Add to category index
        if category not in self._by_category:
            self._by_category[category] = []
        self._by_category[category].append(item)

        return item

    def retrieve(
        self,
        query: str | None = None,
        category: str | None = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """
        Retrieve relevant items from working memory.

        Args:
            query: Optional search query
            category: Filter by category
            min_importance: Minimum importance threshold
            limit: Maximum items to return

        Returns:
            List of matching items
        """
        # Check decay
        self._check_decay()

        # Get candidate items
        if category:
            candidates = self._by_category.get(category, [])
        else:
            candidates = list(self._buffer)

        # Filter by importance
        filtered = [
            item for item in candidates
            if item.importance >= min_importance
        ]

        # Calculate relevance scores
        scored = [(item, item.relevance_score()) for item in filtered]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top N
        results = [item for item, _ in scored[:limit]]

        # Update access stats
        for item in results:
            item.access_count += 1
            item.last_accessed = datetime.now(timezone.utc)

        return results

    def get_recent(
        self,
        seconds: int = 300,
        category: str | None = None,
    ) -> list[MemoryItem]:
        """
        Get recent items.

        Args:
            seconds: Time window in seconds
            category: Optional category filter

        Returns:
            List of recent items
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=seconds)

        items = (
            self._by_category.get(category, [])
            if category else list(self._buffer)
        )

        return [
            item for item in items
            if item.timestamp >= cutoff
        ]

    def get_context_window(
        self,
        token_limit: int = 8000,
        categories: list[str] | None = None,
    ) -> ContextWindow:
        """
        Build context window for LLM.

        Args:
            token_limit: Maximum tokens for context
            categories: Categories to include

        Returns:
            Context window with selected items
        """
        # Check decay
        self._check_decay()

        # Get relevant items
        if categories:
            all_items = []
            for cat in categories:
                all_items.extend(self._by_category.get(cat, []))
        else:
            all_items = list(self._buffer)

        # Sort by relevance
        scored = [(item, item.relevance_score()) for item in all_items]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build context within token limit
        context_items = []
        total_tokens = 0

        for item, _ in scored:
            # Rough token estimation (4 chars per token)
            item_tokens = len(item.content) // 4 + 10  # +10 for overhead

            if total_tokens + item_tokens > token_limit:
                break

            context_items.append(item)
            total_tokens += item_tokens

        return ContextWindow(
            items=context_items,
            total_tokens=total_tokens,
            token_limit=token_limit,
        )

    def update_importance(
        self,
        content: str,
        new_importance: float,
    ) -> bool:
        """
        Update importance of an item.

        Args:
            content: Content to match
            new_importance: New importance value

        Returns:
            True if item was found and updated
        """
        for item in self._buffer:
            if item.content == content:
                item.importance = max(0.0, min(1.0, new_importance))
                return True
        return False

    def clear(self, category: str | None = None) -> int:
        """
        Clear memory items.

        Args:
            category: Clear only this category, or all if None

        Returns:
            Number of items cleared
        """
        if category:
            # Clear specific category
            count = len(self._by_category.get(category, []))

            # Remove from buffer
            self._buffer = deque(
                [item for item in self._buffer if item.category != category],
                maxlen=self.max_items,
            )

            # Clear from category index
            self._by_category[category] = []

            return count
        else:
            # Clear all
            count = len(self._buffer)
            self._buffer.clear()
            self._by_category.clear()
            return count

    # ═══════════════════════════════════════════════════════════════════
    # DECAY AND MAINTENANCE
    # ═══════════════════════════════════════════════════════════════════

    def _check_decay(self) -> None:
        """Check if decay should run."""
        now = time.time()
        if now - self._last_decay > self.decay_interval:
            self._decay_old_items()
            self._last_decay = now

    def _decay_old_items(self) -> None:
        """Decay importance of old items."""
        now = datetime.now(timezone.utc)

        for item in self._buffer:
            age_hours = (now - item.timestamp).total_seconds() / 3600

            if age_hours > 1:  # Decay items older than 1 hour
                # Decay factor based on age
                decay = 0.9 ** (age_hours / 6)  # 10% decay every 6 hours
                item.importance *= decay

    def consolidate_to_long_term(
        self,
        min_importance: float = 0.7,
        min_access_count: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Identify items to consolidate to long-term memory.

        Args:
            min_importance: Minimum importance to consider
            min_access_count: Minimum access count to consider

        Returns:
            List of items ready for consolidation
        """
        to_consolidate = []

        for item in self._buffer:
            if (item.importance >= min_importance and
                item.access_count >= min_access_count):
                to_consolidate.append(item.to_dict())

        return to_consolidate

    # ═══════════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════════

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        now = datetime.now(timezone.utc)

        # Calculate category distribution
        category_counts = {
            cat: len(items) for cat, items in self._by_category.items()
        }

        # Calculate average importance
        avg_importance = sum(item.importance for item in self._buffer) / max(1, len(self._buffer))

        # Find oldest and newest
        ages = [(now - item.timestamp).total_seconds() for item in self._buffer]

        return {
            "total_items": len(self._buffer),
            "max_items": self.max_items,
            "categories": list(self._by_category.keys()),
            "category_counts": category_counts,
            "avg_importance": avg_importance,
            "oldest_item_seconds": max(ages) if ages else 0,
            "newest_item_seconds": min(ages) if ages else 0,
        }

    def to_list(self) -> list[dict[str, Any]]:
        """Export all items as list of dicts."""
        return [item.to_dict() for item in self._buffer]

    def save_to_file(self, path: str) -> None:
        """Save memory to JSON file."""
        data = {
            "items": self.to_list(),
            "stats": self.get_stats(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, path: str) -> None:
        """Load memory from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        self.clear()

        for item_data in data.get("items", []):
            item = MemoryItem.from_dict(item_data)
            self._buffer.append(item)

            # Rebuild category index
            if item.category not in self._by_category:
                self._by_category[item.category] = []
            self._by_category[item.category].append(item)

    def __len__(self) -> int:
        """Get number of items in memory."""
        return len(self._buffer)

    def __iter__(self) -> Iterator[MemoryItem]:
        """Iterate over memory items."""
        return iter(self._buffer)

    def __contains__(self, content: str) -> bool:
        """Check if content exists in memory."""
        return any(item.content == content for item in self._buffer)


# ═══════════════════════════════════════════════════════════════════════
# CONTEXT MANAGER FOR LLM INTERACTIONS
# ═══════════════════════════════════════════════════════════════════════

class WorkingMemoryContext:
    """
    Context manager for LLM interactions.

    Automatically manages working memory during conversations.
    """

    def __init__(
        self,
        memory: WorkingMemoryBuffer | None = None,
        system_prompt: str | None = None,
        context_categories: list[str] | None = None,
    ) -> None:
        """Initialize context.

        Args:
            memory: Working memory buffer
            system_prompt: System prompt to include
            context_categories: Categories to include in context
        """
        self.memory = memory or WorkingMemoryBuffer()
        self.system_prompt = system_prompt
        self.context_categories = context_categories or ["system", "user", "agent"]
        self.messages: list[dict[str, Any]] = []

    def add_message(
        self,
        role: str,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to context."""
        # Add to memory
        self.memory.add(
            content=content,
            category=role,
            importance=importance,
            source=role,
            metadata=metadata or {},
        )

        # Add to message history
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_messages(
        self,
        include_system: bool = True,
        recent_only: int | None = None,
    ) -> list[dict[str, str]]:
        """Get messages formatted for LLM."""
        messages = []

        # Add system prompt
        if include_system and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add context from working memory
        context = self.memory.get_context_window(
            categories=self.context_categories,
        )
        if context.items:
            context_str = context.to_prompt_context()
            messages.append({"role": "system", "content": f"Context:\n{context_str}"})

        # Add recent messages
        msgs = self.messages
        if recent_only:
            msgs = msgs[-recent_only:]

        for msg in msgs:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        return messages

    def get_context_string(self) -> str:
        """Get context as formatted string."""
        context = self.memory.get_context_window(
            categories=self.context_categories,
        )
        return context.to_prompt_context()


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

_default_buffer: WorkingMemoryBuffer | None = None


def get_working_memory() -> WorkingMemoryBuffer:
    """Get global working memory instance."""
    global _default_buffer
    if _default_buffer is None:
        _default_buffer = WorkingMemoryBuffer()
    return _default_buffer


def remember(
    content: str,
    category: str = "general",
    importance: float = 0.5,
) -> MemoryItem:
    """Quickly add to working memory."""
    return get_working_memory().add(content, category, importance)


def recall(
    query: str | None = None,
    category: str | None = None,
    limit: int = 5,
) -> list[MemoryItem]:
    """Quickly retrieve from working memory."""
    return get_working_memory().retrieve(query, category, limit=limit)


def context(
    token_limit: int = 4000,
) -> str:
    """Get working memory context."""
    return get_working_memory().get_context_window(token_limit).to_prompt_context()


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create working memory
    memory = WorkingMemoryBuffer(max_items=50)

    # Add items
    memory.add(
        content="User wants to build a Python web app",
        category="goal",
        importance=0.9,
    )
    memory.add(
        content="User prefers FastAPI over Flask",
        category="preference",
        importance=0.7,
    )
    memory.add(
        content="Current project: Lya AI Agent",
        category="context",
        importance=0.8,
    )

    # Retrieve relevant items
    print("Top items:")
    for item in memory.retrieve(limit=5):
        print(f"  [{item.category}] {item.content[:50]}...")
        print(f"    Score: {item.relevance_score():.2f}")

    # Get context window
    context = memory.get_context_window(token_limit=1000)
    print(f"\nContext window ({context.total_tokens} tokens):")
    print(context.to_prompt_context())

    # Stats
    print(f"\nStats: {memory.get_stats()}")
