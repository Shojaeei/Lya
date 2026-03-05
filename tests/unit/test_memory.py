"""Tests for Memory domain model."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from uuid import UUID

from lya.domain.models.memory import (
    Memory,
    MemoryType,
    MemoryImportance,
    MemoryContext,
)


class TestMemory:
    """Tests for Memory class."""

    def test_create_basic_memory(self) -> None:
        """Test creating a basic memory."""
        memory = Memory(
            content="Test content",
            memory_type=MemoryType.EPISODIC,
        )

        assert memory.content == "Test content"
        assert memory.type == MemoryType.EPISODIC
        assert memory.importance == MemoryImportance.MEDIUM
        assert isinstance(memory.id, UUID)
        assert memory.access_count == 0
        assert not memory.is_consolidated

    def test_create_memory_with_context(self) -> None:
        """Test creating memory with context."""
        context = MemoryContext(
            source="test",
            tags=["important", "test"],
            metadata={"key": "value"},
        )

        memory = Memory(
            content="Test with context",
            memory_type=MemoryType.SEMANTIC,
            importance=MemoryImportance.HIGH,
            context=context,
        )

        assert memory.context.source == "test"
        assert memory.context.tags == ["important", "test"]
        assert memory.context.metadata == {"key": "value"}

    def test_memory_access(self) -> None:
        """Test accessing a memory updates counters."""
        memory = Memory(content="Test")

        assert memory.access_count == 0
        assert memory.accessed_at is None

        memory.access()

        assert memory.access_count == 1
        assert memory.accessed_at is not None

    def test_update_content(self) -> None:
        """Test updating memory content."""
        memory = Memory(content="Original")
        memory.set_embedding([0.1, 0.2, 0.3])

        assert memory.content == "Original"
        assert memory.embedding is not None

        memory.update_content("Updated")

        assert memory.content == "Updated"
        assert memory.embedding is None  # Embedding cleared on update

    def test_mark_consolidated(self) -> None:
        """Test marking memory as consolidated."""
        memory = Memory(content="Test")

        assert not memory.is_consolidated

        memory.mark_consolidated()

        assert memory.is_consolidated

    def test_age_hours(self) -> None:
        """Test age calculation."""
        memory = Memory(content="Test")

        # Memory should be very young
        assert memory.age_hours < 0.1

    def test_relevance_score_with_embedding(self) -> None:
        """Test relevance score calculation."""
        memory = Memory(content="Test")
        memory.set_embedding([1.0, 0.0, 0.0])

        # Same direction should give high score
        score = memory.calculate_relevance_score([1.0, 0.0, 0.0])
        assert score > 0

        # Opposite direction should give lower score
        score = memory.calculate_relevance_score([-1.0, 0.0, 0.0])
        assert score < 0.5

    def test_relevance_score_without_embedding(self) -> None:
        """Test relevance score without embedding."""
        memory = Memory(content="Test")
        # No embedding set

        score = memory.calculate_relevance_score([1.0, 0.0, 0.0])
        assert score == 0.0

    def test_should_forget_critical(self) -> None:
        """Test that critical memories are not forgotten."""
        memory = Memory(
            content="Critical info",
            importance=MemoryImportance.CRITICAL,
        )

        # Critical memories should never be forgotten
        assert not memory.should_forget(threshold_days=1)

    def test_should_forget_old_unused(self) -> None:
        """Test forgetting old, unused memories."""
        # Create memory with very old timestamp
        memory = Memory(
            content="Old info",
            importance=MemoryImportance.LOW,
        )

        # Manually set creation time to long ago
        from datetime import datetime, timedelta
        memory._created_at = datetime.now(timezone.utc) - timedelta(days=100)

        # Should suggest forgetting old, low importance, unaccessed memory
        assert memory.should_forget(threshold_days=30)

    def test_generate_summary(self) -> None:
        """Test summary generation."""
        memory = Memory(
            content="This is a very long content that needs to be summarized",
            memory_type=MemoryType.EPISODIC,
        )

        summary = memory.generate_summary(max_length=20)

        assert memory.type.name in summary
        assert len(summary) <= 40  # Type name prefix + truncated content + ellipsis

    def test_serialization(self) -> None:
        """Test memory serialization/deserialization."""
        original = Memory(
            content="Test content",
            memory_type=MemoryType.SEMANTIC,
            importance=MemoryImportance.HIGH,
        )
        original.access()

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = Memory.from_dict(data)

        assert restored.content == original.content
        assert restored.type == original.type
        assert restored.importance == original.importance
        assert restored.id == original.id
        assert restored.access_count == original.access_count


class TestMemoryContext:
    """Tests for MemoryContext class."""

    def test_create_context(self) -> None:
        """Test creating memory context."""
        context = MemoryContext(
            source="test_source",
            tags=["tag1", "tag2"],
        )

        assert context.source == "test_source"
        assert context.tags == ["tag1", "tag2"]
        assert context.goal_id is None
        assert context.task_id is None

    def test_context_serialization(self) -> None:
        """Test context serialization."""
        original = MemoryContext(
            source="test",
            tags=["a", "b"],
            metadata={"key": "value"},
        )

        data = original.to_dict()
        restored = MemoryContext.from_dict(data)

        assert restored.source == original.source
        assert restored.tags == original.tags
        assert restored.metadata == original.metadata
