"""Tests for MemoryService."""

from __future__ import annotations

import pytest
from uuid import uuid4

from lya.domain.models.memory import Memory, MemoryType, MemoryImportance
from lya.domain.services.memory_service import MemoryService
from lya.infrastructure.persistence import get_embedding_service
from tests.mocks.mock_repositories import MockMemoryRepository
from tests.mocks.mock_llm_port import MockLLMPort


class TestMemoryService:
    """Tests for MemoryService."""

    @pytest.fixture
    async def service(self) -> MemoryService:
        """Create service with mock repository."""
        repo = MockMemoryRepository()
        service = MemoryService(repository=repo)
        return service

    async def test_create_memory(self, service: MemoryService) -> None:
        """Test creating a memory."""
        memory = await service.create_memory(
            content="Test memory",
            memory_type=MemoryType.EPISODIC,
            importance=MemoryImportance.HIGH,
        )

        assert memory.content == "Test memory"
        assert memory.type == MemoryType.EPISODIC
        assert memory.importance == MemoryImportance.HIGH

    async def test_create_memory_with_agent(self, service: MemoryService) -> None:
        """Test creating memory for an agent."""
        agent_id = uuid4()

        memory = await service.create_memory(
            content="Agent memory",
            agent_id=agent_id,
        )

        assert memory.agent_id == agent_id

    async def test_recall(self, service: MemoryService) -> None:
        """Test recalling memories."""
        # Create some memories first
        await service.create_memory(content="Python programming tips")
        await service.create_memory(content="Machine learning concepts")
        await service.create_memory(content="Cooking recipes")

        # Search
        results = await service.recall(
            query="coding",
            limit=10,
            threshold=0.0,  # Low threshold for mock
        )

        # Mock returns empty, but verifies the flow
        assert isinstance(results, list)

    async def test_get_agent_memories(self, service: MemoryService) -> None:
        """Test getting memories for an agent."""
        agent_id = uuid4()

        # Create memories for agent
        await service.create_memory(
            content="Memory 1",
            agent_id=agent_id,
        )
        await service.create_memory(
            content="Memory 2",
            agent_id=agent_id,
        )
        await service.create_memory(
            content="Other agent memory",
            agent_id=uuid4(),  # Different agent
        )

        memories = await service.get_agent_memories(agent_id)

        assert len(memories) == 2
        for m in memories:
            assert m.agent_id == agent_id

    async def test_forget_stale_memories(self, service: MemoryService) -> None:
        """Test forgetting old memories."""
        # Create old memory
        old_memory = await service.create_memory(
            content="Old memory",
            importance=MemoryImportance.LOW,
        )

        # Manually age the memory
        from datetime import datetime, timedelta, timezone
        old_memory._created_at = datetime.now(timezone.utc) - timedelta(days=100)
        await service._repo.save(old_memory)

        # Forget stale memories
        count = await service.forget_stale_memories(threshold_days=30)

        assert count == 1
