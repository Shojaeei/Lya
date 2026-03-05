"""Pytest configuration and fixtures."""

from __future__ import annotations

import asyncio
from pathlib import Path
from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from lya.domain.models.agent import Agent, AgentConfig
from lya.domain.models.goal import Goal, GoalPriority
from lya.domain.models.memory import Memory, MemoryType, MemoryImportance
from lya.domain.models.task import Task, TaskPriority


# ═══════════════════════════════════════════════════════════════
# Event Loop Fixture
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ═══════════════════════════════════════════════════════════════
# Domain Model Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def agent() -> Agent:
    """Create a test agent."""
    return Agent(
        config=AgentConfig(name="TestAgent"),
    )


@pytest.fixture
def goal() -> Goal:
    """Create a test goal."""
    return Goal(
        description="Complete a test task",
        priority=GoalPriority.HIGH,
    )


@pytest.fixture
def task(goal: Goal) -> Task:
    """Create a test task."""
    return Task(
        description="Execute test action",
        priority=TaskPriority.MEDIUM,
        goal_id=goal.id,
    )


@pytest.fixture
def memory(agent: Agent) -> Memory:
    """Create a test memory."""
    return Memory(
        agent_id=agent.id,
        content="Test memory content",
        memory_type=MemoryType.EPISODIC,
        importance=MemoryImportance.HIGH,
    )


# ═══════════════════════════════════════════════════════════════
# Async Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest_asyncio.fixture
async def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


# ═══════════════════════════════════════════════════════════════
# Mock Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_llm_port() -> MagicMock:
    """Create a mock LLM port."""
    mock = MagicMock()
    mock.generate = AsyncMock(return_value="Generated text")
    mock.chat = AsyncMock(return_value="Chat response")
    mock.embed = AsyncMock(return_value=[0.1] * 384)
    mock.generate_structured = AsyncMock(return_value={"result": "success"})
    return mock


@pytest.fixture
def mock_memory_repository() -> MagicMock:
    """Create a mock memory repository."""
    mock = MagicMock()
    mock.get = AsyncMock(return_value=None)
    mock.save = AsyncMock()
    mock.delete = AsyncMock(return_value=True)
    mock.search = AsyncMock(return_value=[])
    mock.get_by_agent = AsyncMock(return_value=[])
    mock.count = AsyncMock(return_value=0)
    return mock


@pytest.fixture
def mock_file_manager() -> MagicMock:
    """Create a mock file manager."""
    mock = MagicMock()
    mock.read_file = AsyncMock(return_value="File content")
    mock.write_file = AsyncMock()
    mock.exists = AsyncMock(return_value=True)
    mock.list_files = AsyncMock(return_value=[])
    return mock


# ═══════════════════════════════════════════════════════════════
# Configuration Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def test_settings():
    """Create test settings."""
    from lya.infrastructure.config.settings import Settings

    return Settings(
        environment="testing",
        debug=True,
        workspace_path="./test_workspace",
    )


# ═══════════════════════════════════════════════════════════════
# Integration Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest_asyncio.fixture
async def chroma_memory_repo(tmp_path: Path) -> AsyncGenerator:
    """Create a temporary ChromaDB repository."""
    from lya.infrastructure.persistence import ChromaMemoryRepository

    persist_dir = tmp_path / "chroma_test"
    repo = ChromaMemoryRepository(
        persist_directory=str(persist_dir),
        collection_name="test_memories",
    )
    await repo.connect()

    yield repo

    await repo.disconnect()


# ═══════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════

def create_mock_embedding(text: str, dimension: int = 384) -> list[float]:
    """Create a deterministic mock embedding."""
    import hashlib

    # Create deterministic embedding from text hash
    hash_bytes = hashlib.sha256(text.encode()).digest()
    # Normalize to create embedding vector
    import random
    random.seed(int.from_bytes(hash_bytes[:4], "big"))
    embedding = [random.uniform(-1, 1) for _ in range(dimension)]
    # Normalize
    norm = sum(x * x for x in embedding) ** 0.5
    return [x / norm for x in embedding]
