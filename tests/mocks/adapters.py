"""Mock adapters for testing."""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

from lya.application.ports.outgoing.llm_port import LLMPort


class MockLLMAdapter(LLMPort):
    """Mock LLM adapter for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.default_response = "Mock response"
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Mock text generation."""
        self.calls.append({
            "method": "generate",
            "prompt": prompt,
            "temperature": temperature,
            "system_prompt": system_prompt,
        })
        return self.responses.get(prompt, self.default_response)

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Mock chat completion."""
        self.calls.append({
            "method": "chat",
            "messages": messages,
            "temperature": temperature,
        })
        return self.default_response

    async def generate_structured(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Mock structured generation."""
        self.calls.append({
            "method": "generate_structured",
            "prompt": prompt,
            "schema": output_schema,
        })
        return {"result": "mock"}

    async def generate_stream(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Mock streaming generation."""
        yield "Mock "
        yield "streaming "
        yield "response"

    async def embed(self, text: str) -> list[float]:
        """Mock embedding generation."""
        # Return fixed-size embedding
        return [0.1] * 384

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Mock batch embedding."""
        return [[0.1] * 384 for _ in texts]

    def get_model_info(self) -> dict[str, Any]:
        """Get mock model info."""
        return {
            "provider": "mock",
            "model": "mock-model",
        }

    async def list_models(self) -> list[dict[str, Any]]:
        """List mock models."""
        return [
            {"id": "mock-model", "name": "Mock Model"},
        ]

    async def close(self) -> None:
        """Close mock adapter."""
        pass


class MockGitAdapter:
    """Mock Git adapter for testing."""

    def __init__(self):
        self.commits = []
        self.branches = ["main"]
        self.current_branch = "main"

    async def init(self, path):
        return True

    async def clone(self, url, destination=None, branch=None):
        return destination or path("/mock/clone")

    async def commit(self, message, repo_path=None):
        from lya.infrastructure.tools.git.git_adapter import GitCommit
        commit = GitCommit(
            hash="abc123",
            message=message,
            author="Test",
            timestamp="2026-01-01T00:00:00Z",
        )
        self.commits.append(commit)
        return commit

    async def status(self, repo_path=None):
        from lya.infrastructure.tools.git.git_adapter import GitStatus
        return GitStatus(
            branch=self.current_branch,
            is_clean=True,
            modified=[],
            staged=[],
            untracked=[],
        )
