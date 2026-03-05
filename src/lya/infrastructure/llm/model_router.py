"""Multi-Model Router — pure Python 3.14.

Routes different task types to different LLM models.
Uses keyword classification to detect task type and select
the best model from configured routes.
"""

from __future__ import annotations

import re
from typing import Any

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)

# Keyword patterns for task classification
_TASK_PATTERNS: dict[str, list[str]] = {
    "code": [
        "code", "function", "class", "debug", "error", "bug", "implement",
        "python", "javascript", "typescript", "java", "rust", "go",
        "algorithm", "refactor", "```", "def ", "import ", "from ",
        "compile", "syntax", "programming", "developer", "api",
    ],
    "vision": [
        "image", "photo", "picture", "screenshot", "diagram",
        "describe what you see", "what's in this",
    ],
    "analysis": [
        "analyze", "analysis", "compare", "evaluate", "review",
        "assess", "benchmark", "metrics", "statistics", "data",
        "report", "summary of data",
    ],
    "creative": [
        "write a story", "poem", "creative", "fiction", "imagine",
        "roleplay", "scenario", "narrative", "song",
    ],
}


class ModelRouter:
    """Route tasks to appropriate models based on content classification."""

    def __init__(
        self,
        default_model: str | None = None,
        routes: dict[str, str] | None = None,
    ):
        self.default_model = default_model or settings.llm.model
        self.routes = routes or dict(settings.llm.model_routes)
        self.enabled = settings.llm.routing_enabled and bool(self.routes)

        if self.enabled:
            logger.info(
                "model_router_enabled",
                default=self.default_model,
                routes=self.routes,
            )

    def classify_task(self, text: str) -> str:
        """Classify a task based on keyword matching.

        Returns task type: "code", "vision", "analysis", "creative", or "chat"
        """
        text_lower = text.lower()

        scores: dict[str, int] = {}
        for task_type, keywords in _TASK_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[task_type] = score

        if not scores:
            return "chat"

        return max(scores, key=scores.get)

    def get_model_for_task(self, text: str) -> str:
        """Get the best model name for a given task.

        Returns the routed model or default model.
        """
        if not self.enabled:
            return self.default_model

        task_type = self.classify_task(text)
        model = self.routes.get(task_type, self.default_model)

        if model != self.default_model:
            logger.debug(
                "model_routed",
                task_type=task_type,
                model=model,
            )

        return model

    def get_model_info(self) -> dict[str, Any]:
        """Get router configuration info."""
        return {
            "enabled": self.enabled,
            "default_model": self.default_model,
            "routes": dict(self.routes),
        }
