"""Conversation Summarizer — pure Python 3.14.

Summarizes old conversation history to compress context.
Uses the LLM to generate concise summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class ConversationSummarizer:
    """Summarize old conversations to compress context window."""

    def __init__(self, workspace: Path, threshold: int = 40):
        self._workspace = workspace / "summaries"
        self._workspace.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold

    def _file_for(self, chat_id: int) -> Path:
        return self._workspace / f"{chat_id}.json"

    def load_summary(self, chat_id: int) -> str:
        """Load existing summary for a chat."""
        f = self._file_for(chat_id)
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                return data.get("summary", "")
            except (json.JSONDecodeError, OSError):
                return ""
        return ""

    def save_summary(self, chat_id: int, summary: str) -> None:
        """Save summary for a chat."""
        f = self._file_for(chat_id)
        data = {"chat_id": chat_id, "summary": summary}
        f.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )

    def should_summarize(self, history: list[dict[str, Any]]) -> bool:
        """Check if history is long enough to need summarization."""
        return len(history) > self.threshold

    async def summarize(
        self,
        llm: Any,
        messages: list[dict[str, Any]],
        existing_summary: str = "",
    ) -> str:
        """Summarize a batch of messages using the LLM.

        Args:
            llm: LLM adapter with chat() method
            messages: List of message dicts with 'text', 'is_user', 'username'
            existing_summary: Previous summary to build upon

        Returns:
            New summary text
        """
        # Format messages for summarization
        formatted = []
        for msg in messages:
            role = msg.get("username", "User") if msg.get("is_user") else "Lya"
            text = msg.get("text", "")[:200]
            formatted.append(f"{role}: {text}")

        convo = "\n".join(formatted)

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a conversation summarizer. Create a concise summary "
                    "of the conversation below. Include: key topics discussed, "
                    "any facts or preferences the user mentioned, any tasks or "
                    "requests. Max 200 words."
                ),
            },
        ]

        if existing_summary:
            prompt_messages.append({
                "role": "user",
                "content": (
                    f"Previous summary:\n{existing_summary}\n\n"
                    f"New conversation to incorporate:\n{convo}\n\n"
                    "Create an updated summary combining both."
                ),
            })
        else:
            prompt_messages.append({
                "role": "user",
                "content": f"Summarize this conversation:\n{convo}",
            })

        try:
            summary = await llm.chat(
                messages=prompt_messages,
                temperature=0.3,
                max_tokens=500,
            )
            return summary.strip()
        except Exception as e:
            logger.error("summarization_failed", error=str(e))
            return existing_summary
