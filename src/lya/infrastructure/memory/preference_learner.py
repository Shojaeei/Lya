"""Preference Learner — pure Python 3.14.

Detects user preferences from conversation patterns
and adapts bot behavior accordingly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)

# Patterns for detecting explicit preference signals
_PREFERENCE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Language preferences
    (re.compile(r"i\s+(?:prefer|like|use)\s+(python|javascript|typescript|java|rust|go|c\+\+)", re.I),
     "code_language", ""),
    # Verbosity preferences
    (re.compile(r"(?:keep\s+it|be\s+(?:more\s+)?)(brief|concise|short|detailed|verbose)", re.I),
     "verbosity", ""),
    # Language (spoken)
    (re.compile(r"(?:speak|reply|respond|talk)\s+(?:in|to\s+me\s+in)\s+(\w+)", re.I),
     "language", ""),
    (re.compile(r"(?:i\s+speak|my\s+language\s+is)\s+(\w+)", re.I),
     "language", ""),
    # Timezone
    (re.compile(r"(?:my\s+timezone?\s+is|i'?m?\s+in)\s+([A-Z]{2,5}(?:[+-]\d+)?)", re.I),
     "timezone", ""),
]


@dataclass
class UserPreferences:
    """Learned user preferences."""
    language: str | None = None
    timezone: str | None = None
    verbosity: str = "balanced"  # "concise" | "balanced" | "detailed"
    topics_of_interest: list[str] = field(default_factory=list)
    code_language: str | None = None
    custom: dict[str, str] = field(default_factory=dict)

    def to_prompt_context(self) -> str:
        """Format preferences for injection into system prompt."""
        parts = []
        if self.language:
            parts.append(f"Preferred language: {self.language}")
        if self.timezone:
            parts.append(f"Timezone: {self.timezone}")
        if self.verbosity != "balanced":
            parts.append(f"Response style: {self.verbosity}")
        if self.code_language:
            parts.append(f"Preferred coding language: {self.code_language}")
        if self.topics_of_interest:
            parts.append(f"Topics of interest: {', '.join(self.topics_of_interest[:5])}")
        for k, v in list(self.custom.items())[:5]:
            parts.append(f"{k}: {v}")

        if not parts:
            return ""
        return "## User Preferences\n" + "\n".join(f"- {p}" for p in parts)


class PreferenceLearner:
    """Learn and persist user preferences."""

    def __init__(self, workspace: Path):
        self._workspace = workspace / "preferences"
        self._workspace.mkdir(parents=True, exist_ok=True)

    def _file_for(self, chat_id: int) -> Path:
        return self._workspace / f"{chat_id}.json"

    def load_preferences(self, chat_id: int) -> UserPreferences:
        """Load preferences for a chat."""
        f = self._file_for(chat_id)
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                return UserPreferences(**{
                    k: v for k, v in data.items()
                    if k in UserPreferences.__dataclass_fields__
                })
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        return UserPreferences()

    def save_preferences(self, chat_id: int, prefs: UserPreferences) -> None:
        """Save preferences for a chat."""
        f = self._file_for(chat_id)
        f.write_text(
            json.dumps(asdict(prefs), ensure_ascii=False),
            encoding="utf-8",
        )

    def detect_preference_signal(self, text: str) -> dict[str, str] | None:
        """Detect explicit preference statements in text.

        Returns dict of {key: value} if a preference is detected, else None.
        """
        for pattern, key, _ in _PREFERENCE_PATTERNS:
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                return {key: value}
        return None

    def update_from_signal(
        self, chat_id: int, signal: dict[str, str],
    ) -> UserPreferences:
        """Update preferences from a detected signal."""
        prefs = self.load_preferences(chat_id)

        for key, value in signal.items():
            if key == "language":
                prefs.language = value
            elif key == "timezone":
                prefs.timezone = value
            elif key == "verbosity":
                verb_map = {
                    "brief": "concise", "concise": "concise", "short": "concise",
                    "detailed": "detailed", "verbose": "detailed",
                }
                prefs.verbosity = verb_map.get(value.lower(), "balanced")
            elif key == "code_language":
                prefs.code_language = value
            else:
                prefs.custom[key] = value

        self.save_preferences(chat_id, prefs)
        logger.info("preference_updated", chat_id=chat_id, signal=signal)
        return prefs

    async def analyze_conversation(
        self,
        llm: Any,
        messages: list[dict[str, Any]],
        existing: UserPreferences,
    ) -> UserPreferences:
        """Use LLM to analyze conversation and extract preferences.

        This is called periodically (every N messages).
        """
        formatted = []
        for msg in messages[-20:]:
            role = "User" if msg.get("is_user") else "Lya"
            formatted.append(f"{role}: {msg.get('text', '')[:150]}")

        convo = "\n".join(formatted)

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "Analyze the conversation and extract user preferences. "
                    "Return ONLY a JSON object with these optional fields: "
                    '"language", "timezone", "verbosity" (concise/balanced/detailed), '
                    '"topics_of_interest" (list), "code_language". '
                    "Only include fields you're confident about."
                ),
            },
            {
                "role": "user",
                "content": f"Conversation:\n{convo}",
            },
        ]

        try:
            result = await llm.chat(
                messages=prompt_messages,
                temperature=0.1,
                max_tokens=300,
            )

            # Parse JSON from response
            result = result.strip()
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]

            data = json.loads(result.strip())

            # Update preferences
            if lang := data.get("language"):
                existing.language = lang
            if tz := data.get("timezone"):
                existing.timezone = tz
            if verb := data.get("verbosity"):
                existing.verbosity = verb
            if topics := data.get("topics_of_interest"):
                existing.topics_of_interest = list(set(
                    existing.topics_of_interest + topics
                ))[:10]
            if code_lang := data.get("code_language"):
                existing.code_language = code_lang

            return existing

        except (json.JSONDecodeError, Exception) as e:
            logger.debug("preference_analysis_failed", error=str(e))
            return existing
