"""Telegram Inline Keyboard Builder — pure Python 3.14.

Parses button syntax from LLM output and builds Telegram
InlineKeyboardMarkup JSON for interactive buttons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# Pattern to find button syntax: [Button: label|callback_data]
BUTTON_PATTERN = re.compile(
    r"\[Button:\s*(.+?)\|(.+?)\]",
)


@dataclass
class InlineButton:
    """A single inline button."""
    text: str
    callback_data: str


@dataclass
class InlineKeyboard:
    """Collection of button rows."""
    rows: list[list[InlineButton]]

    def to_markup(self) -> dict[str, Any]:
        """Convert to Telegram InlineKeyboardMarkup JSON."""
        return {
            "inline_keyboard": [
                [
                    {"text": btn.text, "callback_data": btn.callback_data}
                    for btn in row
                ]
                for row in self.rows
            ]
        }


def parse_llm_buttons(text: str) -> tuple[str, InlineKeyboard | None]:
    """Parse button syntax from LLM output.

    Finds all [Button: label|callback_data] patterns, removes them
    from text, and returns the cleaned text + keyboard.

    Args:
        text: LLM output text

    Returns:
        Tuple of (cleaned_text, keyboard_or_None)
    """
    matches = BUTTON_PATTERN.findall(text)
    if not matches:
        return text, None

    # Remove button syntax from text
    cleaned = BUTTON_PATTERN.sub("", text).strip()
    # Clean up extra whitespace/newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Build keyboard (max 3 buttons per row)
    buttons = [
        InlineButton(text=label.strip(), callback_data=data.strip())
        for label, data in matches
    ]

    rows: list[list[InlineButton]] = []
    for i in range(0, len(buttons), 3):
        rows.append(buttons[i:i + 3])

    keyboard = InlineKeyboard(rows=rows)
    return cleaned, keyboard


def build_settings_keyboard() -> InlineKeyboard:
    """Build a built-in settings keyboard."""
    return InlineKeyboard(rows=[
        [
            InlineButton(text="Memory Stats", callback_data="/recall memory"),
            InlineButton(text="History", callback_data="/history"),
        ],
        [
            InlineButton(text="Help", callback_data="/help"),
        ],
    ])
