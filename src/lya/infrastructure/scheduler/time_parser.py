"""Time expression parser — pure Python 3.14.

Parses natural language time expressions into datetime objects.
Uses regex patterns for common expressions.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)

# Patterns for common time expressions
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "in X minutes/hours/seconds/days"
    (re.compile(r"in\s+(\d+)\s*(?:min(?:ute)?s?)", re.I), "minutes"),
    (re.compile(r"in\s+(\d+)\s*(?:hour?s?|hr?s?)", re.I), "hours"),
    (re.compile(r"in\s+(\d+)\s*(?:sec(?:ond)?s?)", re.I), "seconds"),
    (re.compile(r"in\s+(\d+)\s*(?:day?s?)", re.I), "days"),
    (re.compile(r"in\s+(\d+)\s*(?:week?s?)", re.I), "weeks"),
]

# "at HH:MM" pattern
_AT_TIME = re.compile(
    r"at\s+(\d{1,2}):(\d{2})(?:\s*(am|pm))?", re.I,
)

# "tomorrow" with optional time
_TOMORROW = re.compile(
    r"tomorrow(?:\s+(?:at\s+)?(\d{1,2}):(\d{2})(?:\s*(am|pm))?)?", re.I,
)

# Recurring patterns
_RECURRING_PATTERN = re.compile(
    r"every\s+(hour|day|week|daily|weekly|hourly)", re.I,
)


def parse_time_expression(
    text: str,
    reference: datetime | None = None,
) -> tuple[datetime | None, str | None]:
    """Parse a natural language time expression.

    Args:
        text: The text containing a time expression
        reference: Reference datetime (defaults to now UTC)

    Returns:
        Tuple of (trigger_time, recurring_type) or (None, None) if unparsable
    """
    now = reference or datetime.now(timezone.utc)
    text_lower = text.lower().strip()

    # Check for recurring pattern
    recurring = None
    rec_match = _RECURRING_PATTERN.search(text_lower)
    if rec_match:
        rec_type = rec_match.group(1).lower()
        recurring_map = {
            "hour": "hourly", "hourly": "hourly",
            "day": "daily", "daily": "daily",
            "week": "weekly", "weekly": "weekly",
        }
        recurring = recurring_map.get(rec_type)

    # Try "in X units" patterns
    for pattern, unit in _PATTERNS:
        match = pattern.search(text_lower)
        if match:
            value = int(match.group(1))
            delta = timedelta(**{unit: value})
            return now + delta, recurring

    # Try "at HH:MM" pattern
    at_match = _AT_TIME.search(text_lower)
    if at_match:
        hour = int(at_match.group(1))
        minute = int(at_match.group(2))
        ampm = at_match.group(3)
        if ampm:
            if ampm.lower() == "pm" and hour < 12:
                hour += 12
            elif ampm.lower() == "am" and hour == 12:
                hour = 0
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target, recurring

    # Try "tomorrow"
    tom_match = _TOMORROW.search(text_lower)
    if tom_match:
        tomorrow = now + timedelta(days=1)
        if tom_match.group(1):
            hour = int(tom_match.group(1))
            minute = int(tom_match.group(2))
            ampm = tom_match.group(3)
            if ampm:
                if ampm.lower() == "pm" and hour < 12:
                    hour += 12
                elif ampm.lower() == "am" and hour == 12:
                    hour = 0
            tomorrow = tomorrow.replace(
                hour=hour, minute=minute, second=0, microsecond=0,
            )
        else:
            tomorrow = tomorrow.replace(
                hour=9, minute=0, second=0, microsecond=0,
            )
        return tomorrow, recurring

    # If only recurring found but no time, default to next interval
    if recurring:
        delta_map = {
            "hourly": timedelta(hours=1),
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
        }
        delta = delta_map.get(recurring, timedelta(hours=1))
        return now + delta, recurring

    return None, None


def format_task_time(iso_time: str) -> str:
    """Format an ISO datetime for display."""
    try:
        dt = datetime.fromisoformat(iso_time)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = dt - now

        if diff.total_seconds() < 0:
            return "overdue"
        elif diff.total_seconds() < 60:
            return f"in {int(diff.total_seconds())}s"
        elif diff.total_seconds() < 3600:
            return f"in {int(diff.total_seconds() / 60)}min"
        elif diff.total_seconds() < 86400:
            return f"in {diff.total_seconds() / 3600:.1f}h"
        else:
            return dt.strftime("%Y-%m-%d %H:%M UTC")

    except (ValueError, TypeError):
        return iso_time
