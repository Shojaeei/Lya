"""Capability templates for self-improvement.

Pre-built capability implementations that can be:
- Registered directly with the capability registry
- Used as templates for LLM-generated capabilities
- Extended and customized for specific use cases
"""

from pathlib import Path
from typing import Any

# Available templates
TEMPLATES = {
    "browser_operations": "browser_operations.py",
    "filesystem_operations": "filesystem_operations.py",
    "git_operations": "git_operations.py",
}


def get_template_path(template_name: str) -> Path | None:
    """Get the path to a capability template."""
    if template_name not in TEMPLATES:
        return None
    return Path(__file__).parent / TEMPLATES[template_name]


def list_templates() -> list[str]:
    """List available capability templates."""
    return list(TEMPLATES.keys())


def load_template(template_name: str) -> str | None:
    """Load a template file content."""
    path = get_template_path(template_name)
    if not path or not path.exists():
        return None
    return path.read_text(encoding="utf-8")


__all__ = [
    "TEMPLATES",
    "get_template_path",
    "list_templates",
    "load_template",
]
