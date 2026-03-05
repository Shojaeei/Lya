"""Outgoing port for tool operations."""

from typing import Any, Protocol


class ToolPort(Protocol):
    """
    Outgoing port for tool execution.

    This port abstracts tool implementations from the application layer.
    Infrastructure adapters implement this protocol.
    """

    async def execute(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool."""
        ...

    def get_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        """Get schema for a tool."""
        ...

    def list_tools(self) -> list[dict[str, Any]]:
        """List all available tools."""
        ...

    def register_tool(
        self,
        name: str,
        handler: callable,
        schema: dict[str, Any],
    ) -> None:
        """Register a new tool."""
        ...

    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available."""
        ...
