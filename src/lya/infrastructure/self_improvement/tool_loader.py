"""Tool Loader for self-improvement.

Dynamically loads and attaches tools to the agent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class ToolLoader:
    """
    Dynamically load and attach tools to the agent.

    Features:
    - Load tools from Python files
    - Hot reload support
    - Tool caching
    - Safe execution context
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.tools_dir = workspace / "tools"
        self.generated_dir = workspace / "tools" / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        self.loaded_tools: dict[str, Callable] = {}
        self.tool_modules: dict[str, Any] = {}

    def load_tool(self, tool_name: str) -> Callable | None:
        """
        Load a tool from the tools directory.

        Args:
            tool_name: Name of the tool to load

        Returns:
            Loaded tool function or None
        """
        # Check if already loaded
        if tool_name in self.loaded_tools:
            return self.loaded_tools[tool_name]

        # Check generated tools first
        tool_path = self.generated_dir / f"{tool_name}.py"
        if not tool_path.exists():
            tool_path = self.tools_dir / f"{tool_name}.py"

        if not tool_path.exists():
            logger.warning("Tool file not found", tool_name=tool_name)
            return None

        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(tool_name, tool_path)
            if not spec or not spec.loader:
                logger.error("Failed to create module spec", tool_name=tool_name)
                return None

            module = importlib.util.module_from_spec(spec)

            # Remove from sys.modules if exists (for reloading)
            if tool_name in sys.modules:
                del sys.modules[tool_name]

            sys.modules[tool_name] = module
            spec.loader.exec_module(module)

            # Find the main function
            tool_func = None

            # Look for 'main' function first
            if hasattr(module, "main"):
                tool_func = getattr(module, "main")
            # Then look for function matching tool name
            elif hasattr(module, tool_name):
                tool_func = getattr(module, tool_name)
            # Look for any callable that's not a module import
            else:
                for name in dir(module):
                    obj = getattr(module, name)
                    if callable(obj) and not name.startswith("_"):
                        tool_func = obj
                        break

            if tool_func:
                self.loaded_tools[tool_name] = tool_func
                self.tool_modules[tool_name] = module

                logger.info("Tool loaded successfully", tool_name=tool_name)
                return tool_func
            else:
                logger.error("No callable found in tool module", tool_name=tool_name)
                return None

        except Exception as e:
            logger.error("Error loading tool", tool_name=tool_name, error=str(e))
            return None

    def unload_tool(self, tool_name: str) -> bool:
        """
        Unload a tool.

        Args:
            tool_name: Name of the tool to unload

        Returns:
            True if unloaded successfully
        """
        if tool_name in self.loaded_tools:
            del self.loaded_tools[tool_name]

        if tool_name in self.tool_modules:
            del self.tool_modules[tool_name]

        if tool_name in sys.modules:
            del sys.modules[tool_name]

        logger.info("Tool unloaded", tool_name=tool_name)
        return True

    def reload_tool(self, tool_name: str) -> Callable | None:
        """
        Reload a tool.

        Args:
            tool_name: Name of the tool to reload

        Returns:
            Reloaded tool function or None
        """
        self.unload_tool(tool_name)
        return self.load_tool(tool_name)

    def list_tools(self) -> list[str]:
        """List all available tools."""
        tools = []

        # Core tools
        if self.tools_dir.exists():
            tools.extend(
                [f.stem for f in self.tools_dir.glob("*.py") if f.stem != "__init__"]
            )

        # Generated tools
        if self.generated_dir.exists():
            generated = [
                f"generated/{f.stem}"
                for f in self.generated_dir.glob("*.py")
                if f.stem != "__init__"
            ]
            tools.extend(generated)

        return sorted(set(tools))

    def list_generated_tools(self) -> list[str]:
        """List only generated tools."""
        tools = []
        if self.generated_dir.exists():
            tools = [
                f.stem
                for f in self.generated_dir.glob("*.py")
                if f.stem != "__init__"
            ]
        return tools

    def get_tool_info(self, tool_name: str) -> dict[str, Any] | None:
        """Get information about a loaded tool."""
        if tool_name not in self.loaded_tools:
            return None

        func = self.loaded_tools[tool_name]
        module = self.tool_modules.get(tool_name)

        import inspect

        info = {
            "name": tool_name,
            "loaded": True,
            "docstring": inspect.getdoc(func),
            "signature": str(inspect.signature(func)),
            "file": str(module.__file__) if module else None,
        }

        return info

    def is_loaded(self, tool_name: str) -> bool:
        """Check if a tool is loaded."""
        return tool_name in self.loaded_tools

    async def execute_tool(
        self,
        tool_name: str,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute a loaded tool.

        Args:
            tool_name: Name of the tool
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tool execution result
        """
        tool = self.loaded_tools.get(tool_name)
        if not tool:
            # Try to load it
            tool = self.load_tool(tool_name)
            if not tool:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found or could not be loaded",
                }

        try:
            import asyncio

            # Check if tool is async
            if asyncio.iscoroutinefunction(tool):
                result = await tool(*args, **kwargs)
            else:
                result = tool(*args, **kwargs)

            # Normalize result
            if isinstance(result, dict):
                if "success" not in result:
                    result["success"] = True
                return result
            else:
                return {"success": True, "result": result}

        except Exception as e:
            logger.error("Tool execution failed", tool_name=tool_name, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
            }
