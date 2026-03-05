"""Tool Generator for self-improvement.

Generates new tools from natural language descriptions.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from lya.domain.models.improvement import ToolDefinition, ImprovementType, Improvement
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class ToolGenerator:
    """
    Generates new tools from natural language descriptions.

    Uses LLM to generate Python code for tools based on descriptions.
    Validates generated code for safety before saving.
    """

    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        "exec",
        "eval",
        "__import__",
        "os.system",
        "subprocess.call",
        "subprocess.run",
        "subprocess.Popen",
        "compile",
        "open('/etc",
        "open('C:/Windows",
    ]

    def __init__(self, workspace: Path, llm_interface):
        self.workspace = workspace
        self.tools_dir = workspace / "tools" / "generated"
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.model = llm_interface

    async def generate_tool(
        self,
        description: str,
        tool_name: str | None = None,
        test_cases: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a new tool from natural language description.

        Args:
            description: Natural language description of what the tool should do
            tool_name: Optional name for the tool
            test_cases: Optional test cases the tool should pass

        Returns:
            Dictionary with generation results
        """
        if not tool_name:
            tool_name = f"tool_{uuid4().hex[:8]}"

        # Build generation prompt
        prompt = self._build_generation_prompt(description, test_cases)

        logger.info(
            "Generating tool",
            tool_name=tool_name,
            description=description[:100],
        )

        try:
            # Generate code using LLM
            generated_code = await self.model.generate_code(prompt)

            # Clean up the code (remove markdown code blocks if present)
            generated_code = self._clean_code(generated_code)

            # Validate the generated code
            validation = self._validate_code(generated_code)

            if not validation["valid"]:
                logger.error(
                    "Generated code validation failed",
                    tool_name=tool_name,
                    error=validation["error"],
                )
                return {
                    "success": False,
                    "error": validation["error"],
                    "code": generated_code,
                }

            # Save the tool
            tool_path = self.tools_dir / f"{tool_name}.py"
            tool_path.write_text(generated_code, encoding="utf-8")

            logger.info(
                "Tool generated successfully",
                tool_name=tool_name,
                path=str(tool_path),
            )

            # Create tool definition
            tool_def = ToolDefinition(
                name=tool_name,
                description=description,
                code=generated_code,
                test_cases=test_cases or [],
                validated=True,
                category="generated",
            )

            return {
                "success": True,
                "tool_name": tool_name,
                "tool_path": str(tool_path),
                "code": generated_code,
                "validation": validation,
                "tool_definition": tool_def,
            }

        except Exception as e:
            logger.error("Tool generation failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    async def improve_tool(
        self,
        tool_name: str,
        improvement_request: str,
        current_code: str | None = None,
    ) -> dict[str, Any]:
        """
        Improve an existing tool.

        Args:
            tool_name: Name of the tool to improve
            improvement_request: What to improve
            current_code: Current code (will load from file if not provided)

        Returns:
            Improvement results
        """
        # Load current code if not provided
        if current_code is None:
            tool_path = self.tools_dir / f"{tool_name}.py"
            if not tool_path.exists():
                return {
                    "success": False,
                    "error": f"Tool {tool_name} not found",
                }
            current_code = tool_path.read_text(encoding="utf-8")

        # Build improvement prompt
        prompt = f"""Improve this Python tool based on the request.

IMPROVEMENT REQUEST: {improvement_request}

CURRENT CODE:
```python
{current_code}
```

Requirements:
1. Maintain the same function signature
2. Keep error handling
3. Add improvements as requested
4. Include docstrings
5. Return complete working code

Provide ONLY the complete improved code, no explanations."""

        logger.info(
            "Improving tool",
            tool_name=tool_name,
            request=improvement_request[:100],
        )

        try:
            # Generate improved code
            improved_code = await self.model.generate_code(prompt)
            improved_code = self._clean_code(improved_code)

            # Validate
            validation = self._validate_code(improved_code)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                }

            # Backup old version
            tool_path = self.tools_dir / f"{tool_name}.py"
            backup_path = tool_path.with_suffix(".py.bak")
            backup_path.write_text(current_code, encoding="utf-8")

            # Write improved version
            tool_path.write_text(improved_code, encoding="utf-8")

            logger.info("Tool improved successfully", tool_name=tool_name)

            return {
                "success": True,
                "tool_name": tool_name,
                "backup_path": str(backup_path),
                "code": improved_code,
            }

        except Exception as e:
            logger.error("Tool improvement failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    def _build_generation_prompt(
        self,
        description: str,
        test_cases: list[str] | None = None,
    ) -> str:
        """Build the prompt for code generation."""
        prompt = f"""You are a Python expert. Create a complete, working Python tool based on this description:

DESCRIPTION: {description}

Requirements:
1. Create a complete Python script with proper imports
2. Include an async main function that can be called
3. Add error handling with try/except
4. Include detailed docstrings
5. Make it runnable as a script (if __name__ == "__main__")
6. Use type hints where appropriate
7. Return results as a dictionary with "success" key
8. Use only safe, standard libraries
9. No dangerous operations (exec, eval, system calls)

The tool should be saved in: tools/generated/

Provide ONLY the Python code, no explanation."""

        if test_cases:
            prompt += f"\n\nTest cases it should handle:\n" + "\n".join(
                f"- {tc}" for tc in test_cases
            )

        return prompt

    def _clean_code(self, code: str) -> str:
        """Clean up generated code."""
        # Remove markdown code blocks
        code = re.sub(r"```python\n", "", code)
        code = re.sub(r"```\n", "", code)
        code = re.sub(r"```", "", code)

        # Strip whitespace
        code = code.strip()

        return code

    def _validate_code(self, code: str) -> dict[str, Any]:
        """
        Validate generated Python code.

        Checks:
        - Syntax validity
        - Dangerous patterns
        - AST structure
        """
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in code.lower():
                return {
                    "valid": False,
                    "error": f"Dangerous pattern detected: {pattern}",
                }

        # Check for imports of dangerous modules
        dangerous_modules = ["subprocess", "os.system", "eval", "exec"]
        for module in dangerous_modules:
            if f"import {module}" in code or f"from {module}" in code:
                # Allow subprocess in specific safe contexts if needed
                pass

        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Syntax error: {e}",
            }

        # Check AST for dangerous nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check function calls
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["exec", "eval", "compile"]:
                        return {
                            "valid": False,
                            "error": f"Dangerous function call: {node.func.id}",
                        }

            # Check for __import__
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                    return {
                        "valid": False,
                        "error": "Dangerous: __import__ detected",
                    }

        return {
            "valid": True,
            "error": None,
        }

    def list_generated_tools(self) -> list[str]:
        """List all generated tools."""
        tools = []
        if self.tools_dir.exists():
            for f in self.tools_dir.glob("*.py"):
                if f.stem != "__init__":
                    tools.append(f.stem)
        return tools

    def get_tool_code(self, tool_name: str) -> str | None:
        """Get the code for a generated tool."""
        tool_path = self.tools_dir / f"{tool_name}.py"
        if tool_path.exists():
            return tool_path.read_text(encoding="utf-8")
        return None
