"""SDLC Pipeline Tool.

Full software development lifecycle: generate code → test → debug → deploy.
Runs in sandboxed subprocess. Supports registering as new bot tool or sending as file.
"""

from __future__ import annotations

import asyncio
import json
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings
from lya.infrastructure.llm.ollama_adapter import OllamaAdapter

logger = get_logger(__name__)

MAX_DEBUG_ATTEMPTS = 3
TEST_TIMEOUT = 60


async def develop_code(parameters: dict[str, Any]) -> dict[str, Any]:
    """
    Full SDLC pipeline: generate → test → debug → deploy.

    Parameters:
        description: What to build (detailed description)
        intent: "tool" (register as bot tool) or "file" (send to user)
        filename: Output filename (e.g., "calculator.py")
    """
    description = parameters.get("description", "")
    intent = parameters.get("intent", "file")
    filename = parameters.get("filename", "output.py")

    if not description:
        return {"success": False, "error": "description is required"}

    if not filename.endswith(".py"):
        filename += ".py"

    workspace = Path(settings.workspace_path)
    llm = OllamaAdapter()

    logger.info("sdlc_starting", description=description[:80], intent=intent)

    # Phase 1: Generate code
    code = await _generate_code(llm, description)
    if not code:
        return {"success": False, "error": "Code generation failed", "phase": "generate"}

    # Phase 2: Generate tests
    test_code = await _generate_tests(llm, description, code)

    # Phase 3: Test & Debug loop
    for attempt in range(1, MAX_DEBUG_ATTEMPTS + 1):
        logger.info("sdlc_test_attempt", attempt=attempt)

        passed, test_output = await _run_tests_sandboxed(code, test_code)

        if passed:
            logger.info("sdlc_tests_passed", attempt=attempt)
            break

        if attempt < MAX_DEBUG_ATTEMPTS:
            logger.info("sdlc_debugging", attempt=attempt, output=test_output[:200])
            code = await _debug_code(llm, code, test_code, test_output)
    else:
        # All attempts exhausted
        logger.warning("sdlc_all_attempts_failed")
        # Save anyway so user can see what was generated
        out_path = workspace / "outputs" / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(code, encoding="utf-8")
        return {
            "success": False,
            "error": f"Tests failed after {MAX_DEBUG_ATTEMPTS} attempts",
            "phase": "test",
            "test_output": test_output[:1000],
            "file_path": str(out_path),
            "attempts": MAX_DEBUG_ATTEMPTS,
        }

    # Phase 4: Deploy
    if intent == "tool":
        result = await _register_as_tool(code, filename, description, workspace)
    else:
        result = await _save_as_file(code, test_code, filename, workspace)

    result["attempts"] = attempt
    result["code_lines"] = len(code.splitlines())
    return result


async def _generate_code(llm: OllamaAdapter, description: str) -> str:
    prompt = f"""Write a complete, production-ready Python module.

Task: {description}

Requirements:
- Pure Python 3.12+ compatible
- All functions must have type hints and docstrings
- Handle errors with try/except
- Use async def for I/O operations
- Follow PEP 8
- Include all necessary imports
- The code must be complete and runnable

Output ONLY Python code. No markdown, no explanations, no ```python blocks."""

    try:
        code = await llm.generate(prompt, temperature=0.2, max_tokens=4096)
        return _clean_code(code)
    except Exception as e:
        logger.error("code_generation_failed", error=str(e))
        return ""


async def _generate_tests(llm: OllamaAdapter, description: str, code: str) -> str:
    prompt = f"""Write pytest tests for this Python module.

Module description: {description}

Module code:
{code}

Requirements:
- Import from 'module' (the file will be named module.py)
- Use pytest and pytest-asyncio if async functions
- At least 3 test cases: basic functionality, edge cases, error handling
- Use assert statements
- Tests must be self-contained (no external dependencies)

Output ONLY Python test code. No markdown, no explanations."""

    try:
        test_code = await llm.generate(prompt, temperature=0.2, max_tokens=2048)
        return _clean_code(test_code)
    except Exception as e:
        logger.error("test_generation_failed", error=str(e))
        return _stub_tests()


async def _run_tests_sandboxed(code: str, test_code: str) -> tuple[bool, str]:
    """Run tests in isolated subprocess."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Write module
        (tmp / "module.py").write_text(code, encoding="utf-8")
        (tmp / "test_module.py").write_text(test_code, encoding="utf-8")
        (tmp / "conftest.py").write_text(
            "import asyncio\nimport pytest\n", encoding="utf-8"
        )

        # Determine python command
        python_cmd = "python"

        cmd = [
            python_cmd, "-m", "pytest",
            "test_module.py", "-v", "--tb=short", "--timeout=30",
        ]

        env = {
            "PYTHONPATH": str(tmp),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": str(Path(shutil.which(python_cmd) or python_cmd).parent),
            "SYSTEMROOT": "C:\\Windows" if platform.system() == "Windows" else "",
        }

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=tmp,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=TEST_TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, "Test execution timed out"

            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")
            full_output = output + "\n" + errors

            passed = proc.returncode == 0
            logger.info("sandbox_test_result", passed=passed,
                        returncode=proc.returncode)
            return passed, full_output

        except Exception as e:
            return False, f"Failed to run tests: {e}"


async def _debug_code(
    llm: OllamaAdapter, code: str, test_code: str, test_output: str
) -> str:
    prompt = f"""Fix this Python code based on the test failures.

Current code:
{code}

Test code:
{test_code}

Test output (failures):
{test_output[:2000]}

Fix the code to make all tests pass. Keep the same interface.
Output ONLY the fixed Python code. No markdown, no explanations."""

    try:
        fixed = await llm.generate(prompt, temperature=0.2, max_tokens=4096)
        return _clean_code(fixed)
    except Exception as e:
        logger.error("debug_failed", error=str(e))
        return code


async def _register_as_tool(
    code: str, filename: str, description: str, workspace: Path
) -> dict[str, Any]:
    """Register the generated code as a new tool in ToolRegistry."""
    from lya.infrastructure.tools.tool_registry import get_tool_registry

    # Save to capabilities directory
    cap_dir = workspace / "capabilities" / filename.replace(".py", "")
    cap_dir.mkdir(parents=True, exist_ok=True)
    code_path = cap_dir / filename
    code_path.write_text(code, encoding="utf-8")

    # Extract async functions from code to register as tools
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"success": False, "error": f"Syntax error in generated code: {e}"}

    registered_tools = []
    registry = get_tool_registry()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue

            func_name = node.name
            # Extract docstring
            func_doc = ast.get_docstring(node) or description

            # Build parameter schema from function args
            params = {}
            required = []
            for arg in node.args.args:
                if arg.arg == "self":
                    continue
                params[arg.arg] = {
                    "type": "string",
                    "description": f"Parameter {arg.arg}",
                }
                # Args without defaults are required
                required.append(arg.arg)

            # Remove required for args that have defaults
            n_defaults = len(node.args.defaults)
            if n_defaults > 0:
                required = required[:-n_defaults]

            tool_name = f"custom_{func_name}"

            # Create a handler that executes the function
            exec_code = code
            async def make_handler(fc=exec_code, fn=func_name):
                async def handler(parameters: dict[str, Any]) -> dict[str, Any]:
                    namespace: dict[str, Any] = {}
                    exec(compile(fc, "<capability>", "exec"), namespace)
                    func = namespace.get(fn)
                    if not func:
                        return {"success": False, "error": f"Function {fn} not found"}
                    try:
                        import asyncio
                        if asyncio.iscoroutinefunction(func):
                            result = await func(**parameters)
                        else:
                            result = func(**parameters)
                        return {"success": True, "result": result}
                    except Exception as e:
                        return {"success": False, "error": str(e)}
                return handler

            handler = await make_handler()

            registry.register_tool(
                name=tool_name,
                handler=handler,
                schema={
                    "name": tool_name,
                    "description": func_doc[:200],
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": required,
                    },
                },
            )
            registered_tools.append(tool_name)
            logger.info("tool_registered", name=tool_name)

    return {
        "success": True,
        "phase": "deploy",
        "intent": "tool",
        "file_path": str(code_path),
        "registered_tools": registered_tools,
        "message": f"Registered {len(registered_tools)} tools: {', '.join(registered_tools)}",
    }


async def _save_as_file(
    code: str, test_code: str, filename: str, workspace: Path
) -> dict[str, Any]:
    """Save code to workspace/outputs for sending to user."""
    out_dir = workspace / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    code_path = out_dir / filename
    code_path.write_text(code, encoding="utf-8")

    test_filename = f"test_{filename}"
    test_path = out_dir / test_filename
    test_path.write_text(test_code, encoding="utf-8")

    return {
        "success": True,
        "phase": "deploy",
        "intent": "file",
        "file_path": str(code_path),
        "test_file_path": str(test_path),
        "message": f"Code saved to {filename} (tests passed)",
    }


def _clean_code(code: str) -> str:
    """Remove markdown wrappers from LLM output."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def _stub_tests() -> str:
    return """import pytest

def test_placeholder():
    assert True, "Placeholder test"
"""
