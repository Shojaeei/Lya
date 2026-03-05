"""Tool-calling middleware for prompt-based function calling.

Bridges Ollama (no native tool calling) and the existing ToolRegistry
by injecting tool descriptions into the system prompt and parsing
structured tool_call blocks from LLM output.

Supports multiple tool calls per LLM turn for efficient multi-file operations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.tools.tool_registry import ToolRegistry

logger = get_logger(__name__)

# Matches one complete tool_call block
TOOL_CALL_PATTERN = re.compile(
    r"```tool_call\s*\n(.*?)\n\s*```",
    re.DOTALL,
)

# Matches ALL complete tool_call blocks (for multi-tool extraction)
TOOL_CALL_ALL_PATTERN = re.compile(
    r"```tool_call\s*\n(.*?)\n\s*```",
    re.DOTALL,
)

# Detects a tool_call block that was truncated (no closing ```)
TRUNCATED_TOOL_CALL_PATTERN = re.compile(
    r"```tool_call\s*\n.*",
    re.DOTALL,
)

MAX_TOOL_ITERATIONS = 15
MAX_TOOL_RESULT_LENGTH = 3000


@dataclass
class ToolCallResult:
    tool_name: str
    parameters: dict[str, Any]
    result: dict[str, Any]
    success: bool
    error: str | None = None


@dataclass
class ToolCallingContext:
    iterations: int = 0
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    final_response: str | None = None


def build_tool_descriptions(registry: ToolRegistry) -> str:
    tools = registry.list_tools()
    if not tools:
        return ""

    lines = ["## Available Tools\n"]
    for tool in tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "")
        params = tool.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        # Compact signature: tool_name(param1*, param2) — description
        sig_parts = []
        for pname, pinfo in properties.items():
            req_mark = "*" if pname in required else ""
            sig_parts.append(f"{pname}{req_mark}")
        sig = ", ".join(sig_parts) if sig_parts else ""
        lines.append(f"- `{name}({sig})` — {desc}")

    return "\n".join(lines)


TOOL_CALLING_INSTRUCTIONS = """
## Tool Usage

When you need real-time data, file access, or to run a command, use tools.
Output one or more tool_call blocks per message:

```tool_call
{"name": "<tool_name>", "parameters": {<parameters_object>}}
```

Rules:
- You can output MULTIPLE tool_call blocks in a single message for parallel operations (e.g. writing several files at once).
- Each tool_call block must be a separate fenced block with valid JSON.
- The JSON must be valid. "name" is the tool name, "parameters" is an object.
- You may include brief text BEFORE and BETWEEN tool_call blocks.
- After receiving tool results, either call more tools or give your final answer as plain text.
- When you do NOT need a tool, just respond with plain text (no tool_call block).
- NEVER fabricate tool results. Always call the tool to get real data.
- For large projects, write ALL files in one message using multiple tool_call blocks.

CRITICAL — Writing Code:
- NEVER paste code into your response. Always use the `file_write` tool to write code to a file.
- When the user asks you to write code, create an app, build something, etc., use `file_write` for EVERY file.
- Your text response should only contain brief explanations, NOT code. The code goes in file_write calls.

CRITICAL — Be Autonomous:
- You are an autonomous agent. DO things, don't instruct the user how to do things.
- NEVER say "run this command" or "go to this URL". Use your tools to do it yourself.
- For downloads: call `download_video` or `download_file`, then call `upload_file` to send the result.
- For scheduling: call `schedule_task`. Do NOT write scheduler/cron code.
- For file operations: use `file_write`, `file_edit`, `file_delete`, `file_move`, `file_copy`.
- Always ACT using tools. Never just DESCRIBE what to do.
""".strip()


def build_system_prompt_with_tools(
    base_system_prompt: str,
    registry: ToolRegistry,
) -> str:
    tool_descriptions = build_tool_descriptions(registry)
    if not tool_descriptions:
        return base_system_prompt

    return (
        f"{base_system_prompt}\n\n"
        f"{tool_descriptions}\n"
        f"{TOOL_CALLING_INSTRUCTIONS}"
    )


def parse_tool_call(llm_output: str) -> dict[str, Any] | None:
    """Parse the first tool_call block from LLM output."""
    match = TOOL_CALL_PATTERN.search(llm_output)
    if not match:
        return None

    raw_json = match.group(1).strip()
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logger.warning("tool_call_json_parse_error", raw=raw_json[:200], error=str(e))
        return None

    name = parsed.get("name")
    parameters = parsed.get("parameters", {})

    if not name or not isinstance(name, str):
        return None
    if not isinstance(parameters, dict):
        return None

    return {"name": name, "parameters": parameters}


def _parse_multi_json(raw: str) -> list[dict[str, Any]]:
    """Parse one or more JSON objects from a raw string.

    Handles both single JSON and multiple JSON objects separated by newlines
    (the LLM sometimes puts multiple tool calls in one fenced block).
    """
    raw = raw.strip()
    objects = []

    # Try single JSON first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    # Try splitting by lines — each line might be a JSON object
    for line in raw.split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            continue

    # If line-split didn't work, try finding JSON objects by brace matching
    if not objects:
        depth = 0
        start = None
        for i, ch in enumerate(raw):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(raw[start : i + 1])
                        if isinstance(obj, dict):
                            objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None

    return objects


def parse_all_tool_calls(llm_output: str) -> list[dict[str, Any]]:
    """Parse ALL tool_call blocks from LLM output (supports multiple per message).

    Handles:
    - Multiple separate ```tool_call blocks
    - Multiple JSON objects inside a single block (LLM puts them on separate lines)
    """
    matches = TOOL_CALL_ALL_PATTERN.findall(llm_output)
    if not matches:
        return []

    results = []
    for raw_json in matches:
        parsed_objects = _parse_multi_json(raw_json)
        if not parsed_objects:
            logger.warning("tool_call_json_parse_error", raw=raw_json[:200], error="no valid JSON found")
            continue

        for parsed in parsed_objects:
            name = parsed.get("name")
            parameters = parsed.get("parameters", {})

            if not name or not isinstance(name, str):
                continue
            if not isinstance(parameters, dict):
                continue

            results.append({"name": name, "parameters": parameters})

    return results


def clean_response(text: str) -> str:
    """Strip any leaked tool_call blocks from the final user-facing response."""
    # Remove complete tool_call blocks
    text = TOOL_CALL_PATTERN.sub("", text)
    # Remove truncated tool_call blocks (opening ``` without closing)
    text = TRUNCATED_TOOL_CALL_PATTERN.sub("", text)
    return text.strip()


def is_truncated_tool_call(text: str) -> bool:
    """Check if the output contains a tool_call block that was cut off (no closing ```)."""
    if "```tool_call" not in text:
        return False
    # Has opening but no proper closing
    return TOOL_CALL_PATTERN.search(text) is None


def truncate_result(result: dict[str, Any], max_length: int = MAX_TOOL_RESULT_LENGTH) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) > max_length:
        text = text[:max_length] + "... (truncated)"
    return text


async def execute_tool_call(
    tool_call: dict[str, Any],
    registry: ToolRegistry,
) -> ToolCallResult:
    name = tool_call["name"]
    parameters = tool_call["parameters"]

    logger.info("executing_tool_call", tool=name, params=list(parameters.keys()))

    if not registry.is_tool_available(name):
        return ToolCallResult(
            tool_name=name, parameters=parameters,
            result={"success": False, "error": f"Unknown tool: {name}"},
            success=False, error=f"Unknown tool: {name}",
        )

    try:
        result = await registry.execute(name, parameters)
        success = result.get("success", True)
        return ToolCallResult(
            tool_name=name, parameters=parameters,
            result=result, success=success,
            error=result.get("error") if not success else None,
        )
    except Exception as e:
        logger.error("tool_execution_error", tool=name, error=str(e))
        return ToolCallResult(
            tool_name=name, parameters=parameters,
            result={"success": False, "error": str(e)},
            success=False, error=str(e),
        )


async def run_tool_calling_loop(
    llm,
    messages: list[dict[str, str]],
    registry: ToolRegistry,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    temperature: float = 0.7,
    max_tokens: int = 16384,
    images: list[str] | None = None,
) -> tuple[str, ToolCallingContext]:
    ctx = ToolCallingContext()
    working_messages = list(messages)

    for iteration in range(max_iterations):
        ctx.iterations = iteration + 1

        # Only pass images on the first iteration
        chat_kwargs: dict[str, Any] = {
            "messages": working_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if images and iteration == 0:
            chat_kwargs["images"] = images

        llm_output = await llm.chat(**chat_kwargs)
        llm_output = llm_output.strip()

        logger.debug(
            "tool_loop_iteration",
            iteration=iteration + 1,
            length=len(llm_output),
            has_tool_call="tool_call" in llm_output,
        )

        # ── Parse ALL tool calls from this turn ──
        tool_calls = parse_all_tool_calls(llm_output)

        if not tool_calls:
            # Check if the tool_call was truncated due to max_tokens
            if is_truncated_tool_call(llm_output):
                logger.warning(
                    "tool_call_truncated",
                    iteration=iteration + 1,
                    output_length=len(llm_output),
                )
                # Ask the LLM to retry with a simpler approach
                working_messages.append({"role": "assistant", "content": llm_output})
                working_messages.append({
                    "role": "user",
                    "content": (
                        "[System] Your tool_call was cut off because the content was too long. "
                        "Break the task into smaller steps — write one file at a time with shorter content, "
                        "or split large files into multiple writes. Try again."
                    ),
                })
                continue

            # No tool call — this is the final response. Clean any remnants.
            cleaned = clean_response(llm_output)
            ctx.final_response = cleaned
            return cleaned, ctx

        # ── Execute ALL tool calls from this turn ──
        logger.info(
            "executing_batch_tool_calls",
            count=len(tool_calls),
            tools=[tc["name"] for tc in tool_calls],
        )

        results_summary = []
        for tc in tool_calls:
            tool_result = await execute_tool_call(tc, registry)
            ctx.tool_calls.append(tool_result)

            result_text = truncate_result(tool_result.result)
            status = "OK" if tool_result.success else "FAILED"
            results_summary.append(
                f"[{tc['name']}] {status}: {result_text}"
            )

        working_messages.append({"role": "assistant", "content": llm_output})
        working_messages.append({
            "role": "user",
            "content": (
                f"[Tool Results — {len(tool_calls)} tool(s) executed]\n"
                + "\n".join(results_summary)
                + "\n\nContinue with more tool_call blocks if needed, "
                "or provide your final answer as plain text."
            ),
        })

    # Max iterations reached — force a final answer
    logger.warning("tool_loop_max_iterations", iterations=max_iterations)
    working_messages.append({
        "role": "user",
        "content": (
            "Maximum tool calls reached. Provide your best answer now "
            "based on what you have. Do NOT use any more tool_call blocks."
        ),
    })

    final_output = await llm.chat(
        messages=working_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    cleaned = clean_response(final_output)
    ctx.final_response = cleaned
    ctx.iterations = max_iterations + 1
    return cleaned, ctx
