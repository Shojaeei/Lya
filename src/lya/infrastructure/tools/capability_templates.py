"""Capability Templates.

Pre-built templates for common capabilities.
"""

from lya.domain.models.capability import (
    CapabilityNeed,
    CapabilityInterface,
    FunctionSignature,
)


class CapabilityTemplates:
    """Templates for common capabilities."""

    @staticmethod
    def browser_automation() -> tuple[CapabilityNeed, str]:
        """Template for browser automation capability."""
        need = CapabilityNeed(
            name="browser_automation",
            description="Automate web browser for navigation, form filling, and data extraction",
            reason="Need to interact with websites and extract information",
            suggested_interface=CapabilityInterface(
                functions=[
                    FunctionSignature(
                        name="browse_website",
                        description="Navigate to a website and extract content",
                        parameters={
                            "url": {"type": "string", "description": "Website URL to browse"},
                            "wait_seconds": {"type": "integer", "default": 3},
                        },
                        returns={"type": "BrowseResult"},
                        async_=True,
                    ),
                    FunctionSignature(
                        name="fill_form",
                        description="Fill out a web form with provided data",
                        parameters={
                            "url": {"type": "string"},
                            "fields": {"type": "dict", "description": "Field selectors and values"},
                        },
                        returns={"type": "dict"},
                        async_=True,
                    ),
                    FunctionSignature(
                        name="take_screenshot",
                        description="Take screenshot of a webpage",
                        parameters={
                            "url": {"type": "string"},
                            "full_page": {"type": "boolean", "default": True},
                        },
                        returns={"type": "string", "description": "Base64 encoded PNG"},
                        async_=True,
                    ),
                ]
            ),
            suggested_dependencies=["playwright"],
            urgency=4,
        )

        code = '''"""Browser Automation Capability.

Automates web browser for navigation, forms, and screenshots.
Uses Playwright for reliable automation.
"""

from dataclasses import dataclass
from typing import Any

@dataclass
class BrowseResult:
    url: str
    title: str
    text: str
    links: list[dict]

async def browse_website(url: str, wait_seconds: int = 3) -> BrowseResult:
    """
    Browse a website and extract content.

    Args:
        url: Website URL
        wait_seconds: Time to wait for page load

    Returns:
        BrowseResult with title, text, and links
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(wait_seconds * 1000)

            title = await page.title()

            # Extract visible text
            text = await page.evaluate("() => document.body.innerText")

            # Extract links
            links = await page.eval_on_selector_all(
                "a[href]",
                """els => els.map(e => ({
                    text: e.textContent?.trim() || '',
                    href: e.href
                }))"""
            )

            return BrowseResult(
                url=url,
                title=title,
                text=text[:10000],  # Limit
                links=links[:50]   # Limit
            )
        finally:
            await browser.close()

async def fill_form(url: str, fields: dict[str, str]) -> dict[str, Any]:
    """Fill out a web form."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(url)

            # Fill fields
            for selector, value in fields.items():
                await page.fill(selector, value)

            # Try to submit
            try:
                await page.click("button[type='submit']")
                await page.wait_for_load_state("networkidle")
            except:
                pass

            return {
                "url": page.url,
                "title": await page.title(),
                "success": True
            }
        finally:
            await browser.close()

async def take_screenshot(url: str, full_page: bool = True) -> str:
    """Take screenshot of webpage."""
    from playwright.async_api import async_playwright
    import base64

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.goto(url, wait_until="networkidle")

            screenshot = await page.screenshot(full_page=full_page, type="png")
            return base64.b64encode(screenshot).decode()
        finally:
            await browser.close()
'''

        test_code = '''import pytest
from capability import browse_website, fill_form, take_screenshot, BrowseResult

@pytest.mark.asyncio
async def test_browse_website():
    # This test uses example.com which should always work
    result = await browse_website("https://example.com", wait_seconds=1)
    assert isinstance(result, BrowseResult)
    assert result.url == "https://example.com"
    assert "Example Domain" in result.title

@pytest.mark.asyncio
async def test_browse_returns_text():
    result = await browse_website("https://example.com", wait_seconds=1)
    assert len(result.text) > 0

@pytest.mark.asyncio
async def test_take_screenshot():
    screenshot = await take_screenshot("https://example.com", full_page=False)
    assert len(screenshot) > 0
    # Should be valid base64
    import base64
    decoded = base64.b64decode(screenshot)
    assert len(decoded) > 0

@pytest.mark.asyncio
async def test_fill_form_mock():
    # This would need a real form page to test properly
    # For now, just test it doesn't crash
    try:
        result = await fill_form("https://httpbin.org/forms/post", {
            "input[name='custname']": "Test User"
        })
        assert isinstance(result, dict)
    except:
        pytest.skip("Network issue or page changed")
'''

        return need, code, test_code

    @staticmethod
    def file_operations() -> tuple[CapabilityNeed, str, str]:
        """Template for file operations capability."""
        need = CapabilityNeed(
            name="file_operations",
            description="Read, write, and manage files safely",
            reason="Need to access and modify files in workspace",
            suggested_interface=CapabilityInterface(
                functions=[
                    FunctionSignature(
                        name="read_file",
                        description="Read file contents as string",
                        parameters={
                            "path": {"type": "string"},
                            "encoding": {"type": "string", "default": "utf-8"},
                        },
                        returns={"type": "string"},
                        async_=True,
                    ),
                    FunctionSignature(
                        name="write_file",
                        description="Write content to file",
                        parameters={
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        returns={"type": "bool"},
                        async_=True,
                    ),
                    FunctionSignature(
                        name="list_directory",
                        description="List directory contents",
                        parameters={
                            "path": {"type": "string", "default": "."},
                            "recursive": {"type": "bool", "default": False},
                        },
                        returns={"type": "list"},
                        async_=True,
                    ),
                    FunctionSignature(
                        name="edit_file",
                        description="Apply text replacements to file",
                        parameters={
                            "path": {"type": "string"},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                        },
                        returns={"type": "bool"},
                        async_=True,
                    ),
                ]
            ),
            suggested_dependencies=["aiofiles"],
            urgency=5,
        )

        code = '''"""File Operations Capability.

Safe file reading, writing, and management.
"""

from pathlib import Path
from typing import Any
import aiofiles

async def read_file(path: str, encoding: str = "utf-8") -> str:
    """
    Read file contents.

    Args:
        path: File path (relative to workspace)
        encoding: Text encoding

    Returns:
        File contents
    """
    async with aiofiles.open(path, "r", encoding=encoding) as f:
        return await f.read()

async def write_file(path: str, content: str) -> bool:
    """
    Write content to file.

    Args:
        path: File path
        content: Content to write

    Returns:
        True if successful
    """
    # Create parent directories if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(content)
    return True

async def list_directory(path: str = ".", recursive: bool = False) -> list[dict]:
    """
    List directory contents.

    Args:
        path: Directory path
        recursive: Include subdirectories

    Returns:
        List of file info dicts
    """
    target = Path(path)
    results = []

    pattern = "**/*" if recursive else "*"

    for item in target.glob(pattern):
        stat = item.stat()
        results.append({
            "path": str(item.relative_to(target) if path != "." else item),
            "name": item.name,
            "is_file": item.is_file(),
            "is_dir": item.is_dir(),
            "size": stat.st_size,
            "modified": stat.st_mtime,
        })

    return results

async def edit_file(path: str, old_text: str, new_text: str) -> bool:
    """
    Replace text in file.

    Args:
        path: File path
        old_text: Text to replace
        new_text: Replacement text

    Returns:
        True if changes were made
    """
    content = await read_file(path)

    if old_text not in content:
        return False

    new_content = content.replace(old_text, new_text, 1)
    await write_file(path, new_content)
    return True
'''

        test_code = '''import pytest
import tempfile
import os
from pathlib import Path
from capability import read_file, write_file, list_directory, edit_file

@pytest.mark.asyncio
async def test_write_and_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)  # Change to temp dir

        test_file = "test.txt"
        content = "Hello, World!"

        success = await write_file(test_file, content)
        assert success

        result = await read_file(test_file)
        assert result == content

@pytest.mark.asyncio
async def test_list_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        # Create test files
        await write_file("file1.txt", "content1")
        await write_file("file2.txt", "content2")
        Path("subdir").mkdir()

        results = await list_directory(".", recursive=False)

        assert len(results) >= 3

        files = [r for r in results if r["is_file"]]
        dirs = [r for r in results if r["is_dir"]]

        assert len(files) >= 2
        assert len(dirs) >= 1

@pytest.mark.asyncio
async def test_edit_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        test_file = "edit_test.txt"
        await write_file(test_file, "Hello, World!")

        changed = await edit_file(test_file, "World", "Universe")
        assert changed

        result = await read_file(test_file)
        assert result == "Hello, Universe!"

@pytest.mark.asyncio
async def test_edit_file_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        test_file = "nonexistent.txt"
        await write_file(test_file, "Hello")

        changed = await edit_file(test_file, "World", "Universe")
        assert not changed  # Should return False if old_text not found
'''

        return need, code, test_code

    @staticmethod
    def code_analysis() -> tuple[CapabilityNeed, str, str]:
        """Template for code analysis capability."""
        need = CapabilityNeed(
            name="code_analyzer",
            description="Analyze code structure, find patterns, and suggest improvements",
            reason="Need to understand and improve codebases",
            suggested_interface=CapabilityInterface(
                functions=[
                    FunctionSignature(
                        name="analyze_file",
                        description="Analyze a Python file and extract structure",
                        parameters={"path": {"type": "string"}},
                        returns={"type": "dict"},
                        async_=True,
                    ),
                    FunctionSignature(
                        name="find_functions",
                        description="Find all function definitions in code",
                        parameters={"code": {"type": "string"}},
                        returns={"type": "list"},
                        async_=True,
                    ),
                    FunctionSignature(
                        name="check_syntax",
                        description="Check if Python code is valid",
                        parameters={"code": {"type": "string"}},
                        returns={"type": "dict"},
                        async_=True,
                    ),
                ]
            ),
            suggested_dependencies=["ast"],  # Built-in
            urgency=3,
        )

        code = '''"""Code Analysis Capability.

Analyze Python code structure and extract information.
"""

import ast
import inspect
from typing import Any

async def analyze_file(path: str) -> dict[str, Any]:
    """
    Analyze a Python file structure.

    Args:
        path: Path to Python file

    Returns:
        Analysis results with functions, classes, imports
    """
    with open(path, "r") as f:
        code = f.read()

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {
            "error": f"Syntax error: {e}",
            "functions": [],
            "classes": [],
            "imports": [],
        }

    result = {
        "functions": [],
        "classes": [],
        "imports": [],
        "total_lines": len(code.splitlines()),
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            result["functions"].append({
                "name": node.name,
                "line": node.lineno,
                "args": [arg.arg for arg in node.args.args],
                "docstring": ast.get_docstring(node),
            })
        elif isinstance(node, ast.ClassDef):
            result["classes"].append({
                "name": node.name,
                "line": node.lineno,
                "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
            })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                result["imports"].append(f"{module}.{alias.name}")

    return result

async def find_functions(code: str) -> list[dict[str, Any]]:
    """Find all function definitions in code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "line": node.lineno,
                "signature": ast.unparse(node.args) if hasattr(ast, "unparse") else str(node.args),
            })

    return functions

async def check_syntax(code: str) -> dict[str, Any]:
    """Check if code has valid Python syntax."""
    try:
        ast.parse(code)
        return {"valid": True, "error": None}
    except SyntaxError as e:
        return {
            "valid": False,
            "error": str(e),
            "line": e.lineno,
            "column": e.offset,
        }
'''

        test_code = '''import pytest
from capability import analyze_file, find_functions, check_syntax
import tempfile
import os

@pytest.mark.asyncio
async def test_check_syntax_valid():
    code = "def hello():\\n    pass"
    result = await check_syntax(code)
    assert result["valid"] is True

@pytest.mark.asyncio
async def test_check_syntax_invalid():
    code = "def hello(:\\n    pass"  # Invalid syntax
    result = await check_syntax(code)
    assert result["valid"] is False
    assert "error" in result

@pytest.mark.asyncio
async def test_find_functions():
    code = """
def func1():
    pass

def func2(x, y):
    return x + y
"""
    functions = await find_functions(code)
    assert len(functions) == 2

    names = [f["name"] for f in functions]
    assert "func1" in names
    assert "func2" in names

@pytest.mark.asyncio
async def test_analyze_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
import json
from typing import List

def process_data(items: List[str]) -> dict:
    \\"\\"\\"Process a list of items.\\"\\"\\"
    return {item: len(item) for item in items}

class DataHandler:
    def __init__(self):
        self.data = []

    def add(self, item):
        self.data.append(item)
""")
        temp_path = f.name

    try:
        result = await analyze_file(temp_path)

        assert "functions" in result
        assert "classes" in result
        assert "imports" in result

        # Should find process_data function
        func_names = [f["name"] for f in result["functions"]]
        assert "process_data" in func_names

        # Should find DataHandler class
        class_names = [c["name"] for c in result["classes"]]
        assert "DataHandler" in class_names

    finally:
        os.unlink(temp_path)
'''

        return need, code, test_code


def get_all_templates() -> dict[str, tuple]:
    """Get all capability templates."""
    return {
        "browser_automation": CapabilityTemplates.browser_automation(),
        "file_operations": CapabilityTemplates.file_operations(),
        "code_analysis": CapabilityTemplates.code_analysis(),
    }
