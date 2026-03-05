"""Tool Registry for Lya.

Implements the ToolPort protocol and provides a registry for basic tools.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from lya.application.ports.outgoing.tool_port import ToolPort
from lya.infrastructure.config.logging import get_logger
from pathlib import Path

from lya.infrastructure.tools.filesystem import FileManager, get_file_manager
from lya.infrastructure.tools.web_tools import WebTools, get_web_tools
from lya.infrastructure.tools.system_tools import SystemTools, get_system_tools

logger = get_logger(__name__)


class ToolRegistry(ToolPort):
    """
    Tool registry implementing ToolPort.

    Provides access to basic tools:
    - File operations (via FileManager)
    - Web/HTTP operations (via WebTools)
    - System/shell operations (via SystemTools)

    Each tool is registered with its schema for LLM function calling.
    """

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

        # Initialize tool handlers
        self._file_manager = get_file_manager()
        self._web_tools = get_web_tools()
        self._system_tools = get_system_tools()

        # Scheduler (set externally via set_scheduler)
        self._scheduler = None
        self._scheduler_chat_id: int | None = None

        # Git adapter (optional — git may not be installed)
        self._git_adapter = None
        try:
            from lya.infrastructure.tools.git import GitAdapter
            self._git_adapter = GitAdapter()
        except Exception:
            logger.warning("Git not available — git tools disabled")

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools
        self.register_tool(
            name="file_read",
            handler=self._handle_file_read,
            schema={
                "name": "file_read",
                "description": "Read contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read",
                        },
                        "encoding": {
                            "type": "string",
                            "description": "File encoding (default: utf-8)",
                            "default": "utf-8",
                        },
                    },
                    "required": ["path"],
                },
            },
        )

        self.register_tool(
            name="file_write",
            handler=self._handle_file_write,
            schema={
                "name": "file_write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file",
                        },
                        "encoding": {
                            "type": "string",
                            "description": "File encoding (default: utf-8)",
                            "default": "utf-8",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        )

        self.register_tool(
            name="file_list",
            handler=self._handle_file_list,
            schema={
                "name": "file_list",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory path (default: current directory)",
                            "default": ".",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to filter files",
                            "default": "*",
                        },
                    },
                },
            },
        )

        self.register_tool(
            name="file_exists",
            handler=self._handle_file_exists,
            schema={
                "name": "file_exists",
                "description": "Check if a file or directory exists",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to check",
                        },
                    },
                    "required": ["path"],
                },
            },
        )

        self.register_tool(
            name="file_delete",
            handler=self._handle_file_delete,
            schema={
                "name": "file_delete",
                "description": "Delete a file permanently",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to delete"},
                    },
                    "required": ["path"],
                },
            },
        )

        self.register_tool(
            name="file_move",
            handler=self._handle_file_move,
            schema={
                "name": "file_move",
                "description": "Move or rename a file or directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source path"},
                        "destination": {"type": "string", "description": "Destination path"},
                    },
                    "required": ["source", "destination"],
                },
            },
        )

        self.register_tool(
            name="file_copy",
            handler=self._handle_file_copy,
            schema={
                "name": "file_copy",
                "description": "Copy a file to a new location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source file path"},
                        "destination": {"type": "string", "description": "Destination file path"},
                    },
                    "required": ["source", "destination"],
                },
            },
        )

        self.register_tool(
            name="file_append",
            handler=self._handle_file_append,
            schema={
                "name": "file_append",
                "description": "Append content to the end of an existing file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                        "content": {"type": "string", "description": "Content to append"},
                    },
                    "required": ["path", "content"],
                },
            },
        )

        # Web tools
        self.register_tool(
            name="web_request",
            handler=self._handle_web_request,
            schema={
                "name": "web_request",
                "description": "Make an HTTP request to a URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to request",
                        },
                        "method": {
                            "type": "string",
                            "description": "HTTP method (GET, POST, PUT, DELETE)",
                            "default": "GET",
                            "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                        },
                        "headers": {
                            "type": "object",
                            "description": "HTTP headers",
                        },
                        "data": {
                            "type": "string",
                            "description": "Request body data",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Request timeout in seconds",
                            "default": 30,
                        },
                    },
                    "required": ["url"],
                },
            },
        )

        self.register_tool(
            name="web_fetch_json",
            handler=self._handle_web_fetch_json,
            schema={
                "name": "web_fetch_json",
                "description": "Fetch and parse JSON from a URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to fetch JSON from",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Request timeout in seconds",
                            "default": 30,
                        },
                    },
                    "required": ["url"],
                },
            },
        )

        self.register_tool(
            name="web_check_status",
            handler=self._handle_web_check_status,
            schema={
                "name": "web_check_status",
                "description": "Check if a URL is up and responding",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to check",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Request timeout in seconds",
                            "default": 10,
                        },
                    },
                    "required": ["url"],
                },
            },
        )

        # System tools
        self.register_tool(
            name="execute_command",
            handler=self._handle_execute_command,
            schema={
                "name": "execute_command",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory for the command",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Command timeout in seconds",
                            "default": 30,
                        },
                    },
                    "required": ["command"],
                },
            },
        )

        self.register_tool(
            name="check_command_exists",
            handler=self._handle_check_command_exists,
            schema={
                "name": "check_command_exists",
                "description": "Check if a command exists in PATH",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command name to check",
                        },
                    },
                    "required": ["command"],
                },
            },
        )

        self.register_tool(
            name="get_system_info",
            handler=self._handle_get_system_info,
            schema={
                "name": "get_system_info",
                "description": "Get system information (platform, versions, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        )

        self.register_tool(
            name="get_resource_usage",
            handler=self._handle_get_resource_usage,
            schema={
                "name": "get_resource_usage",
                "description": "Get CPU, memory, and disk usage",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        )

        # SDLC tool
        self.register_tool(
            name="develop_code",
            handler=self._handle_develop_code,
            schema={
                "name": "develop_code",
                "description": (
                    "Full SDLC pipeline: generate code, write tests, run tests "
                    "in sandbox, debug if failed, then deploy. Use intent='tool' "
                    "to register as a new bot tool, or intent='file' to send "
                    "the code to the user."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Detailed description of what to build",
                        },
                        "intent": {
                            "type": "string",
                            "description": "Deployment target: 'tool' to register as bot tool, 'file' to send to user",
                            "enum": ["tool", "file"],
                            "default": "file",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Output filename (e.g. calculator.py)",
                            "default": "output.py",
                        },
                    },
                    "required": ["description"],
                },
            },
        )

        # ── Developer tools ───────────────────────────────────────────

        self.register_tool(
            name="file_edit",
            handler=self._handle_file_edit,
            schema={
                "name": "file_edit",
                "description": "Apply search-and-replace edits to a file. Use for targeted changes instead of rewriting entire files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                        "edits": {
                            "type": "array",
                            "description": "List of edits: [{\"old\": \"text to find\", \"new\": \"replacement\"}]",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old": {"type": "string"},
                                    "new": {"type": "string"},
                                },
                                "required": ["old", "new"],
                            },
                        },
                    },
                    "required": ["path", "edits"],
                },
            },
        )

        self.register_tool(
            name="grep",
            handler=self._handle_grep,
            schema={
                "name": "grep",
                "description": "Search file contents by regex pattern. Returns matching lines with file path, line number, and context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "path": {"type": "string", "description": "File or directory to search in (default: workspace root)", "default": "."},
                        "include": {"type": "array", "description": "File extensions to include, e.g. [\".py\", \".ts\"]", "items": {"type": "string"}},
                    },
                    "required": ["pattern"],
                },
            },
        )

        self.register_tool(
            name="find_files",
            handler=self._handle_find_files,
            schema={
                "name": "find_files",
                "description": "Find files by name pattern (glob). Example: '*.py', 'test_*', 'package.json'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name_pattern": {"type": "string", "description": "Glob pattern for file names"},
                        "path": {"type": "string", "description": "Directory to search in", "default": "."},
                    },
                    "required": ["name_pattern"],
                },
            },
        )

        self.register_tool(
            name="create_directory",
            handler=self._handle_create_directory,
            schema={
                "name": "create_directory",
                "description": "Create a directory (and parent directories if needed)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path to create"},
                    },
                    "required": ["path"],
                },
            },
        )

        self.register_tool(
            name="project_tree",
            handler=self._handle_project_tree,
            schema={
                "name": "project_tree",
                "description": "Show the file tree structure of a directory. Use this to understand project layout before coding.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Root directory to scan", "default": "."},
                        "max_depth": {"type": "integer", "description": "Max depth (default: 4)", "default": 4},
                    },
                },
            },
        )

        # ── Git tools ────────────────────────────────────────────────

        if self._git_adapter:
            self.register_tool(
                name="git_init",
                handler=self._handle_git_init,
                schema={
                    "name": "git_init",
                    "description": "Initialize a new git repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory to initialize (default: workspace)"},
                        },
                    },
                },
            )

            self.register_tool(
                name="git_clone",
                handler=self._handle_git_clone,
                schema={
                    "name": "git_clone",
                    "description": "Clone a git repository from a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "Repository URL"},
                            "destination": {"type": "string", "description": "Local path (default: derived from URL)"},
                        },
                        "required": ["url"],
                    },
                },
            )

            self.register_tool(
                name="git_status",
                handler=self._handle_git_status,
                schema={
                    "name": "git_status",
                    "description": "Show git status: current branch, staged/unstaged/untracked files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {"type": "string", "description": "Repository path"},
                        },
                    },
                },
            )

            self.register_tool(
                name="git_add",
                handler=self._handle_git_add,
                schema={
                    "name": "git_add",
                    "description": "Stage files for commit. Use [\".\"] to stage all changes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "files": {"type": "array", "description": "Files to stage", "items": {"type": "string"}, "default": ["."]},
                            "repo_path": {"type": "string", "description": "Repository path"},
                        },
                    },
                },
            )

            self.register_tool(
                name="git_commit",
                handler=self._handle_git_commit,
                schema={
                    "name": "git_commit",
                    "description": "Create a git commit with a message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Commit message"},
                            "repo_path": {"type": "string", "description": "Repository path"},
                        },
                        "required": ["message"],
                    },
                },
            )

            self.register_tool(
                name="git_diff",
                handler=self._handle_git_diff,
                schema={
                    "name": "git_diff",
                    "description": "Show git diff of changes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Specific file to diff"},
                            "repo_path": {"type": "string", "description": "Repository path"},
                        },
                    },
                },
            )

            self.register_tool(
                name="git_log",
                handler=self._handle_git_log,
                schema={
                    "name": "git_log",
                    "description": "Show recent git commit history",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {"type": "integer", "description": "Number of commits (default: 10)", "default": 10},
                            "repo_path": {"type": "string", "description": "Repository path"},
                        },
                    },
                },
            )

            self.register_tool(
                name="git_create_branch",
                handler=self._handle_git_create_branch,
                schema={
                    "name": "git_create_branch",
                    "description": "Create and checkout a new git branch",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Branch name"},
                            "repo_path": {"type": "string", "description": "Repository path"},
                        },
                        "required": ["name"],
                    },
                },
            )

            self.register_tool(
                name="git_checkout",
                handler=self._handle_git_checkout,
                schema={
                    "name": "git_checkout",
                    "description": "Switch to an existing branch",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "branch": {"type": "string", "description": "Branch name to checkout"},
                            "repo_path": {"type": "string", "description": "Repository path"},
                        },
                        "required": ["branch"],
                    },
                },
            )

            self.register_tool(
                name="git_list_branches",
                handler=self._handle_git_list_branches,
                schema={
                    "name": "git_list_branches",
                    "description": "List all local (and optionally remote) branches",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "remote": {"type": "boolean", "description": "Include remote branches", "default": False},
                            "repo_path": {"type": "string", "description": "Repository path"},
                        },
                    },
                },
            )

            self.register_tool(
                name="git_pull",
                handler=self._handle_git_pull,
                schema={
                    "name": "git_pull",
                    "description": "Pull latest changes from remote repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "remote": {"type": "string", "description": "Remote name (default: origin)", "default": "origin"},
                            "branch": {"type": "string", "description": "Branch to pull (optional)"},
                            "repo_path": {"type": "string", "description": "Repository path"},
                        },
                    },
                },
            )

            self.register_tool(
                name="git_push",
                handler=self._handle_git_push,
                schema={
                    "name": "git_push",
                    "description": "Push local commits to remote repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "remote": {"type": "string", "description": "Remote name (default: origin)", "default": "origin"},
                            "branch": {"type": "string", "description": "Branch to push (optional)"},
                            "set_upstream": {"type": "boolean", "description": "Set upstream tracking", "default": False},
                            "repo_path": {"type": "string", "description": "Repository path"},
                        },
                    },
                },
            )

            self.register_tool(
                name="git_stash",
                handler=self._handle_git_stash,
                schema={
                    "name": "git_stash",
                    "description": "Stash current uncommitted changes for later",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Stash description"},
                            "pop": {"type": "boolean", "description": "If true, pop (restore) the latest stash instead of stashing", "default": False},
                            "repo_path": {"type": "string", "description": "Repository path"},
                        },
                    },
                },
            )

        # ── Download / Upload / Scrape tools ─────────────────────────

        self.register_tool(
            name="download_file",
            handler=self._handle_download_file,
            schema={
                "name": "download_file",
                "description": "Download a file from a URL and save it locally. Returns the local file path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL of the file to download"},
                        "filename": {"type": "string", "description": "Output filename (optional, auto-detected from URL)"},
                        "directory": {"type": "string", "description": "Directory to save in (default: workspace/downloads)"},
                    },
                    "required": ["url"],
                },
            },
        )

        self.register_tool(
            name="download_video",
            handler=self._handle_download_video,
            schema={
                "name": "download_video",
                "description": (
                    "Download video/audio from YouTube, Instagram, Twitter/X, TikTok, "
                    "and 1000+ other sites using yt-dlp. Supports quality selection."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Video URL (YouTube, Instagram, Twitter, TikTok, etc.)"},
                        "audio_only": {"type": "boolean", "description": "Extract audio only (mp3)", "default": False},
                        "quality": {
                            "type": "string",
                            "description": "Video quality: best, 1080p, 720p, 480p, 360p, worst",
                            "default": "best",
                        },
                        "directory": {"type": "string", "description": "Output directory (default: workspace/downloads)"},
                    },
                    "required": ["url"],
                },
            },
        )

        self.register_tool(
            name="web_scrape",
            handler=self._handle_web_scrape,
            schema={
                "name": "web_scrape",
                "description": (
                    "Scrape text content from a webpage. Extracts article text, headlines, "
                    "and links. Supports Persian/Farsi and RTL websites. "
                    "Good for news sites like IRNA, Tasnim, ISNA, Fars News, BBC Persian, etc."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL of the webpage to scrape"},
                        "selector": {
                            "type": "string",
                            "description": "CSS selector to target specific content (e.g. 'article', '.post-content', '#main')",
                        },
                        "extract": {
                            "type": "string",
                            "description": "What to extract: text, links, headlines, or all",
                            "default": "text",
                            "enum": ["text", "links", "headlines", "all"],
                        },
                    },
                    "required": ["url"],
                },
            },
        )

        self.register_tool(
            name="upload_file",
            handler=self._handle_upload_file,
            schema={
                "name": "upload_file",
                "description": (
                    "Send a local file to the user via Telegram. "
                    "Use after downloading or creating a file to deliver it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the local file to send"},
                        "caption": {"type": "string", "description": "Optional caption/description for the file"},
                    },
                    "required": ["path"],
                },
            },
        )

        logger.info(
            "Default tools registered",
            count=len(self._tools),
            tools=list(self._tools.keys()),
        )

    # ═══════════════════════════════════════════════════════════════
    # Tool Handlers
    # ═══════════════════════════════════════════════════════════════

    async def _handle_file_read(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Handle file_read tool."""
        try:
            path = parameters["path"]
            encoding = parameters.get("encoding", "utf-8")

            content = await self._file_manager.read_file(path, encoding)
            file_info = self._file_manager._get_file_info(
                self._file_manager._validate_path(path)
            )

            return {
                "success": True,
                "content": content,
                "path": str(path),
                "size": file_info.size,
            }
        except Exception as e:
            logger.error("file_read failed", error=str(e), path=parameters.get("path"))
            return {
                "success": False,
                "error": str(e),
                "path": parameters.get("path"),
            }

    async def _handle_file_write(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Handle file_write tool."""
        try:
            path = parameters["path"]
            content = parameters["content"]
            encoding = parameters.get("encoding", "utf-8")

            await self._file_manager.write_file(path, content, encoding=encoding)

            return {
                "success": True,
                "path": str(path),
                "bytes_written": len(content.encode(encoding)),
            }
        except Exception as e:
            logger.error("file_write failed", error=str(e), path=parameters.get("path"))
            return {
                "success": False,
                "error": str(e),
                "path": parameters.get("path"),
            }

    async def _handle_file_list(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Handle file_list tool."""
        try:
            directory = parameters.get("directory", ".")
            pattern = parameters.get("pattern", "*")

            files = await self._file_manager.list_directory(directory, pattern)

            return {
                "success": True,
                "directory": str(directory),
                "pattern": pattern,
                "files": [
                    {
                        "name": f.path.name,
                        "path": str(f.path),
                        "is_file": f.is_file,
                        "is_dir": f.is_dir,
                        "size": f.size,
                    }
                    for f in files
                ],
                "count": len(files),
            }
        except Exception as e:
            logger.error("file_list failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "files": [],
            }

    async def _handle_file_exists(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Handle file_exists tool."""
        try:
            from pathlib import Path
            path = Path(parameters["path"]).expanduser()
            exists = path.exists()

            return {
                "success": True,
                "path": str(path),
                "exists": exists,
                "is_file": path.is_file() if exists else False,
                "is_dir": path.is_dir() if exists else False,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": parameters.get("path"),
                "exists": False,
            }

    async def _handle_file_delete(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            path = parameters["path"]
            await self._file_manager.delete_file(path, confirm=True)
            return {"success": True, "path": path, "message": f"Deleted {path}"}
        except Exception as e:
            logger.error("file_delete failed", error=str(e))
            return {"success": False, "error": str(e), "path": parameters.get("path")}

    async def _handle_file_move(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            source = parameters["source"]
            destination = parameters["destination"]
            result = await self._file_manager.move_file(source, destination)
            return {"success": True, "source": source, "destination": str(result)}
        except Exception as e:
            logger.error("file_move failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_file_copy(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            source = parameters["source"]
            destination = parameters["destination"]
            result = await self._file_manager.copy_file(source, destination)
            return {"success": True, "source": source, "destination": str(result)}
        except Exception as e:
            logger.error("file_copy failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_file_append(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            path = parameters["path"]
            content = parameters["content"]
            await self._file_manager.append_file(path, content)
            return {"success": True, "path": path, "bytes_appended": len(content.encode("utf-8"))}
        except Exception as e:
            logger.error("file_append failed", error=str(e))
            return {"success": False, "error": str(e), "path": parameters.get("path")}

    async def _handle_web_request(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Handle web_request tool."""
        try:
            url = parameters["url"]
            method = parameters.get("method", "GET")
            headers = parameters.get("headers")
            data = parameters.get("data")
            timeout = parameters.get("timeout", 30)

            response = await self._web_tools.request(
                url=url,
                method=method,
                headers=headers,
                data=data,
                timeout=timeout,
            )

            return response.to_dict()

        except Exception as e:
            logger.error("web_request failed", error=str(e), url=parameters.get("url"))
            return {
                "success": False,
                "error": str(e),
                "url": parameters.get("url"),
            }

    async def _handle_web_fetch_json(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Handle web_fetch_json tool."""
        try:
            url = parameters["url"]
            timeout = parameters.get("timeout", 30)

            return await self._web_tools.fetch_json(url, timeout=timeout)

        except Exception as e:
            logger.error("web_fetch_json failed", error=str(e), url=parameters.get("url"))
            return {
                "success": False,
                "error": str(e),
                "url": parameters.get("url"),
            }

    async def _handle_web_check_status(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Handle web_check_status tool."""
        try:
            url = parameters["url"]
            timeout = parameters.get("timeout", 10)

            return await self._web_tools.check_status(url, timeout=timeout)

        except Exception as e:
            logger.error("web_check_status failed", error=str(e), url=parameters.get("url"))
            return {
                "success": False,
                "error": str(e),
                "url": parameters.get("url"),
                "up": False,
            }

    async def _handle_execute_command(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Handle execute_command tool."""
        try:
            command = parameters["command"]
            cwd = parameters.get("cwd")
            timeout = parameters.get("timeout", 30)

            result = await self._system_tools.execute(
                command=command,
                cwd=cwd,
                timeout=timeout,
            )

            return result.to_dict()

        except Exception as e:
            logger.error("execute_command failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "command": parameters.get("command"),
            }

    async def _handle_check_command_exists(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Handle check_command_exists tool."""
        try:
            command = parameters["command"]
            exists = self._system_tools.check_command_exists(command)

            return {
                "success": True,
                "command": command,
                "exists": exists,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": parameters.get("command"),
                "exists": False,
            }

    async def _handle_get_system_info(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Handle get_system_info tool."""
        return self._system_tools.get_system_info()

    async def _handle_get_resource_usage(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Handle get_resource_usage tool."""
        return self._system_tools.get_resource_usage()

    async def _handle_develop_code(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Handle develop_code SDLC tool."""
        try:
            from lya.infrastructure.tools.sdlc_tool import develop_code
            return await develop_code(parameters)
        except Exception as e:
            logger.error("develop_code failed", error=str(e))
            return {"success": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════
    # Developer Tool Handlers
    # ═══════════════════════════════════════════════════════════════

    async def _handle_file_edit(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await self._file_manager.edit_file(parameters["path"], parameters["edits"])
            return {"success": True, **result}
        except Exception as e:
            logger.error("file_edit failed", error=str(e))
            return {"success": False, "error": str(e), "path": parameters.get("path")}

    async def _handle_grep(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            results = await self._file_manager.grep(
                parameters["pattern"],
                parameters.get("path", "."),
                include=parameters.get("include"),
            )
            return {
                "success": True,
                "matches": results[:50],
                "total_matches": len(results),
            }
        except Exception as e:
            logger.error("grep failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_find_files(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            results = await self._file_manager.find_by_name(
                parameters["name_pattern"],
                parameters.get("path", "."),
            )
            return {
                "success": True,
                "files": [str(p) for p in results[:100]],
                "count": len(results),
            }
        except Exception as e:
            logger.error("find_files failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_create_directory(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            result_path = await self._file_manager.create_directory(parameters["path"])
            return {"success": True, "path": str(result_path)}
        except Exception as e:
            logger.error("create_directory failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_project_tree(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            path = parameters.get("path", ".")
            max_depth = parameters.get("max_depth", 4)
            target = self._file_manager._validate_path(path)

            if not target.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}

            skip_dirs = {
                "__pycache__", "node_modules", ".git", ".venv", "venv",
                ".mypy_cache", ".pytest_cache", ".tox", "dist", "build",
                ".egg-info", ".eggs", ".idea", ".vscode",
            }

            lines = []
            file_count = 0
            dir_count = 0

            def walk(dir_path: Path, prefix: str, depth: int):
                nonlocal file_count, dir_count
                if depth > max_depth:
                    return
                try:
                    entries = sorted(
                        dir_path.iterdir(),
                        key=lambda e: (not e.is_dir(), e.name.lower()),
                    )
                except PermissionError:
                    return

                filtered = [
                    e for e in entries
                    if not e.name.startswith(".")
                    and not (e.is_dir() and e.name in skip_dirs)
                ]

                for i, entry in enumerate(filtered):
                    is_last = i == len(filtered) - 1
                    connector = "└── " if is_last else "├── "
                    extension = "    " if is_last else "│   "

                    if entry.is_dir():
                        dir_count += 1
                        lines.append(f"{prefix}{connector}{entry.name}/")
                        walk(entry, prefix + extension, depth + 1)
                    else:
                        file_count += 1
                        size = entry.stat().st_size
                        size_str = (
                            f"{size}B" if size < 1024
                            else f"{size / 1024:.1f}KB" if size < 1048576
                            else f"{size / 1048576:.1f}MB"
                        )
                        lines.append(f"{prefix}{connector}{entry.name} ({size_str})")

            lines.append(f"{target.name}/")
            walk(target, "", 1)

            tree_output = "\n".join(lines[:500])
            if len(lines) > 500:
                tree_output += f"\n... ({len(lines) - 500} more entries)"

            return {
                "success": True,
                "tree": tree_output,
                "total_files": file_count,
                "total_dirs": dir_count,
            }
        except Exception as e:
            logger.error("project_tree failed", error=str(e))
            return {"success": False, "error": str(e)}

    # ── Git handlers ─────────────────────────────────────────────

    def _get_git_adapter(self, repo_path: str | None = None):
        from lya.infrastructure.tools.git import GitAdapter
        if repo_path:
            target = self._file_manager._validate_path(repo_path)
            return GitAdapter(target)
        return self._git_adapter

    async def _handle_git_init(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            path = parameters.get("path", ".")
            target = self._file_manager._validate_path(path)
            adapter = self._get_git_adapter()
            result = await adapter.init(target)
            return {"success": True, "path": str(result)}
        except Exception as e:
            logger.error("git_init failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_git_clone(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            adapter = self._get_git_adapter()
            result = await adapter.clone(
                parameters["url"],
                destination=parameters.get("destination"),
            )
            return {"success": True, "path": str(result), "url": parameters["url"]}
        except Exception as e:
            logger.error("git_clone failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_git_status(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            adapter = self._get_git_adapter(parameters.get("repo_path"))
            status = await adapter.status()
            return {"success": True, **status}
        except Exception as e:
            logger.error("git_status failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_git_add(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            adapter = self._get_git_adapter(parameters.get("repo_path"))
            files = parameters.get("files", ["."])
            await adapter.add(files)
            return {"success": True, "staged": files}
        except Exception as e:
            logger.error("git_add failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_git_commit(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            adapter = self._get_git_adapter(parameters.get("repo_path"))
            result = await adapter.commit(parameters["message"])
            return {"success": True, "hash": result, "message": parameters["message"]}
        except Exception as e:
            logger.error("git_commit failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_git_diff(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            adapter = self._get_git_adapter(parameters.get("repo_path"))
            diffs = await adapter.diff(path=parameters.get("path"))
            diff_text = "\n".join(d.patch[:1000] for d in diffs) if diffs else "(no changes)"
            return {"success": True, "diff": diff_text[:5000]}
        except Exception as e:
            logger.error("git_diff failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_git_log(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            adapter = self._get_git_adapter(parameters.get("repo_path"))
            commits = await adapter.get_commit_history(n=parameters.get("n", 10))
            return {
                "success": True,
                "commits": [
                    {"hash": c.hash[:8], "message": c.message, "author": c.author, "date": c.date}
                    for c in commits
                ],
            }
        except Exception as e:
            logger.error("git_log failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_git_create_branch(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            adapter = self._get_git_adapter(parameters.get("repo_path"))
            result = await adapter.create_branch(parameters["name"])
            return {"success": True, "branch": result}
        except Exception as e:
            logger.error("git_create_branch failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_git_checkout(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            adapter = self._get_git_adapter(parameters.get("repo_path"))
            await adapter.checkout(parameters["branch"])
            return {"success": True, "branch": parameters["branch"]}
        except Exception as e:
            logger.error("git_checkout failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_git_list_branches(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            adapter = self._get_git_adapter(parameters.get("repo_path"))
            branches = await adapter.list_branches(remote=parameters.get("remote", False))
            return {"success": True, "branches": branches, "count": len(branches)}
        except Exception as e:
            logger.error("git_list_branches failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_git_pull(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            adapter = self._get_git_adapter(parameters.get("repo_path"))
            result = await adapter.pull(
                remote=parameters.get("remote", "origin"),
                branch=parameters.get("branch"),
            )
            return {"success": True, "output": result}
        except Exception as e:
            logger.error("git_pull failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_git_push(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            adapter = self._get_git_adapter(parameters.get("repo_path"))
            result = await adapter.push(
                remote=parameters.get("remote", "origin"),
                branch=parameters.get("branch"),
                set_upstream=parameters.get("set_upstream", False),
            )
            return {"success": True, "output": result}
        except Exception as e:
            logger.error("git_push failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_git_stash(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            adapter = self._get_git_adapter(parameters.get("repo_path"))
            if parameters.get("pop", False):
                result = await adapter.stash_pop()
                return {"success": True, "action": "pop", "output": result}
            else:
                result = await adapter.stash(message=parameters.get("message"))
                return {"success": True, "action": "stash", "output": result}
        except Exception as e:
            logger.error("git_stash failed", error=str(e))
            return {"success": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════
    # Download / Upload / Scrape Handlers
    # ═══════════════════════════════════════════════════════════════

    def _get_downloads_dir(self, directory: str | None = None) -> Path:
        """Get or create downloads directory."""
        if directory:
            target = self._file_manager._validate_path(directory)
        else:
            target = self._file_manager._validate_path("downloads")
        target.mkdir(parents=True, exist_ok=True)
        return target

    async def _handle_download_file(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Download a file from URL."""
        try:
            import httpx
            from urllib.parse import urlparse, unquote

            url = parameters["url"]
            directory = self._get_downloads_dir(parameters.get("directory"))

            # Determine filename
            filename = parameters.get("filename")
            if not filename:
                parsed = urlparse(url)
                filename = unquote(parsed.path.split("/")[-1]) or "downloaded_file"

            filepath = directory / filename

            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=120.0,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                filepath.write_bytes(response.content)

            size = filepath.stat().st_size
            size_str = (
                f"{size}B" if size < 1024
                else f"{size / 1024:.1f}KB" if size < 1048576
                else f"{size / 1048576:.1f}MB"
            )

            return {
                "success": True,
                "path": str(filepath),
                "filename": filename,
                "size": size_str,
                "size_bytes": size,
            }
        except Exception as e:
            logger.error("download_file failed", error=str(e), url=parameters.get("url"))
            return {"success": False, "error": str(e), "url": parameters.get("url")}

    async def _handle_download_video(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Download video/audio from YouTube and other platforms via yt-dlp."""
        try:
            import asyncio

            url = parameters["url"]
            audio_only = parameters.get("audio_only", False)
            quality = parameters.get("quality", "best")
            directory = self._get_downloads_dir(parameters.get("directory"))

            # Find ffmpeg location
            import shutil
            ffmpeg_loc = shutil.which("ffmpeg")
            if not ffmpeg_loc:
                # Check common Windows install paths
                for candidate in [
                    Path.home() / "AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe",
                    Path("C:/ProgramData/chocolatey/bin/ffmpeg.exe"),
                ]:
                    if candidate.exists():
                        ffmpeg_loc = str(candidate.parent)
                        break
                # Search WinGet packages
                if not ffmpeg_loc:
                    winget_dir = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
                    if winget_dir.exists():
                        for p in winget_dir.rglob("ffmpeg.exe"):
                            ffmpeg_loc = str(p.parent)
                            break

            # Build yt-dlp options
            ydl_opts = {
                "outtmpl": str(directory / "%(title)s.%(ext)s"),
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "socket_timeout": 60,
            }
            if ffmpeg_loc:
                ydl_opts["ffmpeg_location"] = ffmpeg_loc

            if audio_only:
                ydl_opts["format"] = "bestaudio/best"
                ydl_opts["postprocessors"] = [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }]
            else:
                # Normalize quality: '720' -> '720p', '480' -> '480p', etc.
                q = quality.strip().rstrip("p")
                if q.isdigit():
                    quality = q + "p"
                quality_map = {
                    "best": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "worst": "worst",
                    "1080p": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best",
                    "720p": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best",
                    "480p": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]/best",
                    "360p": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360]/best",
                }
                ydl_opts["format"] = quality_map.get(quality, quality_map["best"])

            # Run yt-dlp in executor to avoid blocking
            def _download():
                import yt_dlp
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    return info

            info = await asyncio.get_event_loop().run_in_executor(None, _download)

            # Find the downloaded file
            title = info.get("title", "video")
            ext = "mp3" if audio_only else info.get("ext", "mp4")
            downloaded_path = directory / f"{title}.{ext}"

            # yt-dlp may sanitize the filename, so search for it
            if not downloaded_path.exists():
                # Find most recently created file in downloads dir
                candidates = sorted(directory.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                downloaded_path = candidates[0] if candidates else downloaded_path

            size = downloaded_path.stat().st_size if downloaded_path.exists() else 0
            size_str = (
                f"{size}B" if size < 1024
                else f"{size / 1024:.1f}KB" if size < 1048576
                else f"{size / 1048576:.1f}MB"
            )

            return {
                "success": True,
                "path": str(downloaded_path),
                "title": title,
                "duration": info.get("duration"),
                "uploader": info.get("uploader"),
                "size": size_str,
                "format": ext,
            }
        except Exception as e:
            logger.error("download_video failed", error=str(e), url=parameters.get("url"))
            return {"success": False, "error": str(e), "url": parameters.get("url")}

    async def _handle_web_scrape(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Scrape text content from a webpage with Persian/Farsi support."""
        try:
            import httpx

            url = parameters["url"]
            selector = parameters.get("selector")
            extract = parameters.get("extract", "text")

            # Fetch with browser-like headers for compatibility with Persian news sites
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=30.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "fa-IR,fa;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Accept-Encoding": "gzip, deflate",
                },
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Detect encoding — many Persian sites use windows-1256 or utf-8
                content_type = response.headers.get("content-type", "")
                if "charset=" in content_type:
                    encoding = content_type.split("charset=")[-1].strip().split(";")[0]
                else:
                    encoding = response.encoding or "utf-8"

                html = response.content.decode(encoding, errors="replace")

            try:
                from bs4 import BeautifulSoup
            except ImportError:
                return {"success": False, "error": "beautifulsoup4 not installed. Run: pip install beautifulsoup4"}

            soup = BeautifulSoup(html, "html.parser")

            # Remove script, style, nav, footer, header tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()

            # Apply CSS selector if provided
            if selector:
                target = soup.select_one(selector)
                if not target:
                    # Try broader match
                    targets = soup.select(selector)
                    if targets:
                        target = targets[0]
                    else:
                        return {
                            "success": False,
                            "error": f"CSS selector '{selector}' not found on page",
                            "url": url,
                        }
                soup = target

            result: dict[str, Any] = {"success": True, "url": url}

            if extract in ("text", "all"):
                text = soup.get_text(separator="\n", strip=True)
                # Clean up excessive whitespace
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                result["text"] = "\n".join(lines)[:8000]
                result["text_length"] = len(result["text"])

            if extract in ("links", "all"):
                links = []
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    link_text = a.get_text(strip=True)
                    if href and not href.startswith(("#", "javascript:")):
                        # Make absolute URLs
                        if href.startswith("/"):
                            from urllib.parse import urljoin
                            href = urljoin(url, href)
                        links.append({"text": link_text[:100], "url": href})
                result["links"] = links[:50]
                result["link_count"] = len(links)

            if extract in ("headlines", "all"):
                headlines = []
                for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
                    text = tag.get_text(strip=True)
                    if text:
                        headlines.append({"level": tag.name, "text": text[:200]})
                result["headlines"] = headlines[:30]

            return result

        except Exception as e:
            logger.error("web_scrape failed", error=str(e), url=parameters.get("url"))
            return {"success": False, "error": str(e), "url": parameters.get("url")}

    async def _handle_upload_file(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Mark a file for sending to the Telegram user.

        The actual sending is handled by the Telegram bot layer when it
        sees upload_file in the tool results. This handler validates the
        file exists and returns metadata.
        """
        try:
            path_str = parameters["path"]
            caption = parameters.get("caption", "")

            file_path = Path(path_str).expanduser()
            if not file_path.is_absolute():
                file_path = self._file_manager._validate_path(path_str)

            if not file_path.exists():
                return {"success": False, "error": f"File not found: {path_str}"}

            if not file_path.is_file():
                return {"success": False, "error": f"Not a file: {path_str}"}

            size = file_path.stat().st_size
            # Telegram max file size is 50MB for bots
            if size > 50 * 1024 * 1024:
                return {
                    "success": False,
                    "error": f"File too large for Telegram ({size / 1048576:.1f}MB). Max is 50MB.",
                }

            size_str = (
                f"{size}B" if size < 1024
                else f"{size / 1024:.1f}KB" if size < 1048576
                else f"{size / 1048576:.1f}MB"
            )

            return {
                "success": True,
                "path": str(file_path),
                "filename": file_path.name,
                "size": size_str,
                "size_bytes": size,
                "caption": caption,
                "_action": "send_to_user",
            }
        except Exception as e:
            logger.error("upload_file failed", error=str(e))
            return {"success": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════
    # Scheduler Tools
    # ═══════════════════════════════════════════════════════════════

    def set_scheduler(self, scheduler, default_chat_id: int | None = None) -> None:
        """Inject the scheduler instance and register scheduler tools."""
        self._scheduler = scheduler
        self._scheduler_chat_id = default_chat_id

        self.register_tool(
            name="schedule_task",
            handler=self._handle_schedule_task,
            schema={
                "name": "schedule_task",
                "description": (
                    "Schedule a recurring or one-time task. The bot will send a message "
                    "when the task is due. Use for reminders, daily news, periodic checks, etc."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description": "What to do when triggered (e.g. 'Send daily news headlines')"},
                        "time": {"type": "string", "description": "When to trigger: ISO datetime or relative like '08:00', '2024-12-25T08:00:00'"},
                        "recurring": {
                            "type": "string",
                            "description": "Repeat interval: daily, hourly, weekly, or null for one-time",
                            "enum": ["daily", "hourly", "weekly"],
                        },
                    },
                    "required": ["description", "time"],
                },
            },
        )

        self.register_tool(
            name="list_scheduled_tasks",
            handler=self._handle_list_tasks,
            schema={
                "name": "list_scheduled_tasks",
                "description": "List all pending scheduled tasks/reminders",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        )

        self.register_tool(
            name="cancel_task",
            handler=self._handle_cancel_task,
            schema={
                "name": "cancel_task",
                "description": "Cancel a scheduled task by its ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "ID of the task to cancel"},
                    },
                    "required": ["task_id"],
                },
            },
        )

        logger.info("scheduler_tools_registered", tools=["schedule_task", "list_scheduled_tasks", "cancel_task"])

    async def _handle_schedule_task(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            if not self._scheduler:
                return {"success": False, "error": "Scheduler not available"}

            from datetime import datetime, timezone, timedelta

            description = parameters["description"]
            time_str = parameters["time"]
            recurring = parameters.get("recurring")

            # Parse time
            now = datetime.now(timezone.utc)

            # Try ISO format first
            trigger_time = None
            try:
                trigger_time = datetime.fromisoformat(time_str)
                if trigger_time.tzinfo is None:
                    trigger_time = trigger_time.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

            # Try HH:MM format (today or tomorrow)
            if trigger_time is None and ":" in time_str:
                try:
                    parts = time_str.strip().split(":")
                    hour, minute = int(parts[0]), int(parts[1])
                    trigger_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    if trigger_time <= now:
                        trigger_time += timedelta(days=1)
                except (ValueError, IndexError):
                    pass

            if trigger_time is None:
                return {"success": False, "error": f"Could not parse time: '{time_str}'. Use HH:MM or ISO format."}

            chat_id = self._scheduler_chat_id or 0
            task = await self._scheduler.schedule(
                chat_id=chat_id,
                description=description,
                trigger_time=trigger_time,
                recurring=recurring,
            )

            return {
                "success": True,
                "task_id": task.id,
                "description": description,
                "trigger_time": task.trigger_time,
                "recurring": recurring or "one-time",
            }
        except Exception as e:
            logger.error("schedule_task failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_list_tasks(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            if not self._scheduler:
                return {"success": False, "error": "Scheduler not available"}

            tasks = await self._scheduler.list_tasks(chat_id=self._scheduler_chat_id)
            return {
                "success": True,
                "tasks": [
                    {
                        "id": t.id,
                        "description": t.description,
                        "trigger_time": t.trigger_time,
                        "recurring": t.recurring or "one-time",
                    }
                    for t in tasks
                ],
                "count": len(tasks),
            }
        except Exception as e:
            logger.error("list_tasks failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def _handle_cancel_task(self, parameters: dict[str, Any]) -> dict[str, Any]:
        try:
            if not self._scheduler:
                return {"success": False, "error": "Scheduler not available"}

            task_id = parameters["task_id"]
            cancelled = await self._scheduler.cancel(task_id)
            return {
                "success": cancelled,
                "task_id": task_id,
                "message": f"Task {task_id} cancelled" if cancelled else f"Task {task_id} not found",
            }
        except Exception as e:
            logger.error("cancel_task failed", error=str(e))
            return {"success": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════
    # ToolPort Implementation
    # ═══════════════════════════════════════════════════════════════

    async def execute(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a tool.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters

        Returns:
            Tool execution result
        """
        if tool_name not in self._tools:
            logger.error("Tool not found", tool_name=tool_name)
            return {
                "success": False,
                "error": f"Tool not found: {tool_name}",
            }

        handler = self._tools[tool_name]

        logger.debug(
            "Executing tool",
            tool_name=tool_name,
            parameters=parameters,
        )

        try:
            result = await handler(parameters)
            return result
        except Exception as e:
            logger.error(
                "Tool execution failed",
                tool_name=tool_name,
                error=str(e),
            )
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
            }

    def get_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        """Get schema for a tool."""
        return self._schemas.get(tool_name)

    def list_tools(self) -> list[dict[str, Any]]:
        """List all available tools with their schemas."""
        return [
            {
                "name": name,
                **schema,
            }
            for name, schema in self._schemas.items()
        ]

    def register_tool(
        self,
        name: str,
        handler: Callable,
        schema: dict[str, Any],
    ) -> None:
        """
        Register a new tool.

        Args:
            name: Tool name
            handler: Tool handler function
            schema: Tool schema for LLM
        """
        self._tools[name] = handler
        self._schemas[name] = schema

        logger.debug("Tool registered", name=name, description=schema.get("description", ""))

    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available."""
        return tool_name in self._tools

    def get_tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())


# Global instance
_tool_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry
