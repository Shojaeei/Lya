"""Direct Access Module for Lya.

Provides direct file, network, and system access without external dependencies.
Uses only Python 3.14+ standard library.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from collections.abc import Sequence


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    returncode: int
    stdout: str
    stderr: str
    command: str
    duration_ms: float = 0.0
    error: str | None = None


@dataclass
class HttpResponse:
    """HTTP response wrapper."""
    success: bool
    status: int
    content: str
    headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    url: str = ""


@dataclass
class FileInfo:
    """File information."""
    path: str
    exists: bool
    is_file: bool = False
    is_dir: bool = False
    size: int = 0
    modified: float = 0.0
    error: str | None = None


class DirectAccess:
    """
    Direct system access using only Python standard library.

    Provides:
    - File operations (read/write/list)
    - Network requests (GET/POST)
    - Command execution
    - System information
    """

    # Dangerous commands to block
    BLOCKED_COMMANDS: Sequence[str] = (
        "rm -rf /", "rm -rf /*", "> /dev/sda", "> /dev/hda",
        "mkfs", "dd if=/dev/zero", "format c:", ":(){ :|: & };:",
        "chmod -R 777 /", "chown -R", "del /f /s /q c:\\" ,
    )

    def __init__(self, workspace: str | Path | None = None) -> None:
        """Initialize direct access.

        Args:
            workspace: Default workspace directory
        """
        self.workspace = Path(workspace).expanduser() if workspace else Path.home() / ".lya"
        self.workspace.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════
    # FILE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════

    def read_file(self, path: str | Path, encoding: str = "utf-8") -> dict[str, Any]:
        """Read file contents directly.

        Args:
            path: File path to read
            encoding: Text encoding

        Returns:
            Dict with success, content, and metadata
        """
        try:
            file_path = Path(path).expanduser()

            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {path}",
                    "path": str(file_path),
                }

            content = file_path.read_text(encoding=encoding)
            stat = file_path.stat()

            return {
                "success": True,
                "content": content,
                "path": str(file_path),
                "size": len(content.encode(encoding)),
                "modified": stat.st_mtime,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": str(path),
            }

    def write_file(
        self,
        path: str | Path,
        content: str,
        encoding: str = "utf-8",
        append: bool = False
    ) -> dict[str, Any]:
        """Write content to file directly.

        Args:
            path: File path to write
            content: Content to write
            encoding: Text encoding
            append: Append instead of overwrite

        Returns:
            Dict with success status and metadata
        """
        try:
            file_path = Path(path).expanduser()
            file_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            file_path.write_text(content, encoding=encoding)

            return {
                "success": True,
                "path": str(file_path),
                "bytes_written": len(content.encode(encoding)),
                "operation": "append" if append else "write",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": str(path),
            }

    def list_directory(
        self,
        path: str | Path = ".",
        pattern: str = "*",
        recursive: bool = False
    ) -> dict[str, Any]:
        """List directory contents.

        Args:
            path: Directory path
            pattern: Glob pattern
            recursive: Include subdirectories

        Returns:
            Dict with file list and metadata
        """
        try:
            dir_path = Path(path).expanduser()

            if not dir_path.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {path}",
                    "files": [],
                }

            if recursive:
                iterator = dir_path.rglob(pattern)
            else:
                iterator = dir_path.glob(pattern)

            files = []
            for item in iterator:
                try:
                    stat = item.stat()
                    files.append({
                        "name": item.name,
                        "path": str(item),
                        "is_file": item.is_file(),
                        "is_dir": item.is_dir(),
                        "size": stat.st_size if item.is_file() else 0,
                        "modified": stat.st_mtime,
                    })
                except (OSError, PermissionError):
                    continue

            return {
                "success": True,
                "path": str(dir_path),
                "pattern": pattern,
                "recursive": recursive,
                "files": files,
                "count": len(files),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": str(path),
                "files": [],
            }

    def file_info(self, path: str | Path) -> dict[str, Any]:
        """Get file information.

        Args:
            path: File or directory path

        Returns:
            Dict with file metadata
        """
        try:
            file_path = Path(path).expanduser()

            if not file_path.exists():
                return {
                    "success": False,
                    "exists": False,
                    "path": str(file_path),
                }

            stat = file_path.stat()

            return {
                "success": True,
                "exists": True,
                "path": str(file_path),
                "is_file": file_path.is_file(),
                "is_dir": file_path.is_dir(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "created": stat.st_ctime,
                "permissions": oct(stat.st_mode)[-3:],
            }
        except Exception as e:
            return {
                "success": False,
                "exists": False,
                "error": str(e),
                "path": str(path),
            }

    def delete_file(self, path: str | Path, recursive: bool = False) -> dict[str, Any]:
        """Delete file or directory.

        Args:
            path: Path to delete
            recursive: Delete directories recursively

        Returns:
            Dict with success status
        """
        try:
            file_path = Path(path).expanduser()

            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"Path not found: {path}",
                }

            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                if recursive:
                    import shutil
                    shutil.rmtree(file_path)
                else:
                    file_path.rmdir()

            return {
                "success": True,
                "path": str(file_path),
                "type": "directory" if file_path.is_dir() else "file",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": str(path),
            }

    def copy_file(
        self,
        source: str | Path,
        destination: str | Path
    ) -> dict[str, Any]:
        """Copy file or directory.

        Args:
            source: Source path
            destination: Destination path

        Returns:
            Dict with success status
        """
        try:
            src_path = Path(source).expanduser()
            dst_path = Path(destination).expanduser()

            if not src_path.exists():
                return {
                    "success": False,
                    "error": f"Source not found: {source}",
                }

            import shutil

            if src_path.is_file():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
            else:
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

            return {
                "success": True,
                "source": str(src_path),
                "destination": str(dst_path),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "source": str(source),
                "destination": str(destination),
            }

    # ═══════════════════════════════════════════════════════════════════
    # NETWORK OPERATIONS
    # ═══════════════════════════════════════════════════════════════════

    def http_get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 30
    ) -> dict[str, Any]:
        """Make HTTP GET request.

        Args:
            url: URL to request
            headers: Optional HTTP headers
            timeout: Request timeout in seconds

        Returns:
            Dict with response data
        """
        try:
            request = urllib.request.Request(
                url,
                headers=headers or {},
                method="GET"
            )

            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read().decode("utf-8", errors="replace")

                return {
                    "success": True,
                    "status": response.status,
                    "content": content,
                    "url": url,
                    "headers": dict(response.headers),
                }
        except urllib.error.HTTPError as e:
            return {
                "success": False,
                "status": e.code,
                "error": str(e.reason),
                "url": url,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
            }

    def http_post(
        self,
        url: str,
        data: dict[str, Any] | str,
        headers: dict[str, str] | None = None,
        timeout: int = 30
    ) -> dict[str, Any]:
        """Make HTTP POST request.

        Args:
            url: URL to request
            data: POST data (dict or string)
            headers: Optional HTTP headers
            timeout: Request timeout in seconds

        Returns:
            Dict with response data
        """
        try:
            if isinstance(data, dict):
                json_data = json.dumps(data).encode("utf-8")
                content_type = "application/json"
            else:
                json_data = data.encode("utf-8")
                content_type = "application/x-www-form-urlencoded"

            request_headers = headers or {}
            request_headers.setdefault("Content-Type", content_type)

            request = urllib.request.Request(
                url,
                data=json_data,
                headers=request_headers,
                method="POST"
            )

            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read().decode("utf-8", errors="replace")

                return {
                    "success": True,
                    "status": response.status,
                    "content": content,
                    "url": url,
                    "headers": dict(response.headers),
                }
        except urllib.error.HTTPError as e:
            return {
                "success": False,
                "status": e.code,
                "error": str(e.reason),
                "url": url,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
            }

    def download_file(
        self,
        url: str,
        save_path: str | Path,
        timeout: int = 60
    ) -> dict[str, Any]:
        """Download file from URL.

        Args:
            url: URL to download
            save_path: Local path to save file
            timeout: Download timeout in seconds

        Returns:
            Dict with download result
        """
        try:
            file_path = Path(save_path).expanduser()
            file_path.parent.mkdir(parents=True, exist_ok=True)

            urllib.request.urlretrieve(url, file_path, timeout=timeout)

            stat = file_path.stat()

            return {
                "success": True,
                "path": str(file_path),
                "size": stat.st_size,
                "url": url,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "path": str(save_path),
            }

    def check_internet(self) -> dict[str, Any]:
        """Check if internet connection is available.

        Returns:
            Dict with connectivity status
        """
        try:
            # Try to reach Google DNS
            urllib.request.urlopen(
                "https://8.8.8.8",
                timeout=3
            )
            return {
                "success": True,
                "connected": True,
                "message": "Internet connection available",
            }
        except Exception:
            return {
                "success": True,
                "connected": False,
                "message": "No internet connection",
            }

    # ═══════════════════════════════════════════════════════════════════
    # SYSTEM OPERATIONS
    # ═══════════════════════════════════════════════════════════════════

    def _check_command_safety(self, command: str) -> tuple[bool, str | None]:
        """Check if command is safe to execute."""
        cmd_lower = command.lower().strip()

        for blocked in self.BLOCKED_COMMANDS:
            if blocked.lower() in cmd_lower:
                return False, f"Command blocked for safety: {blocked}"

        return True, None

    def execute(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: int = 30,
        env: dict[str, str] | None = None,
        shell: bool = True
    ) -> dict[str, Any]:
        """Execute shell command directly.

        Args:
            command: Command to execute
            cwd: Working directory
            timeout: Timeout in seconds
            env: Environment variables
            shell: Use shell execution

        Returns:
            Dict with command result
        """
        import time

        start_time = time.time()

        # Safety check
        is_safe, error = self._check_command_safety(command)
        if not is_safe:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": error,
                "command": command,
                "duration_ms": 0.0,
            }

        try:
            result = subprocess.run(
                command,
                shell=shell,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, **(env or {})},
            )

            duration_ms = (time.time() - start_time) * 1000

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command,
                "duration_ms": duration_ms,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "command": command,
                "duration_ms": timeout * 1000,
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "command": command,
                "duration_ms": (time.time() - start_time) * 1000,
            }

    def command_exists(self, command: str) -> bool:
        """Check if command exists in PATH.

        Args:
            command: Command name

        Returns:
            True if command exists
        """
        import shutil
        return shutil.which(command) is not None

    def get_system_info(self) -> dict[str, Any]:
        """Get system information.

        Returns:
            Dict with system details
        """
        import platform

        return {
            "success": True,
            "platform": platform.system(),
            "platform_version": platform.version(),
            "platform_release": platform.release(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node_name": platform.node(),
            "architecture": platform.architecture(),
        }

    def get_environment(self) -> dict[str, Any]:
        """Get environment variables.

        Returns:
            Dict with environment info
        """
        return {
            "success": True,
            "variables": dict(os.environ),
            "path": os.environ.get("PATH", "").split(os.pathsep),
            "home": os.path.expanduser("~"),
            "cwd": os.getcwd(),
        }


# ═══════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE & CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

_direct_access: DirectAccess | None = None


def get_direct_access(workspace: str | Path | None = None) -> DirectAccess:
    """Get global direct access instance."""
    global _direct_access
    if _direct_access is None:
        _direct_access = DirectAccess(workspace)
    return _direct_access


# Convenience functions for quick access
def read_file(path: str, encoding: str = "utf-8") -> dict[str, Any]:
    """Read file."""
    return get_direct_access().read_file(path, encoding)


def write_file(path: str, content: str, encoding: str = "utf-8") -> dict[str, Any]:
    """Write file."""
    return get_direct_access().write_file(path, content, encoding)


def list_files(path: str = ".", pattern: str = "*", recursive: bool = False) -> dict[str, Any]:
    """List directory."""
    return get_direct_access().list_directory(path, pattern, recursive)


def http_get(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    """HTTP GET."""
    return get_direct_access().http_get(url, headers, timeout)


def http_post(url: str, data: dict[str, Any] | str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """HTTP POST."""
    return get_direct_access().http_post(url, data, headers)


def execute(command: str, cwd: str | None = None, timeout: int = 30) -> dict[str, Any]:
    """Execute command."""
    return get_direct_access().execute(command, cwd, timeout)


def download(url: str, save_path: str, timeout: int = 60) -> dict[str, Any]:
    """Download file."""
    return get_direct_access().download_file(url, save_path, timeout)
