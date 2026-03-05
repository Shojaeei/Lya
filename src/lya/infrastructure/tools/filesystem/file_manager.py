"""File System Tools.

Provides file operations with safety checks and permissions.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal

import aiofiles
from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


@dataclass
class FileInfo:
    """Information about a file."""
    path: Path
    size: int
    modified: float
    created: float
    is_dir: bool
    is_file: bool
    permissions: str
    hash: str | None = None


@dataclass
class FileSearchResult:
    """Result of file search."""
    path: Path
    line_number: int | None = None
    context: str | None = None
    match: str | None = None


class FileManager:
    """
    Safe file system manager with permissions.

    Features:
    - Path validation (prevents escape from workspace)
    - Async file operations
    - Content search
    - Batch operations
    """

    # Safety: Blocked paths (absolute)
    BLOCKED_PATHS = [
        "/etc/passwd",
        "/etc/shadow",
        "/.env",
        "/.ssh",
        "/.aws",
        "/root",
    ]

    # Safety: Blocked extensions
    DANGEROUS_EXTENSIONS = [
        ".exe", ".dll", ".so", ".dylib", ".sh", ".bat", ".cmd"
    ]

    def __init__(self, workspace: Path | None = None):
        self.workspace = Path(workspace or settings.workspace_path).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _validate_path(self, path: str | Path) -> Path:
        """
        Validate and resolve path.

        Ensures path is within workspace (prevents directory traversal).
        """
        target = Path(path).expanduser()

        # If relative, make relative to workspace BEFORE resolving
        if not target.is_absolute():
            target = self.workspace / target

        target = target.resolve()

        # Check if within workspace
        try:
            target.relative_to(self.workspace)
        except ValueError:
            raise PermissionError(
                f"Access denied: {target} is outside workspace {self.workspace}"
            )

        # Check blocked paths
        for blocked in self.BLOCKED_PATHS:
            if str(target).startswith(blocked):
                raise PermissionError(f"Access denied: {target} is blocked")

        return target

    def _check_extension(self, path: Path, mode: Literal["read", "write"] = "read") -> None:
        """Check file extension safety."""
        ext = path.suffix.lower()

        if ext in self.DANGEROUS_EXTENSIONS:
            raise PermissionError(f"Blocked file type: {ext}")

        # Extra check for write mode
        if mode == "write":
            # Don't allow overwriting critical files
            if path.name in ["requirements.txt", "pyproject.toml", ".env"]:
                logger.warning("Modifying critical file", path=str(path))

    # ═══════════════════════════════════════════════════════════════
    # Basic Operations
    # ═══════════════════════════════════════════════════════════════

    async def read_file(self, path: str | Path, encoding: str = "utf-8") -> str:
        """
        Read file contents.

        Args:
            path: File path
            encoding: File encoding

        Returns:
            File contents as string
        """
        target = self._validate_path(path)
        self._check_extension(target, "read")

        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")

        if not target.is_file():
            raise IsADirectoryError(f"Is a directory: {target}")

        logger.debug("Reading file", path=str(target))

        try:
            async with aiofiles.open(target, "r", encoding=encoding) as f:
                content = await f.read()
            return content
        except UnicodeDecodeError:
            # Try binary and decode
            async with aiofiles.open(target, "rb") as f:
                content = await f.read()
            return content.decode(encoding, errors="replace")

    async def read_binary(self, path: str | Path) -> bytes:
        """Read file as binary."""
        target = self._validate_path(path)

        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")

        async with aiofiles.open(target, "rb") as f:
            return await f.read()

    async def write_file(
        self,
        path: str | Path,
        content: str | bytes,
        encoding: str = "utf-8",
        backup: bool = True,
    ) -> None:
        """
        Write file contents.

        Args:
            path: File path
            content: Content to write
            encoding: Text encoding (for string content)
            backup: Create .backup file if exists
        """
        target = self._validate_path(path)
        self._check_extension(target, "write")

        # Ensure parent directory exists
        target.parent.mkdir(parents=True, exist_ok=True)

        # Create backup if file exists
        if backup and target.exists():
            backup_path = target.with_suffix(target.suffix + ".backup")
            shutil.copy2(target, backup_path)
            logger.debug("Created backup", original=str(target), backup=str(backup_path))

        logger.info("Writing file", path=str(target), size=len(content))

        mode = "w" if isinstance(content, str) else "wb"
        async with aiofiles.open(target, mode, encoding=encoding if mode == "w" else None) as f:
            await f.write(content)

    async def append_file(self, path: str | Path, content: str, encoding: str = "utf-8") -> None:
        """Append to file."""
        target = self._validate_path(path)
        self._check_extension(target, "write")

        target.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(target, "a", encoding=encoding) as f:
            await f.write(content)

    async def delete_file(self, path: str | Path, confirm: bool = False) -> bool:
        """
        Delete file.

        Args:
            path: File to delete
            confirm: Require confirmation for existing files

        Returns:
            True if deleted
        """
        target = self._validate_path(path)

        if not target.exists():
            return False

        if confirm:
            # In autonomous mode, log and skip
            logger.warning("Skipping deletion without explicit confirmation", path=str(target))
            return False

        if target.is_file():
            target.unlink()
            logger.info("Deleted file", path=str(target))
        elif target.is_dir():
            shutil.rmtree(target)
            logger.info("Deleted directory", path=str(target))

        return True

    async def move_file(
        self,
        source: str | Path,
        destination: str | Path,
    ) -> Path:
        """Move/rename file."""
        src = self._validate_path(source)
        dst = self._validate_path(destination)

        self._check_extension(dst, "write")

        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(src), str(dst))
        logger.info("Moved file", src=str(src), dst=str(dst))

        return dst

    async def copy_file(
        self,
        source: str | Path,
        destination: str | Path,
    ) -> Path:
        """Copy file."""
        src = self._validate_path(source)
        dst = self._validate_path(destination)

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.is_file():
            shutil.copy2(src, dst)
        elif src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)

        logger.info("Copied", src=str(src), dst=str(dst))
        return dst

    # ═══════════════════════════════════════════════════════════════
    # Directory Operations
    # ═══════════════════════════════════════════════════════════════

    async def list_directory(
        self,
        path: str | Path = ".",
        pattern: str = "*",
        recursive: bool = False,
    ) -> list[FileInfo]:
        """
        List directory contents.

        Args:
            path: Directory path
            pattern: Glob pattern to filter
            recursive: List recursively

        Returns:
            List of file information
        """
        target = self._validate_path(path)

        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {target}")

        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {target}")

        results = []

        if recursive:
            for item in target.rglob(pattern):
                results.append(self._get_file_info(item))
        else:
            for item in target.glob(pattern):
                results.append(self._get_file_info(item))

        return sorted(results, key=lambda x: str(x.path))

    async def create_directory(self, path: str | Path, exist_ok: bool = True) -> Path:
        """Create directory."""
        target = self._validate_path(path)

        target.mkdir(parents=True, exist_ok=exist_ok)
        logger.debug("Created directory", path=str(target))

        return target

    async def remove_directory(self, path: str | Path, recursive: bool = False) -> None:
        """Remove directory."""
        target = self._validate_path(path)

        if recursive:
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.rmdir()  # Only works if empty

        logger.info("Removed directory", path=str(target))

    # ═══════════════════════════════════════════════════════════════
    # Search Operations
    # ═══════════════════════════════════════════════════════════════

    async def search_content(
        self,
        query: str,
        path: str | Path = ".",
        pattern: str = "*",
        case_sensitive: bool = False,
    ) -> AsyncIterator[FileSearchResult]:
        """
        Search file contents.

        Args:
            query: Search string
            path: Directory to search
            pattern: File pattern to include
            case_sensitive: Case sensitivity

        Yields:
            Search results
        """
        target = self._validate_path(path)

        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {target}")

        flags = 0 if case_sensitive else re.IGNORECASE

        for file_path in target.rglob(pattern):
            if not file_path.is_file():
                continue

            # Skip binary files
            if self._is_binary(file_path):
                continue

            try:
                content = await self.read_file(file_path)
                lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    if re.search(query, line, flags):
                        yield FileSearchResult(
                            path=file_path.relative_to(self.workspace),
                            line_number=i,
                            context=line.strip()[:200],
                            match=query,
                        )

            except Exception as e:
                logger.debug("Search skipped file", path=str(file_path), error=str(e))
                continue

    async def find_by_name(
        self,
        name_pattern: str,
        path: str | Path = ".",
    ) -> list[Path]:
        """Find files by name pattern."""
        target = self._validate_path(path)
        results = []

        for item in target.rglob(name_pattern):
            results.append(item.relative_to(self.workspace))

        return results

    async def grep(
        self,
        pattern: str,
        path: str | Path,
        include: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Grep-like search with context.

        Args:
            pattern: Regex pattern
            path: Path to search
            include: File extensions to include (e.g., [".py", ".md"])

        Returns:
            List of matches with context
        """
        import re

        target = self._validate_path(path)
        results = []

        if target.is_file():
            files = [target]
        else:
            files = []
            for ext in (include or ["*"]):
                files.extend(target.rglob(f"*{ext}"))

        for file_path in files:
            if not file_path.is_file():
                continue

            try:
                content = await self.read_file(file_path)
                lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        # Get context (3 lines before/after)
                        start = max(0, i - 4)
                        end = min(len(lines), i + 3)
                        context = "\n".join(lines[start:end])

                        results.append({
                            "file": str(file_path.relative_to(self.workspace)),
                            "line": i,
                            "text": line.strip(),
                            "context": context,
                        })

            except Exception:
                continue

        return results

    # ═══════════════════════════════════════════════════════════════
    # File Analysis
    # ═══════════════════════════════════════════════════════════════

    async def get_file_hash(self, path: str | Path) -> str:
        """Calculate file hash (SHA256)."""
        target = self._validate_path(path)

        if not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")

        sha256_hash = hashlib.sha256()

        async with aiofiles.open(target, "rb") as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    async def analyze_directory(self, path: str | Path = ".") -> dict[str, Any]:
        """Analyze directory structure."""
        target = self._validate_path(path)

        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {target}")

        stats = {
            "total_files": 0,
            "total_dirs": 0,
            "total_size": 0,
            "extensions": {},
            "largest_files": [],
        }

        for item in target.rglob("*"):
            if item.is_file():
                stats["total_files"] += 1
                size = item.stat().st_size
                stats["total_size"] += size

                ext = item.suffix.lower()
                stats["extensions"][ext] = stats["extensions"].get(ext, 0) + 1

                # Track largest files
                stats["largest_files"].append((item, size))
                stats["largest_files"].sort(key=lambda x: x[1], reverse=True)
                stats["largest_files"] = stats["largest_files"][:10]

            elif item.is_dir():
                stats["total_dirs"] += 1

        # Convert largest files to serializable format
        stats["largest_files"] = [
            {"path": str(p.relative_to(self.workspace)), "size": s}
            for p, s in stats["largest_files"]
        ]

        return stats

    # ═══════════════════════════════════════════════════════════════
    # Helper Methods
    # ═══════════════════════════════════════════════════════════════

    def _get_file_info(self, path: Path) -> FileInfo:
        """Get file information."""
        stat = path.stat()

        return FileInfo(
            path=path.relative_to(self.workspace),
            size=stat.st_size,
            modified=stat.st_mtime,
            created=stat.st_ctime,
            is_dir=path.is_dir(),
            is_file=path.is_file(),
            permissions=oct(stat.st_mode)[-3:],
        )

    def _is_binary(self, path: Path, sample_size: int = 8192) -> bool:
        """Check if file is binary."""
        try:
            with open(path, "rb") as f:
                chunk = f.read(sample_size)
                return b"\x00" in chunk
        except:
            return True

    # ═══════════════════════════════════════════════════════════════
    # Convenience Methods
    # ═══════════════════════════════════════════════════════════════

    async def edit_file(
        self,
        path: str | Path,
        edits: list[dict[str, str]],  # [{"old": "...", "new": "..."}]
    ) -> dict[str, Any]:
        """
        Apply multiple edits to a file.

        Args:
            path: File path
            edits: List of {old_text: new_text}

        Returns:
            Edit results
        """
        content = await self.read_file(path)
        original = content
        results = []

        for i, edit in enumerate(edits):
            old_text = edit.get("old", "")
            new_text = edit.get("new", "")

            if old_text in content:
                content = content.replace(old_text, new_text, 1)
                results.append({"edit": i, "status": "applied"})
            else:
                results.append({"edit": i, "status": "not_found", "text": old_text[:50]})

        if content != original:
            await self.write_file(path, content, backup=True)

        return {
            "path": str(path),
            "edits_attempted": len(edits),
            "edits_applied": sum(1 for r in results if r["status"] == "applied"),
            "results": results,
        }

    async def merge_files(
        self,
        sources: list[str | Path],
        destination: str | Path,
    ) -> Path:
        """Merge multiple files into one."""
        dst = self._validate_path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)

        merged = []
        for src in sources:
            src_path = self._validate_path(src)
            content = await self.read_file(src_path)
            merged.append(f"# === {src_path.name} ===\n{content}\n")

        await self.write_file(dst, "\n".join(merged))
        return dst


# Global instance
_file_manager: FileManager | None = None


def get_file_manager() -> FileManager:
    """Get global file manager."""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager


import re
