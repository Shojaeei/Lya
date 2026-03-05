"""Git Adapter.

Provides git operations with safety checks and logging.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


@dataclass
class Commit:
    """Git commit information."""
    hash: str
    message: str
    author: str
    date: str
    files_changed: list[str]


@dataclass
class Diff:
    """Git diff information."""
    old_path: str
    new_path: str
    change_type: Literal["added", "deleted", "modified", "renamed"]
    additions: int
    deletions: int
    patch: str


@dataclass
class Branch:
    """Git branch information."""
    name: str
    is_current: bool
    is_remote: bool
    ahead: int
    behind: int


class GitAdapter:
    """
    Adapter for git operations.

    Provides safe, logged git operations with:
    - Path validation
    - Command sanitization
    - Async execution
    - Error handling
    """

    # Safety: Blocked remote URLs
    BLOCKED_URL_PATTERNS = [
        r".*@.*:.*",  # SSH URLs (require keys)
        r".*\.(sh|exe|bat)$",  # Executable URLs
    ]

    def __init__(self, working_dir: Path | str | None = None):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self._ensure_git_available()

    def _ensure_git_available(self) -> None:
        """Check if git is installed."""
        try:
            result = self._run_git_command(["--version"], cwd=None)
            if result["returncode"] != 0:
                raise RuntimeError("Git not found")
            logger.debug("Git available", version=result["stdout"].strip())
        except FileNotFoundError:
            raise RuntimeError("Git not installed. Install git first.")

    def _run_git_command(
        self,
        args: list[str],
        cwd: Path | str | None = None,
        check: bool = True,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """
        Run a git command safely.

        Args:
            args: Git command arguments
            cwd: Working directory
            check: Raise on error
            timeout: Command timeout

        Returns:
            Dict with stdout, stderr, returncode
        """
        cmd = ["git"] + args

        cwd = cwd or self.working_dir

        logger.debug("Running git command", cmd=" ".join(cmd), cwd=str(cwd))

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

            if check and result.returncode != 0:
                raise RuntimeError(
                    f"Git command failed: {' '.join(cmd)}\n"
                    f"stderr: {result.stderr}"
                )

            return output

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Git command timed out after {timeout}s")

    def _sanitize_branch_name(self, name: str) -> str:
        """Sanitize branch name for safety."""
        # Remove dangerous characters
        sanitized = re.sub(r"[^a-zA-Z0-9_\-\/]", "-", name)
        # Ensure doesn't start with -
        sanitized = sanitized.lstrip("-")
        # Limit length
        return sanitized[:100]

    def _is_safe_remote_url(self, url: str) -> bool:
        """Check if remote URL is safe."""
        for pattern in self.BLOCKED_URL_PATTERNS:
            if re.match(pattern, url, re.IGNORECASE):
                return False
        return True

    # ═══════════════════════════════════════════════════════════════
    # Repository Operations
    # ═══════════════════════════════════════════════════════════════

    async def clone(
        self,
        url: str,
        destination: Path | str | None = None,
        branch: str | None = None,
        depth: int | None = None,
    ) -> Path:
        """
        Clone a git repository.

        Args:
            url: Repository URL
            destination: Local path (optional)
            branch: Branch to clone
            depth: Shallow clone depth

        Returns:
            Path to cloned repository
        """
        if not self._is_safe_remote_url(url):
            raise ValueError(f"URL blocked: {url}")

        # Determine destination
        if destination:
            dest_path = Path(destination)
        else:
            # Extract repo name from URL
            repo_name = url.split("/")[-1].replace(".git", "")
            dest_path = self.working_dir / repo_name

        if dest_path.exists():
            raise FileExistsError(f"Destination already exists: {dest_path}")

        # Build command
        cmd = ["clone"]
        if branch:
            cmd.extend(["--branch", branch])
        if depth:
            cmd.extend(["--depth", str(depth)])
        cmd.extend([url, str(dest_path)])

        logger.info("Cloning repository", url=url, destination=str(dest_path))

        result = self._run_git_command(cmd, cwd=None)

        logger.info("Clone completed", destination=str(dest_path))

        return dest_path

    async def init(self, path: Path | str | None = None, bare: bool = False) -> Path:
        """Initialize a new git repository."""
        repo_path = Path(path) if path else self.working_dir

        cmd = ["init"]
        if bare:
            cmd.append("--bare")
        cmd.append(str(repo_path))

        self._run_git_command(cmd, cwd=None)

        logger.info("Repository initialized", path=str(repo_path))

        return repo_path

    # ═══════════════════════════════════════════════════════════════
    # Branch Operations
    # ═══════════════════════════════════════════════════════════════

    async def create_branch(
        self,
        name: str,
        checkout: bool = True,
        from_branch: str | None = None,
    ) -> str:
        """
        Create a new branch.

        Args:
            name: Branch name
            checkout: Switch to new branch
            from_branch: Base branch (default: current)

        Returns:
            Branch name
        """
        sanitized = self._sanitize_branch_name(name)

        # Create branch
        cmd = ["checkout", "-b", sanitized]
        if from_branch:
            cmd.append(from_branch)

        self._run_git_command(cmd)

        logger.info("Branch created", branch=sanitized, checkout=checkout)

        return sanitized

    async def checkout(self, branch: str, create: bool = False) -> None:
        """Checkout a branch."""
        sanitized = self._sanitize_branch_name(branch)

        if create:
            cmd = ["checkout", "-b", sanitized]
        else:
            cmd = ["checkout", sanitized]

        self._run_git_command(cmd)

        logger.info("Checked out branch", branch=sanitized)

    async def list_branches(self, remote: bool = False) -> list[Branch]:
        """List branches."""
        if remote:
            cmd = ["branch", "-r", "--format", "%(refname:short)"]
        else:
            cmd = ["branch", "-v", "--format", "%(refname:short) %(upstream:track)"]

        result = self._run_git_command(cmd)
        branches = []

        for line in result["stdout"].strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            name = parts[0]

            # Parse tracking info
            is_current = name.startswith("*")
            if is_current:
                name = name[1:].strip()

            branches.append(Branch(
                name=name,
                is_current=is_current,
                is_remote=remote,
                ahead=0,  # Would need git rev-list to get actual values
                behind=0,
            ))

        return branches

    async def delete_branch(
        self,
        name: str,
        force: bool = False,
        remote: bool = False,
    ) -> None:
        """Delete a branch."""
        sanitized = self._sanitize_branch_name(name)

        if remote:
            cmd = ["push", "origin", "--delete", sanitized]
        else:
            cmd = ["branch", "-D" if force else "-d", sanitized]

        self._run_git_command(cmd)

        logger.info("Branch deleted", branch=sanitized, remote=remote)

    # ═══════════════════════════════════════════════════════════════
    # Commit Operations
    # ═══════════════════════════════════════════════════════════════

    async def add(self, files: list[str] | str | None = None) -> None:
        """Stage files."""
        cmd = ["add"]

        if files is None:
            cmd.append(".")
        elif isinstance(files, str):
            cmd.append(files)
        else:
            cmd.extend(files)

        self._run_git_command(cmd)

        logger.debug("Files staged", files=files)

    async def commit(
        self,
        message: str,
        files: list[str] | None = None,
        allow_empty: bool = False,
    ) -> str:
        """
        Create a commit.

        Args:
            message: Commit message
            files: Specific files to commit (None = all staged)
            allow_empty: Allow empty commit

        Returns:
            Commit hash
        """
        # Add files if specified
        if files:
            await self.add(files)

        # Create commit
        cmd = ["commit", "-m", message]
        if allow_empty:
            cmd.append("--allow-empty")

        self._run_git_command(cmd)

        # Get commit hash
        result = self._run_git_command(["rev-parse", "HEAD"])
        commit_hash = result["stdout"].strip()

        logger.info("Commit created", hash=commit_hash[:8], message=message[:50])

        return commit_hash

    async def get_commit_history(
        self,
        n: int = 10,
        path: str | None = None,
    ) -> list[Commit]:
        """Get commit history."""
        cmd = [
            "log",
            f"-{n}",
            "--pretty=format:%H|%s|%an|%ad",
            "--date=short",
        ]

        if path:
            cmd.append("--")
            cmd.append(path)

        result = self._run_git_command(cmd)

        commits = []
        for line in result["stdout"].strip().split("\n"):
            if "|" not in line:
                continue

            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append(Commit(
                    hash=parts[0],
                    message=parts[1],
                    author=parts[2],
                    date=parts[3],
                    files_changed=[],  # Would need separate call
                ))

        return commits

    async def get_last_commit(self) -> Commit | None:
        """Get last commit."""
        commits = await self.get_commit_history(1)
        return commits[0] if commits else None

    async def amend_commit(self, message: str | None = None) -> None:
        """Amend last commit."""
        cmd = ["commit", "--amend"]

        if message:
            cmd.extend(["-m", message])
        else:
            cmd.append("--no-edit")

        self._run_git_command(cmd)

        logger.info("Commit amended")

    # ═══════════════════════════════════════════════════════════════
    # Diff Operations
    # ═══════════════════════════════════════════════════════════════

    async def diff(
        self,
        from_ref: str = "HEAD",
        to_ref: str | None = None,
        path: str | None = None,
    ) -> str:
        """
        Get diff between references.

        Args:
            from_ref: Starting reference (e.g., "HEAD", commit hash)
            to_ref: Ending reference (default: working tree)
            path: Specific file or directory

        Returns:
            Diff as string
        """
        cmd = ["diff", from_ref]

        if to_ref:
            cmd.append(to_ref)
        if path:
            cmd.append("--")
            cmd.append(path)

        result = self._run_git_command(cmd, check=False)

        return result["stdout"]

    async def get_changed_files(
        self,
        from_ref: str = "HEAD",
        to_ref: str | None = None,
    ) -> list[Diff]:
        """Get list of changed files."""
        cmd = ["diff", "--name-status", from_ref]
        if to_ref:
            cmd.append(to_ref)

        result = self._run_git_command(cmd, check=False)

        changes = []
        for line in result["stdout"].strip().split("\n"):
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                status = parts[0]
                path = parts[1]

                change_type = "modified"
                if status.startswith("A"):
                    change_type = "added"
                elif status.startswith("D"):
                    change_type = "deleted"
                elif status.startswith("R"):
                    change_type = "renamed"

                changes.append(Diff(
                    old_path=path,
                    new_path=parts[2] if len(parts) > 2 else path,
                    change_type=change_type,  # type: ignore
                    additions=0,
                    deletions=0,
                    patch="",
                ))

        return changes

    async def show_file_content(self, path: str, ref: str = "HEAD") -> str:
        """Show file content at specific commit."""
        cmd = ["show", f"{ref}:{path}"]
        result = self._run_git_command(cmd, check=False)
        return result["stdout"]

    # ═══════════════════════════════════════════════════════════════
    # Status and Info
    # ═══════════════════════════════════════════════════════════════

    async def status(self) -> dict[str, Any]:
        """Get repository status."""
        # Porcelain status for parsing
        result = self._run_git_command(
            ["status", "--porcelain", "-b"],
            check=False,
        )

        staged = []
        unstaged = []
        untracked = []
        branch = ""

        for line in result["stdout"].split("\n"):
            if line.startswith("##"):
                branch = line[2:].strip().split("...")[0]
            elif len(line) >= 2:
                index_status = line[0]
                worktree_status = line[1]
                filename = line[3:]

                if index_status != " " and index_status != "?":
                    staged.append(filename)
                if worktree_status != " " and worktree_status != "?":
                    unstaged.append(filename)
                if index_status == "?" and worktree_status == "?":
                    untracked.append(filename)

        return {
            "branch": branch,
            "staged": staged,
            "unstaged": unstaged,
            "untracked": untracked,
            "is_clean": len(staged) == 0 and len(unstaged) == 0,
        }

    async def is_repo(self, path: Path | str | None = None) -> bool:
        """Check if path is a git repository."""
        check_path = Path(path) if path else self.working_dir

        try:
            result = self._run_git_command(
                ["rev-parse", "--git-dir"],
                cwd=check_path,
                check=False,
            )
            return result["returncode"] == 0
        except:
            return False

    # ═══════════════════════════════════════════════════════════════
    # Remote Operations
    # ═══════════════════════════════════════════════════════════════

    async def add_remote(self, name: str, url: str) -> None:
        """Add remote."""
        self._run_git_command(["remote", "add", name, url])
        logger.info("Remote added", name=name, url=url)

    async def fetch(self, remote: str = "origin") -> None:
        """Fetch from remote."""
        self._run_git_command(["fetch", remote])
        logger.info("Fetched", remote=remote)

    async def pull(self, remote: str = "origin", branch: str | None = None) -> None:
        """Pull from remote."""
        cmd = ["pull", remote]
        if branch:
            cmd.append(branch)

        self._run_git_command(cmd)
        logger.info("Pulled", remote=remote, branch=branch)

    async def push(
        self,
        remote: str = "origin",
        branch: str | None = None,
        force: bool = False,
        set_upstream: bool = False,
    ) -> None:
        """Push to remote."""
        cmd = ["push"]

        if force:
            cmd.append("--force-with-lease")
        if set_upstream:
            cmd.append("-u")

        cmd.append(remote)

        if branch:
            cmd.append(branch)

        self._run_git_command(cmd)
        logger.info("Pushed", remote=remote, branch=branch)

    async def get_remotes(self) -> list[dict[str, str]]:
        """List remotes."""
        result = self._run_git_command(["remote", "-v"])

        remotes = []
        seen = set()

        for line in result["stdout"].strip().split("\n"):
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                url = parts[1]

                if name not in seen:
                    seen.add(name)
                    remotes.append({"name": name, "url": url})

        return remotes

    # ═══════════════════════════════════════════════════════════════
    # Stash Operations
    # ═══════════════════════════════════════════════════════════════

    async def stash(self, message: str | None = None, include_untracked: bool = False) -> None:
        """Stash changes."""
        cmd = ["stash", "push"]

        if message:
            cmd.extend(["-m", message])
        if include_untracked:
            cmd.append("-u")

        self._run_git_command(cmd)
        logger.info("Stashed changes", message=message)

    async def stash_pop(self, index: int = 0) -> None:
        """Apply stashed changes."""
        cmd = ["stash", "pop"]
        if index > 0:
            cmd.append(f"stash@{{{index}}}")

        self._run_git_command(cmd)
        logger.info("Stash applied")

    async def stash_list(self) -> list[dict[str, str]]:
        """List stashes."""
        result = self._run_git_command(["stash", "list"], check=False)

        stashes = []
        for i, line in enumerate(result["stdout"].strip().split("\n")):
            if line:
                stashes.append({
                    "index": str(i),
                    "message": line,
                })

        return stashes

    # ═══════════════════════════════════════════════════════════════
    # Advanced Operations
    # ═══════════════════════════════════════════════════════════════

    async def cherry_pick(self, commit_hash: str) -> None:
        """Cherry-pick a commit."""
        self._run_git_command(["cherry-pick", commit_hash])
        logger.info("Cherry-picked", commit=commit_hash[:8])

    async def revert(self, commit_hash: str) -> None:
        """Revert a commit."""
        self._run_git_command(["revert", "--no-edit", commit_hash])
        logger.info("Reverted", commit=commit_hash[:8])

    async def reset(
        self,
        ref: str = "HEAD",
        mode: Literal["soft", "mixed", "hard"] = "mixed",
    ) -> None:
        """Reset to reference."""
        cmd = ["reset", f"--{mode}", ref]
        self._run_git_command(cmd)
        logger.info("Reset", mode=mode, ref=ref)

    async def clean(self, force: bool = False, directories: bool = False) -> None:
        """Clean untracked files."""
        cmd = ["clean"]

        if force:
            cmd.append("-f")
        if directories:
            cmd.append("-d")

        self._run_git_command(cmd)
        logger.info("Cleaned untracked files")


# Global instance
_git_adapter: GitAdapter | None = None


def get_git_adapter(working_dir: Path | str | None = None) -> GitAdapter:
    """Get global git adapter."""
    global _git_adapter
    if _git_adapter is None:
        _git_adapter = GitAdapter(working_dir)
    return _git_adapter
