"""Git Integration Tools.

Provides git operations, code review, and repository management.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import aiofiles

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


@dataclass
class GitCommit:
    """Represents a git commit."""
    hash: str
    message: str
    author: str
    date: datetime
    files_changed: list[str] = field(default_factory=list)


@dataclass
class FileDiff:
    """Diff for a single file."""
    path: str
    status: Literal["added", "modified", "deleted", "renamed"]
    additions: int
    deletions: int
    diff_content: str | None = None


@dataclass
class PullRequest:
    """Pull request information."""
    number: int
    title: str
    description: str
    branch: str
    base: str
    status: str
    url: str


@dataclass
class CodeReviewComment:
    """Code review comment."""
    path: str
    line: int
    message: str
    severity: Literal["info", "warning", "error"]
    suggestion: str | None = None


class GitRepository:
    """
    Represents a git repository with safe operations.
    """

    # Safety: Maximum file size to diff
    MAX_DIFF_SIZE = 1024 * 1024  # 1MB

    # Safety: Blocked patterns in commit messages
    DANGEROUS_PATTERNS = [
        r"--.*",  # Command injection attempts
        r";\s*rm\s",  # Delete commands
        r"\|\s*bash",
    ]

    def __init__(self, path: str | Path, allow_dangerous: bool = False):
        """
        Initialize git repository.

        Args:
            path: Path to repository
            allow_dangerous: Allow operations outside workspace (use with caution)
        """
        self.path = Path(path).expanduser().resolve()

        if not allow_dangerous:
            # Validate path is within workspace
            workspace = settings.workspace_path.expanduser().resolve()
            try:
                self.path.relative_to(workspace)
            except ValueError:
                raise PermissionError(
                    f"Repository {self.path} is outside workspace {workspace}"
                )

        self._git_configured = False

    async def _run_git(
        self,
        args: list[str],
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run git command."""
        cmd = ["git"] + args

        try:
            result = subprocess.run(
                cmd,
                cwd=self.path,
                check=check,
                capture_output=capture_output,
                text=True,
                timeout=60,
            )
            return result
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Git command timed out: {' '.join(args)}")
        except subprocess.CalledProcessError as e:
            logger.error("Git command failed", cmd=args, stderr=e.stderr)
            raise

    def _is_git_repo(self) -> bool:
        """Check if path is a git repository."""
        git_dir = self.path / ".git"
        return git_dir.exists() and git_dir.is_dir()

    async def init(self, bare: bool = False) -> None:
        """Initialize a new git repository."""
        self.path.mkdir(parents=True, exist_ok=True)

        args = ["init"]
        if bare:
            args.append("--bare")

        await self._run_git(args)
        logger.info("Git repository initialized", path=str(self.path))

    async def clone(
        self,
        url: str,
        branch: str | None = None,
        depth: int | None = None,
    ) -> None:
        """
        Clone a repository.

        Args:
            url: Repository URL
            branch: Specific branch to clone
            depth: Shallow clone depth
        """
        if self.path.exists() and any(self.path.iterdir()):
            raise FileExistsError(f"Directory not empty: {self.path}")

        self.path.parent.mkdir(parents=True, exist_ok=True)

        args = ["clone", url, str(self.path)]

        if branch:
            args.extend(["--branch", branch])
        if depth:
            args.extend(["--depth", str(depth)])

        await self._run_git(args, check=False)

        if not self._is_git_repo():
            raise RuntimeError(f"Clone failed: {url}")

        logger.info("Repository cloned", url=url, path=str(self.path))

    async def status(self) -> dict[str, Any]:
        """Get repository status."""
        if not self._is_git_repo():
            return {"is_repo": False}

        # Get branch
        try:
            branch_result = await self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
            branch = branch_result.stdout.strip()
        except:
            branch = "unknown"

        # Get status
        status_result = await self._run_git(
            ["status", "--porcelain", "-u"],
            check=False,
        )

        staged = []
        unstaged = []
        untracked = []

        for line in status_result.stdout.split("\n"):
            if not line.strip():
                continue

            status_code = line[:2]
            filename = line[3:].strip()

            if status_code[0] != " ":
                staged.append({"file": filename, "status": status_code[0]})
            if status_code[1] != " ":
                unstaged.append({"file": filename, "status": status_code[1]})
            if status_code == "??":
                untracked.append(filename)

        return {
            "is_repo": True,
            "branch": branch,
            "staged": staged,
            "unstaged": unstaged,
            "untracked": untracked,
            "clean": not (staged or unstaged or untracked),
        }

    async def add(self, paths: list[str] | str) -> None:
        """Stage files."""
        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            # Validate path doesn't escape repo
            target = self.path / path
            try:
                target.relative_to(self.path)
            except ValueError:
                raise PermissionError(f"Path escapes repository: {path}")

        await self._run_git(["add"] + paths)
        logger.debug("Files staged", paths=paths)

    async def commit(
        self,
        message: str,
        author_name: str | None = None,
        author_email: str | None = None,
    ) -> str:
        """
        Create commit.

        Args:
            message: Commit message
            author_name: Optional author name
            author_email: Optional author email

        Returns:
            Commit hash
        """
        # Validate message
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                raise ValueError(f"Potentially dangerous pattern in commit message: {message[:50]}")

        # Configure git if needed
        if not self._git_configured:
            await self._configure_git(author_name, author_email)

        # Create commit
        result = await self._run_git(["commit", "-m", message])

        # Get commit hash
        hash_result = await self._run_git(["rev-parse", "HEAD"])
        commit_hash = hash_result.stdout.strip()

        logger.info("Commit created", hash=commit_hash[:8], message=message[:50])
        return commit_hash

    async def _configure_git(
        self,
        name: str | None = None,
        email: str | None = None,
    ) -> None:
        """Configure git user."""
        if not name:
            name = settings.get("git_name", "Lya Bot")
        if not email:
            email = settings.get("git_email", "lya@localhost")

        await self._run_git(["config", "user.name", name], check=False)
        await self._run_git(["config", "user.email", email], check=False)

        self._git_configured = True

    async def create_branch(self, name: str, base: str = "HEAD") -> None:
        """Create and checkout new branch."""
        # Validate branch name
        if not re.match(r"^[\w\-\/\.]+$", name):
            raise ValueError(f"Invalid branch name: {name}")

        await self._run_git(["checkout", "-b", name, base])
        logger.info("Branch created", name=name, base=base)

    async def checkout(self, ref: str) -> None:
        """Checkout reference."""
        await self._run_git(["checkout", ref])

    async def diff(
        self,
        staged: bool = False,
        file_path: str | None = None,
    ) -> list[FileDiff]:
        """
        Get diff of changes.

        Args:
            staged: Show staged changes
            file_path: Show diff for specific file

        Returns:
            List of file diffs
        """
        args = ["diff", "--numstat"]

        if staged:
            args.append("--staged")

        if file_path:
            args.append(file_path)

        result = await self._run_git(args, check=False)

        diffs = []
        for line in result.stdout.split("\n"):
            if not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    additions = int(parts[0]) if parts[0] != "-" else 0
                    deletions = int(parts[1]) if parts[1] != "-" else 0
                    filename = parts[2]

                    diffs.append(FileDiff(
                        path=filename,
                        status="modified",
                        additions=additions,
                        deletions=deletions,
                    ))
                except ValueError:
                    continue

        return diffs

    async def log(
        self,
        n: int = 10,
        file_path: str | None = None,
    ) -> list[GitCommit]:
        """Get commit history."""
        args = [
            "log",
            f"-{n}",
            "--pretty=format:%H|%s|%an|%ad",
            "--date=iso",
        ]

        if file_path:
            args.append(file_path)

        result = await self._run_git(args)

        commits = []
        for line in result.stdout.split("\n"):
            if not line.strip():
                continue

            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append(GitCommit(
                    hash=parts[0],
                    message=parts[1],
                    author=parts[2],
                    date=datetime.fromisoformat(parts[3]),
                ))

        return commits

    async def get_file_content(self, path: str, ref: str = "HEAD") -> str:
        """Get file content at specific ref."""
        result = await self._run_git(["show", f"{ref}:{path}"], check=False)
        return result.stdout

    async def push(
        self,
        remote: str = "origin",
        branch: str | None = None,
    ) -> None:
        """Push to remote."""
        args = ["push", remote]

        if branch:
            args.append(branch)

        await self._run_git(args)
        logger.info("Pushed to remote", remote=remote, branch=branch)

    async def pull(self, remote: str = "origin", branch: str | None = None) -> None:
        """Pull from remote."""
        args = ["pull", remote]

        if branch:
            args.append(branch)

        await self._run_git(args)

    async def get_remote_url(self) -> str | None:
        """Get remote URL."""
        try:
            result = await self._run_git(["remote", "get-url", "origin"])
            return result.stdout.strip()
        except:
            return None

    async def merge(self, branch: str, message: str | None = None) -> None:
        """Merge branch into current."""
        args = ["merge", branch]

        if message:
            args.extend(["-m", message])

        await self._run_git(args)

    async def reset(self, mode: Literal["soft", "mixed", "hard"] = "mixed", ref: str = "HEAD") -> None:
        """Reset to ref."""
        await self._run_git(["reset", f"--{mode}", ref])


class GitHubIntegration:
    """
    GitHub API integration for PRs and code review.

    Requires GITHUB_TOKEN environment variable.
    """

    def __init__(self, token: str | None = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            logger.warning("GitHub token not set")

    def _get_headers(self) -> dict[str, str]:
        """Get API headers."""
        return {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

    async def create_pull_request(
        self,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
    ) -> PullRequest:
        """
        Create pull request on GitHub.

        Args:
            repo: Owner/repo format (e.g., "user/repo")
            title: PR title
            body: PR description
            head: Branch with changes
            base: Target branch

        Returns:
            PullRequest object
        """
        import httpx

        url = f"https://api.github.com/repos/{repo}/pulls"

        data = {
            "title": title,
            "body": body,
            "head": head,
            "base": base,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=self._get_headers(),
                json=data,
            )
            response.raise_for_status()

            pr_data = response.json()

            return PullRequest(
                number=pr_data["number"],
                title=pr_data["title"],
                description=pr_data["body"],
                branch=head,
                base=base,
                status=pr_data["state"],
                url=pr_data["html_url"],
            )

    async def create_review_comment(
        self,
        repo: str,
        pr_number: int,
        path: str,
        line: int,
        message: str,
        suggestion: str | None = None,
    ) -> dict[str, Any]:
        """Create code review comment."""
        import httpx

        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"

        # Create review with comment
        data = {
            "body": "Code review by Lya",
            "event": "COMMENT",
            "comments": [
                {
                    "path": path,
                    "line": line,
                    "body": message,
                }
            ],
        }

        if suggestion:
            data["comments"][0]["body"] += f"\n\n```suggestion\n{suggestion}\n```"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=self._get_headers(),
                json=data,
            )
            response.raise_for_status()
            return response.json()

    async def get_pull_request_files(
        self,
        repo: str,
        pr_number: int,
    ) -> list[dict[str, Any]]:
        """Get files changed in PR."""
        import httpx

        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._get_headers())
            response.raise_for_status()
            return response.json()


class CodeReviewer:
    """
    Automated code review using LLM.
    """

    def __init__(self, llm_adapter):
        self.llm = llm_adapter

    async def review_code(
        self,
        code: str,
        file_path: str,
        context: str | None = None,
    ) -> list[CodeReviewComment]:
        """
        Review code and generate comments.

        Args:
            code: Code to review
            file_path: Path to file (for context)
            context: Additional context

        Returns:
            List of review comments
        """
        prompt = f"""Review this code and identify issues:

File: {file_path}

```python
{code}
```

Context: {context or "None"}

Look for:
1. Security issues (SQL injection, XSS, unsafe eval, etc.)
2. Performance problems
3. Code smells
4. Style violations
5. Missing error handling
6. Type safety issues

Output as JSON array of comments:
[
  {{
    "line": 5,
    "message": "Description of issue",
    "severity": "error|warning|info",
    "suggestion": "How to fix it"
  }}
]

Be specific and actionable."""

        try:
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "line": {"type": "integer"},
                        "message": {"type": "string"},
                        "severity": {"type": "string", "enum": ["error", "warning", "info"]},
                        "suggestion": {"type": "string"},
                    },
                    "required": ["line", "message", "severity"],
                },
            }

            result = await self.llm.generate_structured(prompt, schema, temperature=0.2)

            comments = []
            for item in result:
                comments.append(CodeReviewComment(
                    path=file_path,
                    line=item.get("line", 1),
                    message=item["message"],
                    severity=item["severity"],
                    suggestion=item.get("suggestion"),
                ))

            return comments

        except Exception as e:
            logger.error("Code review failed", error=str(e))
            return []

    async def review_diff(
        self,
        diffs: list[FileDiff],
        repo_path: Path,
    ) -> dict[str, list[CodeReviewComment]]:
        """Review a set of diffs."""
        all_comments = {}

        for diff in diffs:
            if diff.status == "deleted":
                continue

            file_path = repo_path / diff.path

            if not file_path.exists():
                continue

            try:
                code = file_path.read_text()
                comments = await self.review_code(code, diff.path)
                if comments:
                    all_comments[diff.path] = comments
            except Exception as e:
                logger.error("Failed to review file", path=diff.path, error=str(e))

        return all_comments


class AutoCommitter:
    """
    Automatically commit changes with intelligent messages.
    """

    def __init__(self, repo: GitRepository, llm_adapter=None):
        self.repo = repo
        self.llm = llm_adapter

    async def commit_changes(
        self,
        paths: list[str],
        custom_message: str | None = None,
    ) -> str | None:
        """
        Commit changes with automatic message generation.

        Args:
            paths: Files to commit
            custom_message: Optional custom message

        Returns:
            Commit hash or None if no changes
        """
        # Stage files
        await self.repo.add(paths)

        # Get diff
        diffs = await self.repo.diff(staged=True)

        if not diffs:
            logger.info("No changes to commit")
            return None

        # Generate message if not provided
        if not custom_message:
            message = await self._generate_commit_message(diffs)
        else:
            message = custom_message

        # Create commit
        commit_hash = await self.repo.commit(message)

        return commit_hash

    async def _generate_commit_message(self, diffs: list[FileDiff]) -> str:
        """Generate commit message from diffs."""
        if not self.llm:
            # Simple message
            files = [d.path for d in diffs[:3]]
            return f"Update {', '.join(files)}"

        # Use LLM for better messages
        prompt = f"""Generate a git commit message for these changes:

Files changed:
{chr(10).join(f"- {d.path} (+{d.additions} -{d.deletions})" for d in diffs[:10])}

Write a concise commit message (50 chars or less) describing the changes.
Follow conventional commits format if applicable.

Message:"""

        try:
            message = await self.llm.generate(prompt, temperature=0.3)
            return message.strip()[:72]  # Limit length
        except:
            files = [d.path for d in diffs[:3]]
            return f"Update {', '.join(files)}"

    async def commit_capability(
        self,
        capability_id: str,
        capability_name: str,
    ) -> str | None:
        """Commit a generated capability."""
        message = f"Add capability: {capability_name}\n\nGenerated by Lya self-improvement system.\nCapability ID: {capability_id}"

        return await self.commit_changes(
            ["."],  # All changes
            message,
        )


def get_git_capability_code() -> str:
    """Generate the git integration capability code."""
    return '''"""Git Integration Capability.

Clone repositories, create branches, commit changes, and manage code.
"""

from pathlib import Path
from typing import Any
import subprocess

class GitRepo:
    """Simple git repository wrapper."""

    def __init__(self, path: str):
        self.path = Path(path)

    async def commit(self, message: str) -> str:
        """Stage all and commit."""
        subprocess.run(["git", "add", "."], cwd=self.path, check=True)
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.path,
            check=True,
            capture_output=True,
            text=True
        )
        # Get commit hash
        hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.path,
            capture_output=True,
            text=True
        )
        return hash_result.stdout.strip()

    async def create_branch(self, name: str) -> None:
        """Create and checkout new branch."""
        subprocess.run(
            ["git", "checkout", "-b", name],
            cwd=self.path,
            check=True
        )

    async def push(self, remote: str = "origin", branch: str | None = None) -> None:
        """Push to remote."""
        cmd = ["git", "push", remote]
        if branch:
            cmd.append(branch)
        subprocess.run(cmd, cwd=self.path, check=True)

async def clone_repository(url: str, destination: str) -> GitRepo:
    """Clone a git repository."""
    import subprocess

    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "clone", url, str(dest)],
        check=True
    )

    return GitRepo(str(dest))

async def init_repository(path: str) -> GitRepo:
    """Initialize a new git repository."""
    repo_path = Path(path)
    repo_path.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "init"],
        cwd=repo_path,
        check=True
    )

    return GitRepo(str(repo_path))
'''
