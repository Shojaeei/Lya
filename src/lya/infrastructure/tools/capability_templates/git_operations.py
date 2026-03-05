"""
Capability: Git Operations
Description: Comprehensive git operations including repository management,
             branching, committing, and remote operations.
Tags: git, vcs, repository, version-control
"""

from pathlib import Path
from dataclasses import dataclass

from lya.infrastructure.tools.git import GitAdapter, GitConfig, GitCommit, GitBranch


@dataclass
class GitOperationResult:
    """Result of a git operation."""
    success: bool
    message: str
    data: dict = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


class GitOperationsCapability:
    """
    Git operations capability for repository management.

    Provides safe, controlled access to git functionality with
    workspace isolation and validation.
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self._adapter: GitAdapter | None = None

    async def __aenter__(self):
        self._adapter = GitAdapter(workspace=self.workspace)
        await self._adapter.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._adapter:
            await self._adapter.__aexit__(exc_type, exc_val, exc_tb)

    # ═══════════════════════════════════════════════════════════════
    # Repository Operations
    # ═══════════════════════════════════════════════════════════════

    async def init_repository(self, name: str) -> GitOperationResult:
        """Initialize a new git repository."""
        try:
            repo_path = self.workspace / name
            await self._adapter.init(repo_path)
            return GitOperationResult(
                success=True,
                message=f"Repository initialized at {repo_path}",
                data={"path": str(repo_path)}
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to initialize repository: {e}"
            )

    async def clone_repository(
        self,
        url: str,
        destination: str | None = None,
        branch: str | None = None
    ) -> GitOperationResult:
        """Clone a remote repository."""
        try:
            dest_path = await self._adapter.clone(url, destination, branch)
            return GitOperationResult(
                success=True,
                message=f"Repository cloned to {dest_path}",
                data={"path": str(dest_path)}
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to clone repository: {e}"
            )

    async def get_status(self, repo_path: Path | None = None) -> GitOperationResult:
        """Get repository status."""
        try:
            status = await self._adapter.status(repo_path)
            return GitOperationResult(
                success=True,
                message=f"Status for {status.branch}",
                data={
                    "branch": status.branch,
                    "is_clean": status.is_clean,
                    "modified": status.modified,
                    "staged": status.staged,
                    "untracked": status.untracked,
                }
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to get status: {e}"
            )

    # ═══════════════════════════════════════════════════════════════
    # Branch Operations
    # ═══════════════════════════════════════════════════════════════

    async def create_branch(
        self,
        branch_name: str,
        from_branch: str = "main",
        repo_path: Path | None = None
    ) -> GitOperationResult:
        """Create a new branch."""
        try:
            branch = await self._adapter.create_branch(branch_name, from_branch, repo_path)
            return GitOperationResult(
                success=True,
                message=f"Branch '{branch.name}' created",
                data={
                    "name": branch.name,
                    "is_current": branch.is_current,
                    "tracking": branch.tracking,
                }
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to create branch: {e}"
            )

    async def checkout_branch(
        self,
        branch_name: str,
        repo_path: Path | None = None
    ) -> GitOperationResult:
        """Checkout a branch."""
        try:
            await self._adapter.checkout(branch_name, repo_path)
            return GitOperationResult(
                success=True,
                message=f"Switched to branch '{branch_name}'",
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to checkout branch: {e}"
            )

    async def list_branches(self, repo_path: Path | None = None) -> GitOperationResult:
        """List all branches."""
        try:
            branches = await self._adapter.list_branches(repo_path)
            return GitOperationResult(
                success=True,
                message=f"Found {len(branches)} branches",
                data={
                    "branches": [
                        {
                            "name": b.name,
                            "is_current": b.is_current,
                            "tracking": b.tracking,
                        }
                        for b in branches
                    ]
                }
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to list branches: {e}"
            )

    async def merge_branch(
        self,
        branch_name: str,
        message: str | None = None,
        repo_path: Path | None = None
    ) -> GitOperationResult:
        """Merge a branch into current branch."""
        try:
            await self._adapter.merge(branch_name, message, repo_path)
            return GitOperationResult(
                success=True,
                message=f"Merged branch '{branch_name}'",
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to merge branch: {e}"
            )

    # ═══════════════════════════════════════════════════════════════
    # Commit Operations
    # ═══════════════════════════════════════════════════════════════

    async def stage_files(
        self,
        files: list[str],
        repo_path: Path | None = None
    ) -> GitOperationResult:
        """Stage files for commit."""
        try:
            await self._adapter.add(files, repo_path)
            return GitOperationResult(
                success=True,
                message=f"Staged {len(files)} files",
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to stage files: {e}"
            )

    async def commit_changes(
        self,
        message: str,
        repo_path: Path | None = None
    ) -> GitOperationResult:
        """Create a commit."""
        try:
            commit = await self._adapter.commit(message, repo_path)
            return GitOperationResult(
                success=True,
                message=f"Committed: {commit.message}",
                data={
                    "hash": commit.hash,
                    "message": commit.message,
                    "author": commit.author,
                    "timestamp": commit.timestamp,
                }
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to commit: {e}"
            )

    async def get_commit_history(
        self,
        limit: int = 10,
        repo_path: Path | None = None
    ) -> GitOperationResult:
        """Get commit history."""
        try:
            commits = await self._adapter.get_commit_history(limit, repo_path)
            return GitOperationResult(
                success=True,
                message=f"Retrieved {len(commits)} commits",
                data={
                    "commits": [
                        {
                            "hash": c.hash,
                            "message": c.message,
                            "author": c.author,
                            "timestamp": c.timestamp,
                        }
                        for c in commits
                    ]
                }
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to get history: {e}"
            )

    # ═══════════════════════════════════════════════════════════════
    # Remote Operations
    # ═══════════════════════════════════════════════════════════════

    async def push_branch(
        self,
        branch: str | None = None,
        remote: str = "origin",
        set_upstream: bool = False,
        repo_path: Path | None = None
    ) -> GitOperationResult:
        """Push branch to remote."""
        try:
            await self._adapter.push(branch, remote, set_upstream, repo_path)
            return GitOperationResult(
                success=True,
                message=f"Pushed to {remote}",
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to push: {e}"
            )

    async def pull_changes(
        self,
        branch: str | None = None,
        remote: str = "origin",
        rebase: bool = False,
        repo_path: Path | None = None
    ) -> GitOperationResult:
        """Pull changes from remote."""
        try:
            await self._adapter.pull(branch, remote, rebase, repo_path)
            return GitOperationResult(
                success=True,
                message=f"Pulled from {remote}",
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to pull: {e}"
            )

    async def fetch_remote(
        self,
        remote: str = "origin",
        repo_path: Path | None = None
    ) -> GitOperationResult:
        """Fetch from remote."""
        try:
            await self._adapter.fetch(remote, repo_path)
            return GitOperationResult(
                success=True,
                message=f"Fetched from {remote}",
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to fetch: {e}"
            )

    # ═══════════════════════════════════════════════════════════════
    # Utility Operations
    # ═══════════════════════════════════════════════════════════════

    async def get_diff(
        self,
        target: str,
        source: str | None = None,
        repo_path: Path | None = None
    ) -> GitOperationResult:
        """Get diff between commits/branches."""
        try:
            diff = await self._adapter.get_diff(target, source, repo_path)
            return GitOperationResult(
                success=True,
                message="Diff retrieved",
                data={"diff": diff}
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to get diff: {e}"
            )

    async def stash_changes(
        self,
        message: str | None = None,
        include_untracked: bool = False,
        repo_path: Path | None = None
    ) -> GitOperationResult:
        """Stash current changes."""
        try:
            await self._adapter.stash(message, include_untracked, repo_path)
            return GitOperationResult(
                success=True,
                message="Changes stashed",
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to stash: {e}"
            )

    async def unstash_changes(
        self,
        stash_ref: str = "stash@{0}",
        repo_path: Path | None = None
    ) -> GitOperationResult:
        """Apply stashed changes."""
        try:
            await self._adapter.unstash(stash_ref, repo_path)
            return GitOperationResult(
                success=True,
                message="Stash applied",
            )
        except Exception as e:
            return GitOperationResult(
                success=False,
                message=f"Failed to unstash: {e}"
            )


# ═══════════════════════════════════════════════════════════════
# Capability Metadata
# ═══════════════════════════════════════════════════════════════

CAPABILITY_METADATA = {
    "name": "git_operations",
    "version": "1.0.0",
    "description": "Git repository management operations",
    "author": "Lya",
    "tags": ["git", "vcs", "repository", "version-control"],
    "dependencies": ["GitAdapter"],
    "requires_config": ["workspace"],
    "entry_point": "GitOperationsCapability",
    "methods": [
        "init_repository",
        "clone_repository",
        "get_status",
        "create_branch",
        "checkout_branch",
        "list_branches",
        "merge_branch",
        "stage_files",
        "commit_changes",
        "get_commit_history",
        "push_branch",
        "pull_changes",
        "fetch_remote",
        "get_diff",
        "stash_changes",
        "unstash_changes",
    ],
}
