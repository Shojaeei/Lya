"""Git integration tools."""

from .git_adapter import (
    GitAdapter,
    Commit,
    Diff,
    Branch,
    get_git_adapter,
)
from .github_adapter import (
    GitHubAdapter,
    PullRequest,
    ReviewComment,
    CodeReviewService,
)
from .gitlab_adapter import (
    GitLabAdapter,
    GitLabMergeRequest,
)

__all__ = [
    # Git
    "GitAdapter",
    "Commit",
    "Diff",
    "Branch",
    "get_git_adapter",
    # GitHub
    "GitHubAdapter",
    "PullRequest",
    "ReviewComment",
    "CodeReviewService",
    # GitLab
    "GitLabAdapter",
    "GitLabMergeRequest",
]
