"""GitHub API Adapter.

Provides GitHub API operations for pull requests, reviews, and issues.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


@dataclass
class PullRequest:
    """Pull request information."""
    number: int
    title: str
    body: str
    branch: str
    base: str
    state: Literal["open", "closed", "merged"]
    url: str
    author: str
    created_at: str


@dataclass
class ReviewComment:
    """Code review comment."""
    path: str
    line: int
    body: str
    side: Literal["LEFT", "RIGHT"] = "RIGHT"


class GitHubAdapter:
    """
    Adapter for GitHub API operations.

    Supports:
    - Pull requests (create, list, merge)
    - Code reviews (comments, approve)
    - Issues (create, update, list)
    - Repository info
    """

    API_BASE = "https://api.github.com"

    def __init__(self, token: str | None = None):
        if not HAS_HTTPX:
            raise ImportError("httpx required. Run: pip install httpx")

        self.token = token or os.environ.get("GITHUB_TOKEN") or settings.get("github_token")
        if not self.token:
            logger.warning("GitHub token not set. Set GITHUB_TOKEN environment variable.")

        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "GitHubAdapter":
        """Async context manager entry."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Lya-Agent/1.0",
        }

        if self.token:
            headers["Authorization"] = f"token {self.token}"

        self._client = httpx.AsyncClient(
            base_url=self.API_BASE,
            headers=headers,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if self._client is None:
            raise RuntimeError("GitHubAdapter not in async context")
        return self._client

    def _parse_repo_url(self, url: str) -> tuple[str, str]:
        """Parse owner and repo from URL."""
        # Handle: https://github.com/owner/repo or git@github.com:owner/repo.git
        if url.startswith("git@github.com:"):
            match = re.match(r"git@github\.com:(.+)/(.+)\.git?", url)
            if match:
                return match.group(1), match.group(2)
        else:
            parsed = urlparse(url)
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2:
                return parts[0], parts[1].replace(".git", "")

        raise ValueError(f"Could not parse GitHub URL: {url}")

    # ═══════════════════════════════════════════════════════════════
    # Pull Requests
    # ═══════════════════════════════════════════════════════════════

    async def create_pull_request(
        self,
        repo_url: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        draft: bool = False,
    ) -> PullRequest:
        """
        Create a pull request.

        Args:
            repo_url: Repository URL
            title: PR title
            body: PR description
            head_branch: Branch with changes
            base_branch: Target branch
            draft: Create as draft PR

        Returns:
            Created pull request
        """
        owner, repo = self._parse_repo_url(repo_url)

        data = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch,
            "draft": draft,
        }

        try:
            response = await self._get_client().post(
                f"/repos/{owner}/{repo}/pulls",
                json=data,
            )
            response.raise_for_status()

            result = response.json()

            logger.info(
                "Pull request created",
                repo=f"{owner}/{repo}",
                number=result["number"],
                title=title[:50],
            )

            return PullRequest(
                number=result["number"],
                title=result["title"],
                body=result["body"] or "",
                branch=result["head"]["ref"],
                base=result["base"]["ref"],
                state=result["state"],
                url=result["html_url"],
                author=result["user"]["login"],
                created_at=result["created_at"],
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                "Failed to create PR",
                error=e.response.text,
                status=e.response.status_code,
            )
            raise RuntimeError(f"GitHub API error: {e.response.text}")

    async def list_pull_requests(
        self,
        repo_url: str,
        state: Literal["open", "closed", "all"] = "open",
    ) -> list[PullRequest]:
        """List pull requests."""
        owner, repo = self._parse_repo_url(repo_url)

        response = await self._get_client().get(
            f"/repos/{owner}/{repo}/pulls",
            params={"state": state},
        )
        response.raise_for_status()

        return [
            PullRequest(
                number=pr["number"],
                title=pr["title"],
                body=pr["body"] or "",
                branch=pr["head"]["ref"],
                base=pr["base"]["ref"],
                state=pr["state"],
                url=pr["html_url"],
                author=pr["user"]["login"],
                created_at=pr["created_at"],
            )
            for pr in response.json()
        ]

    async def get_pull_request(self, repo_url: str, number: int) -> PullRequest:
        """Get a specific pull request."""
        owner, repo = self._parse_repo_url(repo_url)

        response = await self._get_client().get(
            f"/repos/{owner}/{repo}/pulls/{number}",
        )
        response.raise_for_status()

        pr = response.json()
        return PullRequest(
            number=pr["number"],
            title=pr["title"],
            body=pr["body"] or "",
            branch=pr["head"]["ref"],
            base=pr["base"]["ref"],
            state=pr["state"],
            url=pr["html_url"],
            author=pr["user"]["login"],
            created_at=pr["created_at"],
        )

    async def merge_pull_request(
        self,
        repo_url: str,
        number: int,
        commit_title: str | None = None,
        commit_message: str | None = None,
        method: Literal["merge", "squash", "rebase"] = "merge",
    ) -> dict[str, Any]:
        """Merge a pull request."""
        owner, repo = self._parse_repo_url(repo_url)

        data = {"merge_method": method}
        if commit_title:
            data["commit_title"] = commit_title
        if commit_message:
            data["commit_message"] = commit_message

        response = await self._get_client().put(
            f"/repos/{owner}/{repo}/pulls/{number}/merge",
            json=data,
        )
        response.raise_for_status()

        logger.info("Pull request merged", repo=f"{owner}/{repo}", number=number)

        return response.json()

    async def update_pull_request(
        self,
        repo_url: str,
        number: int,
        title: str | None = None,
        body: str | None = None,
        state: Literal["open", "closed"] | None = None,
    ) -> PullRequest:
        """Update a pull request."""
        owner, repo = self._parse_repo_url(repo_url)

        data: dict[str, Any] = {}
        if title:
            data["title"] = title
        if body:
            data["body"] = body
        if state:
            data["state"] = state

        response = await self._get_client().patch(
            f"/repos/{owner}/{repo}/pulls/{number}",
            json=data,
        )
        response.raise_for_status()

        pr = response.json()
        return PullRequest(
            number=pr["number"],
            title=pr["title"],
            body=pr["body"] or "",
            branch=pr["head"]["ref"],
            base=pr["base"]["ref"],
            state=pr["state"],
            url=pr["html_url"],
            author=pr["user"]["login"],
            created_at=pr["created_at"],
        )

    # ═══════════════════════════════════════════════════════════════
    # Code Reviews
    # ═══════════════════════════════════════════════════════════════

    async def create_review(
        self,
        repo_url: str,
        pr_number: int,
        body: str,
        event: Literal["APPROVE", "REQUEST_CHANGES", "COMMENT"] = "COMMENT",
        comments: list[ReviewComment] | None = None,
    ) -> dict[str, Any]:
        """
        Create a code review on a pull request.

        Args:
            repo_url: Repository URL
            pr_number: Pull request number
            body: Review body text
            event: Review action (APPROVE, REQUEST_CHANGES, COMMENT)
            comments: Line-specific comments

        Returns:
            Review result
        """
        owner, repo = self._parse_repo_url(repo_url)

        data: dict[str, Any] = {
            "body": body,
            "event": event,
        }

        if comments:
            data["comments"] = [
                {
                    "path": c.path,
                    "line": c.line,
                    "body": c.body,
                    "side": c.side,
                }
                for c in comments
            ]

        response = await self._get_client().post(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
            json=data,
        )
        response.raise_for_status()

        logger.info(
            "Review created",
            repo=f"{owner}/{repo}",
            pr=pr_number,
            event=event,
        )

        return response.json()

    async def submit_review(
        self,
        repo_url: str,
        pr_number: int,
        review_id: int,
        event: Literal["APPROVE", "REQUEST_CHANGES", "COMMENT"],
        body: str | None = None,
    ) -> dict[str, Any]:
        """Submit a pending review."""
        owner, repo = self._parse_repo_url(repo_url)

        data = {"event": event}
        if body:
            data["body"] = body

        response = await self._get_client().post(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews/{review_id}/events",
            json=data,
        )
        response.raise_for_status()

        return response.json()

    async def list_reviews(self, repo_url: str, pr_number: int) -> list[dict[str, Any]]:
        """List reviews on a pull request."""
        owner, repo = self._parse_repo_url(repo_url)

        response = await self._get_client().get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
        )
        response.raise_for_status()

        return response.json()

    # ═══════════════════════════════════════════════════════════════
    # Issues
    # ═══════════════════════════════════════════════════════════════

    async def create_issue(
        self,
        repo_url: str,
        title: str,
        body: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create an issue."""
        owner, repo = self._parse_repo_url(repo_url)

        data: dict[str, Any] = {"title": title}
        if body:
            data["body"] = body
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees

        response = await self._get_client().post(
            f"/repos/{owner}/{repo}/issues",
            json=data,
        )
        response.raise_for_status()

        logger.info(
            "Issue created",
            repo=f"{owner}/{repo}",
            title=title[:50],
        )

        return response.json()

    async def list_issues(
        self,
        repo_url: str,
        state: Literal["open", "closed", "all"] = "open",
        labels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List issues."""
        owner, repo = self._parse_repo_url(repo_url)

        params: dict[str, Any] = {"state": state}
        if labels:
            params["labels"] = ",".join(labels)

        response = await self._get_client().get(
            f"/repos/{owner}/{repo}/issues",
            params=params,
        )
        response.raise_for_status()

        return response.json()

    async def update_issue(
        self,
        repo_url: str,
        number: int,
        title: str | None = None,
        body: str | None = None,
        state: Literal["open", "closed"] | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update an issue."""
        owner, repo = self._parse_repo_url(repo_url)

        data: dict[str, Any] = {}
        if title:
            data["title"] = title
        if body:
            data["body"] = body
        if state:
            data["state"] = state
        if labels:
            data["labels"] = labels

        response = await self._get_client().patch(
            f"/repos/{owner}/{repo}/issues/{number}",
            json=data,
        )
        response.raise_for_status()

        return response.json()

    # ═══════════════════════════════════════════════════════════════
    # Repository Info
    # ═══════════════════════════════════════════════════════════════

    async def get_repository(self, repo_url: str) -> dict[str, Any]:
        """Get repository information."""
        owner, repo = self._parse_repo_url(repo_url)

        response = await self._get_client().get(f"/repos/{owner}/{repo}")
        response.raise_for_status()

        return response.json()

    async def get_rate_limit(self) -> dict[str, Any]:
        """Get current rate limit status."""
        response = await self._get_client().get("/rate_limit")
        response.raise_for_status()

        return response.json()

    # ═══════════════════════════════════════════════════════════════
    # Utility Methods
    # ═══════════════════════════════════════════════════════════════

    async def create_pr_from_changes(
        self,
        repo_url: str,
        branch_name: str,
        title: str,
        description: str,
        base_branch: str = "main",
    ) -> PullRequest:
        """
        Complete workflow: Create branch, push, create PR.

        Note: This requires local git operations to be done first.
        """
        # The actual git operations (branch, commit, push) should be done
        # with GitAdapter before calling this method

        return await self.create_pull_request(
            repo_url=repo_url,
            title=title,
            body=description,
            head_branch=branch_name,
            base_branch=base_branch,
        )

    async def generate_pr_description(
        self,
        commits: list[str],
        files_changed: list[str],
    ) -> str:
        """Generate PR description from changes."""
        # This could use LLM to generate descriptions
        description = "## Changes\n\n"

        if commits:
            description += "### Commits\n"
            for commit in commits:
                description += f"- {commit}\n"
            description += "\n"

        if files_changed:
            description += "### Files Changed\n"
            for f in files_changed:
                description += f"- `{f}`\n"
            description += "\n"

        description += "---\n*This PR was created by Lya*"

        return description


class GitLabAdapter:
    """GitLab API Adapter (similar structure to GitHub)."""

    API_BASE = "https://gitlab.com/api/v4"

    def __init__(self, token: str | None = None, base_url: str | None = None):
        self.token = token or os.environ.get("GITLAB_TOKEN")
        self.base_url = base_url or self.API_BASE

    # Similar methods for GitLab API
    # Implementation omitted for brevity


class CodeReviewService:
    """Service for automated code reviews."""

    def __init__(self, github: GitHubAdapter, llm: Any):
        self.github = github
        self.llm = llm

    async def review_pr(
        self,
        repo_url: str,
        pr_number: int,
        diff: str,
    ) -> dict[str, Any]:
        """
        Automated code review using LLM.

        Args:
            repo_url: Repository URL
            pr_number: PR number
            diff: Diff content

        Returns:
            Review result
        """
        # Analyze diff with LLM
        analysis = await self._analyze_diff(diff)

        # Create review comments
        comments = []
        for issue in analysis.get("issues", []):
            comments.append(ReviewComment(
                path=issue["file"],
                line=issue["line"],
                body=issue["comment"],
            ))

        # Submit review
        event = "REQUEST_CHANGES" if analysis.get("has_issues") else "APPROVE"

        review = await self.github.create_review(
            repo_url=repo_url,
            pr_number=pr_number,
            body=analysis.get("summary", ""),
            event=event,
            comments=comments if comments else None,
        )

        return review

    async def _analyze_diff(self, diff: str) -> dict[str, Any]:
        """Analyze diff using LLM."""
        # This would call LLM to analyze code
        # Placeholder implementation
        return {
            "has_issues": False,
            "issues": [],
            "summary": "Code looks good!",
        }

    async def generate_pr_description(
        self,
        repo_url: str,
        pr_number: int,
    ) -> str:
        """Generate PR description from commits and changed files."""
        # Get PR details
        pr = await self.github.get_pull_request(repo_url, pr_number)

        # Get commits
        owner, repo = self.github._parse_repo_url(repo_url)
        response = await self.github._get_client().get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/commits"
        )
        response.raise_for_status()
        commits = response.json()

        # Get changed files
        response = await self.github._get_client().get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/files"
        )
        response.raise_for_status()
        files = response.json()

        # Generate description
        description = "## Summary\n\n"
        description += "This PR includes the following changes:\n\n"

        if commits:
            description += "### Commits\n\n"
            for commit in commits:
                msg = commit["commit"]["message"].split("\n")[0]  # First line only
                description += f"- {msg}\n"
            description += "\n"

        if files:
            description += "### Files Changed\n\n"
            for f in files:
                status = f["status"]
                changes = f"(+{f['additions']}/-{f['deletions']})"
                description += f"- `{f['filename']}` ({status}) {changes}\n"
            description += "\n"

        description += "---\n\n"
        description += "🤖 *This description was generated by Lya*"

        return description

    async def suggest_reviewers(
        self,
        repo_url: str,
        pr_number: int,
    ) -> list[str]:
        """Suggest reviewers based on code changes."""
        # Get changed files
        owner, repo = self.github._parse_repo_url(repo_url)
        response = await self.github._get_client().get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/files"
        )
        response.raise_for_status()
        files = response.json()

        # Get file paths
        file_paths = [f["filename"] for f in files]

        # Get contributors for these files
        contributors: dict[str, int] = {}
        for path in file_paths[:5]:  # Limit to first 5 files
            try:
                response = await self.github._get_client().get(
                    f"/repos/{owner}/{repo}/commits",
                    params={"path": path, "per_page": 10}
                )
                response.raise_for_status()
                commits = response.json()

                for commit in commits:
                    author = commit.get("author", {}).get("login")
                    if author:
                        contributors[author] = contributors.get(author, 0) + 1
            except Exception:
                continue

        # Sort by contribution count
        sorted_reviewers = sorted(
            contributors.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [r[0] for r in sorted_reviewers[:3]]  # Top 3 reviewers

    async def check_pr_quality(
        self,
        repo_url: str,
        pr_number: int,
    ) -> dict[str, Any]:
        """Check PR quality metrics."""
        # Get PR details
        pr = await self.github.get_pull_request(repo_url, pr_number)

        # Get changed files
        owner, repo = self.github._parse_repo_url(repo_url)
        response = await self.github._get_client().get(
            f"/repos/{owner}/{repo}/pulls/{pr_number}/files"
        )
        response.raise_for_status()
        files = response.json()

        total_additions = sum(f.get("additions", 0) for f in files)
        total_deletions = sum(f.get("deletions", 0) for f in files)
        total_changes = total_additions + total_deletions

        # Check description quality
        description = pr.body or ""
        has_description = len(description.strip()) > 50
        has_tests = any("test" in f["filename"].lower() for f in files)
        has_docs = any(
            any(d in f["filename"].lower() for d in ["readme", "docs", ".md"])
            for f in files
        )

        # Calculate score
        score = 0
        checks = {
            "has_description": has_description,
            "reasonable_size": total_changes < 500,
            "has_tests": has_tests,
            "has_documentation": has_docs,
        }

        for check, passed in checks.items():
            if passed:
                score += 25

        return {
            "score": score,
            "grade": "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 50 else "D",
            "total_changes": total_changes,
            "files_changed": len(files),
            "checks": checks,
            "recommendations": [
                "Add PR description" if not has_description else None,
                "Consider splitting large PR" if total_changes >= 500 else None,
                "Add tests" if not has_tests else None,
                "Add documentation" if not has_docs else None,
            ],
        }
