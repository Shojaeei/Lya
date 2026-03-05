"""GitLab API Adapter.

Provides GitLab API operations for merge requests and issues.
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
class GitLabMergeRequest:
    """GitLab merge request information."""
    iid: int
    title: str
    description: str
    source_branch: str
    target_branch: str
    state: Literal["opened", "closed", "merged", "locked"]
    web_url: str
    author: str
    created_at: str
    labels: list[str]


class GitLabAdapter:
    """
    Adapter for GitLab API operations.

    Supports:
    - Merge requests (create, list, merge)
    - Issues (create, update, list)
    - Repository info
    - CI/CD pipelines
    """

    DEFAULT_BASE = "https://gitlab.com/api/v4"

    def __init__(self, token: str | None = None, base_url: str | None = None):
        if not HAS_HTTPX:
            raise ImportError("httpx required. Run: pip install httpx")

        self.token = token or os.environ.get("GITLAB_TOKEN") or settings.get("gitlab_token")
        self.base_url = base_url or settings.get("gitlab_base_url") or self.DEFAULT_BASE

        if not self.token:
            logger.warning("GitLab token not set. Set GITLAB_TOKEN environment variable.")

        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "GitLabAdapter":
        """Async context manager entry."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "Lya-Agent/1.0",
        }

        if self.token:
            headers["PRIVATE-TOKEN"] = self.token

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
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
            raise RuntimeError("GitLabAdapter not in async context")
        return self._client

    def _parse_repo_url(self, url: str) -> tuple[str, str]:
        """Parse namespace and project from GitLab URL."""
        # Handle: https://gitlab.com/namespace/project or git@gitlab.com:namespace/project.git
        if url.startswith("git@gitlab.com:"):
            match = re.match(r"git@gitlab\.com:(.+)/(.+)\.git?", url)
            if match:
                return match.group(1), match.group(2)
        else:
            parsed = urlparse(url)
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2:
                return "/".join(parts[:-1]), parts[-1].replace(".git", "")

        raise ValueError(f"Could not parse GitLab URL: {url}")

    def _encode_project_path(self, namespace: str, project: str) -> str:
        """Encode project path for URL."""
        return f"{namespace}%2F{project}"

    # ═══════════════════════════════════════════════════════════════
    # Merge Requests
    # ═══════════════════════════════════════════════════════════════

    async def create_merge_request(
        self,
        repo_url: str,
        title: str,
        description: str,
        source_branch: str,
        target_branch: str = "main",
        draft: bool = False,
        labels: list[str] | None = None,
    ) -> GitLabMergeRequest:
        """
        Create a merge request.

        Args:
            repo_url: Repository URL
            title: MR title
            description: MR description
            source_branch: Source branch
            target_branch: Target branch
            draft: Create as draft MR
            labels: List of labels

        Returns:
            Created merge request
        """
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        data: dict[str, Any] = {
            "title": f"Draft: {title}" if draft else title,
            "description": description,
            "source_branch": source_branch,
            "target_branch": target_branch,
        }

        if labels:
            data["labels"] = ",".join(labels)

        try:
            response = await self._get_client().post(
                f"/projects/{project_path}/merge_requests",
                json=data,
            )
            response.raise_for_status()

            result = response.json()

            logger.info(
                "Merge request created",
                project=f"{namespace}/{project}",
                iid=result["iid"],
                title=title[:50],
            )

            return GitLabMergeRequest(
                iid=result["iid"],
                title=result["title"],
                description=result["description"] or "",
                source_branch=result["source_branch"],
                target_branch=result["target_branch"],
                state=result["state"],
                web_url=result["web_url"],
                author=result["author"]["username"],
                created_at=result["created_at"],
                labels=result.get("labels", []),
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                "Failed to create MR",
                error=e.response.text,
                status=e.response.status_code,
            )
            raise RuntimeError(f"GitLab API error: {e.response.text}")

    async def list_merge_requests(
        self,
        repo_url: str,
        state: Literal["opened", "closed", "locked", "merged", "all"] = "opened",
    ) -> list[GitLabMergeRequest]:
        """List merge requests."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        response = await self._get_client().get(
            f"/projects/{project_path}/merge_requests",
            params={"state": state},
        )
        response.raise_for_status()

        return [
            GitLabMergeRequest(
                iid=mr["iid"],
                title=mr["title"],
                description=mr["description"] or "",
                source_branch=mr["source_branch"],
                target_branch=mr["target_branch"],
                state=mr["state"],
                web_url=mr["web_url"],
                author=mr["author"]["username"],
                created_at=mr["created_at"],
                labels=mr.get("labels", []),
            )
            for mr in response.json()
        ]

    async def get_merge_request(self, repo_url: str, iid: int) -> GitLabMergeRequest:
        """Get a specific merge request."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        response = await self._get_client().get(
            f"/projects/{project_path}/merge_requests/{iid}",
        )
        response.raise_for_status()

        mr = response.json()
        return GitLabMergeRequest(
            iid=mr["iid"],
            title=mr["title"],
            description=mr["description"] or "",
            source_branch=mr["source_branch"],
            target_branch=mr["target_branch"],
            state=mr["state"],
            web_url=mr["web_url"],
            author=mr["author"]["username"],
            created_at=mr["created_at"],
            labels=mr.get("labels", []),
        )

    async def accept_merge_request(
        self,
        repo_url: str,
        iid: int,
        commit_message: str | None = None,
        should_remove_source_branch: bool = True,
    ) -> dict[str, Any]:
        """Merge a merge request."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        data: dict[str, Any] = {}
        if commit_message:
            data["merge_commit_message"] = commit_message
        data["should_remove_source_branch"] = should_remove_source_branch

        response = await self._get_client().put(
            f"/projects/{project_path}/merge_requests/{iid}/merge",
            json=data,
        )
        response.raise_for_status()

        logger.info("Merge request merged", project=f"{namespace}/{project}", iid=iid)

        return response.json()

    async def update_merge_request(
        self,
        repo_url: str,
        iid: int,
        title: str | None = None,
        description: str | None = None,
        state: Literal["opened", "closed"] | None = None,
        labels: list[str] | None = None,
    ) -> GitLabMergeRequest:
        """Update a merge request."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        data: dict[str, Any] = {}
        if title:
            data["title"] = title
        if description:
            data["description"] = description
        if state:
            data["state_event"] = state
        if labels:
            data["labels"] = ",".join(labels)

        response = await self._get_client().put(
            f"/projects/{project_path}/merge_requests/{iid}",
            json=data,
        )
        response.raise_for_status()

        mr = response.json()
        return GitLabMergeRequest(
            iid=mr["iid"],
            title=mr["title"],
            description=mr["description"] or "",
            source_branch=mr["source_branch"],
            target_branch=mr["target_branch"],
            state=mr["state"],
            web_url=mr["web_url"],
            author=mr["author"]["username"],
            created_at=mr["created_at"],
            labels=mr.get("labels", []),
        )

    # ═══════════════════════════════════════════════════════════════
    # Issues
    # ═══════════════════════════════════════════════════════════════

    async def create_issue(
        self,
        repo_url: str,
        title: str,
        description: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
        milestone: str | None = None,
    ) -> dict[str, Any]:
        """Create an issue."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        data: dict[str, Any] = {"title": title}
        if description:
            data["description"] = description
        if labels:
            data["labels"] = ",".join(labels)
        if assignees:
            # GitLab uses assignee_ids
            assignee_ids = []
            for username in assignees:
                user = await self._get_user_id(username)
                if user:
                    assignee_ids.append(user)
            if assignee_ids:
                data["assignee_ids"] = assignee_ids

        response = await self._get_client().post(
            f"/projects/{project_path}/issues",
            json=data,
        )
        response.raise_for_status()

        logger.info(
            "Issue created",
            project=f"{namespace}/{project}",
            title=title[:50],
        )

        return response.json()

    async def list_issues(
        self,
        repo_url: str,
        state: Literal["opened", "closed", "all"] = "opened",
        labels: list[str] | None = None,
        assignee: str | None = None,
    ) -> list[dict[str, Any]]:
        """List issues."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        params: dict[str, Any] = {"state": state}
        if labels:
            params["labels"] = ",".join(labels)
        if assignee:
            params["assignee_username"] = assignee

        response = await self._get_client().get(
            f"/projects/{project_path}/issues",
            params=params,
        )
        response.raise_for_status()

        return response.json()

    async def update_issue(
        self,
        repo_url: str,
        iid: int,
        title: str | None = None,
        description: str | None = None,
        state: Literal["opened", "closed"] | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update an issue."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        data: dict[str, Any] = {}
        if title:
            data["title"] = title
        if description:
            data["description"] = description
        if state:
            data["state_event"] = state
        if labels:
            data["labels"] = ",".join(labels)

        response = await self._get_client().put(
            f"/projects/{project_path}/issues/{iid}",
            json=data,
        )
        response.raise_for_status()

        return response.json()

    # ═══════════════════════════════════════════════════════════════
    # Comments/Notes
    # ═══════════════════════════════════════════════════════════════

    async def create_merge_request_note(
        self,
        repo_url: str,
        mr_iid: int,
        body: str,
        position: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a note (comment) on a merge request.

        Args:
            repo_url: Repository URL
            mr_iid: Merge request IID
            body: Note content
            position: Optional code position for inline comments
        """
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        data: dict[str, Any] = {"body": body}
        if position:
            data["position"] = position

        response = await self._get_client().post(
            f"/projects/{project_path}/merge_requests/{mr_iid}/notes",
            json=data,
        )
        response.raise_for_status()

        return response.json()

    async def create_merge_request_diff_note(
        self,
        repo_url: str,
        mr_iid: int,
        body: str,
        file_path: str,
        new_line: int | None = None,
        old_line: int | None = None,
        base_sha: str | None = None,
        head_sha: str | None = None,
        start_sha: str | None = None,
    ) -> dict[str, Any]:
        """Create a diff note on a merge request."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        data: dict[str, Any] = {
            "body": body,
            "position": {
                "base_sha": base_sha,
                "head_sha": head_sha,
                "start_sha": start_sha,
                "position_type": "text",
                "new_path": file_path,
                "old_path": file_path,
            },
        }

        if new_line:
            data["position"]["new_line"] = new_line
        if old_line:
            data["position"]["old_line"] = old_line

        response = await self._get_client().post(
            f"/projects/{project_path}/merge_requests/{mr_iid}/diff_notes",
            json=data,
        )
        response.raise_for_status()

        return response.json()

    # ═══════════════════════════════════════════════════════════════
    # Repository Info
    # ═══════════════════════════════════════════════════════════════

    async def get_project(self, repo_url: str) -> dict[str, Any]:
        """Get project information."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        response = await self._get_client().get(f"/projects/{project_path}")
        response.raise_for_status()

        return response.json()

    async def get_repository_tree(
        self,
        repo_url: str,
        path: str = "",
        ref: str = "main",
    ) -> list[dict[str, Any]]:
        """Get repository file tree."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        params: dict[str, Any] = {"ref": ref}
        if path:
            params["path"] = path

        response = await self._get_client().get(
            f"/projects/{project_path}/repository/tree",
            params=params,
        )
        response.raise_for_status()

        return response.json()

    async def get_file_content(
        self,
        repo_url: str,
        file_path: str,
        ref: str = "main",
    ) -> str:
        """Get file content from repository."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        file_path_encoded = file_path.replace("/", "%2F")

        response = await self._get_client().get(
            f"/projects/{project_path}/repository/files/{file_path_encoded}/raw",
            params={"ref": ref},
        )
        response.raise_for_status()

        return response.text

    # ═══════════════════════════════════════════════════════════════
    # CI/CD
    # ═══════════════════════════════════════════════════════════════

    async def list_pipelines(
        self,
        repo_url: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List CI/CD pipelines."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        params: dict[str, Any] = {}
        if status:
            params["status"] = status

        response = await self._get_client().get(
            f"/projects/{project_path}/pipelines",
            params=params,
        )
        response.raise_for_status()

        return response.json()

    async def get_pipeline(self, repo_url: str, pipeline_id: int) -> dict[str, Any]:
        """Get pipeline details."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        response = await self._get_client().get(
            f"/projects/{project_path}/pipelines/{pipeline_id}",
        )
        response.raise_for_status()

        return response.json()

    async def create_pipeline(
        self,
        repo_url: str,
        ref: str,
        variables: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Trigger a new pipeline."""
        namespace, project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(namespace, project)

        data: dict[str, Any] = {"ref": ref}
        if variables:
            data["variables"] = [
                {"key": k, "value": v} for k, v in variables.items()
            ]

        response = await self._get_client().post(
            f"/projects/{project_path}/pipeline",
            json=data,
        )
        response.raise_for_status()

        logger.info("Pipeline created", project=f"{namespace}/{project}", ref=ref)

        return response.json()

    # ═══════════════════════════════════════════════════════════════
    # Utility Methods
    # ═══════════════════════════════════════════════════════════════

    async def _get_user_id(self, username: str) -> int | None:
        """Get user ID from username."""
        response = await self._get_client().get(
            "/users",
            params={"username": username},
        )
        response.raise_for_status()

        users = response.json()
        if users:
            return users[0]["id"]
        return None

    async def fork_project(
        self,
        repo_url: str,
        namespace: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Fork a project."""
        src_namespace, src_project = self._parse_repo_url(repo_url)
        project_path = self._encode_project_path(src_namespace, src_project)

        data: dict[str, Any] = {}
        if namespace:
            data["namespace_path"] = namespace
        if name:
            data["name"] = name

        response = await self._get_client().post(
            f"/projects/{project_path}/fork",
            json=data,
        )
        response.raise_for_status()

        logger.info(
            "Project forked",
            project=f"{src_namespace}/{src_project}",
            namespace=namespace,
        )

        return response.json()
