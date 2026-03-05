"""Infrastructure tools for Lya.

Provides adapters for external services and operations:
- Browser automation (Playwright)
- File system operations
- Git version control
- Code review services
- Web/HTTP operations
- System/shell operations
"""

from .filesystem import (
    FileManager,
    FileInfo,
    FileSearchResult,
    get_file_manager,
)
from .browser import (
    BrowserSession,
    BrowserAction,
    ExtractedData,
    ScreenshotResult,
    PlaywrightBrowserAdapter,
    get_browser_capability_code,
)
from .git import (
    GitAdapter,
    Commit,
    Diff,
    Branch,
    get_git_adapter,
    GitHubAdapter,
    PullRequest,
    ReviewComment,
    CodeReviewService,
    GitLabAdapter,
    GitLabMergeRequest,
)
from .web_tools import (
    WebTools,
    WebResponse,
    get_web_tools,
)
from .system_tools import (
    SystemTools,
    CommandResult,
    get_system_tools,
)
from .tool_registry import (
    ToolRegistry,
    get_tool_registry,
)

__all__ = [
    # File System
    "FileManager",
    "FileInfo",
    "FileSearchResult",
    "get_file_manager",
    # Browser
    "BrowserSession",
    "BrowserAction",
    "ExtractedData",
    "ScreenshotResult",
    "PlaywrightBrowserAdapter",
    "get_browser_capability_code",
    # Git
    "GitAdapter",
    "Commit",
    "Diff",
    "Branch",
    "get_git_adapter",
    "GitHubAdapter",
    "PullRequest",
    "ReviewComment",
    "CodeReviewService",
    "GitLabAdapter",
    "GitLabMergeRequest",
    # Web Tools
    "WebTools",
    "WebResponse",
    "get_web_tools",
    # System Tools
    "SystemTools",
    "CommandResult",
    "get_system_tools",
    # Tool Registry
    "ToolRegistry",
    "get_tool_registry",
]
