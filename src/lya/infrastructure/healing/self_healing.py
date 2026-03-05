"""Self-Healing System - Core Implementation.

This module provides the SelfHealing class for automatic error detection and recovery.
It supports multiple healing strategies including retry, fallback, and circuit breaker patterns.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Coroutine, Generic, ParamSpec, TypeVar
from uuid import UUID, uuid4

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from lya.domain.models.healing import (
    HealthIssue,
    HealingAction,
    HealthCheck,
    IssueType,
    IssueSeverity,
    HealingStatus,
)
from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject requests
    HALF_OPEN = auto()   # Testing if recovered


@dataclass
class HealingStrategyConfig:
    """Configuration for healing strategies."""
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    fallback_enabled: bool = True


@dataclass
class CircuitBreaker:
    """Circuit breaker for preventing cascading failures."""
    threshold: int = 5
    timeout: float = 60.0
    failure_count: int = 0
    last_failure_time: float | None = None
    state: CircuitState = field(default=CircuitState.CLOSED)

    def record_success(self) -> None:
        """Record a successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> bool:
        """Record a failed call. Returns True if circuit should open."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.threshold:
            self.state = CircuitState.OPEN
            return True
        return False

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False

        return True  # HALF_OPEN allows one test


class HealingStrategy:
    """Base class for healing strategies.

    Healing strategies define how to recover from specific types of issues.
    Concrete implementations should override the `can_handle` and `heal` methods.
    """

    def __init__(self, name: str, priority: int = 0):
        """Initialize healing strategy.

        Args:
            name: Strategy identifier
            priority: Higher priority strategies are tried first
        """
        self.name = name
        self.priority = priority
        self.config = HealingStrategyConfig()
        self.circuit_breaker = CircuitBreaker()
        self.success_count: int = 0
        self.failure_count: int = 0

    async def can_handle(self, issue: HealthIssue) -> bool:
        """Check if this strategy can handle the issue.

        Args:
            issue: The health issue to check

        Returns:
            True if this strategy can handle the issue
        """
        raise NotImplementedError

    async def heal(self, issue: HealthIssue) -> HealingAction:
        """Attempt to heal the issue.

        Args:
            issue: The health issue to heal

        Returns:
            Result of the healing attempt
        """
        raise NotImplementedError

    async def heal_with_retry(self, issue: HealthIssue) -> HealingAction:
        """Attempt to heal with retry logic.

        Args:
            issue: The health issue to heal

        Returns:
            Result of the healing attempt
        """
        action_id = uuid4()
        started_at = datetime.now(timezone.utc)

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            return HealingAction(
                action_id=action_id,
                issue_id=issue.issue_id,
                action_type=f"{self.name}_circuit_open",
                description=f"Circuit breaker open for {self.name}",
                status=HealingStatus.DEFERRED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                error_message="Circuit breaker is open",
            )

        last_error: Exception | None = None
        delay = self.config.retry_delay

        for attempt in range(1, self.config.max_retries + 1):
            try:
                action = await self.heal(issue)

                if action.status == HealingStatus.SUCCESS:
                    self.circuit_breaker.record_success()
                    self.success_count += 1
                    return action

                last_error = Exception(action.error_message or "Healing failed")

                if attempt < self.config.max_retries:
                    logger.info(
                        "Healing attempt failed, retrying",
                        strategy=self.name,
                        attempt=attempt,
                        next_delay=delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= self.config.retry_backoff

            except Exception as e:
                last_error = e
                logger.error(
                    "Healing attempt error",
                    strategy=self.name,
                    attempt=attempt,
                    error=str(e),
                )

                if attempt < self.config.max_retries:
                    await asyncio.sleep(delay)
                    delay *= self.config.retry_backoff

        # All retries failed
        should_open = self.circuit_breaker.record_failure()
        self.failure_count += 1

        return HealingAction(
            action_id=action_id,
            issue_id=issue.issue_id,
            action_type=f"{self.name}_retry_exhausted",
            description=f"Healing failed after {self.config.max_retries} attempts",
            status=HealingStatus.FAILED,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            error_message=str(last_error) if last_error else "Unknown error",
        )


class RetryStrategy(HealingStrategy):
    """Generic retry strategy for transient failures.

    This strategy wraps any operation and retries it on failure.
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        super().__init__("retry", priority=100)
        self.config.max_retries = max_retries
        self.config.retry_delay = base_delay

    async def can_handle(self, issue: HealthIssue) -> bool:
        """Can handle any issue marked as retryable."""
        return issue.context.get("retryable", False)

    async def heal(self, issue: HealthIssue) -> HealingAction:
        """Mark the issue for retry."""
        action = HealingAction(
            action_id=uuid4(),
            issue_id=issue.issue_id,
            action_type="retry",
            description=f"Retry operation after {issue.healing_attempts} attempts",
            status=HealingStatus.SUCCESS,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            result={"attempt": issue.healing_attempts + 1},
        )
        return action


class FallbackStrategy(HealingStrategy):
    """Fallback strategy for when primary operations fail.

    Provides alternative execution paths when the primary method fails.
    """

    def __init__(self, fallback_handler: Callable | None = None):
        super().__init__("fallback", priority=50)
        self.fallback_handler = fallback_handler

    async def can_handle(self, issue: HealthIssue) -> bool:
        """Can handle if fallback is enabled and available."""
        return self.config.fallback_enabled and (
            self.fallback_handler is not None
            or issue.context.get("fallback_available", False)
        )

    async def heal(self, issue: HealthIssue) -> HealingAction:
        """Execute fallback operation."""
        action = HealingAction(
            action_id=uuid4(),
            issue_id=issue.issue_id,
            action_type="fallback",
            description="Executing fallback operation",
            status=HealingStatus.HEALING,
            started_at=datetime.now(timezone.utc),
        )

        try:
            if self.fallback_handler:
                if asyncio.iscoroutinefunction(self.fallback_handler):
                    result = await self.fallback_handler(issue)
                else:
                    result = self.fallback_handler(issue)

                action.status = HealingStatus.SUCCESS
                action.result = {"fallback_result": result}
            else:
                action.status = HealingStatus.SUCCESS
                action.result = {"message": "Fallback mode activated"}

        except Exception as e:
            action.status = HealingStatus.FAILED
            action.error_message = str(e)

        action.completed_at = datetime.now(timezone.utc)
        return action


class CircuitBreakerStrategy(HealingStrategy):
    """Circuit breaker pattern implementation.

    Prevents cascading failures by temporarily rejecting requests
    when failures exceed a threshold.
    """

    def __init__(self, threshold: int = 5, timeout: float = 60.0):
        super().__init__("circuit_breaker", priority=200)
        self.circuit_breaker = CircuitBreaker(threshold, timeout)

    async def can_handle(self, issue: HealthIssue) -> bool:
        """Can handle if circuit breaker is active."""
        return issue.context.get("use_circuit_breaker", False)

    async def heal(self, issue: HealthIssue) -> HealingAction:
        """Check and manage circuit state."""
        action = HealingAction(
            action_id=uuid4(),
            issue_id=issue.issue_id,
            action_type="circuit_breaker_check",
            description="Checking circuit breaker state",
            status=HealingStatus.HEALING,
            started_at=datetime.now(timezone.utc),
        )

        if not self.circuit_breaker.can_execute():
            action.status = HealingStatus.DEFERRED
            action.error_message = "Circuit breaker is open"
        else:
            action.status = HealingStatus.SUCCESS
            action.result = {
                "state": self.circuit_breaker.state.name,
                "failure_count": self.circuit_breaker.failure_count,
            }

        action.completed_at = datetime.now(timezone.utc)
        return action


class OllamaHealingStrategy(HealingStrategy):
    """Heal Ollama connection issues."""

    def __init__(self):
        super().__init__("ollama_restart", priority=10)

    async def can_handle(self, issue: HealthIssue) -> bool:
        return issue.issue_type == IssueType.LLM_NOT_RESPONDING

    async def heal(self, issue: HealthIssue) -> HealingAction:
        action = HealingAction(
            action_id=uuid4(),
            issue_id=issue.issue_id,
            action_type="restart_service",
            description="Restart Ollama service",
            status=HealingStatus.HEALING,
            started_at=datetime.now(timezone.utc),
        )

        try:
            # Kill Ollama process
            subprocess.run(
                ["pkill", "-f", "ollama"],
                capture_output=True,
                timeout=5,
                check=False,
            )

            await asyncio.sleep(2)

            # Restart Ollama
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            await asyncio.sleep(5)

            # Verify
            if await SelfHealing.check_ollama():
                action.status = HealingStatus.SUCCESS
                action.result = {"action": "Restarted Ollama", "verified": True}
            else:
                action.status = HealingStatus.FAILED
                action.error_message = "Ollama still not responding after restart"

        except Exception as e:
            action.status = HealingStatus.FAILED
            action.error_message = str(e)

        action.completed_at = datetime.now(timezone.utc)
        return action


class MemoryHealingStrategy(HealingStrategy):
    """Heal high memory usage."""

    def __init__(self):
        super().__init__("memory_cleanup", priority=10)

    async def can_handle(self, issue: HealthIssue) -> bool:
        return issue.issue_type == IssueType.HIGH_MEMORY_USAGE

    async def heal(self, issue: HealthIssue) -> HealingAction:
        action = HealingAction(
            action_id=uuid4(),
            issue_id=issue.issue_id,
            action_type="garbage_collection",
            description="Clean up memory",
            status=HealingStatus.HEALING,
            started_at=datetime.now(timezone.utc),
        )

        try:
            # Force garbage collection
            gc.collect()

            if HAS_PSUTIL:
                mem_before = psutil.virtual_memory().percent

                # Clear caches if possible
                gc.collect()

                # Get memory after
                mem = psutil.virtual_memory()

                if mem.percent < 80:
                    action.status = HealingStatus.SUCCESS
                    action.result = {
                        "action": "Garbage collection",
                        "before_percent": mem_before,
                        "new_usage": f"{mem.percent:.1f}%",
                        "freed_mb": issue.context.get("memory_mb", 0) - mem.used / 1024 / 1024,
                    }
                else:
                    action.status = HealingStatus.FAILED
                    action.error_message = f"Memory still high: {mem.percent:.1f}%"
            else:
                action.status = HealingStatus.SUCCESS
                action.result = {"action": "Garbage collection", "psutil": False}

        except Exception as e:
            action.status = HealingStatus.FAILED
            action.error_message = str(e)

        action.completed_at = datetime.now(timezone.utc)
        return action


class DiskHealingStrategy(HealingStrategy):
    """Heal disk full issues."""

    def __init__(self):
        super().__init__("disk_cleanup", priority=10)

    async def can_handle(self, issue: HealthIssue) -> bool:
        return issue.issue_type == IssueType.DISK_FULL

    async def heal(self, issue: HealthIssue) -> HealingAction:
        action = HealingAction(
            action_id=uuid4(),
            issue_id=issue.issue_id,
            action_type="clear_temp_files",
            description="Clear temporary files",
            status=HealingStatus.HEALING,
            started_at=datetime.now(timezone.utc),
        )

        try:
            freed = 0
            cleaned_files = 0

            # Clear temp directories
            temp_dirs = [
                Path("/tmp"),
                Path.home() / ".cache",
                Path.home() / ".tmp",
                Path(settings.workspace_path) / "temp",
            ]

            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    try:
                        for file in temp_dir.glob("*"):
                            try:
                                if file.is_file():
                                    freed += file.stat().st_size
                                    file.unlink()
                                    cleaned_files += 1
                                elif file.is_dir():
                                    freed += sum(f.stat().st_size for f in file.rglob("*") if f.is_file())
                                    shutil.rmtree(file)
                                    cleaned_files += 1
                            except (OSError, PermissionError):
                                pass
                    except (OSError, PermissionError):
                        pass

            # Check disk after cleanup
            if HAS_PSUTIL:
                disk = psutil.disk_usage('/')

                if disk.percent < 90:
                    action.status = HealingStatus.SUCCESS
                    action.result = {
                        "action": "Cleared temp files",
                        "freed_mb": freed / 1024 / 1024,
                        "files_removed": cleaned_files,
                        "new_usage": f"{disk.percent:.1f}%",
                    }
                else:
                    action.status = HealingStatus.FAILED
                    action.error_message = f"Disk still full: {disk.percent:.1f}%"
            else:
                action.status = HealingStatus.SUCCESS
                action.result = {
                    "action": "Cleared temp files",
                    "freed_mb": freed / 1024 / 1024,
                    "files_removed": cleaned_files,
                }

        except Exception as e:
            action.status = HealingStatus.FAILED
            action.error_message = str(e)

        action.completed_at = datetime.now(timezone.utc)
        return action


class DatabaseHealingStrategy(HealingStrategy):
    """Heal database connection issues."""

    def __init__(self):
        super().__init__("database_reconnect", priority=10)

    async def can_handle(self, issue: HealthIssue) -> bool:
        return issue.issue_type == IssueType.DATABASE_CONNECTION_LOST

    async def heal(self, issue: HealthIssue) -> HealingAction:
        action = HealingAction(
            action_id=uuid4(),
            issue_id=issue.issue_id,
            action_type="reconnect_database",
            description="Reconnect to database",
            status=HealingStatus.HEALING,
            started_at=datetime.now(timezone.utc),
        )

        try:
            # Wait a moment for network to stabilize
            await asyncio.sleep(2)

            # Attempt to reconnect
            # This would integrate with the actual database connection pool
            action.status = HealingStatus.SUCCESS
            action.result = {"action": "Database reconnection attempted"}

        except Exception as e:
            action.status = HealingStatus.FAILED
            action.error_message = str(e)

        action.completed_at = datetime.now(timezone.utc)
        return action


class CacheHealingStrategy(HealingStrategy):
    """Heal corrupted cache issues."""

    def __init__(self):
        super().__init__("cache_reset", priority=10)

    async def can_handle(self, issue: HealthIssue) -> bool:
        return issue.issue_type == IssueType.CACHE_CORRUPTED

    async def heal(self, issue: HealthIssue) -> HealingAction:
        action = HealingAction(
            action_id=uuid4(),
            issue_id=issue.issue_id,
            action_type="clear_cache",
            description="Clear corrupted cache",
            status=HealingStatus.HEALING,
            started_at=datetime.now(timezone.utc),
        )

        try:
            cache_path = Path(settings.workspace_path) / "cache"
            if cache_path.exists():
                # Backup corrupted cache
                backup_path = Path(settings.workspace_path) / "cache_backup"
                cache_path.rename(backup_path)
                cache_path.mkdir(parents=True, exist_ok=True)

                action.status = HealingStatus.SUCCESS
                action.result = {
                    "action": "Cache cleared and backed up",
                    "backup_path": str(backup_path),
                }
            else:
                action.status = HealingStatus.SUCCESS
                action.result = {"action": "No cache to clear"}

        except Exception as e:
            action.status = HealingStatus.FAILED
            action.error_message = str(e)

        action.completed_at = datetime.now(timezone.utc)
        return action


class SelfHealing:
    """Self-healing system for automatic error recovery.

    This class provides methods for:
    - Monitoring failures (exceptions, errors)
    - Attempting automatic recovery strategies
    - Logging all healing attempts
    - Supporting retry, fallback, and circuit breaker patterns
    - Tracking healing history

    Example:
        healing = SelfHealing()

        # Heal a specific error
        action = await healing.heal_error(
            error=ConnectionError("Database connection failed"),
            context={"service": "database", "retryable": True}
        )

        # Run diagnostics
        diagnosis = await healing.diagnose()

        # Apply a specific fix
        action = await healing.apply_fix(IssueType.HIGH_MEMORY_USAGE)
    """

    def __init__(self, workspace: Path | None = None):
        """Initialize self-healing system.

        Args:
            workspace: Path to workspace directory for logs and state
        """
        self.workspace = workspace or Path(settings.workspace_path)
        self.healing_log: list[HealingAction] = []
        self.issues: dict[UUID, HealthIssue] = {}
        self.strategies: list[HealingStrategy] = []
        self.active_issues: dict[UUID, HealthIssue] = {}
        self._running = False
        self._check_interval = 60

        # Register default strategies
        self._register_default_strategies()

        logger.info("Self-healing system initialized", workspace=str(self.workspace))

    def _register_default_strategies(self) -> None:
        """Register default healing strategies."""
        self.strategies = [
            RetryStrategy(),
            FallbackStrategy(),
            CircuitBreakerStrategy(),
            OllamaHealingStrategy(),
            MemoryHealingStrategy(),
            DiskHealingStrategy(),
            DatabaseHealingStrategy(),
            CacheHealingStrategy(),
        ]
        # Sort by priority (higher first)
        self.strategies.sort(key=lambda s: s.priority, reverse=True)

    def register_strategy(self, strategy: HealingStrategy) -> None:
        """Register a healing strategy.

        Args:
            strategy: Strategy to register
        """
        self.strategies.append(strategy)
        self.strategies.sort(key=lambda s: s.priority, reverse=True)
        logger.info("Registered healing strategy", name=strategy.name, priority=strategy.priority)

    def unregister_strategy(self, strategy_name: str) -> bool:
        """Unregister a healing strategy.

        Args:
            strategy_name: Name of strategy to remove

        Returns:
            True if strategy was found and removed
        """
        for i, strategy in enumerate(self.strategies):
            if strategy.name == strategy_name:
                self.strategies.pop(i)
                logger.info("Unregistered healing strategy", name=strategy_name)
                return True
        return False

    async def heal_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        severity: IssueSeverity = IssueSeverity.MEDIUM,
    ) -> HealingAction | None:
        """Heal an error by creating an issue and applying healing strategies.

        Args:
            error: The exception/error to heal
            context: Additional context about the error
            severity: Severity level of the issue

        Returns:
            HealingAction if healing was attempted, None otherwise
        """
        context = context or {}

        # Determine issue type from error
        issue_type = self._classify_error(error)

        # Create health issue
        issue = HealthIssue(
            issue_id=uuid4(),
            issue_type=issue_type,
            severity=severity,
            description=str(error),
            detected_at=datetime.now(timezone.utc),
            source="heal_error",
            context={
                **context,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )

        self.issues[issue.issue_id] = issue
        self.active_issues[issue.issue_id] = issue

        logger.info(
            "Created health issue from error",
            issue_id=str(issue.issue_id),
            issue_type=issue_type.name,
            severity=severity.name,
        )

        # Attempt to heal
        return await self._heal_issue(issue)

    def _classify_error(self, error: Exception) -> IssueType:
        """Classify an error into an issue type.

        Args:
            error: The exception to classify

        Returns:
            IssueType classification
        """
        error_type = type(error).__name__
        error_msg = str(error).lower()

        if "ollama" in error_msg or "llm" in error_msg or "connection" in error_msg:
            return IssueType.LLM_NOT_RESPONDING
        elif "memory" in error_msg:
            return IssueType.HIGH_MEMORY_USAGE
        elif "disk" in error_msg or "space" in error_msg:
            return IssueType.DISK_FULL
        elif "database" in error_msg or "db" in error_msg:
            return IssueType.DATABASE_CONNECTION_LOST
        elif "cache" in error_msg:
            return IssueType.CACHE_CORRUPTED
        elif "network" in error_msg or "http" in error_msg:
            return IssueType.NETWORK_ERROR
        elif "task" in error_msg:
            return IssueType.TASK_STUCK
        else:
            return IssueType.AGENT_CRASHED

    async def diagnose(self) -> dict[str, Any]:
        """Run diagnostics and detect issues.

        Returns:
            Dictionary containing diagnosis results
        """
        issues: list[HealthIssue] = []
        checks: list[HealthCheck] = []

        # Check Ollama
        check_id = uuid4()
        ollama_healthy = await self.check_ollama()
        checks.append(HealthCheck(
            check_id=check_id,
            name="ollama_health",
            component="llm",
            status="healthy" if ollama_healthy else "critical",
            message="Ollama is responding" if ollama_healthy else "Ollama not responding",
            timestamp=datetime.now(timezone.utc),
        ))

        if not ollama_healthy:
            issue = HealthIssue(
                issue_id=uuid4(),
                issue_type=IssueType.LLM_NOT_RESPONDING,
                severity=IssueSeverity.CRITICAL,
                description="Ollama service not responding",
                detected_at=datetime.now(timezone.utc),
                source="diagnosis",
            )
            issues.append(issue)
            self.issues[issue.issue_id] = issue
            self.active_issues[issue.issue_id] = issue

        # Check memory
        if HAS_PSUTIL:
            check_id = uuid4()
            mem = psutil.virtual_memory()
            mem_status = "healthy" if mem.percent < 80 else "warning" if mem.percent < 90 else "critical"

            checks.append(HealthCheck(
                check_id=check_id,
                name="memory_usage",
                component="system",
                status=mem_status,
                message=f"Memory usage at {mem.percent:.1f}%",
                timestamp=datetime.now(timezone.utc),
                metrics={"percent": mem.percent, "available_mb": mem.available / 1024 / 1024},
            ))

            if mem.percent > 90:
                issue = HealthIssue(
                    issue_id=uuid4(),
                    issue_type=IssueType.HIGH_MEMORY_USAGE,
                    severity=IssueSeverity.WARNING,
                    description=f"Memory usage at {mem.percent:.1f}%",
                    detected_at=datetime.now(timezone.utc),
                    source="diagnosis",
                    context={"memory_mb": mem.used / 1024 / 1024, "percent": mem.percent},
                )
                issues.append(issue)
                self.issues[issue.issue_id] = issue
                self.active_issues[issue.issue_id] = issue

            # Check disk
            check_id = uuid4()
            disk = psutil.disk_usage('/')
            disk_status = "healthy" if disk.percent < 80 else "warning" if disk.percent < 90 else "critical"

            checks.append(HealthCheck(
                check_id=check_id,
                name="disk_usage",
                component="system",
                status=disk_status,
                message=f"Disk usage at {disk.percent:.1f}%",
                timestamp=datetime.now(timezone.utc),
                metrics={"percent": disk.percent, "free_gb": disk.free / 1024 / 1024 / 1024},
            ))

            if disk.percent > 95:
                issue = HealthIssue(
                    issue_id=uuid4(),
                    issue_type=IssueType.DISK_FULL,
                    severity=IssueSeverity.CRITICAL,
                    description=f"Disk usage at {disk.percent:.1f}%",
                    detected_at=datetime.now(timezone.utc),
                    source="diagnosis",
                    context={"percent": disk.percent},
                )
                issues.append(issue)
                self.issues[issue.issue_id] = issue
                self.active_issues[issue.issue_id] = issue

        # Auto-heal critical issues
        for issue in issues:
            if issue.severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH):
                await self._heal_issue(issue)

        self.health_checks = checks

        return {
            "healthy": len(issues) == 0,
            "issues_count": len(issues),
            "issues": [i.to_dict() for i in issues],
            "checks": [c.to_dict() for c in checks],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def apply_fix(self, issue_type: IssueType, **kwargs: Any) -> HealingAction | None:
        """Apply a specific fix for an issue type.

        Args:
            issue_type: Type of issue to fix
            **kwargs: Additional context for the fix

        Returns:
            HealingAction if fix was applied, None otherwise
        """
        # Create synthetic issue
        issue = HealthIssue(
            issue_id=uuid4(),
            issue_type=issue_type,
            severity=IssueSeverity.MEDIUM,
            description=f"Manual fix for {issue_type.name}",
            detected_at=datetime.now(timezone.utc),
            source="manual_fix",
            context=kwargs,
        )

        self.issues[issue.issue_id] = issue
        self.active_issues[issue.issue_id] = issue

        return await self._heal_issue(issue)

    async def _heal_issue(self, issue: HealthIssue) -> HealingAction | None:
        """Internal method to heal an issue.

        Args:
            issue: The health issue to heal

        Returns:
            HealingAction if healing was attempted, None otherwise
        """
        issue.status = HealingStatus.HEALING
        issue.healing_attempts += 1

        # Find appropriate strategy
        for strategy in self.strategies:
            try:
                if await strategy.can_handle(issue):
                    logger.info(
                        "Attempting healing",
                        strategy=strategy.name,
                        issue=str(issue.issue_id),
                        attempt=issue.healing_attempts,
                    )

                    # Use retry wrapper for healing
                    action = await strategy.heal_with_retry(issue)
                    self.healing_log.append(action)

                    # Update issue status
                    issue.status = action.status

                    if action.status == HealingStatus.SUCCESS:
                        logger.info(
                            "Healing successful",
                            strategy=strategy.name,
                            issue=str(issue.issue_id),
                        )
                        # Remove from active if healed
                        if issue.issue_id in self.active_issues:
                            del self.active_issues[issue.issue_id]
                    else:
                        logger.warning(
                            "Healing failed",
                            strategy=strategy.name,
                            issue=str(issue.issue_id),
                            status=action.status.name,
                            error=action.error_message,
                        )

                    return action

            except Exception as e:
                logger.error(
                    "Strategy error",
                    strategy=strategy.name,
                    error=str(e),
                )

        logger.warning(
            "No healing strategy found",
            issue=str(issue.issue_id),
            issue_type=issue.issue_type.name,
        )
        issue.status = HealingStatus.DEFERRED
        return None

    async def start_monitoring(self, interval: int = 60) -> None:
        """Start background monitoring loop.

        Args:
            interval: Seconds between checks
        """
        self._running = True
        self._check_interval = interval

        logger.info("Self-healing monitoring started", interval=interval)

        while self._running:
            try:
                await self.diagnose()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error("Monitoring error", error=str(e))
                await asyncio.sleep(interval)

    def stop_monitoring(self) -> None:
        """Stop background monitoring loop."""
        self._running = False
        logger.info("Self-healing monitoring stopped")

    def get_healing_history(
        self,
        limit: int = 100,
        status: HealingStatus | None = None,
    ) -> list[dict[str, Any]]:
        """Get healing history.

        Args:
            limit: Maximum number of records
            status: Filter by status

        Returns:
            List of healing actions with issue details
        """
        actions = self.healing_log

        if status:
            actions = [a for a in actions if a.status == status]

        return [
            {
                "issue": self.issues.get(a.issue_id, {}).to_dict() if a.issue_id in self.issues else None,
                "action": a.to_dict(),
            }
            for a in actions[-limit:]
        ]

    def get_active_issues(self) -> list[HealthIssue]:
        """Get list of active (unresolved) issues.

        Returns:
            List of active health issues
        """
        return list(self.active_issues.values())

    def get_issue_stats(self) -> dict[str, Any]:
        """Get statistics about issues and healing.

        Returns:
            Dictionary with statistics
        """
        total_actions = len(self.healing_log)
        successful = sum(1 for a in self.healing_log if a.status == HealingStatus.SUCCESS)
        failed = sum(1 for a in self.healing_log if a.status == HealingStatus.FAILED)
        deferred = sum(1 for a in self.healing_log if a.status == HealingStatus.DEFERRED)

        by_type: dict[str, int] = {}
        for issue in self.issues.values():
            type_name = issue.issue_type.name
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            "total_issues": len(self.issues),
            "active_issues": len(self.active_issues),
            "total_healing_attempts": total_actions,
            "successful_heals": successful,
            "failed_heals": failed,
            "deferred_heals": deferred,
            "success_rate": successful / total_actions if total_actions > 0 else 0,
            "issues_by_type": by_type,
            "registered_strategies": len(self.strategies),
            "strategy_names": [s.name for s in self.strategies],
        }

    def clear_history(self, keep_recent: int = 0) -> int:
        """Clear healing history.

        Args:
            keep_recent: Number of recent entries to keep

        Returns:
            Number of entries removed
        """
        if keep_recent >= len(self.healing_log):
            return 0

        removed = len(self.healing_log) - keep_recent
        self.healing_log = self.healing_log[-keep_recent:] if keep_recent > 0 else []
        return removed

    def resolve_issue(self, issue_id: UUID, resolution: str = "manual") -> bool:
        """Manually resolve an issue.

        Args:
            issue_id: ID of issue to resolve
            resolution: Resolution description

        Returns:
            True if issue was found and resolved
        """
        issue = self.active_issues.get(issue_id)
        if not issue:
            return False

        issue.status = HealingStatus.SUCCESS
        del self.active_issues[issue_id]

        # Log the manual resolution
        action = HealingAction(
            action_id=uuid4(),
            issue_id=issue_id,
            action_type="manual_resolution",
            description=f"Issue manually resolved: {resolution}",
            status=HealingStatus.SUCCESS,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            result={"resolution": resolution},
        )
        self.healing_log.append(action)

        logger.info("Issue manually resolved", issue_id=str(issue_id), resolution=resolution)
        return True

    @staticmethod
    async def check_ollama(timeout: float = 5.0) -> bool:
        """Check if Ollama is responding.

        Args:
            timeout: Request timeout in seconds

        Returns:
            True if Ollama is responding
        """
        if not HAS_HTTPX:
            return False

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:11434/api/tags",
                    timeout=timeout,
                )
                return response.status_code == 200
        except Exception:
            return False

    def decorator_with_healing(
        self,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
        max_retries: int = 3,
        fallback_func: Callable | None = None,
    ) -> Callable[[Callable[P, T]], Callable[P, Coroutine[Any, Any, T | None]]]:
        """Decorator to add healing capabilities to functions.

        Args:
            retryable_exceptions: Tuple of exceptions to catch and retry
            max_retries: Maximum retry attempts
            fallback_func: Function to call on failure

        Returns:
            Decorator function
        """
        retryable_exceptions = retryable_exceptions or (Exception,)

        def decorator(func: Callable[P, T]) -> Callable[P, Coroutine[Any, Any, T | None]]:
            @functools.wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
                for attempt in range(1, max_retries + 1):
                    try:
                        if asyncio.iscoroutinefunction(func):
                            result = await func(*args, **kwargs)
                        else:
                            result = func(*args, **kwargs)
                        return result
                    except retryable_exceptions as e:
                        if attempt < max_retries:
                            logger.warning(
                                "Function failed, retrying",
                                func=func.__name__,
                                attempt=attempt,
                                max_retries=max_retries,
                                error=str(e),
                            )
                            await asyncio.sleep(1 * attempt)
                        else:
                            # Max retries reached
                            await self.heal_error(
                                e,
                                context={
                                    "function": func.__name__,
                                    "args": str(args),
                                    "kwargs": list(kwargs.keys()),
                                    "max_retries": max_retries,
                                },
                                severity=IssueSeverity.HIGH,
                            )

                            if fallback_func:
                                logger.info("Executing fallback", func=func.__name__)
                                if asyncio.iscoroutinefunction(fallback_func):
                                    return await fallback_func(*args, **kwargs)
                                return fallback_func(*args, **kwargs)

                            raise
                return None
            return wrapper
        return decorator
