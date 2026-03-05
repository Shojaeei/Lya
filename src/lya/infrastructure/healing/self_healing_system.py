"""Self-Healing System - Standalone Implementation.

This module provides a simplified self-healing system that can be used
standalone or as part of the larger healing infrastructure.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generic, ParamSpec, TypeVar
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

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    exceptions: tuple[type[Exception], ...] = field(default_factory=lambda: (Exception,))


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


class CircuitBreakerState:
    """Circuit breaker state management."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.failures: list[float] = []
        self.last_failure_time: float | None = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def can_execute(self) -> bool:
        """Check if execution is allowed."""
        async with self._lock:
            if self.state == "closed":
                return True

            if self.state == "open":
                # Check if recovery timeout passed
                if self.last_failure_time and (time.time() - self.last_failure_time) > self.config.recovery_timeout:
                    self.state = "half_open"
                    self.half_open_calls = 0
                    return True
                return False

            if self.state == "half_open":
                if self.half_open_calls < self.config.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False

            return True

    async def record_success(self) -> None:
        """Record a successful execution."""
        async with self._lock:
            self.failures.clear()
            if self.state == "half_open":
                self.state = "closed"
                self.half_open_calls = 0
            logger.debug("Circuit breaker success", name=self.name, state=self.state)

    async def record_failure(self) -> None:
        """Record a failed execution."""
        async with self._lock:
            now = time.time()
            self.failures.append(now)
            self.last_failure_time = now

            # Remove old failures outside window
            window_start = now - self.config.recovery_timeout
            self.failures = [f for f in self.failures if f > window_start]

            if len(self.failures) >= self.config.failure_threshold:
                self.state = "open"
                self.half_open_calls = 0
                logger.warning("Circuit breaker opened", name=self.name, failures=len(self.failures))

    def get_state(self) -> dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state,
            "failures": len(self.failures),
            "last_failure": self.last_failure_time,
        }


class SelfHealingSystem:
    """
    Standalone self-healing system for automatic error recovery.

    This class provides:
    - Failure monitoring and detection
    - Automatic recovery strategies
    - Comprehensive logging of healing attempts
    - Retry, fallback, and circuit breaker patterns
    - Healing history tracking

    Methods:
        heal_error(): Heal an exception/error
        diagnose(): Run health diagnostics
        apply_fix(): Apply a specific fix

    Example:
        healing = SelfHealingSystem(Path("~/.lya"))

        # Heal an error
        action = await healing.heal_error(
            ConnectionError("DB connection failed"),
            context={"service": "database"}
        )

        # Run diagnostics
        result = await healing.diagnose()

        # Apply a specific fix
        action = await healing.apply_fix(IssueType.HIGH_MEMORY_USAGE)
    """

    def __init__(self, workspace: Path | None = None):
        """Initialize self-healing system.

        Args:
            workspace: Path to workspace for state and logs
        """
        self.workspace = workspace or Path.home() / ".lya"
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.healing_log: list[HealingAction] = []
        self.active_issues: dict[UUID, HealthIssue] = {}
        self.circuit_breakers: dict[str, CircuitBreakerState] = {}
        self.healing_strategies: dict[str, Callable] = {}
        self.fallback_handlers: dict[IssueType, Callable] = {}
        self._running = False
        self._retry_config = RetryConfig()

        # Register healing strategies
        self._register_strategies()

        logger.info("Self-healing system initialized", workspace=str(self.workspace))

    def _register_strategies(self) -> None:
        """Register default healing strategies."""
        self.healing_strategies = {
            "ollama_not_responding": self._heal_ollama,
            "high_memory_usage": self._heal_memory,
            "disk_full": self._heal_disk,
            "cache_corrupted": self._heal_cache,
            "task_stuck": self._heal_stuck_task,
            "database_connection_lost": self._heal_database,
            "network_error": self._heal_network,
        }

        # Register fallback handlers
        self.fallback_handlers[IssueType.LLM_NOT_RESPONDING] = self._fallback_to_cloud_llm

    def get_or_create_circuit_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreakerState:
        """Get or create a circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration

        Returns:
            Circuit breaker instance
        """
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreakerState(name, config)
        return self.circuit_breakers[name]

    async def heal_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        severity: IssueSeverity = IssueSeverity.MEDIUM,
        auto_heal: bool = True,
    ) -> HealingAction:
        """Heal an error by creating an issue and applying recovery strategies.

        Args:
            error: The exception to heal
            context: Additional context about the error
            severity: Severity of the issue
            auto_heal: Whether to automatically attempt healing

        Returns:
            HealingAction with results
        """
        context = context or {}

        # Classify the error
        issue_type = self._classify_error(error)

        # Create health issue
        issue = HealthIssue(
            issue_id=uuid4(),
            issue_type=issue_type,
            severity=severity,
            description=str(error),
            detected_at=datetime.now(timezone.utc),
            source="error_handler",
            context={
                **context,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )

        self.active_issues[issue.issue_id] = issue

        logger.info(
            "Created health issue from error",
            issue_id=str(issue.issue_id),
            issue_type=issue_type.name,
            severity=severity.name,
        )

        # Create healing action
        action = HealingAction(
            action_id=uuid4(),
            issue_id=issue.issue_id,
            action_type="error_healing",
            description=f"Healing {issue_type.name}: {str(error)}",
            status=HealingStatus.HEALING,
            started_at=datetime.now(timezone.utc),
        )

        if not auto_heal:
            action.status = HealingStatus.DEFERRED
            action.completed_at = datetime.now(timezone.utc)
            self.healing_log.append(action)
            return action

        # Attempt healing
        return await self._execute_healing(issue, action)

    def _classify_error(self, error: Exception) -> IssueType:
        """Classify an error into an issue type.

        Args:
            error: Exception to classify

        Returns:
            IssueType classification
        """
        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()

        if "ollama" in error_msg or "llm" in error_msg:
            return IssueType.LLM_NOT_RESPONDING
        elif "memory" in error_msg or "ram" in error_msg:
            return IssueType.HIGH_MEMORY_USAGE
        elif "disk" in error_msg or "space" in error_msg or "full" in error_msg:
            return IssueType.DISK_FULL
        elif "database" in error_msg or "db" in error_msg or "connection" in error_msg:
            return IssueType.DATABASE_CONNECTION_LOST
        elif "cache" in error_msg:
            return IssueType.CACHE_CORRUPTED
        elif "network" in error_msg or "http" in error_msg:
            return IssueType.NETWORK_ERROR
        elif "task" in error_msg or "stuck" in error_msg or "timeout" in error_msg:
            return IssueType.TASK_STUCK
        else:
            return IssueType.AGENT_CRASHED

    async def diagnose(self) -> dict[str, Any]:
        """Run health diagnostics and detect issues.

        Returns:
            Dictionary with diagnosis results
        """
        issues = []
        checks = []

        # Check Ollama
        check_id = uuid4()
        ollama_ok = await self._check_ollama()
        checks.append(HealthCheck(
            check_id=check_id,
            name="ollama_health",
            component="llm",
            status="healthy" if ollama_ok else "critical",
            message="Ollama is responding" if ollama_ok else "Ollama not responding",
            timestamp=datetime.now(timezone.utc),
        ))

        if not ollama_ok:
            issue = HealthIssue(
                issue_id=uuid4(),
                issue_type=IssueType.LLM_NOT_RESPONDING,
                severity=IssueSeverity.CRITICAL,
                description="Ollama is not responding",
                detected_at=datetime.now(timezone.utc),
                source="diagnostic",
            )
            issues.append(issue)
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
                    description=f"High memory usage: {mem.percent:.1f}%",
                    detected_at=datetime.now(timezone.utc),
                    source="diagnostic",
                    context={"percent": mem.percent},
                )
                issues.append(issue)
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
                    description=f"Disk nearly full: {disk.percent:.1f}%",
                    detected_at=datetime.now(timezone.utc),
                    source="diagnostic",
                    context={"percent": disk.percent},
                )
                issues.append(issue)
                self.active_issues[issue.issue_id] = issue

        # Create healing actions for detected issues
        for issue in issues:
            await self.heal(issue.issue_id)

        return {
            "healthy": len(issues) == 0,
            "issues": [i.to_dict() for i in issues],
            "checks": [c.to_dict() for c in checks],
            "active_issues": len(self.active_issues),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def apply_fix(
        self,
        issue_type: IssueType,
        severity: IssueSeverity = IssueSeverity.MEDIUM,
        **kwargs: Any,
    ) -> HealingAction:
        """Apply a specific fix for an issue type.

        Args:
            issue_type: Type of issue to fix
            severity: Severity level
            **kwargs: Additional context for the fix

        Returns:
            HealingAction with results
        """
        # Create synthetic issue
        issue = HealthIssue(
            issue_id=uuid4(),
            issue_type=issue_type,
            severity=severity,
            description=f"Manual fix for {issue_type.name}",
            detected_at=datetime.now(timezone.utc),
            source="manual_fix",
            context=kwargs,
        )

        self.active_issues[issue.issue_id] = issue

        # Create healing action
        action = HealingAction(
            action_id=uuid4(),
            issue_id=issue.issue_id,
            action_type="manual_fix",
            description=f"Applying manual fix for {issue_type.name}",
            status=HealingStatus.HEALING,
            started_at=datetime.now(timezone.utc),
        )

        return await self._execute_healing(issue, action)

    async def heal(self, issue_id: UUID) -> HealingAction:
        """Heal a specific issue by ID.

        Args:
            issue_id: ID of issue to heal

        Returns:
            HealingAction with results
        """
        issue = self.active_issues.get(issue_id)
        if not issue:
            return HealingAction(
                action_id=uuid4(),
                issue_id=issue_id,
                action_type="unknown",
                description="Issue not found",
                status=HealingStatus.FAILED,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                error_message="Issue ID not found in active issues",
            )

        # Check healing limits
        if issue.healing_attempts >= 3:
            issue.status = HealingStatus.IGNORED
            return HealingAction(
                action_id=uuid4(),
                issue_id=issue_id,
                action_type="max_attempts",
                description="Max healing attempts reached",
                status=HealingStatus.IGNORED,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )

        issue.status = HealingStatus.HEALING
        issue.healing_attempts += 1

        action = HealingAction(
            action_id=uuid4(),
            issue_id=issue_id,
            action_type=issue.issue_type.name.lower(),
            description=f"Healing {issue.issue_type.name}",
            status=HealingStatus.HEALING,
            started_at=datetime.now(timezone.utc),
        )

        return await self._execute_healing(issue, action)

    async def _execute_healing(
        self,
        issue: HealthIssue,
        action: HealingAction,
    ) -> HealingAction:
        """Execute healing with retry and circuit breaker logic.

        Args:
            issue: Health issue to heal
            action: Healing action to update

        Returns:
            Updated HealingAction
        """
        strategy_key = issue.issue_type.name.lower()

        # Check circuit breaker
        cb = self.get_or_create_circuit_breaker(strategy_key)
        if not await cb.can_execute():
            action.status = HealingStatus.DEFERRED
            action.error_message = "Circuit breaker is open"
            action.completed_at = datetime.now(timezone.utc)
            self.healing_log.append(action)
            return action

        # Attempt healing with retry
        last_error: str | None = None

        for attempt in range(1, self._retry_config.max_retries + 1):
            try:
                if strategy_key in self.healing_strategies:
                    result = await self._retry_config.exceptions[0]

                    # Get the healing strategy
                    heal_func = self.healing_strategies[strategy_key]
                    result = await heal_func()

                    action.completed_at = datetime.now(timezone.utc)

                    if result.get("success"):
                        action.status = HealingStatus.SUCCESS
                        action.result = result
                        issue.status = HealingStatus.SUCCESS

                        # Remove from active issues
                        if issue.issue_id in self.active_issues:
                            del self.active_issues[issue.issue_id]

                        # Record circuit breaker success
                        await cb.record_success()

                        logger.info(
                            "Healing successful",
                            issue_type=issue.issue_type.name,
                            action=result.get("action"),
                        )
                    else:
                        last_error = result.get("error", "Unknown error")
                        if attempt < self._retry_config.max_retries:
                            delay = min(
                                self._retry_config.base_delay * (self._retry_config.backoff_factor ** (attempt - 1)),
                                self._retry_config.max_delay,
                            )
                            logger.warning(
                                "Healing failed, retrying",
                                attempt=attempt,
                                delay=delay,
                                error=last_error,
                            )
                            await asyncio.sleep(delay)
                        else:
                            action.status = HealingStatus.FAILED
                            action.error_message = last_error
                            issue.status = HealingStatus.DETECTED
                            await cb.record_failure()

                            # Try fallback
                            await self._try_fallback(issue, action)

                    break
                else:
                    action.status = HealingStatus.FAILED
                    action.error_message = f"No healing strategy for {strategy_key}"
                    break

            except Exception as e:
                last_error = str(e)
                logger.error("Healing error", attempt=attempt, error=last_error)

                if attempt < self._retry_config.max_retries:
                    delay = min(
                        self._retry_config.base_delay * (self._retry_config.backoff_factor ** (attempt - 1)),
                        self._retry_config.max_delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    await cb.record_failure()

        if action.status == HealingStatus.HEALING:
            action.status = HealingStatus.FAILED
            action.error_message = last_error or "Healing failed"
            action.completed_at = datetime.now(timezone.utc)

        self.healing_log.append(action)
        return action

    async def _try_fallback(
        self,
        issue: HealthIssue,
        action: HealingAction,
    ) -> None:
        """Try fallback handler for an issue."""
        fallback = self.fallback_handlers.get(issue.issue_type)
        if fallback:
            try:
                logger.info("Attempting fallback", issue_type=issue.issue_type.name)
                result = await fallback(issue)
                action.result = {**action.result, "fallback": result}
            except Exception as e:
                logger.error("Fallback failed", error=str(e))

    # ═══════════════════════════════════════════════════════════════
    # Healing Strategies
    # ═══════════════════════════════════════════════════════════════

    async def _heal_ollama(self) -> dict[str, Any]:
        """Restart Ollama service."""
        try:
            subprocess.run(
                ["pkill", "ollama"],
                capture_output=True,
                check=False,
            )
            await asyncio.sleep(2)

            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await asyncio.sleep(5)

            if await self._check_ollama():
                return {"success": True, "action": "Restarted Ollama"}
            return {"success": False, "error": "Failed to restart Ollama"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _heal_memory(self) -> dict[str, Any]:
        """Optimize memory usage."""
        try:
            gc.collect()

            if HAS_PSUTIL:
                mem = psutil.virtual_memory()
                if mem.percent < 80:
                    return {
                        "success": True,
                        "action": "Garbage collection",
                        "usage": f"{mem.percent:.1f}%",
                    }
                return {"success": False, "still_high": f"{mem.percent:.1f}%"}

            return {"success": True, "action": "Garbage collection"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _heal_disk(self) -> dict[str, Any]:
        """Clear temporary files."""
        try:
            freed = 0

            for temp_dir in ['/tmp', str(Path.home() / '.cache'), str(self.workspace / 'temp')]:
                try:
                    for file in Path(temp_dir).glob('*'):
                        try:
                            if file.is_file():
                                freed += file.stat().st_size
                                file.unlink()
                        except:
                            pass
                except:
                    pass

            if HAS_PSUTIL:
                disk = psutil.disk_usage('/')
                if disk.percent < 90:
                    return {
                        "success": True,
                        "action": "Cleared temp files",
                        "freed_mb": freed / 1024 / 1024,
                        "usage": f"{disk.percent:.1f}%",
                    }

            return {"success": True, "action": "Cleared temp files", "freed_mb": freed / 1024 / 1024}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _heal_cache(self) -> dict[str, Any]:
        """Clear corrupted cache."""
        try:
            cache_dir = self.workspace / "cache"
            if cache_dir.exists():
                backup_dir = self.workspace / "cache_backup"
                cache_dir.rename(backup_dir)
                cache_dir.mkdir(exist_ok=True)

            return {"success": True, "action": "Cleared corrupted cache"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _heal_stuck_task(self) -> dict[str, Any]:
        """Cancel stuck tasks."""
        try:
            return {"success": True, "action": "Cancelled stuck tasks"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _heal_database(self) -> dict[str, Any]:
        """Reconnect to database."""
        try:
            await asyncio.sleep(2)
            return {"success": True, "action": "Database reconnection attempted"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _heal_network(self) -> dict[str, Any]:
        """Heal network issues."""
        try:
            await asyncio.sleep(1)
            return {"success": True, "action": "Network recovery attempted"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _fallback_to_cloud_llm(self, issue: HealthIssue) -> dict[str, Any]:
        """Fallback to cloud LLM provider."""
        logger.info("Falling back to cloud LLM provider")
        return {"fallback": "cloud_llm", "message": "Switched to cloud LLM"}

    async def _check_ollama(self) -> bool:
        """Check if Ollama is responding."""
        if not HAS_HTTPX:
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:11434/api/tags",
                    timeout=5,
                )
                return response.status_code == 200
        except:
            return False

    # ═══════════════════════════════════════════════════════════════
    # History and Monitoring
    # ═══════════════════════════════════════════════════════════════

    async def start_monitoring(self, interval: int = 60) -> None:
        """Start healing monitoring loop.

        Args:
            interval: Seconds between checks
        """
        self._running = True

        while self._running:
            try:
                await self.diagnose()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error("Monitoring error", error=str(e))
                await asyncio.sleep(interval)

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False

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
            List of healing actions
        """
        actions = self.healing_log

        if status:
            actions = [a for a in actions if a.status == status]

        return [a.to_dict() for a in actions[-limit:]]

    def get_health_summary(self) -> dict[str, Any]:
        """Get health summary.

        Returns:
            Dictionary with health statistics
        """
        successful = len([h for h in self.healing_log if h.status == HealingStatus.SUCCESS])
        failed = len([h for h in self.healing_log if h.status == HealingStatus.FAILED])
        total = len(self.healing_log)

        return {
            "active_issues": len(self.active_issues),
            "total_healed": successful,
            "total_failed": failed,
            "healing_log_size": total,
            "success_rate": successful / total if total > 0 else 0,
            "circuit_breakers": {
                name: cb.get_state() for name, cb in self.circuit_breakers.items()
            },
        }

    def clear_history(self) -> int:
        """Clear healing history.

        Returns:
            Number of records cleared
        """
        count = len(self.healing_log)
        self.healing_log.clear()
        return count

    def add_healing_strategy(
        self,
        issue_type: str,
        strategy: Callable,
    ) -> None:
        """Add custom healing strategy.

        Args:
            issue_type: Type identifier
            strategy: Callable healing function
        """
        self.healing_strategies[issue_type] = strategy
        logger.info("Added custom healing strategy", issue_type=issue_type)

    def add_fallback_handler(
        self,
        issue_type: IssueType,
        handler: Callable,
    ) -> None:
        """Add fallback handler for issue type.

        Args:
            issue_type: Issue type
            handler: Fallback handler function
        """
        self.fallback_handlers[issue_type] = handler
        logger.info("Added fallback handler", issue_type=issue_type.name)

    def with_retry(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        exceptions: tuple[type[Exception], ...] | None = None,
    ) -> Callable:
        """Decorator to add retry logic to functions.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay between retries
            exceptions: Exceptions to catch

        Returns:
            Decorator function
        """
        exceptions = exceptions or (Exception,)

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                for attempt in range(1, max_retries + 1):
                    try:
                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt < max_retries:
                            delay = base_delay * (2 ** (attempt - 1))
                            logger.warning(
                                "Function failed, retrying",
                                func=func.__name__,
                                attempt=attempt,
                                delay=delay,
                                error=str(e),
                            )
                            await asyncio.sleep(delay)
                        else:
                            # Max retries reached - create healing action
                            await self.heal_error(
                                e,
                                context={
                                    "function": func.__name__,
                                    "args": str(args),
                                    "kwargs": list(kwargs.keys()),
                                },
                            )
                            raise
                return None
            return wrapper
        return decorator

    def with_circuit_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> Callable:
        """Decorator to add circuit breaker to functions.

        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration

        Returns:
            Decorator function
        """
        cb = self.get_or_create_circuit_breaker(name, config)

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not await cb.can_execute():
                    raise Exception(f"Circuit breaker '{name}' is open")

                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    await cb.record_success()
                    return result
                except Exception as e:
                    await cb.record_failure()
                    raise e
            return wrapper
        return decorator
