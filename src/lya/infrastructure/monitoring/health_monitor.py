"""Health Monitoring and Healing System for Lya.

Provides self-monitoring, diagnosis, and auto-healing capabilities.
Pure Python 3.14+ compatible.
"""

from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    CRITICAL = auto()


class IssueType(Enum):
    """Types of health issues."""
    MEMORY_HIGH = auto()
    CPU_HIGH = auto()
    DISK_FULL = auto()
    NETWORK_ERROR = auto()
    SERVICE_DOWN = auto()
    ERROR_RATE_HIGH = auto()
    LATENCY_HIGH = auto()
    DEADLOCK = auto()
    CORRUPTION = auto()


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    response_time_ms: float = 0.0


@dataclass
class HealthReport:
    """Overall health report."""
    status: HealthStatus
    checks: list[HealthCheck]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    uptime_seconds: float = 0.0
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.name,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.name,
                    "message": c.message,
                    "details": c.details,
                    "timestamp": c.timestamp,
                    "response_time_ms": c.response_time_ms,
                }
                for c in self.checks
            ],
            "timestamp": self.timestamp,
            "uptime_seconds": self.uptime_seconds,
            "issues": [
                {
                    "type": i.type.name,
                    "severity": i.severity.name,
                    "description": i.description,
                    "detected_at": i.detected_at,
                }
                for i in self.issues
            ],
        }


@dataclass
class Issue:
    """Detected health issue."""
    type: IssueType
    severity: HealthStatus
    description: str
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context: dict[str, Any] = field(default_factory=dict)
    auto_heal_attempts: int = 0
    resolved: bool = False
    resolved_at: str | None = None


@dataclass
class HealingAction:
    """Healing action taken."""
    issue_type: IssueType
    action: str
    success: bool
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HealthChecker:
    """
    Performs health checks on system components.

    Checks memory, CPU, disk, network, and application state.
    """

    def __init__(
        self,
        memory_threshold: float = 85.0,
        cpu_threshold: float = 80.0,
        disk_threshold: float = 90.0,
    ) -> None:
        """Initialize health checker.

        Args:
            memory_threshold: Memory usage threshold (%)
            cpu_threshold: CPU usage threshold (%)
            disk_threshold: Disk usage threshold (%)
        """
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.disk_threshold = disk_threshold

    async def check_memory(self) -> HealthCheck:
        """Check memory usage."""
        start = time.time()

        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_percent = memory.percent

            if usage_percent > self.memory_threshold:
                status = HealthStatus.DEGRADED if usage_percent < 95 else HealthStatus.UNHEALTHY
                message = f"High memory usage: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage: {usage_percent:.1f}%"

            return HealthCheck(
                name="memory",
                status=status,
                message=message,
                details={
                    "percent": usage_percent,
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                },
                response_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check memory: {e}",
                response_time_ms=(time.time() - start) * 1000,
            )

    async def check_cpu(self) -> HealthCheck:
        """Check CPU usage."""
        start = time.time()

        try:
            import psutil
            # Get CPU usage over 1 second
            cpu_percent = psutil.cpu_percent(interval=1)

            if cpu_percent > self.cpu_threshold:
                status = HealthStatus.DEGRADED if cpu_percent < 95 else HealthStatus.UNHEALTHY
                message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage: {cpu_percent:.1f}%"

            return HealthCheck(
                name="cpu",
                status=status,
                message=message,
                details={
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                },
                response_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheck(
                name="cpu",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check CPU: {e}",
                response_time_ms=(time.time() - start) * 1000,
            )

    async def check_disk(self) -> HealthCheck:
        """Check disk usage."""
        start = time.time()

        try:
            import psutil
            disk = psutil.disk_usage("/")
            usage_percent = disk.percent

            if usage_percent > self.disk_threshold:
                status = HealthStatus.DEGRADED if usage_percent < 95 else HealthStatus.UNHEALTHY
                message = f"High disk usage: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage: {usage_percent:.1f}%"

            return HealthCheck(
                name="disk",
                status=status,
                message=message,
                details={
                    "percent": usage_percent,
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                },
                response_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheck(
                name="disk",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check disk: {e}",
                response_time_ms=(time.time() - start) * 1000,
            )

    async def check_network(self) -> HealthCheck:
        """Check network connectivity."""
        start = time.time()

        try:
            import urllib.request
            # Try to reach Google DNS
            urllib.request.urlopen("https://8.8.8.8", timeout=5)

            return HealthCheck(
                name="network",
                status=HealthStatus.HEALTHY,
                message="Network connectivity OK",
                response_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheck(
                name="network",
                status=HealthStatus.UNHEALTHY,
                message=f"Network check failed: {e}",
                response_time_ms=(time.time() - start) * 1000,
            )

    async def check_python_health(self) -> HealthCheck:
        """Check Python interpreter health."""
        start = time.time()

        try:
            # Check garbage collection
            import gc
            gc.collect()
            gc_stats = gc.get_stats()

            return HealthCheck(
                name="python",
                status=HealthStatus.HEALTHY,
                message=f"Python {platform.python_version()} healthy",
                details={
                    "version": platform.python_version(),
                    "gc_stats": str(gc_stats),
                },
                response_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheck(
                name="python",
                status=HealthStatus.UNHEALTHY,
                message=f"Python health check failed: {e}",
                response_time_ms=(time.time() - start) * 1000,
            )

    async def run_all_checks(self) -> list[HealthCheck]:
        """Run all health checks."""
        checks = await asyncio.gather(
            self.check_memory(),
            self.check_cpu(),
            self.check_disk(),
            self.check_network(),
            self.check_python_health(),
        )
        return list(checks)


class HealingEngine:
    """
    Auto-healing engine for Lya.

    Detects issues and applies healing actions.
    """

    def __init__(self) -> None:
        """Initialize healing engine."""
        self._healing_actions: dict[IssueType, list[Callable[[Issue], Awaitable[HealingAction]]]] = {}
        self._healing_history: list[HealingAction] = []

        # Register default healing actions
        self._register_default_healers()

    def _register_default_healers(self) -> None:
        """Register default healing actions."""
        self.register_healer(IssueType.MEMORY_HIGH, self._heal_memory)
        self.register_healer(IssueType.CPU_HIGH, self._heal_cpu)
        self.register_healer(IssueType.DISK_FULL, self._heal_disk)
        self.register_healer(IssueType.SERVICE_DOWN, self._heal_service)

    def register_healer(
        self,
        issue_type: IssueType,
        handler: Callable[[Issue], Awaitable[HealingAction]],
    ) -> None:
        """Register a healing handler.

        Args:
            issue_type: Type of issue this handler addresses
            handler: Async handler function
        """
        if issue_type not in self._healing_actions:
            self._healing_actions[issue_type] = []
        self._healing_actions[issue_type].append(handler)

    async def heal(self, issue: Issue) -> list[HealingAction]:
        """Attempt to heal an issue.

        Args:
            issue: Issue to heal

        Returns:
            List of healing actions taken
        """
        actions: list[HealingAction] = []

        handlers = self._healing_actions.get(issue.type, [])

        for handler in handlers:
            try:
                action = await handler(issue)
                actions.append(action)

                if action.success:
                    issue.resolved = True
                    issue.resolved_at = datetime.now(timezone.utc).isoformat()
                    break

            except Exception as e:
                actions.append(HealingAction(
                    issue_type=issue.type,
                    action=f"{handler.__name__}",
                    success=False,
                    message=f"Healing failed: {e}",
                ))

        issue.auto_heal_attempts += len(actions)
        self._healing_history.extend(actions)

        return actions

    async def _heal_memory(self, issue: Issue) -> HealingAction:
        """Heal high memory usage."""
        try:
            import gc
            gc.collect()

            # Clear caches if possible
            # This would integrate with Lya's cache system

            return HealingAction(
                issue_type=IssueType.MEMORY_HIGH,
                action="garbage_collect",
                success=True,
                message="Garbage collection completed",
            )
        except Exception as e:
            return HealingAction(
                issue_type=IssueType.MEMORY_HIGH,
                action="garbage_collect",
                success=False,
                message=str(e),
            )

    async def _heal_cpu(self, issue: Issue) -> HealingAction:
        """Heal high CPU usage."""
        try:
            # Sleep briefly to reduce load
            await asyncio.sleep(0.5)

            return HealingAction(
                issue_type=IssueType.CPU_HIGH,
                action="throttle",
                success=True,
                message="Throttled execution",
            )
        except Exception as e:
            return HealingAction(
                issue_type=IssueType.CPU_HIGH,
                action="throttle",
                success=False,
                message=str(e),
            )

    async def _heal_disk(self, issue: Issue) -> HealingAction:
        """Heal disk space issue."""
        try:
            import shutil
            # Clear temp directories
            import tempfile
            temp_dir = Path(tempfile.gettempdir())

            cleared = 0
            for item in temp_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                        cleared += 1
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                        cleared += 1
                except Exception:
                    pass

            return HealingAction(
                issue_type=IssueType.DISK_FULL,
                action="clear_temp",
                success=True,
                message=f"Cleared {cleared} temp items",
            )
        except Exception as e:
            return HealingAction(
                issue_type=IssueType.DISK_FULL,
                action="clear_temp",
                success=False,
                message=str(e),
            )

    async def _heal_service(self, issue: Issue) -> HealingAction:
        """Heal service down issue."""
        # This would integrate with service management
        return HealingAction(
            issue_type=IssueType.SERVICE_DOWN,
            action="restart",
            success=False,
            message="Service restart not implemented",
        )

    def get_healing_history(self) -> list[HealingAction]:
        """Get healing action history."""
        return self._healing_history.copy()


class HealthMonitor:
    """
    Main health monitoring system.

    Coordinates health checks, issue detection, and healing.
    """

    def __init__(
        self,
        check_interval_seconds: float = 60.0,
        auto_heal: bool = True,
        max_healing_attempts: int = 3,
    ) -> None:
        """Initialize health monitor.

        Args:
            check_interval_seconds: Time between health checks
            auto_heal: Whether to auto-heal detected issues
            max_healing_attempts: Max healing attempts per issue
        """
        self.checker = HealthChecker()
        self.healer = HealingEngine()
        self.check_interval = check_interval_seconds
        self.auto_heal = auto_heal
        self.max_healing_attempts = max_healing_attempts

        self._running = False
        self._start_time = time.time()
        self._issues: list[Issue] = []
        self._health_history: list[HealthReport] = []
        self._max_history = 100

    def _checks_to_issues(self, checks: list[HealthCheck]) -> list[Issue]:
        """Convert health checks to issues."""
        issues = []

        for check in checks:
            if check.status == HealthStatus.HEALTHY:
                continue

            # Map check to issue type
            issue_type = self._check_to_issue_type(check.name)

            if issue_type:
                # Check if already exists and unresolved
                existing = next(
                    (i for i in self._issues if i.type == issue_type and not i.resolved),
                    None
                )

                if not existing:
                    issues.append(Issue(
                        type=issue_type,
                        severity=check.status,
                        description=check.message,
                        context=check.details,
                    ))

        return issues

    def _check_to_issue_type(self, check_name: str) -> IssueType | None:
        """Map check name to issue type."""
        mapping = {
            "memory": IssueType.MEMORY_HIGH,
            "cpu": IssueType.CPU_HIGH,
            "disk": IssueType.DISK_FULL,
            "network": IssueType.NETWORK_ERROR,
            "python": IssueType.CORRUPTION,
        }
        return mapping.get(check_name)

    def _calculate_overall_status(self, checks: list[HealthCheck]) -> HealthStatus:
        """Calculate overall health status."""
        statuses = [c.status for c in checks]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    async def check_health(self) -> HealthReport:
        """Perform health check."""
        checks = await self.checker.run_all_checks()
        new_issues = self._checks_to_issues(checks)

        # Add new issues
        for issue in new_issues:
            self._issues.append(issue)

        # Auto-heal if enabled
        if self.auto_heal:
            for issue in self._issues:
                if not issue.resolved and issue.auto_heal_attempts < self.max_healing_attempts:
                    await self.healer.heal(issue)

        # Calculate overall status
        status = self._calculate_overall_status(checks)

        report = HealthReport(
            status=status,
            checks=checks,
            uptime_seconds=time.time() - self._start_time,
            issues=[i for i in self._issues if not i.resolved],
        )

        # Store history
        self._health_history.append(report)
        if len(self._health_history) > self._max_history:
            self._health_history.pop(0)

        return report

    async def run(self) -> None:
        """Run monitoring loop."""
        self._running = True

        while self._running:
            try:
                report = await self.check_health()

                if report.status != HealthStatus.HEALTHY:
                    print(f"Health check: {report.status.name}")
                    for issue in report.issues:
                        print(f"  Issue: {issue.type.name} - {issue.description}")

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                print(f"Health monitor error: {e}")
                await asyncio.sleep(self.check_interval)

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False

    def get_latest_report(self) -> HealthReport | None:
        """Get latest health report."""
        if self._health_history:
            return self._health_history[-1]
        return None

    def get_health_history(self, count: int = 10) -> list[HealthReport]:
        """Get recent health history."""
        return self._health_history[-count:]

    def export_report(self, path: str | Path) -> None:
        """Export health report to file.

        Args:
            path: Export path
        """
        report = self.get_latest_report()
        if report:
            Path(path).write_text(json.dumps(report.to_dict(), indent=2))

    def is_healthy(self) -> bool:
        """Quick health check."""
        report = self.get_latest_report()
        if not report:
            return True  # Assume healthy if no checks yet
        return report.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

async def main():
    """Example health monitoring."""
    monitor = HealthMonitor(check_interval_seconds=5)

    # Single check
    report = await monitor.check_health()
    print(f"Health status: {report.status.name}")
    print(f"Uptime: {report.uptime_seconds:.1f}s")
    print("\nChecks:")
    for check in report.checks:
        print(f"  {check.name}: {check.status.name} - {check.message}")

    # Export report
    monitor.export_report("./health_report.json")
    print("\nReport exported to health_report.json")


if __name__ == "__main__":
    asyncio.run(main())
