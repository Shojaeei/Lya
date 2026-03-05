"""Monitoring System - Infrastructure Implementation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


class HealthCheck:
    """Health check result."""

    def __init__(
        self,
        name: str,
        status: str,  # healthy, warning, critical
        message: str,
        metrics: dict[str, Any] | None = None,
    ):
        self.name = name
        self.status = status
        self.message = message
        self.timestamp = datetime.now(timezone.utc)
        self.metrics = metrics or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
        }


class Alert:
    """System alert."""

    def __init__(
        self,
        level: str,  # info, warning, critical
        message: str,
        source: str,
        metrics: dict[str, Any] | None = None,
    ):
        self.level = level
        self.message = message
        self.source = source
        self.timestamp = datetime.now(timezone.utc)
        self.metrics = metrics or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
        }


class MonitoringSystem:
    """
    System monitoring and health checks.

    Features:
    - Resource monitoring (CPU, memory, disk)
    - Health checks
    - Alert management
    - Dashboard data
    - Alert handlers
    """

    def __init__(self, workspace: Path | None = None):
        self.workspace = workspace or Path(settings.workspace_path)
        self.alerts_file = self.workspace / "monitoring" / "alerts.json"
        self.metrics_file = self.workspace / "monitoring" / "metrics.json"
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)

        self.health_checks: list[HealthCheck] = []
        self.alerts: list[Alert] = []
        self.alert_handlers: list[Callable] = []
        self._running = False
        self._interval = 60  # seconds

        # Thresholds
        self.thresholds = {
            "memory_warning": 80,
            "memory_critical": 95,
            "cpu_warning": 70,
            "cpu_critical": 90,
            "disk_warning": 80,
            "disk_critical": 95,
        }

        logger.info("Monitoring system initialized")

    def add_alert_handler(self, handler: Callable) -> None:
        """Add alert handler callback."""
        self.alert_handlers.append(handler)

    async def start(self, interval: int = 60) -> None:
        """Start monitoring loop."""
        self._running = True
        self._interval = interval

        logger.info("Monitoring started", interval=interval)

        while self._running:
            try:
                await self.run_health_checks()
                await self.check_thresholds()
                await self._save_metrics()
            except Exception as e:
                logger.error("Monitoring error", error=str(e))

            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        logger.info("Monitoring stopped")

    async def run_health_checks(self) -> list[HealthCheck]:
        """Run all health checks."""
        checks = []

        # Memory check
        mem = psutil.virtual_memory()
        if mem.percent > self.thresholds["memory_critical"]:
            mem_status = "critical"
        elif mem.percent > self.thresholds["memory_warning"]:
            mem_status = "warning"
        else:
            mem_status = "healthy"

        checks.append(HealthCheck(
            name="memory",
            status=mem_status,
            message=f"{mem.percent:.1f}% used ({mem.used / 1024 / 1024 / 1024:.1f} GB / {mem.total / 1024 / 1024 / 1024:.1f} GB)",
            metrics={
                "percent": mem.percent,
                "used_gb": mem.used / 1024 / 1024 / 1024,
                "total_gb": mem.total / 1024 / 1024 / 1024,
                "available_gb": mem.available / 1024 / 1024 / 1024,
            },
        ))

        # CPU check
        cpu = psutil.cpu_percent(interval=1)
        if cpu > self.thresholds["cpu_critical"]:
            cpu_status = "critical"
        elif cpu > self.thresholds["cpu_warning"]:
            cpu_status = "warning"
        else:
            cpu_status = "healthy"

        checks.append(HealthCheck(
            name="cpu",
            status=cpu_status,
            message=f"{cpu:.1f}% usage",
            metrics={"percent": cpu, "count": psutil.cpu_count(), "freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0},
        ))

        # Disk check
        disk = psutil.disk_usage('/')
        if disk.percent > self.thresholds["disk_critical"]:
            disk_status = "critical"
        elif disk.percent > self.thresholds["disk_warning"]:
            disk_status = "warning"
        else:
            disk_status = "healthy"

        checks.append(HealthCheck(
            name="disk",
            status=disk_status,
            message=f"{disk.percent:.1f}% used ({disk.used / 1024 / 1024 / 1024:.1f} GB / {disk.total / 1024 / 1024 / 1024:.1f} GB)",
            metrics={
                "percent": disk.percent,
                "used_gb": disk.used / 1024 / 1024 / 1024,
                "total_gb": disk.total / 1024 / 1024 / 1024,
                "free_gb": disk.free / 1024 / 1024 / 1024,
            },
        ))

        # Process check (if running)
        try:
            process = psutil.Process()
            checks.append(HealthCheck(
                name="process",
                status="healthy",
                message=f"Running (PID: {process.pid})",
                metrics={
                    "pid": process.pid,
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "cpu_percent": process.cpu_percent(),
                    "threads": process.num_threads(),
                },
            ))
        except Exception:
            pass

        self.health_checks = checks
        return checks

    async def check_thresholds(self) -> None:
        """Check thresholds and trigger alerts."""
        for check in self.health_checks:
            if check.status == "critical":
                await self.trigger_alert(
                    level="critical",
                    message=f"{check.name}: {check.message}",
                    source=check.name,
                    metrics=check.metrics,
                )
            elif check.status == "warning":
                await self.trigger_alert(
                    level="warning",
                    message=f"{check.name}: {check.message}",
                    source=check.name,
                    metrics=check.metrics,
                )

    async def trigger_alert(
        self,
        level: str,
        message: str,
        source: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Trigger an alert."""
        alert = Alert(
            level=level,
            message=message,
            source=source,
            metrics=metrics,
        )

        self.alerts.append(alert)

        # Keep only recent alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]

        # Log alert
        icon = "🚨" if level == "critical" else "⚠️" if level == "warning" else "ℹ️"
        logger.warning(f"{icon} ALERT [{level.upper()}]: {message}", source=source)

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error("Alert handler failed", error=str(e))

        await self._save_alerts()

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get dashboard data."""
        healthy = sum(1 for h in self.health_checks if h.status == "healthy")
        warning = sum(1 for h in self.health_checks if h.status == "warning")
        critical = sum(1 for h in self.health_checks if h.status == "critical")

        recent_alerts = self.alerts[-20:]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health_checks": [h.to_dict() for h in self.health_checks],
            "alerts": [a.to_dict() for a in recent_alerts],
            "summary": {
                "healthy": healthy,
                "warning": warning,
                "critical": critical,
                "total_alerts": len(self.alerts),
            },
            "system": {
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                "uptime_seconds": (datetime.now() - datetime.fromtimestamp(psutil.boot_time())).total_seconds(),
            },
        }

    async def _save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": [h.to_dict() for h in self.health_checks],
            }

            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save metrics", error=str(e))

    async def _save_alerts(self) -> None:
        """Save alerts to file."""
        try:
            with open(self.alerts_file, 'w', encoding='utf-8') as f:
                json.dump([a.to_dict() for a in self.alerts[-100:]], f, indent=2)
        except Exception as e:
            logger.error("Failed to save alerts", error=str(e))

    def get_health_summary(self) -> dict[str, Any]:
        """Get quick health summary."""
        overall = "healthy"
        if any(h.status == "critical" for h in self.health_checks):
            overall = "critical"
        elif any(h.status == "warning" for h in self.health_checks):
            overall = "warning"

        return {
            "status": overall,
            "checks": len(self.health_checks),
            "healthy": sum(1 for h in self.health_checks if h.status == "healthy"),
            "warnings": sum(1 for h in self.health_checks if h.status == "warning"),
            "critical": sum(1 for h in self.health_checks if h.status == "critical"),
        }
