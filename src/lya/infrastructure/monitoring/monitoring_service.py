"""Monitoring Service - Infrastructure Implementation."""

from __future__ import annotations

import asyncio
import json
import psutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from lya.domain.models.healing import HealthCheck, HealthIssue, IssueSeverity
from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


class MonitoringService:
    """
    Service for monitoring system health.

    Tracks:
    - CPU usage
    - Memory usage
    - Disk usage
    - Agent metrics
    """

    def __init__(self, workspace: Path | None = None):
        self.workspace = workspace or Path(settings.workspace_path)
        self.alerts_file = self.workspace / "monitoring" / "alerts.json"
        self.metrics_file = self.workspace / "monitoring" / "metrics.json"
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)

        self.health_checks: list[HealthCheck] = []
        self.alerts: list[dict] = []
        self.running = False
        self._alert_handlers: list[callable] = []

        # Metrics history
        self.metrics_history: list[dict] = []

        logger.info("Monitoring service initialized")

    async def start_monitoring(self, interval: int = 60) -> None:
        """Start continuous monitoring."""
        if not HAS_PSUTIL:
            logger.warning("psutil not installed, monitoring disabled")
            return

        self.running = True
        logger.info("Monitoring started", interval=interval)

        while self.running:
            try:
                await self.run_health_checks()
                await self.check_thresholds()
                await self._collect_metrics()
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error("Monitoring error", error=str(e))
                await asyncio.sleep(10)

    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.running = False
        logger.info("Monitoring stopped")

    async def run_health_checks(self) -> list[HealthCheck]:
        """Run all health checks."""
        if not HAS_PSUTIL:
            return []

        checks = []

        # Memory check
        try:
            mem = psutil.virtual_memory()
            mem_status = "critical" if mem.percent > 95 else "warning" if mem.percent > 80 else "healthy"
            checks.append(HealthCheck(
                check_id=uuid4(),
                name="memory",
                component="system",
                status=mem_status,
                message=f"Memory: {mem.percent}% ({mem.used // 1024 // 1024}MB / {mem.total // 1024 // 1024}MB)",
                timestamp=datetime.now(timezone.utc),
                metrics={"percent": mem.percent, "used_mb": mem.used // 1024 // 1024},
            ))
        except Exception as e:
            logger.warning("Memory check failed", error=str(e))

        # CPU check
        try:
            cpu = psutil.cpu_percent(interval=1)
            cpu_status = "critical" if cpu > 90 else "warning" if cpu > 70 else "healthy"
            checks.append(HealthCheck(
                check_id=uuid4(),
                name="cpu",
                component="system",
                status=cpu_status,
                message=f"CPU: {cpu}%",
                timestamp=datetime.now(timezone.utc),
                metrics={"percent": cpu},
            ))
        except Exception as e:
            logger.warning("CPU check failed", error=str(e))

        # Disk check
        try:
            disk = psutil.disk_usage('/')
            disk_status = "critical" if disk.percent > 95 else "warning" if disk.percent > 80 else "healthy"
            checks.append(HealthCheck(
                check_id=uuid4(),
                name="disk",
                component="system",
                status=disk_status,
                message=f"Disk: {disk.percent}% used",
                timestamp=datetime.now(timezone.utc),
                metrics={"percent": disk.percent, "free_gb": disk.free // 1024 // 1024 // 1024},
            ))
        except Exception as e:
            logger.warning("Disk check failed", error=str(e))

        self.health_checks = checks
        return checks

    async def check_thresholds(self) -> list[dict]:
        """Check thresholds and trigger alerts."""
        triggered = []

        for check in self.health_checks:
            if check.status == "critical":
                alert = await self._trigger_alert(
                    severity=IssueSeverity.CRITICAL,
                    message=check.message,
                    source=check.component,
                    check_name=check.name,
                )
                triggered.append(alert)

            elif check.status == "warning":
                alert = await self._trigger_alert(
                    severity=IssueSeverity.HIGH,
                    message=check.message,
                    source=check.component,
                    check_name=check.name,
                )
                triggered.append(alert)

        return triggered

    async def _trigger_alert(
        self,
        severity: IssueSeverity,
        message: str,
        source: str,
        check_name: str,
    ) -> dict:
        """Trigger an alert."""
        alert = {
            "alert_id": str(uuid4()),
            "level": severity.name,
            "message": message,
            "source": source,
            "check": check_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.alerts.append(alert)

        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        self._save_alerts()

        # Log with emoji
        icon = "🚨" if severity == IssueSeverity.CRITICAL else "⚠️"
        logger.warning(f"{icon} ALERT [{severity.name}]: {message}")

        # Notify handlers
        for handler in self._alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error("Alert handler failed", error=str(e))

        return alert

    async def _collect_metrics(self) -> None:
        """Collect system metrics."""
        if not HAS_PSUTIL:
            return

        try:
            metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
            }

            self.metrics_history.append(metrics)

            # Keep last 1000 metrics
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]

            # Save to file
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f)

        except Exception as e:
            logger.warning("Failed to collect metrics", error=str(e))

    def _save_alerts(self) -> None:
        """Save alerts to file."""
        try:
            with open(self.alerts_file, 'w') as f:
                json.dump(self.alerts, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save alerts", error=str(e))

    def register_alert_handler(self, handler: callable) -> None:
        """Register a handler for alerts."""
        self._alert_handlers.append(handler)

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get dashboard data."""
        healthy = sum(1 for h in self.health_checks if h.status == "healthy")
        warning = sum(1 for h in self.health_checks if h.status == "warning")
        critical = sum(1 for h in self.health_checks if h.status == "critical")

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health_checks": [h.to_dict() for h in self.health_checks],
            "alerts": self.alerts[-20:],
            "summary": {
                "healthy": healthy,
                "warning": warning,
                "critical": critical,
                "total": len(self.health_checks),
            },
        }

    def get_health_summary(self) -> str:
        """Get a quick health summary."""
        if not self.health_checks:
            return "No health checks available"

        healthy = sum(1 for h in self.health_checks if h.status == "healthy")
        warning = sum(1 for h in self.health_checks if h.status == "warning")
        critical = sum(1 for h in self.health_checks if h.status == "critical")

        if critical > 0:
            return f"🚨 {critical} critical issues detected"
        elif warning > 0:
            return f"⚠️ {warning} warnings"
        else:
            return f"✅ All systems healthy ({healthy}/{len(self.health_checks)} checks passed)"

    def get_metrics_history(self, limit: int = 100) -> list[dict]:
        """Get metrics history."""
        return self.metrics_history[-limit:]
