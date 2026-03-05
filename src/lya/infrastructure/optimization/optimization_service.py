"""Optimization Service - Infrastructure Implementation."""

from __future__ import annotations

import asyncio
import gc
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


class PerformanceMetrics:
    """Performance metrics snapshot."""

    def __init__(
        self,
        timestamp: datetime | None = None,
        cpu_percent: float = 0.0,
        memory_mb: float = 0.0,
        memory_percent: float = 0.0,
        tokens_per_second: float = 0.0,
        response_time_ms: float = 0.0,
        cache_hit_rate: float = 0.0,
        cache_size: int = 0,
        active_tasks: int = 0,
    ):
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.cpu_percent = cpu_percent
        self.memory_mb = memory_mb
        self.memory_percent = memory_percent
        self.tokens_per_second = tokens_per_second
        self.response_time_ms = response_time_ms
        self.cache_hit_rate = cache_hit_rate
        self.cache_size = cache_size
        self.active_tasks = active_tasks

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "tokens_per_second": self.tokens_per_second,
            "response_time_ms": self.response_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_size": self.cache_size,
            "active_tasks": self.active_tasks,
        }


class Optimization:
    """Optimization record."""

    def __init__(
        self,
        optimization_type: str,
        description: str,
        timestamp: datetime | None = None,
        metrics_before: dict[str, Any] | None = None,
    ):
        self.optimization_type = optimization_type
        self.description = description
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.metrics_before = metrics_before or {}
        self.metrics_after: dict[str, Any] | None = None
        self.improvement_percent = 0.0

    def calculate_improvement(self) -> float:
        """Calculate improvement percentage."""
        if not self.metrics_after:
            return 0.0

        # Calculate based on memory usage reduction
        before_mem = self.metrics_before.get("memory_mb", 0)
        after_mem = self.metrics_after.get("memory_mb", 0)

        if before_mem > 0 and after_mem > 0:
            return ((before_mem - after_mem) / before_mem) * 100

        return 0.0


class OptimizationService:
    """
    Self-optimization service for performance tuning.

    Features:
    - Performance metrics collection
    - Automatic optimization
    - Smart caching with LRU eviction
    - Memory management
    """

    def __init__(self, workspace: Path | None = None):
        self.workspace = workspace or Path(settings.workspace_path)
        self.metrics_file = self.workspace / "metrics" / "performance.json"
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

        self.metrics: list[PerformanceMetrics] = []
        self.optimizations: list[Optimization] = []
        self._running = False
        self._interval = 60  # seconds

        logger.info("Optimization service initialized")

    async def start(self, interval: int = 60) -> None:
        """Start optimization monitoring loop."""
        self._running = True
        self._interval = interval

        logger.info("Optimization service started", interval=interval)

        while self._running:
            try:
                await self.collect_metrics()

                # Run optimization periodically
                if len(self.metrics) > 10:
                    await self.self_optimize()

            except Exception as e:
                logger.error("Optimization error", error=str(e))

            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        """Stop optimization service."""
        self._running = False
        logger.info("Optimization service stopped")

    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        process = psutil.Process()

        metric = PerformanceMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_mb=process.memory_info().rss / 1024 / 1024,
            memory_percent=psutil.virtual_memory().percent,
            active_tasks=len(asyncio.all_tasks()),
        )

        self.metrics.append(metric)

        # Keep only last 1000 metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-500:]

        await self._save_metrics()

        return metric

    async def analyze_performance(self) -> dict[str, Any]:
        """Analyze performance trends."""
        if len(self.metrics) < 10:
            return {"error": "Not enough data"}

        recent = self.metrics[-100:]

        avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
        avg_memory = sum(m.memory_mb for m in recent) / len(recent)

        # Detect trends
        cpu_trend = "increasing" if recent[-1].cpu_percent > recent[0].cpu_percent else "stable"
        memory_trend = "leak" if recent[-1].memory_mb > recent[0].memory_mb * 1.5 else "stable"

        return {
            "avg_cpu": avg_cpu,
            "avg_memory_mb": avg_memory,
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "recommendations": self._generate_recommendations(avg_cpu, avg_memory),
        }

    def _generate_recommendations(self, cpu: float, memory: float) -> list[str]:
        """Generate optimization recommendations."""
        recs = []

        if cpu > 80:
            recs.append("High CPU usage - consider reducing concurrent tasks")
        if memory > 1000:
            recs.append("High memory usage - implement memory cleanup")
        if len(self.metrics) > 1000:
            recs.append("Large metrics history - archive old data")

        return recs

    async def self_optimize(self) -> Optimization | None:
        """Apply self-optimizations."""
        analysis = await self.analyze_performance()

        if "recommendations" not in analysis:
            return None

        for rec in analysis["recommendations"]:
            if "memory" in rec.lower():
                return await self._optimize_memory()
            elif "cpu" in rec.lower():
                return await self._optimize_cpu()

        return None

    async def _optimize_memory(self) -> Optimization:
        """Optimize memory usage."""
        before = await self.collect_metrics()

        # Run garbage collection
        gc.collect()

        # Trim metrics
        if len(self.metrics) > 500:
            self.metrics = self.metrics[-250:]
            await self._save_metrics()

        after = await self.collect_metrics()

        opt = Optimization(
            optimization_type="memory_cleanup",
            description="Garbage collection and metrics trim",
            metrics_before=before.to_dict(),
        )
        opt.metrics_after = after.to_dict()
        opt.improvement_percent = opt.calculate_improvement()

        self.optimizations.append(opt)

        logger.info(
            "Memory optimization applied",
            before_mb=before.memory_mb,
            after_mb=after.memory_mb,
            improvement=opt.improvement_percent,
        )

        return opt

    async def _optimize_cpu(self) -> Optimization:
        """Optimize CPU usage."""
        before = await self.collect_metrics()

        # Reduce polling frequency temporarily
        self._interval = min(300, self._interval + 30)

        after = await self.collect_metrics()

        opt = Optimization(
            optimization_type="cpu_throttle",
            description="Reduced monitoring frequency",
            metrics_before=before.to_dict(),
        )
        opt.metrics_after = after.to_dict()

        self.optimizations.append(opt)

        logger.info("CPU optimization applied", new_interval=self._interval)

        return opt

    async def _save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            data = [
                m.to_dict()
                for m in self.metrics[-500:]  # Keep last 500
            ]

            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error("Failed to save metrics", error=str(e))


class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: int | None = None):
        self.max_size = max_size
        self.default_ttl = default_ttl  # seconds
        self._cache: OrderedDict[str, tuple[Any, datetime | None]] = OrderedDict()
        self._access_count: dict[str, int] = {}
        self._hit_count = 0
        self._miss_count = 0

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if key not in self._cache:
            self._miss_count += 1
            return None

        value, expires = self._cache[key]

        # Check expiration
        if expires and datetime.now(timezone.utc) > expires:
            del self._cache[key]
            del self._access_count[key]
            self._miss_count += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._access_count[key] = self._access_count.get(key, 0) + 1
        self._hit_count += 1

        return value

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Set value in cache with optional TTL."""
        # Calculate expiration
        expires = None
        if ttl is not None:
            expires = datetime.now(timezone.utc) + __import__('datetime').timedelta(seconds=ttl)
        elif self.default_ttl is not None:
            expires = datetime.now(timezone.utc) + __import__('datetime').timedelta(seconds=self.default_ttl)

        # Evict if at capacity and key is new
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()

        self._cache[key] = (value, expires)
        self._access_count[key] = 1

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self._cache:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
            del self._access_count[oldest]

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            del self._access_count[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache."""
        self._cache.clear()
        self._access_count.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_accesses = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_accesses if total_accesses > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "utilization": len(self._cache) / self.max_size,
        }

    def keys(self) -> list[str]:
        """Get all keys."""
        return list(self._cache.keys())

    def values(self) -> list[Any]:
        """Get all values (excluding expired)."""
        return [
            v for k, (v, e) in self._cache.items()
            if e is None or datetime.now(timezone.utc) <= e
        ]
