"""Optimization Engine - Infrastructure Implementation."""

from __future__ import annotations

import asyncio
import gc
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import psutil

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


class PerformanceMetrics:
    """Performance metrics snapshot."""

    def __init__(
        self,
        cpu_percent: float = 0.0,
        memory_mb: float = 0.0,
        memory_percent: float = 0.0,
        tokens_per_second: float = 0.0,
        response_time_ms: float = 0.0,
        cache_hit_rate: float = 0.0,
        cache_size: int = 0,
        active_tasks: int = 0,
    ):
        self.metric_id = uuid4()
        self.timestamp = datetime.now(timezone.utc)
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
            "metric_id": str(self.metric_id),
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


class CacheOptimizer:
    """
    LRU Cache with intelligent eviction.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.access_count: dict[str, int] = {}
        self.created_at: dict[str, datetime] = {}
        self.last_accessed: dict[str, datetime] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        """Get from cache."""
        if key in self.cache:
            # Check TTL
            if self._is_expired(key):
                self._remove(key)
                self.misses += 1
                return None

            # Update access stats
            self.access_count[key] += 1
            self.last_accessed[key] = datetime.now(timezone.utc)
            self.cache.move_to_end(key)

            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set cache with LRU eviction."""
        now = datetime.now(timezone.utc)

        # Evict if needed
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()

        # Set value
        self.cache[key] = value
        self.cache.move_to_end(key)

        if key not in self.access_count:
            self.created_at[key] = now
            self.access_count[key] = 0

        self.last_accessed[key] = now

        # Set custom TTL if provided
        if ttl:
            self.created_at[key] = now

    def _is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        if key not in self.created_at:
            return False
        age = (datetime.now(timezone.utc) - self.created_at[key]).total_seconds()
        return age > self.ttl_seconds

    def _remove(self, key: str) -> None:
        """Remove key from cache."""
        self.cache.pop(key, None)
        self.access_count.pop(key, None)
        self.created_at.pop(key, None)
        self.last_accessed.pop(key, None)

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self.cache:
            oldest = next(iter(self.cache))
            self._remove(oldest)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "hits": self.hits,
            "misses": self.misses,
            "unique_keys": len(self.cache),
        }

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()
        self.created_at.clear()
        self.last_accessed.clear()


class OptimizationEngine:
    """
    Self-optimization engine for performance tuning.

    Features:
    - Performance metrics collection
    - Trend analysis
    - Automatic optimization
    - Cache management
    - Memory cleanup
    """

    def __init__(self, workspace: Path | None = None):
        self.workspace = workspace or Path(settings.workspace_path)
        self.metrics_file = self.workspace / "metrics" / "performance.json"
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

        self.metrics: list[PerformanceMetrics] = []
        self.optimizations: list[dict[str, Any]] = []
        self.cache = CacheOptimizer()
        self.learning_rate = 0.1
        self._running = False

        # Load existing metrics
        self._load_metrics()

        logger.info("Optimization engine initialized")

    def _load_metrics(self) -> None:
        """Load historical metrics."""
        if not self.metrics_file.exists():
            return

        try:
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load recent metrics (keep last 1000)
            self.metrics = [
                PerformanceMetrics(
                    cpu_percent=m.get("cpu_percent", 0),
                    memory_mb=m.get("memory_mb", 0),
                    memory_percent=m.get("memory_percent", 0),
                    tokens_per_second=m.get("tokens_per_second", 0),
                    response_time_ms=m.get("response_time_ms", 0),
                    cache_hit_rate=m.get("cache_hit_rate", 0),
                    cache_size=m.get("cache_size", 0),
                    active_tasks=m.get("active_tasks", 0),
                )
                for m in data.get("metrics", [])[-1000:]
            ]

        except Exception as e:
            logger.error("Failed to load metrics", error=str(e))

    async def start(self, interval: int = 60) -> None:
        """Start optimization loop."""
        self._running = True

        while self._running:
            try:
                await self.collect_metrics()

                if len(self.metrics) >= 10:
                    analysis = self.analyze_performance()
                    if "recommendations" in analysis:
                        for rec in analysis["recommendations"]:
                            await self.apply_optimization(rec)

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error("Optimization loop error", error=str(e))
                await asyncio.sleep(interval)

    async def stop(self) -> None:
        """Stop optimization."""
        self._running = False

    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            metric = PerformanceMetrics(
                cpu_percent=psutil.cpu_percent(interval=1),
                memory_mb=memory_info.rss / 1024 / 1024,
                memory_percent=memory_percent,
                tokens_per_second=0.0,  # Would track from LLM
                response_time_ms=0.0,   # Would track from responses
                cache_hit_rate=self.cache.get_stats()["hit_rate"],
                cache_size=len(self.cache.cache),
                active_tasks=0,  # Would track from executor
            )

            self.metrics.append(metric)

            # Keep only last 1000 metrics
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]

            await self._save_metrics()

            return metric

        except Exception as e:
            logger.error("Failed to collect metrics", error=str(e))
            return PerformanceMetrics()

    def analyze_performance(self) -> dict[str, Any]:
        """Analyze performance trends."""
        if len(self.metrics) < 10:
            return {"error": "Not enough data"}

        recent = self.metrics[-100:]  # Last 100 measurements

        avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
        avg_memory = sum(m.memory_mb for m in recent) / len(recent)
        avg_memory_percent = sum(m.memory_percent for m in recent) / len(recent)

        # Detect trends
        cpu_trend = "increasing" if recent[-1].cpu_percent > recent[0].cpu_percent else "stable"
        memory_trend = "leak" if recent[-1].memory_mb > recent[0].memory_mb * 1.5 else "stable"

        # Generate recommendations
        recommendations = []

        if avg_cpu > 80:
            recommendations.append({"type": "cpu", "action": "throttle", "priority": "high"})
        if avg_memory_percent > 85:
            recommendations.append({"type": "memory", "action": "cleanup", "priority": "critical"})
        if self.cache.get_stats()["hit_rate"] < 0.5:
            recommendations.append({"type": "cache", "action": "resize", "priority": "medium"})
        if len(self.metrics) > 5000:
            recommendations.append({"type": "metrics", "action": "archive", "priority": "low"})

        return {
            "avg_cpu": avg_cpu,
            "avg_memory_mb": avg_memory,
            "avg_memory_percent": avg_memory_percent,
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "cache_stats": self.cache.get_stats(),
            "recommendations": recommendations,
        }

    async def apply_optimization(self, recommendation: dict[str, Any]) -> dict[str, Any]:
        """Apply an optimization."""
        opt_type = recommendation.get("type")
        action = recommendation.get("action")

        optimization = {
            "optimization_id": str(uuid4()),
            "type": opt_type,
            "action": action,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "applied",
        }

        try:
            if opt_type == "memory":
                await self._optimize_memory()
                optimization["result"] = "Memory optimized"
            elif opt_type == "cpu":
                await self._optimize_cpu()
                optimization["result"] = "CPU throttled"
            elif opt_type == "cache":
                await self._optimize_cache()
                optimization["result"] = "Cache optimized"
            elif opt_type == "metrics":
                await self._archive_metrics()
                optimization["result"] = "Metrics archived"

            optimization["status"] = "success"

        except Exception as e:
            optimization["status"] = "failed"
            optimization["error"] = str(e)
            logger.error("Optimization failed", opt_type=opt_type, error=str(e))

        self.optimizations.append(optimization)
        return optimization

    async def _optimize_memory(self) -> None:
        """Optimize memory usage."""
        # Force garbage collection
        gc.collect()

        # Trim metrics history
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-500:]
            await self._save_metrics()

        logger.info("Memory optimization applied")

    async def _optimize_cpu(self) -> None:
        """Optimize CPU usage."""
        # Add small delay to reduce CPU load
        await asyncio.sleep(0.1)
        logger.info("CPU optimization applied")

    async def _optimize_cache(self) -> None:
        """Optimize cache."""
        stats = self.cache.get_stats()

        # If hit rate is low, clear and resize
        if stats["hit_rate"] < 0.3:
            self.cache.clear()
            logger.info("Cache cleared due to low hit rate")

    async def _archive_metrics(self) -> None:
        """Archive old metrics."""
        if len(self.metrics) > 1000:
            archived = self.metrics[:-500]
            self.metrics = self.metrics[-500:]

            # Save archive
            archive_file = self.workspace / "metrics" / f"metrics_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            with open(archive_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {"metrics": [m.to_dict() for m in archived]},
                    f,
                    indent=2,
                )

            await self._save_metrics()
            logger.info(f"Archived {len(archived)} metrics")

    async def _save_metrics(self) -> None:
        """Save metrics to disk."""
        try:
            data = {
                "metrics": [m.to_dict() for m in self.metrics[-1000:]],
                "optimizations": self.optimizations[-100:],
            }

            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error("Failed to save metrics", error=str(e))

    def get_cache(self) -> CacheOptimizer:
        """Get cache instance."""
        return self.cache

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get optimization summary."""
        recent = self.optimizations[-10:]

        return {
            "total_optimizations": len(self.optimizations),
            "recent_optimizations": recent,
            "cache_stats": self.cache.get_stats(),
            "metrics_count": len(self.metrics),
        }
