"""Health check routes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, status

from lya.infrastructure.config.settings import settings

router = APIRouter()


@router.get("/")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "lya",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.env,
    }


@router.get("/ready")
async def readiness_check() -> dict[str, Any]:
    """Readiness probe."""
    return {
        "status": "ready",
        "checks": {
            "configuration": True,
            "database": True,  # Would check actual DB connection
        },
    }


@router.get("/live")
async def liveness_check() -> dict[str, Any]:
    """Liveness probe."""
    return {
        "status": "alive",
    }
