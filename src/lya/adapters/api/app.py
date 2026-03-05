"""FastAPI application for Lya REST API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lya.infrastructure.config.logging import get_logger, configure_logging
from lya.infrastructure.config.settings import settings
from lya.adapters.api.routes import health, goals, tasks, memories, capabilities
from lya.adapters.api.middleware import LoggingMiddleware, RateLimitMiddleware
from lya.adapters.api.websocket import ConnectionManager

logger = get_logger(__name__)

# WebSocket connection manager
ws_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    configure_logging()
    logger.info(
        "Lya API starting",
        version="1.0.0",
        environment=settings.env,
    )

    # Initialize services
    await ws_manager.initialize()

    yield

    # Shutdown
    logger.info("Lya API shutting down")
    await ws_manager.close()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Lya API",
        description="Autonomous AGI Agent API",
        version="1.0.0",
        docs_url="/docs" if settings.env == "development" else None,
        redoc_url="/redoc" if settings.env == "development" else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(goals.router, prefix="/api/v1/goals", tags=["Goals"])
    app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["Tasks"])
    app.include_router(memories.router, prefix="/api/v1/memories", tags=["Memories"])
    app.include_router(capabilities.router, prefix="/api/v1/capabilities", tags=["Capabilities"])

    # Static files for dashboard
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def serve_dashboard():
            """Serve the dashboard HTML."""
            index_file = static_dir / "index.html"
            if index_file.exists():
                return FileResponse(str(index_file))
            raise HTTPException(status_code=404, detail="Dashboard not found")

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time communication."""
        await ws_manager.connect(websocket)

        try:
            while True:
                # Receive message
                data = await websocket.receive_json()

                # Handle message
                response = await ws_manager.handle_message(websocket, data)

                # Send response
                if response:
                    await websocket.send_json(response)

        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)
        except Exception as e:
            logger.error("WebSocket error", error=str(e))
            await ws_manager.disconnect(websocket)

    return app


# Create app instance
app = create_app()
