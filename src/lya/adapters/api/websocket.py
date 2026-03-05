"""WebSocket connection manager."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import UUID, uuid4

from fastapi import WebSocket

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class WebSocketConnection:
    """Represents a WebSocket connection."""

    def __init__(self, websocket: WebSocket, connection_id: str):
        self.websocket = websocket
        self.id = connection_id
        self.agent_id: str | None = None
        self.subscriptions: set[str] = set()
        self.connected_at = asyncio.get_event_loop().time()
        self.last_ping: float = self.connected_at

    async def send(self, message: dict[str, Any]) -> None:
        """Send message to connection."""
        await self.websocket.send_json(message)


class ConnectionManager:
    """
    Manages WebSocket connections.

    Features:
    - Connection tracking
    - Message broadcasting
    - Event subscription
    - Ping/pong heartbeat
    """

    def __init__(self):
        self._connections: dict[str, WebSocketConnection] = {}
        self._agent_connections: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()
        self._running = False

    async def initialize(self) -> None:
        """Initialize the manager."""
        self._running = True
        asyncio.create_task(self._heartbeat_loop())
        logger.info("WebSocket manager initialized")

    async def close(self) -> None:
        """Close all connections."""
        self._running = False

        async with self._lock:
            for conn in list(self._connections.values()):
                try:
                    await conn.websocket.close()
                except Exception:
                    pass
            self._connections.clear()
            self._agent_connections.clear()

        logger.info("WebSocket manager closed")

    async def connect(self, websocket: WebSocket) -> WebSocketConnection:
        """Accept and track a new connection."""
        await websocket.accept()

        connection_id = str(uuid4())
        connection = WebSocketConnection(websocket, connection_id)

        async with self._lock:
            self._connections[connection_id] = connection

        logger.info("WebSocket connected", connection_id=connection_id)
        return connection

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove and close a connection."""
        async with self._lock:
            # Find connection
            conn_id = None
            for cid, conn in self._connections.items():
                if conn.websocket == websocket:
                    conn_id = cid
                    break

            if conn_id:
                conn = self._connections[conn_id]

                # Remove from agent connections
                if conn.agent_id and conn.agent_id in self._agent_connections:
                    self._agent_connections[conn.agent_id].discard(conn_id)

                # Remove from connections
                del self._connections[conn_id]

                logger.info("WebSocket disconnected", connection_id=conn_id)

    async def handle_message(
        self,
        websocket: WebSocket,
        data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Handle incoming WebSocket message."""
        message_type = data.get("type", "unknown")
        connection = self._get_connection(websocket)

        if not connection:
            return None

        handlers = {
            "authenticate": self._handle_authenticate,
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe,
            "ping": self._handle_ping,
            "message": self._handle_message,
        }

        handler = handlers.get(message_type, self._handle_unknown)
        return await handler(connection, data)

    def _get_connection(self, websocket: WebSocket) -> WebSocketConnection | None:
        """Get connection by WebSocket."""
        for conn in self._connections.values():
            if conn.websocket == websocket:
                return conn
        return None

    async def _handle_authenticate(
        self,
        connection: WebSocketConnection,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle authentication message."""
        agent_id = data.get("agent_id")

        if not agent_id:
            return {
                "type": "error",
                "error": "Missing agent_id",
            }

        connection.agent_id = agent_id

        async with self._lock:
            if agent_id not in self._agent_connections:
                self._agent_connections[agent_id] = set()
            self._agent_connections[agent_id].add(connection.id)

        logger.info("WebSocket authenticated", connection_id=connection.id, agent_id=agent_id)

        return {
            "type": "authenticated",
            "connection_id": connection.id,
        }

    async def _handle_subscribe(
        self,
        connection: WebSocketConnection,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle subscription message."""
        channel = data.get("channel")

        if not channel:
            return {
                "type": "error",
                "error": "Missing channel",
            }

        connection.subscriptions.add(channel)

        return {
            "type": "subscribed",
            "channel": channel,
        }

    async def _handle_unsubscribe(
        self,
        connection: WebSocketConnection,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle unsubscription message."""
        channel = data.get("channel")

        if channel:
            connection.subscriptions.discard(channel)

        return {
            "type": "unsubscribed",
            "channel": channel,
        }

    async def _handle_ping(
        self,
        connection: WebSocketConnection,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle ping message."""
        connection.last_ping = asyncio.get_event_loop().time()
        return {"type": "pong"}

    async def _handle_message(
        self,
        connection: WebSocketConnection,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle chat/action message."""
        content = data.get("content", "")

        # TODO: Connect to agent's message handler
        return {
            "type": "message",
            "content": f"Message received: {content}",
            "agent_id": connection.agent_id,
            "status": "not_implemented",
        }

    async def _handle_unknown(
        self,
        connection: WebSocketConnection,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle unknown message type."""
        return {
            "type": "error",
            "error": f"Unknown message type: {data.get('type')}",
        }

    # ═══════════════════════════════════════════════════════════════
    # Broadcasting
    # ═══════════════════════════════════════════════════════════════

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connections."""
        disconnected = []

        for conn in self._connections.values():
            try:
                await conn.send(message)
            except Exception:
                disconnected.append(conn.id)

        # Clean up disconnected
        for conn_id in disconnected:
            if conn_id in self._connections:
                del self._connections[conn_id]

    async def broadcast_to_agent(
        self,
        agent_id: str,
        message: dict[str, Any],
    ) -> None:
        """Broadcast message to an agent's connections."""
        if agent_id not in self._agent_connections:
            return

        connection_ids = self._agent_connections[agent_id].copy()
        disconnected = []

        for conn_id in connection_ids:
            if conn_id in self._connections:
                try:
                    await self._connections[conn_id].send(message)
                except Exception:
                    disconnected.append(conn_id)
            else:
                disconnected.append(conn_id)

        # Clean up disconnected
        for conn_id in disconnected:
            self._agent_connections[agent_id].discard(conn_id)

    async def broadcast_to_channel(
        self,
        channel: str,
        message: dict[str, Any],
    ) -> None:
        """Broadcast message to channel subscribers."""
        for conn in self._connections.values():
            if channel in conn.subscriptions:
                try:
                    await conn.send(message)
                except Exception:
                    pass

    # ═══════════════════════════════════════════════════════════════
    # Heartbeat
    # ═══════════════════════════════════════════════════════════════

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats and clean up stale connections."""
        while self._running:
            await asyncio.sleep(30)  # 30 second interval

            if not self._running:
                break

            current_time = asyncio.get_event_loop().time()
            disconnected = []

            for conn_id, conn in list(self._connections.items()):
                # Check if stale (no ping for 5 minutes)
                if current_time - conn.last_ping > 300:
                    disconnected.append(conn_id)
                    continue

                # Send ping
                try:
                    await conn.send({"type": "ping"})
                except Exception:
                    disconnected.append(conn_id)

            # Clean up stale connections
            for conn_id in disconnected:
                if conn_id in self._connections:
                    conn = self._connections[conn_id]
                    if conn.agent_id and conn.agent_id in self._agent_connections:
                        self._agent_connections[conn.agent_id].discard(conn_id)
                    del self._connections[conn_id]
                    logger.info("Cleaned up stale connection", connection_id=conn_id)

    # ═══════════════════════════════════════════════════════════════
    # Stats
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self._connections),
            "agent_connections": {
                agent_id: len(conn_ids)
                for agent_id, conn_ids in self._agent_connections.items()
            },
        }
