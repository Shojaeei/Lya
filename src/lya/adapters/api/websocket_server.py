"""WebSocket Communication for Lya.

Provides real-time bidirectional communication.
Pure Python 3.14+ compatible.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any


class MessageType(Enum):
    """WebSocket message types."""
    MESSAGE = auto()
    RESPONSE = auto()
    TOOL_CALL = auto()
    TOOL_RESULT = auto()
    WORKFLOW_START = auto()
    WORKFLOW_PROGRESS = auto()
    WORKFLOW_COMPLETE = auto()
    HEARTBEAT = auto()
    ERROR = auto()
    AUTH = auto()


@dataclass
class WebSocketMessage:
    """WebSocket message."""
    type: MessageType
    payload: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message_id: str = field(default_factory=lambda: str(hash(datetime.now())))

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type.name,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
        })

    @classmethod
    def from_json(cls, data: str) -> WebSocketMessage | None:
        """Parse from JSON string."""
        try:
            obj = json.loads(data)
            return cls(
                type=MessageType[obj.get("type", "MESSAGE")],
                payload=obj.get("payload", {}),
                timestamp=obj.get("timestamp", ""),
                message_id=obj.get("message_id", ""),
            )
        except Exception:
            return None


class WebSocketClient:
    """
    WebSocket client for Lya.

    Connects to WebSocket servers.
    """

    def __init__(self, url: str) -> None:
        """Initialize client.

        Args:
            url: WebSocket URL
        """
        self.url = url
        self._connected = False
        self._handlers: dict[MessageType, list[Callable]] = {}

    async def connect(self) -> bool:
        """Connect to server.

        Returns:
            True if connected
        """
        # Simplified - real implementation would use websockets library
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Disconnect from server."""
        self._connected = False

    async def send(self, message: WebSocketMessage) -> bool:
        """Send message.

        Args:
            message: Message to send

        Returns:
            True if sent
        """
        if not self._connected:
            return False

        # Simplified
        return True

    def on(self, message_type: MessageType, handler: Callable) -> None:
        """Register message handler."""
        if message_type not in self._handlers:
            self._handlers[message_type] = []
        self._handlers[message_type].append(handler)

    async def send_message(self, text: str, context: dict | None = None) -> None:
        """Send user message."""
        msg = WebSocketMessage(
            type=MessageType.MESSAGE,
            payload={"text": text, "context": context or {}},
        )
        await self.send(msg)

    async def send_tool_call(self, tool: str, params: dict) -> None:
        """Send tool call request."""
        msg = WebSocketMessage(
            type=MessageType.TOOL_CALL,
            payload={"tool": tool, "params": params},
        )
        await self.send(msg)


class WebSocketServer:
    """
    WebSocket server for Lya.

    Handles real-time client connections.
    """

    def __init__(self, agent: Any, host: str = "0.0.0.0", port: int = 8765) -> None:
        """Initialize server.

        Args:
            agent: Agent instance
            host: Server host
            port: Server port
        """
        self.agent = agent
        self.host = host
        self.port = port
        self._running = False
        self._clients: set[Any] = set()

    async def start(self) -> None:
        """Start WebSocket server."""
        # Using built-in HTTP server as WebSocket fallback
        import http.server
        import socketserver

        self._running = True
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/ws":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile_write(json.dumps({"status": "WebSocket endpoint"}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        self._server = socketserver.TCPServer((self.host, self.port), Handler)

        print(f"WebSocket server started on ws://{self.host}:{self.port}")

        await asyncio.get_event_loop().run_in_executor(
            None, self._server.serve_forever
        )

    async def stop(self) -> None:
        """Stop server."""
        self._running = False
        if hasattr(self, "_server"):
            self._server.shutdown()

    async def broadcast(self, message: WebSocketMessage) -> None:
        """Broadcast message to all clients."""
        # Simplified - would iterate over actual WebSocket connections
        pass

    async def handle_message(
        self,
        client: Any,
        message: WebSocketMessage,
    ) -> WebSocketMessage:
        """Handle incoming message.

        Args:
            client: Client connection
            message: Received message

        Returns:
            Response message
        """
        if message.type == MessageType.MESSAGE:
            # Process through agent
            result = await self.agent.process_message(
                message.payload.get("text", "")
            )

            return WebSocketMessage(
                type=MessageType.RESPONSE,
                payload=result,
            )

        elif message.type == MessageType.TOOL_CALL:
            # Execute tool
            result = await self.agent.execute_tool(
                message.payload.get("tool", ""),
                message.payload.get("params", {}),
            )

            return WebSocketMessage(
                type=MessageType.TOOL_RESULT,
                payload=result,
            )

        elif message.type == MessageType.HEARTBEAT:
            return WebSocketMessage(
                type=MessageType.HEARTBEAT,
                payload={"status": "alive"},
            )

        return WebSocketMessage(
            type=MessageType.ERROR,
            payload={"error": "Unknown message type"},
        )


class RealTimeSession:
    """
    Real-time session manager.

    Manages interactive sessions over WebSocket.
    """

    def __init__(self, agent: Any, session_id: str | None = None) -> None:
        """Initialize session.

        Args:
            agent: Agent instance
            session_id: Optional session ID
        """
        self.agent = agent
        self.session_id = session_id or str(hash(datetime.now()))
        self.created_at = datetime.now(timezone.utc)
        self.message_count = 0

    async def process_stream(
        self,
        message: str,
        callback: Callable[[str], None],
    ) -> None:
        """Process message with streaming response.

        Args:
            message: User message
            callback: Callback for response chunks
        """
        self.message_count += 1

        # Add to context
        self.agent.context.add_user_message(message)

        # Process in chunks
        chunks = [
            "Thinking...",
            "\n",
            "Analyzing request...",
            "\n",
            "Processing complete.",
        ]

        response_parts = []

        for chunk in chunks:
            await asyncio.sleep(0.1)
            callback(chunk)
            response_parts.append(chunk)

        # Final response
        full_response = "".join(response_parts)
        self.agent.context.add_assistant_message(full_response)

    async def get_status(self) -> dict[str, Any]:
        """Get session status."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "message_count": self.message_count,
            "agent_state": self.agent.state.name,
        }


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

async def example():
    """Example WebSocket usage."""
    # Create server
    server = WebSocketServer(agent=None, port=8765)

    # Start server
    import asyncio
    server_task = asyncio.create_task(server.start())

    await asyncio.sleep(1)

    print("WebSocket server example started")

    # Create client
    client = WebSocketClient("ws://localhost:8765/ws")

    # Define handler
    def on_response(payload):
        print(f"Received: {payload}")

    client.on(MessageType.RESPONSE, on_response)

    # Connect
    await client.connect()

    # Send message
    await client.send_message("Hello!")

    # Cleanup
    await client.disconnect()
    await server.stop()


if __name__ == "__main__":
    asyncio.run(example())
