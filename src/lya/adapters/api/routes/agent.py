"""Agent API Routes for Lya.

Exposes agent functionality via REST API.
Pure Python 3.14+ compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from lya.core.agent_core import AgentCore, AgentState, create_agent
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class AgentAPI:
    """
    REST API for agent interactions.

    Provides endpoints for messaging, tool execution, and monitoring.
    """

    def __init__(self, agent: AgentCore | None = None) -> None:
        """Initialize API.

        Args:
            agent: Agent instance (creates new if None)
        """
        self.agent = agent
        self._routes = self._build_routes()

    async def _ensure_agent(self) -> AgentCore:
        """Ensure agent is initialized."""
        if self.agent is None:
            self.agent = await create_agent()
        return self.agent

    def _build_routes(self) -> dict:
        """Build route handlers."""
        return {
            # Agent management
            "POST /agent/start": self.start_agent,
            "POST /agent/stop": self.stop_agent,
            "GET /agent/status": self.get_status,
            "GET /agent/stats": self.get_stats,

            # Messaging
            "POST /agent/message": self.process_message,
            "GET /agent/context": self.get_context,
            "DELETE /agent/context": self.clear_context,

            # Tools
            "POST /agent/tools/execute": self.execute_tool,
            "GET /agent/tools/list": self.list_tools,
            "GET /agent/tools/{name}": self.get_tool_info,

            # Memory
            "GET /agent/memory/working": self.get_working_memory,
            "POST /agent/memory/search": self.search_memory,
            "DELETE /agent/memory": self.clear_memory,

            # Workflows
            "POST /agent/workflows/{name}/execute": self.execute_workflow,
            "GET /agent/workflows/list": self.list_workflows,

            # Health
            "GET /agent/health": self.get_health,
            "POST /agent/health/check": self.check_health,

            # Security
            "GET /agent/security/report": self.security_report,
            "POST /agent/security/validate": self.validate_input,
        }

    # ═══════════════════════════════════════════════════════════════════
    # AGENT MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    async def start_agent(self, request: dict) -> dict[str, Any]:
        """Start the agent."""
        try:
            agent = await self._ensure_agent()
            await agent.start()
            return {
                "success": True,
                "agent_id": agent.id,
                "state": agent.state.name,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def stop_agent(self, request: dict) -> dict[str, Any]:
        """Stop the agent."""
        try:
            if self.agent:
                await self.agent.stop()
            return {"success": True, "message": "Agent stopped"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_status(self, request: dict) -> dict[str, Any]:
        """Get agent status."""
        try:
            agent = await self._ensure_agent()
            stats = agent.get_stats()
            return {
                "success": True,
                "agent_id": agent.id,
                "state": agent.state.name,
                "name": agent.config.name,
                "uptime_seconds": stats.uptime_seconds,
                "workspace": str(agent.workspace),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_stats(self, request: dict) -> dict[str, Any]:
        """Get detailed stats."""
        try:
            agent = await self._ensure_agent()
            stats = agent.get_stats()
            return {
                "success": True,
                **stats.__dict__,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════
    # MESSAGING
    # ═══════════════════════════════════════════════════════════════════

    async def process_message(self, request: dict) -> dict[str, Any]:
        """Process a message."""
        try:
            agent = await self._ensure_agent()

            message = request.get("message", "")
            context = request.get("context", {})

            if not message:
                return {"success": False, "error": "Message is required"}

            result = await agent.process_message(message, context)
            return result

        except Exception as e:
            logger.error("Message processing failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def get_context(self, request: dict) -> dict[str, Any]:
        """Get current context."""
        try:
            agent = await self._ensure_agent()
            context = agent.context.get_context_for_llm()

            return {
                "success": True,
                "context": context,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def clear_context(self, request: dict) -> dict[str, Any]:
        """Clear context."""
        try:
            agent = await self._ensure_agent()
            agent.context.clear()

            return {
                "success": True,
                "message": "Context cleared",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════
    # TOOLS
    # ═══════════════════════════════════════════════════════════════════

    async def execute_tool(self, request: dict) -> dict[str, Any]:
        """Execute a tool."""
        try:
            agent = await self._ensure_agent()

            tool_name = request.get("tool", "")
            params = request.get("params", {})

            if not tool_name:
                return {"success": False, "error": "Tool name is required"}

            result = await agent.execute_tool(tool_name, params)
            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_tools(self, request: dict) -> dict[str, Any]:
        """List available tools."""
        try:
            agent = await self._ensure_agent()
            tools = agent.tool_registry.list_tools()

            return {
                "success": True,
                "tools": tools,
                "count": len(tools),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_tool_info(self, request: dict, name: str) -> dict[str, Any]:
        """Get tool information."""
        try:
            agent = await self._ensure_agent()
            schema = agent.tool_registry.get_tool_schema(name)

            if schema is None:
                return {"success": False, "error": f"Tool not found: {name}"}

            return {
                "success": True,
                "tool": name,
                "schema": schema,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════
    # MEMORY
    # ═══════════════════════════════════════════════════════════════════

    async def get_working_memory(self, request: dict) -> dict[str, Any]:
        """Get working memory."""
        try:
            agent = await self._ensure_agent()
            summary = agent.working_memory.get_summary()

            return {
                "success": True,
                **summary.to_dict(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def search_memory(self, request: dict) -> dict[str, Any]:
        """Search working memory."""
        try:
            agent = await self._ensure_agent()

            query = request.get("query", "")
            top_k = request.get("top_k", 5)

            results = agent.working_memory.query(query, top_k=top_k)

            return {
                "success": True,
                "query": query,
                "results": [
                    {
                        "id": r[0].id,
                        "content": r[0].content[:200],
                        "score": r[1],
                        "importance": r[0].importance,
                    }
                    for r in results
                ],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def clear_memory(self, request: dict) -> dict[str, Any]:
        """Clear working memory."""
        try:
            agent = await self._ensure_agent()
            count = agent.working_memory.clear()

            return {
                "success": True,
                "cleared_items": count,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════
    # WORKFLOWS
    # ═══════════════════════════════════════════════════════════════════

    async def execute_workflow(self, request: dict, name: str) -> dict[str, Any]:
        """Execute a workflow."""
        try:
            agent = await self._ensure_agent()
            initial_state = request.get("state", {})

            result = await agent._execute_workflow(name, initial_state)
            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_workflows(self, request: dict) -> dict[str, Any]:
        """List available workflows."""
        try:
            agent = await self._ensure_agent()
            workflows = agent.workflow_manager.list_workflows()

            return {
                "success": True,
                "workflows": workflows,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════
    # HEALTH
    # ═══════════════════════════════════════════════════════════════════

    async def get_health(self, request: dict) -> dict[str, Any]:
        """Get health status."""
        try:
            agent = await self._ensure_agent()

            if not agent.health_monitor:
                return {
                    "success": True,
                    "status": "monitoring_disabled",
                }

            report = agent.health_monitor.get_latest_report()
            if report:
                return {
                    "success": True,
                    **report.to_dict(),
                }
            else:
                return {
                    "success": True,
                    "status": "no_report_yet",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def check_health(self, request: dict) -> dict[str, Any]:
        """Run health check."""
        try:
            agent = await self._ensure_agent()

            if not agent.health_monitor:
                return {
                    "success": False,
                    "error": "Health monitoring disabled",
                }

            report = await agent.health_monitor.check_health()
            return {
                "success": True,
                **report.to_dict(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════
    # SECURITY
    # ═══════════════════════════════════════════════════════════════════

    async def security_report(self, request: dict) -> dict[str, Any]:
        """Get security report."""
        try:
            agent = await self._ensure_agent()
            report = agent.security.generate_report()

            return {
                "success": True,
                **report,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def validate_input(self, request: dict) -> dict[str, Any]:
        """Validate input for security."""
        try:
            agent = await self._ensure_agent()

            text = request.get("text", "")
            context = request.get("context", "general")

            is_valid, error = agent.security.validate_input(text, context)

            return {
                "success": True,
                "valid": is_valid,
                "error": error if not is_valid else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════
    # ROUTING
    # ═══════════════════════════════════════════════════════════════════

    def handle_request(self, method: str, path: str, request: dict) -> dict[str, Any]:
        """Handle HTTP request.

        Args:
            method: HTTP method
            path: Request path
            request: Request body

        Returns:
            Response dict
        """
        route_key = f"{method} {path}"

        # Check exact match
        if route_key in self._routes:
            handler = self._routes[route_key]
            # Note: async handlers would need await here
            # This is a simplified version
            return handler(request)

        # Check pattern match for dynamic routes
        for key, handler in self._routes.items():
            if "{" in key and "}" in key:
                # Simple pattern matching
                pattern = key.replace("{", "").replace("}", "")
                if path.startswith(pattern.split("{")[0]):
                    # Extract parameter
                    parts = path.split("/")
                    param = parts[-1] if parts else ""
                    return handler(request, param)

        return {"success": False, "error": "Route not found", "route": route_key}


def create_fastapi_app(agent: AgentCore | None = None) -> Any:
    """Create FastAPI application.

    Args:
        agent: Optional agent instance

    Returns:
        FastAPI app
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError:
        # Fallback to simple app
        return SimpleHTTPApp(agent)

    app = FastAPI(title="Lya API", version="2.0.0")
    api = AgentAPI(agent)

    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "lya"}

    @app.post("/agent/message")
    async def message(request: Request):
        body = await request.json()
        result = await api.process_message(body)
        return JSONResponse(result)

    @app.get("/agent/status")
    async def status():
        result = await api.get_status({})
        return JSONResponse(result)

    @app.post("/agent/tools/execute")
    async def tool(request: Request):
        body = await request.json()
        result = await api.execute_tool(body)
        return JSONResponse(result)

    return app


class SimpleHTTPApp:
    """Simple HTTP app without FastAPI."""

    def __init__(self, agent: AgentCore | None = None) -> None:
        self.agent = agent
        self.api = AgentAPI(agent)

    async def handle(self, method: str, path: str, body: dict) -> dict:
        """Handle request."""
        return self.api.handle_request(method, path, body)


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio

    async def demo():
        # Create API
        api = AgentAPI()

        # Start agent
        result = await api.start_agent({})
        print(f"Start result: {result}")

        # Get status
        status = await api.get_status({})
        print(f"Status: {status}")

        # Process message
        result = await api.process_message({"message": "Hello!"})
        print(f"Message result: {result}")

        # List tools
        tools = await api.list_tools({})
        print(f"Tools: {tools}")

        # Stop
        await api.stop_agent({})

    asyncio.run(demo())
