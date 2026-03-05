"""Agent API Routes for Lya.

Exposes agent functionality via REST endpoints.
Pure Python 3.14+ compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lya.core.agent_core import AgentCore, AgentConfig
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


# Simple HTTP response builder for pure Python
def json_response(data: dict, status: int = 200) -> tuple[str, int, dict]:
    """Build JSON response."""
    return (
        json.dumps(data, default=str),
        status,
        {"Content-Type": "application/json"},
    )


def error_response(message: str, status: int = 400) -> tuple[str, int, dict]:
    """Build error response."""
    return json_response(
        {"success": False, "error": message, "timestamp": datetime.now(timezone.utc).isoformat()},
        status,
    )


class AgentRoutes:
    """
    API routes for agent management.

    Provides REST endpoints for:
    - Message processing
    - Tool execution
    - Workflow management
    - Status and monitoring
    """

    def __init__(self, agent: AgentCore) -> None:
        """Initialize routes.

        Args:
            agent: Agent instance
        """
        self.agent = agent

    # ═══════════════════════════════════════════════════════════════════
    # MESSAGE ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════

    async def process_message(self, request: dict) -> tuple[str, int, dict]:
        """POST /api/v1/agent/message

        Process a user message.
        """
        try:
            message = request.get("message", "")
            context = request.get("context", {})

            if not message:
                return error_response("Message is required", 400)

            result = await self.agent.process_message(message, context)
            return json_response(result)

        except Exception as e:
            logger.error("Message processing failed", error=str(e))
            return error_response(str(e), 500)

    async def get_conversation_context(self, request: dict) -> tuple[str, int, dict]:
        """GET /api/v1/agent/context

        Get current conversation context.
        """
        try:
            context = self.agent.context.get_context_for_llm(
                include_history=True,
                include_working_memory=True,
            )

            return json_response({
                "success": True,
                "context": context,
                "working_memory_items": len(self.agent.working_memory._items),
            })

        except Exception as e:
            return error_response(str(e), 500)

    async def clear_context(self, request: dict) -> tuple[str, int, dict]:
        """POST /api/v1/agent/context/clear

        Clear conversation context.
        """
        try:
            source = request.get("source")  # Optional: clear only specific source
            self.agent.context.clear()

            return json_response({
                "success": True,
                "message": "Context cleared",
                "source_filter": source,
            })

        except Exception as e:
            return error_response(str(e), 500)

    # ═══════════════════════════════════════════════════════════════════
    # TOOL ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════

    async def execute_tool(self, request: dict) -> tuple[str, int, dict]:
        """POST /api/v1/agent/tools/execute

        Execute a tool.
        """
        try:
            tool_name = request.get("tool", "")
            params = request.get("params", {})

            if not tool_name:
                return error_response("Tool name is required", 400)

            result = await self.agent.execute_tool(tool_name, params)
            return json_response(result)

        except Exception as e:
            logger.error("Tool execution failed", tool=tool_name, error=str(e))
            return error_response(str(e), 500)

    async def list_tools(self, request: dict) -> tuple[str, int, dict]:
        """GET /api/v1/agent/tools

        List available tools.
        """
        try:
            tools = self.agent.tool_registry.list_tools()

            return json_response({
                "success": True,
                "tools": tools,
                "count": len(tools),
            })

        except Exception as e:
            return error_response(str(e), 500)

    async def direct_access(self, request: dict) -> tuple[str, int, dict]:
        """POST /api/v1/agent/tools/direct

        Direct access operations (read/write/exec).
        """
        try:
            operation = request.get("operation", "")
            path = request.get("path", "")
            content = request.get("content", "")

            if operation == "read":
                result = self.agent.direct_access.read_file(path)
            elif operation == "write":
                result = self.agent.direct_access.write_file(path, content)
            elif operation == "list":
                result = self.agent.direct_access.list_directory(path)
            elif operation == "exec":
                command = request.get("command", "")
                result = self.agent.direct_access.execute(command)
            elif operation == "http_get":
                url = request.get("url", "")
                result = self.agent.direct_access.http_get(url)
            else:
                return error_response(f"Unknown operation: {operation}", 400)

            return json_response(result)

        except Exception as e:
            return error_response(str(e), 500)

    # ═══════════════════════════════════════════════════════════════════
    # WORKFLOW ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════

    async def list_workflows(self, request: dict) -> tuple[str, int, dict]:
        """GET /api/v1/agent/workflows

        List available workflows.
        """
        try:
            workflows = self.agent.workflow_manager.list_workflows()

            return json_response({
                "success": True,
                "workflows": workflows,
            })

        except Exception as e:
            return error_response(str(e), 500)

    async def execute_workflow(self, request: dict) -> tuple[str, int, dict]:
        """POST /api/v1/agent/workflows/{name}/execute

        Execute a workflow.
        """
        try:
            workflow_name = request.get("workflow", "")
            initial_state = request.get("state", {})

            if not workflow_name:
                return error_response("Workflow name is required", 400)

            result = await self.agent._execute_workflow(workflow_name, initial_state)
            return json_response(result)

        except Exception as e:
            return error_response(str(e), 500)

    # ═══════════════════════════════════════════════════════════════════
    # MEMORY ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════

    async def search_memory(self, request: dict) -> tuple[str, int, dict]:
        """POST /api/v1/agent/memory/search

        Search working memory.
        """
        try:
            query = request.get("query", "")
            top_k = request.get("top_k", 10)

            if not query:
                return error_response("Query is required", 400)

            results = self.agent.working_memory.query(query, top_k=top_k)

            return json_response({
                "success": True,
                "query": query,
                "results": [
                    {
                        "id": item.id,
                        "content": item.content[:200],
                        "score": score,
                        "importance": item.importance,
                    }
                    for item, score in results
                ],
            })

        except Exception as e:
            return error_response(str(e), 500)

    async def add_memory(self, request: dict) -> tuple[str, int, dict]:
        """POST /api/v1/agent/memory

        Add item to working memory.
        """
        try:
            content = request.get("content", "")
            importance = request.get("importance", 0.5)
            source = request.get("source", "api")

            if not content:
                return error_response("Content is required", 400)

            item_id = self.agent.working_memory.add(
                content=content,
                importance=importance,
                source=source,
            )

            return json_response({
                "success": True,
                "item_id": item_id,
            })

        except Exception as e:
            return error_response(str(e), 500)

    async def get_memory_stats(self, request: dict) -> tuple[str, int, dict]:
        """GET /api/v1/agent/memory/stats

        Get working memory statistics.
        """
        try:
            summary = self.agent.working_memory.get_summary()

            return json_response({
                "success": True,
                "stats": {
                    "total_items": summary.total_items,
                    "expired_items": summary.expired_items,
                    "total_importance": summary.total_importance,
                    "oldest_item_hours": summary.oldest_item_age_hours,
                    "top_sources": summary.top_sources,
                },
            })

        except Exception as e:
            return error_response(str(e), 500)

    # ═══════════════════════════════════════════════════════════════════
    # HEALTH ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════

    async def get_health(self, request: dict) -> tuple[str, int, dict]:
        """GET /api/v1/agent/health

        Get agent health status.
        """
        try:
            if self.agent.health_monitor:
                report = self.agent.health_monitor.get_latest_report()
                if report:
                    return json_response({
                        "success": True,
                        "status": report.status.name,
                        "uptime_seconds": report.uptime_seconds,
                        "checks": [
                            {
                                "name": c.name,
                                "status": c.status.name,
                                "message": c.message,
                            }
                            for c in report.checks
                        ],
                    })

            # Fallback to basic status
            stats = self.agent.get_stats()
            return json_response({
                "success": True,
                "status": stats.state.name,
                "uptime_seconds": stats.uptime_seconds,
            })

        except Exception as e:
            return error_response(str(e), 500)

    async def get_stats(self, request: dict) -> tuple[str, int, dict]:
        """GET /api/v1/agent/stats

        Get detailed agent statistics.
        """
        try:
            stats = self.agent.get_stats()

            return json_response({
                "success": True,
                "agent_id": self.agent.id,
                "name": self.agent.config.name,
                "state": stats.state.name,
                "uptime_seconds": stats.uptime_seconds,
                "workspace": str(self.agent.workspace),
                "config": {
                    "llm_provider": self.agent.config.llm_provider,
                    "llm_model": self.agent.config.llm_model,
                    "max_context_tokens": self.agent.config.max_context_tokens,
                },
            })

        except Exception as e:
            return error_response(str(e), 500)

    # ═══════════════════════════════════════════════════════════════════
    # SECURITY ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════

    async def validate_input(self, request: dict) -> tuple[str, int, dict]:
        """POST /api/v1/agent/security/validate

        Validate input for security.
        """
        try:
            text = request.get("text", "")
            context = request.get("context", "general")

            is_valid, error = self.agent.security.validate_input(text, context)

            return json_response({
                "success": True,
                "valid": is_valid,
                "error": error if not is_valid else None,
            })

        except Exception as e:
            return error_response(str(e), 500)

    async def get_security_report(self, request: dict) -> tuple[str, int, dict]:
        """GET /api/v1/agent/security/report

        Get security audit report.
        """
        try:
            report = self.agent.security.generate_report()

            return json_response({
                "success": True,
                "report": report,
            })

        except Exception as e:
            return error_response(str(e), 500)

    # ═══════════════════════════════════════════════════════════════════
    # SELF-IMPROVEMENT ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════

    async def trigger_improvement(self, request: dict) -> tuple[str, int, dict]:
        """POST /api/v1/agent/improvement/trigger

        Trigger self-improvement analysis.
        """
        try:
            if not self.agent.improvement_loop:
                return error_response("Self-improvement disabled", 400)

            patterns = self.agent._collect_usage_patterns()
            existing = self.agent._get_existing_capabilities()

            results = self.agent.improvement_loop.analyze_and_improve(
                patterns,
                existing,
            )

            return json_response({
                "success": True,
                "results": results,
            })

        except Exception as e:
            return error_response(str(e), 500)

    async def list_capabilities(self, request: dict) -> tuple[str, int, dict]:
        """GET /api/v1/agent/improvement/capabilities

        List generated capabilities.
        """
        try:
            caps_dir = self.agent.workspace / "generated_capabilities"
            if not caps_dir.exists():
                return json_response({
                    "success": True,
                    "capabilities": [],
                })

            capabilities = []
            for f in caps_dir.glob("*_meta.json"):
                meta = json.loads(f.read_text())
                capabilities.append(meta)

            return json_response({
                "success": True,
                "capabilities": capabilities,
                "count": len(capabilities),
            })

        except Exception as e:
            return error_response(str(e), 500)


# ═══════════════════════════════════════════════════════════════════════
# ROUTE REGISTRY
# ═══════════════════════════════════════════════════════════════════════

def register_agent_routes(app: Any, agent: AgentCore) -> None:
    """Register all agent routes with Flask/FastAPI app.

    Args:
        app: Web framework app instance
        agent: Agent instance
    """
    routes = AgentRoutes(agent)

    # Message routes
    app.add_url_rule("/api/v1/agent/message", "process_message",
                     lambda req: routes.process_message(req), methods=["POST"])
    app.add_url_rule("/api/v1/agent/context", "get_context",
                     lambda req: routes.get_conversation_context(req), methods=["GET"])
    app.add_url_rule("/api/v1/agent/context/clear", "clear_context",
                     lambda req: routes.clear_context(req), methods=["POST"])

    # Tool routes
    app.add_url_rule("/api/v1/agent/tools", "list_tools",
                     lambda req: routes.list_tools(req), methods=["GET"])
    app.add_url_rule("/api/v1/agent/tools/execute", "execute_tool",
                     lambda req: routes.execute_tool(req), methods=["POST"])
    app.add_url_rule("/api/v1/agent/tools/direct", "direct_access",
                     lambda req: routes.direct_access(req), methods=["POST"])

    # Workflow routes
    app.add_url_rule("/api/v1/agent/workflows", "list_workflows",
                     lambda req: routes.list_workflows(req), methods=["GET"])
    app.add_url_rule("/api/v1/agent/workflows/execute", "execute_workflow",
                     lambda req: routes.execute_workflow(req), methods=["POST"])

    # Memory routes
    app.add_url_rule("/api/v1/agent/memory", "add_memory",
                     lambda req: routes.add_memory(req), methods=["POST"])
    app.add_url_rule("/api/v1/agent/memory/search", "search_memory",
                     lambda req: routes.search_memory(req), methods=["POST"])
    app.add_url_rule("/api/v1/agent/memory/stats", "memory_stats",
                     lambda req: routes.get_memory_stats(req), methods=["GET"])

    # Health routes
    app.add_url_rule("/api/v1/agent/health", "get_health",
                     lambda req: routes.get_health(req), methods=["GET"])
    app.add_url_rule("/api/v1/agent/stats", "get_stats",
                     lambda req: routes.get_stats(req), methods=["GET"])

    # Security routes
    app.add_url_rule("/api/v1/agent/security/validate", "validate_input",
                     lambda req: routes.validate_input(req), methods=["POST"])
    app.add_url_rule("/api/v1/agent/security/report", "security_report",
                     lambda req: routes.get_security_report(req), methods=["GET"])

    # Improvement routes
    app.add_url_rule("/api/v1/agent/improvement/trigger", "trigger_improvement",
                     lambda req: routes.trigger_improvement(req), methods=["POST"])
    app.add_url_rule("/api/v1/agent/improvement/capabilities", "list_capabilities",
                     lambda req: routes.list_capabilities(req), methods=["GET"])


# ═══════════════════════════════════════════════════════════════════════
# SIMPLE HTTP SERVER EXAMPLE
# ═══════════════════════════════════════════════════════════════════════

class SimpleAgentServer:
    """Simple HTTP server for agent API."""

    def __init__(self, agent: AgentCore, port: int = 8080) -> None:
        """Initialize server.

        Args:
            agent: Agent instance
            port: Server port
        """
        self.agent = agent
        self.port = port
        self.routes = AgentRoutes(agent)

    def handle_request(self, method: str, path: str, body: bytes) -> tuple[str, int, dict]:
        """Handle HTTP request."""
        import json

        # Parse body
        try:
            request = json.loads(body) if body else {}
        except:
            request = {}

        # Route matching
        routes = {
            ("POST", "/api/v1/agent/message"): self.routes.process_message,
            ("GET", "/api/v1/agent/health"): self.routes.get_health,
            ("GET", "/api/v1/agent/stats"): self.routes.get_stats,
            ("GET", "/api/v1/agent/tools"): self.routes.list_tools,
            ("POST", "/api/v1/agent/tools/execute"): self.routes.execute_tool,
            ("POST", "/api/v1/agent/tools/direct"): self.routes.direct_access,
            ("GET", "/api/v1/agent/workflows"): self.routes.list_workflows,
            ("POST", "/api/v1/agent/workflows/execute"): self.routes.execute_workflow,
            ("POST", "/api/v1/agent/memory"): self.routes.add_memory,
            ("POST", "/api/v1/agent/memory/search"): self.routes.search_memory,
            ("GET", "/api/v1/agent/memory/stats"): self.routes.get_memory_stats,
        }

        handler = routes.get((method, path))

        if handler:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(handler(request))

        return json_response({"error": "Not found"}, 404)

    def start(self) -> None:
        """Start server."""
        from http.server import HTTPServer, BaseHTTPRequestHandler

        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def _handle(self, method: str):
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)

                response, status, headers = server.handle_request(method, self.path, body)

                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response.encode())

            def log_message(self, format, *args):
                logger.info(format % args)

        httpd = HTTPServer(("", self.port), Handler)
        logger.info(f"Agent API server started on port {self.port}")
        httpd.serve_forever()


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        from lya.core.agent_core import create_agent

        agent = await create_agent(name="Lya-API")
        server = SimpleAgentServer(agent, port=8080)
        server.start()

    asyncio.run(main())
