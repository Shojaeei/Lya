"""Core Agent Loop for Lya.

Integrates all subsystems into a unified autonomous agent.
Pure Python 3.14+ compatible.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any
from uuid import uuid4

from lya.infrastructure.tools.direct_access import DirectAccess, get_direct_access
from lya.infrastructure.tools.tool_registry import get_tool_registry
from lya.infrastructure.memory.working_buffer import ContextManager, WorkingMemoryBuffer
from lya.infrastructure.workflows.state_graph import StateGraph, WorkflowManager, create_react_workflow
from lya.infrastructure.self_improvement.capability_generator import SelfImprovementLoop, CapabilityGenerator
from lya.infrastructure.monitoring.health_monitor import HealthMonitor, HealthStatus
from lya.infrastructure.security.security_hardening import SecurityManager, SecurityLevel
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class AgentState(Enum):
    """Agent lifecycle states."""
    INITIALIZING = auto()
    IDLE = auto()
    PROCESSING = auto()
    LEARNING = auto()
    SELF_IMPROVING = auto()
    ERROR = auto()
    SHUTTING_DOWN = auto()


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    name: str = "Lya"
    workspace: str = "~/.lya"
    security_level: SecurityLevel = SecurityLevel.HIGH
    enable_self_improvement: bool = True
    enable_health_monitoring: bool = True
    max_context_tokens: int = 8000
    working_memory_size: int = 100
    auto_save_interval: int = 300  # seconds
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"


@dataclass
class AgentStats:
    """Agent runtime statistics."""
    state: AgentState
    uptime_seconds: float = 0.0
    messages_processed: int = 0
    tools_executed: int = 0
    workflows_completed: int = 0
    capabilities_generated: int = 0
    errors_encountered: int = 0
    last_activity: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AgentCore:
    """
    Core agent integrating all Lya subsystems.

    Provides unified interface for autonomous operation.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize agent core.

        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        self.id = str(uuid4())
        self.state = AgentState.INITIALIZING
        self._start_time = time.time()

        # Initialize workspace
        self.workspace = Path(self.config.workspace).expanduser()
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Initialize subsystems
        self._init_security()
        self._init_memory()
        self._init_tools()
        self._init_workflows()
        self._init_self_improvement()
        self._init_health_monitoring()

        # Event handlers
        self._message_handlers: list[Callable[[dict], Any]] = []
        self._state_change_handlers: list[Callable[[AgentState, AgentState], Any]] = []

        self._running = False
        self._shutdown_event = asyncio.Event()

        logger.info(
            "Agent core initialized",
            agent_id=self.id,
            name=self.config.name,
            workspace=str(self.workspace),
        )

    def _init_security(self) -> None:
        """Initialize security manager."""
        self.security = SecurityManager(
            security_level=self.config.security_level,
            workspace=self.workspace,
        )
        logger.info("Security manager initialized", level=self.config.security_level.name)

    def _init_memory(self) -> None:
        """Initialize memory systems."""
        # Working memory
        self.working_memory = WorkingMemoryBuffer(
            max_items=self.config.working_memory_size,
            persist_path=self.workspace / "working_memory.json",
        )

        # Context manager
        self.context = ContextManager(
            working_memory=self.working_memory,
            context_window_size=self.config.max_context_tokens,
        )

        logger.info("Memory systems initialized")

    def _init_tools(self) -> None:
        """Initialize tool systems."""
        self.direct_access = get_direct_access(self.workspace)
        self.tool_registry = get_tool_registry()

        logger.info("Tool systems initialized")

    def _init_workflows(self) -> None:
        """Initialize workflow manager."""
        self.workflow_manager = WorkflowManager()

        # Register default workflows
        react_workflow = create_react_workflow(
            tool_registry=self.tool_registry,
            llm_client=None,  # Will be set at runtime
        )
        self.workflow_manager.register("react", react_workflow)

        logger.info("Workflow manager initialized")

    def _init_self_improvement(self) -> None:
        """Initialize self-improvement system."""
        if self.config.enable_self_improvement:
            generator = CapabilityGenerator(
                output_dir=self.workspace / "generated_capabilities"
            )
            self.improvement_loop = SelfImprovementLoop(
                generator=generator,
                check_interval_minutes=5,
            )
            logger.info("Self-improvement system initialized")
        else:
            self.improvement_loop = None

    def _init_health_monitoring(self) -> None:
        """Initialize health monitoring."""
        if self.config.enable_health_monitoring:
            self.health_monitor = HealthMonitor(
                check_interval_seconds=60,
                auto_heal=True,
            )
            logger.info("Health monitoring initialized")
        else:
            self.health_monitor = None

    async def start(self) -> None:
        """Start the agent."""
        logger.info("Starting agent...")

        self._running = True
        self._set_state(AgentState.IDLE)

        # Start background tasks
        tasks = []

        if self.health_monitor:
            tasks.append(asyncio.create_task(self._health_loop()))

        if self.improvement_loop:
            tasks.append(asyncio.create_task(self._improvement_loop()))

        tasks.append(asyncio.create_task(self._auto_save_loop()))

        logger.info("Agent started", tasks=len(tasks))

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        logger.info("Stopping agent...")

        self._set_state(AgentState.SHUTTING_DOWN)
        self._running = False
        self._shutdown_event.set()

        # Save state
        await self._save_state()

        logger.info("Agent stopped")

    def _set_state(self, new_state: AgentState) -> None:
        """Change agent state."""
        old_state = self.state
        self.state = new_state

        for handler in self._state_change_handlers:
            try:
                handler(old_state, new_state)
            except Exception as e:
                logger.error("State change handler error", error=str(e))

    async def process_message(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a user message.

        Args:
            message: User message
            context: Additional context

        Returns:
            Response dict
        """
        self._set_state(AgentState.PROCESSING)
        start_time = time.time()

        try:
            # Security check
            is_valid, error = self.security.validate_input(message, "user_message")
            if not is_valid:
                return {
                    "success": False,
                    "error": error,
                    "type": "security_rejection",
                }

            # Add to context
            self.context.add_user_message(message)

            # Get context for LLM
            llm_context = self.context.get_context_for_llm()

            # Process through workflow
            workflow_result = await self._execute_workflow(
                "react",
                {
                    "question": message,
                    "context": llm_context,
                }
            )

            # Generate response
            response = await self._generate_response(
                message,
                workflow_result,
                llm_context,
            )

            # Store in context
            self.context.add_assistant_message(response)

            # Update stats
            duration = time.time() - start_time

            result = {
                "success": True,
                "response": response,
                "workflow_result": workflow_result,
                "duration_ms": duration * 1000,
            }

            logger.info(
                "Message processed",
                duration_ms=result["duration_ms"],
                response_length=len(response),
            )

            return result

        except Exception as e:
            logger.error("Message processing failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "type": "processing_error",
            }

        finally:
            self._set_state(AgentState.IDLE)

    async def execute_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool.

        Args:
            tool_name: Tool name
            params: Tool parameters

        Returns:
            Tool result
        """
        # Security validation
        if tool_name in ["execute_command", "exec"]:
            command = params.get("command", "")
            is_valid, error = self.security.validate_command(command)
            if not is_valid:
                return {"success": False, "error": error}

        if tool_name in ["file_read", "file_write"]:
            path = params.get("path", "")
            is_valid, error = self.security.validate_path(path)
            if not is_valid:
                return {"success": False, "error": error}

        # Check rate limit
        allowed, info = self.security.check_rate_limit(f"tool:{tool_name}")
        if not allowed:
            return {
                "success": False,
                "error": "Rate limit exceeded",
                "retry_after": info.get("retry_after"),
            }

        # Execute tool
        try:
            # Use direct access for system tools
            if tool_name == "file_read":
                result = self.direct_access.read_file(params.get("path", ""))
            elif tool_name == "file_write":
                result = self.direct_access.write_file(
                    params.get("path", ""),
                    params.get("content", ""),
                )
            elif tool_name == "http_get":
                result = self.direct_access.http_get(params.get("url", ""))
            elif tool_name == "execute_command":
                result = self.direct_access.execute(params.get("command", ""))
            else:
                # Use registry for other tools
                result = await self.tool_registry.execute(tool_name, params)

            # Store in context
            self.context.add_tool_result(tool_name, result)

            return result

        except Exception as e:
            logger.error("Tool execution failed", tool=tool_name, error=str(e))
            return {"success": False, "error": str(e)}

    async def _execute_workflow(
        self,
        workflow_name: str,
        initial_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a workflow."""
        from lya.infrastructure.workflows.state_graph import State

        workflow = self.workflow_manager.get_workflow(workflow_name)
        if not workflow:
            return {"error": f"Workflow not found: {workflow_name}"}

        state = State(initial_state)
        result = await workflow.execute(state)

        return {
            "success": result.success,
            "path": result.path,
            "final_state": result.final_state.to_dict(),
        }

    async def _generate_response(
        self,
        message: str,
        workflow_result: dict[str, Any],
        context: str,
    ) -> str:
        """Generate final response."""
        # This would integrate with actual LLM
        # For now, return a structured response

        response_parts = [
            f"Processed: {message[:50]}...",
            f"Workflow path: {' -> '.join(workflow_result.get('path', []))}",
        ]

        if workflow_result.get("final_state", {}).get("answer"):
            response_parts.append(f"Answer: {workflow_result['final_state']['answer']}")

        return "\n".join(response_parts)

    async def _health_loop(self) -> None:
        """Background health monitoring loop."""
        while self._running:
            try:
                report = await self.health_monitor.check_health()

                if report.status in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL):
                    logger.warning(
                        "Health degraded",
                        status=report.status.name,
                        issues=len(report.issues),
                    )

                await asyncio.sleep(self.health_monitor.check_interval)

            except Exception as e:
                logger.error("Health loop error", error=str(e))
                await asyncio.sleep(60)

    async def _improvement_loop(self) -> None:
        """Background self-improvement loop."""
        if not self.improvement_loop:
            return

        while self._running:
            try:
                # Analyze usage patterns
                patterns = self._collect_usage_patterns()
                existing = self._get_existing_capabilities()

                # Run improvement
                self._set_state(AgentState.SELF_IMPROVING)
                results = self.improvement_loop.analyze_and_improve(
                    patterns,
                    existing,
                )

                if results.get("generated"):
                    logger.info(
                        "Capabilities generated",
                        count=len(results["generated"]),
                    )

                self._set_state(AgentState.IDLE)

                await asyncio.sleep(self.improvement_loop.check_interval)

            except Exception as e:
                logger.error("Improvement loop error", error=str(e))
                await asyncio.sleep(60)

    def _collect_usage_patterns(self) -> list[dict[str, Any]]:
        """Collect usage patterns for analysis."""
        # This would analyze actual usage
        return [
            {"description": "need csv processing", "resolved": False},
            {"description": "compress files", "resolved": False},
        ]

    def _get_existing_capabilities(self) -> list[str]:
        """Get list of existing capabilities."""
        return self.tool_registry.get_tool_names()

    async def _auto_save_loop(self) -> None:
        """Auto-save state periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.config.auto_save_interval)
                await self._save_state()
                logger.debug("Auto-saved state")
            except Exception as e:
                logger.error("Auto-save failed", error=str(e))

    async def _save_state(self) -> None:
        """Save agent state."""
        state_file = self.workspace / "agent_state.json"

        state = {
            "agent_id": self.id,
            "name": self.config.name,
            "state": self.state.name,
            "stats": self.get_stats(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        state_file.write_text(json.dumps(state, indent=2, default=str))

    def get_stats(self) -> AgentStats:
        """Get current agent statistics."""
        return AgentStats(
            state=self.state,
            uptime_seconds=time.time() - self._start_time,
        )

    def on_state_change(self, handler: Callable[[AgentState, AgentState], Any]) -> None:
        """Register state change handler."""
        self._state_change_handlers.append(handler)

    def on_message(self, handler: Callable[[dict], Any]) -> None:
        """Register message handler."""
        self._message_handlers.append(handler)


class AgentRunner:
    """
    Runner for managing agent lifecycle.

    Handles startup, shutdown, and signal handling.
    """

    def __init__(self, agent: AgentCore) -> None:
        """Initialize runner.

        Args:
            agent: Agent instance
        """
        self.agent = agent
        self._shutdown_event = asyncio.Event()

    async def run(self) -> None:
        """Run the agent."""
        # Setup signal handlers
        import signal

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self._shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Start agent
            await self.agent.start()

            # Keep running
            await self._shutdown_event.wait()

        except Exception as e:
            logger.error("Agent runner error", error=str(e))
            raise

        finally:
            await self.agent.stop()

    async def _shutdown(self) -> None:
        """Initiate shutdown."""
        self._shutdown_event.set()


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

async def create_agent(
    name: str = "Lya",
    **config_kwargs,
) -> AgentCore:
    """Create and initialize an agent.

    Args:
        name: Agent name
        **config_kwargs: Additional config options

    Returns:
        Initialized agent
    """
    config = AgentConfig(name=name, **config_kwargs)
    agent = AgentCore(config)
    await agent.start()
    return agent


async def run_agent_interactive(agent: AgentCore) -> None:
    """Run agent in interactive mode.

    Args:
        agent: Agent instance
    """
    print(f"🤖 {agent.config.name} is ready!")
    print("Type 'exit' to quit, 'status' for stats\n")

    while True:
        try:
            message = input("You: ").strip()

            if not message:
                continue

            if message.lower() == "exit":
                break

            if message.lower() == "status":
                stats = agent.get_stats()
                print(f"\nState: {stats.state.name}")
                print(f"Uptime: {stats.uptime_seconds:.1f}s\n")
                continue

            result = await agent.process_message(message)

            if result["success"]:
                print(f"\n{agent.config.name}: {result['response']}\n")
            else:
                print(f"\nError: {result.get('error', 'Unknown error')}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    await agent.stop()
    print(f"\n👋 {agent.config.name} says goodbye!")


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

async def main():
    """Example agent usage."""
    # Create agent
    agent = await create_agent(
        name="Lya",
        enable_self_improvement=True,
        enable_health_monitoring=True,
    )

    # Process a message
    result = await agent.process_message("Hello, can you help me with a task?")
    print(f"Result: {result}")

    # Execute a tool
    tool_result = await agent.execute_tool("file_list", {"directory": "."})
    print(f"Tool result: {tool_result}")

    # Get stats
    stats = agent.get_stats()
    print(f"Stats: {stats}")

    # Cleanup
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
