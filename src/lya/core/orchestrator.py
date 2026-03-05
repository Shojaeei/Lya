"""Orchestrator for managing multiple agents and workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from lya.core.agent import AgentCore
from lya.core.event_bus import EventBus, Event, get_event_bus
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OrchestrationConfig:
    """Configuration for orchestration."""

    max_concurrent_agents: int = 10
    auto_scale: bool = True
    enable_self_improvement: bool = True
    enable_memory_consolidation: bool = True
    task_timeout_seconds: int = 300


class Orchestrator:
    """
    Orchestrator for managing multiple agents and workflows.

    Features:
    - Multi-agent management
    - Workflow orchestration
    - Load balancing
    - Self-improvement coordination
    """

    def __init__(
        self,
        config: OrchestrationConfig | None = None,
        event_bus: EventBus | None = None,
    ):
        self.config = config or OrchestrationConfig()
        self._event_bus = event_bus or get_event_bus()
        self._agents: dict[UUID, AgentCore] = {}
        self._workflows: dict[str, dict] = {}
        self._running = False

    async def start(self) -> None:
        """Start the orchestrator."""
        self._running = True

        # Subscribe to events
        self._event_bus.subscribe("agent_created", self._on_agent_created)
        self._event_bus.subscribe("task_completed", self._on_task_completed)
        self._event_bus.subscribe("capability_generated", self._on_capability_generated)

        # Start background tasks
        import asyncio
        asyncio.create_task(self._self_improvement_loop())
        asyncio.create_task(self._memory_consolidation_loop())

        logger.info("Orchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False

        # Stop all agents
        for agent in self._agents.values():
            await agent.stop()

        logger.info("Orchestrator stopped")

    # ═══════════════════════════════════════════════════════════════
    # Agent Management
    # ═══════════════════════════════════════════════════════════════

    async def create_agent(
        self,
        name: str = "Agent",
        llm: Any = None,
        **kwargs,
    ) -> AgentCore:
        """Create and register a new agent."""
        agent = AgentCore(name=name, llm=llm, **kwargs)

        self._agents[agent.id] = agent

        # Start the agent
        await agent.start()

        # Publish event
        await self._event_bus.publish(Event(
            type="agent_created",
            payload={
                "agent_id": str(agent.id),
                "name": name,
            },
            source="orchestrator",
        ))

        logger.info("Agent created", agent_id=str(agent.id), name=name)
        return agent

    async def get_agent(self, agent_id: UUID) -> AgentCore | None:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    async def list_agents(self) -> list[AgentCore]:
        """List all agents."""
        return list(self._agents.values())

    async def terminate_agent(self, agent_id: UUID) -> bool:
        """Terminate an agent."""
        agent = self._agents.pop(agent_id, None)
        if agent:
            await agent.stop()

            await self._event_bus.publish(Event(
                type="agent_terminated",
                payload={"agent_id": str(agent_id)},
                source="orchestrator",
            ))

            return True
        return False

    # ═══════════════════════════════════════════════════════════════
    # Workflow Management
    # ═══════════════════════════════════════════════════════════════

    async def create_workflow(
        self,
        name: str,
        steps: list[dict[str, Any]],
    ) -> str:
        """
        Create a workflow.

        Args:
            name: Workflow name
            steps: List of workflow steps

        Returns:
            Workflow ID
        """
        from uuid import uuid4
        workflow_id = str(uuid4())

        self._workflows[workflow_id] = {
            "id": workflow_id,
            "name": name,
            "steps": steps,
            "status": "pending",
        }

        return workflow_id

    async def execute_workflow(
        self,
        workflow_id: str,
        agent_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Execute a workflow."""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        # Get or create agent
        agent = await self.get_agent(agent_id) if agent_id else None
        if not agent:
            agent = await self.create_agent(name="WorkflowAgent")

        workflow["status"] = "running"
        results = []

        for step in workflow["steps"]:
            step_type = step.get("type")

            try:
                if step_type == "task":
                    result = await self._execute_step_task(agent, step)
                elif step_type == "goal":
                    result = await self._execute_step_goal(agent, step)
                elif step_type == "memory":
                    result = await self._execute_step_memory(agent, step)
                else:
                    result = {"status": "skipped", "reason": "unknown_type"}

                results.append(result)

            except Exception as e:
                logger.error("Workflow step failed", step=step, error=str(e))
                results.append({"status": "error", "error": str(e)})

        workflow["status"] = "completed"
        workflow["results"] = results

        return {
            "workflow_id": workflow_id,
            "status": "completed",
            "results": results,
        }

    async def _execute_step_task(self, agent: AgentCore, step: dict) -> dict:
        """Execute a task step."""
        # Would create and execute task
        return {"status": "completed", "type": "task"}

    async def _execute_step_goal(self, agent: AgentCore, step: dict) -> dict:
        """Execute a goal step."""
        description = step.get("description", "")
        goal = await agent.set_goal(description)
        return {"status": "completed", "type": "goal", "goal_id": str(goal.id)}

    async def _execute_step_memory(self, agent: AgentCore, step: dict) -> dict:
        """Execute a memory step."""
        # Would interact with memory system
        return {"status": "completed", "type": "memory"}

    # ═══════════════════════════════════════════════════════════════
    # Background Loops
    # ═══════════════════════════════════════════════════════════════

    async def _self_improvement_loop(self) -> None:
        """Background loop for self-improvement."""
        import asyncio

        while self._running:
            try:
                if self.config.enable_self_improvement:
                    await self._check_improvement_opportunities()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error("Self-improvement loop error", error=str(e))
                await asyncio.sleep(60)

    async def _check_improvement_opportunities(self) -> None:
        """Check for self-improvement opportunities."""
        # Analyze agent performance
        # Identify missing capabilities
        # Trigger capability generation if needed
        pass

    async def _memory_consolidation_loop(self) -> None:
        """Background loop for memory consolidation."""
        import asyncio

        while self._running:
            try:
                if self.config.enable_memory_consolidation:
                    await self._consolidate_memories()

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error("Memory consolidation loop error", error=str(e))
                await asyncio.sleep(300)

    async def _consolidate_memories(self) -> None:
        """Consolidate memories across agents."""
        # Would consolidate memories
        pass

    # ═══════════════════════════════════════════════════════════════
    # Event Handlers
    # ═══════════════════════════════════════════════════════════════

    async def _on_agent_created(self, event: Event) -> None:
        """Handle agent created event."""
        logger.debug("Agent created event", payload=event.payload)

    async def _on_task_completed(self, event: Event) -> None:
        """Handle task completed event."""
        logger.debug("Task completed event", payload=event.payload)

    async def _on_capability_generated(self, event: Event) -> None:
        """Handle capability generated event."""
        logger.info("New capability generated", payload=event.payload)

    # ═══════════════════════════════════════════════════════════════
    # Status
    # ═══════════════════════════════════════════════════════════════

    async def get_status(self) -> dict[str, Any]:
        """Get orchestrator status."""
        return {
            "running": self._running,
            "agents_count": len(self._agents),
            "workflows_count": len(self._workflows),
            "config": {
                "max_concurrent_agents": self.config.max_concurrent_agents,
                "auto_scale": self.config.auto_scale,
                "enable_self_improvement": self.config.enable_self_improvement,
            },
        }
