"""Agent Core - The heart of Lya."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID, uuid4

from lya.domain.models.agent import Agent, AgentState
from lya.domain.models.goal import Goal
from lya.domain.models.task import Task, TaskStatus
from lya.domain.models.memory import Memory, MemoryType, MemoryImportance
from lya.domain.services.memory_service import MemoryService
from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


class AgentCore:
    """
    Core agent implementation.

    The agent loop follows: OBSERVE → THINK → ACT
    - Observe: Perceive environment and gather information
    - Think: Reason about goals, plan actions
    - Act: Execute tasks and interact with environment

    Features:
    - Event-driven architecture
    - Goal-oriented behavior
    - Memory-augmented reasoning
    - Self-improvement capabilities
    """

    def __init__(
        self,
        agent_id: UUID | None = None,
        name: str = "Lya",
        llm: Any = None,
        memory_service: MemoryService | None = None,
        capability_registry: Any = None,
    ):
        self.id = agent_id or uuid4()
        self.name = name
        self._llm = llm
        self._memory_service = memory_service
        self._capability_registry = capability_registry

        # Agent state
        self._state = AgentState.IDLE
        self._current_goal: Goal | None = None
        self._current_task: Task | None = None
        self._conversation_history: list[dict[str, Any]] = []

        # Event handling
        self._event_handlers: dict[str, list[Callable]] = {}
        self._running = False
        self._task_queue: asyncio.Queue = asyncio.Queue()

        # Metrics
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._start_time = datetime.now(timezone.utc)

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    # ═══════════════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """Start the agent."""
        self._running = True
        self._state = AgentState.IDLE

        logger.info(
            "Agent started",
            agent_id=str(self.id),
            name=self.name,
        )

        # Start the main loop
        asyncio.create_task(self._main_loop())

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self._running = False
        self._state = AgentState.TERMINATED

        logger.info(
            "Agent stopped",
            agent_id=str(self.id),
            tasks_completed=self._tasks_completed,
            tasks_failed=self._tasks_failed,
        )

    async def pause(self) -> None:
        """Pause the agent."""
        self._state = AgentState.PAUSED
        logger.info("Agent paused", agent_id=str(self.id))

    async def resume(self) -> None:
        """Resume the agent."""
        self._state = AgentState.IDLE
        logger.info("Agent resumed", agent_id=str(self.id))

    # ═══════════════════════════════════════════════════════════════
    # Main Loop
    # ═══════════════════════════════════════════════════════════════

    async def _main_loop(self) -> None:
        """
        Main agent loop: OBSERVE → THINK → ACT
        """
        while self._running:
            try:
                if self._state == AgentState.PAUSED:
                    await asyncio.sleep(1)
                    continue

                # OBSERVE: Gather information
                observation = await self._observe()

                # THINK: Reason and plan
                action = await self._think(observation)

                # ACT: Execute action
                if action:
                    await self._act(action)

                # Brief pause to prevent busy-waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error("Error in main loop", error=str(e))
                await asyncio.sleep(1)

    async def _observe(self) -> dict[str, Any]:
        """
        Observe: Gather information about the environment.

        Returns:
            Observation data
        """
        observation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pending_tasks": await self._get_pending_tasks(),
            "current_goal": self._current_goal,
            "conversation": self._conversation_history[-5:] if self._conversation_history else [],
        }

        # Get relevant memories
        if self._memory_service:
            # Search for contextually relevant memories
            context = f"Current goal: {self._current_goal.description if self._current_goal else 'None'}"
            memories = await self._memory_service.recall(
                query=context,
                limit=5,
                threshold=0.6,
            )
            observation["relevant_memories"] = [
                {"content": m.content, "score": s}
                for m, s in memories
            ]

        return observation

    async def _think(self, observation: dict[str, Any]) -> dict[str, Any] | None:
        """
        Think: Reason about observations and plan actions.

        Args:
            observation: Current observation

        Returns:
            Action to take or None
        """
        # Update state based on observation
        if observation["pending_tasks"]:
            self._state = AgentState.WORKING

            # Get next task
            task = observation["pending_tasks"][0]
            return {
                "type": "execute_task",
                "task": task,
            }

        elif self._conversation_history:
            # Check for user messages needing response
            last_message = self._conversation_history[-1]
            if last_message.get("role") == "user":
                return {
                    "type": "respond",
                    "message": last_message,
                }

        # No immediate action needed
        self._state = AgentState.IDLE
        return None

    async def _act(self, action: dict[str, Any]) -> None:
        """
        Act: Execute the planned action.

        Args:
            action: Action to execute
        """
        action_type = action.get("type")

        if action_type == "execute_task":
            await self._execute_task(action["task"])
        elif action_type == "respond":
            await self._respond_to_message(action["message"])
        elif action_type == "research":
            await self._research_capability(action["need"])

    # ═══════════════════════════════════════════════════════════════
    # Task Execution
    # ═══════════════════════════════════════════════════════════════

    async def _execute_task(self, task_data: dict[str, Any]) -> None:
        """Execute a task."""
        self._state = AgentState.WORKING

        task_id = task_data.get("id")
        description = task_data.get("description", "")

        logger.info("Executing task", task_id=task_id, description=description)

        try:
            # Start task
            task = Task(
                description=description,
                priority=task_data.get("priority", "medium"),
            )
            task.start()

            # Execute based on task type
            task_type = task_data.get("type", "generic")

            if task_type == "research":
                await self._execute_research_task(task, task_data)
            elif task_type == "code_generation":
                await self._execute_code_task(task, task_data)
            elif task_type == "file_operation":
                await self._execute_file_task(task, task_data)
            else:
                # Generic task
                await self._execute_generic_task(task, task_data)

            # Complete task
            task.complete()
            self._tasks_completed += 1

            # Store success memory
            if self._memory_service:
                await self._memory_service.create_memory(
                    content=f"Successfully completed task: {description}",
                    memory_type=MemoryType.EPISODIC,
                    importance=MemoryImportance.MEDIUM,
                    agent_id=self.id,
                )

            await self._emit_event("task_completed", {
                "task_id": task_id,
                "description": description,
            })

        except Exception as e:
            logger.error("Task execution failed", task_id=task_id, error=str(e))
            self._tasks_failed += 1

            await self._emit_event("task_failed", {
                "task_id": task_id,
                "error": str(e),
            })

    async def _execute_research_task(self, task: Task, data: dict[str, Any]) -> None:
        """Execute a research task."""
        topic = data.get("topic", "")
        logger.info("Researching topic", topic=topic)

        # Would use browser/tools to research
        # For now, simulate
        await asyncio.sleep(1)

        # Store research findings
        if self._memory_service:
            await self._memory_service.create_memory(
                content=f"Research on {topic}: [Findings would be stored here]",
                memory_type=MemoryType.SEMANTIC,
                importance=MemoryImportance.HIGH,
                agent_id=self.id,
            )

    async def _execute_code_task(self, task: Task, data: dict[str, Any]) -> None:
        """Execute a code generation task."""
        requirement = data.get("requirement", "")
        logger.info("Generating code", requirement=requirement[:50])

        # Would use LLM to generate code
        await asyncio.sleep(1)

    async def _execute_file_task(self, task: Task, data: dict[str, Any]) -> None:
        """Execute a file operation task."""
        operation = data.get("operation", "")
        path = data.get("path", "")
        logger.info("File operation", operation=operation, path=path)

    async def _execute_generic_task(self, task: Task, data: dict[str, Any]) -> None:
        """Execute a generic task."""
        description = data.get("description", "")
        logger.info("Executing generic task", description=description[:50])

    async def _respond_to_message(self, message: dict[str, Any]) -> None:
        """Respond to a user message."""
        content = message.get("content", "")

        # Generate response using LLM
        if self._llm:
            # Would format conversation and get response
            response = f"I understand: {content}"
        else:
            response = "I'm thinking about that..."

        # Store in conversation history
        self._conversation_history.append({
            "role": "assistant",
            "content": response,
        })

        # Emit response event
        await self._emit_event("message_response", {
            "content": response,
            "in_response_to": message.get("id"),
        })

    # ═══════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════

    async def send_message(self, content: str, source: str = "user") -> None:
        """Send a message to the agent."""
        message = {
            "id": str(uuid4()),
            "role": "user" if source == "user" else "system",
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._conversation_history.append(message)
        await self._task_queue.put({"type": "message", "data": message})

    async def set_goal(self, description: str) -> Goal:
        """Set a new goal for the agent."""
        goal = Goal(description=description)
        self._current_goal = goal

        await self._emit_event("goal_set", {
            "goal_id": str(goal.id),
            "description": description,
        })

        return goal

    async def get_status(self) -> dict[str, Any]:
        """Get agent status."""
        return {
            "id": str(self.id),
            "name": self.name,
            "state": self._state.name,
            "running": self._running,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "current_goal": self._current_goal.description if self._current_goal else None,
            "uptime": (datetime.now(timezone.utc) - self._start_time).total_seconds(),
        }

    # ═══════════════════════════════════════════════════════════════
    # Event System
    # ═══════════════════════════════════════════════════════════════

    def on(self, event: str, handler: Callable) -> None:
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def off(self, event: str, handler: Callable) -> None:
        """Unregister event handler."""
        if event in self._event_handlers:
            self._event_handlers[event].remove(handler)

    async def _emit_event(self, event: str, data: dict[str, Any]) -> None:
        """Emit an event."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error("Event handler failed", event=event, error=str(e))

    # ═══════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════

    async def _get_pending_tasks(self) -> list[dict[str, Any]]:
        """Get pending tasks."""
        # Would query task repository
        return []

    async def _research_capability(self, need: str) -> None:
        """Research and generate a new capability."""
        logger.info("Researching capability", need=need)
        # Would trigger self-improvement
