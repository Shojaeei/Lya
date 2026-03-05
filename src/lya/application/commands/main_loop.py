"""Main Execution Loop for Lya.

Implements the core autonomous loop that orchestrates goal processing,
planning, execution, and self-improvement following Clean Architecture.
"""

from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from lya.domain.models.goal import Goal, GoalPriority, GoalStatus
from lya.domain.models.memory import MemoryType, MemoryImportance
from lya.domain.models.events import (
    EventPublisher,
    GoalStarted,
    GoalCompleted,
    GoalFailed,
)
from lya.domain.repositories.goal_repo import GoalRepository
from lya.domain.services.planning_service import PlanningService
from lya.domain.services.reasoning_service import ReasoningService
from lya.domain.services.memory_service import MemoryService
from lya.domain.services.self_improvement_service import SelfImprovementService
from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.config.settings import settings

logger = get_logger(__name__)


class ToolPort(Protocol):
    """Protocol for tool execution."""

    async def execute(self, tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]: ...
    def list_tools(self) -> list[dict[str, Any]]: ...


class LLMPort(Protocol):
    """Protocol for LLM connectivity check."""

    async def generate(self, prompt: str, **kwargs: Any) -> str: ...


@dataclass
class MainLoopStatus:
    """Status report for the main loop."""

    running: bool = False
    paused: bool = False
    iteration_count: int = 0
    goals_total: int = 0
    goals_pending: int = 0
    goals_in_progress: int = 0
    goals_completed: int = 0
    goals_failed: int = 0
    current_goal_id: str | None = None
    last_error: str | None = None
    started_at: datetime | None = None
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert status to dictionary."""
        return {
            "running": self.running,
            "paused": self.paused,
            "iteration_count": self.iteration_count,
            "goals_total": self.goals_total,
            "goals_pending": self.goals_pending,
            "goals_in_progress": self.goals_in_progress,
            "goals_completed": self.goals_completed,
            "goals_failed": self.goals_failed,
            "current_goal_id": self.current_goal_id,
            "last_error": self.last_error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "uptime_seconds": self.uptime_seconds,
        }


@dataclass
class AddGoalRequest:
    """Request to add a goal to the main loop."""

    description: str
    priority: int = 3  # 1-5, lower is higher priority
    agent_id: UUID | None = None
    context: str | None = None


@dataclass
class AddGoalResult:
    """Result of adding a goal."""

    goal_id: UUID | None = None
    success: bool = False
    message: str = ""


class MainLoop:
    """
    Main autonomous execution loop for Lya.

    Orchestrates the continuous operation of the agent:
    1. Goal management (add, process, track)
    2. Planning via PlanningService
    3. Execution via tool port
    4. Memory via MemoryService
    5. Self-improvement via SelfImprovementService
    6. Reasoning via ReasoningService

    Usage:
        loop = MainLoop(
            agent_id=agent_id,
            goal_repository=goal_repo,
            event_publisher=event_bus,
            planning_service=planning_service,
            reasoning_service=reasoning_service,
            memory_service=memory_service,
            self_improvement_service=improvement_service,
            tool_port=tool_registry,
            llm_port=llm_adapter,
        )
        await loop.start()

    Attributes:
        running: Whether the loop is currently running
        paused: Whether execution is paused
        status: Current status report
    """

    def __init__(
        self,
        agent_id: UUID,
        goal_repository: GoalRepository,
        event_publisher: EventPublisher,
        planning_service: PlanningService,
        reasoning_service: ReasoningService,
        memory_service: MemoryService,
        self_improvement_service: SelfImprovementService | None,
        tool_port: ToolPort,
        llm_port: LLMPort,
        workspace: Path | None = None,
    ) -> None:
        """
        Initialize the main loop with all dependencies.

        Args:
            agent_id: The ID of the agent running this loop
            goal_repository: Repository for goal persistence
            event_publisher: Event bus for publishing domain events
            planning_service: Service for creating and refining plans
            reasoning_service: Service for reasoning and reflection
            memory_service: Service for memory operations
            self_improvement_service: Service for self-improvement (optional)
            tool_port: Port for tool execution
            llm_port: Port for LLM connectivity
            workspace: Workspace directory (defaults to settings)
        """
        self._agent_id = agent_id
        self._goal_repo = goal_repository
        self._events = event_publisher
        self._planning = planning_service
        self._reasoning = reasoning_service
        self._memory = memory_service
        self._improvement = self_improvement_service
        self._tools = tool_port
        self._llm = llm_port

        self._workspace = workspace or settings.workspace_path
        self._workspace.mkdir(parents=True, exist_ok=True)

        # State
        self._running = False
        self._paused = False
        self._stop_event: asyncio.Event | None = None
        self._status = MainLoopStatus()

        logger.info(
            "MainLoop initialized",
            agent_id=str(agent_id),
            workspace=str(self._workspace),
        )

    async def check_llm_connectivity(self) -> bool:
        """
        Check if LLM is available and responsive.

        Returns:
            True if LLM is connected and responding
        """
        try:
            # Simple connectivity check
            response = await self._llm.generate(
                prompt="Hello",
                temperature=0.0,
                max_tokens=10,
            )
            logger.debug("LLM connectivity check passed")
            return True
        except Exception as e:
            logger.error("LLM connectivity check failed", error=str(e))
            return False

    async def add_goal(self, request: AddGoalRequest) -> AddGoalResult:
        """
        Add a new goal to the processing queue.

        Args:
            request: Goal creation request

        Returns:
            Result with goal ID if successful
        """
        try:
            # Validate priority
            if not 1 <= request.priority <= 5:
                return AddGoalResult(
                    success=False,
                    message="Priority must be between 1 (highest) and 5 (lowest)",
                )

            # Map priority
            priority_map = {
                1: GoalPriority.CRITICAL,
                2: GoalPriority.HIGH,
                3: GoalPriority.MEDIUM,
                4: GoalPriority.LOW,
                5: GoalPriority.TRIVIAL,
            }
            priority = priority_map.get(request.priority, GoalPriority.MEDIUM)

            # Create goal
            goal = Goal(
                agent_id=request.agent_id or self._agent_id,
                description=request.description.strip(),
                priority=priority,
            )

            # Save to repository
            await self._goal_repo.save(goal)

            # Create memory about the new goal
            await self._memory.create_memory(
                content=f"New goal added: {request.description}",
                memory_type=MemoryType.GOAL,
                importance=MemoryImportance.HIGH,
                agent_id=request.agent_id or self._agent_id,
                context={
                    "goal_id": str(goal.id),
                    "priority": request.priority,
                    "description": request.description,
                },
            )

            logger.info(
                "Goal added to main loop",
                goal_id=str(goal.id),
                description=request.description[:50],
                priority=request.priority,
            )

            return AddGoalResult(
                goal_id=goal.id,
                success=True,
                message=f"Goal added with ID {goal.id}",
            )

        except Exception as e:
            logger.error("Failed to add goal", error=str(e))
            return AddGoalResult(
                success=False,
                message=f"Error: {str(e)}",
            )

    async def process_goal(self, goal: Goal) -> dict[str, Any]:
        """
        Process a single goal through planning and execution.

        Args:
            goal: The goal to process

        Returns:
            Processing result with status and details
        """
        result = {"success": False, "goal_id": str(goal.id), "steps_executed": 0}

        try:
            logger.info("Processing goal", goal_id=str(goal.id), description=goal.description[:50])

            # Update status
            self._status.current_goal_id = str(goal.id)

            # Publish goal started event
            await self._events.publish(
                GoalStarted(
                    goal_id=str(goal.id),
                    agent_id=str(goal.agent_id) if goal.agent_id else str(self._agent_id),
                )
            )

            # Start goal execution
            goal.start_execution()
            await self._goal_repo.save(goal)

            # Get available tools
            available_tools = self._tools.list_tools()

            # Create plan if none exists
            if not goal.plan:
                logger.info("Creating plan for goal", goal_id=str(goal.id))
                plan = await self._planning.create_plan(
                    goal=goal,
                    available_tools=available_tools,
                    context=None,
                )
                goal.set_plan(plan)
                await self._goal_repo.save(goal)

            # Execute plan steps
            if goal.plan:
                logger.info(
                    "Executing plan",
                    goal_id=str(goal.id),
                    steps=len(goal.plan.steps),
                )

                for i, step in enumerate(goal.plan.steps):
                    step_result = await self._execute_step(goal, step, i + 1)
                    result["steps_executed"] += 1

                    if not step_result.get("success"):
                        # Step failed - attempt to reason and recover
                        logger.warning(
                            "Step failed, attempting recovery",
                            goal_id=str(goal.id),
                            step=i + 1,
                            error=step_result.get("error"),
                        )

                        reasoning = await self._reasoning.reason(
                            question=f"How should I recover from this step failure: {step_result.get('error')}?",
                            context=f"Goal: {goal.description}\nFailed step: {step.get('description')}",
                        )

                        logger.info(
                            "Recovery reasoning generated",
                            goal_id=str(goal.id),
                            conclusion=reasoning.conclusion[:100],
                        )

                        # Mark step as completed in plan (even on failure)
                        step["completed"] = True

            # Mark goal as completed
            goal.complete(result={"steps_executed": result["steps_executed"]})
            await self._goal_repo.save(goal)

            # Publish completion event
            await self._events.publish(
                GoalCompleted(
                    goal_id=str(goal.id),
                    agent_id=str(goal.agent_id) if goal.agent_id else str(self._agent_id),
                    result_summary=f"Completed with {result['steps_executed']} steps",
                    duration_seconds=0.0,  # Could calculate actual duration
                )
            )

            # Create success memory
            await self._memory.create_memory(
                content=f"Goal completed: {goal.description}",
                memory_type=MemoryType.EPISODIC,
                importance=MemoryImportance.HIGH,
                agent_id=goal.agent_id or self._agent_id,
                context={"goal_id": str(goal.id), "steps": result["steps_executed"]},
            )

            result["success"] = True

            logger.info(
                "Goal completed successfully",
                goal_id=str(goal.id),
                steps=result["steps_executed"],
            )

        except Exception as e:
            logger.error("Goal processing failed", goal_id=str(goal.id), error=str(e))

            # Mark goal as failed
            goal.fail(str(e))
            await self._goal_repo.save(goal)

            # Publish failure event
            await self._events.publish(
                GoalFailed(
                    goal_id=str(goal.id),
                    agent_id=str(goal.agent_id) if goal.agent_id else str(self._agent_id),
                    error_message=str(e),
                    can_retry=goal.can_retry,
                )
            )

            # Create failure memory
            await self._memory.create_memory(
                content=f"Goal failed: {goal.description}. Error: {str(e)[:100]}",
                memory_type=MemoryType.EPISODIC,
                importance=MemoryImportance.CRITICAL,
                agent_id=goal.agent_id or self._agent_id,
                context={"goal_id": str(goal.id), "error": str(e)},
            )

            self._status.last_error = str(e)
            result["error"] = str(e)

        finally:
            self._status.current_goal_id = None

        return result

    async def _execute_step(
        self,
        goal: Goal,
        step: dict[str, Any],
        step_number: int,
    ) -> dict[str, Any]:
        """
        Execute a single plan step.

        Args:
            goal: Parent goal
            step: Step definition
            step_number: Step index for logging

        Returns:
            Execution result
        """
        description = step.get("description", "Unknown step")
        tool_name = step.get("tool")
        parameters = step.get("parameters", {})

        logger.debug(
            "Executing step",
            goal_id=str(goal.id),
            step=step_number,
            description=description[:50],
            tool=tool_name,
        )

        # If a tool is specified, execute it
        if tool_name:
            result = await self._tools.execute(tool_name, parameters)

            # Create memory about tool execution
            await self._memory.create_memory(
                content=f"Executed tool '{tool_name}' for step {step_number}: {description}",
                memory_type=MemoryType.PROCEDURAL,
                importance=MemoryImportance.MEDIUM,
                agent_id=goal.agent_id or self._agent_id,
                context={
                    "goal_id": str(goal.id),
                    "step": step_number,
                    "tool": tool_name,
                    "success": result.get("success", False),
                },
            )

            return result

        # No tool - just mark as completed
        return {"success": True, "message": "Step completed (no tool required)"}

    async def run_once(self) -> bool:
        """
        Run one iteration of the main loop.

        Returns:
            True if work was done, False if no pending goals
        """
        try:
            # Get pending goals
            pending = await self._goal_repo.get_pending_goals(
                agent_id=self._agent_id,
                limit=10,
            )

            if pending:
                # Sort by priority (lower number = higher priority)
                pending.sort(key=lambda g: g.priority_value)

                # Process highest priority goal
                goal = pending[0]
                await self.process_goal(goal)
                return True

            # No pending goals - idle
            logger.debug("No pending goals, loop idling")
            return False

        except Exception as e:
            logger.error("Error in run_once iteration", error=str(e))
            self._status.last_error = str(e)
            return False

    async def start(self) -> None:
        """
        Start the main autonomous loop.

        This method runs indefinitely until stop() is called or
        Ctrl+C is pressed (SIGINT).
        """
        logger.info("Starting Lya main loop")

        # Check LLM connectivity
        logger.info("Checking LLM connectivity...")
        if not await self.check_llm_connectivity():
            logger.error(
                "LLM not available at %s. Please ensure LLM is running.",
                settings.llm.base_url,
            )
            return

        logger.info("LLM connectivity confirmed")

        # Initialize state
        self._running = True
        self._paused = False
        self._stop_event = asyncio.Event()
        self._status.running = True
        self._status.started_at = datetime.now(timezone.utc)

        # Set up signal handlers
        self._setup_signal_handlers()

        logger.info("Main loop is now running! Press Ctrl+C to stop")

        # Run main loop
        try:
            while self._running:
                if not self._paused:
                    try:
                        worked = await self.run_once()
                        if not worked:
                            # No work to do - sleep briefly
                            await asyncio.wait_for(
                                self._stop_event.wait(),
                                timeout=1.0,
                            )
                            self._stop_event.clear()
                        else:
                            self._status.iteration_count += 1

                        # Update status
                        await self._update_status()

                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break
                else:
                    # Paused - wait
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
        except Exception as e:
            logger.error("Error in main loop", error=str(e))
            self._status.last_error = str(e)

        finally:
            self._running = False
            self._status.running = False
            logger.info("Main loop stopped")

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        try:
            loop = asyncio.get_running_loop()

            def signal_handler():
                logger.info("Received shutdown signal")
                self.stop()

            loop.add_signal_handler(signal.SIGINT, signal_handler)
            loop.add_signal_handler(signal.SIGTERM, signal_handler)

            logger.debug("Signal handlers registered")
        except (NotImplementedError, ValueError):
            # Windows doesn't support add_signal_handler in some cases
            logger.warning("Could not register signal handlers (platform limitation)")

    def stop(self) -> None:
        """Stop the main loop gracefully."""
        logger.info("Stopping main loop...")
        self._running = False
        self._status.running = False
        if self._stop_event:
            self._stop_event.set()

    def pause(self) -> None:
        """Pause execution (goals remain queued)."""
        if not self._paused:
            logger.info("Pausing main loop")
            self._paused = True
            self._status.paused = True

    def resume(self) -> None:
        """Resume execution after pause."""
        if self._paused:
            logger.info("Resuming main loop")
            self._paused = False
            self._status.paused = False

    async def _update_status(self) -> None:
        """Update internal status counters."""
        try:
            # Get goal counts
            total = await self._goal_repo.count(agent_id=self._agent_id)
            pending = await self._goal_repo.count(
                agent_id=self._agent_id,
                status=GoalStatus.PENDING,
            )
            in_progress = await self._goal_repo.count(
                agent_id=self._agent_id,
                status=GoalStatus.IN_PROGRESS,
            )
            completed = await self._goal_repo.count(
                agent_id=self._agent_id,
                status=GoalStatus.COMPLETED,
            )
            failed = await self._goal_repo.count(
                agent_id=self._agent_id,
                status=GoalStatus.FAILED,
            )

            self._status.goals_total = total
            self._status.goals_pending = pending
            self._status.goals_in_progress = in_progress
            self._status.goals_completed = completed
            self._status.goals_failed = failed

            # Update uptime
            if self._status.started_at:
                self._status.uptime_seconds = (
                    datetime.now(timezone.utc) - self._status.started_at
                ).total_seconds()

        except Exception as e:
            logger.warning("Failed to update status", error=str(e))

    def get_status(self) -> MainLoopStatus:
        """
        Get current status of the main loop.

        Returns:
            Current status report
        """
        return self._status

    async def get_status_dict(self) -> dict[str, Any]:
        """
        Get status as a dictionary (includes tool and memory counts).

        Returns:
            Full status report with all components
        """
        base_status = self._status.to_dict()

        # Get memory stats
        try:
            memory_stats = await self._memory.get_memory_stats(agent_id=self._agent_id)
        except Exception:
            memory_stats = {}

        # Get tool count
        try:
            tools = self._tools.list_tools()
            tool_count = len(tools)
        except Exception:
            tool_count = 0

        return {
            **base_status,
            "tools_available": tool_count,
            "memory_stats": memory_stats,
            "agent_id": str(self._agent_id),
            "improvement_enabled": self._improvement is not None,
        }

    @property
    def running(self) -> bool:
        """Whether the loop is currently running."""
        return self._running

    @property
    def paused(self) -> bool:
        """Whether execution is paused."""
        return self._paused


__all__ = [
    "MainLoop",
    "MainLoopStatus",
    "AddGoalRequest",
    "AddGoalResult",
]