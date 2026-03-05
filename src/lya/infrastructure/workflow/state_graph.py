"""LangGraph-style State Machine for Lya.

Workflow orchestration with state management, cycles, and multi-step reasoning.
Pure Python 3.14+ compatible implementation (no external dependencies).
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, TypeVar, Generic

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class StateStatus(Enum):
    """Status of a state node."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


class EdgeType(Enum):
    """Type of edge between nodes."""
    DEFAULT = auto()
    CONDITIONAL = auto()
    PARALLEL = auto()
    CALLBACK = auto()


@dataclass
class State:
    """State in the state machine."""

    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    current_node: str | None = None
    execution_count: int = 0
    start_time: float | None = None
    end_time: float | None = None

    def update(self, updates: dict[str, Any]) -> State:
        """Update state with new data."""
        self.data.update(updates)
        self.history.append({
            "timestamp": time.time(),
            "updates": updates,
            "node": self.current_node,
        })
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> State:
        """Set value in state."""
        self.data[key] = value
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data": self.data,
            "metadata": self.metadata,
            "history_count": len(self.history),
            "current_node": self.current_node,
            "execution_count": self.execution_count,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class NodeResult:
    """Result of node execution."""

    next_node: str | None
    state_updates: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    status: StateStatus = StateStatus.COMPLETED
    error: str | None = None


@dataclass
class Edge:
    """Edge connecting nodes in the graph."""

    source: str
    target: str
    edge_type: EdgeType = EdgeType.DEFAULT
    condition: Callable[[State], bool] | None = None
    label: str | None = None

    def should_follow(self, state: State) -> bool:
        """Check if this edge should be followed."""
        if self.condition is None:
            return True
        try:
            return self.condition(state)
        except Exception as e:
            logger.error(
                "Edge condition failed",
                edge=f"{self.source}->{self.target}",
                error=str(e),
            )
            return False


class Node(ABC):
    """Abstract base class for state machine nodes."""

    def __init__(
        self,
        name: str,
        description: str = "",
        max_retries: int = 0,
    ) -> None:
        """Initialize node.

        Args:
            name: Node identifier
            description: Node description
            max_retries: Maximum retries on failure
        """
        self.name = name
        self.description = description
        self.max_retries = max_retries
        self._retry_count = 0

    @abstractmethod
    async def execute(self, state: State) -> NodeResult:
        """Execute the node.

        Args:
            state: Current state

        Returns:
            Node result with next node and updates
        """
        pass

    async def on_error(self, state: State, error: Exception) -> NodeResult:
        """Handle execution error.

        Args:
            state: Current state
            error: Exception that occurred

        Returns:
            Fallback result
        """
        logger.error(
            "Node execution failed",
            node=self.name,
            error=str(error),
        )
        return NodeResult(
            next_node=None,
            state_updates={"error": str(error), "failed_node": self.name},
            status=StateStatus.FAILED,
            error=str(error),
        )


class StateGraph:
    """
    State machine graph for workflow orchestration.

    Features:
    - State nodes with transitions
    - Conditional edges
    - Cycles and loops
    - Parallel execution
    - State persistence
    """

    def __init__(self, name: str = "graph") -> None:
        """Initialize state graph.

        Args:
            name: Graph name
        """
        self.name = name
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, list[Edge]] = {}
        self._entry_point: str | None = None

    def add_node(self, node: Node) -> StateGraph:
        """Add a node to the graph.

        Args:
            node: Node to add

        Returns:
            Self for chaining
        """
        self._nodes[node.name] = node
        if node.name not in self._edges:
            self._edges[node.name] = []
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        condition: Callable[[State], bool] | None = None,
        label: str | None = None,
    ) -> StateGraph:
        """Add an edge between nodes.

        Args:
            source: Source node name
            target: Target node name
            condition: Optional condition function
            label: Edge label

        Returns:
            Self for chaining
        """
        edge_type = EdgeType.CONDITIONAL if condition else EdgeType.DEFAULT

        edge = Edge(
            source=source,
            target=target,
            edge_type=edge_type,
            condition=condition,
            label=label,
        )

        if source not in self._edges:
            self._edges[source] = []
        self._edges[source].append(edge)

        return self

    def add_conditional_edges(
        self,
        source: str,
        conditions: dict[str, Callable[[State], bool]],
    ) -> StateGraph:
        """Add multiple conditional edges from a node.

        Args:
            source: Source node name
            conditions: Dict of target -> condition function

        Returns:
            Self for chaining
        """
        for target, condition in conditions.items():
            self.add_edge(source, target, condition, label=f"condition_{target}")
        return self

    def set_entry_point(self, node_name: str) -> StateGraph:
        """Set the entry point node.

        Args:
            node_name: Name of entry node

        Returns:
            Self for chaining
        """
        self._entry_point = node_name
        return self

    def compile(self) -> CompiledGraph:
        """Compile the graph for execution.

        Returns:
            Compiled graph ready for execution
        """
        if not self._entry_point:
            raise ValueError("Entry point not set. Call set_entry_point()")

        if self._entry_point not in self._nodes:
            raise ValueError(f"Entry point node not found: {self._entry_point}")

        return CompiledGraph(
            name=self.name,
            nodes=self._nodes.copy(),
            edges=self._edges.copy(),
            entry_point=self._entry_point,
        )


class CompiledGraph:
    """Compiled state graph ready for execution."""

    def __init__(
        self,
        name: str,
        nodes: dict[str, Node],
        edges: dict[str, list[Edge]],
        entry_point: str,
    ) -> None:
        """Initialize compiled graph."""
        self.name = name
        self._nodes = nodes
        self._edges = edges
        self._entry_point = entry_point

    async def execute(
        self,
        initial_state: dict[str, Any] | None = None,
        max_steps: int = 100,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Execute the graph.

        Args:
            initial_state: Initial state data
            max_steps: Maximum steps to prevent infinite loops
            timeout_seconds: Optional timeout

        Returns:
            Final state
        """
        state = State(
            data=initial_state or {},
            start_time=time.time(),
        )
        state.current_node = self._entry_point

        start_time = time.time()
        step_count = 0

        while state.current_node and step_count < max_steps:
            # Check timeout
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                state.update({"timeout": True})
                break

            node_name = state.current_node
            node = self._nodes.get(node_name)

            if not node:
                logger.error("Node not found", node=node_name)
                state.update({"error": f"Node not found: {node_name}"})
                break

            # Execute node
            state.execution_count += 1
            try:
                result = await node.execute(state)
                state.update(result.state_updates)

                if result.output is not None:
                    state.set(f"{node_name}_output", result.output)

                if result.status == StateStatus.FAILED:
                    state.update({"failed_at": node_name, "error": result.error})
                    break

                # Determine next node
                if result.next_node is None:
                    # Follow edges
                    next_node = self._get_next_node(node_name, state)
                    state.current_node = next_node
                else:
                    state.current_node = result.next_node

            except Exception as e:
                result = await node.on_error(state, e)
                state.update(result.state_updates)
                state.current_node = result.next_node

                if result.status == StateStatus.FAILED:
                    break

            step_count += 1

            # Detect potential infinite loop
            if step_count >= max_steps:
                state.update({"max_steps_reached": True})

        state.end_time = time.time()
        return state.to_dict()

    def _get_next_node(self, current: str, state: State) -> str | None:
        """Determine next node based on edges."""
        edges = self._edges.get(current, [])

        for edge in edges:
            if edge.should_follow(state):
                return edge.target

        return None

    def visualize(self) -> str:
        """Generate Mermaid diagram of the graph.

        Returns:
            Mermaid diagram string
        """
        lines = ["stateDiagram-v2"]

        for node_name in self._nodes:
            lines.append(f"    {node_name}")

        for source, edges in self._edges.items():
            for edge in edges:
                label = edge.label or ""
                if label:
                    lines.append(f"    {source} --> {edge.target}: {label}")
                else:
                    lines.append(f"    {source} --> {edge.target}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# PRE-BUILT NODES
# ═══════════════════════════════════════════════════════════════════════

class FunctionNode(Node):
    """Node that executes a function."""

    def __init__(
        self,
        name: str,
        func: Callable[[State], dict[str, Any]],
        description: str = "",
        next_node: str | None = None,
    ) -> None:
        """Initialize function node.

        Args:
            name: Node name
            func: Function to execute (takes State, returns updates dict)
            description: Node description
            next_node: Default next node
        """
        super().__init__(name, description)
        self.func = func
        self.next_node = next_node

    async def execute(self, state: State) -> NodeResult:
        """Execute function."""
        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(state)
            else:
                result = self.func(state)

            if not isinstance(result, dict):
                result = {"result": result}

            return NodeResult(
                next_node=self.next_node,
                state_updates=result,
                status=StateStatus.COMPLETED,
            )
        except Exception as e:
            return await self.on_error(state, e)


class LLMNode(Node):
    """Node that calls an LLM."""

    def __init__(
        self,
        name: str,
        prompt_template: str,
        llm_client: Any,
        description: str = "",
        output_key: str = "llm_response",
        next_node: str | None = None,
    ) -> None:
        """Initialize LLM node.

        Args:
            name: Node name
            prompt_template: Template for prompt
            llm_client: LLM client to use
            description: Node description
            output_key: Key to store response in state
            next_node: Default next node
        """
        super().__init__(name, description)
        self.prompt_template = prompt_template
        self.llm_client = llm_client
        self.output_key = output_key
        self.next_node = next_node

    async def execute(self, state: State) -> NodeResult:
        """Execute LLM call."""
        try:
            # Format prompt with state
            prompt = self.prompt_template.format(**state.data)

            # Call LLM (assumes async client)
            response = await self.llm_client.complete(prompt)

            return NodeResult(
                next_node=self.next_node,
                state_updates={self.output_key: response},
                output=response,
                status=StateStatus.COMPLETED,
            )
        except Exception as e:
            return await self.on_error(state, e)


class ToolNode(Node):
    """Node that executes a tool."""

    def __init__(
        self,
        name: str,
        tool_registry: Any,
        tool_name: str,
        description: str = "",
        output_key: str = "tool_result",
        next_node: str | None = None,
    ) -> None:
        """Initialize tool node.

        Args:
            name: Node name
            tool_registry: Tool registry
            tool_name: Name of tool to execute
            description: Node description
            output_key: Key to store result in state
            next_node: Default next node
        """
        super().__init__(name, description)
        self.tool_registry = tool_registry
        self.tool_name = tool_name
        self.output_key = output_key
        self.next_node = next_node

    async def execute(self, state: State) -> NodeResult:
        """Execute tool."""
        try:
            # Get tool params from state
            params = state.get("tool_params", {})

            # Execute tool
            result = await self.tool_registry.execute(self.tool_name, params)

            return NodeResult(
                next_node=self.next_node,
                state_updates={
                    self.output_key: result,
                    f"{self.tool_name}_result": result,
                },
                output=result,
                status=StateStatus.COMPLETED,
            )
        except Exception as e:
            return await self.on_error(state, e)


class DecisionNode(Node):
    """Node that makes a decision based on state."""

    def __init__(
        self,
        name: str,
        decision_fn: Callable[[State], tuple[str, dict[str, Any]]],
        description: str = "",
    ) -> None:
        """Initialize decision node.

        Args:
            name: Node name
            decision_fn: Function that returns (next_node, updates)
            description: Node description
        """
        super().__init__(name, description)
        self.decision_fn = decision_fn

    async def execute(self, state: State) -> NodeResult:
        """Execute decision."""
        try:
            if asyncio.iscoroutinefunction(self.decision_fn):
                next_node, updates = await self.decision_fn(state)
            else:
                next_node, updates = self.decision_fn(state)

            return NodeResult(
                next_node=next_node,
                state_updates=updates,
                status=StateStatus.COMPLETED,
            )
        except Exception as e:
            return await self.on_error(state, e)


class ParallelNode(Node):
    """Node that executes multiple branches in parallel."""

    def __init__(
        self,
        name: str,
        branches: list[Node],
        description: str = "",
        next_node: str | None = None,
    ) -> None:
        """Initialize parallel node.

        Args:
            name: Node name
            branches: Nodes to execute in parallel
            description: Node description
            next_node: Default next node after all complete
        """
        super().__init__(name, description)
        self.branches = branches
        self.next_node = next_node

    async def execute(self, state: State) -> NodeResult:
        """Execute branches in parallel."""
        try:
            results = await asyncio.gather(
                *[branch.execute(state) for branch in self.branches],
                return_exceptions=True,
            )

            # Merge state updates
            all_updates = {}
            outputs = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    all_updates[f"branch_{i}_error"] = str(result)
                    outputs.append(None)
                else:
                    all_updates.update(result.state_updates)
                    outputs.append(result.output)

            all_updates["parallel_outputs"] = outputs

            return NodeResult(
                next_node=self.next_node,
                state_updates=all_updates,
                status=StateStatus.COMPLETED,
            )
        except Exception as e:
            return await self.on_error(state, e)


class LoopNode(Node):
    """Node that loops until condition is met."""

    def __init__(
        self,
        name: str,
        body: Node,
        condition: Callable[[State], bool],
        max_iterations: int = 10,
        description: str = "",
        next_node: str | None = None,
    ) -> None:
        """Initialize loop node.

        Args:
            name: Node name
            body: Node to execute in loop
            condition: Condition to continue looping
            max_iterations: Maximum iterations
            description: Node description
            next_node: Node to go to after loop
        """
        super().__init__(name, description)
        self.body = body
        self.condition = condition
        self.max_iterations = max_iterations
        self.next_node = next_node

    async def execute(self, state: State) -> NodeResult:
        """Execute loop."""
        iteration = 0

        while iteration < self.max_iterations:
            if not self.condition(state):
                break

            result = await self.body.execute(state)
            state.update(result.state_updates)

            if result.status == StateStatus.FAILED:
                return result

            iteration += 1
            state.set(f"{self.name}_iteration", iteration)

        if iteration >= self.max_iterations:
            state.set(f"{self.name}_max_iterations_reached", True)

        return NodeResult(
            next_node=self.next_node,
            state_updates={f"{self.name}_iterations": iteration},
            status=StateStatus.COMPLETED,
        )


# ═══════════════════════════════════════════════════════════════════════
# WORKFLOW BUILDERS
# ═══════════════════════════════════════════════════════════════════════

class WorkflowBuilder:
    """Helper class for building common workflows."""

    @staticmethod
    def create_react_agent(
        llm_client: Any,
        tool_registry: Any,
        max_iterations: int = 5,
    ) -> CompiledGraph:
        """Create a ReAct (Reasoning + Acting) agent workflow.

        Args:
            llm_client: LLM client
            tool_registry: Tool registry
            max_iterations: Maximum thinking-acting cycles

        Returns:
            Compiled graph
        """
        graph = StateGraph("react_agent")

        # Thought node
        thought = LLMNode(
            name="thought",
            prompt_template="""You are a reasoning agent. Given the task and previous actions, decide what to do next.

Task: {task}
Previous actions: {action_history}
Observations: {observations}

Think step by step. What should you do next?""",
            llm_client=llm_client,
            description="Generate next thought",
            output_key="thought",
        )

        # Action node
        action = ToolNode(
            name="action",
            tool_registry=tool_registry,
            tool_name="{selected_tool}",  # Dynamic tool selection
            description="Execute action",
            output_key="observation",
        )

        # Decision node
        def decide_next(state: State) -> tuple[str, dict[str, Any]]:
            iterations = state.get("iteration", 0)
            if iterations >= max_iterations:
                return "finish", {"max_iterations_reached": True}

            thought_text = state.get("thought", "")
            if "FINAL ANSWER" in thought_text.upper():
                return "finish", {}

            return "action", {"iteration": iterations + 1}

        decision = DecisionNode(
            name="decide",
            decision_fn=decide_next,
            description="Decide next step",
        )

        # Finish node
        finish = FunctionNode(
            name="finish",
            func=lambda s: {"final_answer": s.get("thought", "")},
            description="Finish and return answer",
        )

        # Add nodes
        graph.add_node(thought)
        graph.add_node(action)
        graph.add_node(decision)
        graph.add_node(finish)

        # Add edges
        graph.add_edge("thought", "decide")
        graph.add_edge("decide", "action")  # Continue loop
        graph.add_edge("action", "thought")  # Back to thought
        graph.add_edge("decide", "finish")  # Done

        graph.set_entry_point("thought")

        return graph.compile()

    @staticmethod
    def create_sequential_pipeline(
        steps: list[tuple[str, Callable[[State], dict[str, Any]]]],
    ) -> CompiledGraph:
        """Create a simple sequential pipeline.

        Args:
            steps: List of (name, function) tuples

        Returns:
            Compiled graph
        """
        graph = StateGraph("pipeline")

        prev_node = None
        for name, func in steps:
            node = FunctionNode(
                name=name,
                func=func,
                description=f"Step: {name}",
            )
            graph.add_node(node)

            if prev_node:
                graph.add_edge(prev_node, name)
            else:
                graph.set_entry_point(name)

            prev_node = name

        return graph.compile()


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: Simple workflow
    async def demo():
        graph = StateGraph("demo")

        # Create nodes
        start = FunctionNode(
            name="start",
            func=lambda s: {"started": True, "step": 1},
            description="Initialize",
            next_node="process",
        )

        process = FunctionNode(
            name="process",
            func=lambda s: {"processed": True, "data": s.get("input", "") * 2},
            description="Process data",
        )

        finish = FunctionNode(
            name="finish",
            func=lambda s: {"finished": True, "result": s.get("data", "")},
            description="Finish",
        )

        # Build graph
        graph.add_node(start)
        graph.add_node(process)
        graph.add_node(finish)

        graph.add_edge("start", "process")
        graph.add_edge("process", "finish")
        graph.set_entry_point("start")

        # Compile and run
        compiled = graph.compile()
        result = await compiled.execute(initial_state={"input": "hello"})

        print("Result:", json.dumps(result, indent=2))
        print("\nVisualization:")
        print(compiled.visualize())

    # Run demo
    asyncio.run(demo())
