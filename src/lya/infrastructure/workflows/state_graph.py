"""State Machine Workflows for Lya.

Pure Python implementation of state machine graphs for agent workflows.
Compatible with Python 3.14+.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any
from uuid import uuid4


class NodeType(Enum):
    """Types of workflow nodes."""
    START = auto()
    END = auto()
    ACTION = auto()
    CONDITION = auto()
    PARALLEL = auto()
    WAIT = auto()


@dataclass
class State:
    """Workflow state."""
    data: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        self.data[key] = value

    def update(self, other: dict[str, Any]) -> None:
        """Update state with dict."""
        self.data.update(other)

    def copy(self) -> State:
        """Create a copy of state."""
        return State(
            data=self.data.copy(),
            history=self.history.copy(),
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data": self.data,
            "history": self.history,
            "metadata": self.metadata,
        }


@dataclass
class Edge:
    """Edge connecting two nodes."""
    from_node: str
    to_node: str
    condition: Callable[[State], bool] | None = None
    name: str = ""

    async def can_traverse(self, state: State) -> bool:
        """Check if edge can be traversed."""
        if self.condition is None:
            return True
        result = self.condition(state)
        if asyncio.iscoroutine(result):
            return await result
        return result


@dataclass
class Node:
    """Node in the workflow graph."""
    id: str
    name: str
    node_type: NodeType
    action: Callable[[State], Awaitable[State]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    success: bool
    final_state: State
    path: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    error: str | None = None


class StateGraph:
    """
    State machine graph for agent workflows.

    Provides LangGraph-like functionality for defining and executing
    state-based workflows with conditions, cycles, and parallelism.
    """

    def __init__(self, name: str = "workflow") -> None:
        """Initialize state graph.

        Args:
            name: Graph name
        """
        self.name = name
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, list[Edge]] = {}  # from_node -> edges
        self._start_node: str | None = None
        self._end_nodes: set[str] = set()

    def add_node(
        self,
        name: str,
        action: Callable[[State], Awaitable[State]],
        node_type: NodeType = NodeType.ACTION,
    ) -> StateGraph:
        """Add a node to the graph.

        Args:
            name: Node name
            action: Async function to execute
            node_type: Type of node

        Returns:
            Self for chaining
        """
        node_id = f"{name}_{len(self._nodes)}"
        self._nodes[node_id] = Node(
            id=node_id,
            name=name,
            node_type=node_type,
            action=action,
        )
        self._edges[node_id] = []

        if self._start_node is None:
            self._start_node = node_id

        return self

    def add_start_node(self, name: str = "start") -> StateGraph:
        """Add a start node.

        Args:
            name: Node name

        Returns:
            Self for chaining
        """
        node_id = f"{name}_{len(self._nodes)}"
        self._nodes[node_id] = Node(
            id=node_id,
            name=name,
            node_type=NodeType.START,
            action=None,
        )
        self._edges[node_id] = []
        self._start_node = node_id
        return self

    def add_end_node(self, name: str = "end") -> StateGraph:
        """Add an end node.

        Args:
            name: Node name

        Returns:
            Self for chaining
        """
        node_id = f"{name}_{len(self._nodes)}"
        self._nodes[node_id] = Node(
            id=node_id,
            name=name,
            node_type=NodeType.END,
            action=None,
        )
        self._edges[node_id] = []
        self._end_nodes.add(node_id)
        return self

    def add_conditional_node(
        self,
        name: str,
        action: Callable[[State], Awaitable[State]],
    ) -> StateGraph:
        """Add a conditional/decision node.

        Args:
            name: Node name
            action: Async function that returns updated state

        Returns:
            Self for chaining
        """
        node_id = f"{name}_{len(self._nodes)}"
        self._nodes[node_id] = Node(
            id=node_id,
            name=name,
            node_type=NodeType.CONDITION,
            action=action,
        )
        self._edges[node_id] = []
        return self

    def add_parallel_node(
        self,
        name: str,
        actions: list[Callable[[State], Awaitable[State]]],
    ) -> StateGraph:
        """Add a parallel execution node.

        Args:
            name: Node name
            actions: List of async functions to run in parallel

        Returns:
            Self for chaining
        """
        async def parallel_action(state: State) -> State:
            """Execute actions in parallel."""
            results = await asyncio.gather(
                *[action(state.copy()) for action in actions],
                return_exceptions=True,
            )

            # Merge results
            new_state = state.copy()
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    new_state.set(f"parallel_{i}_error", str(result))
                else:
                    new_state.set(f"parallel_{i}", result.data)

            return new_state

        node_id = f"{name}_{len(self._nodes)}"
        self._nodes[node_id] = Node(
            id=node_id,
            name=name,
            node_type=NodeType.PARALLEL,
            action=parallel_action,
        )
        self._edges[node_id] = []
        return self

    def add_edge(
        self,
        from_name: str,
        to_name: str,
        condition: Callable[[State], bool] | None = None,
    ) -> StateGraph:
        """Add an edge between nodes.

        Args:
            from_name: Source node name
            to_name: Target node name
            condition: Optional condition function

        Returns:
            Self for chaining
        """
        from_id = self._get_node_id(from_name)
        to_id = self._get_node_id(to_name)

        if from_id not in self._edges:
            self._edges[from_id] = []

        edge_name = f"{from_name}_to_{to_name}"
        self._edges[from_id].append(Edge(
            from_node=from_id,
            to_node=to_id,
            condition=condition,
            name=edge_name,
        ))

        return self

    def add_conditional_edges(
        self,
        from_name: str,
        conditions: dict[str, Callable[[State], bool]],
    ) -> StateGraph:
        """Add multiple conditional edges from a node.

        Args:
            from_name: Source node name
            conditions: Dict of target name -> condition function

        Returns:
            Self for chaining
        """
        for to_name, condition in conditions.items():
            self.add_edge(from_name, to_name, condition)

        return self

    def set_entry_point(self, name: str) -> StateGraph:
        """Set the entry point node.

        Args:
            name: Node name

        Returns:
            Self for chaining
        """
        node_id = self._get_node_id(name)
        self._start_node = node_id
        return self

    def _get_node_id(self, name: str) -> str:
        """Get node ID from name."""
        for node_id, node in self._nodes.items():
            if node.name == name or node_id.startswith(f"{name}_"):
                return node_id
        raise ValueError(f"Node not found: {name}")

    async def execute(
        self,
        initial_state: State | None = None,
        max_steps: int = 100,
    ) -> WorkflowResult:
        """Execute the workflow.

        Args:
            initial_state: Initial state
            max_steps: Maximum steps to prevent infinite loops

        Returns:
            Workflow result
        """
        import time

        start_time = time.time()
        state = initial_state or State()
        path: list[str] = []

        if self._start_node is None:
            return WorkflowResult(
                success=False,
                final_state=state,
                error="No start node defined",
            )

        current_node = self._start_node
        steps = 0

        try:
            while steps < max_steps:
                steps += 1
                node = self._nodes[current_node]
                path.append(node.name)

                # Record step in history
                state.history.append({
                    "step": steps,
                    "node": node.name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                # Execute node action
                if node.action is not None:
                    state = await node.action(state)

                # Check if end node
                if node.node_type == NodeType.END or current_node in self._end_nodes:
                    break

                # Find next node
                edges = self._edges.get(current_node, [])

                if not edges:
                    return WorkflowResult(
                        success=False,
                        final_state=state,
                        path=path,
                        error=f"Dead end at node {node.name}",
                    )

                # Find traversable edge
                next_node = None
                for edge in edges:
                    if await edge.can_traverse(state):
                        next_node = edge.to_node
                        break

                if next_node is None:
                    return WorkflowResult(
                        success=False,
                        final_state=state,
                        path=path,
                        error=f"No valid edge from node {node.name}",
                    )

                current_node = next_node

            else:
                return WorkflowResult(
                    success=False,
                    final_state=state,
                    path=path,
                    error=f"Max steps ({max_steps}) exceeded",
                )

            execution_time = (time.time() - start_time) * 1000

            return WorkflowResult(
                success=True,
                final_state=state,
                path=path,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return WorkflowResult(
                success=False,
                final_state=state,
                path=path,
                execution_time_ms=execution_time,
                error=str(e),
            )

    def visualize(self) -> str:
        """Generate Mermaid diagram of the graph.

        Returns:
            Mermaid diagram as string
        """
        lines = ["graph TD"]

        # Add nodes
        for node_id, node in self._nodes.items():
            if node.node_type == NodeType.START:
                lines.append(f"    {node_id}[({node.name})]")
            elif node.node_type == NodeType.END:
                lines.append(f"    {node_id}(({node.name}))")
            elif node.node_type == NodeType.CONDITION:
                lines.append(f"    {node_id}{{{node.name}}}")
            else:
                lines.append(f"    {node_id}[{node.name}]")

        # Add edges
        for from_id, edges in self._edges.items():
            for edge in edges:
                name = edge.name if edge.name else ""
                if edge.condition:
                    lines.append(f"    {from_id} -->|{name}| {edge.to_node}")
                else:
                    lines.append(f"    {from_id} --> {edge.to_node}")

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save graph to file.

        Args:
            path: Save path
        """
        data = {
            "name": self.name,
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "type": n.node_type.name,
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "from": e.from_node,
                    "to": e.to_node,
                    "name": e.name,
                }
                for edges in self._edges.values()
                for e in edges
            ],
            "start_node": self._start_node,
            "end_nodes": list(self._end_nodes),
        }

        Path(path).write_text(json.dumps(data, indent=2))


class WorkflowManager:
    """
    Manager for multiple workflows.

    Provides registration, execution, and monitoring of workflows.
    """

    def __init__(self) -> None:
        """Initialize workflow manager."""
        self._workflows: dict[str, StateGraph] = {}
        self._executions: dict[str, list[WorkflowResult]] = {}

    def register(self, name: str, graph: StateGraph) -> None:
        """Register a workflow.

        Args:
            name: Workflow name
            graph: State graph
        """
        self._workflows[name] = graph
        self._executions[name] = []

    async def run(
        self,
        name: str,
        initial_state: State | None = None,
    ) -> WorkflowResult:
        """Run a workflow.

        Args:
            name: Workflow name
            initial_state: Initial state

        Returns:
            Workflow result
        """
        if name not in self._workflows:
            return WorkflowResult(
                success=False,
                final_state=initial_state or State(),
                error=f"Workflow not found: {name}",
            )

        graph = self._workflows[name]
        result = await graph.execute(initial_state)

        self._executions[name].append(result)

        return result

    def get_workflow(self, name: str) -> StateGraph | None:
        """Get a workflow by name."""
        return self._workflows.get(name)

    def list_workflows(self) -> list[str]:
        """List all workflow names."""
        return list(self._workflows.keys())

    def get_execution_history(self, name: str) -> list[WorkflowResult]:
        """Get execution history for a workflow."""
        return self._executions.get(name, [])


# ═══════════════════════════════════════════════════════════════════════
# COMMON WORKFLOW PATTERNS
# ═══════════════════════════════════════════════════════════════════════

def create_react_workflow(
    tool_registry: Any,
    llm_client: Any,
    max_iterations: int = 5,
) -> StateGraph:
    """
    Create a ReAct (Reasoning + Acting) workflow.

    The agent thinks, acts, and observes in a loop until done.

    Args:
        tool_registry: Tool registry
        llm_client: LLM client
        max_iterations: Maximum iterations

    Returns:
        ReAct workflow graph
    """
    graph = StateGraph(name="react_agent")

    async def think(state: State) -> State:
        """Reason about what to do next."""
        question = state.get("question", "")
        observations = state.get("observations", [])

        # Call LLM to decide next action
        context = f"Question: {question}\nObservations: {observations}"
        # In real implementation, call llm_client.generate()

        new_state = state.copy()
        new_state.set("thought", f"Analyzing: {question}")
        new_state.set("action_needed", True)

        return new_state

    async def act(state: State) -> State:
        """Execute action using tools."""
        thought = state.get("thought", "")

        # Determine which tool to use
        new_state = state.copy()
        new_state.set("action", "search")
        new_state.set("action_input", thought)

        return new_state

    async def observe(state: State) -> State:
        """Observe results of action."""
        action = state.get("action", "")
        action_input = state.get("action_input", "")

        # Execute tool and observe
        # tool_result = await tool_registry.execute(action, action_input)

        new_state = state.copy()
        observations = new_state.get("observations", [])
        observations.append(f"Action {action} completed")
        new_state.set("observations", observations)

        # Check if we have an answer
        iteration = new_state.get("iteration", 0) + 1
        new_state.set("iteration", iteration)

        if iteration >= max_iterations:
            new_state.set("done", True)
            new_state.set("answer", "Final answer after analysis")

        return new_state

    async def finish(state: State) -> State:
        """Finalize with answer."""
        new_state = state.copy()
        new_state.set("status", "completed")
        return new_state

    # Build graph
    graph.add_start_node("start")
    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.add_node("observe", observe)
    graph.add_node("finish", finish)
    graph.add_end_node("end")

    graph.add_edge("start", "think")
    graph.add_edge("think", "act")
    graph.add_edge("act", "observe")

    # Cycle back or finish
    def should_continue(state: State) -> bool:
        return not state.get("done", False)

    graph.add_conditional_edges("observe", {
        "think": should_continue,
        "finish": lambda s: not should_continue(s),
    })

    graph.add_edge("finish", "end")

    return graph


def create_plan_and_execute_workflow(
    planner: Any,
    executor: Any,
) -> StateGraph:
    """
    Create a plan-and-execute workflow.

    First plans all steps, then executes them sequentially.

    Args:
        planner: Planning component
        executor: Execution component

    Returns:
        Plan-and-execute workflow graph
    """
    graph = StateGraph(name="plan_and_execute")

    async def plan(state: State) -> State:
        """Create plan."""
        goal = state.get("goal", "")

        # planner.create_plan(goal)
        plan_steps = ["Step 1", "Step 2", "Step 3"]

        new_state = state.copy()
        new_state.set("plan", plan_steps)
        new_state.set("current_step", 0)

        return new_state

    async def execute_step(state: State) -> State:
        """Execute one step of the plan."""
        plan = state.get("plan", [])
        current = state.get("current_step", 0)

        if current < len(plan):
            # executor.execute(plan[current])

            new_state = state.copy()
            new_state.set("current_step", current + 1)

            results = new_state.get("results", [])
            results.append(f"Executed {plan[current]}")
            new_state.set("results", results)

            return new_state

        return state.copy()

    async def synthesize(state: State) -> State:
        """Synthesize final answer."""
        results = state.get("results", [])

        new_state = state.copy()
        new_state.set("answer", f"Synthesized from {len(results)} steps")

        return new_state

    # Build graph
    graph.add_start_node("start")
    graph.add_node("plan", plan)
    graph.add_node("execute", execute_step)
    graph.add_node("synthesize", synthesize)
    graph.add_end_node("end")

    graph.add_edge("start", "plan")
    graph.add_edge("plan", "execute")

    def has_more_steps(state: State) -> bool:
        plan = state.get("plan", [])
        current = state.get("current_step", 0)
        return current < len(plan)

    graph.add_conditional_edges("execute", {
        "execute": has_more_steps,
        "synthesize": lambda s: not has_more_steps(s),
    })

    graph.add_edge("synthesize", "end")

    return graph


def create_reflection_workflow(
    generator: Any,
    critic: Any,
    max_reflections: int = 3,
) -> StateGraph:
    """
    Create a reflection workflow.

    Generates output, reflects on it, and improves iteratively.

    Args:
        generator: Generation component
        critic: Critic component
        max_reflections: Maximum reflection cycles

    Returns:
        Reflection workflow graph
    """
    graph = StateGraph(name="reflection")

    async def generate(state: State) -> State:
        """Generate initial output."""
        prompt = state.get("prompt", "")

        # output = generator.generate(prompt)
        new_state = state.copy()
        new_state.set("output", f"Generated: {prompt}")

        return new_state

    async def reflect(state: State) -> State:
        """Reflect on output."""
        output = state.get("output", "")

        # feedback = critic.evaluate(output)
        new_state = state.copy()
        new_state.set("feedback", "Good but could be improved")

        reflection_count = new_state.get("reflection_count", 0) + 1
        new_state.set("reflection_count", reflection_count)

        if reflection_count >= max_reflections:
            new_state.set("satisfied", True)

        return new_state

    async def improve(state: State) -> State:
        """Improve based on feedback."""
        output = state.get("output", "")
        feedback = state.get("feedback", "")

        # improved = generator.improve(output, feedback)
        new_state = state.copy()
        new_state.set("output", f"Improved: {output}")

        return new_state

    async def finalize(state: State) -> State:
        """Finalize output."""
        new_state = state.copy()
        new_state.set("final_output", new_state.get("output"))
        return new_state

    # Build graph
    graph.add_start_node("start")
    graph.add_node("generate", generate)
    graph.add_node("reflect", reflect)
    graph.add_node("improve", improve)
    graph.add_node("finalize", finalize)
    graph.add_end_node("end")

    graph.add_edge("start", "generate")
    graph.add_edge("generate", "reflect")

    def needs_improvement(state: State) -> bool:
        return not state.get("satisfied", False)

    graph.add_conditional_edges("reflect", {
        "improve": needs_improvement,
        "finalize": lambda s: not needs_improvement(s),
    })

    graph.add_edge("improve", "reflect")
    graph.add_edge("finalize", "end")

    return graph


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

async def main():
    """Example workflow execution."""
    # Create a simple workflow
    graph = StateGraph(name="example")

    async def step1(state: State) -> State:
        print("Step 1: Processing input")
        new_state = state.copy()
        new_state.set("processed", True)
        return new_state

    async def step2(state: State) -> State:
        print("Step 2: Analyzing data")
        new_state = state.copy()
        new_state.set("analyzed", True)
        return new_state

    async def step3(state: State) -> State:
        print("Step 3: Finalizing")
        new_state = state.copy()
        new_state.set("finalized", True)
        return new_state

    graph.add_start_node("start")
    graph.add_node("process", step1)
    graph.add_node("analyze", step2)
    graph.add_node("finalize", step3)
    graph.add_end_node("end")

    graph.add_edge("start", "process")
    graph.add_edge("process", "analyze")
    graph.add_edge("analyze", "finalize")
    graph.add_edge("finalize", "end")

    # Execute
    result = await graph.execute(State({"input": "test"}))
    print(f"\nResult: {result}")
    print(f"\nPath: {result.path}")
    print(f"Final state: {result.final_state.to_dict()}")

    # Visualize
    print(f"\nVisualization:\n{graph.visualize()}")

    # ReAct workflow example
    print("\n--- ReAct Workflow ---")
    react = create_react_workflow(None, None, max_iterations=2)
    react_result = await react.execute(State({"question": "What is AI?"}))
    print(f"ReAct result: {react_result.success}")
    print(f"Path: {react_result.path}")


if __name__ == "__main__":
    asyncio.run(main())
