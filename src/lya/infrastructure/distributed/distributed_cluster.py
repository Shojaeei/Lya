"""Distributed Computing - Infrastructure Implementation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import aiohttp

from lya.domain.models.distributed import (
    Node, DistributedTask, ClusterState,
    NodeStatus, TaskDistributionStrategy,
)
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class DistributedCluster:
    """
    Distributed computing cluster manager.

    Features:
    - Node registration and management
    - Task distribution with multiple strategies
    - Health monitoring
    - Load balancing
    - Failover handling
    """

    def __init__(
        self,
        node_id: UUID | None = None,
        node_name: str | None = None,
        host: str = "0.0.0.0",
        port: int = 8081,
    ):
        self.node_id = node_id or self._generate_node_id()
        self.node_name = node_name or f"lya-{self.node_id.hex[:8]}"
        self.host = host
        self.port = port

        self.is_coordinator = False
        self.coordinator: Node | None = None
        self.nodes: dict[UUID, Node] = {}
        self.tasks: dict[UUID, DistributedTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()

        self._running = False
        self._strategy = TaskDistributionStrategy.LEAST_LOADED
        self._session: aiohttp.ClientSession | None = None

        logger.info(
            "Distributed cluster initialized",
            node_id=str(self.node_id),
            name=self.node_name,
        )

    def _generate_node_id(self) -> UUID:
        """Generate unique node ID."""
        import socket
        hostname = socket.gethostname()
        unique_str = f"{hostname}:{datetime.now(timezone.utc).timestamp()}"
        return UUID(hashlib.md5(unique_str.encode()).hexdigest()[:32])

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def start(self) -> None:
        """Start the cluster node."""
        self._running = True

        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._task_processor())

        logger.info("Cluster node started", node=str(self.node_id))

    async def stop(self) -> None:
        """Stop the cluster node."""
        self._running = False

        if self._session and not self._session.closed:
            await self._session.close()

        logger.info("Cluster node stopped")

    async def register_node(self, host: str, port: int, name: str | None = None) -> Node | None:
        """Register a new node in the cluster."""
        try:
            # Check if node already exists
            for node in self.nodes.values():
                if node.host == host and node.port == port:
                    return node

            # Create node
            node_id = uuid4()
            node = Node(
                node_id=node_id,
                name=name or f"node-{node_id.hex[:8]}",
                host=host,
                port=port,
                status=NodeStatus.ONLINE,
            )

            # Check if node is reachable
            if await self._ping_node(node):
                self.nodes[node_id] = node
                logger.info("Node registered", node=str(node_id), host=host, port=port)
                return node
            else:
                logger.warning("Node unreachable", host=host, port=port)
                return None

        except Exception as e:
            logger.error("Failed to register node", error=str(e))
            return None

    async def unregister_node(self, node_id: UUID) -> bool:
        """Unregister a node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info("Node unregistered", node=str(node_id))
            return True
        return False

    async def _ping_node(self, node: Node) -> bool:
        """Check if node is reachable."""
        try:
            session = await self._get_session()
            async with session.get(
                f"http://{node.host}:{node.port}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def _heartbeat_loop(self) -> None:
        """Send heartbeats to all nodes."""
        while self._running:
            try:
                for node in list(self.nodes.values()):
                    if not await self._ping_node(node):
                        node.status = NodeStatus.OFFLINE
                        logger.warning("Node offline", node=str(node.node_id))
                    else:
                        node.last_heartbeat = datetime.now(timezone.utc)
                        if node.status == NodeStatus.OFFLINE:
                            node.status = NodeStatus.ONLINE

                await asyncio.sleep(30)  # Every 30 seconds

            except Exception as e:
                logger.error("Heartbeat error", error=str(e))
                await asyncio.sleep(30)

    async def submit_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        priority: int = 5,
    ) -> DistributedTask:
        """Submit a task to the cluster."""
        task = DistributedTask(
            task_id=uuid4(),
            task_type=task_type,
            payload=payload,
            priority=priority,
        )

        self.tasks[task.task_id] = task
        await self.task_queue.put(task)

        logger.info("Task submitted", task=str(task.task_id), type=task_type)
        return task

    async def _task_processor(self) -> None:
        """Process tasks from queue.""""
        while self._running:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                await self._distribute_task(task)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Task processor error", error=str(e))

    async def _distribute_task(self, task: DistributedTask) -> None:
        """Distribute task to available node."""
        task.status = "assigning"
        task.assigned_at = datetime.now(timezone.utc)

        # Find best node
        node = await self._select_node(task)

        if node:
            task.node_id = node.node_id
            task.status = "running"
            node.current_load = min(1.0, node.current_load + 0.1)

            try:
                result = await self._send_task_to_node(node, task)
                task.status = "completed"
                task.result = result
                task.completed_at = datetime.now(timezone.utc)
                node.current_load = max(0.0, node.current_load - 0.1)
                node.total_tasks_completed += 1

            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                node.total_tasks_failed += 1
                logger.error("Task execution failed", task=str(task.task_id), error=str(e))

        else:
            task.status = "failed"
            task.error = "No available nodes"
            logger.error("No nodes available for task", task=str(task.task_id))

    async def _select_node(self, task: DistributedTask) -> Node | None:
        """Select best node for task."""
        available = [
            n for n in self.nodes.values()
            if n.status == NodeStatus.ONLINE and n.current_load < 0.9
        ]

        if not available:
            return None

        if self._strategy == TaskDistributionStrategy.LEAST_LOADED:
            return min(available, key=lambda n: n.current_load)
        elif self._strategy == TaskDistributionStrategy.ROUND_ROBIN:
            # Simple round-robin based on hash
            index = hash(task.task_id) % len(available)
            return available[index]
        elif self._strategy == TaskDistributionStrategy.CAPABILITY_MATCH:
            # Find node with matching capabilities
            for node in available:
                if task.task_type in node.capabilities:
                    return node
            return available[0]  # Fallback
        else:
            return available[0]

    async def _send_task_to_node(self, node: Node, task: DistributedTask) -> Any:
        """Send task to remote node."""
        session = await self._get_session()

        async with session.post(
            f"http://{node.host}:{node.port}/execute",
            json={
                "task_id": str(task.task_id),
                "task_type": task.task_type,
                "payload": task.payload,
            },
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            data = await response.json()
            return data.get("result")

    async def broadcast(self, message: dict[str, Any]) -> dict[UUID, bool]:
        """Broadcast message to all nodes."""
        results = {}

        for node in self.nodes.values():
            if node.status == NodeStatus.ONLINE:
                try:
                    session = await self._get_session()
                    async with session.post(
                        f"http://{node.host}:{node.port}/broadcast",
                        json=message,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ):
                        results[node.node_id] = True
                except Exception:
                    results[node.node_id] = False
                    node.status = NodeStatus.OFFLINE

        return results

    def get_cluster_state(self) -> ClusterState:
        """Get current cluster state.""""
        online_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]
        total_load = sum(n.current_load for n in online_nodes) if online_nodes else 0
        avg_load = total_load / len(online_nodes) if online_nodes else 0

        return ClusterState(
            cluster_id=uuid4(),
            coordinator_id=self.coordinator.node_id if self.coordinator else None,
            nodes=list(self.nodes.values()),
            total_tasks=len(self.tasks),
            completed_tasks=sum(1 for t in self.tasks.values() if t.status == "completed"),
            failed_tasks=sum(1 for t in self.tasks.values() if t.status == "failed"),
            average_load=avg_load,
        )

    def get_node_stats(self) -> dict[str, Any]:
        """Get node statistics."""
        return {
            "this_node": str(self.node_id),
            "is_coordinator": self.is_coordinator,
            "total_nodes": len(self.nodes),
            "online_nodes": sum(1 for n in self.nodes.values() if n.status == NodeStatus.ONLINE),
            "nodes": [n.to_dict() for n in self.nodes.values()],
        }
