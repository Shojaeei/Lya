"""Distributed Cluster Manager - Infrastructure Implementation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from lya.domain.models.distributed import (
    Node, DistributedTask, ClusterState,
    NodeStatus, TaskDistributionStrategy,
)
from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


class ClusterManager:
    """
    Distributed cluster manager for multi-node scaling.

    Features:
    - Node registration and management
    - Task distribution with load balancing
    - Health monitoring of nodes
    - Failover handling
    - Cluster state management
    """

    def __init__(
        self,
        node_id: UUID | None = None,
        strategy: TaskDistributionStrategy = TaskDistributionStrategy.ROUND_ROBIN,
    ):
        self.node_id = node_id or self._generate_node_id()
        self.strategy = strategy
        self.is_coordinator = False

        # Cluster state
        self.nodes: dict[UUID, Node] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.results: dict[UUID, Any] = {}
        self._task_counter = 0
        self._lock = asyncio.Lock()

        # Cluster stats
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0

        logger.info(
            "Cluster manager initialized",
            node_id=str(self.node_id),
            strategy=strategy.name,
        )

    def _generate_node_id(self) -> UUID:
        """Generate unique node ID."""
        import socket
        hostname = socket.gethostname()
        hash_val = hashlib.md5(f"{hostname}{datetime.now(timezone.utc)}".encode()).hexdigest()[:8]
        return uuid4()

    async def register_node(
        self,
        host: str,
        port: int,
        name: str | None = None,
        capabilities: list[str] | None = None,
    ) -> Node | None:
        """Register a new node in cluster."""
        if not HAS_AIOHTTP:
            logger.error("aiohttp required for cluster communication")
            return None

        node_id = uuid4()

        node = Node(
            node_id=node_id,
            name=name or f"node-{str(node_id)[:6]}",
            host=host,
            port=port,
            status=NodeStatus.OFFLINE,
            capabilities=capabilities or [],
        )

        # Check if node is online
        if await self._ping_node(node):
            node.status = NodeStatus.ONLINE
            async with self._lock:
                self.nodes[node_id] = node

            logger.info(
                "Node registered",
                node_id=str(node_id),
                host=host,
                port=port,
                name=node.name,
            )
            return node
        else:
            logger.warning(f"Node {host}:{port} unreachable")
            return None

    async def _ping_node(self, node: Node, timeout: int = 5) -> bool:
        """Check if node is reachable."""
        if not HAS_AIOHTTP:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{node.host}:{node.port}/health",
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    async def heartbeat(self, node_id: UUID) -> None:
        """Update node heartbeat."""
        if node_id in self.nodes:
            self.nodes[node_id].last_heartbeat = datetime.now(timezone.utc)

    async def check_nodes(self) -> None:
        """Check health of all nodes."""
        for node in list(self.nodes.values()):
            is_online = await self._ping_node(node, timeout=3)

            if is_online and node.status == NodeStatus.OFFLINE:
                node.status = NodeStatus.ONLINE
                logger.info("Node came online", node_id=str(node.node_id), name=node.name)
            elif not is_online and node.status == NodeStatus.ONLINE:
                node.status = NodeStatus.OFFLINE
                logger.warning("Node went offline", node_id=str(node.node_id), name=node.name)

    async def submit_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        priority: int = 5,
        required_capabilities: list[str] | None = None,
    ) -> DistributedTask:
        """Submit a task to the cluster."""
        task = DistributedTask(
            task_id=uuid4(),
            task_type=task_type,
            payload=payload,
            priority=priority,
        )

        await self.task_queue.put(task)
        self.total_tasks_submitted += 1

        logger.debug(
            "Task submitted",
            task_id=str(task.task_id),
            type=task_type,
            priority=priority,
        )

        return task

    async def distribute_task(self, task: DistributedTask) -> Any:
        """Distribute task to available node."""
        # Get available nodes
        available = [n for n in self.nodes.values() if n.is_healthy()]

        if not available:
            logger.warning("No available nodes, executing locally")
            return await self._execute_local(task)

        # Select node based on strategy
        node = self._select_node(available, task)

        if node:
            task.node_id = node.node_id
            task.status = "assigned"
            task.assigned_at = datetime.now(timezone.utc)
            node.current_load += 1

            result = await self._send_task_to_node(node, task)

            node.current_load = max(0, node.current_load - 1)
            node.total_tasks_completed += 1

            return result

        return await self._execute_local(task)

    def _select_node(
        self,
        available: list[Node],
        task: DistributedTask,
    ) -> Node | None:
        """Select node based on distribution strategy."""
        if not available:
            return None

        if self.strategy == TaskDistributionStrategy.ROUND_ROBIN:
            # Simple round-robin
            self._task_counter = (self._task_counter + 1) % len(available)
            return available[self._task_counter]

        elif self.strategy == TaskDistributionStrategy.LEAST_LOADED:
            # Select node with lowest load
            return min(available, key=lambda n: n.current_load)

        elif self.strategy == TaskDistributionStrategy.HASH_BASED:
            # Hash-based selection
            hash_val = hash(task.task_id) % len(available)
            return available[hash_val]

        else:
            # Default: least loaded
            return min(available, key=lambda n: n.current_load)

    async def _send_task_to_node(self, node: Node, task: DistributedTask) -> Any:
        """Send task to remote node."""
        if not HAS_AIOHTTP:
            return {"error": "aiohttp not available"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{node.host}:{node.port}/execute",
                    json=task.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        task.status = "completed"
                        task.completed_at = datetime.now(timezone.utc)
                        task.result = data.get("result")
                        return data.get("result")
                    else:
                        raise Exception(f"Node returned {response.status}")

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            node.status = NodeStatus.OFFLINE
            logger.error(
                "Task failed on node",
                node_id=str(node.node_id),
                error=str(e),
            )
            return {"error": str(e)}

    async def _execute_local(self, task: DistributedTask) -> Any:
        """Execute task locally as fallback."""
        task.status = "running"

        # This would call local execution
        # For now, return placeholder
        result = {"executed": "local", "task_type": task.task_type}

        task.status = "completed"
        task.completed_at = datetime.now(timezone.utc)
        task.result = result

        return result

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all nodes."""
        tasks = []
        for node in self.nodes.values():
            if node.status == NodeStatus.ONLINE:
                tasks.append(self._send_to_node(node, message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_to_node(self, node: Node, message: dict[str, Any]) -> None:
        """Send message to single node."""
        if not HAS_AIOHTTP:
            return

        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"http://{node.host}:{node.port}/broadcast",
                    json=message,
                    timeout=aiohttp.ClientTimeout(total=10),
                )
        except Exception:
            node.status = NodeStatus.OFFLINE

    async def unregister_node(self, node_id: UUID) -> bool:
        """Unregister a node."""
        async with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info("Node unregistered", node_id=str(node_id))
                return True
        return False

    def get_cluster_status(self) -> ClusterState:
        """Get cluster status."""
        return ClusterState(
            cluster_id=uuid4(),
            coordinator_id=self.node_id if self.is_coordinator else None,
            nodes=list(self.nodes.values()),
            total_tasks=self.total_tasks_submitted,
            completed_tasks=self.total_tasks_completed,
            failed_tasks=self.total_tasks_failed,
            average_load=sum(n.current_load for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0,
        )

    def get_node_status(self, node_id: UUID) -> Node | None:
        """Get status of specific node."""
        return self.nodes.get(node_id)

    async def start_coordinator(self, interval: int = 30) -> None:
        """Start coordinator duties."""
        self.is_coordinator = True

        while True:
            try:
                await self.check_nodes()

                # Process task queue
                while not self.task_queue.empty():
                    task = await self.task_queue.get()
                    asyncio.create_task(self.distribute_task(task))

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error("Coordinator error", error=str(e))
                await asyncio.sleep(interval)
