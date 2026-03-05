"""Neo4j Graph Memory Adapter for Lya.

Provides graph-based knowledge storage using Neo4j.
Pure Python 3.14+ compatible - uses REST API.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any
from collections.abc import Sequence


class RelationshipType(Enum):
    """Types of relationships in the knowledge graph."""
    KNOWS = "KNOWS"
    RELATES_TO = "RELATES_TO"
    PART_OF = "PART_OF"
    HAS_ATTRIBUTE = "HAS_ATTRIBUTE"
    DEPENDS_ON = "DEPENDS_ON"
    CAUSES = "CAUSES"
    LOCATED_IN = "LOCATED_IN"
    CREATED_BY = "CREATED_BY"
    USED_FOR = "USED_FOR"
    INSTANCE_OF = "INSTANCE_OF"


@dataclass
class Node:
    """Graph node."""
    id: str
    labels: list[str]
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Relationship:
    """Graph relationship."""
    id: str
    type: RelationshipType
    start_node_id: str
    end_node_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Path:
    """Graph path between nodes."""
    nodes: list[Node]
    relationships: list[Relationship]
    length: int = 0


class Neo4jClient:
    """
    HTTP client for Neo4j REST API.

    Uses Neo4j HTTP API for Cypher queries.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        """Initialize Neo4j client.

        Args:
            uri: Neo4j URI (e.g., http://localhost:7474)
            user: Username
            password: Password
            database: Database name
        """
        self.uri = uri.rstrip("/")
        self.user = user
        self.password = password
        self.database = database

        # Create auth header
        import base64
        credentials = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request to Neo4j.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body

        Returns:
            Response data
        """
        url = f"{self.uri}{endpoint}"

        try:
            req = urllib.request.Request(
                url,
                headers=self.headers,
                method=method,
            )

            if data is not None:
                req.data = json.dumps(data).encode("utf-8")

            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode())

        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            raise Neo4jError(f"Neo4j error: {e.code} - {error_body}")
        except Exception as e:
            raise Neo4jError(f"Request failed: {e}")

    def execute_cypher(
        self,
        query: str,
        parameters: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Execute Cypher query.

        Args:
            query: Cypher query
            parameters: Query parameters

        Returns:
            Query results
        """
        data = {
            "statements": [{
                "statement": query,
                "parameters": parameters or {},
            }],
        }

        response = self._request(
            "POST",
            f"/db/{self.database}/tx/commit",
            data,
        )

        results = response.get("results", [])
        errors = response.get("errors", [])

        if errors:
            raise Neo4jError(f"Cypher errors: {errors}")

        return results

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            self.execute_cypher("RETURN 1 AS test")
            return True
        except Exception:
            return False


class GraphMemoryRepository:
    """
    Graph-based memory repository using Neo4j.

    Stores entities, concepts, and relationships.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
    ) -> None:
        """Initialize repository.

        Args:
            uri: Neo4j URI
            user: Username
            password: Password
            database: Database name
        """
        # Get from environment if not provided
        import os

        self.uri = uri or os.environ.get("NEO4J_URI", "http://localhost:7474")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "")
        self.database = database

        self._client: Neo4jClient | None = None

    def _ensure_client(self) -> Neo4jClient:
        """Ensure client is initialized."""
        if self._client is None:
            self._client = Neo4jClient(
                self.uri,
                self.user,
                self.password,
                self.database,
            )
        return self._client

    def create_node(
        self,
        labels: list[str],
        properties: dict[str, Any],
        node_id: str | None = None,
    ) -> Node:
        """Create a node.

        Args:
            labels: Node labels
            properties: Node properties
            node_id: Optional custom ID

        Returns:
            Created node
        """
        client = self._ensure_client()

        node_id = node_id or self._generate_id()
        properties["id"] = node_id
        properties["created_at"] = datetime.now(timezone.utc).isoformat()

        labels_str = ":".join(labels)

        query = f"""
        CREATE (n:{labels_str} $properties)
        RETURN n
        """

        client.execute_cypher(query, {"properties": properties})

        return Node(
            id=node_id,
            labels=labels,
            properties=properties,
        )

    def create_relationship(
        self,
        start_node_id: str,
        end_node_id: str,
        rel_type: RelationshipType,
        properties: dict[str, Any] | None = None,
        rel_id: str | None = None,
    ) -> Relationship:
        """Create a relationship between nodes.

        Args:
            start_node_id: Start node ID
            end_node_id: End node ID
            rel_type: Relationship type
            properties: Relationship properties
            rel_id: Optional custom ID

        Returns:
            Created relationship
        """
        client = self._ensure_client()

        rel_id = rel_id or self._generate_id()
        properties = properties or {}
        properties["id"] = rel_id
        properties["created_at"] = datetime.now(timezone.utc).isoformat()

        query = f"""
        MATCH (a {{id: $start_id}}), (b {{id: $end_id}})
        CREATE (a)-[r:{rel_type.value} $properties]->(b)
        RETURN r
        """

        client.execute_cypher(query, {
            "start_id": start_node_id,
            "end_id": end_node_id,
            "properties": properties,
        })

        return Relationship(
            id=rel_id,
            type=rel_type,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            properties=properties,
        )

    def get_node(self, node_id: str) -> Node | None:
        """Get node by ID.

        Args:
            node_id: Node ID

        Returns:
            Node or None
        """
        client = self._ensure_client()

        query = """
        MATCH (n {id: $id})
        RETURN n
        """

        results = client.execute_cypher(query, {"id": node_id})

        if not results or not results[0].get("data"):
            return None

        data = results[0]["data"][0]["row"][0]

        return Node(
            id=data.get("id", node_id),
            labels=data.get("labels", []),
            properties=data.get("properties", {}),
        )

    def find_nodes(
        self,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[Node]:
        """Find nodes matching criteria.

        Args:
            labels: Filter by labels
            properties: Filter by properties
            limit: Maximum results

        Returns:
            List of matching nodes
        """
        client = self._ensure_client()

        where_clauses = []
        params: dict[str, Any] = {}

        if properties:
            for key, value in properties.items():
                where_clauses.append(f"n.{key} = ${key}")
                params[key] = value

        labels_str = ":".join(labels) if labels else ""
        where_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        MATCH (n:{labels_str}) {where_str}
        RETURN n
        LIMIT {limit}
        """

        results = client.execute_cypher(query, params)

        nodes = []
        for result in results:
            for row in result.get("data", []):
                data = row["row"][0]
                nodes.append(Node(
                    id=data.get("id", ""),
                    labels=data.get("labels", []),
                    properties=data,
                ))

        return nodes

    def get_neighbors(
        self,
        node_id: str,
        rel_type: RelationshipType | None = None,
        direction: str = "both",
    ) -> list[tuple[Node, Relationship, str]]:
        """Get neighboring nodes.

        Args:
            node_id: Center node ID
            rel_type: Filter by relationship type
            direction: "in", "out", or "both"

        Returns:
            List of (node, relationship, direction) tuples
        """
        client = self._ensure_client()

        rel_filter = f":{rel_type.value}" if rel_type else ""

        if direction == "out":
            query = f"""
            MATCH (n {{id: $id}})-[r{rel_filter}]->(m)
            RETURN n, r, m
            """
        elif direction == "in":
            query = f"""
            MATCH (n {{id: $id}})<-[r{rel_filter}]-(m)
            RETURN n, r, m
            """
        else:
            query = f"""
            MATCH (n {{id: $id}})-[r{rel_filter}]-(m)
            RETURN n, r, m
            """

        results = client.execute_cypher(query, {"id": node_id})

        neighbors = []
        for result in results:
            for row in result.get("data", []):
                center_data = row["row"][0]
                rel_data = row["row"][1]
                neighbor_data = row["row"][2]

                rel = Relationship(
                    id=rel_data.get("id", ""),
                    type=RelationshipType(rel_data.get("type", "RELATES_TO")),
                    start_node_id=center_data.get("id", ""),
                    end_node_id=neighbor_data.get("id", ""),
                    properties=rel_data,
                )

                node = Node(
                    id=neighbor_data.get("id", ""),
                    labels=neighbor_data.get("labels", []),
                    properties=neighbor_data,
                )

                neighbors.append((node, rel, "out"))

        return neighbors

    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
    ) -> Path | None:
        """Find shortest path between nodes.

        Args:
            start_id: Start node ID
            end_id: End node ID
            max_depth: Maximum path length

        Returns:
            Path or None
        """
        client = self._ensure_client()

        query = """
        MATCH path = shortestPath(
            (a {id: $start_id})-[:*1..{max_depth}]-(b {id: $end_id})
        )
        RETURN path
        """.format(max_depth=max_depth)

        results = client.execute_cypher(query, {
            "start_id": start_id,
            "end_id": end_id,
        })

        if not results or not results[0].get("data"):
            return None

        # Parse path
        # This is simplified - real implementation would parse Neo4j path format
        return Path(nodes=[], relationships=[], length=0)

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its relationships.

        Args:
            node_id: Node ID

        Returns:
            True if deleted
        """
        client = self._ensure_client()

        query = """
        MATCH (n {id: $id})
        DETACH DELETE n
        """

        client.execute_cypher(query, {"id": node_id})
        return True

    def delete_relationship(self, rel_id: str) -> bool:
        """Delete a relationship.

        Args:
            rel_id: Relationship ID

        Returns:
            True if deleted
        """
        client = self._ensure_client()

        query = """
        MATCH ()-[r {id: $id}]-()
        DELETE r
        """

        client.execute_cypher(query, {"id": rel_id})
        return True

    def store_knowledge(
        self,
        subject: str,
        predicate: str,
        object: str,
        properties: dict[str, Any] | None = None,
    ) -> tuple[Node, Relationship, Node]:
        """Store knowledge as subject-predicate-object triple.

        Args:
            subject: Subject entity
            predicate: Relationship type
            object: Object entity
            properties: Additional properties

        Returns:
            Tuple of (subject_node, relationship, object_node)
        """
        # Create nodes
        sub_node = self.create_node(
            labels=["Entity"],
            properties={"name": subject, "type": "subject"},
        )

        obj_node = self.create_node(
            labels=["Entity"],
            properties={"name": object, "type": "object"},
        )

        # Map predicate to relationship type
        rel_type = self._map_predicate_to_relationship(predicate)

        # Create relationship
        rel = self.create_relationship(
            start_node_id=sub_node.id,
            end_node_id=obj_node.id,
            rel_type=rel_type,
            properties={**properties, "predicate": predicate},
        )

        return sub_node, rel, obj_node

    def query_knowledge(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query knowledge graph.

        Args:
            subject: Subject filter
            predicate: Predicate filter
            object: Object filter

        Returns:
            List of matching triples
        """
        client = self._ensure_client()

        where_clauses = []
        params: dict[str, Any] = {}

        if subject:
            where_clauses.append("s.name = $subject")
            params["subject"] = subject

        if object:
            where_clauses.append("o.name = $object")
            params["object"] = object

        if predicate:
            where_clauses.append("r.predicate = $predicate")
            params["predicate"] = predicate

        where_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
        MATCH (s:Entity)-[r]->(o:Entity)
        {where_str}
        RETURN s.name as subject, r.predicate as predicate, o.name as object, r as properties
        """

        results = client.execute_cypher(query, params)

        triples = []
        for result in results:
            for row in result.get("data", []):
                triples.append({
                    "subject": row["row"][0],
                    "predicate": row["row"][1],
                    "object": row["row"][2],
                    "properties": row["row"][3] if len(row["row"]) > 3 else {},
                })

        return triples

    def get_stats(self) -> dict[str, Any]:
        """Get repository statistics."""
        client = self._ensure_client()

        # Node count
        node_result = client.execute_cypher("MATCH (n) RETURN count(n) as count")
        node_count = node_result[0]["data"][0]["row"][0] if node_result else 0

        # Relationship count
        rel_result = client.execute_cypher("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = rel_result[0]["data"][0]["row"][0] if rel_result else 0

        return {
            "nodes": node_count,
            "relationships": rel_count,
            "database": self.database,
            "connected": client.test_connection(),
        }

    def _generate_id(self) -> str:
        """Generate unique ID."""
        import uuid
        return str(uuid.uuid4())

    def _map_predicate_to_relationship(self, predicate: str) -> RelationshipType:
        """Map natural language predicate to relationship type."""
        mapping = {
            "knows": RelationshipType.KNOWS,
            "is": RelationshipType.INSTANCE_OF,
            "has": RelationshipType.HAS_ATTRIBUTE,
            "part of": RelationshipType.PART_OF,
            "causes": RelationshipType.CAUSES,
            "in": RelationshipType.LOCATED_IN,
            "by": RelationshipType.CREATED_BY,
            "for": RelationshipType.USED_FOR,
        }

        return mapping.get(predicate.lower(), RelationshipType.RELATES_TO)


class InMemoryGraphRepository:
    """
    In-memory graph repository (fallback when Neo4j unavailable).

    Uses simple adjacency list representation.
    """

    def __init__(self) -> None:
        """Initialize in-memory repository."""
        self._nodes: dict[str, Node] = {}
        self._relationships: dict[str, Relationship] = {}
        self._adjacency: dict[str, list[tuple[str, str]]] = {}  # node_id -> [(rel_id, target_id)]

    def create_node(
        self,
        labels: list[str],
        properties: dict[str, Any],
        node_id: str | None = None,
    ) -> Node:
        """Create node in memory."""
        import uuid

        node_id = node_id or str(uuid.uuid4())
        properties["id"] = node_id

        node = Node(
            id=node_id,
            labels=labels,
            properties=properties,
        )

        self._nodes[node_id] = node
        self._adjacency[node_id] = []

        return node

    def create_relationship(
        self,
        start_node_id: str,
        end_node_id: str,
        rel_type: RelationshipType,
        properties: dict[str, Any] | None = None,
        rel_id: str | None = None,
    ) -> Relationship:
        """Create relationship in memory."""
        import uuid

        rel_id = rel_id or str(uuid.uuid4())

        rel = Relationship(
            id=rel_id,
            type=rel_type,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            properties=properties or {},
        )

        self._relationships[rel_id] = rel
        self._adjacency[start_node_id].append((rel_id, end_node_id))

        return rel

    def get_node(self, node_id: str) -> Node | None:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def find_nodes(
        self,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[Node]:
        """Find nodes."""
        results = []

        for node in self._nodes.values():
            if labels and not any(l in node.labels for l in labels):
                continue

            if properties:
                match = all(
                    node.properties.get(k) == v
                    for k, v in properties.items()
                )
                if not match:
                    continue

            results.append(node)

            if len(results) >= limit:
                break

        return results

    def get_neighbors(
        self,
        node_id: str,
        rel_type: RelationshipType | None = None,
        direction: str = "both",
    ) -> list[tuple[Node, Relationship, str]]:
        """Get neighbors."""
        neighbors = []

        for rel_id, target_id in self._adjacency.get(node_id, []):
            rel = self._relationships.get(rel_id)
            node = self._nodes.get(target_id)

            if rel and node:
                if rel_type is None or rel.type == rel_type:
                    neighbors.append((node, rel, "out"))

        return neighbors

    def query_knowledge(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query knowledge."""
        results = []

        for rel in self._relationships.values():
            sub = self._nodes.get(rel.start_node_id)
            obj = self._nodes.get(rel.end_node_id)

            if not sub or not obj:
                continue

            if subject and sub.properties.get("name") != subject:
                continue

            if object and obj.properties.get("name") != object:
                continue

            if predicate and rel.properties.get("predicate") != predicate:
                continue

            results.append({
                "subject": sub.properties.get("name", ""),
                "predicate": rel.properties.get("predicate", ""),
                "object": obj.properties.get("name", ""),
                "properties": rel.properties,
            })

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get stats."""
        return {
            "nodes": len(self._nodes),
            "relationships": len(self._relationships),
            "database": "in_memory",
            "connected": True,
        }


class Neo4jError(Exception):
    """Neo4j-specific error."""
    pass


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Use in-memory version for example
    repo = InMemoryGraphRepository()

    # Create nodes
    person1 = repo.create_node(
        labels=["Person"],
        properties={"name": "Alice", "age": 30},
    )

    person2 = repo.create_node(
        labels=["Person"],
        properties={"name": "Bob", "age": 25},
    )

    # Create relationship
    rel = repo.create_relationship(
        start_node_id=person1.id,
        end_node_id=person2.id,
        rel_type=RelationshipType.KNOWS,
        properties={"since": "2020"},
    )

    print(f"Created: {person1.properties['name']} -{rel.type.value}-> {person2.properties['name']}")

    # Store knowledge
    sub, r, obj = repo.store_knowledge(
        subject="Python",
        predicate="is",
        object="Programming Language",
        properties={"popularity": "high"},
    )

    # Query
    results = repo.query_knowledge(subject="Python")
    print(f"\nQuery results: {results}")

    # Stats
    print(f"\nStats: {repo.get_stats()}")
