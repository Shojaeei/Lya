"""Pinecone Vector Store Adapter for Lya.

Provides cloud-based vector storage using Pinecone API.
Pure Python 3.14+ compatible - uses only urllib.

Note: Pinecone recommends their client library, but this implementation
uses the REST API directly for zero-dependency operation.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from lya.infrastructure.config.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorRecord:
    """A vector record in Pinecone."""
    id: str
    values: list[float]
    metadata: dict[str, Any]
    namespace: str = ""


@dataclass
class SearchResult:
    """Vector search result."""
    id: str
    score: float
    metadata: dict[str, Any]
    namespace: str = ""


class PineconeClient:
    """
    HTTP client for Pinecone REST API.

    Uses only Python standard library (urllib).
    """

    BASE_URL = "https://api.pinecone.io"
    PINECONE_VERSION = "2024-07"

    def __init__(
        self,
        api_key: str,
        environment: str = "us-west1-gcp",
    ) -> None:
        """Initialize Pinecone client.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
        """
        self.api_key = api_key
        self.environment = environment
        self.headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": self.PINECONE_VERSION,
        }
        self._index_host: str | None = None

    def _request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request to Pinecone API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body
            base_url: Override base URL

        Returns:
            Response data
        """
        url = f"{base_url or self.BASE_URL}{endpoint}"

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
            raise PineconeError(
                f"Pinecone API error: {e.code} - {error_body}",
                status_code=e.code,
                response=error_body,
            )
        except Exception as e:
            raise PineconeError(f"Request failed: {e}")

    def list_indexes(self) -> list[dict[str, Any]]:
        """List all indexes.

        Returns:
            List of index information
        """
        response = self._request("GET", "/indexes")
        return response.get("indexes", [])

    def create_index(
        self,
        name: str,
        dimension: int = 1536,
        metric: str = "cosine",
        spec: dict | None = None,
    ) -> dict[str, Any]:
        """Create a new index.

        Args:
            name: Index name
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
            spec: Index specification

        Returns:
            Index information
        """
        spec = spec or {
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1",
            }
        }

        data = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "spec": spec,
        }

        return self._request("POST", "/indexes", data)

    def delete_index(self, name: str) -> None:
        """Delete an index.

        Args:
            name: Index name
        """
        self._request("DELETE", f"/indexes/{name}")

    def describe_index(self, name: str) -> dict[str, Any]:
        """Get index information.

        Args:
            name: Index name

        Returns:
            Index details
        """
        return self._request("GET", f"/indexes/{name}")

    def _get_index_host(self, index_name: str) -> str:
        """Get index host URL.

        Args:
            index_name: Index name

        Returns:
            Index host URL
        """
        if self._index_host is None:
            info = self.describe_index(index_name)
            self._index_host = info.get("host", "")
        return f"https://{self._index_host}"


class PineconeIndex:
    """
    Interface to a specific Pinecone index.

    Provides vector operations (upsert, query, fetch, delete).
    """

    def __init__(
        self,
        client: PineconeClient,
        index_name: str,
        dimension: int = 1536,
    ) -> None:
        """Initialize index interface.

        Args:
            client: Pinecone client
            index_name: Index name
            dimension: Vector dimension
        """
        self.client = client
        self.index_name = index_name
        self.dimension = dimension
        self._host: str | None = None

    def _get_host(self) -> str:
        """Get index host URL."""
        if self._host is None:
            self._host = self.client._get_index_host(self.index_name)
        return self._host

    def upsert(
        self,
        vectors: list[VectorRecord],
        namespace: str = "",
    ) -> dict[str, Any]:
        """Upsert vectors into index.

        Args:
            vectors: Vectors to upsert
            namespace: Namespace

        Returns:
            Upsert response
        """
        data = {
            "vectors": [
                {
                    "id": v.id,
                    "values": v.values,
                    "metadata": v.metadata,
                }
                for v in vectors
            ],
        }

        if namespace:
            data["namespace"] = namespace

        return self.client._request(
            "POST",
            "/vectors/upsert",
            data,
            base_url=self._get_host(),
        )

    def query(
        self,
        vector: list[float] | None = None,
        top_k: int = 10,
        namespace: str = "",
        filter_dict: dict | None = None,
        id: str | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> list[SearchResult]:
        """Query vectors by similarity.

        Args:
            vector: Query vector
            top_k: Number of results
            namespace: Namespace
            filter_dict: Metadata filter
            id: Query by ID instead of vector
            include_metadata: Include metadata in results
            include_values: Include vector values

        Returns:
            List of search results
        """
        data: dict[str, Any] = {
            "topK": top_k,
            "includeMetadata": include_metadata,
            "includeValues": include_values,
        }

        if vector is not None:
            data["vector"] = vector
        elif id is not None:
            data["id"] = id
        else:
            raise ValueError("Either vector or id must be provided")

        if namespace:
            data["namespace"] = namespace

        if filter_dict:
            data["filter"] = filter_dict

        response = self.client._request(
            "POST",
            "/query",
            data,
            base_url=self._get_host(),
        )

        matches = response.get("matches", [])
        return [
            SearchResult(
                id=m.get("id", ""),
                score=m.get("score", 0.0),
                metadata=m.get("metadata", {}),
                namespace=namespace,
            )
            for m in matches
        ]

    def fetch(
        self,
        ids: list[str],
        namespace: str = "",
    ) -> list[VectorRecord]:
        """Fetch vectors by ID.

        Args:
            ids: Vector IDs to fetch
            namespace: Namespace

        Returns:
            List of vector records
        """
        ids_param = ",".join(ids)
        endpoint = f"/vectors/fetch?ids={ids_param}"

        if namespace:
            endpoint += f"&namespace={namespace}"

        response = self.client._request(
            "GET",
            endpoint,
            base_url=self._get_host(),
        )

        vectors = response.get("vectors", {})
        return [
            VectorRecord(
                id=vid,
                values=vdata.get("values", []),
                metadata=vdata.get("metadata", {}),
                namespace=namespace,
            )
            for vid, vdata in vectors.items()
        ]

    def delete(
        self,
        ids: list[str] | None = None,
        namespace: str = "",
        filter_dict: dict | None = None,
        delete_all: bool = False,
    ) -> None:
        """Delete vectors.

        Args:
            ids: IDs to delete
            namespace: Namespace
            filter_dict: Filter for deletion
            delete_all: Delete all vectors
        """
        data: dict[str, Any] = {}

        if delete_all:
            data["deleteAll"] = True
        elif ids:
            data["ids"] = ids
        elif filter_dict:
            data["filter"] = filter_dict
        else:
            raise ValueError("Must provide ids, filter, or delete_all=True")

        if namespace:
            data["namespace"] = namespace

        self.client._request(
            "POST",
            "/vectors/delete",
            data,
            base_url=self._get_host(),
        )

    def describe_index_stats(self) -> dict[str, Any]:
        """Get index statistics.

        Returns:
            Index statistics
        """
        return self.client._request(
            "GET",
            "/describe_index_stats",
            base_url=self._get_host(),
        )

    def list_namespaces(self) -> list[str]:
        """List all namespaces.

        Returns:
            List of namespace names
        """
        stats = self.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        return list(namespaces.keys())


class PineconeRepository:
    """
    Repository implementation for Lya using Pinecone.

    Provides semantic memory storage using Pinecone vector database.
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str = "lya-memory",
        dimension: int = 1536,
        namespace: str = "default",
    ) -> None:
        """Initialize Pinecone repository.

        Args:
            api_key: Pinecone API key (from env if None)
            index_name: Index name
            dimension: Vector dimension
            namespace: Default namespace
        """
        self.api_key = api_key or self._get_api_key_from_env()
        self.index_name = index_name
        self.dimension = dimension
        self.namespace = namespace

        self._client: PineconeClient | None = None
        self._index: PineconeIndex | None = None

    def _get_api_key_from_env(self) -> str:
        """Get API key from environment."""
        import os
        key = os.environ.get("PINECONE_API_KEY")
        if not key:
            raise PineconeError("PINECONE_API_KEY not set")
        return key

    def _ensure_client(self) -> PineconeClient:
        """Ensure client is initialized."""
        if self._client is None:
            self._client = PineconeClient(self.api_key)
        return self._client

    def _ensure_index(self) -> PineconeIndex:
        """Ensure index is initialized."""
        if self._index is None:
            client = self._ensure_client()
            self._index = PineconeIndex(
                client,
                self.index_name,
                self.dimension,
            )
        return self._index

    def create_index_if_not_exists(self) -> None:
        """Create index if it doesn't exist."""
        client = self._ensure_client()

        try:
            client.describe_index(self.index_name)
            logger.info("Index exists", index=self.index_name)
        except PineconeError as e:
            if "not found" in str(e).lower() or e.status_code == 404:
                logger.info("Creating index", index=self.index_name)
                client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                )
            else:
                raise

    def store(
        self,
        id: str,
        vector: list[float],
        metadata: dict[str, Any],
        namespace: str | None = None,
    ) -> None:
        """Store a vector.

        Args:
            id: Vector ID
            vector: Vector values
            metadata: Metadata dictionary
            namespace: Optional namespace override
        """
        index = self._ensure_index()

        record = VectorRecord(
            id=id,
            values=vector,
            metadata=metadata,
            namespace=namespace or self.namespace,
        )

        index.upsert([record], namespace=record.namespace)

    def search(
        self,
        vector: list[float],
        top_k: int = 10,
        namespace: str | None = None,
        filter_dict: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            vector: Query vector
            top_k: Number of results
            namespace: Optional namespace override
            filter_dict: Metadata filter

        Returns:
            Search results
        """
        index = self._ensure_index()

        return index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace or self.namespace,
            filter_dict=filter_dict,
        )

    def get(self, id: str, namespace: str | None = None) -> VectorRecord | None:
        """Get vector by ID.

        Args:
            id: Vector ID
            namespace: Optional namespace

        Returns:
            Vector record or None
        """
        index = self._ensure_index()

        results = index.fetch([id], namespace=namespace or self.namespace)

        return results[0] if results else None

    def delete(self, id: str, namespace: str | None = None) -> None:
        """Delete vector by ID.

        Args:
            id: Vector ID
            namespace: Optional namespace
        """
        index = self._ensure_index()
        index.delete([id], namespace=namespace or self.namespace)

    def get_stats(self) -> dict[str, Any]:
        """Get repository statistics.

        Returns:
            Statistics dictionary
        """
        index = self._ensure_index()
        return index.describe_index_stats()

    def text_to_vector_id(self, text: str) -> str:
        """Generate a deterministic ID from text.

        Args:
            text: Input text

        Returns:
            Hash-based ID
        """
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def store_text(
        self,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> str:
        """Store text with embedding.

        Args:
            text: Original text
            embedding: Vector embedding
            metadata: Additional metadata
            namespace: Optional namespace

        Returns:
            Stored ID
        """
        id = self.text_to_vector_id(text)

        meta = metadata or {}
        meta["text"] = text
        meta["stored_at"] = datetime.now(timezone.utc).isoformat()

        self.store(id, embedding, meta, namespace)
        return id

    def search_by_text(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        namespace: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Search and return text results.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            namespace: Optional namespace

        Returns:
            List of (text, score, metadata) tuples
        """
        results = self.search(query_embedding, top_k, namespace)

        return [
            (
                r.metadata.get("text", ""),
                r.score,
                r.metadata,
            )
            for r in results
        ]


class PineconeError(Exception):
    """Pinecone-specific error."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response: str | None = None,
    ) -> None:
        """Initialize error.

        Args:
            message: Error message
            status_code: HTTP status code
            response: Response body
        """
        super().__init__(message)
        self.status_code = status_code
        self.response = response


# ═══════════════════════════════════════════════════════════════════════
# FALLBACK: Local Memory Repository
# ═══════════════════════════════════════════════════════════════════════

class LocalMemoryRepository:
    """
    Fallback local memory repository using simple vector similarity.

    Used when Pinecone is not available.
    """

    def __init__(self, dimension: int = 1536) -> None:
        """Initialize local repository.

        Args:
            dimension: Vector dimension
        """
        self.dimension = dimension
        self._vectors: dict[str, tuple[list[float], dict]] = {}

    def store(
        self,
        id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Store vector locally."""
        self._vectors[id] = (vector, metadata)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def search(
        self,
        vector: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search by similarity."""
        scored = []

        for id, (vec, meta) in self._vectors.items():
            score = self._cosine_similarity(vector, vec)
            scored.append((id, score, meta))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(
                id=id,
                score=score,
                metadata=meta,
            )
            for id, score, meta in scored[:top_k]
        ]

    def get(self, id: str) -> VectorRecord | None:
        """Get by ID."""
        if id in self._vectors:
            vec, meta = self._vectors[id]
            return VectorRecord(id=id, values=vec, metadata=meta)
        return None

    def delete(self, id: str) -> None:
        """Delete by ID."""
        self._vectors.pop(id, None)

    def get_stats(self) -> dict[str, Any]:
        """Get stats."""
        return {"total_vectors": len(self._vectors)}


# ═══════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example with fallback local repo
    repo = LocalMemoryRepository()

    # Store some vectors
    repo.store("1", [1.0, 0.0, 0.0], {"text": "red"})
    repo.store("2", [0.0, 1.0, 0.0], {"text": "green"})
    repo.store("3", [0.0, 0.0, 1.0], {"text": "blue"})

    # Search
    results = repo.search([0.9, 0.1, 0.0], top_k=2)

    print("Search results:")
    for r in results:
        print(f"  {r.id}: {r.score:.3f} - {r.metadata}")
