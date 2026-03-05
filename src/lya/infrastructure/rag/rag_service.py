"""RAG Service — pure Python 3.14.

Retrieval-Augmented Generation: ingest documents, embed chunks,
and retrieve relevant content for LLM context augmentation.

Storage is JSON-based per chat at workspace/rag/{chat_id}/.
Uses hash-based embeddings by default, real embeddings if available.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from lya.infrastructure.config.logging import get_logger
from lya.infrastructure.rag.chunker import TextChunker
from lya.infrastructure.rag.document_loader import DocumentLoader

logger = get_logger(__name__)


def _hash_embedding(text: str, dimension: int = 128) -> list[float]:
    """Generate a deterministic hash-based embedding."""
    text = text.lower().strip()
    vectors: list[float] = []
    for seed in range(0, dimension, 32):
        h = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
        vectors.extend(b / 255.0 for b in h)
    vectors = vectors[:dimension]
    norm = math.sqrt(sum(x * x for x in vectors))
    if norm > 0:
        vectors = [x / norm for x in vectors]
    return vectors


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_score(query: str, content: str) -> float:
    """Simple keyword overlap score."""
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    if not query_words:
        return 0.0
    overlap = query_words & content_words
    return len(overlap) / len(query_words)


class RAGService:
    """Document-based retrieval-augmented generation service."""

    def __init__(
        self,
        workspace: Path,
        embedding_service: Any | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self._workspace = workspace / "rag"
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._embedding_service = embedding_service
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._chunker = TextChunker()

    def _db_path(self, chat_id: int) -> Path:
        """Get document DB path for a chat."""
        p = self._workspace / str(chat_id)
        p.mkdir(parents=True, exist_ok=True)
        return p / "documents.json"

    def _load_docs(self, chat_id: int) -> list[dict[str, Any]]:
        """Load stored document chunks for a chat."""
        db = self._db_path(chat_id)
        if db.exists():
            try:
                data = json.loads(db.read_text(encoding="utf-8"))
                return data.get("chunks", [])
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_docs(self, chat_id: int, chunks: list[dict[str, Any]]) -> None:
        """Save document chunks."""
        db = self._db_path(chat_id)
        data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(chunks),
            "chunks": chunks,
        }
        tmp = db.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        tmp.replace(db)

    async def _embed(self, text: str) -> list[float]:
        """Generate embedding using real service or hash fallback."""
        if self._embedding_service:
            try:
                return await self._embedding_service.embed(text)
            except Exception:
                pass
        return _hash_embedding(text)

    async def ingest_document(
        self,
        file_path: Path,
        chat_id: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ingest a document: load → chunk → embed → store.

        Returns the number of chunks created.
        """
        # Load content
        content = await DocumentLoader.load(file_path)
        if not content.strip():
            return 0

        # Chunk
        chunks = self._chunker.chunk(
            content,
            chunk_size=self._chunk_size,
            overlap=self._chunk_overlap,
        )

        if not chunks:
            return 0

        # Embed and store
        existing = self._load_docs(chat_id)
        doc_id = str(uuid4())[:8]
        meta = metadata or {}
        meta["doc_id"] = doc_id
        meta["filename"] = meta.get("filename", file_path.name)

        for i, chunk_text in enumerate(chunks):
            embedding = await self._embed(chunk_text)
            existing.append({
                "id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "content": chunk_text,
                "embedding": embedding,
                "metadata": meta,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

        self._save_docs(chat_id, existing)
        logger.info(
            "document_ingested",
            chat_id=chat_id,
            filename=meta.get("filename"),
            chunks=len(chunks),
        )
        return len(chunks)

    async def ingest_text(
        self,
        text: str,
        chat_id: int,
        source: str = "text",
    ) -> int:
        """Ingest raw text content."""
        chunks = self._chunker.chunk(
            text, chunk_size=self._chunk_size, overlap=self._chunk_overlap,
        )
        if not chunks:
            return 0

        existing = self._load_docs(chat_id)
        doc_id = str(uuid4())[:8]

        for i, chunk_text in enumerate(chunks):
            embedding = await self._embed(chunk_text)
            existing.append({
                "id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "content": chunk_text,
                "embedding": embedding,
                "metadata": {"source": source, "doc_id": doc_id},
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

        self._save_docs(chat_id, existing)
        return len(chunks)

    async def query(
        self,
        question: str,
        chat_id: int,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query stored documents for relevant chunks.

        Returns list of {content, similarity, metadata} dicts.
        """
        chunks = self._load_docs(chat_id)
        if not chunks:
            return []

        query_embedding = await self._embed(question)

        scored: list[tuple[dict[str, Any], float]] = []
        for chunk in chunks:
            emb = chunk.get("embedding", [])
            sim = _cosine_similarity(query_embedding, emb) if emb else 0.0
            kw = _keyword_score(question, chunk["content"])
            combined = 0.4 * sim + 0.5 * kw + 0.1
            if combined > 0.15:
                scored.append((chunk, combined))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in scored[:top_k]:
            results.append({
                "content": chunk["content"],
                "similarity": round(score, 3),
                "metadata": chunk.get("metadata", {}),
            })

        logger.debug(
            "rag_query",
            question=question[:50],
            results=len(results),
            chat_id=chat_id,
        )
        return results

    async def list_documents(self, chat_id: int) -> list[dict[str, Any]]:
        """List ingested documents for a chat."""
        chunks = self._load_docs(chat_id)
        docs: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            doc_id = chunk.get("doc_id") or chunk.get("metadata", {}).get("doc_id", "unknown")
            if doc_id not in docs:
                meta = chunk.get("metadata", {})
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "chunks": 0,
                    "created_at": chunk.get("created_at", ""),
                }
            docs[doc_id]["chunks"] += 1
        return list(docs.values())

    async def clear_documents(self, chat_id: int) -> int:
        """Clear all documents for a chat."""
        chunks = self._load_docs(chat_id)
        count = len(chunks)
        self._save_docs(chat_id, [])
        return count
