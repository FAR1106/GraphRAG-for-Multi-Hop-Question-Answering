"""
app/core/vector_retriever.py — Conventional vector-only RAG retrieval.
Uses the Neo4j vector index for ANN search.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.core.embeddings import embed_query
from app.config import get_settings
from app.db.neo4j_client import Neo4jClient
from app.db import queries

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with provenance and retrieval metadata."""
    id: str
    text: str
    document_title: str
    paragraph_index: int
    source_dataset: str
    score: float
    retrieval_source: str  # "vector" | "graph" | "both"
    entity_name: str = ""


@dataclass
class VectorRetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    latency_s: float = 0.0
    query: str = ""
    top_k: int = 10


def retrieve_by_vector(
    client: Neo4jClient,
    question: str,
    top_k: int | None = None,
) -> VectorRetrievalResult:
    """
    Embed the question and retrieve top-k chunks by cosine similarity.
    """
    cfg = get_settings()
    k = top_k or cfg.vector_top_k
    t0 = time.monotonic()

    if not client.is_connected():
        from app.db.in_memory_graph import InMemoryGraphStore
        store = InMemoryGraphStore.get_instance()
        chunks = store.search_vector(question, top_k=k)
        return VectorRetrievalResult(
            chunks=chunks,
            latency_s=round(time.monotonic() - t0, 4),
            query=question,
            top_k=k,
        )

    q_emb = embed_query(question)
    rows = queries.vector_search_chunks(client, q_emb, top_k=k)

    chunks: list[RetrievedChunk] = []
    for row in rows:
        chunk_data = row.get("chunk", {})
        chunks.append(
            RetrievedChunk(
                id=chunk_data.get("id", ""),
                text=chunk_data.get("text", ""),
                document_title=chunk_data.get("document_title", ""),
                paragraph_index=chunk_data.get("paragraph_index", 0),
                source_dataset=chunk_data.get("source_dataset", ""),
                score=float(row.get("score", 0.0)),
                retrieval_source="vector",
            )
        )

    return VectorRetrievalResult(
        chunks=chunks,
        latency_s=round(time.monotonic() - t0, 4),
        query=question,
        top_k=k,
    )
