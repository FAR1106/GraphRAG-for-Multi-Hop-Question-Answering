"""
app/core/hybrid_retriever.py — Merge vector + graph retrieval candidates.

Merges results from vector search and graph traversal, deduplicates by chunk ID,
and records retrieval provenance for each chunk.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.core.vector_retriever import RetrievedChunk, retrieve_by_vector
from app.core.graph_retriever import retrieve_by_graph, GraphRetrievalResult
from app.core.reranker import rerank_chunks, RerankResult
from app.config import get_settings
from app.db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class HybridRetrievalResult:
    final_chunks: list[RetrievedChunk] = field(default_factory=list)
    vector_chunks: list[RetrievedChunk] = field(default_factory=list)
    graph_chunks: list[RetrievedChunk] = field(default_factory=list)
    merged_candidate_count: int = 0
    vector_candidate_count: int = 0
    graph_candidate_count: int = 0
    matched_question_entities: list[str] = field(default_factory=list)
    relationship_triples: list[dict] = field(default_factory=list)
    traversed_entities: list[dict] = field(default_factory=list)
    reranked: bool = False
    vector_latency_s: float = 0.0
    graph_latency_s: float = 0.0
    rerank_latency_s: float = 0.0
    total_latency_s: float = 0.0
    query: str = ""


def retrieve_hybrid(
    client: Neo4jClient,
    question: str,
    vector_top_k: int | None = None,
    graph_max_depth: int | None = None,
    graph_max_candidates: int | None = None,
    final_context_size: int | None = None,
) -> HybridRetrievalResult:
    """
    Run hybrid retrieval:
    1. Vector ANN search.
    2. Graph traversal from question entities.
    3. Merge and deduplicate candidates.
    4. Cross-encoder reranking.
    """
    cfg = get_settings()
    t0 = time.monotonic()

    # ── Vector retrieval ──────────────────────────────────────────────────────
    vec_result = retrieve_by_vector(client, question, top_k=vector_top_k)
    vector_chunks = vec_result.chunks
    vector_latency = vec_result.latency_s

    # ── Graph retrieval ───────────────────────────────────────────────────────
    graph_result: GraphRetrievalResult = retrieve_by_graph(
        client,
        question,
        max_depth=graph_max_depth,
        max_candidates=graph_max_candidates,
    )
    graph_chunks = graph_result.chunks
    graph_latency = graph_result.latency_s

    # ── Merge and deduplicate ─────────────────────────────────────────────────
    seen_ids: dict[str, RetrievedChunk] = {}

    for ch in vector_chunks:
        seen_ids[ch.id] = ch

    for ch in graph_chunks:
        if ch.id in seen_ids:
            # Mark as retrieved by both
            seen_ids[ch.id].retrieval_source = "both"
        else:
            seen_ids[ch.id] = ch

    merged = list(seen_ids.values())
    merged_count = len(merged)

    # ── Reranking ─────────────────────────────────────────────────────────────
    final_n = final_context_size or cfg.final_context_size
    rerank_result: RerankResult = rerank_chunks(question, merged, top_n=final_n)

    total_latency = time.monotonic() - t0

    return HybridRetrievalResult(
        final_chunks=rerank_result.chunks,
        vector_chunks=vector_chunks,
        graph_chunks=graph_chunks,
        merged_candidate_count=merged_count,
        vector_candidate_count=len(vector_chunks),
        graph_candidate_count=len(graph_chunks),
        matched_question_entities=graph_result.matched_question_entities,
        relationship_triples=graph_result.relationship_triples,
        traversed_entities=graph_result.traversed_entities,
        reranked=rerank_result.reranked,
        vector_latency_s=vector_latency,
        graph_latency_s=graph_latency,
        rerank_latency_s=rerank_result.latency_s,
        total_latency_s=round(total_latency, 4),
        query=question,
    )
