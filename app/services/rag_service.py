"""
app/services/rag_service.py — High-level RAG service orchestrating retrieval + answer generation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from app.core.vector_retriever import retrieve_by_vector, RetrievedChunk
from app.core.hybrid_retriever import retrieve_hybrid, HybridRetrievalResult
from app.core.answer_generator import generate_answer, AnswerResult
from app.core.question_decomposer import decompose_question
from app.config import get_settings
from app.db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

Method = Literal["vector", "graphrag", "decomposed"]


@dataclass
class RAGResponse:
    question: str
    method: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    evidence_chunks: list[dict] = field(default_factory=list)
    relationship_triples: list[dict] = field(default_factory=list)
    retrieval_diagnostics: dict = field(default_factory=dict)
    total_latency_s: float = 0.0
    answer_latency_s: float = 0.0
    retrieval_latency_s: float = 0.0


def ask(
    client: Neo4jClient,
    question: str,
    method: Method = "graphrag",
    vector_top_k: int | None = None,
    graph_max_depth: int | None = None,
    final_context_size: int | None = None,
) -> RAGResponse:
    """
    Main entry point for the RAG service.

    Parameters
    ----------
    client : Neo4jClient
    question : str
    method : 'vector' | 'graphrag' | 'decomposed'
    """
    cfg = get_settings()
    t0 = time.monotonic()

    if method == "vector":
        return _ask_vector(client, question, vector_top_k, final_context_size, t0)
    elif method in ("graphrag", "hybrid"):
        return _ask_graphrag(client, question, vector_top_k, graph_max_depth, final_context_size, t0)
    elif method == "graph":
        return _ask_graph(client, question, graph_max_depth, final_context_size, t0)
    elif method == "decomposed":
        return _ask_decomposed(client, question, vector_top_k, graph_max_depth, final_context_size, t0)
    else:
        return _ask_graphrag(client, question, vector_top_k, graph_max_depth, final_context_size, t0)


def _ask_vector(
    client, question, vector_top_k, final_context_size, t0
) -> RAGResponse:
    cfg = get_settings()
    k = vector_top_k or cfg.vector_top_k
    ctx = final_context_size or cfg.final_context_size

    vec_result = retrieve_by_vector(client, question, top_k=k)
    chunks = vec_result.chunks[:ctx]
    retrieval_latency = vec_result.latency_s

    answer_result = generate_answer(question, chunks, relationship_triples=[])
    total_latency = time.monotonic() - t0

    return RAGResponse(
        question=question,
        method="vector",
        answer=answer_result.answer,
        citations=answer_result.citations,
        evidence_chunks=[_chunk_to_dict(c) for c in chunks],
        relationship_triples=[],
        retrieval_diagnostics={
            "vector_candidates": len(vec_result.chunks),
            "final_context": len(chunks),
        },
        total_latency_s=round(total_latency, 4),
        answer_latency_s=answer_result.latency_s,
        retrieval_latency_s=retrieval_latency,
    )


def _ask_graph(
    client, question, graph_max_depth, final_context_size, t0
) -> RAGResponse:
    from app.core.graph_retriever import retrieve_by_graph
    cfg = get_settings()
    depth = graph_max_depth or cfg.graph_max_depth
    ctx = final_context_size or cfg.final_context_size

    graph_result = retrieve_by_graph(client, question, max_depth=depth)
    chunks = graph_result.chunks[:ctx]
    retrieval_latency = graph_result.latency_s

    answer_result = generate_answer(question, chunks, relationship_triples=graph_result.relationship_triples)
    total_latency = time.monotonic() - t0

    return RAGResponse(
        question=question,
        method="graph",
        answer=answer_result.answer,
        citations=answer_result.citations,
        evidence_chunks=[_chunk_to_dict(c) for c in chunks],
        relationship_triples=graph_result.relationship_triples,
        retrieval_diagnostics={
            "graph_candidates": len(graph_result.chunks),
            "final_context": len(chunks),
        },
        total_latency_s=round(total_latency, 4),
        answer_latency_s=answer_result.latency_s,
        retrieval_latency_s=retrieval_latency,
    )


def _ask_graphrag(
    client, question, vector_top_k, graph_max_depth, final_context_size, t0
) -> RAGResponse:
    result: HybridRetrievalResult = retrieve_hybrid(
        client,
        question,
        vector_top_k=vector_top_k,
        graph_max_depth=graph_max_depth,
        final_context_size=final_context_size,
    )

    answer_result = generate_answer(
        question, result.final_chunks, relationship_triples=result.relationship_triples
    )
    total_latency = time.monotonic() - t0

    return RAGResponse(
        question=question,
        method="graphrag",
        answer=answer_result.answer,
        citations=answer_result.citations,
        evidence_chunks=[_chunk_to_dict(c) for c in result.final_chunks],
        relationship_triples=result.relationship_triples,
        retrieval_diagnostics={
            "vector_candidates": result.vector_candidate_count,
            "graph_candidates": result.graph_candidate_count,
            "merged_candidates": result.merged_candidate_count,
            "final_context": len(result.final_chunks),
            "matched_entities": result.matched_question_entities,
            "reranked": result.reranked,
            "vector_latency_s": result.vector_latency_s,
            "graph_latency_s": result.graph_latency_s,
            "rerank_latency_s": result.rerank_latency_s,
        },
        total_latency_s=round(total_latency, 4),
        answer_latency_s=answer_result.latency_s,
        retrieval_latency_s=result.total_latency_s,
    )


def _ask_decomposed(
    client, question, vector_top_k, graph_max_depth, final_context_size, t0
) -> RAGResponse:
    """Decompose the question, retrieve for each sub-question, accumulate evidence."""
    decomposed = decompose_question(question)
    all_chunks: list[RetrievedChunk] = []
    all_triples: list[dict] = []

    for sq in decomposed.sub_questions:
        result: HybridRetrievalResult = retrieve_hybrid(
            client,
            sq.question,
            vector_top_k=vector_top_k,
            graph_max_depth=graph_max_depth,
        )
        all_chunks.extend(result.final_chunks)
        all_triples.extend(result.relationship_triples)

    # Deduplicate by chunk id, keep first occurrence
    seen: dict[str, RetrievedChunk] = {}
    for ch in all_chunks:
        if ch.id not in seen:
            seen[ch.id] = ch
    unique_chunks = list(seen.values())

    cfg = get_settings()
    ctx = final_context_size or cfg.final_context_size
    final_chunks = unique_chunks[:ctx]

    answer_result = generate_answer(question, final_chunks, relationship_triples=all_triples)
    total_latency = time.monotonic() - t0

    return RAGResponse(
        question=question,
        method="decomposed",
        answer=answer_result.answer,
        citations=answer_result.citations,
        evidence_chunks=[_chunk_to_dict(c) for c in final_chunks],
        relationship_triples=all_triples[:20],
        retrieval_diagnostics={
            "sub_questions": [sq.question for sq in decomposed.sub_questions],
            "final_context": len(final_chunks),
        },
        total_latency_s=round(total_latency, 4),
        answer_latency_s=answer_result.latency_s,
        retrieval_latency_s=0.0,
    )


def _chunk_to_dict(chunk: RetrievedChunk) -> dict:
    return {
        "id": chunk.id,
        "text": chunk.text,
        "document_title": chunk.document_title,
        "paragraph_index": chunk.paragraph_index,
        "score": round(chunk.score, 4),
        "retrieval_source": chunk.retrieval_source,
    }
