"""
app/core/reranker.py — Cross-encoder reranking of candidate chunks.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.core.embeddings import rerank
from app.core.vector_retriever import RetrievedChunk
from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    latency_s: float = 0.0
    reranked: bool = False


def rerank_chunks(
    query: str,
    candidates: list[RetrievedChunk],
    top_n: int | None = None,
) -> RerankResult:
    """
    Rerank candidate chunks using the cross-encoder.

    Parameters
    ----------
    query : str
        The original question.
    candidates : list[RetrievedChunk]
        Pool of candidate chunks from vector and/or graph retrieval.
    top_n : int | None
        If set, return only the top N results.

    Returns
    -------
    RerankResult
        Reranked and (optionally) trimmed list of chunks.
    """
    cfg = get_settings()
    t0 = time.monotonic()

    if not candidates:
        return RerankResult(chunks=[], latency_s=0.0, reranked=False)

    final_n = top_n or cfg.final_context_size

    if not cfg.reranker_enabled:
        # No reranking — sort by original score descending
        sorted_chunks = sorted(candidates, key=lambda c: c.score, reverse=True)
        return RerankResult(
            chunks=sorted_chunks[:final_n],
            latency_s=round(time.monotonic() - t0, 4),
            reranked=False,
        )

    passages = [c.text for c in candidates]
    ranked = rerank(query, passages, top_n=final_n)

    result_chunks: list[RetrievedChunk] = []
    for orig_idx, score in ranked:
        chunk = candidates[orig_idx]
        chunk.score = float(score)
        result_chunks.append(chunk)

    return RerankResult(
        chunks=result_chunks,
        latency_s=round(time.monotonic() - t0, 4),
        reranked=True,
    )
