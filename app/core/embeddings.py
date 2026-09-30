"""
app/core/embeddings.py — Embedding generation and management.

Uses sentence-transformers with lazy loading and caching.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from app.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_bi_encoder() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    cfg = get_settings()
    logger.info("Loading embedding model: %s", cfg.embedding_model)
    model = SentenceTransformer(cfg.embedding_model)
    logger.info("Embedding model loaded.")
    return model


@lru_cache(maxsize=1)
def _load_cross_encoder() -> "SentenceTransformer":
    """Load the cross-encoder model for reranking."""
    from sentence_transformers import CrossEncoder

    cfg = get_settings()
    logger.info("Loading cross-encoder: %s", cfg.reranker_model)
    model = CrossEncoder(cfg.reranker_model)
    logger.info("Cross-encoder loaded.")
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Embed a list of texts.

    Returns a list of float lists (JSON-serializable, Neo4j-compatible).
    """
    if not texts:
        return []

    model = _load_bi_encoder()
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 200,
        normalize_embeddings=True,  # cosine via dot product
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]


def rerank(
    query: str,
    passages: list[str],
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """
    Rerank passages against a query using the cross-encoder.

    Returns a list of (original_index, score) tuples sorted descending by score.
    """
    cfg = get_settings()
    if not cfg.reranker_enabled or not passages:
        # Identity ranking
        return [(i, 0.0) for i in range(len(passages))]

    model = _load_cross_encoder()
    pairs = [(query, p) for p in passages]
    scores: list[float] = model.predict(pairs).tolist()

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))
