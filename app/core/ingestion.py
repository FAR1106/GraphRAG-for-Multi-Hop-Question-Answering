"""
app/core/ingestion.py — Document ingestion pipeline.
Embeds chunks and writes them to Neo4j with their embeddings.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.core.chunking import Chunk, chunk_dataset_samples
from app.core.embeddings import embed_texts
from app.config import get_settings
from app.db.neo4j_client import Neo4jClient
from app.db import queries

logger = logging.getLogger(__name__)


def ingest_samples(
    client: Neo4jClient,
    samples: list[dict],
    source_dataset: str = "hotpotqa",
    batch_size: int = 64,
) -> dict[str, Any]:
    """
    Full ingestion pipeline:
    1. Chunk documents.
    2. Embed chunks.
    3. Upsert chunks into Neo4j with embeddings.

    Returns ingestion statistics.
    """
    t0 = time.monotonic()
    cfg = get_settings()

    # ── Chunking ──────────────────────────────────────────────────────────────
    logger.info("Chunking %d samples …", len(samples))
    t_chunk = time.monotonic()
    chunks: list[Chunk] = chunk_dataset_samples(samples, source_dataset)
    chunk_time = time.monotonic() - t_chunk
    logger.info("Produced %d unique chunks in %.1fs.", len(chunks), chunk_time)

    # ── Embedding ─────────────────────────────────────────────────────────────
    logger.info("Embedding %d chunks (model=%s) …", len(chunks), cfg.embedding_model)
    t_embed = time.monotonic()
    texts = [ch.normalized_text for ch in chunks]
    embeddings: list[list[float]] = embed_texts(texts, batch_size=batch_size)
    embed_time = time.monotonic() - t_embed
    logger.info("Embedding done in %.1fs.", embed_time)

    # ── Build Neo4j payloads ──────────────────────────────────────────────────
    chunk_dicts: list[dict] = []
    for ch, emb in zip(chunks, embeddings):
        d = ch.to_dict()
        d["embedding"] = emb
        chunk_dicts.append(d)

    # ── Insert to Neo4j ───────────────────────────────────────────────────────
    logger.info("Writing %d chunks to Neo4j …", len(chunk_dicts))
    t_insert = time.monotonic()
    for i in range(0, len(chunk_dicts), batch_size):
        batch = chunk_dicts[i: i + batch_size]
        queries.batch_upsert_chunks(client, batch)
    insert_time = time.monotonic() - t_insert
    logger.info("Neo4j write done in %.1fs.", insert_time)

    total_time = time.monotonic() - t0
    return {
        "num_samples": len(samples),
        "num_chunks": len(chunks),
        "chunk_time_s": round(chunk_time, 2),
        "embed_time_s": round(embed_time, 2),
        "insert_time_s": round(insert_time, 2),
        "total_time_s": round(total_time, 2),
    }


def get_all_chunk_dicts(
    client: Neo4jClient,
    limit: int = 10_000,
) -> list[dict]:
    """Retrieve all chunk records from Neo4j (for downstream use)."""
    cypher = "MATCH (c:Chunk) RETURN c{.*} AS chunk LIMIT $limit"
    rows = client.run_query(cypher, {"limit": limit})
    return [r["chunk"] for r in rows]
