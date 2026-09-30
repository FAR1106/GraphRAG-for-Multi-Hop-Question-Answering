"""
app/core/graph_builder.py — Build the Neo4j knowledge graph from extraction results.

Idempotent: uses MERGE on stable IDs so re-running never creates duplicates.
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from app.core.entity_resolution import normalize_entity_name, resolve_entities
from app.core.extraction import ExtractionResult
from app.config import get_settings
from app.db.neo4j_client import Neo4jClient
from app.db import queries

logger = logging.getLogger(__name__)


def _make_entity_id(name: str, entity_type: str) -> str:
    raw = f"{entity_type}::{normalize_entity_name(name)}"
    return "ent_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Build graph from extraction results
# ---------------------------------------------------------------------------

def build_graph(
    client: Neo4jClient,
    extraction_results: list[ExtractionResult],
    batch_size: int = 100,
) -> dict[str, Any]:
    """
    Insert entities, relationships and chunk↔entity mentions into Neo4j.

    Returns a dict with indexing statistics.
    """
    cfg = get_settings()
    t0 = time.monotonic()
    stats: dict[str, Any] = {
        "raw_entities": 0,
        "raw_relationships": 0,
        "resolved_entities": 0,
        "inserted_entities": 0,
        "inserted_relationships": 0,
        "inserted_mentions": 0,
        "entity_resolution_accepted": 0,
        "entity_resolution_rejected": 0,
    }

    # ── Collect all raw entities across all chunks ────────────────────────────
    raw_entities_by_name: dict[str, dict] = {}
    for result in extraction_results:
        for ent in result.entities:
            eid = _make_entity_id(ent.name, ent.entity_type)
            if eid not in raw_entities_by_name:
                raw_entities_by_name[eid] = {
                    "id": eid,
                    "name": ent.name,
                    "normalized_name": ent.normalized_name,
                    "entity_type": ent.entity_type,
                    "aliases": [],
                    "description": ent.description,
                }
            stats["raw_entities"] += 1

    # ── Entity resolution ─────────────────────────────────────────────────────
    raw_entity_list = list(raw_entities_by_name.values())
    resolved_entities, id_mapping, res_stats = resolve_entities(
        raw_entity_list,
        mode=cfg.entity_resolution_mode,
        fuzzy_threshold=cfg.entity_fuzzy_threshold * 100,  # 0-1 → 0-100
        embedding_threshold=cfg.entity_embedding_threshold,
    )
    stats["resolved_entities"] = len(resolved_entities)
    stats["entity_resolution_accepted"] = res_stats.accepted_merges
    stats["entity_resolution_rejected"] = res_stats.rejected_merges

    # ── Insert entities in batches ────────────────────────────────────────────
    for i in range(0, len(resolved_entities), batch_size):
        batch = resolved_entities[i: i + batch_size]
        queries.batch_upsert_entities(client, batch)
    stats["inserted_entities"] = len(resolved_entities)

    # ── Collect relationships and mentions ────────────────────────────────────
    all_relationships: list[dict] = []
    all_mentions: list[dict] = []

    for result in extraction_results:
        chunk_id = result.chunk_id

        # Mentions: which canonical entities appear in this chunk?
        chunk_entity_ids: set[str] = set()
        for ent in result.entities:
            raw_eid = _make_entity_id(ent.name, ent.entity_type)
            canonical_eid = id_mapping.get(raw_eid, raw_eid)
            if canonical_eid not in chunk_entity_ids:
                chunk_entity_ids.add(canonical_eid)
                all_mentions.append({"chunk_id": chunk_id, "entity_id": canonical_eid})

        # Relationships
        for rel in result.relationships:
            src_raw = _make_entity_id(rel.source_entity, "OTHER")
            # Try to find the type from extracted entities
            src_canonical = id_mapping.get(src_raw, src_raw)
            tgt_raw = _make_entity_id(rel.target_entity, "OTHER")
            tgt_canonical = id_mapping.get(tgt_raw, tgt_raw)

            all_relationships.append({
                "source_id": src_canonical,
                "target_id": tgt_canonical,
                "rel_type": rel.relation_type,
                "confidence": rel.confidence,
                "evidence_chunk_id": chunk_id,
            })
            stats["raw_relationships"] += 1

    # ── Insert mentions in batches ────────────────────────────────────────────
    for i in range(0, len(all_mentions), batch_size):
        batch = all_mentions[i: i + batch_size]
        queries.batch_upsert_mentions(client, batch)
    stats["inserted_mentions"] = len(all_mentions)

    # ── Insert relationships ──────────────────────────────────────────────────
    for i in range(0, len(all_relationships), batch_size):
        batch = all_relationships[i: i + batch_size]
        queries.batch_upsert_relationships(client, batch)
    stats["inserted_relationships"] = len(all_relationships)

    stats["graph_build_time_s"] = round(time.monotonic() - t0, 2)
    logger.info("Graph build complete: %s", stats)
    return stats
