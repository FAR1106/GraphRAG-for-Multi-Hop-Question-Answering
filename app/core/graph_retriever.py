"""
app/core/graph_retriever.py — Graph-based retrieval via entity matching and traversal.

Steps:
1. Extract entities mentioned in the question.
2. Match them to canonical entities in Neo4j.
3. Traverse the graph up to max_depth hops.
4. Collect supporting chunks for all traversed entities.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.core.entity_resolution import normalize_entity_name
from app.core.extraction import extract_from_chunk
from app.core.vector_retriever import RetrievedChunk
from app.config import get_settings
from app.db.neo4j_client import Neo4jClient
from app.db import queries

logger = logging.getLogger(__name__)


@dataclass
class GraphRetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    traversed_entities: list[dict] = field(default_factory=list)
    matched_question_entities: list[str] = field(default_factory=list)
    relationship_triples: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    query: str = ""
    graph_depth: int = 2


def _extract_question_entity_names(question: str) -> list[str]:
    """
    Lightweight entity extraction from a question string.
    We use the same LLM extraction path but on the question text.
    Falls back to noun phrases if the LLM is unavailable.
    """
    result = extract_from_chunk(chunk_id="question_tmp", text=question)
    names = [e.name for e in result.entities]
    if not names:
        # Heuristic fallback: capitalised tokens
        import re
        tokens = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        names = list(dict.fromkeys(tokens))  # deduplicate preserving order
    return names


def _match_entities_in_graph(
    client: Neo4jClient, names: list[str]
) -> list[dict]:
    """Return graph entity records that match the extracted names."""
    matched: list[dict] = []
    seen_ids: set[str] = set()
    for name in names:
        norm = normalize_entity_name(name)
        rows = queries.search_entities_by_name(client, norm)
        for row in rows:
            if row.get("id") not in seen_ids:
                seen_ids.add(row["id"])
                matched.append(row)
    return matched


def retrieve_by_graph(
    client: Neo4jClient,
    question: str,
    max_depth: int | None = None,
    max_candidates: int | None = None,
) -> GraphRetrievalResult:
    """
    Graph-based retrieval starting from question entities.
    """
    cfg = get_settings()
    depth = max_depth if max_depth is not None else cfg.graph_max_depth
    max_cands = max_candidates or cfg.graph_max_candidates
    t0 = time.monotonic()

    if not client.is_connected():
        from app.db.in_memory_graph import InMemoryGraphStore
        store = InMemoryGraphStore.get_instance()
        chunks, paths = store.multi_hop_reasoning(question, max_depth=depth)
        return GraphRetrievalResult(
            chunks=chunks[:max_cands],
            matched_question_entities=[],
            relationship_triples=paths,
            traversed_entities=[],
            latency_s=round(time.monotonic() - t0, 4),
            query=question,
            graph_depth=depth,
        )

    # Step 1: extract entity names from question
    q_entity_names = _extract_question_entity_names(question)
    logger.debug("Question entities: %s", q_entity_names)

    # Step 2: match to graph
    matched_entities = _match_entities_in_graph(client, q_entity_names)
    matched_ids = [e["id"] for e in matched_entities]
    logger.debug("Matched %d entities in graph.", len(matched_ids))

    if not matched_ids:
        return GraphRetrievalResult(
            query=question,
            matched_question_entities=q_entity_names,
            latency_s=round(time.monotonic() - t0, 4),
            graph_depth=depth,
        )

    # Step 3: traverse
    all_entity_ids: set[str] = set(matched_ids)
    traversed_entities: list[dict] = list(matched_entities)

    for eid in matched_ids:
        neighbors = queries.get_entity_neighbors(client, eid, max_depth=depth)
        for row in neighbors:
            neighbor = row.get("entity", {})
            nid = neighbor.get("id", "")
            if nid and nid not in all_entity_ids:
                all_entity_ids.add(nid)
                traversed_entities.append(neighbor)
            if len(all_entity_ids) >= max_cands:
                break

    # Step 4: collect chunks for traversed entities
    entity_id_list = list(all_entity_ids)[:max_cands]
    chunk_rows = queries.get_chunks_for_entities(client, entity_id_list)

    seen_chunk_ids: set[str] = set()
    graph_chunks: list[RetrievedChunk] = []
    for row in chunk_rows:
        chunk_data = row.get("chunk", {})
        cid = chunk_data.get("id", "")
        if not cid or cid in seen_chunk_ids:
            continue
        seen_chunk_ids.add(cid)
        graph_chunks.append(
            RetrievedChunk(
                id=cid,
                text=chunk_data.get("text", ""),
                document_title=chunk_data.get("document_title", ""),
                paragraph_index=chunk_data.get("paragraph_index", 0),
                source_dataset=chunk_data.get("source_dataset", ""),
                score=0.5,  # graph-derived chunks get a fixed prior score
                retrieval_source="graph",
                entity_name=row.get("entity_name", ""),
            )
        )

    # Step 5: collect relationship triples
    triples = queries.get_relationship_triples(client, entity_id_list)

    return GraphRetrievalResult(
        chunks=graph_chunks,
        traversed_entities=traversed_entities,
        matched_question_entities=q_entity_names,
        relationship_triples=triples,
        latency_s=round(time.monotonic() - t0, 4),
        query=question,
        graph_depth=depth,
    )
