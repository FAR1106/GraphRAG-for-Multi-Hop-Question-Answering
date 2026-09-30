"""
app/db/queries.py — Parameterized Cypher queries used across the application.
All queries here accept explicit parameters; no string interpolation of user data.
"""
from __future__ import annotations

from typing import Any

from app.db.neo4j_client import Neo4jClient


# =============================================================================
# Chunk operations
# =============================================================================

def upsert_chunk(client: Neo4jClient, chunk: dict[str, Any]) -> None:
    """Insert or update a Chunk node (idempotent via MERGE on id)."""
    cypher = """
    MERGE (c:Chunk {id: $id})
    SET c.text             = $text,
        c.document_title   = $document_title,
        c.paragraph_index  = $paragraph_index,
        c.source_dataset   = $source_dataset,
        c.embedding        = $embedding
    """
    client.run_write(cypher, chunk)


def batch_upsert_chunks(client: Neo4jClient, chunks: list[dict[str, Any]]) -> None:
    """Batch-upsert chunks using UNWIND for efficiency."""
    cypher = """
    UNWIND $chunks AS row
    MERGE (c:Chunk {id: row.id})
    SET c.text             = row.text,
        c.document_title   = row.document_title,
        c.paragraph_index  = row.paragraph_index,
        c.source_dataset   = row.source_dataset,
        c.embedding        = row.embedding
    """
    client.run_write(cypher, {"chunks": chunks})


def get_chunk_by_id(client: Neo4jClient, chunk_id: str) -> dict[str, Any] | None:
    rows = client.run_query(
        "MATCH (c:Chunk {id: $id}) RETURN c{.*} AS chunk", {"id": chunk_id}
    )
    return rows[0]["chunk"] if rows else None


# =============================================================================
# Entity operations
# =============================================================================

def upsert_entity(client: Neo4jClient, entity: dict[str, Any]) -> None:
    cypher = """
    MERGE (e:Entity {id: $id})
    SET e.name            = $name,
        e.normalized_name = $normalized_name,
        e.entity_type     = $entity_type,
        e.aliases         = $aliases,
        e.description     = $description
    """
    client.run_write(cypher, entity)


def batch_upsert_entities(client: Neo4jClient, entities: list[dict[str, Any]]) -> None:
    cypher = """
    UNWIND $entities AS row
    MERGE (e:Entity {id: row.id})
    SET e.name            = row.name,
        e.normalized_name = row.normalized_name,
        e.entity_type     = row.entity_type,
        e.aliases         = row.aliases,
        e.description     = row.description
    """
    client.run_write(cypher, {"entities": entities})


def get_entity_by_id(client: Neo4jClient, entity_id: str) -> dict[str, Any] | None:
    rows = client.run_query(
        "MATCH (e:Entity {id: $id}) RETURN e{.*} AS entity", {"id": entity_id}
    )
    return rows[0]["entity"] if rows else None


def search_entities_by_name(
    client: Neo4jClient, normalized_name: str, entity_type: str | None = None
) -> list[dict[str, Any]]:
    if entity_type:
        cypher = """
        MATCH (e:Entity)
        WHERE e.normalized_name CONTAINS $name AND e.entity_type = $entity_type
        RETURN e{.*} AS entity LIMIT 20
        """
        rows = client.run_query(cypher, {"name": normalized_name, "entity_type": entity_type})
    else:
        cypher = """
        MATCH (e:Entity)
        WHERE e.normalized_name CONTAINS $name
        RETURN e{.*} AS entity LIMIT 20
        """
        rows = client.run_query(cypher, {"name": normalized_name})
    return [r["entity"] for r in rows]


# =============================================================================
# Relationship operations
# =============================================================================

def upsert_relationship(
    client: Neo4jClient,
    source_id: str,
    target_id: str,
    rel_type: str,
    confidence: float,
    evidence_chunk_id: str,
) -> None:
    cypher = """
    MATCH (src:Entity {id: $src_id})
    MATCH (tgt:Entity {id: $tgt_id})
    MERGE (src)-[r:REL {type: $rel_type, evidence_chunk_id: $evidence_chunk_id}]->(tgt)
    SET r.confidence = $confidence
    """
    client.run_write(
        cypher,
        {
            "src_id": source_id,
            "tgt_id": target_id,
            "rel_type": rel_type,
            "confidence": confidence,
            "evidence_chunk_id": evidence_chunk_id,
        },
    )


def batch_upsert_relationships(
    client: Neo4jClient, relationships: list[dict[str, Any]]
) -> None:
    cypher = """
    UNWIND $rels AS row
    MATCH (src:Entity {id: row.source_id})
    MATCH (tgt:Entity {id: row.target_id})
    MERGE (src)-[r:REL {type: row.rel_type, evidence_chunk_id: row.evidence_chunk_id}]->(tgt)
    SET r.confidence = row.confidence
    """
    client.run_write(cypher, {"rels": relationships})


def upsert_mentions(
    client: Neo4jClient, chunk_id: str, entity_id: str
) -> None:
    cypher = """
    MATCH (c:Chunk {id: $chunk_id})
    MATCH (e:Entity {id: $entity_id})
    MERGE (c)-[:MENTIONS]->(e)
    """
    client.run_write(cypher, {"chunk_id": chunk_id, "entity_id": entity_id})


def batch_upsert_mentions(
    client: Neo4jClient, mentions: list[dict[str, str]]
) -> None:
    """mentions: list of {chunk_id, entity_id}"""
    cypher = """
    UNWIND $mentions AS row
    MATCH (c:Chunk {id: row.chunk_id})
    MATCH (e:Entity {id: row.entity_id})
    MERGE (c)-[:MENTIONS]->(e)
    """
    client.run_write(cypher, {"mentions": mentions})


# =============================================================================
# Graph traversal
# =============================================================================

def get_entity_neighbors(
    client: Neo4jClient, entity_id: str, max_depth: int = 2
) -> list[dict[str, Any]]:
    """Return entities reachable within max_depth hops."""
    cypher = f"""
    MATCH path = (src:Entity {{id: $entity_id}})-[:REL*1..{max_depth}]-(neighbor:Entity)
    WITH neighbor, length(path) AS depth, [r IN relationships(path) | r.type] AS rels
    RETURN DISTINCT neighbor{{.*}} AS entity, depth, rels
    ORDER BY depth
    LIMIT 50
    """
    return client.run_query(cypher, {"entity_id": entity_id})


def get_chunks_for_entities(
    client: Neo4jClient, entity_ids: list[str]
) -> list[dict[str, Any]]:
    """Return chunks that mention any of the given entity IDs."""
    cypher = """
    UNWIND $entity_ids AS eid
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {id: eid})
    RETURN DISTINCT c{.*} AS chunk, e.name AS entity_name, e.id AS entity_id
    """
    return client.run_query(cypher, {"entity_ids": entity_ids})


def get_relationship_triples(
    client: Neo4jClient, entity_ids: list[str]
) -> list[dict[str, Any]]:
    """Return relationship triples among the given entity IDs."""
    cypher = """
    UNWIND $entity_ids AS eid
    MATCH (src:Entity {id: eid})-[r:REL]->(tgt:Entity)
    WHERE tgt.id IN $entity_ids
    RETURN src.name AS source, r.type AS relation, tgt.name AS target,
           r.confidence AS confidence, r.evidence_chunk_id AS evidence_chunk_id
    """
    return client.run_query(cypher, {"entity_ids": entity_ids})


# =============================================================================
# Vector search
# =============================================================================

def vector_search_chunks(
    client: Neo4jClient, query_embedding: list[float], top_k: int = 10
) -> list[dict[str, Any]]:
    """Approximate nearest-neighbour search using the Neo4j vector index."""
    cypher = """
    CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $embedding)
    YIELD node AS c, score
    RETURN c{.*} AS chunk, score
    ORDER BY score DESC
    """
    return client.run_query(cypher, {"top_k": top_k, "embedding": query_embedding})


# =============================================================================
# Graph explorer
# =============================================================================

def get_subgraph_for_entity(
    client: Neo4jClient, entity_id: str, depth: int = 1, limit: int = 30
) -> dict[str, list]:
    """Return nodes and edges for the graph explorer visualization."""
    cypher = f"""
    MATCH path = (src:Entity {{id: $entity_id}})-[:REL*0..{depth}]-(neighbor:Entity)
    WITH nodes(path) AS ns, relationships(path) AS rs
    UNWIND ns AS n
    WITH COLLECT(DISTINCT n{{.*}}) AS nodes_list, rs
    UNWIND rs AS r
    WITH nodes_list,
         COLLECT(DISTINCT {{
             source: startNode(r).id,
             target: endNode(r).id,
             type:   r.type,
             confidence: r.confidence
         }}) AS edges_list
    RETURN nodes_list[0..$limit] AS nodes, edges_list AS edges
    """
    rows = client.run_query(cypher, {"entity_id": entity_id, "limit": limit})
    if not rows:
        return {"nodes": [], "edges": []}
    return {"nodes": rows[0].get("nodes", []), "edges": rows[0].get("edges", [])}
