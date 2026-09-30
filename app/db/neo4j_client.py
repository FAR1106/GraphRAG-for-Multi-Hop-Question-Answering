"""
app/db/neo4j_client.py — Thread-safe Neo4j driver with connection pooling.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

from neo4j import GraphDatabase, Driver, Session, exceptions as neo4j_exc

from app.config import get_settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Singleton-style Neo4j client that manages the driver lifecycle."""

    _driver: Driver | None = None
    _connected: bool = False

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the Neo4j driver (call once at startup)."""
        cfg = self._settings
        try:
            self._driver = GraphDatabase.driver(
                cfg.neo4j_uri,
                auth=(cfg.neo4j_username, cfg.neo4j_password),
                max_connection_pool_size=50,
                connection_timeout=2.0,
            )
            self._driver.verify_connectivity()
            self._connected = True
            logger.info("Connected to Neo4j at %s", cfg.neo4j_uri)
        except Exception as exc:
            self.close()
            self._connected = False
            logger.error("Neo4j unavailable at %s: %s", cfg.neo4j_uri, exc)
            raise

    def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None
        self._connected = False
        logger.info("Neo4j driver closed.")

    def is_connected(self) -> bool:
        """Return True if driver is active and connected."""
        return bool(self._driver and self._connected)

    # ── Session helpers ───────────────────────────────────────────────────────

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
        cfg = self._settings
        with self._driver.session(database=cfg.neo4j_database) as session:
            yield session

    def run_query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return all records as dicts."""
        with self.session() as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

    def run_write(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Execute a write Cypher statement inside an explicit write transaction."""
        with self.session() as session:
            session.execute_write(lambda tx: tx.run(cypher, parameters or {}))

    # ── Schema bootstrap ──────────────────────────────────────────────────────

    def initialize_schema(self) -> None:
        """Create constraints and indexes required by the GraphRAG schema."""
        constraints = [
            # Chunk
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            # Entity
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            # Indexes
            "CREATE INDEX chunk_doc_title IF NOT EXISTS FOR (c:Chunk) ON (c.document_title)",
            "CREATE INDEX entity_norm_name IF NOT EXISTS FOR (e:Entity) ON (e.normalized_name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
        ]
        with self.session() as session:
            for stmt in constraints:
                try:
                    session.run(stmt)
                except neo4j_exc.ClientError as exc:
                    # Already exists — safe to ignore
                    logger.debug("Schema stmt skipped (already exists): %s", exc)
        logger.info("Neo4j schema initialized.")

    def create_vector_index(self) -> None:
        """Create the chunk embedding vector index (idempotent)."""
        cfg = self._settings
        cypher = f"""
        CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {cfg.embedding_dimension},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        with self.session() as session:
            try:
                session.run(cypher)
                logger.info(
                    "Vector index created/confirmed (dim=%d).", cfg.embedding_dimension
                )
            except neo4j_exc.ClientError as exc:
                logger.warning("Vector index creation: %s", exc)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, int]:
        """Return basic graph statistics."""
        queries = {
            "chunk_count": "MATCH (c:Chunk) RETURN count(c) AS n",
            "entity_count": "MATCH (e:Entity) RETURN count(e) AS n",
            "relationship_count": "MATCH ()-[r:REL]->() RETURN count(r) AS n",
            "mentions_count": "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS n",
        }
        stats: dict[str, int] = {}
        for key, cypher in queries.items():
            try:
                rows = self.run_query(cypher)
                stats[key] = rows[0]["n"] if rows else 0
            except Exception as exc:
                logger.warning("Stats query failed for %s: %s", key, exc)
                stats[key] = -1
        return stats

    def reset_database(self) -> None:
        """
        Delete all nodes, relationships and indexes.
        CAUTION: irreversible. Only call via the reset_database.py script.
        """
        logger.warning("Resetting Neo4j database — all data will be deleted.")
        with self.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Database reset complete.")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_client: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    """Return the module-level Neo4j client singleton."""
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client
