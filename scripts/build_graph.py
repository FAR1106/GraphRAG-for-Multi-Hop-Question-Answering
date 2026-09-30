"""
scripts/build_graph.py — Extract entities/relationships and build the knowledge graph.

Usage:
    python scripts/build_graph.py
    python scripts/build_graph.py --mock   (use mock LLM for pipeline testing)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.db.neo4j_client import get_neo4j_client
from app.core.ingestion import get_all_chunk_dicts
from app.core.extraction import extract_from_chunks
from app.core.graph_builder import build_graph

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use mock LLM extraction.")
    args = parser.parse_args()

    if args.mock:
        os.environ["LLM_PROVIDER"] = "mock"
        logger.warning(
            "Using MOCK extraction — results are not suitable for research experiments."
        )

    cfg = get_settings()
    client = get_neo4j_client()

    try:
        client.connect()
    except Exception as exc:
        logger.error("Neo4j connection failed: %s", exc)
        sys.exit(1)

    try:
        chunk_dicts = get_all_chunk_dicts(client)
        if not chunk_dicts:
            logger.error("No chunks found in Neo4j. Run scripts/build_index.py first.")
            sys.exit(1)

        logger.info("Extracting from %d chunks …", len(chunk_dicts))
        results = extract_from_chunks(
            chunk_dicts,
            progress_callback=lambda i, n: logger.info("%d/%d chunks extracted …", i, n) if i % 50 == 0 else None,
        )

        total_entities = sum(len(r.entities) for r in results)
        total_rels = sum(len(r.relationships) for r in results)
        cached = sum(1 for r in results if r.from_cache)
        logger.info(
            "Extraction complete: %d entities, %d relationships, %d from cache.",
            total_entities, total_rels, cached
        )

        stats = build_graph(client, results)

        print("\n=== Graph Build Complete ===")
        print(f"  Raw entities      : {stats['raw_entities']}")
        print(f"  Resolved entities : {stats['resolved_entities']}")
        print(f"  Relationships     : {stats['inserted_relationships']}")
        print(f"  Mentions          : {stats['inserted_mentions']}")
        print(f"  Build time        : {stats['graph_build_time_s']}s")
    finally:
        client.close()


if __name__ == "__main__":
    main()
