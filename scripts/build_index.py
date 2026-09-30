"""
scripts/build_index.py — Build vector index from processed samples.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --dataset hotpotqa --size 100
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.db.neo4j_client import get_neo4j_client
from app.core.ingestion import ingest_samples
from scripts.sample_dataset import load_or_sample

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--size", type=int, default=None)
    args = parser.parse_args()

    cfg = get_settings()
    client = get_neo4j_client()

    try:
        client.connect()
        client.initialize_schema()
        client.create_vector_index()
    except Exception as exc:
        logger.error("Neo4j connection failed: %s", exc)
        sys.exit(1)

    try:
        samples = load_or_sample(args.dataset, args.size)
        logger.info("Loaded %d samples.", len(samples))

        stats = ingest_samples(client, samples, source_dataset=args.dataset)

        print("\n=== Ingestion Complete ===")
        print(f"  Samples      : {stats['num_samples']}")
        print(f"  Chunks       : {stats['num_chunks']}")
        print(f"  Embed time   : {stats['embed_time_s']}s")
        print(f"  Insert time  : {stats['insert_time_s']}s")
        print(f"  Total time   : {stats['total_time_s']}s")

        graph_stats = client.get_stats()
        print(f"\nNeo4j now has:")
        print(f"  Chunks    : {graph_stats['chunk_count']}")
        print(f"  Entities  : {graph_stats['entity_count']}")
        print(f"  Relations : {graph_stats['relationship_count']}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
