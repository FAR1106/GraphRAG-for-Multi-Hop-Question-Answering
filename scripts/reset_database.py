"""
scripts/reset_database.py — Explicitly reset (wipe) the Neo4j development database.

CAUTION: This permanently deletes all data. Requires explicit --confirm flag.

Usage:
    python scripts/reset_database.py --confirm
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true", help="Confirm database reset.")
    args = parser.parse_args()

    if not args.confirm:
        print("ERROR: Pass --confirm to reset the database. This is irreversible.")
        sys.exit(1)

    client = get_neo4j_client()
    try:
        client.connect()
        client.reset_database()
        print("Database reset complete.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
