"""
evaluation/run.py — CLI entry point for reproducible experiments.

Usage:
    python -m evaluation.run --systems vector,graphrag --max-samples 50
    python -m evaluation.run --dataset data/processed/eval.json --systems graphrag
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.db.neo4j_client import get_neo4j_client
from app.services.evaluation_service import run_evaluation

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GraphRAG evaluation experiment."
    )
    parser.add_argument(
        "--systems",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=["vector", "graphrag"],
        help="Comma-separated list of systems: vector,graphrag,decomposed",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the evaluation JSON file (default: data/processed/eval.json).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Custom experiment ID (default: auto-generated).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_settings()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    client = get_neo4j_client()
    try:
        client.connect()
    except Exception as exc:
        logger.error("Cannot connect to Neo4j: %s", exc)
        sys.exit(1)

    try:
        result = run_evaluation(
            client,
            systems=args.systems,
            eval_path=args.dataset,
            max_samples=args.max_samples,
            experiment_id=args.experiment_id,
        )
    finally:
        client.close()

    # Print summary to stdout
    print("\n" + "=" * 60)
    print(f"Experiment ID : {result['experiment_id']}")
    print(f"Samples       : {result['num_samples']}")
    print(f"Total time    : {result['total_time_s']:.1f}s")
    print()
    for system, metrics in result["metrics"].items():
        print(f"[{system.upper()}]")
        print(f"  EM                    : {metrics.get('exact_match', 'N/A')}")
        print(f"  F1                    : {metrics.get('f1', 'N/A')}")
        print(f"  SF Recall             : {metrics.get('supporting_fact_recall', 'N/A')}")
        print(f"  Latency (median)      : {metrics.get('latency_median_s', 'N/A')}s")
        print(f"  Latency (p95)         : {metrics.get('latency_p95_s', 'N/A')}s")
        print()
    print("=" * 60)


if __name__ == "__main__":
    main()
