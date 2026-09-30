"""
scripts/sample_dataset.py — Sample and prepare the evaluation dataset.

Produces:
  data/processed/eval.json          — full sample
  data/processed/eval_multihop.json — multi-hop subset
  data/processed/eval_singlehop.json — single-hop subset

Also provides a load_or_sample() function used by the API.

Usage:
    python scripts/sample_dataset.py --dataset hotpotqa --size 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic dataset for pipeline testing (not for benchmark reporting)
# ---------------------------------------------------------------------------

SYNTHETIC_SAMPLES = [
    {
        "id": "syn_001",
        "question": "Who directed the film starring Marie Curie that won a Nobel Prize?",
        "answer": "Marie Noëlle",
        "type": "bridge",
        "supporting_facts": [["Marie Curie (film)", 0], ["Nobel Prize in Physics", 0]],
        "context": [
            ["Marie Curie (film)", ["Marie Curie: The Courage of Knowledge is a 2016 biographical film directed by Marie Noëlle."]],
            ["Nobel Prize in Physics", ["The Nobel Prize in Physics is awarded by the Royal Swedish Academy of Sciences."]],
            ["Albert Einstein", ["Albert Einstein developed the theory of relativity."]],
        ],
    },
    {
        "id": "syn_002",
        "question": "What city is the headquarters of the company that makes iPhone?",
        "answer": "Cupertino",
        "type": "bridge",
        "supporting_facts": [["Apple Inc.", 0], ["Cupertino", 0]],
        "context": [
            ["Apple Inc.", ["Apple Inc. is an American multinational technology company headquartered in Cupertino, California."]],
            ["Cupertino", ["Cupertino is a city in Santa Clara County, California."]],
        ],
    },
    {
        "id": "syn_003",
        "question": "In what country was the author of Harry Potter born?",
        "answer": "United Kingdom",
        "type": "bridge",
        "supporting_facts": [["J. K. Rowling", 0]],
        "context": [
            ["J. K. Rowling", ["J. K. Rowling is a British author born in Yate, Gloucestershire, England."]],
            ["Harry Potter", ["Harry Potter is a series of seven fantasy novels written by J. K. Rowling."]],
        ],
    },
]


def _normalize_hotpotqa_sample(raw: dict) -> dict:
    """Normalize a raw HotpotQA record to our internal format."""
    return {
        "id": raw.get("id", ""),
        "question": raw.get("question", ""),
        "answer": raw.get("answer", ""),
        "type": raw.get("type", ""),
        "level": raw.get("level", ""),
        "supporting_facts": raw.get("supporting_facts", []),
        "context": raw.get("context", []),
    }


def _validate_sample(sample: dict) -> list[str]:
    """Return list of validation errors for a sample (empty = valid)."""
    errors: list[str] = []
    if not sample.get("question", "").strip():
        errors.append("missing question")
    if not sample.get("answer", "").strip():
        errors.append("missing answer")
    if not sample.get("supporting_facts"):
        errors.append("missing supporting_facts")
    if not sample.get("context"):
        errors.append("missing context")
    return errors


def sample_hotpotqa(
    raw_path: Path,
    sample_size: int,
    seed: int,
) -> list[dict]:
    """Load raw HotpotQA, validate, and sample."""
    logger.info("Loading raw HotpotQA from %s …", raw_path)
    raw_records = json.loads(raw_path.read_text(encoding="utf-8"))

    valid: list[dict] = []
    invalid_count = 0
    for raw in raw_records:
        sample = _normalize_hotpotqa_sample(raw)
        errs = _validate_sample(sample)
        if errs:
            invalid_count += 1
            logger.debug("Skipping invalid sample %s: %s", sample.get("id"), errs)
        else:
            valid.append(sample)

    logger.info("Valid: %d / Total: %d (invalid: %d)", len(valid), len(raw_records), invalid_count)

    rng = random.Random(seed)
    sampled = rng.sample(valid, min(sample_size, len(valid)))
    return sampled


def load_or_sample(
    dataset_name: str,
    sample_size: int | None = None,
) -> list[dict]:
    """
    Load existing processed samples or trigger sampling.
    Used by the API /ingest endpoint.
    """
    cfg = get_settings()
    size = sample_size or cfg.dataset_sample_size

    if dataset_name == "synthetic":
        return SYNTHETIC_SAMPLES

    # Check for existing processed file
    if cfg.eval_data_path.exists():
        samples = json.loads(cfg.eval_data_path.read_text(encoding="utf-8"))
        if len(samples) >= size:
            return samples[:size]

    # Try to load from raw
    raw_path = cfg.raw_data_dir / f"{dataset_name}_validation.json"
    if not raw_path.exists():
        raw_path = cfg.raw_data_dir / f"{dataset_name}_train.json"

    if not raw_path.exists():
        logger.warning(
            "Raw dataset not found at %s — using synthetic fallback. "
            "Run scripts/download_dataset.py first for real experiments.",
            raw_path,
        )
        return SYNTHETIC_SAMPLES

    return sample_hotpotqa(raw_path, size, cfg.dataset_random_seed)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = get_settings()

    if args.dataset == "synthetic":
        samples = SYNTHETIC_SAMPLES
    else:
        raw_path = cfg.raw_data_dir / f"{args.dataset}_validation.json"
        if not raw_path.exists():
            print(f"Raw file not found: {raw_path}")
            print("Run scripts/download_dataset.py first.")
            sys.exit(1)
        samples = sample_hotpotqa(raw_path, args.size, args.seed)

    # Save full eval set
    cfg.processed_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.eval_data_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved %d samples to %s", len(samples), cfg.eval_data_path)

    # Save split subsets
    from evaluation.metrics import split_by_hop_type

    # Convert to metric-compatible format
    splits = split_by_hop_type(
        [{"gold_supporting_facts": s.get("supporting_facts", [])} for s in samples]
    )
    multihop_samples = [samples[i] for i, s in enumerate(samples)
                        if {"gold_supporting_facts": s.get("supporting_facts", [])} in splits.get("multihop", [])]
    singlehop_samples = [samples[i] for i, s in enumerate(samples)
                         if {"gold_supporting_facts": s.get("supporting_facts", [])} in splits.get("singlehop", [])]

    cfg.multihop_eval_path.write_text(
        json.dumps(multihop_samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cfg.singlehop_eval_path.write_text(
        json.dumps(singlehop_samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nSampled {len(samples)} total samples.")
    print(f"Saved to {cfg.eval_data_path}")


if __name__ == "__main__":
    main()
