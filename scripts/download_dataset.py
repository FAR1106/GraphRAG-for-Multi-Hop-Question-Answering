"""
scripts/download_dataset.py — Download and cache HotpotQA (or 2WikiMultiHopQA).

Usage:
    python scripts/download_dataset.py --dataset hotpotqa
    python scripts/download_dataset.py --dataset hotpotqa --split validation
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings

logger = logging.getLogger(__name__)


DATASET_CONFIGS = {
    "hotpotqa": {
        "hf_name": "hotpot_qa",
        "hf_config": "fullwiki",
        "splits": ["train", "validation"],
        "required_fields": ["question", "answer", "supporting_facts", "context"],
    },
    "2wikimultihopqa": {
        "hf_name": "wikimedia/wikipedia",  # placeholder — use official source
        "hf_config": None,
        "splits": ["train", "dev"],
        "required_fields": ["question", "answer", "supporting_facts"],
    },
}


def download_hotpotqa(split: str, raw_dir: Path) -> Path:
    """Download HotpotQA via Hugging Face datasets and save as JSON."""
    from datasets import load_dataset

    logger.info("Downloading HotpotQA (%s) …", split)
    ds = load_dataset("hotpot_qa", "fullwiki", split=split, trust_remote_code=True)

    out_path = raw_dir / f"hotpotqa_{split}.json"
    records = [dict(row) for row in ds]
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved %d records to %s", len(records), out_path)
    return out_path


def download_dataset(dataset_name: str, split: str) -> Path:
    cfg = get_settings()
    raw_dir = cfg.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "hotpotqa":
        return download_hotpotqa(split, raw_dir)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not yet supported for download.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa", choices=list(DATASET_CONFIGS))
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    path = download_dataset(args.dataset, args.split)
    print(f"Download complete: {path}")


if __name__ == "__main__":
    main()
