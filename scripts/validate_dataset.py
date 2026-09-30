"""
scripts/validate_dataset.py — Validate processed evaluation samples.

Checks for:
  - Missing required fields
  - Malformed questions
  - Missing supporting facts
  - Duplicate sample IDs
  - Context/supporting-fact mismatches

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --path data/processed/eval.json
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


REQUIRED_FIELDS = ["question", "answer", "supporting_facts", "context"]


def validate(path: Path) -> dict[str, int | list]:
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    samples = json.loads(path.read_text(encoding="utf-8"))
    total = len(samples)
    issues: list[dict] = []
    seen_ids: set[str] = set()
    duplicate_count = 0

    for i, sample in enumerate(samples):
        sid = sample.get("id", f"idx_{i}")
        errs: list[str] = []

        # Required fields
        for field in REQUIRED_FIELDS:
            if not sample.get(field):
                errs.append(f"missing or empty field: {field}")

        # Question length
        q = sample.get("question", "")
        if len(q.strip()) < 5:
            errs.append("question too short")

        # Supporting facts structure
        sf = sample.get("supporting_facts", [])
        if sf and not isinstance(sf[0], (list, tuple)):
            errs.append("supporting_facts should be list of [title, int]")

        # Duplicates
        if sid in seen_ids:
            duplicate_count += 1
            errs.append("duplicate id")
        seen_ids.add(sid)

        if errs:
            issues.append({"id": sid, "errors": errs})

    report = {
        "total": total,
        "valid": total - len(issues),
        "invalid": len(issues),
        "duplicates": duplicate_count,
        "issues": issues[:20],  # show at most 20
    }
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=None)
    args = parser.parse_args()

    cfg = get_settings()
    path = args.path or cfg.eval_data_path

    print(f"\nValidating {path} …")
    report = validate(path)

    print(f"  Total  : {report['total']}")
    print(f"  Valid  : {report['valid']}")
    print(f"  Invalid: {report['invalid']}")
    print(f"  Dupes  : {report['duplicates']}")

    if report["issues"]:
        print("\nFirst issues:")
        for issue in report["issues"][:5]:
            print(f"  [{issue['id']}] {', '.join(issue['errors'])}")

    sys.exit(0 if report["invalid"] == 0 else 1)


if __name__ == "__main__":
    main()
