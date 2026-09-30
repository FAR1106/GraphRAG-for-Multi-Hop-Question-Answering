"""
evaluation/compare.py — Generate comparison tables from experiment results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings


def load_all_experiments() -> list[dict]:
    cfg = get_settings()
    experiments = []
    for exp_dir in cfg.results_dir.iterdir():
        exp_file = exp_dir / "experiment.json"
        if exp_file.exists():
            try:
                experiments.append(json.loads(exp_file.read_text(encoding="utf-8")))
            except Exception:
                pass
    return sorted(experiments, key=lambda x: x.get("timestamp", ""), reverse=True)


def build_comparison_df(experiment: dict) -> pd.DataFrame:
    """Build a comparison DataFrame from a single experiment record."""
    rows = []
    for system, metrics in experiment.get("metrics", {}).items():
        row = {
            "System": system,
            "EM": metrics.get("exact_match", "N/A"),
            "F1": metrics.get("f1", "N/A"),
            "SF Recall": metrics.get("supporting_fact_recall", "N/A"),
            "Median Latency (s)": metrics.get("latency_median_s", "N/A"),
            "P95 Latency (s)": metrics.get("latency_p95_s", "N/A"),
            "Samples": metrics.get("num_samples", "N/A"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    experiments = load_all_experiments()
    if not experiments:
        print("No experiments found. Run evaluation first.")
        sys.exit(0)

    latest = experiments[0]
    print(f"\nLatest experiment: {latest['experiment_id']} ({latest['timestamp']})")
    df = build_comparison_df(latest)
    print(df.to_string(index=False))

    # Save CSV
    cfg = get_settings()
    out_csv = cfg.results_dir / latest["experiment_id"] / "comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved comparison to {out_csv}")
