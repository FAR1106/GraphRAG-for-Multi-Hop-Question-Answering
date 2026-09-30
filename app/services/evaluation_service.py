"""
app/services/evaluation_service.py — Evaluation orchestration.
Loads evaluation samples, runs all configured methods, records results.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.db.neo4j_client import Neo4jClient
from app.services.rag_service import ask, RAGResponse

logger = logging.getLogger(__name__)


def load_eval_samples(path: Path | None = None) -> list[dict]:
    """Load evaluation samples from the processed data directory."""
    cfg = get_settings()
    eval_path = path or cfg.eval_data_path
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")
    with open(eval_path, encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    client: Neo4jClient,
    systems: list[str],
    eval_path: Path | None = None,
    max_samples: int | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """
    Run evaluation for one or more systems and save results.

    Parameters
    ----------
    systems : list[str]
        e.g. ['vector', 'graphrag']
    eval_path : Path | None
    max_samples : int | None
        If set, limit evaluation to first N samples.
    experiment_id : str | None

    Returns
    -------
    dict with experiment metadata and per-system metrics
    """
    from evaluation.metrics import compute_metrics

    cfg = get_settings()
    exp_id = experiment_id or str(uuid.uuid4())[:8]
    t0 = time.monotonic()

    samples = load_eval_samples(eval_path)
    if max_samples:
        samples = samples[:max_samples]

    logger.info(
        "Evaluating %d samples with systems: %s (exp_id=%s)",
        len(samples), systems, exp_id
    )

    per_system_results: dict[str, list[dict]] = {sys: [] for sys in systems}
    per_system_predictions: dict[str, list[dict]] = {sys: [] for sys in systems}

    for i, sample in enumerate(samples):
        question = sample.get("question", "")
        gold_answer = sample.get("answer", "")
        gold_facts = sample.get("supporting_facts", [])

        for system in systems:
            try:
                response: RAGResponse = ask(
                    client,
                    question,
                    method=system,  # type: ignore[arg-type]
                )
                pred_answer = response.answer
                retrieved_titles = [
                    ch.get("document_title", "") for ch in response.evidence_chunks
                ]
                retrieved_chunk_ids = [
                    ch.get("id", "") for ch in response.evidence_chunks
                ]
            except Exception as exc:
                logger.warning("System %s failed on sample %d: %s", system, i, exc)
                pred_answer = ""
                retrieved_titles = []
                retrieved_chunk_ids = []
                response = None  # type: ignore[assignment]

            # Build prediction record
            pred_record = {
                "sample_id": sample.get("id", f"s{i}"),
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": pred_answer,
                "gold_supporting_facts": gold_facts,
                "retrieved_titles": retrieved_titles,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "latency_s": getattr(response, "total_latency_s", 0.0) if response else 0.0,
                "retrieval_diagnostics": getattr(response, "retrieval_diagnostics", {}) if response else {},
            }
            per_system_predictions[system].append(pred_record)

        if (i + 1) % 10 == 0:
            logger.info("Evaluated %d/%d samples …", i + 1, len(samples))

    # ── Compute metrics ───────────────────────────────────────────────────────
    per_system_metrics: dict[str, dict] = {}
    for system in systems:
        preds = per_system_predictions[system]
        metrics = compute_metrics(preds)
        per_system_metrics[system] = metrics

    total_time = time.monotonic() - t0

    # ── Save results ──────────────────────────────────────────────────────────
    experiment_record = {
        "experiment_id": exp_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "systems": systems,
        "num_samples": len(samples),
        "total_time_s": round(total_time, 2),
        "metrics": per_system_metrics,
        "predictions": per_system_predictions,
        "config": {
            "embedding_model": cfg.embedding_model,
            "llm_model": cfg.llm_model,
            "llm_provider": cfg.llm_provider,
            "vector_top_k": cfg.vector_top_k,
            "graph_max_depth": cfg.graph_max_depth,
            "reranker_enabled": cfg.reranker_enabled,
            "entity_resolution_mode": cfg.entity_resolution_mode,
        },
    }

    results_dir = cfg.results_dir / exp_id
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "experiment.json").write_text(
        json.dumps(experiment_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Also save per-system predictions separately
    for system, preds in per_system_predictions.items():
        (results_dir / f"{system}_predictions.json").write_text(
            json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info("Experiment %s complete. Results saved to %s.", exp_id, results_dir)
    return experiment_record
