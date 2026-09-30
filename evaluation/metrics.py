"""
evaluation/metrics.py — Evaluation metrics for multi-hop QA.

Implements:
  - Exact Match (EM) with normalization
  - Token-level F1
  - Supporting-fact recall
  - Latency statistics (median, p95)
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

import numpy as np


# =============================================================================
# Text normalization (HotpotQA official conventions)
# =============================================================================

def normalize_answer(s: str) -> str:
    """
    Normalize a string for fair EM/F1 comparison.
    Follows the HotpotQA evaluation script conventions.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# =============================================================================
# Per-question metrics
# =============================================================================

def exact_match(prediction: str, ground_truth: str) -> int:
    """Return 1 if prediction matches ground truth after normalization."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> dict[str, float]:
    """
    Compute token-level precision, recall and F1.

    Returns dict with keys 'precision', 'recall', 'f1'.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def supporting_fact_recall(
    gold_facts: list[list],
    retrieved_titles: list[str],
) -> float:
    """
    Compute the fraction of gold supporting facts whose document was retrieved.

    gold_facts: list of [title, sentence_index] pairs (HotpotQA format).
    retrieved_titles: list of document titles in the retrieved context.
    """
    if not gold_facts:
        return 1.0  # vacuously true

    gold_titles = {fact[0] for fact in gold_facts if fact}
    retrieved_set = set(retrieved_titles)
    matched = gold_titles & retrieved_set
    return len(matched) / len(gold_titles)


# =============================================================================
# Aggregate metrics
# =============================================================================

def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate metrics over a list of prediction records.

    Each prediction record must have:
      - 'gold_answer' : str
      - 'predicted_answer' : str
      - 'gold_supporting_facts' : list of [title, sent_idx]
      - 'retrieved_titles' : list of str
      - 'latency_s' : float

    Returns a dict of scalar metrics.
    """
    if not predictions:
        return {"error": "No predictions provided"}

    em_scores: list[int] = []
    f1_scores: list[float] = []
    sf_recall_scores: list[float] = []
    latencies: list[float] = []

    for pred in predictions:
        gold = pred.get("gold_answer", "")
        predicted = pred.get("predicted_answer", "")
        gold_facts = pred.get("gold_supporting_facts", [])
        retrieved_titles = pred.get("retrieved_titles", [])
        latency = pred.get("latency_s", 0.0)

        em_scores.append(exact_match(predicted, gold))
        f1 = token_f1(predicted, gold)
        f1_scores.append(f1["f1"])
        sf_recall_scores.append(supporting_fact_recall(gold_facts, retrieved_titles))
        latencies.append(latency)

    n = len(predictions)
    lat_arr = np.array(latencies)

    return {
        "num_samples": n,
        "exact_match": round(float(np.mean(em_scores)), 4),
        "f1": round(float(np.mean(f1_scores)), 4),
        "supporting_fact_recall": round(float(np.mean(sf_recall_scores)), 4),
        "latency_median_s": round(float(np.median(lat_arr)), 4),
        "latency_p95_s": round(float(np.percentile(lat_arr, 95)), 4) if n >= 5 else None,
        "latency_mean_s": round(float(np.mean(lat_arr)), 4),
    }


def split_by_hop_type(predictions: list[dict]) -> dict[str, list[dict]]:
    """
    Split predictions into multi-hop and single-hop subsets.
    Uses 'num_supporting_facts' field if available, else infers from gold facts length.
    """
    multihop: list[dict] = []
    singlehop: list[dict] = []

    for pred in predictions:
        facts = pred.get("gold_supporting_facts", [])
        # Single-hop: one unique source document
        unique_docs = len({f[0] for f in facts if f})
        if unique_docs <= 1:
            singlehop.append(pred)
        else:
            multihop.append(pred)

    return {"multihop": multihop, "singlehop": singlehop}
