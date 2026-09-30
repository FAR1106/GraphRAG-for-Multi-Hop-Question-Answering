"""
app/core/question_decomposer.py — Optional question decomposition extension.

Decomposes complex multi-hop questions into ordered sub-questions,
retrieves evidence for each, and chains answers to guide subsequent retrieval.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from app.core.extraction import _call_llm
from app.config import get_settings

logger = logging.getLogger(__name__)


class SubQuestion(BaseModel):
    """A single decomposed sub-question."""
    index: int
    question: str = Field(..., min_length=3)
    depends_on: list[int] = Field(default_factory=list)


class DecomposedQuestion(BaseModel):
    """Result of question decomposition."""
    original: str
    sub_questions: list[SubQuestion]


def decompose_question(question: str) -> DecomposedQuestion:
    """
    Ask the LLM to decompose a complex question into ordered sub-questions.
    Returns a validated DecomposedQuestion.
    Falls back to the original question as a single sub-question on failure.
    """
    cfg = get_settings()
    if not cfg.question_decomposition_enabled:
        return DecomposedQuestion(
            original=question,
            sub_questions=[SubQuestion(index=0, question=question)],
        )

    system = (
        "You decompose complex questions into simpler, ordered sub-questions. "
        "Respond with valid JSON only:\n"
        '{"sub_questions": [{"index": 0, "question": "...", "depends_on": []}]}'
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}"},
    ]

    try:
        raw = _call_llm(messages)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:]).rstrip("`").strip()
        data = json.loads(raw)
        sub_qs = [SubQuestion.model_validate(sq) for sq in data.get("sub_questions", [])]
        if not sub_qs:
            raise ValueError("Empty sub-question list")
        return DecomposedQuestion(original=question, sub_questions=sub_qs)
    except Exception as exc:
        logger.warning("Question decomposition failed (%s); using original.", exc)
        return DecomposedQuestion(
            original=question,
            sub_questions=[SubQuestion(index=0, question=question)],
        )
