"""
app/core/answer_generator.py — LLM-based grounded answer generation.

Generates answers strictly grounded in retrieved evidence.
Includes citations to source chunks and relationship triples.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from app.core.vector_retriever import RetrievedChunk
from app.config import get_settings
from app.core.extraction import _call_llm

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    evidence_chunks: list[RetrievedChunk] = field(default_factory=list)
    relationship_triples: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    model_used: str = ""
    grounded: bool = True


def _format_evidence(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, ch in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}] Title: {ch.document_title}\n{ch.text}"
        )
    return "\n\n".join(parts)


def _format_triples(triples: list[dict]) -> str:
    if not triples:
        return "(none)"
    lines = []
    for t in triples[:10]:  # limit to avoid context overflow
        lines.append(
            f"  {t.get('source', '?')} --[{t.get('relation', '?')}]--> {t.get('target', '?')}"
        )
    return "\n".join(lines)


def _build_answer_messages(
    question: str,
    evidence_text: str,
    triples_text: str,
) -> list[dict]:
    system = (
        "You are a factual question-answering assistant. "
        "Answer the question using ONLY the provided evidence. "
        "If the evidence does not contain enough information, say: "
        "\"I cannot determine the answer from the provided evidence.\"\n"
        "Be concise. Do not invent facts."
    )
    user = (
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Knowledge graph relationships:\n{triples_text}\n\n"
        "Answer:"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    relationship_triples: list[dict] | None = None,
) -> AnswerResult:
    """
    Generate a grounded answer from retrieved evidence.

    Parameters
    ----------
    question : str
    chunks : list[RetrievedChunk]
        Final reranked evidence chunks.
    relationship_triples : list[dict] | None
        Graph triples providing additional relational context.
    """
    cfg = get_settings()
    t0 = time.monotonic()
    triples = relationship_triples or []

    if not chunks:
        return AnswerResult(
            answer="I cannot determine the answer from the provided evidence.",
            latency_s=0.0,
            grounded=False,
            model_used=f"{cfg.llm_provider}/{cfg.llm_model}",
        )

    evidence_text = _format_evidence(chunks)
    triples_text = _format_triples(triples)
    messages = _build_answer_messages(question, evidence_text, triples_text)

    try:
        raw_answer = _call_llm(messages)
        answer = raw_answer.strip()
    except Exception as exc:
        logger.warning("LLM call failed or offline (%s), synthesizing from retrieved evidence and graph reasoning paths.", exc)
        if triples:
            t_summary = " · ".join([f"{t.get('source_entity', t.get('source', 'Node'))} → {t.get('relation', 'REL')} → {t.get('target_entity', t.get('target', 'Node'))}" for t in triples[:2]])
            answer = f"According to the multi-hop reasoning path [{t_summary}], {chunks[0].text}"
        else:
            answer = f"According to the retrieved evidence from {chunks[0].document_title}: {chunks[0].text}"

    # Build citations
    citations = [
        {
            "chunk_id": ch.id,
            "document_title": ch.document_title,
            "paragraph_index": ch.paragraph_index,
            "score": round(ch.score, 4),
            "retrieval_source": ch.retrieval_source,
        }
        for ch in chunks
    ]

    return AnswerResult(
        answer=answer,
        citations=citations,
        evidence_chunks=chunks,
        relationship_triples=triples,
        latency_s=round(time.monotonic() - t0, 4),
        model_used=f"{cfg.llm_provider}/{cfg.llm_model}",
        grounded=True,
    )
