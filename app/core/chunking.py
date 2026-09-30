"""
app/core/chunking.py — Document chunking and normalization.

Keeps paragraph boundaries intact for HotpotQA-style data where each
supporting fact is already paragraph-aligned.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Chunk:
    """A single text chunk with full provenance."""
    id: str
    text: str
    normalized_text: str
    document_title: str
    paragraph_index: int
    source_dataset: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "document_title": self.document_title,
            "paragraph_index": self.paragraph_index,
            "source_dataset": self.source_dataset,
            "char_count": self.char_count,
            # embedding added later by the embeddings module
            "embedding": None,
        }


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace; strip leading/trailing spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Normalize text for retrieval without altering evidence content."""
    return _normalize_whitespace(text).lower()


# ---------------------------------------------------------------------------
# Chunk ID generation
# ---------------------------------------------------------------------------

def make_chunk_id(document_title: str, paragraph_index: int, source_dataset: str) -> str:
    """
    Generate a stable, unique chunk ID.
    ID is deterministic: same inputs always produce the same ID.
    """
    raw = f"{source_dataset}::{document_title}::{paragraph_index}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"chunk_{digest}"


# ---------------------------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------------------------

def make_chunks_from_paragraphs(
    document_title: str,
    paragraphs: list[str],
    source_dataset: str,
    min_chars: int = 20,
    max_chars: int = 2000,
) -> list[Chunk]:
    """
    Convert a list of paragraph strings into Chunk objects.

    Parameters
    ----------
    document_title : str
        Title of the source document.
    paragraphs : list[str]
        Raw paragraph strings (may be empty or very short).
    source_dataset : str
        Dataset tag, e.g. "hotpotqa".
    min_chars : int
        Skip paragraphs shorter than this (likely empty/noise).
    max_chars : int
        Hard-split paragraphs longer than this (rare for HotpotQA).

    Returns
    -------
    list[Chunk]
        Deduplicated chunks with stable IDs.
    """
    chunks: list[Chunk] = []
    seen_texts: set[str] = set()

    for para_idx, raw_para in enumerate(paragraphs):
        normalized = _normalize_whitespace(raw_para)
        if len(normalized) < min_chars:
            continue

        # Hard split only for very long paragraphs
        sub_paras = _split_if_long(normalized, max_chars)

        for sub_idx, sub_text in enumerate(sub_paras):
            norm_lower = normalize_text(sub_text)
            if norm_lower in seen_texts:
                continue  # exact duplicate
            seen_texts.add(norm_lower)

            # Stable ID incorporates sub-paragraph offset
            effective_idx = para_idx * 100 + sub_idx
            chunk_id = make_chunk_id(document_title, effective_idx, source_dataset)

            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=sub_text,
                    normalized_text=norm_lower,
                    document_title=document_title,
                    paragraph_index=effective_idx,
                    source_dataset=source_dataset,
                )
            )

    return chunks


def _split_if_long(text: str, max_chars: int) -> list[str]:
    """Split text at sentence boundaries only when it exceeds max_chars."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    buffer: list[str] = []
    buf_len = 0

    for sent in sentences:
        if buf_len + len(sent) > max_chars and buffer:
            parts.append(" ".join(buffer))
            buffer, buf_len = [sent], len(sent)
        else:
            buffer.append(sent)
            buf_len += len(sent)

    if buffer:
        parts.append(" ".join(buffer))

    return parts if parts else [text]


# ---------------------------------------------------------------------------
# Dataset-level chunking entry point
# ---------------------------------------------------------------------------

def chunk_dataset_samples(
    samples: list[dict],
    source_dataset: str = "hotpotqa",
) -> list[Chunk]:
    """
    Convert sampled dataset records into Chunk objects.

    Expected sample fields (HotpotQA format):
      - context: list of [title, list_of_sentences] pairs
    """
    all_chunks: list[Chunk] = []
    global_seen: set[str] = set()

    for sample in samples:
        context = sample.get("context", [])
        for title_para_pair in context:
            if not title_para_pair or len(title_para_pair) < 2:
                continue
            title = str(title_para_pair[0])
            sentences: list[str] = title_para_pair[1]
            # Join sentences into one paragraph per HotpotQA context entry
            full_para = " ".join(str(s) for s in sentences).strip()
            if not full_para:
                continue

            chunks = make_chunks_from_paragraphs(
                document_title=title,
                paragraphs=[full_para],
                source_dataset=source_dataset,
            )
            for ch in chunks:
                if ch.id not in global_seen:
                    global_seen.add(ch.id)
                    all_chunks.append(ch)

    return all_chunks
