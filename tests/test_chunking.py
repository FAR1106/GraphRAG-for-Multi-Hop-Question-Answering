"""tests/test_chunking.py — Unit tests for chunking module."""
from __future__ import annotations

import pytest
from app.core.chunking import (
    make_chunk_id,
    make_chunks_from_paragraphs,
    normalize_text,
    chunk_dataset_samples,
)


class TestChunkIdStability:
    def test_same_inputs_same_id(self):
        id1 = make_chunk_id("Title A", 0, "hotpotqa")
        id2 = make_chunk_id("Title A", 0, "hotpotqa")
        assert id1 == id2

    def test_different_index_different_id(self):
        id1 = make_chunk_id("Title A", 0, "hotpotqa")
        id2 = make_chunk_id("Title A", 1, "hotpotqa")
        assert id1 != id2

    def test_different_title_different_id(self):
        id1 = make_chunk_id("Title A", 0, "hotpotqa")
        id2 = make_chunk_id("Title B", 0, "hotpotqa")
        assert id1 != id2

    def test_id_has_prefix(self):
        chunk_id = make_chunk_id("T", 0, "test")
        assert chunk_id.startswith("chunk_")


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_text("  a   b  ") == "a b"

    def test_unicode_normalization(self):
        # NFKC: fullwidth space → space
        assert normalize_text("café") == "café"


class TestMakeChunksFromParagraphs:
    def test_basic(self):
        chunks = make_chunks_from_paragraphs("Doc", ["Hello world. This is a test."], "hotpotqa")
        assert len(chunks) == 1
        assert chunks[0].document_title == "Doc"
        assert chunks[0].paragraph_index == 0

    def test_short_paragraphs_skipped(self):
        chunks = make_chunks_from_paragraphs("Doc", ["Hi", "A long enough paragraph here."], "hotpotqa", min_chars=20)
        assert len(chunks) == 1

    def test_deduplication(self):
        same = "This is the exact same text."
        chunks = make_chunks_from_paragraphs("Doc", [same, same], "hotpotqa")
        assert len(chunks) == 1

    def test_preserves_provenance(self):
        chunks = make_chunks_from_paragraphs("My Document", ["Some valid text here for testing."], "hotpotqa")
        assert chunks[0].document_title == "My Document"
        assert chunks[0].source_dataset == "hotpotqa"


class TestChunkDatasetSamples:
    def test_basic_sample(self):
        samples = [
            {
                "context": [
                    ["Doc1", ["Sentence one.", "Sentence two."]],
                    ["Doc2", ["Another sentence."]],
                ]
            }
        ]
        chunks = chunk_dataset_samples(samples, "hotpotqa")
        assert len(chunks) >= 1

    def test_global_deduplication(self):
        same_context = [["Doc1", ["Repeated text in both samples."]]]
        samples = [{"context": same_context}, {"context": same_context}]
        chunks = chunk_dataset_samples(samples, "hotpotqa")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs across samples"

    def test_missing_context_handled(self):
        samples = [{"context": []}, {"no_context_field": True}]
        # Should not raise
        chunks = chunk_dataset_samples(samples, "hotpotqa")
        assert isinstance(chunks, list)
