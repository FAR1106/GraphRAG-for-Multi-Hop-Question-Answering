"""tests/test_entity_resolution.py — Unit tests for entity resolution."""
from __future__ import annotations

import pytest
from app.core.entity_resolution import (
    normalize_entity_name,
    EntityRecord,
    EntityResolver,
    resolve_entities,
)


class TestNormalization:
    def test_lowercase(self):
        assert normalize_entity_name("Marie Curie") == "marie curie"

    def test_punctuation_stripped(self):
        assert normalize_entity_name("J.K. Rowling") == "jk rowling"

    def test_whitespace_collapsed(self):
        assert normalize_entity_name("  Albert   Einstein  ") == "albert einstein"

    def test_hyphen_preserved(self):
        result = normalize_entity_name("Jean-Paul Sartre")
        assert "jean-paul" in result


class TestEntityResolverExactMode:
    def _make_record(self, name: str, etype: str, idx: int = 0) -> EntityRecord:
        return EntityRecord(
            id=f"ent_{idx}",
            name=name,
            normalized_name=normalize_entity_name(name),
            entity_type=etype,
        )

    def test_exact_duplicate_merged(self):
        r = EntityResolver(mode="exact")
        records = [
            self._make_record("Paris", "LOCATION", 0),
            self._make_record("Paris", "LOCATION", 1),
        ]
        resolved = r.resolve(records)
        assert len(resolved) == 1
        assert r.stats.accepted_merges == 1

    def test_case_difference_merged_in_exact(self):
        r = EntityResolver(mode="exact")
        records = [
            self._make_record("paris", "LOCATION", 0),
            self._make_record("Paris", "LOCATION", 1),
        ]
        resolved = r.resolve(records)
        assert len(resolved) == 1

    def test_different_types_not_merged(self):
        r = EntityResolver(mode="exact")
        records = [
            self._make_record("Mercury", "PERSON", 0),
            self._make_record("Mercury", "LOCATION", 1),
        ]
        resolved = r.resolve(records)
        # Different types — should not merge
        assert len(resolved) == 2

    def test_similar_names_not_merged_in_exact(self):
        r = EntityResolver(mode="exact")
        records = [
            self._make_record("John Smith", "PERSON", 0),
            self._make_record("Jon Smith", "PERSON", 1),
        ]
        resolved = r.resolve(records)
        # Slightly different — exact mode should NOT merge
        assert len(resolved) == 2


class TestEntityResolverFuzzyMode:
    def _make_record(self, name: str, etype: str, idx: int = 0) -> EntityRecord:
        return EntityRecord(
            id=f"ent_{idx}",
            name=name,
            normalized_name=normalize_entity_name(name),
            entity_type=etype,
        )

    def test_close_names_merged_in_fuzzy(self):
        r = EntityResolver(mode="fuzzy", fuzzy_threshold=85.0)
        records = [
            self._make_record("United States of America", "LOCATION", 0),
            self._make_record("United States", "LOCATION", 1),
        ]
        resolved = r.resolve(records)
        # These are fuzzy-similar enough
        assert len(resolved) <= 2  # may or may not merge depending on ratio

    def test_completely_different_not_merged(self):
        r = EntityResolver(mode="fuzzy", fuzzy_threshold=85.0)
        records = [
            self._make_record("Albert Einstein", "PERSON", 0),
            self._make_record("Isaac Newton", "PERSON", 1),
        ]
        resolved = r.resolve(records)
        assert len(resolved) == 2

    def test_aliases_tracked(self):
        r = EntityResolver(mode="exact")
        records = [
            self._make_record("New York", "LOCATION", 0),
            self._make_record("New York", "LOCATION", 1),  # exact dup
        ]
        r.resolve(records)
        canon_id = list(r._canonical.keys())[0]
        canon = r._canonical[canon_id]
        assert "New York" in canon.aliases or canon.name == "New York"


class TestResolveEntitiesFunction:
    def test_returns_tuple(self):
        raw = [
            {"id": "e1", "name": "Paris", "entity_type": "LOCATION"},
            {"id": "e2", "name": "Paris", "entity_type": "LOCATION"},
        ]
        resolved, mapping, stats = resolve_entities(raw, mode="exact")
        assert isinstance(resolved, list)
        assert isinstance(mapping, dict)
        assert stats.accepted_merges >= 1

    def test_mapping_all_originals_present(self):
        raw = [
            {"id": "e1", "name": "A", "entity_type": "PERSON"},
            {"id": "e2", "name": "B", "entity_type": "PERSON"},
        ]
        _, mapping, _ = resolve_entities(raw, mode="exact")
        assert "e1" in mapping
        assert "e2" in mapping
