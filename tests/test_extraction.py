"""tests/test_extraction.py — Unit tests for extraction schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.extraction import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    VALID_ENTITY_TYPES,
    VALID_RELATION_TYPES,
)


class TestExtractedEntity:
    def test_valid_entity(self):
        e = ExtractedEntity(name="Marie Curie", entity_type="PERSON")
        assert e.name == "Marie Curie"
        assert e.entity_type == "PERSON"

    def test_invalid_type_falls_back_to_other(self):
        e = ExtractedEntity(name="X", entity_type="SPACESHIP")
        assert e.entity_type == "OTHER"

    def test_normalized_name(self):
        e = ExtractedEntity(name="Marie Curie!", entity_type="PERSON")
        assert e.normalized_name == "marie curie"

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            ExtractedEntity(name="", entity_type="PERSON")

    def test_description_defaults_empty(self):
        e = ExtractedEntity(name="Test", entity_type="OTHER")
        assert e.description == ""


class TestExtractedRelationship:
    def test_valid_relationship(self):
        r = ExtractedRelationship(
            source_entity="A", target_entity="B", relation_type="BORN_IN"
        )
        assert r.relation_type == "BORN_IN"

    def test_invalid_type_falls_back(self):
        r = ExtractedRelationship(
            source_entity="A", target_entity="B", relation_type="INVENTED"
        )
        assert r.relation_type == "RELATED_TO"

    def test_confidence_bounds(self):
        r = ExtractedRelationship(
            source_entity="A", target_entity="B",
            relation_type="BORN_IN", confidence=0.5
        )
        assert 0.0 <= r.confidence <= 1.0

    def test_confidence_exceeds_max_rejected(self):
        with pytest.raises(ValidationError):
            ExtractedRelationship(
                source_entity="A", target_entity="B",
                relation_type="BORN_IN", confidence=2.0
            )

    def test_empty_source_rejected(self):
        with pytest.raises(ValidationError):
            ExtractedRelationship(source_entity="", target_entity="B", relation_type="BORN_IN")


class TestExtractionResult:
    def test_empty_result_valid(self):
        r = ExtractionResult(chunk_id="c1")
        assert r.entities == []
        assert r.relationships == []

    def test_populated_result(self):
        e = ExtractedEntity(name="Einstein", entity_type="PERSON")
        r_item = ExtractedRelationship(
            source_entity="Einstein", target_entity="Germany", relation_type="BORN_IN"
        )
        result = ExtractionResult(
            entities=[e],
            relationships=[r_item],
            chunk_id="c1",
        )
        assert len(result.entities) == 1
        assert len(result.relationships) == 1
