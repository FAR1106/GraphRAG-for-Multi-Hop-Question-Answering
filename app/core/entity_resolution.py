"""
app/core/entity_resolution.py — Entity deduplication and canonical merging.

Supports three resolution modes:
  exact     — case/whitespace normalization only
  fuzzy     — + rapidfuzz token-sort ratio
  embedding — + embedding cosine similarity for ambiguous pairs

Never merges entities of different types.
Records all candidate merges, accepted merges and rejections.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


# =============================================================================
# Normalization
# =============================================================================

def normalize_entity_name(name: str) -> str:
    """
    Canonicalize an entity name for comparison:
    - Unicode NFKC normalization
    - Lowercase
    - Strip punctuation (except hyphens)
    - Collapse whitespace
    """
    name = unicodedata.normalize("NFKC", name)
    name = name.lower()
    name = re.sub(r"[^\w\s-]", "", name)  # keep hyphens
    name = re.sub(r"\s+", " ", name).strip()
    return name


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class EntityRecord:
    """Internal entity record during resolution."""
    id: str
    name: str
    normalized_name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    # After resolution, points to the canonical entity id
    canonical_id: str = ""

    def __post_init__(self) -> None:
        if not self.canonical_id:
            self.canonical_id = self.id


@dataclass
class ResolutionStats:
    candidates_considered: int = 0
    accepted_merges: int = 0
    rejected_merges: int = 0
    final_entity_count: int = 0


# =============================================================================
# Resolution engine
# =============================================================================

class EntityResolver:
    """
    Deduplicates entities extracted from multiple chunks.

    Parameters
    ----------
    mode : str
        Resolution aggressiveness: 'exact', 'fuzzy', or 'embedding'.
    fuzzy_threshold : float
        Minimum token-sort ratio (0–100 scale) to consider a fuzzy match.
    embedding_threshold : float
        Minimum cosine similarity to merge ambiguous fuzzy candidates.
    """

    def __init__(
        self,
        mode: Literal["exact", "fuzzy", "embedding"] = "fuzzy",
        fuzzy_threshold: float = 85.0,
        embedding_threshold: float = 0.90,
    ) -> None:
        self.mode = mode
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold
        self.stats = ResolutionStats()

        # canonical_id -> EntityRecord
        self._canonical: dict[str, EntityRecord] = {}
        # original_id -> canonical_id
        self._mapping: dict[str, str] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(self, entities: list[EntityRecord]) -> list[EntityRecord]:
        """
        Deduplicate a list of EntityRecord objects and return the canonical set.
        Updates self.stats and self._mapping.
        """
        self._canonical = {}
        self._mapping = {}

        for entity in entities:
            self._process_entity(entity)

        resolved = list(self._canonical.values())
        self.stats.final_entity_count = len(resolved)
        logger.info(
            "Entity resolution (%s): %d input → %d canonical | "
            "accepted=%d rejected=%d",
            self.mode,
            len(entities),
            len(resolved),
            self.stats.accepted_merges,
            self.stats.rejected_merges,
        )
        return resolved

    def get_canonical_id(self, original_id: str) -> str:
        """Look up the canonical entity ID for an original entity ID."""
        return self._mapping.get(original_id, original_id)

    def get_mapping(self) -> dict[str, str]:
        """Return the full original → canonical mapping."""
        return dict(self._mapping)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_entity(self, entity: EntityRecord) -> None:
        match_id = self._find_match(entity)
        if match_id:
            self.stats.accepted_merges += 1
            self._mapping[entity.id] = match_id
            # Absorb alias
            canon = self._canonical[match_id]
            if entity.name not in canon.aliases:
                canon.aliases.append(entity.name)
        else:
            # New canonical entity
            self._canonical[entity.id] = entity
            self._mapping[entity.id] = entity.id

    def _find_match(self, entity: EntityRecord) -> str | None:
        """Return the canonical_id of the best match, or None."""
        for canon_id, canon in self._canonical.items():
            # Strict type check first
            if canon.entity_type != entity.entity_type:
                continue

            self.stats.candidates_considered += 1

            # 1. Exact normalized match
            if canon.normalized_name == entity.normalized_name:
                return canon_id

            if self.mode == "exact":
                continue

            # 2. Fuzzy string match
            score = self._fuzzy_score(canon.normalized_name, entity.normalized_name)
            if score >= self.fuzzy_threshold:
                if self.mode == "fuzzy":
                    return canon_id

                # 3. Confirm with embedding similarity
                if self.mode == "embedding":
                    sim = self._embedding_similarity(entity.name, canon.name)
                    if sim >= self.embedding_threshold:
                        return canon_id
                    else:
                        self.stats.rejected_merges += 1
                        continue

            self.stats.rejected_merges += 1

        return None

    @staticmethod
    def _fuzzy_score(a: str, b: str) -> float:
        try:
            from rapidfuzz.fuzz import token_sort_ratio
            return token_sort_ratio(a, b)
        except ImportError:
            # Fallback: simple character overlap
            set_a, set_b = set(a.split()), set(b.split())
            union = set_a | set_b
            if not union:
                return 0.0
            return 100.0 * len(set_a & set_b) / len(union)

    @staticmethod
    def _embedding_similarity(name_a: str, name_b: str) -> float:
        try:
            from app.core.embeddings import embed_texts, cosine_similarity
            embeddings = embed_texts([name_a, name_b])
            return cosine_similarity(embeddings[0], embeddings[1])
        except Exception as exc:
            logger.warning("Embedding similarity failed: %s", exc)
            return 0.0


# =============================================================================
# Convenience function
# =============================================================================

def resolve_entities(
    raw_entities: list[dict[str, Any]],
    mode: str = "fuzzy",
    fuzzy_threshold: float = 85.0,
    embedding_threshold: float = 0.90,
) -> tuple[list[dict[str, Any]], dict[str, str], ResolutionStats]:
    """
    High-level entry point for entity resolution.

    Parameters
    ----------
    raw_entities : list[dict]
        Each dict must have keys: id, name, normalized_name, entity_type.
    mode : str
        'exact' | 'fuzzy' | 'embedding'
    fuzzy_threshold : float
        0–100 scale threshold for rapidfuzz.
    embedding_threshold : float
        0–1 cosine similarity threshold.

    Returns
    -------
    (resolved_dicts, id_mapping, stats)
    """
    records = [
        EntityRecord(
            id=e["id"],
            name=e["name"],
            normalized_name=e.get("normalized_name", normalize_entity_name(e["name"])),
            entity_type=e.get("entity_type", "OTHER"),
            aliases=e.get("aliases", []),
            description=e.get("description", ""),
        )
        for e in raw_entities
    ]

    resolver = EntityResolver(
        mode=mode,  # type: ignore[arg-type]
        fuzzy_threshold=fuzzy_threshold,
        embedding_threshold=embedding_threshold,
    )
    resolved = resolver.resolve(records)
    mapping = resolver.get_mapping()
    stats = resolver.stats

    resolved_dicts = [
        {
            "id": r.id,
            "name": r.name,
            "normalized_name": r.normalized_name,
            "entity_type": r.entity_type,
            "aliases": r.aliases,
            "description": r.description,
            "canonical_id": r.canonical_id,
        }
        for r in resolved
    ]
    return resolved_dicts, mapping, stats
