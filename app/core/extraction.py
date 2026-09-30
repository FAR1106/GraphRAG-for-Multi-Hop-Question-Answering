"""
app/core/extraction.py — LLM-assisted structured entity and relation extraction.

Supports Ollama (local), OpenAI, Anthropic, Groq and a mock mode for testing.
All outputs are validated against strict Pydantic schemas before use.
Successful extractions are cached to avoid repeated LLM calls.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Schemas
# =============================================================================

VALID_ENTITY_TYPES = {
    "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "WORK", "DATE", "OTHER"
}

VALID_RELATION_TYPES = {
    "BORN_IN", "DIED_IN", "FOUNDED_BY", "FOUNDED_IN", "LOCATED_IN", "PART_OF",
    "MEMBER_OF", "WORKS_FOR", "CREATED_BY", "DIRECTED_BY", "PRODUCED_BY",
    "WRITTEN_BY", "PUBLISHED_BY", "AWARDED_TO", "MARRIED_TO", "CHILD_OF",
    "PARENT_OF", "SIBLING_OF", "SUCCEEDED_BY", "PRECEDED_BY", "AFFILIATED_WITH",
    "STUDIED_AT", "TAUGHT_AT", "KNOWN_FOR", "RELATED_TO", "INSTANCE_OF",
    "NATIONALITY", "OCCUPATION", "GENRE", "GENRE_OF",
}


class ExtractedEntity(BaseModel):
    name: str = Field(..., min_length=1, max_length=300)
    entity_type: str
    description: str = ""

    @field_validator("entity_type", mode="before")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v_upper = str(v).upper().strip()
        return v_upper if v_upper in VALID_ENTITY_TYPES else "OTHER"

    @property
    def normalized_name(self) -> str:
        import re
        name = re.sub(r"[^\w\s]", "", self.name).strip().lower()
        return " ".join(name.split())


class ExtractedRelationship(BaseModel):
    source_entity: str = Field(..., min_length=1)
    target_entity: str = Field(..., min_length=1)
    relation_type: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("relation_type", mode="before")
    @classmethod
    def validate_rel_type(cls, v: str) -> str:
        v_upper = str(v).upper().strip().replace(" ", "_")
        return v_upper if v_upper in VALID_RELATION_TYPES else "RELATED_TO"


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
    chunk_id: str = ""
    extraction_time_s: float = 0.0
    model_used: str = ""
    from_cache: bool = False


# =============================================================================
# LLM Provider Abstraction
# =============================================================================

def _build_extraction_prompt(text: str) -> list[dict[str, str]]:
    system = """You are an expert information extraction assistant.
Extract named entities and relationships from the provided text.
You MUST respond with valid JSON only — no markdown, no explanation.

Entity types allowed: PERSON, ORGANIZATION, LOCATION, EVENT, WORK, DATE, OTHER

Relationship types allowed:
BORN_IN, DIED_IN, FOUNDED_BY, FOUNDED_IN, LOCATED_IN, PART_OF, MEMBER_OF,
WORKS_FOR, CREATED_BY, DIRECTED_BY, PRODUCED_BY, WRITTEN_BY, PUBLISHED_BY,
AWARDED_TO, MARRIED_TO, CHILD_OF, PARENT_OF, SIBLING_OF, SUCCEEDED_BY,
PRECEDED_BY, AFFILIATED_WITH, STUDIED_AT, TAUGHT_AT, KNOWN_FOR, RELATED_TO,
INSTANCE_OF, NATIONALITY, OCCUPATION, GENRE, GENRE_OF

Return JSON with this exact structure:
{
  "entities": [
    {"name": "...", "entity_type": "...", "description": "..."}
  ],
  "relationships": [
    {"source_entity": "...", "target_entity": "...", "relation_type": "...", "confidence": 0.9}
  ]
}

Rules:
- Only extract entities explicitly mentioned in the text.
- Only create relationships supported by text evidence.
- Use canonical entity names as written in the text.
- If no entities or relationships exist, return empty lists.
"""
    user = f"Text to analyse:\n\n{text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _call_ollama(messages: list[dict], model: str, temperature: float) -> str:
    import httpx

    cfg = get_settings()
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = httpx.post(
        f"{cfg.ollama_base_url}/api/chat",
        json=payload,
        timeout=3.0,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _call_openai(messages: list[dict], model: str, temperature: float) -> str:
    from openai import OpenAI

    cfg = get_settings()
    client = OpenAI(api_key=cfg.openai_api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content or ""


def _call_anthropic(messages: list[dict], model: str, temperature: float) -> str:
    import anthropic

    cfg = get_settings()
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    system_msg = messages[0]["content"] if messages[0]["role"] == "system" else ""
    user_msgs = [m for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model=model,
        system=system_msg,
        messages=user_msgs,  # type: ignore[arg-type]
        max_tokens=1024,
        temperature=temperature,
    )
    return resp.content[0].text


def _call_groq(messages: list[dict], model: str, temperature: float) -> str:
    from groq import Groq

    cfg = get_settings()
    client = Groq(api_key=cfg.groq_api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
    )
    return completion.choices[0].message.content or ""


def _mock_extraction(text: str) -> dict:
    """Return a minimal mock extraction for pipeline testing only."""
    return {
        "entities": [{"name": "MockEntity", "entity_type": "OTHER", "description": "mock"}],
        "relationships": [],
    }


def _call_llm(messages: list[dict]) -> str:
    cfg = get_settings()
    provider = cfg.llm_provider
    model = cfg.llm_model
    temperature = cfg.llm_temperature

    if provider == "mock":
        return json.dumps(_mock_extraction(""))
    elif provider == "ollama":
        return _call_ollama(messages, model, temperature)
    elif provider == "openai":
        return _call_openai(messages, model, temperature)
    elif provider == "anthropic":
        return _call_anthropic(messages, model, temperature)
    elif provider == "groq":
        return _call_groq(messages, model, temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# =============================================================================
# Cache helpers
# =============================================================================

def _cache_path(chunk_id: str) -> Path:
    cfg = get_settings()
    return cfg.cache_dir / "extractions" / f"{chunk_id}.json"


def _load_cache(chunk_id: str) -> dict | None:
    cfg = get_settings()
    if not cfg.extraction_cache_enabled:
        return None
    path = _cache_path(chunk_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_cache(chunk_id: str, data: dict) -> None:
    cfg = get_settings()
    if not cfg.extraction_cache_enabled:
        return
    path = _cache_path(chunk_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# Main extraction function
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _extract_with_retry(text: str) -> dict:
    messages = _build_extraction_prompt(text)
    raw = _call_llm(messages)
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        raw = raw.rstrip("`").strip()
    return json.loads(raw)


def extract_from_chunk(chunk_id: str, text: str) -> ExtractionResult:
    """
    Extract entities and relationships from a single chunk.
    Uses cache, validates via Pydantic, retries on malformed output.
    """
    cfg = get_settings()
    t0 = time.monotonic()

    # --- Cache check ---
    cached = _load_cache(chunk_id)
    if cached:
        try:
            result = ExtractionResult.model_validate(cached)
            result.from_cache = True
            return result
        except Exception:
            pass  # Cache corrupted — re-extract

    raw_dict: dict = {}
    try:
        raw_dict = _extract_with_retry(text)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failure for chunk %s: %s", chunk_id, exc)
        raw_dict = {"entities": [], "relationships": []}
    except Exception as exc:
        logger.error("LLM extraction failed for chunk %s: %s", chunk_id, exc)
        raw_dict = {"entities": [], "relationships": []}

    try:
        result = ExtractionResult(
            entities=[ExtractedEntity.model_validate(e) for e in raw_dict.get("entities", [])],
            relationships=[
                ExtractedRelationship.model_validate(r)
                for r in raw_dict.get("relationships", [])
            ],
            chunk_id=chunk_id,
            extraction_time_s=time.monotonic() - t0,
            model_used=f"{cfg.llm_provider}/{cfg.llm_model}",
        )
    except Exception as exc:
        logger.warning("Pydantic validation failed for chunk %s: %s", chunk_id, exc)
        result = ExtractionResult(
            chunk_id=chunk_id,
            extraction_time_s=time.monotonic() - t0,
            model_used=f"{cfg.llm_provider}/{cfg.llm_model}",
        )

    _save_cache(chunk_id, result.model_dump())
    return result


def extract_from_chunks(
    chunks: list[dict],
    progress_callback=None,
) -> list[ExtractionResult]:
    """Extract from a list of chunk dicts (must have 'id' and 'text')."""
    results: list[ExtractionResult] = []
    for i, chunk in enumerate(chunks):
        result = extract_from_chunk(chunk["id"], chunk["text"])
        results.append(result)
        if progress_callback:
            progress_callback(i + 1, len(chunks))
    return results
