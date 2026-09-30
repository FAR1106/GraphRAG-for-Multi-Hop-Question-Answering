"""
app/api/schemas.py — Pydantic request/response models for the FastAPI API.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Request models
# =============================================================================

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    method: Literal["vector", "graphrag", "decomposed", "hybrid", "graph"] = "hybrid"
    vector_top_k: int | None = Field(default=None, ge=1, le=50)
    graph_max_depth: int | None = Field(default=None, ge=1, le=4)
    final_context_size: int | None = Field(default=None, ge=1, le=20)


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=3)
    method: Literal["vector", "graph", "hybrid", "graphrag"] = "hybrid"
    vector_top_k: int | None = Field(default=None, ge=1, le=50)
    graph_max_depth: int | None = Field(default=None, ge=1, le=4)


class IngestRequest(BaseModel):
    dataset: Literal["hotpotqa", "2wikimultihopqa", "synthetic"] = "hotpotqa"
    sample_size: int = Field(default=100, ge=1, le=1000)
    run_extraction: bool = False  # set True to also run KG extraction


class EvaluateRequest(BaseModel):
    systems: list[Literal["vector", "graphrag", "decomposed", "hybrid", "graph"]] = Field(
        default=["vector", "hybrid"]
    )
    max_samples: int | None = Field(default=None, ge=1, le=500)
    eval_path: str | None = None


# =============================================================================
# Response models
# =============================================================================

class CitationModel(BaseModel):
    chunk_id: str
    document_title: str
    paragraph_index: int
    score: float
    retrieval_source: str


class EvidenceChunkModel(BaseModel):
    id: str
    text: str
    document_title: str
    paragraph_index: int
    score: float | None = None
    retrieval_source: str | None = None


class AskResponse(BaseModel):
    question: str
    method: str
    answer: str
    citations: list[CitationModel] = []
    evidence_chunks: list[EvidenceChunkModel] = []
    retrieved_chunks: list[dict[str, Any]] = []
    relationship_triples: list[dict[str, Any]] = []
    graph_paths: list[dict[str, Any]] = []
    sub_queries: list[str] = []
    retrieval_diagnostics: dict[str, Any] = {}
    total_latency_s: float = 0.0
    answer_latency_s: float = 0.0
    retrieval_latency_s: float = 0.0


class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    llm_provider: str
    llm_model: str
    embedding_model: str


class StatsResponse(BaseModel):
    chunk_count: int
    entity_count: int
    relationship_count: int
    mentions_count: int
    embedding_model: str
    llm_model: str
    dataset_name: str


class IngestResponse(BaseModel):
    message: str
    num_chunks: int
    total_time_s: float
    details: dict[str, Any]


class EntityResponse(BaseModel):
    id: str
    name: str
    normalized_name: str
    entity_type: str
    aliases: list[str]
    description: str


class NeighborResponse(BaseModel):
    entity: EntityResponse
    depth: int
    relations: list[str]


class SubgraphResponse(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


class ExperimentSummary(BaseModel):
    experiment_id: str
    timestamp: str
    systems: list[str]
    num_samples: int
    total_time_s: float
    status: str = "completed"


class EvaluateResponse(BaseModel):
    experiment_id: str
    message: str
    metrics: dict[str, Any]
