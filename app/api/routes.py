"""
app/api/routes.py — FastAPI route handlers.

Business logic is delegated to service modules.
All inputs are validated by Pydantic schemas.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import (
    AskRequest, AskResponse,
    EvaluateRequest, EvaluateResponse,
    EntityResponse, NeighborResponse, SubgraphResponse,
    HealthResponse, StatsResponse,
    IngestRequest, IngestResponse,
    ExperimentSummary,
)
from app.config import get_settings
from app.db.neo4j_client import get_neo4j_client, Neo4jClient
from app.db import queries
from app.services.rag_service import ask
from app.services.evaluation_service import run_evaluation

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency: Neo4j client
# ---------------------------------------------------------------------------

def get_db() -> Neo4jClient:
    return get_neo4j_client()


# =============================================================================
# Health / Status
# =============================================================================

@router.get("/health", response_model=HealthResponse, tags=["Status"])
def health_check():
    cfg = get_settings()
    client = get_neo4j_client()
    return HealthResponse(
        status="ok",
        neo4j_connected=client.is_connected(),
        llm_provider=cfg.llm_provider,
        llm_model=cfg.llm_model,
        embedding_model=cfg.embedding_model,
    )


@router.get("/stats", response_model=StatsResponse, tags=["Status"])
def get_stats():
    cfg = get_settings()
    client = get_neo4j_client()
    stats = {}
    if client.is_connected():
        try:
            stats = client.get_stats()
        except Exception:
            pass
    else:
        from app.db.in_memory_graph import InMemoryGraphStore
        stats = InMemoryGraphStore.get_instance().get_stats()

    return StatsResponse(
        chunk_count=stats.get("chunk_count", 0),
        entity_count=stats.get("entity_count", 0),
        relationship_count=stats.get("relationship_count", 0),
        mentions_count=stats.get("mentions_count", 0),
        embedding_model=cfg.embedding_model,
        llm_model=cfg.llm_model,
        dataset_name=cfg.dataset_name,
    )


# =============================================================================
# Ingestion
# =============================================================================

@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest(req: IngestRequest, db: Neo4jClient = Depends(get_db)):
    """Ingest the configured dataset into Neo4j."""
    cfg = get_settings()
    try:
        from app.core.ingestion import ingest_samples
        from scripts.sample_dataset import load_or_sample

        samples = load_or_sample(req.dataset, req.sample_size)
        result = ingest_samples(db, samples, source_dataset=req.dataset)

        if req.run_extraction:
            from app.core.extraction import extract_from_chunks
            from app.core.ingestion import get_all_chunk_dicts
            from app.core.graph_builder import build_graph

            chunk_dicts = get_all_chunk_dicts(db)
            extraction_results = extract_from_chunks(chunk_dicts)
            graph_stats = build_graph(db, extraction_results)
            result["graph_stats"] = graph_stats

        return IngestResponse(
            message="Ingestion complete.",
            num_chunks=result.get("num_chunks", 0),
            total_time_s=result.get("total_time_s", 0.0),
            details=result,
        )
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# =============================================================================
# Ask / Retrieve
# =============================================================================

@router.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_question(req: AskRequest, db: Neo4jClient = Depends(get_db)):
    """Answer a question using the configured retrieval method."""
    try:
        response = ask(
            db,
            req.question,
            method=req.method,
            vector_top_k=req.vector_top_k,
            graph_max_depth=req.graph_max_depth,
            final_context_size=req.final_context_size,
        )
        resp_dict = response.__dict__.copy()
        resp_dict["retrieved_chunks"] = resp_dict.get("evidence_chunks", [])
        resp_dict["graph_paths"] = resp_dict.get("relationship_triples", [])
        resp_dict["sub_queries"] = resp_dict.get("retrieval_diagnostics", {}).get("sub_questions", [])
        return AskResponse(**resp_dict)
    except Exception as exc:
        logger.error("Ask failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/retrieve", tags=["RAG"])
def retrieve(req: dict, db: Neo4jClient = Depends(get_db)):
    """Raw retrieval without answer generation."""
    question = req.get("question", "")
    method = req.get("method", "hybrid")
    try:
        if method == "vector":
            from app.core.vector_retriever import retrieve_by_vector
            r = retrieve_by_vector(db, question)
            return {"chunks": [c.__dict__ for c in r.chunks], "latency_s": r.latency_s}
        elif method == "graph":
            from app.core.graph_retriever import retrieve_by_graph
            r = retrieve_by_graph(db, question)
            return {"chunks": [c.__dict__ for c in r.chunks], "latency_s": r.latency_s}
        else:
            from app.core.hybrid_retriever import retrieve_hybrid
            r = retrieve_hybrid(db, question)
            return {"chunks": [c.__dict__ for c in r.final_chunks], "latency_s": r.total_latency_s}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# =============================================================================
# Graph Explorer
# =============================================================================

@router.get("/graph/entity/{entity_id}", response_model=EntityResponse, tags=["Graph"])
def get_entity(entity_id: str, db: Neo4jClient = Depends(get_db)):
    if not db.is_connected():
        from app.db.in_memory_graph import InMemoryGraphStore
        store = InMemoryGraphStore.get_instance()
        matches = store.search_entities(entity_id)
        if not matches:
            raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found.")
        m = matches[0]
        return EntityResponse(
            id=m["id"],
            name=m["name"],
            type=m.get("type", "OTHER"),
            description=m.get("description", ""),
            aliases=[],
        )
    entity = queries.get_entity_by_id(db, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found.")
    return EntityResponse(**entity)


@router.get("/graph/neighbors/{entity_id}", tags=["Graph"])
def get_neighbors(entity_id: str, depth: int = 1, db: Neo4jClient = Depends(get_db)):
    if not db.is_connected():
        from app.db.in_memory_graph import InMemoryGraphStore
        store = InMemoryGraphStore.get_instance()
        sub = store.traverse_subgraph(entity_id, depth=depth)
        return {"entity_id": entity_id, "neighbors": sub["nodes"]}
    rows = queries.get_entity_neighbors(db, entity_id, max_depth=depth)
    return {"entity_id": entity_id, "neighbors": rows}


@router.get("/graph/subgraph/{entity_id}", response_model=SubgraphResponse, tags=["Graph"])
def get_subgraph(entity_id: str, depth: int = 1, limit: int = 30, db: Neo4jClient = Depends(get_db)):
    if not db.is_connected():
        from app.db.in_memory_graph import InMemoryGraphStore
        store = InMemoryGraphStore.get_instance()
        subgraph = store.traverse_subgraph(entity_id, depth=depth, limit=limit)
        return SubgraphResponse(**subgraph)
    subgraph = queries.get_subgraph_for_entity(db, entity_id, depth=depth, limit=limit)
    return SubgraphResponse(**subgraph)


@router.get("/graph/search", tags=["Graph"])
def search_entities(q: str, entity_type: str | None = None, db: Neo4jClient = Depends(get_db)):
    if not db.is_connected():
        from app.db.in_memory_graph import InMemoryGraphStore
        store = InMemoryGraphStore.get_instance()
        results = store.search_entities(q)
        return {"query": q, "results": results, "count": len(results)}
    from app.core.entity_resolution import normalize_entity_name
    results = queries.search_entities_by_name(db, normalize_entity_name(q), entity_type)
    return {"query": q, "results": results, "count": len(results)}


# =============================================================================
# Experiments
# =============================================================================

@router.get("/experiments", tags=["Evaluation"])
def list_experiments():
    cfg = get_settings()
    experiments = []
    for exp_dir in cfg.results_dir.iterdir():
        exp_file = exp_dir / "experiment.json"
        if exp_file.exists():
            try:
                data = json.loads(exp_file.read_text(encoding="utf-8"))
                experiments.append(
                    ExperimentSummary(
                        experiment_id=data.get("experiment_id", exp_dir.name),
                        timestamp=data.get("timestamp", ""),
                        systems=data.get("systems", []),
                        num_samples=data.get("num_samples", 0),
                        total_time_s=data.get("total_time_s", 0.0),
                    )
                )
            except Exception:
                pass
    return {"experiments": [e.model_dump() for e in experiments]}


@router.get("/experiments/{experiment_id}", tags=["Evaluation"])
def get_experiment(experiment_id: str):
    cfg = get_settings()
    exp_file = cfg.results_dir / experiment_id / "experiment.json"
    if not exp_file.exists():
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found.")
    return json.loads(exp_file.read_text(encoding="utf-8"))


@router.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
def run_eval(req: EvaluateRequest, db: Neo4jClient = Depends(get_db)):
    try:
        eval_path = Path(req.eval_path) if req.eval_path else None
        result = run_evaluation(
            db,
            systems=req.systems,
            eval_path=eval_path,
            max_samples=req.max_samples,
        )
        return EvaluateResponse(
            experiment_id=result["experiment_id"],
            message="Evaluation complete.",
            metrics=result["metrics"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
