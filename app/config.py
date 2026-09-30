"""
app/config.py — Central configuration management for GraphRAG.
All settings are loaded from environment variables (via .env).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Neo4j ───────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="graphrag_password", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # ── LLM ─────────────────────────────────────────────────────────────────
    llm_provider: Literal["ollama", "openai", "anthropic", "groq", "mock"] = Field(
        default="ollama", alias="LLM_PROVIDER"
    )
    llm_model: str = Field(default="llama3.2", alias="LLM_MODEL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")

    # ── Reranker ─────────────────────────────────────────────────────────────
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    reranker_enabled: bool = Field(default=True, alias="RERANKER_ENABLED")

    # ── Retrieval ────────────────────────────────────────────────────────────
    vector_top_k: int = Field(default=10, alias="VECTOR_TOP_K")
    graph_max_depth: int = Field(default=2, alias="GRAPH_MAX_DEPTH")
    graph_max_candidates: int = Field(default=20, alias="GRAPH_MAX_CANDIDATES")
    final_context_size: int = Field(default=5, alias="FINAL_CONTEXT_SIZE")

    # ── Entity Resolution ────────────────────────────────────────────────────
    entity_resolution_mode: Literal["exact", "fuzzy", "embedding"] = Field(
        default="fuzzy", alias="ENTITY_RESOLUTION_MODE"
    )
    entity_fuzzy_threshold: float = Field(default=0.85, alias="ENTITY_FUZZY_THRESHOLD")
    entity_embedding_threshold: float = Field(default=0.90, alias="ENTITY_EMBEDDING_THRESHOLD")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset_name: Literal["hotpotqa", "2wikimultihopqa", "synthetic"] = Field(
        default="hotpotqa", alias="DATASET_NAME"
    )
    dataset_sample_size: int = Field(default=100, alias="DATASET_SAMPLE_SIZE")
    dataset_random_seed: int = Field(default=42, alias="DATASET_RANDOM_SEED")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")

    # ── Experiment ───────────────────────────────────────────────────────────
    results_dir: Path = Field(default=Path("results"), alias="RESULTS_DIR")
    cache_dir: Path = Field(default=Path("data/cache"), alias="CACHE_DIR")
    extraction_cache_enabled: bool = Field(default=True, alias="EXTRACTION_CACHE_ENABLED")

    # ── API / App ─────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ── Cost Estimation ───────────────────────────────────────────────────────
    cost_input_per_1k: float = Field(default=0.0, alias="COST_INPUT_PER_1K")
    cost_output_per_1k: float = Field(default=0.0, alias="COST_OUTPUT_PER_1K")

    # ── Optional Extensions ───────────────────────────────────────────────────
    question_decomposition_enabled: bool = Field(
        default=False, alias="QUESTION_DECOMPOSITION_ENABLED"
    )

    # ── Computed paths ────────────────────────────────────────────────────────
    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def eval_data_path(self) -> Path:
        return self.processed_data_dir / "eval.json"

    @property
    def multihop_eval_path(self) -> Path:
        return self.processed_data_dir / "eval_multihop.json"

    @property
    def singlehop_eval_path(self) -> Path:
        return self.processed_data_dir / "eval_singlehop.json"

    @field_validator("embedding_dimension", mode="before")
    @classmethod
    def infer_dimension(cls, v: int, info) -> int:
        return int(v)

    def ensure_directories(self) -> None:
        """Create all required directories if they do not exist."""
        dirs = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.cache_dir,
            self.results_dir,
            Path("logs"),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    s = Settings()
    s.ensure_directories()
    return s
