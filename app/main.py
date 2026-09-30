"""
app/main.py — FastAPI application entry point.
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.db.neo4j_client import get_neo4j_client

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

cfg = get_settings()
logging.basicConfig(
    level=getattr(logging, cfg.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to Neo4j and initialize schema. Shutdown: close driver."""
    logger.info("Starting GraphRAG API …")
    client = get_neo4j_client()
    try:
        client.connect()
        client.initialize_schema()
        client.create_vector_index()
        logger.info("Neo4j ready.")
    except Exception as exc:
        logger.warning("Neo4j connection failed at startup: %s — running in degraded mode.", exc)

    yield

    logger.info("Shutting down GraphRAG API …")
    client.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GraphRAG Multi-Hop QA API",
    description=(
        "Research-oriented GraphRAG system combining knowledge graph traversal "
        "with vector search for multi-hop question answering."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Root"])
def root():
    return {
        "name": "GraphRAG Multi-Hop QA",
        "version": "1.0.0",
        "docs": "/docs",
    }
