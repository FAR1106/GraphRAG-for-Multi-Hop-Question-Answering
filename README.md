# 🔬 GraphRAG: Knowledge Graph-Augmented Retrieval for Multi-Hop QA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j 5+](https://img.shields.io/badge/neo4j-5.20+-blue.svg)](https://neo4j.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, research-grade **GraphRAG system designed for complex Multi-Hop Question Answering** across connected document collections (HotpotQA / 2WikiMultiHopQA benchmarks). 

Standard dense vector RAG frequently fails on multi-hop questions when the supporting facts are separated by entity bridges rather than high semantic similarity. GraphRAG overcomes this by pairing deterministic dense vector retrieval with structured Knowledge Graph entity extraction, fuzzy entity resolution, multi-hop traversal, and Reciprocal Rank Fusion (RRF).

---

## 🏗️ System Architecture

```
                                      [ User Query ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ Question Decomposer ]                     [ Dense Vector Search ]
            Sub-queries: [Q1, Q2, ...]                    Cosine Sim (all-MiniLM-L6-v2)
                       │                                           │
                       ▼                                           │
          [ Entity Extraction & Linking ]                          │
                       │                                           │
                       ▼                                           │
           [ Neo4j Multi-Hop Traversal ]                           │
           (1-3 Hops + Type-gated Walk)                            │
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             ▼
                                [ Reciprocal Rank Fusion (RRF) ]
                                             │
                                  [ Context Deduplication ]
                                             │
                               [ LLM Answer & Provenance Generator ]
                               (Ollama / OpenAI / Claude / Groq)
                                             │
                                             ▼
                         [ Structured Answer with Graph Proof Trace ]
```

---

## 📂 Repository Structure

```
GRAPHRAG/
├── app/
│   ├── api/
│   │   ├── routes.py            # FastAPI REST endpoints
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── core/
│   │   ├── chunking.py          # Deterministic SHA-256 chunking & normalization
│   │   ├── embeddings.py        # SentenceTransformers local embedding engine
│   │   ├── extraction.py        # Schema-constrained LLM entity & relation extractor
│   │   ├── entity_resolution.py # Exact & fuzzy alias resolution & deduplication
│   │   ├── graph_builder.py     # Neo4j graph population and MENTIONS link creation
│   │   ├── question_decomposer.py # Multi-hop query decomposition into reasoning steps
│   │   ├── vector_retriever.py  # Dense vector similarity retriever
│   │   ├── graph_retriever.py   # Multi-hop Cypher traversal retriever
│   │   ├── hybrid_retriever.py  # Dual-track RRF fusion retriever
│   │   ├── reranker.py          # Cross-encoder & lexical reranking
│   │   └── answer_generator.py  # Structured synthesis with evidence provenance
│   ├── db/
│   │   ├── neo4j_client.py      # Connection pooling & health checks
│   │   └── queries.py           # Optimized Cypher queries
│   ├── services/
│   │   ├── rag_service.py       # High-level RAG orchestration
│   │   └── evaluation_service.py# Evaluation runner service
│   ├── config.py                # Environment configuration & settings
│   └── main.py                  # FastAPI application entrypoint
├── frontend/
│   ├── streamlit_app.py         # Multi-page research UI with dark mode theme
│   └── pages/
│       ├── overview.py          # System health, architecture & live ingestion controls
│       ├── ask.py               # Interactive multi-hop query runner & graph reasoning traces
│       ├── graph_explorer.py    # Interactive PyVis network visualization & subgraph explorer
│       ├── comparison.py        # Side-by-side RAG comparison (Vector vs Graph vs Hybrid)
│       └── experiments.py       # Benchmark evaluation suite, metrics & IEEE LaTeX export
├── evaluation/
│   ├── metrics.py               # Exact Match (EM), Token F1, Fact Recall, Latency stats
│   ├── run.py                   # CLI benchmark experiment runner
│   └── compare.py               # Aggregated experiment comparison & reporting
├── scripts/
│   ├── download_dataset.py      # HotpotQA & 2WikiMultiHopQA dataset downloader
│   ├── sample_dataset.py        # Reproducible sampling & synthetic dataset fallback
│   ├── validate_dataset.py      # Dataset schema & integrity validator
│   ├── build_index.py           # Dense vector index builder
│   ├── build_graph.py           # Knowledge graph construction runner
│   └── reset_database.py        # Database reset utility (requires --confirm)
├── tests/                       # Complete pytest unit test suite (60 test cases)
├── docker-compose.yml           # Multi-container orchestration (Neo4j + API + Frontend)
├── Dockerfile                   # Production container definition
├── requirements.txt             # Python dependencies
└── .env.example                 # Configuration template
```

---

## 🚀 Quickstart Guide

### 1. Environment Configuration

Clone the repository and copy the environment template:
```bash
cp .env.example .env
```

Key environment variables in `.env`:
- `NEO4J_URI=bolt://localhost:7687`
- `NEO4J_USER=neo4j`
- `NEO4J_PASSWORD=graphrag_password`
- `LLM_PROVIDER=ollama` (or `openai`, `anthropic`, `groq`)
- `LLM_MODEL=mistral` (or `gpt-4o-mini`, `claude-3-haiku-20240307`)
- `EMBEDDING_MODEL=all-MiniLM-L6-v2`

---

### 2. Launching with Docker Compose (Recommended)

Start the Neo4j graph database, backend FastAPI service, and Streamlit frontend in one command:
```bash
docker-compose up -d
```
- **Streamlit Web UI:** `http://localhost:8501`
- **FastAPI Documentation:** `http://localhost:8000/docs`
- **Neo4j Browser:** `http://localhost:7474`

---

### 3. Running Locally (Development Mode)

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Start Neo4j Database
Run Neo4j via Docker:
```bash
docker run -d --name neo4j-graphrag -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/graphrag_password neo4j:5.20-community
```

#### Step 3: Sample Benchmark Dataset & Build Index
```bash
# 1. Download/sample HotpotQA dataset (50 samples default)
python scripts/sample_dataset.py --dataset hotpotqa --size 50

# 2. Build vector embeddings
python scripts/build_index.py --dataset hotpotqa --size 50

# 3. Build knowledge graph
python scripts/build_graph.py
```

#### Step 4: Start Backend API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Step 5: Start Streamlit Frontend
```bash
streamlit run frontend/streamlit_app.py
```

---

## 🧪 Reproducible Research Evaluation

Run automated evaluation experiments across all three paradigms (**Vector-Only**, **Graph-Only**, and **Hybrid GraphRAG**):

```bash
# Run benchmark on multi-hop evaluation set
python -m evaluation.run --systems vector,graph,hybrid --max-samples 50

# Compare experiment runs and generate summaries
python -m evaluation.compare
```

### Metrics Tracked:
- **Exact Match (EM %):** Strict normalized text matching against ground-truth answers.
- **Token F1 (%):** Precision and Recall harmonic mean on word tokens.
- **Supporting-Fact Recall (%):** Percentage of required gold provenance facts successfully retrieved.
- **Latency (p50 / p95):** Latency percentiles in seconds.

---

## 🔬 Interactive Streamlit Research Interface

The web interface provides 5 specialized modules:
1. **Overview & Ingestion:** Live graph stats, architecture breakdown, and dataset loading triggers.
2. **Ask Questions:** Interactive multi-hop queries with sub-query breakdowns, reasoning graph edges, and evidence attribution.
3. **Graph Explorer:** Dynamic PyVis network visualization with physics layout, node coloring by type, and neighborhood exploration.
4. **RAG Comparison:** Side-by-side comparison of Vector-Only, Graph-Only, and Hybrid GraphRAG on identical prompts.
5. **Experiment Logs & IEEE Export:** Run benchmarks, view Plotly comparison charts, and copy pre-formatted LaTeX tables for IEEE paper manuscripts.

---

## 🧪 Running Unit Tests

Run the full unit test suite:
```bash
python -m pytest tests -v
```

All 60 tests validate chunking stability, entity extraction schemas, fuzzy entity resolution, and evaluation metrics.
