"""
frontend/pages/overview.py — System Overview and Pipeline Status.
"""
from __future__ import annotations

import streamlit as st
import requests
import pandas as pd
import plotly.express as px


def render(api_base: str):
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.2rem; font-weight: 800; color: #58a6ff; margin-bottom: 0.2rem;">
                🔬 GraphRAG: Multi-Hop Question Answering
            </h1>
            <p style="color: #8b949e; font-size: 1.05rem; line-height: 1.5;">
                Research-grade hybrid retrieval-augmented generation combining Knowledge Graphs with Vector Similarity Search for complex multi-step reasoning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------------------
    # System Status & Metrics Fetching
    # ---------------------------------------------------------------------------
    health_data = {}
    stats_data = {}
    api_online = False

    try:
        r_h = requests.get(f"{api_base}/health", timeout=2)
        if r_h.status_code == 200:
            health_data = r_h.json()
            api_online = True
    except Exception:
        pass

    if api_online:
        try:
            r_s = requests.get(f"{api_base}/stats", timeout=3)
            if r_s.status_code == 200:
                stats_data = r_s.json()
        except Exception:
            pass

    # Top Metric Banner
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        neo4j_ok = health_data.get('neo4j_connected', False)
        st.markdown(
            f"""
            <div class="grag-card">
                <div class="grag-card-title">Graph Engine</div>
                <div class="grag-card-value" style="font-size:1.15rem;">
                    <span class="badge {'badge-green' if neo4j_ok else 'badge-blue'}">
                        {'NEO4J ONLINE' if neo4j_ok else 'EMBEDDED KG ACTIVE'}
                    </span>
                </div>
                <div class="grag-card-sub">{'Connected Cluster' if neo4j_ok else 'In-Memory Store'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="grag-card">
                <div class="grag-card-title">Indexed Chunks</div>
                <div class="grag-card-value">{stats_data.get('chunk_count', 0):,}</div>
                <div class="grag-card-sub">Vector Embeddings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="grag-card">
                <div class="grag-card-title">Entities</div>
                <div class="grag-card-value">{stats_data.get('entity_count', 0):,}</div>
                <div class="grag-card-sub">Resolved KG Nodes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="grag-card">
                <div class="grag-card-title">Relations</div>
                <div class="grag-card-value">{stats_data.get('relationship_count', 0):,}</div>
                <div class="grag-card-sub">Typed KG Edges</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
            <div class="grag-card">
                <div class="grag-card-title">LLM Provider</div>
                <div class="grag-card-value" style="font-size:1.15rem; color:#bc8cff;">
                    {health_data.get('llm_provider', 'ollama').upper()}
                </div>
                <div class="grag-card-sub">{health_data.get('llm_model', 'mistral')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------------------------
    # Main Tabs: Architecture / Ingestion & Controls / Research Context
    # ---------------------------------------------------------------------------
    tab_arch, tab_ingest, tab_research = st.tabs([
        "🏗️ System Architecture",
        "⚡ Data Ingestion & Build",
        "📄 Research & Hypotheses",
    ])

    with tab_arch:
        st.markdown("### Hybrid GraphRAG Multi-Hop Architecture")
        st.markdown(
            """
            GraphRAG addresses the fundamental limitation of standard vector RAG in multi-hop question answering:
            **Vector search finds semantic neighbors in embedding space, but fails when required supporting facts are connected via transitive entities rather than topical similarity.**
            """
        )

        st.markdown(
            """
            ```
            +---------------------------------------------------------------------------------------------------+
            |                                  HYBRID GRAPHRAG RETRIEVAL PIPELINE                               |
            +---------------------------------------------------------------------------------------------------+
                                                                                                                
              [ User Question ]                                                                                 
                     │                                                                                          
                     ├──► [ Question Decomposer ] ──► Sub-queries: [Q1, Q2]                                    
                     │                                                                                          
                     ├──► [ Dense Vector Search ] ──► Chunks via Cosine Similarity (Top-K)                      
                     │                                         │                                                
                     └──► [ KG Multi-Hop Traversal ]            │                                                
                                 │                              │                                                
                          Extract Entities                      │                                                
                                 │                              │                                                
                          Traverse Relations (Depth 1..3)       │                                                
                                 │                              │                                                
                          Collect Reasoning Subgraph            │                                                
                                 │                              │                                                
                                 └───────────────┬──────────────┘                                                
                                                 ▼                                                              
                                     [ Reciprocal Rank Fusion ]                                                 
                                                 │                                                              
                                    [ Reranking & Deduplication ]                                               
                                                 │                                                              
                                    [ Evidence & Path Packaging ]                                               
                                                 │                                                              
                                      [ LLM Answer Generator ]                                                  
                                                 │                                                              
                                 [ Structured Answer with Provenance ]                                          
            ```
            """
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                """
                <div class="grag-card">
                    <div class="grag-section-header">1. Knowledge Graph Construction</div>
                    <ul style="color: #8b949e; font-size: 0.9rem; line-height: 1.6;">
                        <li><strong>Deterministic Chunking:</strong> Paragraph-level token preservation with SHA-256 stable chunk IDs.</li>
                        <li><strong>LLM Entity & Relation Extraction:</strong> Schema-constrained extraction with typed predicates.</li>
                        <li><strong>Entity Resolution:</strong> Exact & fuzzy deduplication with alias clustering and type-gating.</li>
                        <li><strong>Provenance Links:</strong> Explicit <code>MENTIONS</code> edges connecting chunks to graph entities.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                """
                <div class="grag-card">
                    <div class="grag-section-header">2. Multi-Hop Reasoning & Synthesis</div>
                    <ul style="color: #8b949e; font-size: 0.9rem; line-height: 1.6;">
                        <li><strong>Query Decomposition:</strong> Breaks complex questions into sequential reasoning steps.</li>
                        <li><strong>Dual-Track Retrieval:</strong> Combines dense vector similarity with graph multi-hop walks.</li>
                        <li><strong>Hybrid Fusion:</strong> Reciprocal Rank Fusion (RRF) prioritizes intersecting evidence.</li>
                        <li><strong>Transparent Attribution:</strong> Generates answer with exact supporting facts and graph reasoning paths.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_ingest:
        st.markdown("### Dataset Ingestion & Graph Indexing")
        st.markdown(
            "Load benchmarks (HotpotQA / 2WikiMultiHopQA), compute dense vector embeddings, and construct the Neo4j knowledge graph."
        )

        with st.form("ingest_form"):
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                dataset_choice = st.selectbox("Benchmark Dataset", ["hotpotqa", "2wikimultihopqa", "synthetic"], index=0)
            with col_i2:
                sample_count = st.slider("Sample Size (Questions)", min_value=10, max_value=500, value=50, step=10)
            with col_i3:
                do_extract = st.checkbox("Extract Knowledge Graph (LLM)", value=True, help="Extract entities, relationships, and build Neo4j graph nodes")

            submit_ingest = st.form_submit_button("🚀 Start Ingestion & Build", use_container_width=True)

        if submit_ingest:
            if not api_online:
                st.error("Backend API is offline. Please make sure FastAPI backend is running on `http://localhost:8000`.")
            else:
                with st.spinner(f"Ingesting {sample_count} samples from {dataset_choice} and building vector/graph index..."):
                    try:
                        payload = {
                            "dataset": dataset_choice,
                            "sample_size": sample_count,
                            "run_extraction": do_extract,
                        }
                        res = requests.post(f"{api_base}/ingest", json=payload, timeout=300)
                        if res.status_code == 200:
                            data = res.json()
                            st.success(f"✓ {data.get('message', 'Ingestion complete!')} ({data.get('num_chunks', 0)} chunks in {data.get('total_time_s', 0):.2f}s)")
                            st.json(data.get("details", {}))
                            st.rerun()
                        else:
                            st.error(f"Ingestion failed: {res.status_code} - {res.text}")
                    except Exception as e:
                        st.error(f"Request error: {e}")

    with tab_research:
        st.markdown("### Research Contributions & Target Metrics")
        st.markdown(
            """
            This implementation is structured as an experimental testbed for an IEEE paper on Multi-Hop Knowledge Graph RAG.
            
            #### 🎯 Evaluated Research Questions (RQ)
            1. **RQ1 (Accuracy):** Does hybrid GraphRAG achieve higher Exact Match (EM) and Token F1 on multi-hop questions requiring $\ge 2$ reasoning steps compared to vector-only baseline?
            2. **RQ2 (Attribution & Recall):** Does graph traversal improve Supporting-Fact Recall and reduce hallucination rates?
            3. **RQ3 (Efficiency Trade-off):** What is the latency and token overhead introduced by graph traversal vs precision gains?
            """
        )

        st.markdown("#### Baseline Comparisons")
        comp_df = pd.DataFrame([
            {"System": "Standard Vector RAG", "Retrieval Mechanism": "Dense bi-encoder cosine similarity", "Multi-Hop Capability": "Low (misses bridge entities)", "Explainability": "Black-box embeddings"},
            {"System": "Graph-Only Traversal", "Retrieval Mechanism": "Entity linking + Cypher multi-hop walk", "Multi-Hop Capability": "High (structured paths)", "Explainability": "Explicit graph paths"},
            {"System": "Hybrid GraphRAG (Ours)", "Retrieval Mechanism": "Vector + Graph Traversal + Reciprocal Rank Fusion", "Multi-Hop Capability": "State-of-the-Art", "Explainability": "Unified evidence & path provenance"},
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
