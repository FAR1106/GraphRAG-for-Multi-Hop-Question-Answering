"""
frontend/pages/ask.py — Multi-Hop Question Answering & Reasoning Trace.
"""
from __future__ import annotations

import streamlit as st
import requests
import json
import time

PRESET_QUESTIONS = [
    {
        "label": "HotpotQA · Director & Political Office (Multi-Hop)",
        "question": "What government position was held by the director of the film 'The Great McGinty'?",
        "expected_answer": "Member of the California State Assembly / Politician (Preston Sturges)",
    },
    {
        "label": "HotpotQA · Birth Year Comparison (Multi-Hop)",
        "question": "Were Arthur Conan Doyle and Agatha Christie born in the same century?",
        "expected_answer": "Yes (both 19th century: 1859 and 1890)",
    },
    {
        "label": "HotpotQA · Shared Location / Institution (Multi-Hop)",
        "question": "Which university did both the creator of Python and the author of Design Patterns attend?",
        "expected_answer": "University of Amsterdam / relevant institution",
    },
    {
        "label": "HotpotQA · Bridge Entity Reasoning (Multi-Hop)",
        "question": "Who was the spouse of the author who wrote 'The Hobbit'?",
        "expected_answer": "Edith Tolkien (Edith Bratt)",
    },
]


def render(api_base: str):
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <h2 style="font-size: 1.8rem; font-weight: 700; color: #58a6ff; margin-bottom: 0.2rem;">
                💬 Ask Questions · Multi-Hop Reasoning
            </h2>
            <p style="color: #8b949e; font-size: 0.95rem;">
                Test multi-hop retrieval and inspect the full reasoning trace, graph traversal paths, and supporting evidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------------------
    # Presets & Input
    # ---------------------------------------------------------------------------
    col_preset, col_method = st.columns([2, 1])

    with col_preset:
        preset_options = ["-- Custom Question --"] + [p["label"] for p in PRESET_QUESTIONS]
        selected_preset = st.selectbox("Load Example Multi-Hop Question", preset_options, index=1)

    with col_method:
        method = st.selectbox(
            "Retrieval Method",
            ["hybrid", "vector", "graph"],
            index=0,
            format_func=lambda x: {
                "hybrid": "🔬 Hybrid GraphRAG (Vector + KG)",
                "vector": "⚡ Vector Only (Cosine Sim)",
                "graph": "🕸️ Graph Only (KG Traversal)",
            }[x],
        )

    # Initial value from preset if chosen
    default_q = ""
    default_expected = ""
    if selected_preset != "-- Custom Question --":
        preset_item = next(p for p in PRESET_QUESTIONS if p["label"] == selected_preset)
        default_q = preset_item["question"]
        default_expected = preset_item["expected_answer"]

    question_input = st.text_area(
        "Question",
        value=default_q,
        height=75,
        placeholder="Enter a multi-hop question requiring information from multiple connected sources...",
    )

    # ---------------------------------------------------------------------------
    # Advanced Retrieval Hyperparameters
    # ---------------------------------------------------------------------------
    with st.expander("⚙️ Pipeline Hyperparameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            vector_top_k = st.slider("Vector Top-K Candidates", min_value=1, max_value=20, value=5)
        with c2:
            graph_max_depth = st.slider("Graph Traversal Depth", min_value=1, max_value=3, value=2)
        with c3:
            final_context_size = st.slider("Final Context Chunk Limit", min_value=1, max_value=10, value=5)

    ask_button = st.button("🚀 Run Multi-Hop Query", use_container_width=True)

    # ---------------------------------------------------------------------------
    # Query Execution & Results Rendering
    # ---------------------------------------------------------------------------
    if ask_button and question_input.strip():
        payload = {
            "question": question_input.strip(),
            "method": method,
            "vector_top_k": vector_top_k,
            "graph_max_depth": graph_max_depth,
            "final_context_size": final_context_size,
        }

        with st.spinner(f"Executing {method.upper()} multi-hop retrieval and synthesis..."):
            try:
                start_time = time.time()
                res = requests.post(f"{api_base}/ask", json=payload, timeout=60)
                client_duration = time.time() - start_time

                if res.status_code == 200:
                    data = res.json()
                    render_results(data, client_duration, default_expected)
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Failed to communicate with GraphRAG API: {e}")


def render_results(data: dict, total_latency: float, expected_answer: str = ""):
    st.markdown("---")

    # 1. Answer Box
    st.markdown('<div class="grag-section-header">Generated Synthesis</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="answer-box">
            <div style="font-size:1.15rem; font-weight:600; color:#e6edf3; margin-bottom:0.5rem;">
                {data.get('answer', 'No answer generated.')}
            </div>
            <div style="font-size:0.75rem; color:#8b949e;">
                Method: <code>{data.get('method', 'unknown')}</code> · 
                Total Latency: <strong>{data.get('total_latency_s', total_latency):.3f}s</strong> · 
                Context Chunks: <strong>{len(data.get('retrieved_chunks', []))}</strong> · 
                KG Triples: <strong>{len(data.get('graph_paths', []))}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if expected_answer:
        st.markdown(
            f"""
            <div style="background: rgba(88,166,255,0.05); border: 1px dashed rgba(88,166,255,0.3); border-radius:6px; padding:8px 12px; margin-bottom:1rem; font-size:0.85rem; color:#8b949e;">
                🎯 <strong>Ground Truth Reference:</strong> <span style="color:#58a6ff;">{expected_answer}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Reasoning & Decomposition Trace
    col_dec, col_graph = st.columns(2)

    with col_dec:
        st.markdown('<div class="grag-section-header">🔍 Sub-Query Decomposition</div>', unsafe_allow_html=True)
        sub_queries = data.get("sub_queries", [])
        if sub_queries:
            for idx, sq in enumerate(sub_queries, 1):
                st.markdown(
                    f"""
                    <div style="background: #161b22; border: 1px solid #30363d; border-radius:6px; padding:8px 12px; margin-bottom:6px; font-size:0.85rem;">
                        <span style="color:#bc8cff; font-weight:600;">Hop {idx}:</span> {sq}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div style='color:#6e7681; font-size:0.85rem;'>Direct query execution (single-hop / no sub-queries generated).</div>", unsafe_allow_html=True)

    with col_graph:
        st.markdown('<div class="grag-section-header">🕸️ KG Reasoning Traversal</div>', unsafe_allow_html=True)
        graph_paths = data.get("graph_paths", [])
        if graph_paths:
            for path in graph_paths:
                st.markdown(
                    f"""
                    <div class="triple-card">
                        <span style="color:#58a6ff; font-weight:600;">{path.get('source_entity', 'Node')}</span>
                        <span style="color:#bc8cff;"> ──[{path.get('relation', 'REL')}]──► </span>
                        <span style="color:#3fb950; font-weight:600;">{path.get('target_entity', 'Node')}</span>
                        {f'<span style="color:#6e7681; font-size:0.75rem;"> (Hop {path.get("depth", 1)})</span>' if "depth" in path else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<div style='color:#6e7681; font-size:0.85rem;'>No multi-hop graph edges traversed for this query method.</div>", unsafe_allow_html=True)

    # 3. Retrieved Supporting Evidence Chunks
    st.markdown('<div class="grag-section-header">📚 Retrieved Context & Evidence Provenance</div>', unsafe_allow_html=True)
    chunks = data.get("retrieved_chunks", [])
    if chunks:
        for idx, chunk in enumerate(chunks, 1):
            source = chunk.get("source_dataset", chunk.get("chunk_id", f"Chunk-{idx}"))
            score = chunk.get("score")
            score_str = f"Score: {score:.4f}" if score is not None else "KG Retrieved"

            with st.expander(f"Evidence #{idx} · {source} ({score_str})", expanded=(idx <= 2)):
                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div class="evidence-source">ID: {chunk.get('chunk_id', 'N/A')}</div>
                        <div style="line-height:1.5;">{chunk.get('text', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No supporting evidence chunks retrieved.")
