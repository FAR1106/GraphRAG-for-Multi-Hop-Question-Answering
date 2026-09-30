"""
frontend/pages/comparison.py — Side-by-Side RAG Paradigm Comparison.
"""
from __future__ import annotations

import streamlit as st
import requests
import time
import pandas as pd
import plotly.graph_objects as go


def render(api_base: str):
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <h2 style="font-size: 1.8rem; font-weight: 700; color: #58a6ff; margin-bottom: 0.2rem;">
                ⚖️ RAG Paradigm Comparison
            </h2>
            <p style="color: #8b949e; font-size: 0.95rem;">
                Compare <strong>Vector-Only RAG</strong>, <strong>Graph-Only Traversal</strong>, and <strong>Hybrid GraphRAG</strong> head-to-head on the exact same multi-hop query.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    compare_presets = [
        "What government position was held by the director of the film 'The Great McGinty'?",
        "Were Arthur Conan Doyle and Agatha Christie born in the same century?",
        "Who was the spouse of the author who wrote 'The Hobbit'?",
        "Which organization was founded by the mentor of the developer of the Linux kernel?",
    ]

    selected_query = st.selectbox("Select or enter a multi-hop test query:", compare_presets)
    custom_q = st.text_input("Or custom test prompt:", value=selected_query)

    run_comp = st.button("⚔️ Run Side-by-Side Comparison", use_container_width=True)

    if run_comp and custom_q.strip():
        q = custom_q.strip()
        methods = ["vector", "graph", "hybrid"]
        results = {}

        progress_bar = st.progress(0, text="Benchmarking retrieval methods...")

        for idx, m in enumerate(methods):
            progress_bar.progress((idx) / len(methods), text=f"Executing {m.upper()} retrieval...")
            try:
                t0 = time.time()
                r = requests.post(f"{api_base}/ask", json={"question": q, "method": m}, timeout=60)
                dur = time.time() - t0
                if r.status_code == 200:
                    data = r.json()
                    data["client_latency"] = dur
                    results[m] = data
                else:
                    results[m] = {"answer": f"Error {r.status_code}: {r.text}", "retrieved_chunks": [], "graph_paths": [], "client_latency": dur}
            except Exception as e:
                results[m] = {"answer": f"Request failed: {e}", "retrieved_chunks": [], "graph_paths": [], "client_latency": 0.0}

        progress_bar.progress(1.0, text="Comparison complete!")
        time.sleep(0.3)
        progress_bar.empty()

        # ---------------------------------------------------------------------------
        # Side-by-Side Cards
        # ---------------------------------------------------------------------------
        col_v, col_g, col_h = st.columns(3)

        with col_v:
            render_method_card(
                "⚡ Vector-Only RAG",
                "Pure cosine similarity on chunk embeddings",
                results.get("vector", {}),
                color="#58a6ff",
            )

        with col_g:
            render_method_card(
                "🕸️ Graph-Only Traversal",
                "Entity extraction + Cypher graph walk",
                results.get("graph", {}),
                color="#bc8cff",
            )

        with col_h:
            render_method_card(
                "🔬 Hybrid GraphRAG (Ours)",
                "Vector + Graph Traversal + Reciprocal Rank Fusion",
                results.get("hybrid", {}),
                color="#3fb950",
            )

        # ---------------------------------------------------------------------------
        # Quantitative Metric Comparison Charts
        # ---------------------------------------------------------------------------
        st.markdown("---")
        st.markdown('<div class="grag-section-header">Performance Metrics Breakdown</div>', unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            # Latency Bar Chart
            latencies = [
                results.get("vector", {}).get("total_latency_s", results.get("vector", {}).get("client_latency", 0)),
                results.get("graph", {}).get("total_latency_s", results.get("graph", {}).get("client_latency", 0)),
                results.get("hybrid", {}).get("total_latency_s", results.get("hybrid", {}).get("client_latency", 0)),
            ]
            fig_lat = go.Figure(
                data=[
                    go.Bar(
                        x=["Vector", "Graph", "Hybrid GraphRAG"],
                        y=latencies,
                        marker_color=["#58a6ff", "#bc8cff", "#3fb950"],
                        text=[f"{v:.3f}s" for v in latencies],
                        textposition="auto",
                    )
                ]
            )
            fig_lat.update_layout(
                title="Response Latency (seconds, lower is better)",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font={"color": "#e6edf3"},
                margin=dict(l=20, r=20, t=40, b=20),
                height=300,
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        with col_c2:
            # Evidence & Paths Count Chart
            chunk_counts = [
                len(results.get("vector", {}).get("retrieved_chunks", [])),
                len(results.get("graph", {}).get("retrieved_chunks", [])),
                len(results.get("hybrid", {}).get("retrieved_chunks", [])),
            ]
            edge_counts = [
                len(results.get("vector", {}).get("graph_paths", [])),
                len(results.get("graph", {}).get("graph_paths", [])),
                len(results.get("hybrid", {}).get("graph_paths", [])),
            ]

            fig_ev = go.Figure(
                data=[
                    go.Bar(name="Context Chunks", x=["Vector", "Graph", "Hybrid"], y=chunk_counts, marker_color="#58a6ff"),
                    go.Bar(name="KG Triples Traversed", x=["Vector", "Graph", "Hybrid"], y=edge_counts, marker_color="#bc8cff"),
                ]
            )
            fig_ev.update_layout(
                title="Retrieved Evidence Units (Chunks & Triples)",
                barmode="group",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font={"color": "#e6edf3"},
                margin=dict(l=20, r=20, t=40, b=20),
                height=300,
            )
            st.plotly_chart(fig_ev, use_container_width=True)


def render_method_card(title: str, subtitle: str, data: dict, color: str):
    answer = data.get("answer", "No response.")
    chunks = data.get("retrieved_chunks", [])
    paths = data.get("graph_paths", [])
    lat = data.get("total_latency_s", data.get("client_latency", 0.0))

    st.markdown(
        f"""
        <div style="background: #161b22; border: 1px solid #30363d; border-top: 4px solid {color}; border-radius: 8px; padding: 1rem; min-height: 400px;">
            <div style="font-weight: 700; font-size: 1.05rem; color: {color};">{title}</div>
            <div style="font-size: 0.75rem; color: #8b949e; margin-bottom: 0.8rem;">{subtitle}</div>
            
            <div style="font-size: 0.75rem; color: #6e7681; margin-bottom: 4px;">SYNTHESIZED ANSWER:</div>
            <div style="font-size: 0.9rem; color: #e6edf3; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 10px; margin-bottom: 0.8rem; line-height: 1.45;">
                {answer}
            </div>
            
            <div style="display:flex; justify-content:space-between; font-size: 0.75rem; color: #8b949e; border-top: 1px solid #21262d; padding-top: 6px;">
                <span>Latency: <strong style="color:#e6edf3;">{lat:.3f}s</strong></span>
                <span>Chunks: <strong style="color:#e6edf3;">{len(chunks)}</strong></span>
                <span>Triples: <strong style="color:#e6edf3;">{len(paths)}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
