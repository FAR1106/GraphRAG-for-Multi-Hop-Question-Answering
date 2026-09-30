"""
frontend/pages/graph_explorer.py — Interactive Knowledge Graph Visualization & Subgraph Explorer.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import requests
import json
from pyvis.network import Network
import tempfile
import os

ENTITY_COLORS = {
    "PERSON": "#58a6ff",         # Blue
    "ORGANIZATION": "#bc8cff",   # Purple
    "LOCATION": "#3fb950",       # Green
    "EVENT": "#d29922",          # Amber
    "WORK_OF_ART": "#f85149",    # Red/Coral
    "CONCEPT": "#39d353",        # Teal
    "DATE": "#79c0ff",           # Light blue
    "OTHER": "#8b949e",          # Gray
}


def render(api_base: str):
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <h2 style="font-size: 1.8rem; font-weight: 700; color: #58a6ff; margin-bottom: 0.2rem;">
                🕸️ Interactive Knowledge Graph Explorer
            </h2>
            <p style="color: #8b949e; font-size: 0.95rem;">
                Inspect entities, traverse multi-hop relationship neighborhoods, and visualize the underlying reasoning graph.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------------------
    # Search & Exploration Controls
    # ---------------------------------------------------------------------------
    col_search, col_depth, col_limit = st.columns([3, 1, 1])

    with col_search:
        search_query = st.text_input("Search Entity by Name", placeholder="e.g. Tolkien, Python, Paris, Turing...")
    with col_depth:
        depth = st.selectbox("Traversal Depth", [1, 2], index=0)
    with col_limit:
        limit = st.slider("Node Limit", min_value=10, max_value=100, value=30, step=10)

    selected_entity_id = None

    if search_query.strip():
        try:
            r = requests.get(f"{api_base}/graph/search", params={"q": search_query.strip()}, timeout=5)
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results:
                    st.markdown(f"<div style='font-size:0.85rem; color:#8b949e; margin-bottom:0.5rem;'>Found {len(results)} matching entities:</div>", unsafe_allow_html=True)
                    
                    options = {f"{e.get('name')} ({e.get('type')}) [ID: {e.get('id')}]": e.get("id") for e in results}
                    choice = st.selectbox("Select Target Entity", list(options.keys()))
                    selected_entity_id = options[choice]
                else:
                    st.warning(f"No entities found matching '{search_query}'. Try a broader term or ingest more data.")
            else:
                st.error(f"Search error: {r.status_code}")
        except Exception as e:
            st.error(f"Failed to query graph: {e}")

    # ---------------------------------------------------------------------------
    # Subgraph Retrieval & PyVis Visualization
    # ---------------------------------------------------------------------------
    if selected_entity_id:
        st.markdown("---")
        with st.spinner("Fetching entity neighborhood and constructing graph layout..."):
            try:
                sub_res = requests.get(
                    f"{api_base}/graph/subgraph/{selected_entity_id}",
                    params={"depth": depth, "limit": limit},
                    timeout=10,
                )
                if sub_res.status_code == 200:
                    subgraph = sub_res.json()
                    nodes = subgraph.get("nodes", [])
                    edges = subgraph.get("edges", [])

                    # Legend
                    legend_html = "<div style='display:flex; flex-wrap:wrap; gap:10px; margin-bottom:12px;'>"
                    for k, v in ENTITY_COLORS.items():
                        legend_html += f"<span style='font-size:0.75rem; background:#161b22; border:1px solid #30363d; border-radius:12px; padding:2px 10px; color:{v}; font-weight:600;'>● {k}</span>"
                    legend_html += "</div>"
                    st.markdown(legend_html, unsafe_allow_html=True)

                    if nodes:
                        render_pyvis_network(nodes, edges, selected_entity_id)
                        
                        # Node & Edge Inspection Tables
                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            st.markdown('<div class="grag-section-header">Connected Entities</div>', unsafe_allow_html=True)
                            st.dataframe(
                                [{"Name": n.get("name"), "Type": n.get("type"), "ID": n.get("id")} for n in nodes],
                                use_container_width=True,
                                hide_index=True,
                            )
                        with col_t2:
                            st.markdown('<div class="grag-section-header">Relational Triples</div>', unsafe_allow_html=True)
                            st.dataframe(
                                [{"Source": e.get("source"), "Relation": e.get("type"), "Target": e.get("target")} for e in edges],
                                use_container_width=True,
                                hide_index=True,
                            )
                    else:
                        st.info("No connections found for this entity.")
                else:
                    st.error(f"Could not load subgraph: {sub_res.status_code}")
            except Exception as e:
                st.error(f"Visualization rendering error: {e}")
    else:
        # Default placeholder guidance
        st.info("💡 Type an entity name above (e.g., 'Tolkien', 'Oxford', 'Film') to explore its neighborhood in the knowledge graph.")


def render_pyvis_network(nodes: list[dict], edges: list[dict], focus_id: str):
    net = Network(
        height="500px",
        width="100%",
        bgcolor="#0d1117",
        font_color="#e6edf3",
        directed=True,
    )

    # Configure physics for smooth layout
    net.set_options(
        """
        {
          "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "font": { "size": 14, "face": "Inter", "color": "#e6edf3" }
          },
          "edges": {
            "color": { "color": "#484f58", "highlight": "#58a6ff" },
            "smooth": { "type": "continuous" },
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
            "font": { "size": 11, "face": "JetBrains Mono", "color": "#8b949e", "align": "middle" }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -4000,
              "centralGravity": 0.3,
              "springLength": 120,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "minVelocity": 0.75
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "tooltipDelay": 150
          }
        }
        """
    )

    added_nodes = set()
    for n in nodes:
        nid = str(n.get("id"))
        if nid not in added_nodes:
            ntype = n.get("type", "OTHER")
            color = ENTITY_COLORS.get(ntype, "#8b949e")
            is_focus = (nid == focus_id)
            
            net.add_node(
                nid,
                label=n.get("name", nid),
                title=f"Type: {ntype}\nID: {nid}",
                color={"background": "#1c2128", "border": color if not is_focus else "#f85149"},
                size=25 if is_focus else 18,
                shape="box" if is_focus else "dot",
            )
            added_nodes.add(nid)

    for e in edges:
        src = str(e.get("source"))
        tgt = str(e.get("target"))
        rel = e.get("type", "RELATED_TO")
        if src in added_nodes and tgt in added_nodes:
            net.add_edge(src, tgt, label=rel, title=rel)

    # Write to temporary file and embed as HTML component
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tf:
        net.save_graph(tf.name)
        temp_path = tf.name

    with open(temp_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    try:
        os.remove(temp_path)
    except Exception:
        pass

    components.html(html_content, height=520, scrolling=False)
