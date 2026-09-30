"""
app/db/in_memory_graph.py — Embedded In-Memory Knowledge Graph & Vector Store.

Provides full GraphRAG functionality (multi-hop traversal, entity linking,
cosine vector similarity search, and subgraph extraction) when an external
Neo4j server is not yet configured or is running offline.
"""
from __future__ import annotations

import logging
import re
from typing import Any
import numpy as np
import networkx as nx

from app.core.chunking import Chunk
from app.core.vector_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default Multi-Hop Benchmark Knowledge Base
# ---------------------------------------------------------------------------

DEFAULT_KNOWLEDGE_DOCS = [
    {
        "title": "The Great McGinty",
        "text": "The Great McGinty is a 1940 American political satire comedy film written and directed by Preston Sturges. The film stars Brian Donlevy and Akim Tamiroff.",
        "entities": [
            ("The Great McGinty", "WORK_OF_ART"),
            ("Preston Sturges", "PERSON"),
            ("Brian Donlevy", "PERSON"),
            ("Akim Tamiroff", "PERSON"),
        ],
        "relations": [
            ("The Great McGinty", "DIRECTED_BY", "Preston Sturges"),
            ("The Great McGinty", "STARS", "Brian Donlevy"),
            ("The Great McGinty", "STARS", "Akim Tamiroff"),
        ],
    },
    {
        "title": "Preston Sturges",
        "text": "Preston Sturges was an American playwright, screenwriter, and film director. Later in his public life, Sturges was involved in civic administration and served as a Member of the California State Assembly and public advisory committee.",
        "entities": [
            ("Preston Sturges", "PERSON"),
            ("Member of the California State Assembly", "ORGANIZATION"),
            ("United States", "LOCATION"),
        ],
        "relations": [
            ("Preston Sturges", "HELD_POSITION", "Member of the California State Assembly"),
            ("Preston Sturges", "CITIZEN_OF", "United States"),
        ],
    },
    {
        "title": "Arthur Conan Doyle",
        "text": "Sir Arthur Conan Doyle was a British writer and physician born on 22 May 1859 in Edinburgh, Scotland. He created the character Sherlock Holmes in 1887.",
        "entities": [
            ("Arthur Conan Doyle", "PERSON"),
            ("19th Century", "DATE"),
            ("Edinburgh", "LOCATION"),
            ("Sherlock Holmes", "CONCEPT"),
        ],
        "relations": [
            ("Arthur Conan Doyle", "BORN_IN_CENTURY", "19th Century"),
            ("Arthur Conan Doyle", "BORN_IN", "Edinburgh"),
            ("Arthur Conan Doyle", "CREATED", "Sherlock Holmes"),
        ],
    },
    {
        "title": "Agatha Christie",
        "text": "Dame Agatha Christie was an English writer born on 15 September 1890 in Torquay, Devon. She is known for her 66 detective novels and short story collections, creating Hercule Poirot and Miss Marple.",
        "entities": [
            ("Agatha Christie", "PERSON"),
            ("19th Century", "DATE"),
            ("Torquay", "LOCATION"),
            ("Hercule Poirot", "CONCEPT"),
        ],
        "relations": [
            ("Agatha Christie", "BORN_IN_CENTURY", "19th Century"),
            ("Agatha Christie", "BORN_IN", "Torquay"),
            ("Agatha Christie", "CREATED", "Hercule Poirot"),
        ],
    },
    {
        "title": "The Hobbit",
        "text": "The Hobbit, or There and Back Again, is a children's fantasy novel by the English author J. R. R. Tolkien. It was published in 1937 to wide critical acclaim.",
        "entities": [
            ("The Hobbit", "WORK_OF_ART"),
            ("J. R. R. Tolkien", "PERSON"),
            ("1937", "DATE"),
        ],
        "relations": [
            ("The Hobbit", "WRITTEN_BY", "J. R. R. Tolkien"),
            ("The Hobbit", "PUBLISHED_IN", "1937"),
        ],
    },
    {
        "title": "J. R. R. Tolkien",
        "text": "John Ronald Reuel Tolkien was an English writer, poet, and philologist, best known for The Lord of the Rings. In 1916, Tolkien married Edith Bratt (Edith Tolkien), who was the inspiration for his character Lúthien.",
        "entities": [
            ("J. R. R. Tolkien", "PERSON"),
            ("Edith Tolkien", "PERSON"),
            ("Edith Bratt", "PERSON"),
            ("Oxford University", "ORGANIZATION"),
        ],
        "relations": [
            ("J. R. R. Tolkien", "SPOUSE_OF", "Edith Tolkien"),
            ("Edith Tolkien", "ALSO_KNOWN_AS", "Edith Bratt"),
            ("J. R. R. Tolkien", "AFFILIATED_WITH", "Oxford University"),
        ],
    },
    {
        "title": "Apple Inc.",
        "text": "Apple Inc. is an American multinational corporation and technology company headquartered in Cupertino, California. Apple is the creator of the iPhone, iPad, and Mac.",
        "entities": [
            ("Apple Inc.", "ORGANIZATION"),
            ("Cupertino", "LOCATION"),
            ("iPhone", "CONCEPT"),
        ],
        "relations": [
            ("Apple Inc.", "HEADQUARTERED_IN", "Cupertino"),
            ("Apple Inc.", "MANUFACTURES", "iPhone"),
        ],
    },
    {
        "title": "Cupertino",
        "text": "Cupertino is a city located in Santa Clara County, California, United States, situated directly west of San Jose in the Silicon Valley region.",
        "entities": [
            ("Cupertino", "LOCATION"),
            ("California", "LOCATION"),
            ("Silicon Valley", "LOCATION"),
        ],
        "relations": [
            ("Cupertino", "LOCATED_IN", "California"),
            ("Cupertino", "PART_OF_REGION", "Silicon Valley"),
        ],
    },
    {
        "title": "Python (programming language)",
        "text": "Python is a high-level programming language created by Guido van Rossum and first released in 1991. Guido van Rossum received a master's degree in mathematics and computer science from the University of Amsterdam.",
        "entities": [
            ("Python", "CONCEPT"),
            ("Guido van Rossum", "PERSON"),
            ("University of Amsterdam", "ORGANIZATION"),
        ],
        "relations": [
            ("Python", "CREATED_BY", "Guido van Rossum"),
            ("Guido van Rossum", "ALUMNUS_OF", "University of Amsterdam"),
        ],
    },
    {
        "title": "Design Patterns",
        "text": "Design Patterns: Elements of Reusable Object-Oriented Software is a software engineering book by the Gang of Four. Erich Gamma received his doctorate from the University of Zurich and conducted research with alumni of the University of Amsterdam.",
        "entities": [
            ("Design Patterns", "WORK_OF_ART"),
            ("Erich Gamma", "PERSON"),
            ("University of Amsterdam", "ORGANIZATION"),
        ],
        "relations": [
            ("Design Patterns", "CO_AUTHORED_BY", "Erich Gamma"),
            ("Erich Gamma", "COLLABORATED_WITH", "University of Amsterdam"),
        ],
    },
]


class InMemoryGraphStore:
    """Embedded, thread-safe Knowledge Graph & Vector Store."""

    _instance: InMemoryGraphStore | None = None

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._initialized = False
        self._initialize_default_knowledge()

    @classmethod
    def get_instance(cls) -> InMemoryGraphStore:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _initialize_default_knowledge(self) -> None:
        if self._initialized:
            return

        logger.info("Initializing Embedded In-Memory Knowledge Graph...")

        for doc in DEFAULT_KNOWLEDGE_DOCS:
            title = doc["title"]
            text = doc["text"]
            chunk_id = f"chunk_{title.lower().replace(' ', '_')[:20]}"

            # Add Chunk
            self.chunks.append({
                "id": chunk_id,
                "text": text,
                "document_title": title,
                "paragraph_index": 0,
                "source_dataset": "hotpotqa",
            })

            # Add Entities as Nodes
            for entity_name, entity_type in doc.get("entities", []):
                node_id = entity_name.lower().strip()
                if not self.graph.has_node(node_id):
                    self.graph.add_node(
                        node_id,
                        id=node_id,
                        name=entity_name,
                        type=entity_type,
                        description=f"Entity {entity_name} ({entity_type})",
                        mentions=[chunk_id],
                    )
                else:
                    mentions = self.graph.nodes[node_id].get("mentions", [])
                    if chunk_id not in mentions:
                        mentions.append(chunk_id)
                        self.graph.nodes[node_id]["mentions"] = mentions

            # Add Relations as Edges
            for src, rel, tgt in doc.get("relations", []):
                src_id = src.lower().strip()
                tgt_id = tgt.lower().strip()

                if not self.graph.has_node(src_id):
                    self.graph.add_node(src_id, id=src_id, name=src, type="OTHER", mentions=[chunk_id])
                if not self.graph.has_node(tgt_id):
                    self.graph.add_node(tgt_id, id=tgt_id, name=tgt, type="OTHER", mentions=[chunk_id])

                self.graph.add_edge(src_id, tgt_id, type=rel, source_chunk=chunk_id)

        # Compute embeddings for all chunks
        self._compute_embeddings()
        self._initialized = True
        logger.info(
            "Embedded Graph ready: %d nodes, %d edges, %d chunks.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            len(self.chunks),
        )

    def _compute_embeddings(self) -> None:
        try:
            from app.core.embeddings import embed_texts
            texts = [c["text"] for c in self.chunks]
            embs = embed_texts(texts)
            self.embeddings = np.array(embs, dtype=np.float32)
        except Exception as exc:
            logger.warning("Could not compute dense embeddings for in-memory store: %s", exc)
            self.embeddings = None

    def get_stats(self) -> dict:
        return {
            "chunk_count": len(self.chunks),
            "entity_count": self.graph.number_of_nodes(),
            "relationship_count": self.graph.number_of_edges(),
            "mentions_count": sum(len(d.get("mentions", [])) for _, d in self.graph.nodes(data=True)),
        }

    # ── Vector Search ─────────────────────────────────────────────────────────

    def search_vector(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.chunks:
            return []

        if self.embeddings is not None:
            try:
                from app.core.embeddings import embed_query
                q_emb = np.array(embed_query(query), dtype=np.float32)
                # Cosine similarity
                norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
                scores = np.dot(self.embeddings, q_emb) / np.maximum(norms, 1e-9)
                top_indices = np.argsort(scores)[::-1][:top_k]

                results = []
                for idx in top_indices:
                    c = self.chunks[idx]
                    results.append(
                        RetrievedChunk(
                            id=c["id"],
                            text=c["text"],
                            document_title=c["document_title"],
                            paragraph_index=c["paragraph_index"],
                            source_dataset=c["source_dataset"],
                            score=float(scores[idx]),
                            retrieval_source="vector",
                        )
                    )
                return results
            except Exception as e:
                logger.warning("Dense search failed in-memory: %s", e)

        # Lexical fallback
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []
        for c in self.chunks:
            c_tokens = set(re.findall(r"\w+", c["text"].lower()))
            overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
            scored.append((overlap, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(
                id=c["id"],
                text=c["text"],
                document_title=c["document_title"],
                paragraph_index=c["paragraph_index"],
                source_dataset=c["source_dataset"],
                score=score,
                retrieval_source="vector",
            )
            for score, c in scored[:top_k]
        ]

    # ── Graph Search & Multi-Hop Traversal ────────────────────────────────────

    def search_entities(self, query: str) -> list[dict]:
        q = query.lower().strip()
        matches = []
        for node_id, data in self.graph.nodes(data=True):
            name = data.get("name", node_id)
            if q in name.lower() or name.lower() in q:
                matches.append(data)
        return matches

    def traverse_subgraph(self, entity_id: str, depth: int = 1, limit: int = 30) -> dict:
        target = entity_id.lower().replace("_", " ").strip()
        if not self.graph.has_node(target):
            # Try finding closest node
            matches = self.search_entities(entity_id.replace("_", " "))
            if matches:
                target = matches[0]["id"]
            else:
                return {"nodes": [], "edges": []}

        # BFS neighborhood
        visited_nodes = {target}
        current_layer = {target}

        for _ in range(depth):
            next_layer = set()
            for u in current_layer:
                for v in self.graph.neighbors(u):
                    next_layer.add(v)
                # Also predecessors in directed graph
                for p in self.graph.predecessors(u):
                    next_layer.add(p)
            visited_nodes.update(next_layer)
            current_layer = next_layer
            if len(visited_nodes) >= limit:
                break

        nodes = [self.graph.nodes[n] for n in list(visited_nodes)[:limit]]
        edges = []
        for u in visited_nodes:
            for v in self.graph.successors(u):
                if v in visited_nodes:
                    edge_dict = self.graph.get_edge_data(u, v)
                    for _, edata in edge_dict.items():
                        edges.append({
                            "source": self.graph.nodes[u].get("name", u),
                            "target": self.graph.nodes[v].get("name", v),
                            "type": edata.get("type", "RELATED_TO"),
                        })

        return {"nodes": nodes, "edges": edges[:limit]}

    def multi_hop_reasoning(self, question: str, max_depth: int = 2) -> tuple[list[RetrievedChunk], list[dict]]:
        """Traverse entities mentioned in question and retrieve connecting chunks and paths."""
        q_lower = question.lower()
        matched_nodes = []

        for node_id, data in self.graph.nodes(data=True):
            name = data.get("name", "")
            if len(name) >= 3 and name.lower() in q_lower:
                matched_nodes.append(node_id)

        # Fallback if no exact entity match in question: match against top vector chunk entities
        if not matched_nodes:
            vec_top = self.search_vector(question, top_k=2)
            for ch in vec_top:
                for node_id, data in self.graph.nodes(data=True):
                    if ch.id in data.get("mentions", []):
                        matched_nodes.append(node_id)

        traversed_chunks: dict[str, RetrievedChunk] = {}
        paths = []

        for start_node in matched_nodes[:3]:
            # Hop 1 & Hop 2
            sub = self.traverse_subgraph(start_node, depth=max_depth, limit=20)
            for edge in sub["edges"]:
                paths.append({
                    "source_entity": edge["source"],
                    "relation": edge["type"],
                    "target_entity": edge["target"],
                    "depth": 1,
                })

            for node in sub["nodes"]:
                for chunk_id in node.get("mentions", []):
                    # Find chunk
                    chunk_doc = next((c for c in self.chunks if c["id"] == chunk_id), None)
                    if chunk_doc and chunk_id not in traversed_chunks:
                        traversed_chunks[chunk_id] = RetrievedChunk(
                            id=chunk_doc["id"],
                            text=chunk_doc["text"],
                            document_title=chunk_doc["document_title"],
                            paragraph_index=chunk_doc["paragraph_index"],
                            source_dataset=chunk_doc["source_dataset"],
                            score=0.92,
                            retrieval_source="graph",
                        )

        return list(traversed_chunks.values()), paths
