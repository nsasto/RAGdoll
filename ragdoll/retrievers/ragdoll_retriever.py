from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

try:  # pragma: no cover - compatibility across Pydantic versions
    from pydantic import ConfigDict  # type: ignore
except ImportError:  # pragma: no cover
    ConfigDict = None  # type: ignore


def _score_from_metadata(doc: Document, key: str, default: float = 0.0) -> float:
    """
    Best-effort float conversion of a score stored in metadata.

    Falls back to the provided default if the key is missing or unparsable.
    """
    try:
        return float((doc.metadata or {}).get(key, default))
    except Exception:
        return default


class RagdollRetriever(BaseRetriever):
    """
    Lightweight hybrid retriever that merges vector hits with graph nodes.

    Design goals:
    - Fast by default (small k, shallow hop expansion, no LLM-in-the-loop)
    - Compatible with LangChain's BaseRetriever interface
    - Works with in-memory simple graph retriever or Neo4j retriever seeds
    """

    if ConfigDict is not None:  # pragma: no cover - exercised via imports in tests
        model_config = ConfigDict(
            arbitrary_types_allowed=True, extra="allow"
        )  # type: ignore
    else:  # pragma: no cover - legacy pydantic fallback
        class Config:
            arbitrary_types_allowed = True
            extra = "allow"

    def __init__(
        self,
        *,
        vector_store: Any,
        graph_retriever: Any = None,
        mode: str = "hybrid",  # "vector" | "graph" | "hybrid"
        top_k_vector: int = 5,
        top_k_graph: int = 5,
        graph_hops: int = 1,
        include_edges: bool = True,
        weight_vector: float = 1.0,
        weight_graph: float = 1.0,
        max_results: int = 10,
    ) -> None:
        super().__init__()
        self.vector_store = vector_store
        self.graph_retriever = graph_retriever
        self.mode = (mode or "hybrid").lower()
        self.top_k_vector = max(1, top_k_vector)
        self.top_k_graph = max(1, top_k_graph)
        self.graph_hops = max(0, graph_hops)
        self.include_edges = include_edges
        self.weight_vector = weight_vector
        self.weight_graph = weight_graph
        self.max_results = max_results

    # ------------------------------------------------------------------ #
    # BaseRetriever interface
    # ------------------------------------------------------------------ #
    def get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)

    async def aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        mode = (kwargs.get("mode") or self.mode or "hybrid").lower()
        top_k_vector = int(kwargs.get("top_k_vector", kwargs.get("top_k", self.top_k_vector)))
        top_k_graph = int(kwargs.get("top_k_graph", self.top_k_graph))
        vector_docs: List[Document] = []
        graph_docs: List[Document] = []

        # Vector search
        if mode in {"vector", "hybrid"} and self.vector_store:
            vector_docs = self.vector_store.similarity_search(
                query, k=top_k_vector
            )
            for doc in vector_docs:
                doc.metadata = {**(doc.metadata or {}), "source_kind": "vector"}

        # Graph seeds
        if mode in {"graph", "hybrid"} and self.graph_retriever:
            graph_docs = self.graph_retriever.get_relevant_documents(query)
            graph_docs = graph_docs[:top_k_graph]
            for doc in graph_docs:
                md = doc.metadata or {}
                md.update({"source_kind": "graph", "hop": 0})
                doc.metadata = md

            graph_docs = self._expand_hops(graph_docs)

        merged = self._merge(vector_docs, graph_docs)
        merged.sort(key=lambda d: d.metadata.get("score_total", 0.0), reverse=True)
        if self.max_results > 0:
            return merged[: self.max_results]
        return merged

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _expand_hops(self, seeds: List[Document]) -> List[Document]:
        """
        Perform bounded hop expansion for in-memory graphs.

        This only runs when the underlying graph retriever exposes a `graph`
        with `nodes` and `edges` attributes (e.g., SimpleGraphRetriever).
        """
        if self.graph_hops <= 0:
            return seeds

        graph = getattr(self.graph_retriever, "graph", None)
        if graph is None or not hasattr(graph, "edges") or not hasattr(graph, "nodes"):
            return seeds

        id_to_doc: Dict[str, Document] = {}
        for doc in seeds:
            node_id = (doc.metadata or {}).get("node_id")
            if node_id:
                id_to_doc[node_id] = doc

        for hop in range(1, self.graph_hops + 1):
            new_docs: Dict[str, Document] = {}
            for edge in graph.edges:
                for src, tgt in ((edge.source, edge.target), (edge.target, edge.source)):
                    if src not in id_to_doc:
                        continue
                    if tgt in id_to_doc:
                        continue
                    node = next((n for n in graph.nodes if n.id == tgt), None)
                    if not node:
                        continue
                    md = dict(node.metadata or {})
                    if self.include_edges:
                        md["connected_to"] = self._connected_ids(graph, tgt)
                    md.setdefault("node_id", node.id)
                    md.setdefault("node_type", node.type)
                    md["source_kind"] = "graph"
                    md["hop"] = hop
                    md["score"] = 1.0 / (hop + 1)  # simple decay
                    new_docs[tgt] = Document(page_content=node.name or "", metadata=md)

            if not new_docs:
                break
            id_to_doc.update(new_docs)

        return list(id_to_doc.values())

    @staticmethod
    def _connected_ids(graph: Any, node_id: str) -> List[str]:
        neighbors: List[str] = []
        for edge in graph.edges:
            if edge.source == node_id:
                neighbors.append(edge.target)
            elif edge.target == node_id:
                neighbors.append(edge.source)
        return neighbors

    def _merge(
        self, vector_docs: List[Document], graph_docs: List[Document]
    ) -> List[Document]:
        merged: Dict[str, Document] = {}

        def _add(doc: Document, origin: str) -> None:
            md = doc.metadata or {}
            key = (
                md.get("node_id")
                or md.get("source")
                or md.get("path")
                or md.get("id")
                or doc.page_content[:50]
            )

            score_v = _score_from_metadata(doc, "score") if origin == "vector" else 0.0
            score_g = _score_from_metadata(doc, "score") if origin == "graph" else 0.0
            score_total = self.weight_vector * score_v + self.weight_graph * score_g

            md.setdefault("score_vector", score_v if origin == "vector" else 0.0)
            md.setdefault("score_graph", score_g if origin == "graph" else 0.0)
            md["score_total"] = score_total
            doc.metadata = md

            current = merged.get(key)
            if current is None or score_total > current.metadata.get("score_total", 0.0):
                merged[key] = doc

        for doc in vector_docs:
            _add(doc, "vector")
        for doc in graph_docs:
            _add(doc, "graph")

        return list(merged.values())
