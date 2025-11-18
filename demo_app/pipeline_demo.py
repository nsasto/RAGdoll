from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

from ragdoll.chunkers import get_text_splitter, split_documents
from ragdoll.embeddings import FakeEmbeddings, get_embedding_model, OpenAIEmbeddings
from ragdoll.entity_extraction import EntityExtractionService
from ragdoll.entity_extraction.graph_persistence import GraphPersistenceService
from ragdoll.entity_extraction.models import Graph, GraphEdge, GraphNode
from ragdoll.ingestion import DocumentLoaderService
from ragdoll.llms import get_llm_caller
from ragdoll.llms.callers import BaseLLMCaller, call_llm_sync

from . import state

# Flag to switch to OpenAI embeddings instead of config-based resolution
USE_OPENAI_EMBEDDINGS = False

logger = logging.getLogger(__name__)


@dataclass
class StagePayload:
    documents: List[Document]
    chunks: List[Document]
    vector_hits: List[Document]
    graph: Graph
    stats: Dict[str, int]
    loader_items: List[Dict[str, str]]
    loader_logs: List[str]


async def run_ingestion_demo(
    *,
    sources: Sequence[str],
    extra_documents: Sequence[Document],
    augment: bool,
    loader_only: bool = False,
) -> StagePayload:
    """Execute the ingestion stages and collect artifacts for visualization.
    
    Args:
        sources: File paths, URLs, or other source identifiers
        extra_documents: Pre-created Document objects to include
        augment: If True, add to existing vector/graph stores. If False, reset them.
        loader_only: If True, only load documents (no chunking/embedding/graph). Don't touch stores.
    """

    print(f"run_ingestion_demo called with sources={sources}, extra_docs count={len(extra_documents)}, augment={augment}, loader_only={loader_only}")
    logger.info(f"run_ingestion_demo called with sources={sources}, extra_docs count={len(extra_documents)}, augment={augment}, loader_only={loader_only}")

    # Ensure directories exist but don't delete anything yet
    state.ensure_state_dirs()
    
    # Only reset stores if we're doing full ingestion AND not augmenting
    if not loader_only and not augment:
        # Reset vector/graph stores but keep uploads intact
        if state.VECTOR_DIR.exists():
            shutil.rmtree(state.VECTOR_DIR)
        if state.GRAPH_JSON.exists():
            state.GRAPH_JSON.unlink()
        if state.GRAPH_PICKLE.exists():
            state.GRAPH_PICKLE.unlink()
        state.ensure_state_dirs()

    loader = DocumentLoaderService(collect_metrics=True, use_cache=False)

    loaded_docs: List[Document] = []
    if sources:
        print(f"Calling loader.ingest_documents with {len(sources)} sources")
        logger.info(f"Calling loader.ingest_documents with {len(sources)} sources")
        raw_docs = loader.ingest_documents(list(sources))
        # raw_docs are already Document objects, not dicts
        for doc in raw_docs:
            if isinstance(doc, dict):
                # Handle dict format if returned
                loaded_docs.append(
                    Document(
                        page_content=doc.get("page_content", ""),
                        metadata=doc.get("metadata") or {},
                    )
                )
            else:
                # Already a Document object
                loaded_docs.append(doc)

    all_documents = loaded_docs + list(extra_documents)
    if not all_documents:
        fallback_docs: List[Document] = []
        for src in list(sources):
            path = Path(src)
            if path.exists() and path.is_file():
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = ""
                fallback_docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(path), "loader": "fallback_file_read"},
                    )
                )
        all_documents.extend(fallback_docs)

    if not all_documents:
        raise ValueError("Please provide at least one document, URL, or text snippet.")

    stats = {
        "document_count": len(all_documents),
        "chunk_count": 0,
        "vector_count": 0,
        "graph_nodes": 0,
        "graph_edges": 0,
    }

    loader_logs: List[str] = []
    loader_logs.append(f"Sources submitted: files/paths={len(sources)}, extra_docs={len(extra_documents)}")
    loader_logs.append(f"Documents loaded: {len(all_documents)}")
    loader_logs.append("Chunking/embeddings/graph steps are skipped in this loader-only view.")
    if loader.collect_metrics:
        loader_logs.append("Monitoring enabled for ingestion stage")
    else:
        loader_logs.append("Monitoring disabled for ingestion stage")

    return StagePayload(
        documents=all_documents,
        chunks=[],
        vector_hits=[],
        graph=Graph(nodes=[], edges=[]),
        stats=stats,
        loader_items=_build_loader_items(all_documents),
        loader_logs=loader_logs,
    )


def summarize_documents(docs: Sequence[Document], *, limit: int = 5) -> List[Dict]:
    summary = []
    for doc in list(docs)[:limit]:
        content = doc.page_content.strip().replace("\n", " ")
        if len(content) > 200:
            content = f"{content[:200]}..."
        summary.append({"content": content, "metadata": doc.metadata or {}})
    return summary


def _build_loader_items(docs: Sequence[Document], *, limit: int = 10) -> List[Dict[str, str]]:
    items = []
    for idx, doc in enumerate(list(docs)[:limit]):
        text = doc.page_content or ""
        preview = " ".join(text.strip().splitlines())[:320]
        source = doc.metadata.get("source") or f"source_{idx+1}"
        title = doc.metadata.get("title") or source
        items.append(
            {
                "id": f"doc-{idx}",
                "title": title,
                "source": source,
                "preview": preview,
                "content": text.strip(),
            }
        )
    return items


def summarize_graph_nodes(nodes: Sequence[GraphNode], limit: int = 5) -> List[Dict]:
    summary = []
    for node in list(nodes)[:limit]:
        summary.append(
            {
                "id": node.id,
                "name": node.name,
                "type": node.type,
                "metadata": node.metadata or {},
            }
        )
    return summary


def summarize_graph_edges(edges: Sequence[GraphEdge], limit: int = 5) -> List[Dict]:
    preview = []
    for edge in list(edges)[:limit]:
        preview.append(
            {
                "id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "type": edge.type,
            }
        )
    return preview


def _resolve_embedding_model():
    import os

    if USE_OPENAI_EMBEDDINGS:
        try:
            model = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
            logger.info("Using OpenAI embeddings as specified by flag")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise ValueError(
                f"OpenAI embeddings requested but failed to initialize: {e}"
            )

    try:
        model = get_embedding_model()
    except Exception as exc:
        logger.warning(
            "Falling back to FakeEmbeddings because get_embedding_model failed: %s", exc
        )
        model = None

    if model is None:
        model = FakeEmbeddings(size=1536)
        logger.info("Using fake embeddings (no embeddings configured)")

    if isinstance(model, FakeEmbeddings):
        logger.info("Pipeline is using fake embeddings")

    return model


def answer_question(question: str) -> Dict:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    embedding = _resolve_embedding_model()
    vector_store = state.load_vector_store(embedding)
    graph = state.load_graph()

    vector_docs: List[Document] = []
    if vector_store:
        vector_docs = vector_store.similarity_search(question, k=3)

    graph_docs: List[Document] = []
    if graph:
        retriever = GraphPersistenceService().create_retriever(
            graph=graph,
            backend="simple",
            top_k=3,
        )
        graph_docs = retriever.get_relevant_documents(question)

    llm_caller = get_llm_caller()
    answer_text: str
    used_llm = False

    if llm_caller and (vector_docs or graph_docs):
        context_lines = []
        for doc in vector_docs:
            context_lines.append(f"Vector doc: {doc.page_content[:200]}")
        for doc in graph_docs:
            context_lines.append(f"Graph node: {doc.page_content[:200]}")

        prompt = (
            "You are a helpful assistant answering questions about the ingested data.\n"
            f"Question: {question}\n"
            f"Context:\n{chr(10).join(context_lines)}\n"
            "Answer concisely using the context above."
        )
        try:
            answer_text = call_llm_sync(llm_caller, prompt)
            used_llm = True
        except Exception:
            answer_text = _fallback_answer(question, vector_docs, graph_docs)
    else:
        answer_text = _fallback_answer(question, vector_docs, graph_docs)

    return {
        "question": question,
        "answer": answer_text,
        "vector_docs": summarize_documents(vector_docs, limit=3),
        "graph_docs": summarize_documents(graph_docs, limit=3),
        "used_llm": used_llm,
    }


def _fallback_answer(
    question: str,
    vector_docs: Sequence[Document],
    graph_docs: Sequence[Document],
) -> str:
    lines = [f"Question: {question}", "No LLM configured, showing retrieved context."]
    if vector_docs:
        lines.append("Vector hits:")
        for doc in vector_docs:
            lines.append(f"- {doc.page_content[:200]}")
    if graph_docs:
        lines.append("Graph hits:")
        for doc in graph_docs:
            lines.append(f"- {doc.page_content[:200]}")
    if not vector_docs and not graph_docs:
        lines.append("No indexed data yet. Run ingestion first.")
    return "\n".join(lines)
