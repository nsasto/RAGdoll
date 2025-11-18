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
from ragdoll.vector_stores import vector_store_from_config

from . import state
from .config_state import get_app_config
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
    source_filename_map: dict[str, str] = None,
) -> StagePayload:
    """
    Run the full ingestion pipeline or just the loader step.
    
    Args:
        sources: File paths or URLs to load
        extra_documents: Additional documents to process
        augment: If False and not loader_only, resets vector/graph stores
        loader_only: If True, only runs document loading (no chunking/embeddings/stores)
        source_filename_map: Maps internal paths to original filenames for display
    """
    import time
    from langchain_community.vectorstores import FAISS

    logger.info(
        f"run_ingestion_demo called with sources={sources}, extra_docs count={len(extra_documents)}, "
        f"augment={augment}, loader_only={loader_only}"
    )

    app_config = get_app_config()

    if not augment and not loader_only:
        logger.info("Resetting state directory (augment=False)")
        state.reset_state_dir()
        state.ensure_state_dirs()

    loader = DocumentLoaderService(
        app_config=app_config, use_cache=False, collect_metrics=False
    )

    def _to_document(entry: object) -> Document:
        if isinstance(entry, Document):
            doc = entry
        elif isinstance(entry, dict):
            doc = Document(
                page_content=str(entry.get("page_content", "")),
                metadata=entry.get("metadata", {}) or {},
            )
        else:
            doc = Document(page_content=str(entry), metadata={})

        if source_filename_map:
            source = doc.metadata.get("source")
            if source and source in source_filename_map:
                doc.metadata.setdefault(
                    "original_filename", source_filename_map[source]
                )
        return doc
    
    # Track loading stats
    load_start = time.time()
    logger.info(f"Calling loader.ingest_documents with {len(sources)} sources")
    loaded_docs = loader.ingest_documents(sources)
    load_duration = time.time() - load_start
    
    all_documents = [_to_document(doc) for doc in list(loaded_docs) + list(extra_documents)]

    if not all_documents:
        raise ValueError("No valid sources found")

    # Calculate comprehensive stats
    doc_types = {}
    total_chars = 0
    total_words = 0
    total_pages = 0
    total_file_size = 0
    
    for doc in all_documents:
        source = doc.metadata.get("source", "unknown")
        if source.startswith("http"):
            doc_type = "URL"
        elif source == "manual_input":
            doc_type = "Text Input"
        else:
            ext = Path(source).suffix.lower()
            doc_type = ext[1:].upper() if ext else "Unknown"
        
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        total_chars += len(doc.page_content)
        total_words += len(doc.page_content.split())
        
        # Count pages
        page = doc.metadata.get("page")
        if page is not None:
            total_pages = max(total_pages, int(page) + 1)
    
    # Calculate file sizes
    for source in sources:
        if not source.startswith("http") and source != "manual_input":
            try:
                total_file_size += Path(source).stat().st_size
            except Exception:
                pass
    
    size_mb = total_file_size / (1024 * 1024) if total_file_size > 0 else 0
    processing_speed = (size_mb / load_duration) if load_duration > 0 and size_mb > 0 else 0
    
    stats = {
        "document_count": len(all_documents),
        "chunk_count": 0,
        "vector_count": 0,
        "graph_nodes": 0,
        "graph_edges": 0,
        "documents_loaded": len(loaded_docs),
        "sources_submitted": len(sources),
        "extra_documents": len(extra_documents),
        "load_duration_seconds": round(load_duration, 2),
        "total_characters": total_chars,
        "total_words": total_words,
        "total_pages": total_pages,
        "total_file_size_mb": round(size_mb, 2),
        "processing_speed_mbps": round(processing_speed, 2),
    }

    loader_logs = []
    loader_logs.append(f"Sources submitted: files/paths={len([s for s in sources if not s.startswith('http')])}, urls={len([s for s in sources if s.startswith('http')])}, extra_docs={len(extra_documents)}")
    loader_logs.append(f"Documents loaded: {len(loaded_docs)}")
    loader_logs.append(f"Total documents: {len(all_documents)}")
    loader_logs.append(f"Load duration: {load_duration:.2f}s")
    loader_logs.append(f"Avg time per document: {(load_duration / len(all_documents)):.3f}s" if all_documents else "Avg time: N/A")
    loader_logs.append(f"Total characters: {total_chars:,}")
    loader_logs.append(f"Total words: {total_words:,}")
    if total_pages > 0:
        loader_logs.append(f"Total pages: {total_pages}")
    if size_mb > 0:
        loader_logs.append(f"Total file size: {size_mb:.2f} MB")
        loader_logs.append(f"Processing speed: {processing_speed:.2f} MB/s")
    
    loader_logs.append("Document types:")
    for doc_type, count in sorted(doc_types.items()):
        loader_logs.append(f"  - {doc_type}: {count}")
    
    if loader_only:
        loader_logs.append("Chunking/embeddings/graph steps skipped (loader-only mode)")
    
    if loader.collect_metrics:
        loader_logs.append("Monitoring enabled for ingestion stage")

    if loader_only:
        return StagePayload(
            documents=all_documents,
            chunks=[],
            vector_hits=[],
            graph=Graph(nodes=[], edges=[]),
            stats=stats,
            loader_items=_build_loader_items(all_documents),
            loader_logs=loader_logs,
        )

    splitter = get_text_splitter(
        config_manager=app_config.config, app_config=app_config
    )
    chunks = split_documents(
        documents=all_documents,
        splitter=splitter,
        batch_size=10,
    )
    stats["chunk_count"] = len(chunks)
    loader_logs.append(f"Chunks created: {stats['chunk_count']}")

    embedding = _resolve_embedding_model(app_config=app_config)
    vector_store = None
    vector_hits: List[Document] = []
    vector_docs = chunks or all_documents

    if embedding:
        if augment:
            try:
                vector_store = state.load_vector_store(embedding)
            except Exception as exc:  # pragma: no cover - defensive log for demo
                logger.warning("Unable to load existing vector store: %s", exc)
                loader_logs.append(f"Could not load existing vector store: {exc}")

        if vector_store is None:
            try:
                vector_store = vector_store_from_config(
                    app_config.config.vector_store_config,
                    embedding=embedding,
                )
            except Exception as exc:
                logger.warning("Vector store from config failed, falling back to FAISS: %s", exc)
                loader_logs.append("Using in-memory FAISS vector store (config not available).")
                vector_store = FAISS.from_documents([], embedding)

        try:
            vector_store.add_documents(vector_docs)
            stats["vector_count"] = len(vector_docs)
            try:
                state.save_vector_store(vector_store)
            except Exception as exc:  # pragma: no cover - defensive log for demo
                logger.warning("Could not persist vector store: %s", exc)
                loader_logs.append(f"Vector store persistence failed: {exc}")

            if vector_docs:
                try:
                    seed_query = vector_docs[0].page_content[:200] or "demo"
                    vector_hits = vector_store.similarity_search(
                        seed_query, k=min(3, len(vector_docs))
                    )
                except Exception as exc:  # pragma: no cover - defensive log for demo
                    logger.warning("Vector similarity preview failed: %s", exc)
                    loader_logs.append(f"Vector similarity preview failed: {exc}")
        except Exception as exc:
            logger.error("Vector indexing failed: %s", exc)
            loader_logs.append(f"Vector indexing failed: {exc}")
    else:
        loader_logs.append("Embedding model unavailable; vector store step skipped.")

    graph = Graph(nodes=[], edges=[])
    try:
        extractor = EntityExtractionService(app_config=app_config)
        graph = await extractor.extract(vector_docs)
        stats["graph_nodes"] = len(graph.nodes)
        stats["graph_edges"] = len(graph.edges)
        try:
            state.save_graph(graph)
        except Exception as exc:  # pragma: no cover - defensive log for demo
            logger.warning("Could not persist graph: %s", exc)
            loader_logs.append(f"Graph persistence failed: {exc}")
    except ImportError as exc:
        logger.warning("Graph extraction dependencies missing: %s", exc)
        loader_logs.append(f"Graph extraction skipped (dependency missing): {exc}")
    except Exception as exc:  # pragma: no cover - defensive log for demo
        logger.warning("Graph extraction failed: %s", exc)
        loader_logs.append(f"Graph extraction failed: {exc}")

    return StagePayload(
        documents=all_documents,
        chunks=chunks,
        vector_hits=vector_hits,
        graph=graph,
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
    """Group documents by source file and concatenate pages together.
    
    For multi-page documents (PDFs, etc), all pages are concatenated into a single
    entry showing the full markdown content.
    """
    from collections import defaultdict
    
    # Group documents by source
    docs_by_source = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("source") or "unknown"
        docs_by_source[source].append(doc)
    
    items = []
    for idx, (source, source_docs) in enumerate(docs_by_source.items()):
        # Sort by page number if available
        source_docs.sort(key=lambda d: d.metadata.get("page", 0))
        
        # Concatenate all pages with separator
        full_content = "\n\n--- Page Break ---\n\n".join(doc.page_content or "" for doc in source_docs)
        preview = " ".join(full_content.strip().splitlines())[:320]
        
        # Try to get original filename from metadata first
        original_filename = None
        if source_docs:
            original_filename = source_docs[0].metadata.get("original_filename")
        
        if original_filename:
            title = original_filename
        elif source != "unknown":
            # Fall back to extracting from path
            source_path = Path(source)
            filename = source_path.name
            # If filename is a GUID (32 hex chars before extension), we've lost the original name
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            if len(base_name) == 32 and all(c in '0123456789abcdef' for c in base_name.lower()):
                # GUID filename - show extension at least
                title = f"Uploaded {filename.rsplit('.', 1)[1].upper() if '.' in filename else 'File'}"
            else:
                title = filename
        else:
            title = f"Document {idx+1}"
        
        page_count = len(source_docs)
        if page_count > 1:
            title = f"{title} ({page_count} pages concatenated)"
        
        items.append(
            {
                "id": f"doc-{idx}",
                "title": title,
                "source": source,
                "preview": preview,
                "content": full_content.strip(),
            }
        )
        
        if len(items) >= limit:
            break
    
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


def _resolve_embedding_model(app_config=None):
    import os

    config_manager = getattr(app_config, "config", None)

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
        model = get_embedding_model(
            config_manager=config_manager, app_config=app_config
        )
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

    app_config = get_app_config()
    embedding = _resolve_embedding_model(app_config=app_config)
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

    llm_caller = get_llm_caller(app_config=app_config)
    answer_text: str
    used_llm = False
    sources_used = []

    import time

    start = time.time()

    if vector_docs:
        sources_used.append("vector")
    if graph_docs:
        sources_used.append("graph")

    if llm_caller and (vector_docs or graph_docs):
        vector_context = "\n".join(
            f"Doc {idx+1} (source={d.metadata.get('source','unknown')}): {d.page_content[:400]}"
            for idx, d in enumerate(vector_docs)
        ) or "No vector hits."
        graph_context = "\n".join(
            f"Node {idx+1} (type={d.metadata.get('node_type','node')}): {d.page_content[:300]}"
            for idx, d in enumerate(graph_docs)
        ) or "No graph hits."
        prompt = (
            "You are a concise assistant that answers using the provided context.\n"
            f"Question: {question}\n"
            f"Vector context:\n{vector_context}\n"
            f"Graph context:\n{graph_context}\n"
            "Answer using both vector and graph context when possible."
        )
        try:
            answer_text = call_llm_sync(llm_caller, prompt)
            used_llm = True
        except Exception:
            answer_text = _fallback_answer(question, vector_docs, graph_docs)
    else:
        answer_text = _fallback_answer(question, vector_docs, graph_docs)

    response_time = round(time.time() - start, 2)

    def _to_payload(docs: Sequence[Document]) -> List[Dict]:
        items: List[Dict] = []
        for doc in docs:
            items.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata or {},
                }
            )
        return items
    return {
        "question": question,
        "answer": answer_text,
        "vector_docs": summarize_documents(vector_docs, limit=3),
        "graph_docs": summarize_documents(graph_docs, limit=3),
        "used_llm": used_llm,
        "response_time": response_time,
        "sources_used": sources_used,
        "vector_hits_raw": _to_payload(vector_docs),
        "graph_hits_raw": _to_payload(graph_docs),
    }


def render_cached_pipeline(request) -> Optional[Dict]:
    """
    Build the template context from the last saved pipeline payload, if present.
    """
    payload = state.load_pipeline_payload()
    if not payload:
        return None

    def _to_docs(items: Sequence[Dict]) -> List[Dict]:
        docs = []
        for item in items:
            content = item.get("page_content", "")
            metadata = item.get("metadata") or {}
            docs.append(
                {
                    "content": content[:200] + ("..." if len(content) > 200 else ""),
                    "metadata": metadata,
                }
            )
        return docs

    graph_json = payload.get("graph") or {}
    graph_nodes = graph_json.get("nodes") or []
    graph_edges = graph_json.get("edges") or []

    return {
        "request": request,
        "stats": payload.get("stats") or {},
        "documents": _to_docs(payload.get("documents") or []),
        "chunks": _to_docs(payload.get("chunks") or []),
        "vector_hits": _to_docs(payload.get("vector_hits") or []),
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "loader_items": payload.get("loader_items") or [],
        "loader_logs": payload.get("loader_logs") or [],
    }


def _serialize_payload(
    *,
    documents: Sequence[Document],
    chunks: Sequence[Document],
    vector_hits: Sequence[Document],
    graph: Graph,
    stats: Dict[str, int],
    loader_items: Sequence[Dict[str, str]],
    loader_logs: Sequence[str],
) -> Dict:
    def _simple_docs(items: Sequence[Document]) -> List[Dict]:
        return [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata or {},
            }
            for doc in items
        ]

    return {
        "documents": _simple_docs(documents),
        "chunks": _simple_docs(chunks),
        "vector_hits": _simple_docs(vector_hits),
        "graph": graph.model_dump() if graph else {},
        "stats": stats,
        "loader_items": list(loader_items),
        "loader_logs": list(loader_logs),
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
