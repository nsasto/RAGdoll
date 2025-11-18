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
    
    logger.info(
        f"run_ingestion_demo called with sources={sources}, extra_docs count={len(extra_documents)}, "
        f"augment={augment}, loader_only={loader_only}"
    )

    if not augment and not loader_only:
        logger.info("Resetting state directory (augment=False)")
        state.reset_state_dir()
        state.ensure_state_dirs()

    loader = DocumentLoaderService()
    
    # Track loading stats
    load_start = time.time()
    logger.info(f"Calling loader.ingest_documents with {len(sources)} sources")
    loaded_docs = loader.ingest_documents(sources)
    load_duration = time.time() - load_start
    
    all_documents = list(loaded_docs) + list(extra_documents)

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

    return StagePayload(
        documents=all_documents,
        chunks=[],
        vector_hits=[],
        graph=Graph(nodes=[], edges=[]),
        stats=stats,
        loader_items=_build_loader_items(all_documents),
        loader_logs=loader_logs,
    )

    # ... rest of the full pipeline code ...


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
