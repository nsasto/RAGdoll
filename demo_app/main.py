from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional, AsyncGenerator, Sequence, Dict

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain_core.documents import Document

from ragdoll import Ragdoll
from ragdoll.pipeline import IngestionOptions

from .state import get_app_state
from . import state
from .config_state import (
    apply_config_yaml,
    current_config_source,
    current_config_yaml,
    initialize_app_config,
    get_app_config,
)
from ragdoll.ingestion import DocumentLoaderService

# Add project root to Python path and load .env from root
import sys
from pathlib import Path
import os

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from dotenv import load_dotenv

load_dotenv(override=True)
k = os.getenv("OPENAI_API_KEY")
if k:
    print("Key len:", len(k), "prefix:", k[:12], "suffix:", k[-6:])
else:
    print("No OPEN_API_KEY found.")
# Add logger configuration
logger = logging.getLogger(__name__)

app = FastAPI(title="RAGdoll Demo")
templates = Jinja2Templates(directory="demo_app/templates")


def summarize_documents(docs: Sequence[Document], *, limit: int = 5) -> List[Dict]:
    summary = []
    for doc in list(docs)[:limit]:
        content = doc.page_content.strip().replace("\n", " ")
        if len(content) > 200:
            content = f"{content[:200]}..."
        summary.append({"content": content, "metadata": doc.metadata or {}})
    return summary


def _build_loader_items(
    docs: Sequence[Document], *, limit: int = 10
) -> List[Dict[str, str]]:
    """Group documents by source file and concatenate pages together."""
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
        full_content = "\n\n--- Page Break ---\n\n".join(
            doc.page_content or "" for doc in source_docs
        )
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
            base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
            if len(base_name) == 32 and all(
                c in "0123456789abcdef" for c in base_name.lower()
            ):
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


@app.on_event("startup")
async def startup_event():
    """Initialize the global AppConfig and Ragdoll instance on application startup."""
    initialize_app_config()
    logger.info("Global AppConfig initialized")
    app_state = get_app_state()
    if not app_state.ragdoll:
        logger.info("Initializing Ragdoll instance...")
        app_state.ragdoll = Ragdoll(app_config=get_app_config())
        logger.info("Ragdoll instance initialized and stored in AppState.")

        # Load any existing vector store and graph from disk
        from ragdoll.embeddings import get_embedding_model
        from ragdoll.vector_stores.base_vector_store import BaseVectorStore

        embedding_model = get_embedding_model()

        # Load vector store if it exists
        loaded_vs = state.load_vector_store(embedding_model)
        if loaded_vs:
            logger.info("Loaded existing vector store from disk")
            wrapped = BaseVectorStore(loaded_vs)
            app_state.ragdoll.vector_store = wrapped
            app_state.vector_store = wrapped
            try:
                if hasattr(loaded_vs, "docstore"):
                    count = len(loaded_vs.docstore._dict)
                    logger.info(f"Vector store contains {count} documents")
            except Exception as e:
                logger.warning(f"Could not get document count: {e}")

        # Load graph if it exists
        loaded_graph = state.load_graph()
        if loaded_graph:
            logger.info("Loaded existing graph from disk")
            app_state.graph = loaded_graph
            try:
                logger.info(
                    f"Graph contains {len(loaded_graph.nodes)} nodes and {len(loaded_graph.edges)} edges"
                )
            except Exception as e:
                logger.warning(f"Could not get graph stats: {e}")


def _config_context(
    request: Request,
    *,
    config_message: Optional[str] = None,
    message_type: Optional[str] = None,
    config_yaml_override: Optional[str] = None,
) -> dict:
    """Build the context shared between the index page and config partial."""

    yaml_text = (
        current_config_yaml() if config_yaml_override is None else config_yaml_override
    )
    return {
        "request": request,
        "config_yaml": yaml_text,
        "config_message": config_message,
        "message_type": message_type,
        "config_source": current_config_source(),
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    # Clear staged files and uploads on app refresh
    print("=== Index page loaded, clearing all staged files and uploads ===")
    staged_entries = state.staged_file_entries()
    print(f"Staged entries: {len(staged_entries)}")

    # Clear the manifest
    try:
        state.clear_staged_manifest(delete_files=False)
        print("Cleared staged manifest")
    except Exception as e:
        print(f"Could not clear manifest: {e}")
        logger.warning(f"Could not clear manifest: {e}")

    # Clear ALL files in uploads directory (not just staged ones)
    upload_dir = state.upload_directory()
    if upload_dir.exists():
        deleted_count = 0
        failed_count = 0
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted_count += 1
                except (OSError, PermissionError) as e:
                    print(f"Could not delete {file_path.name}: {e}")
                    failed_count += 1

        if deleted_count > 0:
            print(f"Deleted {deleted_count} file(s) from uploads directory")
            logger.info(f"Deleted {deleted_count} file(s) from uploads directory")
        if failed_count > 0:
            print(f"Failed to delete {failed_count} file(s) (may be locked)")
            logger.warning(f"Failed to delete {failed_count} file(s) (may be locked)")

    return templates.TemplateResponse("index.html", _config_context(request))


@app.post("/config", response_class=HTMLResponse)
async def update_config(request: Request, config_yaml: str = Form(...)) -> HTMLResponse:
    try:
        apply_config_yaml(config_yaml)
    except Exception as exc:
        context = _config_context(
            request,
            config_message=f"Unable to apply configuration: {exc}",
            message_type="error",
            config_yaml_override=config_yaml,
        )
        return templates.TemplateResponse(
            "partials/config_panel.html",
            context,
            status_code=400,
        )

    context = _config_context(
        request,
        config_message="Configuration saved and activated for the demo.",
        message_type="success",
    )
    return templates.TemplateResponse("partials/config_panel.html", context)


@app.post("/ingest", response_class=HTMLResponse)
async def ingest(request: Request) -> HTMLResponse:
    form = await request.form()

    file_inputs: List[UploadFile] = [
        upload for upload in form.getlist("files") if isinstance(upload, UploadFile)
    ]
    urls = form.get("urls", "") or ""
    text_input = form.get("text_input", "") or ""
    mode = form.get("mode", "augment") or "augment"

    augment = mode != "reset"
    saved_paths = await _persist_uploads(file_inputs)
    url_list = [line.strip() for line in urls.splitlines() if line.strip()]

    manual_docs: List[Document] = []
    if text_input.strip():
        manual_docs.append(
            Document(
                page_content=text_input.strip(),
                metadata={"source": "manual_input"},
            )
        )

    staged_paths = [str(path) for path in state.staged_file_paths()]
    combined_sources = staged_paths + saved_paths + url_list

    # Build filename mapping from staged manifest
    source_filename_map = {}
    staged_entries = state.staged_file_entries()
    upload_dir = state.upload_directory()
    for entry in staged_entries:
        file_path = str((upload_dir / entry["filename"]).resolve())
        original_name = entry.get("original_name", entry["filename"])
        source_filename_map[file_path] = original_name

    success = False
    try:
        app_state = get_app_state()
        ragdoll = app_state.ragdoll
        if not ragdoll:
            raise ValueError("Ragdoll instance not initialized.")

        # Use the shared Ragdoll instance for ingestion
        payload = ragdoll.ingest_with_graph_sync(
            sources=combined_sources + manual_docs,
            options=IngestionOptions(augment=augment),
        )
        success = True
    except ValueError as exc:
        # Preserve staged files on error so the user can retry.
        message = (
            f"{exc} "
            f"(staged_files={len(staged_paths)}, uploads_in_form={len(saved_paths)}, "
            f"urls={len(url_list)}, text={'yes' if manual_docs else 'no'}, "
            f'sources={combined_sources[:3]}{"..." if len(combined_sources) > 3 else ""})'
        )
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": message},
            status_code=400,
        )
    finally:
        if success:
            for path in saved_paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except (OSError, PermissionError) as e:
                    print(f"Could not delete {path}: {e}")
                    logger.warning(f"Could not delete {path}: {e}")

            # Clear staged manifest and attempt to delete files
            try:
                state.clear_staged_manifest(delete_files=True)
            except (OSError, PermissionError) as e:
                print(f"Could not clear staged files (may still be locked): {e}")
                logger.warning(f"Could not clear staged files: {e}")
                # At least clear the manifest even if files are locked
                try:
                    state.clear_staged_manifest(delete_files=False)
                except Exception as e2:
                    logger.error(f"Could not even clear manifest: {e2}")

    graph = payload.get("graph")
    context = {
        "request": request,
        "stats": payload.get("stats"),
        "documents": [],  # Not available in new payload
        "chunks": [],  # Not available in new payload
        "vector_hits": [],  # Not available in new payload
        "graph_nodes": [node.model_dump() for node in graph.nodes] if graph else [],
        "graph_edges": [edge.model_dump() for edge in graph.edges] if graph else [],
        "loader_items": [],  # Not available in new payload
        "loader_logs": [],  # Not available in new payload
    }
    return templates.TemplateResponse("partials/pipeline_results.html", context)


@app.post("/chunk", response_class=HTMLResponse)
async def chunk_documents(
    request: Request, chunk_size: int = Form(1000), chunk_overlap: int = Form(200)
) -> HTMLResponse:
    """
    Chunk the loaded documents using the specified parameters.
    """
    from ragdoll.chunkers import get_text_splitter, split_documents
    import time

    try:
        logger.info(
            f"=== /chunk endpoint called with chunk_size={chunk_size}, chunk_overlap={chunk_overlap} ==="
        )

        app_config = get_app_config()

        # Prefer persisted loaded documents saved by /load
        saved = state.read_loaded_documents()
        documents = []
        if saved:
            documents = [
                Document(
                    page_content=d.get("page_content", ""),
                    metadata=d.get("metadata") or {},
                )
                for d in saved
            ]
        else:
            # Fall back to staged paths and reload via loader
            staged_paths = [str(path) for path in state.staged_file_paths()]
            if not staged_paths:
                return templates.TemplateResponse(
                    "partials/error.html",
                    {
                        "request": request,
                        "message": "No documents loaded. Please load documents first using the Loaders tab.",
                    },
                    status_code=400,
                )
            loader = DocumentLoaderService(
                app_config=app_config, collect_metrics=False, use_cache=False
            )
            chunk_start_time = time.time()
            raw_docs = loader.ingest_documents(staged_paths)
            documents = [
                (
                    doc
                    if isinstance(doc, Document)
                    else Document(
                        page_content=doc.get("page_content", ""),
                        metadata=doc.get("metadata") or {},
                    )
                )
                for doc in raw_docs
            ]

        # Get text splitter and chunk
        chunk_start_time = time.time()
        splitter = get_text_splitter(
            splitter_type="recursive",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            app_config=app_config,
        )
        chunks = split_documents(documents, text_splitter=splitter)
        chunk_duration = time.time() - chunk_start_time

        # Calculate stats
        total_chunks = len(chunks)
        total_chunk_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chunk_chars / total_chunks if total_chunks > 0 else 0

        stats = {
            "total_documents": len(documents),
            "total_chunks": total_chunks,
            "chunk_duration": round(chunk_duration, 2),
            "avg_chunk_size": round(avg_chunk_size, 0),
        }

        config_info = [
            {"key": "Chunker", "value": "RecursiveCharacterTextSplitter"},
            {"key": "Chunk size", "value": f"{chunk_size} characters"},
            {"key": "Chunk overlap", "value": f"{chunk_overlap} characters"},
            {"key": "Documents processed", "value": len(documents)},
            {"key": "Chunks created", "value": total_chunks},
            {"key": "Chunking duration", "value": f"{chunk_duration:.2f}s"},
            {
                "key": "Avg chunks per doc",
                "value": f"{total_chunks / len(documents):.1f}" if documents else "N/A",
            },
            {"key": "Avg chunk size", "value": f"{avg_chunk_size:.0f} characters"},
        ]

        # Serialize chunks to JSON-compatible format
        chunks_json = [
            {"page_content": chunk.page_content, "metadata": chunk.metadata or {}}
            for chunk in chunks[:50]
        ]

        context = {
            "request": request,
            "stats": stats,
            "chunks": chunks_json,
            "config_info": config_info,
        }
        return templates.TemplateResponse("partials/chunk_results.html", context)

    except Exception as exc:
        logger.error(f"Chunking error: {exc}", exc_info=True)
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": f"Chunking failed: {exc}"},
            status_code=500,
        )


@app.get("/pipeline/cached", response_class=HTMLResponse)
async def cached_pipeline(request: Request) -> HTMLResponse:
    """
    Render the last saved pipeline payload (if present) without re-running ingestion.
    """
    # This is now managed by the Ragdoll instance state, so we can't easily show old results.
    # Returning 204 to prevent errors. A better implementation might cache results.
    return HTMLResponse(status_code=204)


@app.post("/populate_vector", response_class=HTMLResponse)
async def populate_vector(request: Request) -> HTMLResponse:
    """
    Populate the vector store with embeddings from loaded/chunked documents.
    """
    import time
    from ragdoll.embeddings import get_embedding_model
    from ragdoll.vector_stores import vector_store_from_config
    from ragdoll.config.base_config import VectorStoreConfig

    try:
        logger.info("=== /populate_vector endpoint called ===")
        app_config = get_app_config()

        # Get loaded documents
        saved = state.read_loaded_documents()
        if not saved:
            return templates.TemplateResponse(
                "partials/error.html",
                {
                    "request": request,
                    "message": "No documents loaded. Please load documents first.",
                },
                status_code=400,
            )

        documents = [
            Document(
                page_content=d.get("page_content", ""), metadata=d.get("metadata") or {}
            )
            for d in saved
        ]

        # Chunk documents
        from ragdoll.chunkers import get_text_splitter, split_documents

        splitter = get_text_splitter(
            splitter_type="recursive",
            chunk_size=1000,
            chunk_overlap=200,
            app_config=app_config,
        )
        chunks = split_documents(documents, text_splitter=splitter)

        # Get embeddings and create EMPTY vector store first
        start_time = time.time()
        embedding_model = get_embedding_model()
        vector_dimension = len(embedding_model.embed_query("test"))

        # Configure vector store (create empty, don't use from_documents)
        vector_config = VectorStoreConfig(enabled=True, store_type="faiss", params={})
        from ragdoll.vector_stores import create_vector_store

        vector_store = create_vector_store(
            vector_config.store_type, embedding=embedding_model
        )

        # Add documents and CAPTURE vector IDs
        vector_ids = vector_store.add_documents(chunks)
        logger.info(
            f"Added {len(chunks)} chunks to vector store, got {len(vector_ids)} IDs back"
        )

        # CRITICAL: Write vector_ids back into chunk metadata
        from datetime import datetime, timezone

        embedding_timestamp = datetime.now(timezone.utc).isoformat()

        if vector_ids and len(vector_ids) == len(chunks):
            for chunk, vector_id in zip(chunks, vector_ids):
                chunk.metadata["vector_id"] = vector_id
                chunk.metadata["embedding_timestamp"] = embedding_timestamp
            logger.info(f"Enriched {len(chunks)} chunks with vector_id metadata")
        else:
            logger.warning(
                f"Vector ID count mismatch: {len(vector_ids)} IDs for {len(chunks)} chunks. "
                "Graph nodes may not link to embeddings properly."
            )

        # Save enriched chunks for graph population
        chunks_json = [
            {"page_content": chunk.page_content, "metadata": chunk.metadata or {}}
            for chunk in chunks
        ]
        state.save_chunked_documents(chunks_json)
        logger.info(
            "Saved enriched chunks with vector_id metadata for graph population"
        )

        # Save to disk AND update global state
        state.save_vector_store(vector_store)
        app_state = get_app_state()
        app_state.vector_store = vector_store
        app_state.ragdoll.vector_store = vector_store
        logger.info("Vector store saved to disk and updated in global state")

        duration = time.time() - start_time

        # Prepare stats
        stats = {
            "documents_embedded": len(documents),
            "chunks_embedded": len(chunks),
            "vector_dimension": vector_dimension,
            "embedding_duration": round(duration, 2),
            "store_type": "FAISS",
        }

        # Sample vectors for preview
        sample_vectors = []
        for chunk in chunks[:3]:
            vector = embedding_model.embed_query(chunk.page_content)
            sample_vectors.append(
                {
                    "content": chunk.page_content,
                    "source": chunk.metadata.get("source", "unknown"),
                    "vector_preview": ", ".join([f"{v:.3f}" for v in vector[:5]])
                    + "...",
                }
            )

        context = {
            "request": request,
            "stats": stats,
            "sample_vectors": sample_vectors,
        }
        return templates.TemplateResponse("partials/vector_results.html", context)

    except Exception as exc:
        logger.error(f"Vector population error: {exc}", exc_info=True)
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": f"Vector population failed: {exc}"},
            status_code=500,
        )


@app.post("/populate_graph", response_class=HTMLResponse)
async def populate_graph(request: Request) -> HTMLResponse:
    """
    Extract entities and relationships, then populate the graph store.
    """
    import time
    from ragdoll.entity_extraction import EntityExtractionService
    from ragdoll.llms import get_llm_caller

    try:
        logger.info("=== /populate_graph endpoint called ===")
        app_config = get_app_config()

        # Try to get enriched chunks with vector_id first, fall back to raw documents
        saved_chunks = state.read_chunked_documents()
        if saved_chunks:
            documents = [
                Document(
                    page_content=d.get("page_content", ""),
                    metadata=d.get("metadata") or {},
                )
                for d in saved_chunks
            ]
            logger.info(
                "Graph populate: using %d enriched chunks with vector_id metadata",
                len(documents),
            )
            # Log sample metadata to verify vector_id presence
            if documents and documents[0].metadata:
                sample_meta = documents[0].metadata
                has_vector_id = "vector_id" in sample_meta
                logger.info(f"Sample chunk metadata has vector_id: {has_vector_id}")
                if has_vector_id:
                    logger.info(f"Sample vector_id: {sample_meta['vector_id']}")
        else:
            # Fallback to raw documents (will result in orphaned nodes)
            saved = state.read_loaded_documents()
            if not saved:
                return templates.TemplateResponse(
                    "partials/error.html",
                    {
                        "request": request,
                        "message": "No documents loaded. Please populate vector store first.",
                    },
                    status_code=400,
                )
            documents = [
                Document(
                    page_content=d.get("page_content", ""),
                    metadata=d.get("metadata") or {},
                )
                for d in saved
            ]
            logger.warning(
                "Graph populate: using %d raw documents (no vector_id metadata). "
                "Graph nodes will be orphaned. Please run /populate_vector first.",
                len(documents),
            )

        # Get LLM caller for entity extraction
        llm_caller = get_llm_caller(app_config=app_config)

        # Extract entities
        start_time = time.time()
        entity_service = EntityExtractionService(
            llm_caller=llm_caller,
            app_config=app_config,
        )
        logger.info(
            "Graph populate: starting extraction (chunk_documents=%s)",
            entity_service.chunk_documents,
        )

        graph = await entity_service.extract(documents)
        logger.info(
            "Graph populate: extraction complete (nodes=%d, edges=%d)",
            len(graph.nodes),
            len(graph.edges),
        )
        duration = time.time() - start_time

        # Save graph to disk AND update global state
        state.save_graph(graph)
        app_state = get_app_state()
        app_state.graph = graph
        logger.info(
            "Graph populate: graph persisted to disk and updated in global state in %.2fs",
            duration,
        )

        # Prepare stats
        unique_entity_names = {
            (node.name or "").strip().lower()
            for node in graph.nodes
            if (node.name or "").strip()
        }
        stats = {
            "entities_extracted": len(graph.nodes),
            "entities_unique": len(unique_entity_names),
            "relationships_extracted": len(graph.edges),
            "extraction_duration": round(duration, 2),
            "store_type": "JSON",
        }

        # Serialize nodes and edges for template
        nodes_json = [
            {
                "id": node.id,
                "name": node.name,
                "type": node.type,
                "metadata": node.metadata or {},
            }
            for node in graph.nodes
        ]

        edges_json = [
            {
                "id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "type": edge.type,
            }
            for edge in graph.edges
        ]

        context = {
            "request": request,
            "stats": stats,
            "nodes": nodes_json,
            "edges": edges_json,
        }
        return templates.TemplateResponse("partials/graph_results.html", context)

    except Exception as exc:
        logger.error(f"Graph population error: {exc}", exc_info=True)
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": f"Graph population failed: {exc}"},
            status_code=500,
        )


@app.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request, question: str = Form(...), retriever: str = Form("vector")
) -> HTMLResponse:
    try:
        app_state = get_app_state()
        ragdoll = app_state.ragdoll
        if not ragdoll:
            raise ValueError(
                "Ragdoll instance not initialized. Please ingest data first."
            )

        # Ensure ragdoll has access to global state's vector store and graph
        if app_state.vector_store and not ragdoll.vector_store:
            ragdoll.vector_store = app_state.vector_store
            logger.info("Loaded vector store from global state into ragdoll")

        # Create graph retriever if needed
        if app_state.graph and not ragdoll.graph_retriever:
            from ragdoll.retrieval import GraphRetriever
            import networkx as nx

            # Convert Graph (Pydantic model) to NetworkX graph
            nx_graph = nx.DiGraph()

            # Add nodes
            for node in app_state.graph.nodes:
                node_attrs = {
                    "type": node.type,
                    "name": node.name,
                }
                if node.label:
                    node_attrs["label"] = node.label
                if node.properties:
                    node_attrs.update(node.properties)
                elif node.metadata:
                    node_attrs.update(node.metadata)
                nx_graph.add_node(node.id, **node_attrs)

            # Add edges
            for edge in app_state.graph.edges:
                nx_graph.add_edge(
                    edge.source,
                    edge.target,
                    type=edge.type,
                    **edge.metadata if edge.metadata else {},
                )

            logger.info(
                f"Created NetworkX graph with {len(nx_graph.nodes)} nodes and {len(nx_graph.edges)} edges"
            )

            ragdoll.graph_retriever = GraphRetriever(
                graph_store=nx_graph,
                vector_store=ragdoll.vector_store if ragdoll.vector_store else None,
                embedding_model=(
                    ragdoll.embedding_model if ragdoll.embedding_model else None
                ),
                top_k=5,
                max_hops=2,
            )
            logger.info("Created GraphRetriever from global state")

            # Rebuild hybrid retriever now that we have graph retriever
            ragdoll.hybrid_retriever = ragdoll._build_retriever()
            logger.info("Rebuilt HybridRetriever with both vector and graph retrievers")

        # Execute query
        result = ragdoll.query(question, retriever_mode=retriever, k=5)
        logger.info(
            f"Query completed: {len(result.get('documents', []))} documents retrieved using {result.get('retriever_used')} retriever"
        )

    except ValueError as exc:
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": str(exc)},
            status_code=400,
        )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": f"Unexpected error: {exc}"},
            status_code=500,
        )

    context = {"request": request, "question": question, **result}
    return templates.TemplateResponse("partials/chat_response.html", context)


@app.post("/stage")
async def stage_files(files: List[UploadFile] = Form(default=[])) -> JSONResponse:
    """
    Stage uploaded files without processing them yet.
    """
    # print("=== /stage endpoint called ===")
    logger.info("=== /stage endpoint called ===")

    ##print(f"Number of files received: {len(files)}")
    logger.info(f"Number of files received: {len(files)}")

    for upload in files:
        print(f"  - {upload.filename}")
        logger.info(f"  - {upload.filename}")

    if not files:
        # This is called by JavaScript on page load to check for existing staged files
        existing = state.staged_file_entries()
        if existing:
            print(f"Returning {len(existing)} existing staged file(s)")
            logger.info(f"Returning {len(existing)} existing staged file(s)")
        # No log needed when empty - this is normal on fresh page load
        return JSONResponse({"staged_files": existing}, status_code=200)

    # Add new files to the existing staged list (don't clear)
    saved_entries = await _stage_uploads(files)
    # print(f"Saved entries: {saved_entries}")
    logger.info(f"Saved entries: {saved_entries}")

    staged = state.add_staged_files(saved_entries)
    # print(f"All staged files after adding: {staged}")
    logger.info(f"All staged files after adding: {staged}")

    return JSONResponse({"staged_files": staged})


async def _stage_uploads(files: List[UploadFile]) -> List[dict]:
    logger.info(f"_stage_uploads called with {len(files)} files")
    saved_entries: List[dict] = []
    upload_dir = state.upload_directory()
    logger.info(f"Upload directory: {upload_dir}")

    for upload in files:
        if not upload.filename:
            logger.warning(f"Skipping file with no filename")
            continue
        suffix = Path(upload.filename).suffix
        dest = upload_dir / f"{uuid.uuid4().hex}{suffix}"
        content = await upload.read()
        logger.info(f"Writing {len(content)} bytes to {dest}")
        dest.write_bytes(content)
        saved_entries.append(
            {
                "filename": dest.name,
                "original_name": upload.filename,
            }
        )
    logger.info(f"_stage_uploads returning {len(saved_entries)} entries")
    return saved_entries


async def _persist_uploads(files: List[UploadFile]) -> List[str]:
    paths: List[str] = []
    upload_dir = state.upload_directory()
    for upload in files:
        if not upload.filename:
            continue
        suffix = Path(upload.filename).suffix
        dest = upload_dir / f"{uuid.uuid4().hex}{suffix}"
        content = await upload.read()
        dest.write_bytes(content)
        paths.append(str(dest))
    return paths


@app.post("/load", response_class=HTMLResponse)
async def load_docs(request: Request) -> HTMLResponse:
    """
    Loader-only endpoint: pulls staged/form sources and converts them to markdown (no chunking/embeddings).
    """
    # print("=== /load endpoint called ===")
    logger.info("=== /load endpoint called ===")

    form = await request.form()
    # print(f"Form data keys: {list(form.keys())}")
    logger.info(f"Form data keys: {list(form.keys())}")

    # Files should already be staged via /stage endpoint
    # Only get URLs and text input from the form
    urls = form.get("urls", "") or ""
    text_input = form.get("text_input", "") or ""

    print(f"urls: '{urls}'")
    print(f"text_input: '{text_input[:100] if text_input else 'None'}'")

    url_list = [line.strip() for line in urls.splitlines() if line.strip()]

    manual_docs: List[Document] = []
    if text_input.strip():
        manual_docs.append(
            Document(
                page_content=text_input.strip(),
                metadata={"source": "manual_input"},
            )
        )

    # Get files that were staged via /stage endpoint
    staged_paths = [str(path) for path in state.staged_file_paths()]
    combined_sources = staged_paths + url_list

    print(f"staged_paths: {staged_paths}")
    print(f"url_list: {url_list}")
    print(f"combined_sources: {combined_sources}")
    print(f"manual_docs count: {len(manual_docs)}")

    # Build filename mapping from staged manifest
    source_filename_map = {}
    staged_entries = state.staged_file_entries()
    upload_dir = state.upload_directory()
    for entry in staged_entries:
        file_path = str((upload_dir / entry["filename"]).resolve())
        original_name = entry.get("original_name", entry["filename"])
        source_filename_map[file_path] = original_name
    print(f"Filename mapping: {source_filename_map}")

    # Simplified loader-only flow using DocumentLoaderService directly
    try:
        app_config = get_app_config()
        loader = DocumentLoaderService(
            app_config=app_config, use_cache=False, collect_metrics=False
        )
        raw_documents = loader.ingest_documents(combined_sources)

        # Normalize to langchain Document objects if needed
        def normalize_documents(raw_docs):
            docs = []
            for entry in raw_docs:
                if isinstance(entry, Document):
                    docs.append(entry)
                elif isinstance(entry, dict):
                    docs.append(
                        Document(
                            page_content=str(entry.get("page_content", "")),
                            metadata=entry.get("metadata", {}) or {},
                        )
                    )
                else:
                    docs.append(Document(page_content=str(entry), metadata={}))
            return docs

        documents = normalize_documents(raw_documents + manual_docs)

        if not documents:
            return templates.TemplateResponse(
                "partials/error.html",
                {
                    "request": request,
                    "message": "No documents found. Please stage files or provide URLs/text.",
                },
                status_code=400,
            )

        # Build loader items for UI and simple stats
        stats = {
            "document_count": len(documents),
            "documents_loaded": len(raw_documents),
            "load_duration_seconds": 0,
            "total_characters": sum(len(d.page_content) for d in documents),
        }

        loader_logs = [
            f"Documents loaded: {len(raw_documents)}",
            f"Total documents: {len(documents)}",
        ]

        # Persist loaded documents for chunking endpoint
        try:
            simple_docs = [
                {"page_content": d.page_content, "metadata": d.metadata or {}}
                for d in documents
            ]
            state.save_loaded_documents(simple_docs)
        except Exception:
            pass

        context = {
            "request": request,
            "stats": stats,
            "documents": summarize_documents(documents),
            "chunks": [],
            "vector_hits": [],
            "graph_nodes": [],
            "graph_edges": [],
            "loader_items": _build_loader_items(documents),
            "loader_logs": loader_logs,
        }
        return templates.TemplateResponse("partials/pipeline_results.html", context)
    except ValueError as exc:
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": str(exc)},
            status_code=400,
        )
    except Exception as exc:
        logger.error(f"Loader failed: {exc}", exc_info=True)
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": f"Loader failed: {exc}"},
            status_code=500,
        )
