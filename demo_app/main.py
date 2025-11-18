from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_core.documents import Document

from . import state
from .config_state import apply_config_yaml, current_config_source, current_config_yaml
from .pipeline_demo import (
    answer_question,
    run_ingestion_demo,
    summarize_documents,
    summarize_graph_edges,
    summarize_graph_nodes,
)

from dotenv import load_dotenv

load_dotenv(override=True)

# Add logger configuration
logger = logging.getLogger(__name__)

app = FastAPI(title="RAGdoll Demo")
templates = Jinja2Templates(directory="demo_app/templates")


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
    success = False
    try:
        payload = await run_ingestion_demo(
            sources=combined_sources,
            extra_documents=manual_docs,
            augment=augment,  # User's choice: add to existing stores or reset them
            loader_only=False,  # Full pipeline: load, chunk, embed, store
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
                except OSError:
                    pass
            state.clear_staged_manifest(delete_files=True)

    context = {
        "request": request,
        "stats": payload.stats,
        "documents": summarize_documents(payload.documents),
        "chunks": summarize_documents(payload.chunks),
        "vector_hits": summarize_documents(payload.vector_hits),
        "graph_nodes": summarize_graph_nodes(payload.graph.nodes),
        "graph_edges": summarize_graph_edges(payload.graph.edges),
        "loader_items": payload.loader_items,
        "loader_logs": payload.loader_logs,
    }
    return templates.TemplateResponse("partials/pipeline_results.html", context)


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, question: str = Form(...)) -> HTMLResponse:
    try:
        result = answer_question(question)
    except ValueError as exc:
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": str(exc)},
            status_code=400,
        )

    context = {"request": request, **result}
    return templates.TemplateResponse("partials/chat_response.html", context)


@app.post("/stage")
async def stage_files(files: List[UploadFile] = Form(default=[])) -> JSONResponse:
    """
    Stage uploaded files without processing them yet.
    """
    print("=== /stage endpoint called ===")
    logger.info("=== /stage endpoint called ===")
    
    print(f"Number of files received: {len(files)}")
    logger.info(f"Number of files received: {len(files)}")
    
    for upload in files:
        print(f"  - {upload.filename}")
        logger.info(f"  - {upload.filename}")
    
    if not files:
        print("No files provided, returning existing staged files")
        logger.info("No files provided, returning existing staged files")
        return JSONResponse(
            {"staged_files": state.staged_file_entries()}, status_code=200
        )
    
    # Always reset the staged manifest and uploaded temp files when new files arrive.
    state.clear_staged_manifest(delete_files=True)
    
    saved_entries = await _stage_uploads(files)
    print(f"Saved entries: {saved_entries}")
    logger.info(f"Saved entries: {saved_entries}")
    
    staged = state.add_staged_files(saved_entries)
    print(f"All staged files after adding: {staged}")
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
    print("=== /load endpoint called ===")
    logger.info("=== /load endpoint called ===")
    
    form = await request.form()
    print(f"Form data keys: {list(form.keys())}")
    logger.info(f"Form data keys: {list(form.keys())}")

    file_inputs: List[UploadFile] = [
        upload for upload in form.getlist("files") if isinstance(upload, UploadFile)
    ]
    urls = form.get("urls", "") or ""
    text_input = form.get("text_input", "") or ""

    print(f"file_inputs count: {len(file_inputs)}")
    print(f"urls: '{urls}'")
    print(f"text_input: '{text_input[:100] if text_input else 'None'}'")

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

    print(f"staged_paths: {staged_paths}")
    print(f"saved_paths: {saved_paths}")
    print(f"url_list: {url_list}")
    print(f"combined_sources: {combined_sources}")
    print(f"manual_docs count: {len(manual_docs)}")

    success = False
    try:
        print("Calling run_ingestion_demo with loader_only=True (no vector/graph operations)")
        logger.info("Calling run_ingestion_demo with loader_only=True (no vector/graph operations)")
        payload = await run_ingestion_demo(
            sources=combined_sources,
            extra_documents=manual_docs,
            augment=False,  # Doesn't matter for loader_only, but semantically "fresh view"
            loader_only=True,  # Don't touch vector/graph stores, just load and show markdown
        )
        print("run_ingestion_demo succeeded")
        logger.info("run_ingestion_demo succeeded")
        success = True
    except ValueError as exc:
        logger.error(f"ValueError caught: {exc}")
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
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": f"Unexpected error: {exc}"},
            status_code=500,
        )
    finally:
        if success:
            for path in saved_paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
            state.clear_staged_manifest(delete_files=True)

    context = {
        "request": request,
        "stats": payload.stats,
        "documents": summarize_documents(payload.documents),
        "chunks": summarize_documents(payload.chunks),
        "vector_hits": summarize_documents(payload.vector_hits),
        "graph_nodes": summarize_graph_nodes(payload.graph.nodes),
        "graph_edges": summarize_graph_edges(payload.graph.edges),
        "loader_items": payload.loader_items,
        "loader_logs": payload.loader_logs,
    }
    return templates.TemplateResponse("partials/pipeline_results.html", context)
