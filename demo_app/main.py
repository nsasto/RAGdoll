from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_core.documents import Document

from . import state
from .config_state import current_config_source, current_config_yaml, apply_config_yaml
from .pipeline_demo import (
    answer_question,
    run_ingestion_demo,
    summarize_documents,
    summarize_graph_edges,
    summarize_graph_nodes,
)

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

    yaml_text = current_config_yaml() if config_yaml_override is None else config_yaml_override
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

    try:
        payload = await run_ingestion_demo(
            sources=saved_paths + url_list,
            extra_documents=manual_docs,
            augment=augment,
        )
    except ValueError as exc:
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": str(exc)},
            status_code=400,
        )
    finally:
        for path in saved_paths:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    context = {
        "request": request,
        "stats": payload.stats,
        "documents": summarize_documents(payload.documents),
        "chunks": summarize_documents(payload.chunks),
        "vector_hits": summarize_documents(payload.vector_hits),
        "graph_nodes": summarize_graph_nodes(payload.graph.nodes),
        "graph_edges": summarize_graph_edges(payload.graph.edges),
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
