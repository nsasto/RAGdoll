from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

from langchain_community.vectorstores import FAISS
from ragdoll.entity_extraction.models import Graph

STATE_DIR = Path("demo_state")
VECTOR_DIR = STATE_DIR / "vector_store"
GRAPH_JSON = STATE_DIR / "graph.json"
GRAPH_PICKLE = STATE_DIR / "graph_output.gpickle"
UPLOAD_DIR = STATE_DIR / "uploads"
STAGED_MANIFEST = STATE_DIR / "staged_manifest.json"
LOADED_DOCUMENTS = STATE_DIR / "loaded_documents.json"


def ensure_state_dirs() -> None:
    """Ensure the demo state directories exist."""

    STATE_DIR.mkdir(exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    STAGED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)


def reset_state_dir() -> None:
    """Wipe demo state (vector store, graphs, uploads)."""

    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR)
    ensure_state_dirs()


def save_graph(graph: Graph) -> None:
    """Persist the latest graph as JSON."""

    ensure_state_dirs()
    GRAPH_JSON.write_text(graph.model_dump_json(indent=2), encoding="utf-8")


def load_graph() -> Optional[Graph]:
    """Load the persisted graph, if any."""

    if not GRAPH_JSON.exists():
        return None
    return Graph.model_validate_json(GRAPH_JSON.read_text(encoding="utf-8"))


def graph_pickle_path() -> Path:
    ensure_state_dirs()
    return GRAPH_PICKLE


def load_vector_store(embedding) -> Optional[FAISS]:
    """Load the persisted FAISS store if available."""

    if not VECTOR_DIR.exists() or not (VECTOR_DIR / "index.faiss").exists():
        return None
    return FAISS.load_local(
        str(VECTOR_DIR), embedding, allow_dangerous_deserialization=True
    )


def save_vector_store(store: FAISS) -> None:
    """Persist the FAISS store to disk."""

    ensure_state_dirs()
    store.save_local(str(VECTOR_DIR))


def upload_directory() -> Path:
    ensure_state_dirs()
    return UPLOAD_DIR


def staged_manifest_path() -> Path:
    ensure_state_dirs()
    return STAGED_MANIFEST


def read_staged_manifest() -> dict:
    path = staged_manifest_path()
    if not path.exists():
        return {"files": []}
    return json.loads(path.read_text(encoding="utf-8"))


def write_staged_manifest(manifest: dict) -> None:
    staged_manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def add_staged_files(entries: Iterable[dict]) -> list[dict]:
    manifest = read_staged_manifest()
    existing = {entry["filename"] for entry in manifest.get("files", [])}
    for entry in entries:
        if entry["filename"] not in existing:
            manifest.setdefault("files", []).append(entry)
            existing.add(entry["filename"])
    write_staged_manifest(manifest)
    return manifest["files"]


def staged_file_entries() -> list[dict]:
    return read_staged_manifest().get("files", [])


def staged_file_paths() -> list[Path]:
    entries = staged_file_entries()
    paths: list[Path] = []
    for entry in entries:
        candidate = (UPLOAD_DIR / entry["filename"]).resolve()
        if candidate.exists():
            paths.append(candidate)
    return paths


def clear_staged_manifest(delete_files: bool = False) -> None:
    manifest = read_staged_manifest()
    if delete_files:
        for entry in manifest.get("files", []):
            path = UPLOAD_DIR / entry["filename"]
            if path.exists():
                path.unlink()
    manifest_path = staged_manifest_path()
    if manifest_path.exists():
        manifest_path.unlink()


def loaded_documents_path() -> Path:
    ensure_state_dirs()
    return LOADED_DOCUMENTS


def save_loaded_documents(docs: list[dict]) -> None:
    """Save a simple JSON representation of loaded documents.

    Each doc should be a dict with keys: `page_content` and `metadata`.
    """
    ensure_state_dirs()
    try:
        loaded_documents_path().write_text(json.dumps(docs, indent=2), encoding="utf-8")
    except Exception:
        # Best-effort persistence; don't raise for demo UI
        pass


def read_loaded_documents() -> list[dict]:
    path = loaded_documents_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
