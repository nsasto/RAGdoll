from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from ragdoll.entity_extraction.models import Graph

STATE_DIR = Path("demo_state")
VECTOR_DIR = STATE_DIR / "vector_store"
GRAPH_JSON = STATE_DIR / "graph.json"
GRAPH_PICKLE = STATE_DIR / "graph_output.gpickle"
UPLOAD_DIR = STATE_DIR / "uploads"


def ensure_state_dirs() -> None:
    """Ensure the demo state directories exist."""

    STATE_DIR.mkdir(exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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

    if not VECTOR_DIR.exists():
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
