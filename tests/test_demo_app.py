from __future__ import annotations

import os
import subprocess
import sys

from fastapi.testclient import TestClient
import httpx

import demo_app.main as demo_main
from demo_app.main import app
from demo_app.state import get_app_state
from ragdoll.errors import GraphWriteError


def test_homepage_starts_when_provider_network_is_unavailable(monkeypatch) -> None:
    original_send = httpx.Client.send

    def reject_provider_requests(client, request, *args, **kwargs):
        if request.url.host != "testserver":
            raise AssertionError(
                "the demo attempted an outbound request during startup"
            )
        return original_send(client, request, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", reject_provider_requests)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "RAGdoll" in response.text


def test_import_does_not_log_api_key_details() -> None:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "sk-test-super-secret-marker-1234567890"

    completed = subprocess.run(
        [sys.executable, "-c", "import demo_app.main"],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout == ""


def test_ingest_route_awaits_the_async_ragdoll_interface() -> None:
    class FakeRagdoll:
        def __init__(self) -> None:
            self.sources = None

        async def ingest_with_graph(self, sources, *, options):
            self.sources = sources
            return {"stats": {"documents": len(sources)}, "graph": None}

    fake = FakeRagdoll()
    app_state = get_app_state()
    previous = app_state.ragdoll
    app_state.ragdoll = fake
    try:
        with TestClient(app) as client:
            response = client.post(
                "/ingest",
                data={"text_input": "Ragdoll is a retrieval toolkit."},
            )
    finally:
        app_state.ragdoll = previous

    assert response.status_code == 200
    assert fake.sources is not None
    assert fake.sources[0].page_content == "Ragdoll is a retrieval toolkit."


def test_chat_route_uses_the_legacy_vector_query_helper() -> None:
    class FakeRagdoll:
        vector_store = object()
        graph_retriever = None

        def query_sync(self, question, *, retriever_mode, k):
            return {
                "answer": f"Answer for: {question}",
                "documents": [],
                "num_documents": 0,
                "retriever_used": retriever_mode,
            }

    app_state = get_app_state()
    previous = app_state.ragdoll
    app_state.ragdoll = FakeRagdoll()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                data={"question": "What is Ragdoll?", "retriever": "vector"},
            )
    finally:
        app_state.ragdoll = previous

    assert response.status_code == 200
    assert "Answer for: What is Ragdoll?" in response.text


def test_first_provider_action_initializes_ragdoll_lazily(monkeypatch) -> None:
    class FakeRagdoll:
        def __init__(self, *, app_config) -> None:
            self.app_config = app_config
            self.embedding_model = object()
            self.vector_store = object()

        async def ingest_with_graph(self, sources, *, options):
            return {"stats": {"documents": len(sources)}, "graph": None}

    monkeypatch.setattr(demo_main, "Ragdoll", FakeRagdoll)
    app_state = get_app_state()
    previous = app_state.ragdoll
    app_state.ragdoll = None
    try:
        with TestClient(app) as client:
            response = client.post(
                "/ingest",
                data={"text_input": "Initialize on demand."},
            )
    finally:
        app_state.ragdoll = previous

    assert response.status_code == 200


def test_provider_initialization_failure_is_rendered_as_an_actionable_error(
    monkeypatch,
) -> None:
    class UnavailableRagdoll:
        def __init__(self, *, app_config) -> None:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(demo_main, "Ragdoll", UnavailableRagdoll)
    app_state = get_app_state()
    previous = app_state.ragdoll
    app_state.ragdoll = None
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/ingest",
                data={"text_input": "Try provider-backed ingestion."},
            )
    finally:
        app_state.ragdoll = previous

    assert response.status_code == 503
    assert "Check the demo configuration and provider credentials" in response.text
    assert "provider unavailable" not in response.text


def test_loader_accepts_a_text_snippet_without_files_or_urls() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/load",
            data={"text_input": "A standalone text snippet."},
        )

    assert response.status_code == 200
    assert "A standalone text snippet." in response.text


def test_ingest_route_renders_graph_persistence_failures() -> None:
    class FailingRagdoll:
        async def ingest_with_graph(self, sources, *, options):
            raise GraphWriteError("Graph store rejected the graph write")

    app_state = get_app_state()
    previous = app_state.ragdoll
    app_state.ragdoll = FailingRagdoll()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/ingest",
                data={"text_input": "A document containing Unicode: A → B"},
            )
    finally:
        app_state.ragdoll = previous

    assert response.status_code == 500
    assert "Unable to persist the extracted graph" in response.text
