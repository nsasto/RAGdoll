import pytest

from ragdoll.entity_extraction.models import Graph, GraphNode
from ragdoll.graph_index import Neo4jGraphBackend


class Session:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def run(self, query, **params):
        self.calls.append((query, params))
        if "RETURN properties(n)" in query:
            return [
                {
                    "properties": {
                        "name": "Authentication",
                        "label": "concept",
                        "node_id": "auth",
                        "node_type": "concept",
                        "vector_id": "chunk-1",
                    }
                }
            ]
        return []


class Driver:
    def __init__(self):
        self.sessions = []

    def session(self):
        session = Session()
        self.sessions.append(session)
        return session


@pytest.mark.asyncio
async def test_neo4j_backend_persists_and_queries_generation_scope():
    driver = Driver()
    backend = Neo4jGraphBackend(driver=driver)
    scope = {
        "tenant_id": "acme",
        "corpus_id": "docs",
        "generation_id": "generation-1",
    }
    await backend.upsert(
        Graph(
            nodes=[
                GraphNode(
                    id="auth",
                    type="concept",
                    name="Authentication",
                    properties={"vector_id": "chunk-1"},
                )
            ]
        ),
        scope,
    )
    results = await backend.search("Authentication", k=4, filters=scope)

    upsert_params = driver.sessions[0].calls[0][1]
    assert upsert_params["tenant_id"] == "acme"
    assert upsert_params["generation_id"] == "generation-1"
    assert results[0].metadata["chunk_id"] == "chunk-1"
