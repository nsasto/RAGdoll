"""Versioned graph indexing published by the corpus generation switch."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping, Protocol, Sequence

from langchain_core.documents import Document

from ragdoll.corpus import StagedGeneration
from ragdoll.entity_extraction.models import Graph
from ragdoll.errors import TenantIsolationError
from ragdoll.generation_state import GenerationStateStore


class GraphBackend(Protocol):
    async def upsert(self, graph: Graph, scope: Mapping[str, str]) -> None: ...
    async def search(
        self, question: str, *, k: int, filters: Mapping[str, object]
    ) -> Sequence[Document]: ...
    async def delete(self, filters: Mapping[str, object]) -> None: ...


class VersionedGraphIndex:
    _reserved = {"tenant_id", "corpus_id", "generation_id"}

    def __init__(self, backend: GraphBackend, state: GenerationStateStore) -> None:
        self.backend = backend
        self.state = state

    async def stage(self, generation: StagedGeneration, graph: Graph) -> None:
        await self.backend.upsert(
            graph,
            {
                "tenant_id": str(generation.tenant),
                "corpus_id": str(generation.corpus),
                "generation_id": str(generation.id),
            },
        )

    async def query(
        self,
        *,
        tenant: str,
        corpus: str,
        question: str,
        k: int = 4,
        filters: Mapping[str, object] | None = None,
    ) -> list[Document]:
        requested = dict(filters or {})
        conflict = self._reserved.intersection(requested)
        if conflict:
            raise TenantIsolationError(
                f"Reserved graph filters cannot be overridden: {sorted(conflict)}"
            )
        active = await self.state.active(tenant, corpus)
        if active is None:
            return []
        enforced = {
            "tenant_id": tenant,
            "corpus_id": corpus,
            "generation_id": str(active),
            **requested,
        }
        return list(await self.backend.search(question, k=k, filters=enforced))

    async def delete(self, tenant: str, corpus: str) -> None:
        await self.backend.delete({"tenant_id": tenant, "corpus_id": corpus})

    async def discard(self, generation: StagedGeneration) -> None:
        await self.backend.delete(
            {
                "tenant_id": str(generation.tenant),
                "corpus_id": str(generation.corpus),
                "generation_id": str(generation.id),
            }
        )


class InMemoryGraphBackend:
    """Local graph adapter with generation-aware entity search."""

    def __init__(self) -> None:
        self._graphs: dict[tuple[str, str, str], Graph] = {}

    async def upsert(self, graph: Graph, scope: Mapping[str, str]) -> None:
        key = (
            scope["tenant_id"],
            scope["corpus_id"],
            scope["generation_id"],
        )
        self._graphs[key] = graph.model_copy(deep=True)

    async def search(
        self, question: str, *, k: int, filters: Mapping[str, object]
    ) -> Sequence[Document]:
        key = (
            str(filters["tenant_id"]),
            str(filters["corpus_id"]),
            str(filters["generation_id"]),
        )
        graph = self._graphs.get(key)
        if graph is None:
            return []
        terms = [term.casefold() for term in question.split() if term]
        documents = []
        for node in graph.nodes:
            properties = dict(node.properties or node.metadata or {})
            extra = {
                key: value
                for key, value in filters.items()
                if key not in VersionedGraphIndex._reserved
            }
            candidate_metadata = {**properties, "node_type": node.type}
            if not all(
                candidate_metadata.get(key) == value for key, value in extra.items()
            ):
                continue
            haystack = f"{node.name} {node.label or ''} {properties}".casefold()
            if terms and not any(term in haystack for term in terms):
                continue
            metadata = {**filters, "node_id": node.id, "node_type": node.type}
            if properties.get("vector_id"):
                metadata["vector_id"] = properties["vector_id"]
                metadata["chunk_id"] = properties["vector_id"]
            documents.append(
                Document(
                    page_content=f"{node.name}: {node.label or properties}",
                    metadata=metadata,
                )
            )
        return documents[:k]

    async def delete(self, filters: Mapping[str, object]) -> None:
        keys = [
            key
            for key in self._graphs
            if key[0] == filters.get("tenant_id")
            and key[1] == filters.get("corpus_id")
            and (
                "generation_id" not in filters or key[2] == filters.get("generation_id")
            )
        ]
        for key in keys:
            del self._graphs[key]


class Neo4jGraphBackend:
    """Generation-aware Neo4j adapter for the scaled graph profile."""

    def __init__(
        self,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        driver: Any | None = None,
    ) -> None:
        if driver is None:
            if not uri or not user or password is None:
                raise ValueError("Neo4j requires uri, user, and password")
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "Neo4j graph indexing requires `pip install python-ragdoll[graph]`"
                ) from exc
            driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver = driver

    async def upsert(self, graph: Graph, scope: Mapping[str, str]) -> None:
        await asyncio.to_thread(self._upsert_sync, graph, dict(scope))

    async def search(
        self, question: str, *, k: int, filters: Mapping[str, object]
    ) -> Sequence[Document]:
        return await asyncio.to_thread(self._search_sync, question, k, dict(filters))

    async def delete(self, filters: Mapping[str, object]) -> None:
        await asyncio.to_thread(self._delete_sync, dict(filters))

    def _upsert_sync(self, graph: Graph, scope: dict[str, str]) -> None:
        with self.driver.session() as session:
            for node in graph.nodes:
                properties = _neo4j_properties(node.properties or node.metadata or {})
                key = _node_key(scope, node.id)
                session.run(
                    """
                    MERGE (n:RagdollNode {ragdoll_key: $key})
                    SET n += $properties,
                        n.node_id = $node_id,
                        n.name = $name,
                        n.node_type = $node_type,
                        n.label = $label,
                        n.tenant_id = $tenant_id,
                        n.corpus_id = $corpus_id,
                        n.generation_id = $generation_id
                    """,
                    key=key,
                    properties=properties,
                    node_id=node.id,
                    name=node.name,
                    node_type=node.type,
                    label=node.label,
                    **scope,
                )
            for edge in graph.edges:
                session.run(
                    """
                    MATCH (source:RagdollNode {ragdoll_key: $source_key})
                    MATCH (target:RagdollNode {ragdoll_key: $target_key})
                    MERGE (source)-[r:RAGDOLL_RELATION {edge_id: $edge_id}]->(target)
                    SET r.relationship_type = $relationship_type,
                        r.metadata_json = $metadata_json,
                        r.source_document_id = $source_document_id,
                        r.tenant_id = $tenant_id,
                        r.corpus_id = $corpus_id,
                        r.generation_id = $generation_id
                    """,
                    source_key=_node_key(scope, edge.source),
                    target_key=_node_key(scope, edge.target),
                    edge_id=edge.id,
                    relationship_type=edge.type,
                    metadata_json=json.dumps(edge.metadata, default=str),
                    source_document_id=edge.source_document_id,
                    **scope,
                )

    def _search_sync(
        self, question: str, k: int, filters: dict[str, object]
    ) -> list[Document]:
        extra = {
            key: value
            for key, value in filters.items()
            if key not in VersionedGraphIndex._reserved
        }
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n:RagdollNode)
                WHERE n.tenant_id = $tenant_id
                  AND n.corpus_id = $corpus_id
                  AND n.generation_id = $generation_id
                  AND (toLower(n.name) CONTAINS toLower($question)
                       OR toLower(coalesce(n.label, '')) CONTAINS toLower($question))
                RETURN properties(n) AS properties
                LIMIT $limit
                """,
                tenant_id=filters["tenant_id"],
                corpus_id=filters["corpus_id"],
                generation_id=filters["generation_id"],
                question=question,
                limit=k * 4 if extra else k,
            )
            rows = [dict(row["properties"]) for row in result]
        rows = [
            row
            for row in rows
            if all(row.get(key) == value for key, value in extra.items())
        ][:k]
        return [
            Document(
                page_content=f"{row.get('name', '')}: {row.get('label', '')}",
                metadata={
                    **filters,
                    "node_id": row.get("node_id"),
                    "node_type": row.get("node_type"),
                    "vector_id": row.get("vector_id"),
                    "chunk_id": row.get("vector_id"),
                },
            )
            for row in rows
        ]

    def _delete_sync(self, filters: dict[str, object]) -> None:
        with self.driver.session() as session:
            session.run(
                """
                MATCH (n:RagdollNode)
                WHERE n.tenant_id = $tenant_id AND n.corpus_id = $corpus_id
                  AND ($generation_id IS NULL OR n.generation_id = $generation_id)
                DETACH DELETE n
                """,
                tenant_id=filters["tenant_id"],
                corpus_id=filters["corpus_id"],
                generation_id=filters.get("generation_id"),
            )


def _node_key(scope: Mapping[str, str], node_id: str) -> str:
    return ":".join(
        [scope["tenant_id"], scope["corpus_id"], scope["generation_id"], node_id]
    )


def _neo4j_properties(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value
            if isinstance(value, (str, int, float, bool)) or value is None
            else json.dumps(value, default=str)
        )
        for key, value in properties.items()
    }
