"""Versioned corpus indexing with promotion-based visibility."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from langchain_core.documents import Document

from ragdoll.contracts import CorpusId, GenerationId, TenantId
from ragdoll.errors import (
    BatchWriteError,
    GenerationNotFoundError,
    TenantIsolationError,
)
from ragdoll.generation_state import (
    FileGenerationStateStore,
    GenerationRecord,
    GenerationStateStore,
    MemoryGenerationStateStore,
)


@dataclass(frozen=True, slots=True)
class IndexCapabilities:
    metadata_filters: bool
    deletion: bool
    versioned_generations: bool = True
    graph_linkage: bool = False


@dataclass(frozen=True, slots=True)
class StagedGeneration:
    id: GenerationId
    tenant: TenantId
    corpus: CorpusId
    document_count: int
    checksum: str
    unchanged: bool = False
    documents: tuple[Document, ...] = ()


class CorpusIndex(Protocol):
    capabilities: IndexCapabilities

    async def stage(
        self,
        tenant: str,
        corpus: str,
        documents: Sequence[Document],
    ) -> StagedGeneration: ...

    async def promote(
        self,
        generation: StagedGeneration | str,
        *,
        tenant: str | None = None,
        corpus: str | None = None,
    ) -> GenerationId: ...

    async def query(
        self,
        tenant: str,
        corpus: str,
        text: str,
        *,
        k: int = 4,
        filters: Mapping[str, object] | None = None,
    ) -> list[Document]: ...

    async def rollback(self, tenant: str, corpus: str) -> GenerationId: ...

    async def discard(self, generation: StagedGeneration | str) -> None: ...

    async def delete(self, tenant: str, corpus: str) -> None: ...


class InMemoryCorpusIndex:
    """Local adapter implementing the same visibility contract as durable stores."""

    capabilities = IndexCapabilities(metadata_filters=True, deletion=True)

    def __init__(self) -> None:
        self._generations: dict[
            GenerationId, tuple[TenantId, CorpusId, list[Document]]
        ] = {}
        self._active: dict[tuple[TenantId, CorpusId], GenerationId] = {}
        self._history: dict[tuple[TenantId, CorpusId], list[GenerationId]] = {}

    async def stage(
        self,
        tenant: str,
        corpus: str,
        documents: Sequence[Document],
    ) -> StagedGeneration:
        tenant_id = TenantId(_identifier(tenant, "tenant"))
        corpus_id = CorpusId(_identifier(corpus, "corpus"))
        generation_id, checksum, prepared = _prepare_generation(
            tenant_id, corpus_id, documents
        )
        key = (tenant_id, corpus_id)
        active = self._active.get(key)
        if active is not None:
            active_documents = self._generations[active][2]
            if _generation_checksum(active_documents) == checksum:
                return StagedGeneration(
                    id=active,
                    tenant=tenant_id,
                    corpus=corpus_id,
                    document_count=len(prepared),
                    checksum=checksum,
                    unchanged=True,
                    documents=tuple(
                        _prepare_document(doc, tenant_id, corpus_id, active, position)
                        for position, doc in enumerate(documents)
                    ),
                )
        self._generations[generation_id] = (tenant_id, corpus_id, prepared)
        return StagedGeneration(
            id=generation_id,
            tenant=tenant_id,
            corpus=corpus_id,
            document_count=len(prepared),
            checksum=checksum,
            documents=tuple(prepared),
        )

    async def promote(
        self,
        generation: StagedGeneration | str,
        *,
        tenant: str | None = None,
        corpus: str | None = None,
    ) -> GenerationId:
        generation_id = GenerationId(
            generation.id if isinstance(generation, StagedGeneration) else generation
        )
        stored = self._generations.get(generation_id)
        if stored is None:
            raise GenerationNotFoundError(f"Unknown generation: {generation_id}")
        actual_tenant, actual_corpus, _ = stored
        if tenant is not None and TenantId(tenant) != actual_tenant:
            raise TenantIsolationError("Generation belongs to another tenant")
        if corpus is not None and CorpusId(corpus) != actual_corpus:
            raise TenantIsolationError("Generation belongs to another corpus")
        key = (actual_tenant, actual_corpus)
        active = self._active.get(key)
        if active != generation_id:
            if active is not None:
                self._history.setdefault(key, []).append(active)
            self._active[key] = generation_id
        return generation_id

    async def rollback(self, tenant: str, corpus: str) -> GenerationId:
        key = (TenantId(tenant), CorpusId(corpus))
        history = self._history.get(key, [])
        if not history:
            raise GenerationNotFoundError("No previous generation is available")
        previous = history.pop()
        current = self._active.get(key)
        if current is not None:
            history.append(current)
        self._active[key] = previous
        return previous

    async def query(
        self,
        tenant: str,
        corpus: str,
        text: str,
        *,
        k: int = 4,
        filters: Mapping[str, object] | None = None,
    ) -> list[Document]:
        key = (TenantId(tenant), CorpusId(corpus))
        generation_id = self._active.get(key)
        if generation_id is None:
            return []
        documents = self._generations[generation_id][2]
        candidates = [doc for doc in documents if _matches(doc, filters)]
        terms = {term.casefold() for term in text.split() if term}
        ranked = sorted(
            candidates,
            key=lambda doc: sum(term in doc.page_content.casefold() for term in terms),
            reverse=True,
        )
        return ranked[:k]

    async def delete(self, tenant: str, corpus: str) -> None:
        key = (TenantId(tenant), CorpusId(corpus))
        generation_ids = [
            generation_id
            for generation_id, (
                item_tenant,
                item_corpus,
                _,
            ) in self._generations.items()
            if (item_tenant, item_corpus) == key
        ]
        for generation_id in generation_ids:
            del self._generations[generation_id]
        self._active.pop(key, None)
        self._history.pop(key, None)

    async def discard(self, generation: StagedGeneration | str) -> None:
        generation_id = GenerationId(
            generation.id if isinstance(generation, StagedGeneration) else generation
        )
        stored = self._generations.get(generation_id)
        if stored is None:
            return
        key = (stored[0], stored[1])
        if self._active.get(key) == generation_id:
            from ragdoll.errors import GenerationActiveError

            raise GenerationActiveError("The active generation cannot be discarded")
        del self._generations[generation_id]
        history = self._history.get(key)
        if history is not None:
            self._history[key] = [item for item in history if item != generation_id]


class VectorCorpusIndex:
    """Vector-store adapter with durable generation visibility metadata."""

    capabilities = IndexCapabilities(metadata_filters=True, deletion=True)
    _reserved_filters = {"tenant_id", "corpus_id", "generation_id"}

    def __init__(
        self,
        store: Any,
        *,
        state_path: str | Path | None = None,
        state_store: GenerationStateStore | None = None,
    ) -> None:
        if state_path is not None and state_store is not None:
            raise ValueError("Provide state_path or state_store, not both")
        self.store = store
        self.state = state_store or (
            FileGenerationStateStore(state_path)
            if state_path is not None
            else MemoryGenerationStateStore()
        )

    async def stage(
        self,
        tenant: str,
        corpus: str,
        documents: Sequence[Document],
    ) -> StagedGeneration:
        tenant_id = TenantId(_identifier(tenant, "tenant"))
        corpus_id = CorpusId(_identifier(corpus, "corpus"))
        generation_id, checksum, prepared = _prepare_generation(
            tenant_id, corpus_id, documents
        )
        active = await self.state.active(str(tenant_id), str(corpus_id))
        if active is not None:
            active_record = await self.state.get(active)
            if active_record.checksum == checksum:
                return StagedGeneration(
                    id=active,
                    tenant=tenant_id,
                    corpus=corpus_id,
                    document_count=len(prepared),
                    checksum=checksum,
                    unchanged=True,
                    documents=tuple(
                        _prepare_document(doc, tenant_id, corpus_id, active, position)
                        for position, doc in enumerate(documents)
                    ),
                )
        desired_ids = [str(document.metadata["vector_id"]) for document in prepared]
        ids = list(await self.store.aadd_documents(prepared, ids=desired_ids))
        valid_ids = [item_id for item_id in ids if item_id]
        if len(valid_ids) != len(prepared):
            if valid_ids:
                self.store.delete(valid_ids)
            raise BatchWriteError(
                "Vector adapter returned incomplete IDs while staging generation",
                failed_count=len(prepared) - len(valid_ids),
                succeeded_ids=valid_ids,
            )
        if valid_ids != desired_ids:
            if valid_ids:
                self.store.delete(valid_ids)
            raise BatchWriteError(
                "Vector adapter did not preserve stable chunk IDs",
                failed_count=len(prepared),
                succeeded_ids=(),
            )
        await self.state.put(
            GenerationRecord(
                id=generation_id,
                tenant=tenant_id,
                corpus=corpus_id,
                vector_ids=tuple(valid_ids),
                document_count=len(prepared),
                checksum=checksum,
            )
        )
        return StagedGeneration(
            id=generation_id,
            tenant=tenant_id,
            corpus=corpus_id,
            document_count=len(prepared),
            checksum=checksum,
            documents=tuple(prepared),
        )

    async def promote(
        self,
        generation: StagedGeneration | str,
        *,
        tenant: str | None = None,
        corpus: str | None = None,
    ) -> GenerationId:
        generation_id = GenerationId(
            generation.id if isinstance(generation, StagedGeneration) else generation
        )
        stored = await self.state.get(generation_id)
        if len(stored.vector_ids) != stored.document_count or not stored.checksum:
            raise BatchWriteError(
                "Generation failed validation and cannot be promoted",
                failed_count=max(0, stored.document_count - len(stored.vector_ids)),
                succeeded_ids=stored.vector_ids,
            )
        return await self.state.promote(generation_id, tenant=tenant, corpus=corpus)

    async def rollback(self, tenant: str, corpus: str) -> GenerationId:
        return await self.state.rollback(tenant, corpus)

    async def query(
        self,
        tenant: str,
        corpus: str,
        text: str,
        *,
        k: int = 4,
        filters: Mapping[str, object] | None = None,
    ) -> list[Document]:
        generation_id = await self.state.active(tenant, corpus)
        if generation_id is None:
            return []
        requested = dict(filters or {})
        conflicting = self._reserved_filters.intersection(requested)
        if conflicting:
            raise TenantIsolationError(
                f"Reserved corpus filters cannot be overridden: {sorted(conflicting)}"
            )
        enforced = {
            "tenant_id": tenant,
            "corpus_id": corpus,
            "generation_id": str(generation_id),
            **requested,
        }
        return list(await self.store.asimilarity_search(text, k=k, filter=enforced))

    async def delete(self, tenant: str, corpus: str) -> None:
        generations = await self.state.list_scope(tenant, corpus)
        vector_ids = [
            vector_id
            for generation in generations
            for vector_id in generation.vector_ids
        ]
        if vector_ids:
            self.store.delete(vector_ids)
        await self.state.delete_scope(tenant, corpus)

    async def discard(self, generation: StagedGeneration | str) -> None:
        generation_id = GenerationId(
            generation.id if isinstance(generation, StagedGeneration) else generation
        )
        record = await self.state.get(generation_id)
        # Remove visibility metadata first. If the backend delete then fails, only
        # unreachable vectors remain; an active generation is never damaged.
        await self.state.delete_generation(generation_id)
        if record.vector_ids:
            self.store.delete(list(record.vector_ids))


def prepare_corpus_documents(
    tenant: str, corpus: str, documents: Sequence[Document]
) -> list[Document]:
    """Attach stable logical identities before a generation allocates vector IDs."""
    tenant_id = TenantId(_identifier(tenant, "tenant"))
    corpus_id = CorpusId(_identifier(corpus, "corpus"))
    # Generation is deliberately absent here; each index adds its staged generation.
    placeholder = GenerationId("staging")
    prepared = [
        _prepare_document(document, tenant_id, corpus_id, placeholder, position)
        for position, document in enumerate(documents)
    ]
    for document in prepared:
        document.metadata.pop("generation_id", None)
        document.metadata.pop("vector_id", None)
    return prepared


def _identifier(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _prepare_document(
    document: Document,
    tenant: TenantId,
    corpus: CorpusId,
    generation: GenerationId,
    position: int,
) -> Document:
    content_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
    metadata = dict(document.metadata)
    source = metadata.get("source") or metadata.get("path")
    document_id = metadata.get("document_id") or (
        hashlib.sha256(f"{tenant}:{corpus}:{source}".encode("utf-8")).hexdigest()
        if source
        else content_hash
    )
    chunk_id = (
        metadata.get("chunk_id")
        or hashlib.sha256(
            f"{document_id}:{position}:{document.page_content}".encode("utf-8")
        ).hexdigest()
    )
    metadata.update(
        tenant_id=str(tenant),
        corpus_id=str(corpus),
        generation_id=str(generation),
        document_id=document_id,
        chunk_id=chunk_id,
        content_hash=content_hash,
        vector_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"ragdoll-vector:{generation}:{chunk_id}",
            )
        ),
    )
    return Document(page_content=document.page_content, metadata=metadata)


def _prepare_generation(
    tenant: TenantId,
    corpus: CorpusId,
    documents: Sequence[Document],
) -> tuple[GenerationId, str, list[Document]]:
    logical = [
        _prepare_document(doc, tenant, corpus, GenerationId("pending"), position)
        for position, doc in enumerate(documents)
    ]
    checksum = _generation_checksum(logical)
    generation = GenerationId(
        str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"ragdoll-generation:{tenant}:{corpus}:{checksum}",
            )
        )
    )
    prepared = [
        _prepare_document(doc, tenant, corpus, generation, position)
        for position, doc in enumerate(documents)
    ]
    return generation, checksum, prepared


def _generation_checksum(documents: Sequence[Document]) -> str:
    digest = hashlib.sha256()
    for chunk_id in sorted(
        str(document.metadata["chunk_id"]) for document in documents
    ):
        digest.update(chunk_id.encode("ascii"))
    return digest.hexdigest()


def _matches(document: Document, filters: Mapping[str, object] | None) -> bool:
    return not filters or all(
        document.metadata.get(key) == value for key, value in filters.items()
    )


def _assert_scope(
    actual_tenant: TenantId,
    actual_corpus: CorpusId,
    tenant: str | None,
    corpus: str | None,
) -> None:
    if tenant is not None and TenantId(tenant) != actual_tenant:
        raise TenantIsolationError("Generation belongs to another tenant")
    if corpus is not None and CorpusId(corpus) != actual_corpus:
        raise TenantIsolationError("Generation belongs to another corpus")
