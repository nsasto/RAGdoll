"""Stable public contracts shared by ingestion, indexing, and querying."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any, Mapping, NewType, Sequence

from langchain_core.documents import Document

TenantId = NewType("TenantId", str)
CorpusId = NewType("CorpusId", str)
GenerationId = NewType("GenerationId", str)
JobId = NewType("JobId", str)

Source = str | Path | Document


def source_identity(source: Source) -> str:
    """Return a stable identity suitable for outcomes and checkpoints."""
    if isinstance(source, Document):
        metadata = source.metadata or {}
        explicit = metadata.get("source") or metadata.get("document_id")
        if explicit:
            return str(explicit)
        return hashlib.sha256(source.page_content.encode("utf-8")).hexdigest()
    return str(source)


def _required_identifier(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class IngestionSpec:
    """Everything a caller must provide to ingest a corpus safely."""

    corpus: CorpusId | str
    sources: Sequence[Source]
    tenant: TenantId | str = TenantId("default")
    idempotency_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tenant", TenantId(_required_identifier(self.tenant, "tenant"))
        )
        object.__setattr__(
            self, "corpus", CorpusId(_required_identifier(self.corpus, "corpus"))
        )
        normalized_sources = tuple(self.sources)
        if not normalized_sources:
            raise ValueError("sources must contain at least one source")
        object.__setattr__(self, "sources", normalized_sources)
        object.__setattr__(self, "metadata", dict(self.metadata))


class ItemStatus(str, Enum):
    INDEXED = "indexed"
    UNCHANGED = "unchanged"
    RETRYABLE = "retryable"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ItemOutcome:
    source_id: str
    status: ItemStatus
    document_id: str | None = None
    chunk_ids: tuple[str, ...] = ()
    code: str | None = None
    detail: str | None = None

    @property
    def retryable(self) -> bool:
        return self.status is ItemStatus.RETRYABLE

    @property
    def succeeded(self) -> bool:
        return self.status in {ItemStatus.INDEXED, ItemStatus.UNCHANGED}


@dataclass(frozen=True, slots=True)
class BatchWriteOutcome:
    ids: tuple[str, ...]
    attempted_count: int

    @property
    def succeeded_count(self) -> int:
        return len(self.ids)
