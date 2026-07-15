"""Durable ingestion jobs with an inline execution adapter."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, Sequence

from langchain_core.documents import Document

from ragdoll.contracts import (
    GenerationId,
    IngestionSpec,
    ItemOutcome,
    ItemStatus,
    JobId,
    source_identity,
)
from ragdoll.corpus import CorpusIndex, StagedGeneration
from ragdoll.errors import CancelledError, JobLeaseUnavailableError, RagdollError
from ragdoll.chunkers import split_documents
from ragdoll.corpus import prepare_corpus_documents
from ragdoll.graph_index import VersionedGraphIndex
from ragdoll.observability import EventSink, NullEventSink
from ragdoll.quarantine import (
    MemoryQuarantineStore,
    QuarantineStore,
    quarantined_source,
)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class JobProgress:
    completed: int
    total: int
    stage: str


@dataclass(frozen=True, slots=True)
class IngestionResult:
    job_id: JobId
    status: JobStatus
    items: tuple[ItemOutcome, ...]
    generation_id: GenerationId | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class JobRecord:
    id: JobId
    spec: IngestionSpec
    status: JobStatus = JobStatus.PENDING
    progress: JobProgress = field(
        default_factory=lambda: JobProgress(completed=0, total=0, stage="pending")
    )
    result: IngestionResult | None = None
    cancel_requested: bool = False
    attempt: int = 0
    worker_id: str | None = None
    lease_until: datetime | None = None
    staged_generation_id: GenerationId | None = None
    staged_checksum: str | None = None
    staged_document_count: int = 0
    checkpoint_items: tuple[ItemOutcome, ...] = ()


class JobStore(Protocol):
    async def create(self, record: JobRecord) -> JobRecord: ...
    async def get(self, job_id: JobId) -> JobRecord: ...
    async def save(self, record: JobRecord) -> None: ...
    async def find_idempotent(self, spec: IngestionSpec) -> JobRecord | None: ...
    async def claim(
        self, job_id: JobId, worker_id: str, lease_seconds: float
    ) -> JobRecord | None: ...


JobRunner = Callable[[JobId], Awaitable[None]]


class ExecutionAdapter(Protocol):
    distributed: bool

    async def submit(self, job_id: JobId, runner: JobRunner) -> None: ...
    async def cancel(self, job_id: JobId) -> None: ...
    async def wait(self, job_id: JobId) -> None: ...


class InlineExecutionAdapter:
    """Zero-infrastructure execution with bounded concurrency in the job module."""

    distributed = False

    def __init__(self) -> None:
        self._tasks: dict[JobId, asyncio.Task[None]] = {}

    async def submit(self, job_id: JobId, runner: JobRunner) -> None:
        task = self._tasks.get(job_id)
        if task is None or task.done():
            self._tasks[job_id] = asyncio.create_task(runner(job_id))

    async def cancel(self, job_id: JobId) -> None:
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()

    async def wait(self, job_id: JobId) -> None:
        task = self._tasks.get(job_id)
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass


class CeleryExecutionAdapter:
    """Queue adapter dispatching durable job IDs to a Celery worker task."""

    distributed = True

    def __init__(
        self, app: Any, *, task_name: str = "ragdoll.execute_ingestion"
    ) -> None:
        self.app = app
        self.task_name = task_name
        self._task_ids: dict[JobId, str] = {}

    async def submit(self, job_id: JobId, runner: JobRunner) -> None:
        result = await asyncio.to_thread(
            self.app.send_task, self.task_name, args=[str(job_id)]
        )
        self._task_ids[job_id] = str(result.id)

    async def cancel(self, job_id: JobId) -> None:
        task_id = self._task_ids.get(job_id)
        if task_id:
            await asyncio.to_thread(self.app.control.revoke, task_id, terminate=False)

    async def wait(self, job_id: JobId) -> None:
        # Completion is authoritative in JobStore, not Celery result backends.
        return None


class MemoryJobStore:
    """In-process job adapter used by the zero-configuration profile."""

    def __init__(self) -> None:
        self._records: dict[JobId, JobRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, record: JobRecord) -> JobRecord:
        async with self._lock:
            if record.spec.idempotency_key:
                for existing in self._records.values():
                    candidate = existing.spec
                    if (
                        candidate.tenant == record.spec.tenant
                        and candidate.corpus == record.spec.corpus
                        and candidate.idempotency_key == record.spec.idempotency_key
                    ):
                        return existing
            self._records[record.id] = record
        return record

    async def get(self, job_id: JobId) -> JobRecord:
        async with self._lock:
            return self._records[job_id]

    async def save(self, record: JobRecord) -> None:
        async with self._lock:
            self._records[record.id] = record

    async def find_idempotent(self, spec: IngestionSpec) -> JobRecord | None:
        if not spec.idempotency_key:
            return None
        async with self._lock:
            for record in self._records.values():
                candidate = record.spec
                if (
                    candidate.tenant == spec.tenant
                    and candidate.corpus == spec.corpus
                    and candidate.idempotency_key == spec.idempotency_key
                ):
                    return record
        return None

    async def claim(
        self, job_id: JobId, worker_id: str, lease_seconds: float
    ) -> JobRecord | None:
        async with self._lock:
            record = self._records[job_id]
            claimed = _claim_record(record, worker_id, lease_seconds)
            if claimed is not None:
                self._records[job_id] = claimed
            return claimed


class FileJobStore:
    """Atomic JSON job adapter for local and single-host deployments."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._lock = asyncio.Lock()

    async def create(self, record: JobRecord) -> JobRecord:
        async with self._lock:
            existing = self._find_idempotent_sync(record.spec)
            if existing is not None:
                return existing
            self._save_sync(record)
            return record

    async def get(self, job_id: JobId) -> JobRecord:
        async with self._lock:
            path = self.directory / f"{job_id}.json"
            return _record_from_json(json.loads(path.read_text(encoding="utf-8")))

    async def save(self, record: JobRecord) -> None:
        async with self._lock:
            self._save_sync(record)

    async def find_idempotent(self, spec: IngestionSpec) -> JobRecord | None:
        if not spec.idempotency_key:
            return None
        async with self._lock:
            return self._find_idempotent_sync(spec)

    async def claim(
        self, job_id: JobId, worker_id: str, lease_seconds: float
    ) -> JobRecord | None:
        async with self._lock:
            path = self.directory / f"{job_id}.json"
            record = _record_from_json(json.loads(path.read_text(encoding="utf-8")))
            claimed = _claim_record(record, worker_id, lease_seconds)
            if claimed is not None:
                self._save_sync(claimed)
            return claimed

    def _find_idempotent_sync(self, spec: IngestionSpec) -> JobRecord | None:
        if not spec.idempotency_key:
            return None
        if not self.directory.exists():
            return None
        for path in self.directory.glob("*.json"):
            record = _record_from_json(json.loads(path.read_text(encoding="utf-8")))
            candidate = record.spec
            if (
                candidate.tenant == spec.tenant
                and candidate.corpus == spec.corpus
                and candidate.idempotency_key == spec.idempotency_key
            ):
                return record
        return None

    def _save_sync(self, record: JobRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{record.id}.json"
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(_record_to_json(record), indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)


class PostgresJobStore:
    """Shared durable job adapter for multi-worker deployments."""

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "ragdoll_ingestion_jobs",
        connect: Callable[..., Any] | None = None,
    ) -> None:
        if not dsn:
            raise ValueError("Postgres job store requires a DSN")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError("Postgres table name must be a simple identifier")
        self.dsn = dsn
        self.table = table
        self._connect_override = connect
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def create(self, record: JobRecord) -> JobRecord:
        await self._ensure_schema()
        return await asyncio.to_thread(self._create_sync, record)

    async def get(self, job_id: JobId) -> JobRecord:
        await self._ensure_schema()
        return await asyncio.to_thread(self._get_sync, job_id)

    async def save(self, record: JobRecord) -> None:
        await self._ensure_schema()
        await asyncio.to_thread(self._save_sync, record)

    async def find_idempotent(self, spec: IngestionSpec) -> JobRecord | None:
        if not spec.idempotency_key:
            return None
        await self._ensure_schema()
        return await asyncio.to_thread(self._find_sync, spec)

    async def claim(
        self, job_id: JobId, worker_id: str, lease_seconds: float
    ) -> JobRecord | None:
        await self._ensure_schema()
        return await asyncio.to_thread(
            self._claim_sync, job_id, worker_id, lease_seconds
        )

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if not self._initialized:
                await asyncio.to_thread(self._initialize_sync)
                self._initialized = True

    def _connect(self):
        if self._connect_override is not None:
            return self._connect_override(self.dsn)
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PostgresJobStore requires `pip install python-ragdoll[scaled]`"
            ) from exc
        return psycopg.connect(self.dsn)

    def _initialize_sync(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    tenant TEXT NOT NULL,
                    corpus TEXT NOT NULL,
                    idempotency_key TEXT NULL,
                    payload JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """)
            cursor.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {self.table}_idempotency_uq
                ON {self.table} (tenant, corpus, idempotency_key)
                WHERE idempotency_key IS NOT NULL
                """)

    def _create_sync(self, record: JobRecord) -> JobRecord:
        payload = json.dumps(_record_to_json(record), default=str)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {self.table}
                    (id, tenant, corpus, idempotency_key, payload)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT DO NOTHING
                RETURNING payload
                """,
                (
                    str(record.id),
                    str(record.spec.tenant),
                    str(record.spec.corpus),
                    record.spec.idempotency_key,
                    payload,
                ),
            )
            row = cursor.fetchone()
            if row is not None:
                return _record_from_json(_json_value(row[0]))
        existing = self._find_sync(record.spec)
        if existing is None:
            raise RuntimeError("Postgres job insert conflicted without an existing job")
        return existing

    def _get_sync(self, job_id: JobId) -> JobRecord:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT payload FROM {self.table} WHERE id = %s", (str(job_id),)
            )
            row = cursor.fetchone()
        if row is None:
            raise KeyError(job_id)
        return _record_from_json(_json_value(row[0]))

    def _save_sync(self, record: JobRecord) -> None:
        payload = json.dumps(_record_to_json(record), default=str)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE {self.table}
                SET payload = %s::jsonb, updated_at = NOW()
                WHERE id = %s
                """,
                (payload, str(record.id)),
            )
            if cursor.rowcount != 1:
                raise KeyError(record.id)

    def _find_sync(self, spec: IngestionSpec) -> JobRecord | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT payload FROM {self.table}
                WHERE tenant = %s AND corpus = %s AND idempotency_key = %s
                """,
                (str(spec.tenant), str(spec.corpus), spec.idempotency_key),
            )
            row = cursor.fetchone()
        return _record_from_json(_json_value(row[0])) if row else None

    def _claim_sync(
        self, job_id: JobId, worker_id: str, lease_seconds: float
    ) -> JobRecord | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT payload FROM {self.table} WHERE id = %s FOR UPDATE",
                (str(job_id),),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(job_id)
            record = _record_from_json(_json_value(row[0]))
            claimed = _claim_record(record, worker_id, lease_seconds)
            if claimed is None:
                return None
            cursor.execute(
                f"UPDATE {self.table} SET payload = %s::jsonb, updated_at = NOW() "
                "WHERE id = %s",
                (
                    json.dumps(_record_to_json(claimed), default=str),
                    str(job_id),
                ),
            )
            return claimed


@dataclass(frozen=True, slots=True)
class PreparationResult:
    documents: tuple[Document, ...]
    failures: tuple[ItemOutcome, ...] = ()


PrepareDocuments = Callable[
    [IngestionSpec],
    Sequence[Document]
    | PreparationResult
    | Awaitable[Sequence[Document] | PreparationResult],
]
GraphBuilder = Callable[[Sequence[Document]], Any | Awaitable[Any]]


class DocumentPreparer:
    """Loader/chunker adapter kept behind durable ingestion's job interface."""

    def __init__(
        self,
        loader: Any,
        splitter: Any,
        *,
        batch_size: int = 10,
        max_concurrent_sources: int = 8,
    ) -> None:
        self.loader = loader
        self.splitter = splitter
        self.batch_size = batch_size
        self.max_concurrent_sources = max_concurrent_sources

    async def __call__(self, spec: IngestionSpec) -> PreparationResult:
        string_sources = [
            str(source) for source in spec.sources if not isinstance(source, Document)
        ]
        documents = [
            _copy_source_document(source)
            for source in spec.sources
            if isinstance(source, Document)
        ]
        failures: list[ItemOutcome] = []
        if string_sources:
            semaphore = asyncio.Semaphore(self.max_concurrent_sources)

            async def load_source(source: str) -> list[Document] | ItemOutcome:
                try:
                    async with semaphore:
                        loaded = await asyncio.to_thread(
                            self.loader.ingest_documents, [source]
                        )
                    result = [_coerce_document(item) for item in loaded]
                    for document in result:
                        document.metadata["ingestion_source_id"] = source
                    return result
                except Exception as exc:
                    retryable = not isinstance(exc, RagdollError) or exc.retryable
                    return ItemOutcome(
                        source_id=source,
                        status=(
                            ItemStatus.RETRYABLE if retryable else ItemStatus.REJECTED
                        ),
                        code=getattr(exc, "code", type(exc).__name__),
                        detail=str(exc),
                    )

            loaded_sources = await asyncio.gather(
                *(load_source(source) for source in string_sources)
            )
            for loaded in loaded_sources:
                if isinstance(loaded, ItemOutcome):
                    failures.append(loaded)
                else:
                    documents.extend(loaded)
        if not documents:
            return PreparationResult((), tuple(failures))
        chunks = await asyncio.to_thread(
            split_documents,
            documents,
            self.splitter,
            batch_size=self.batch_size,
        )
        return PreparationResult(tuple(chunks), tuple(failures))


class IngestionJob:
    def __init__(self, job_id: JobId, owner: "DurableIngestion") -> None:
        self.id = job_id
        self._owner = owner

    async def status(self) -> JobRecord:
        return await self._owner.store.get(self.id)

    async def wait(self) -> IngestionResult:
        await self._owner._wait(self.id)
        record = await self.status()
        if record.result is None:
            raise RuntimeError(f"Job {self.id} finished without a result")
        return record.result

    async def cancel(self) -> None:
        await self._owner._cancel(self.id)


class DurableIngestion:
    """Small job interface hiding scheduling, state, and publication semantics."""

    def __init__(
        self,
        *,
        index: CorpusIndex,
        prepare: PrepareDocuments,
        store: JobStore | None = None,
        execution: ExecutionAdapter | None = None,
        events: EventSink | None = None,
        quarantine: QuarantineStore | None = None,
        graph_builder: GraphBuilder | None = None,
        graph_index: VersionedGraphIndex | None = None,
        max_concurrent_jobs: int = 2,
        lease_seconds: float = 900,
    ) -> None:
        self.index = index
        self.prepare = prepare
        self.store = store or MemoryJobStore()
        self.execution = execution or InlineExecutionAdapter()
        self.events = events or NullEventSink()
        self.quarantine = quarantine or MemoryQuarantineStore()
        if (graph_builder is None) != (graph_index is None):
            raise ValueError("graph_builder and graph_index must be provided together")
        self.graph_builder = graph_builder
        self.graph_index = graph_index
        self._capacity = asyncio.Semaphore(max_concurrent_jobs)
        self.worker_id = uuid.uuid4().hex
        self.lease_seconds = lease_seconds

    async def submit(self, spec: IngestionSpec) -> IngestionJob:
        existing = await self.store.find_idempotent(spec)
        if existing is not None:
            if (
                existing.status is JobStatus.FAILED
                and existing.result is not None
                and any(
                    item.status is ItemStatus.RETRYABLE
                    for item in existing.result.items
                )
            ):
                existing = replace(
                    existing,
                    status=JobStatus.PENDING,
                    progress=JobProgress(0, len(spec.sources), "pending"),
                    result=None,
                    cancel_requested=False,
                    worker_id=None,
                    lease_until=None,
                    staged_generation_id=None,
                    staged_checksum=None,
                    staged_document_count=0,
                    checkpoint_items=(),
                )
                await self.store.save(existing)
            if existing.status is JobStatus.PENDING or (
                existing.status is JobStatus.RUNNING and not self.execution.distributed
            ):
                if existing.status is JobStatus.RUNNING:
                    # File state is a single-host profile. A new SDK instance is
                    # authoritative after restart and may reclaim its stale lease.
                    await self.store.save(replace(existing, lease_until=None))
                await self.execution.submit(existing.id, self.execute)
            return IngestionJob(existing.id, self)
        job_id = JobId(uuid.uuid4().hex)
        record = JobRecord(
            id=job_id,
            spec=spec,
            progress=JobProgress(0, len(spec.sources), "pending"),
        )
        created = await self.store.create(record)
        if created.id == job_id:
            self.events.emit(
                "ingestion.submitted",
                {
                    "job_id": str(job_id),
                    "tenant": str(spec.tenant),
                    "corpus": str(spec.corpus),
                    "sources": len(spec.sources),
                },
            )
            await self.execution.submit(job_id, self.execute)
        return IngestionJob(created.id, self)

    async def execute(self, job_id: JobId | str) -> None:
        """Worker entry point: execute one durable job by identifier."""
        typed_id = JobId(job_id)
        claimed = await self.store.claim(typed_id, self.worker_id, self.lease_seconds)
        if claimed is None:
            current = await self.store.get(typed_id)
            if current.status in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                return
            raise JobLeaseUnavailableError(
                f"Job {typed_id} is already owned by another worker"
            )
        await self._run(typed_id)

    async def _run(self, job_id: JobId) -> None:
        started_at = datetime.now(timezone.utc)
        generation: StagedGeneration | None = None
        published = False
        try:
            async with self._capacity:
                record = await self.store.get(job_id)
                if record.cancel_requested:
                    raise CancelledError("Job was cancelled before execution")
                if (
                    record.staged_generation_id is not None
                    and self.graph_builder is None
                ):
                    generation = StagedGeneration(
                        id=record.staged_generation_id,
                        tenant=record.spec.tenant,
                        corpus=record.spec.corpus,
                        document_count=record.staged_document_count,
                        checksum=record.staged_checksum or "",
                    )
                    await self.index.promote(generation)
                    published = True
                    result = IngestionResult(
                        job_id=job_id,
                        status=JobStatus.COMPLETED,
                        items=record.checkpoint_items,
                        generation_id=generation.id,
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc),
                    )
                    await self.store.save(
                        replace(
                            record,
                            status=JobStatus.COMPLETED,
                            progress=JobProgress(
                                len(record.spec.sources),
                                len(record.spec.sources),
                                "completed",
                            ),
                            result=result,
                            lease_until=None,
                        )
                    )
                    self.events.emit(
                        "ingestion.resumed",
                        {
                            "job_id": str(job_id),
                            "generation_id": str(generation.id),
                            "checkpoint": "staged",
                        },
                    )
                    return
                record = replace(
                    record,
                    status=JobStatus.RUNNING,
                    progress=JobProgress(0, len(record.spec.sources), "preparing"),
                )
                await self.store.save(record)
                self.events.emit(
                    "ingestion.started",
                    {
                        "job_id": str(job_id),
                        "tenant": str(record.spec.tenant),
                        "corpus": str(record.spec.corpus),
                    },
                )
                prepared = self.prepare(record.spec)
                prepared_value = (
                    await prepared if inspect.isawaitable(prepared) else prepared
                )
                preparation_failures: tuple[ItemOutcome, ...] = ()
                if isinstance(prepared_value, PreparationResult):
                    documents = list(prepared_value.documents)
                    preparation_failures = prepared_value.failures
                else:
                    documents = list(prepared_value)
                documents = prepare_corpus_documents(
                    str(record.spec.tenant), str(record.spec.corpus), documents
                )
                preparation_outcomes = _success_outcomes(
                    record.spec,
                    documents,
                    failures=preparation_failures,
                )
                if not documents or any(
                    item.status is ItemStatus.RETRYABLE for item in preparation_outcomes
                ):
                    failed_outcomes = tuple(
                        (
                            replace(
                                item,
                                status=ItemStatus.RETRYABLE,
                                code="batch_not_published",
                                detail=(
                                    "Another source failed retryably; no corpus "
                                    "generation was published"
                                ),
                            )
                            if item.status is ItemStatus.INDEXED
                            else item
                        )
                        for item in preparation_outcomes
                    )
                    result = IngestionResult(
                        job_id,
                        JobStatus.FAILED,
                        failed_outcomes,
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc),
                    )
                    await self._quarantine_rejected(
                        job_id, record.spec, preparation_outcomes
                    )
                    await self.store.save(
                        replace(
                            record,
                            status=JobStatus.FAILED,
                            result=result,
                            lease_until=None,
                        )
                    )
                    self.events.emit(
                        "ingestion.failed",
                        {
                            "job_id": str(job_id),
                            "error_code": "preparation_failed",
                            "retryable": any(
                                item.status is ItemStatus.RETRYABLE
                                for item in preparation_outcomes
                            ),
                        },
                    )
                    return
                record = await self.store.get(job_id)
                if record.cancel_requested:
                    raise CancelledError("Job was cancelled during preparation")
                record = replace(
                    record,
                    progress=JobProgress(
                        len(documents), len(record.spec.sources), "staging"
                    ),
                )
                await self.store.save(record)
                generation = await self.index.stage(
                    str(record.spec.tenant), str(record.spec.corpus), documents
                )
                checkpoint_items = _success_outcomes(
                    record.spec,
                    generation.documents,
                    failures=preparation_failures,
                    status=(
                        ItemStatus.UNCHANGED
                        if generation.unchanged
                        else ItemStatus.INDEXED
                    ),
                )
                record = replace(
                    record,
                    progress=JobProgress(
                        len(documents), len(record.spec.sources), "staged"
                    ),
                    staged_generation_id=generation.id,
                    staged_checksum=generation.checksum,
                    staged_document_count=generation.document_count,
                    checkpoint_items=checkpoint_items,
                )
                await self.store.save(record)
                self.events.emit(
                    "ingestion.staged",
                    {
                        "job_id": str(job_id),
                        "generation_id": str(generation.id),
                        "documents": generation.document_count,
                    },
                )
                record = await self.store.get(job_id)
                if record.cancel_requested:
                    raise CancelledError("Job was cancelled before publication")
                if (
                    not generation.unchanged
                    and self.graph_builder is not None
                    and self.graph_index is not None
                ):
                    graph_value = self.graph_builder(generation.documents)
                    graph = (
                        await graph_value
                        if inspect.isawaitable(graph_value)
                        else graph_value
                    )
                    await self.graph_index.stage(generation, graph)
                if not generation.unchanged:
                    await self.index.promote(generation)
                    published = True
                outcomes = checkpoint_items
                await self._quarantine_rejected(job_id, record.spec, outcomes)
                result = IngestionResult(
                    job_id=job_id,
                    status=JobStatus.COMPLETED,
                    items=outcomes,
                    generation_id=generation.id,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                )
                await self.store.save(
                    replace(
                        record,
                        status=JobStatus.COMPLETED,
                        progress=JobProgress(
                            len(record.spec.sources),
                            len(record.spec.sources),
                            "completed",
                        ),
                        result=result,
                        lease_until=None,
                    )
                )
                self.events.emit(
                    "ingestion.completed",
                    {
                        "job_id": str(job_id),
                        "generation_id": str(generation.id),
                        "sources": len(outcomes),
                    },
                )
        except (CancelledError, asyncio.CancelledError):
            await self._discard_unpublished(job_id, generation, published)
            record = await self.store.get(job_id)
            outcomes = tuple(
                ItemOutcome(
                    source_identity(source), ItemStatus.CANCELLED, code="cancelled"
                )
                for source in record.spec.sources
            )
            result = IngestionResult(
                job_id,
                JobStatus.CANCELLED,
                outcomes,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
            await self.store.save(
                replace(
                    record,
                    status=JobStatus.CANCELLED,
                    result=result,
                    lease_until=None,
                )
            )
            self.events.emit("ingestion.cancelled", {"job_id": str(job_id)})
        except Exception as exc:
            await self._discard_unpublished(job_id, generation, published)
            record = await self.store.get(job_id)
            retryable = not isinstance(exc, RagdollError) or exc.retryable
            status = ItemStatus.RETRYABLE if retryable else ItemStatus.REJECTED
            outcomes = tuple(
                ItemOutcome(
                    source_identity(source),
                    status,
                    code=getattr(exc, "code", type(exc).__name__),
                    detail=str(exc),
                )
                for source in record.spec.sources
            )
            result = IngestionResult(
                job_id,
                JobStatus.FAILED,
                outcomes,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
            await self._quarantine_rejected(job_id, record.spec, outcomes)
            await self.store.save(
                replace(
                    record,
                    status=JobStatus.FAILED,
                    result=result,
                    lease_until=None,
                )
            )
            self.events.emit(
                "ingestion.failed",
                {
                    "job_id": str(job_id),
                    "error_code": getattr(exc, "code", type(exc).__name__),
                    "retryable": retryable,
                },
            )

    async def _discard_unpublished(
        self,
        job_id: JobId,
        generation: StagedGeneration | None,
        published: bool,
    ) -> None:
        if generation is None or generation.unchanged or published:
            return
        errors: list[str] = []
        if self.graph_index is not None:
            discard_graph = getattr(self.graph_index, "discard", None)
            if discard_graph is not None:
                try:
                    await discard_graph(generation)
                except Exception as exc:  # cleanup must not mask the job failure
                    errors.append(f"graph: {exc}")
        try:
            await self.index.discard(generation)
        except Exception as exc:  # cleanup must not mask the job failure
            errors.append(f"corpus: {exc}")
        self.events.emit(
            "ingestion.discarded",
            {
                "job_id": str(job_id),
                "generation_id": str(generation.id),
                "cleanup_errors": errors,
            },
        )

    async def _cancel(self, job_id: JobId) -> None:
        record = await self.store.get(job_id)
        if record.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            return
        await self.store.save(replace(record, cancel_requested=True))
        await self.execution.cancel(job_id)

    async def _wait(self, job_id: JobId) -> None:
        await self.execution.wait(job_id)
        while True:
            record = await self.store.get(job_id)
            if record.status in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                return
            await asyncio.sleep(0.05)

    async def _quarantine_rejected(
        self,
        job_id: JobId,
        spec: IngestionSpec,
        outcomes: Sequence[ItemOutcome],
    ) -> None:
        for outcome in outcomes:
            if outcome.status is ItemStatus.REJECTED:
                await self.quarantine.add(
                    quarantined_source(
                        job_id=job_id,
                        tenant=spec.tenant,
                        corpus=spec.corpus,
                        source_id=outcome.source_id,
                        code=outcome.code or "rejected",
                        detail=outcome.detail,
                    )
                )


def _success_outcomes(
    spec: IngestionSpec,
    documents: Sequence[Document],
    *,
    status: ItemStatus = ItemStatus.INDEXED,
    failures: Sequence[ItemOutcome] = (),
) -> tuple[ItemOutcome, ...]:
    failed_by_source = {item.source_id: item for item in failures}
    outcomes = []
    for source in spec.sources:
        source_id = source_identity(source)
        if source_id in failed_by_source:
            outcomes.append(failed_by_source[source_id])
            continue
        source_documents = [
            document
            for document in documents
            if str(
                document.metadata.get("ingestion_source_id")
                or document.metadata.get("source")
                or document.metadata.get("path")
                or ""
            )
            == source_id
        ]
        by_document: dict[str, list[Document]] = {}
        for document in source_documents:
            document_id = str(document.metadata.get("document_id") or "")
            by_document.setdefault(document_id, []).append(document)
        if not by_document:
            outcomes.append(
                ItemOutcome(
                    source_id=source_id,
                    status=ItemStatus.REJECTED,
                    code="no_documents_produced",
                )
            )
            continue
        for document_id, chunks in by_document.items():
            outcomes.append(
                ItemOutcome(
                    source_id=source_id,
                    status=status,
                    document_id=document_id or None,
                    chunk_ids=tuple(
                        str(chunk.metadata["chunk_id"])
                        for chunk in chunks
                        if chunk.metadata.get("chunk_id")
                    ),
                )
            )
    return tuple(outcomes)


def _source_to_json(source: object) -> dict[str, Any]:
    if isinstance(source, Document):
        return {
            "kind": "document",
            "page_content": source.page_content,
            "metadata": dict(source.metadata),
        }
    if isinstance(source, Path):
        return {"kind": "path", "value": str(source)}
    return {"kind": "string", "value": str(source)}


def _copy_source_document(source: Document) -> Document:
    metadata = dict(source.metadata)
    metadata.setdefault("source", source_identity(source))
    metadata.setdefault("ingestion_source_id", source_identity(source))
    return Document(page_content=source.page_content, metadata=metadata)


def _coerce_document(value: Any) -> Document:
    if isinstance(value, Document):
        return Document(page_content=value.page_content, metadata=dict(value.metadata))
    if isinstance(value, dict):
        return Document(
            page_content=value.get("page_content") or value.get("text") or "",
            metadata=dict(value.get("metadata") or {}),
        )
    if hasattr(value, "page_content"):
        return Document(
            page_content=value.page_content,
            metadata=dict(getattr(value, "metadata", {}) or {}),
        )
    raise TypeError(f"Unsupported document payload: {type(value)!r}")


def _source_from_json(data: dict[str, Any]) -> object:
    if data["kind"] == "document":
        return Document(
            page_content=data["page_content"], metadata=data.get("metadata", {})
        )
    if data["kind"] == "path":
        return Path(data["value"])
    return data["value"]


def _record_to_json(record: JobRecord) -> dict[str, Any]:
    result = None
    if record.result is not None:
        result = {
            "job_id": str(record.result.job_id),
            "status": record.result.status.value,
            "items": [
                {
                    "source_id": item.source_id,
                    "status": item.status.value,
                    "document_id": item.document_id,
                    "chunk_ids": list(item.chunk_ids),
                    "code": item.code,
                    "detail": item.detail,
                }
                for item in record.result.items
            ],
            "generation_id": record.result.generation_id,
            "started_at": (
                record.result.started_at.isoformat()
                if record.result.started_at
                else None
            ),
            "finished_at": (
                record.result.finished_at.isoformat()
                if record.result.finished_at
                else None
            ),
        }
    return {
        "id": str(record.id),
        "spec": {
            "tenant": str(record.spec.tenant),
            "corpus": str(record.spec.corpus),
            "sources": [_source_to_json(source) for source in record.spec.sources],
            "idempotency_key": record.spec.idempotency_key,
            "metadata": dict(record.spec.metadata),
        },
        "status": record.status.value,
        "progress": {
            "completed": record.progress.completed,
            "total": record.progress.total,
            "stage": record.progress.stage,
        },
        "result": result,
        "cancel_requested": record.cancel_requested,
        "attempt": record.attempt,
        "worker_id": record.worker_id,
        "lease_until": (record.lease_until.isoformat() if record.lease_until else None),
        "checkpoint": {
            "generation_id": record.staged_generation_id,
            "checksum": record.staged_checksum,
            "document_count": record.staged_document_count,
            "items": [
                {
                    "source_id": item.source_id,
                    "status": item.status.value,
                    "document_id": item.document_id,
                    "chunk_ids": list(item.chunk_ids),
                    "code": item.code,
                    "detail": item.detail,
                }
                for item in record.checkpoint_items
            ],
        },
    }


def _record_from_json(data: dict[str, Any]) -> JobRecord:
    spec_data = data["spec"]
    spec = IngestionSpec(
        tenant=spec_data["tenant"],
        corpus=spec_data["corpus"],
        sources=[_source_from_json(source) for source in spec_data["sources"]],
        idempotency_key=spec_data.get("idempotency_key"),
        metadata=spec_data.get("metadata", {}),
    )
    progress = JobProgress(**data["progress"])
    result_data = data.get("result")
    result = None
    if result_data is not None:
        result = IngestionResult(
            job_id=JobId(result_data["job_id"]),
            status=JobStatus(result_data["status"]),
            items=tuple(
                ItemOutcome(
                    source_id=item["source_id"],
                    status=ItemStatus(item["status"]),
                    document_id=item.get("document_id"),
                    chunk_ids=tuple(item.get("chunk_ids", [])),
                    code=item.get("code"),
                    detail=item.get("detail"),
                )
                for item in result_data["items"]
            ),
            generation_id=(
                GenerationId(result_data["generation_id"])
                if result_data.get("generation_id")
                else None
            ),
            started_at=(
                datetime.fromisoformat(result_data["started_at"])
                if result_data.get("started_at")
                else None
            ),
            finished_at=(
                datetime.fromisoformat(result_data["finished_at"])
                if result_data.get("finished_at")
                else None
            ),
        )
    checkpoint = data.get("checkpoint") or {}
    return JobRecord(
        id=JobId(data["id"]),
        spec=spec,
        status=JobStatus(data["status"]),
        progress=progress,
        result=result,
        cancel_requested=data.get("cancel_requested", False),
        attempt=int(data.get("attempt", 0)),
        worker_id=data.get("worker_id"),
        lease_until=(
            datetime.fromisoformat(data["lease_until"])
            if data.get("lease_until")
            else None
        ),
        staged_generation_id=(
            GenerationId(checkpoint["generation_id"])
            if checkpoint.get("generation_id")
            else None
        ),
        staged_checksum=checkpoint.get("checksum"),
        staged_document_count=int(checkpoint.get("document_count", 0)),
        checkpoint_items=tuple(
            ItemOutcome(
                source_id=item["source_id"],
                status=ItemStatus(item["status"]),
                document_id=item.get("document_id"),
                chunk_ids=tuple(item.get("chunk_ids", [])),
                code=item.get("code"),
                detail=item.get("detail"),
            )
            for item in checkpoint.get("items", [])
        ),
    )


def _claim_record(
    record: JobRecord, worker_id: str, lease_seconds: float
) -> JobRecord | None:
    now = datetime.now(timezone.utc)
    if record.status in {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }:
        return None
    if (
        record.status is JobStatus.RUNNING
        and record.worker_id != worker_id
        and record.lease_until is not None
        and record.lease_until > now
    ):
        return None
    return replace(
        record,
        status=JobStatus.RUNNING,
        worker_id=worker_id,
        lease_until=now + timedelta(seconds=lease_seconds),
        attempt=record.attempt + 1,
    )


def _json_value(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else value
