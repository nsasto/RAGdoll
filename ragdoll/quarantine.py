"""Quarantine adapters for sources that require human or policy review."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from ragdoll.contracts import CorpusId, JobId, TenantId


@dataclass(frozen=True, slots=True)
class QuarantinedSource:
    job_id: JobId
    tenant: TenantId
    corpus: CorpusId
    source_id: str
    code: str
    detail: str | None
    quarantined_at: datetime


class QuarantineStore(Protocol):
    async def add(self, item: QuarantinedSource) -> None: ...
    async def list(self, tenant: str, corpus: str) -> list[QuarantinedSource]: ...


class MemoryQuarantineStore:
    def __init__(self) -> None:
        self._items: list[QuarantinedSource] = []
        self._lock = asyncio.Lock()

    async def add(self, item: QuarantinedSource) -> None:
        async with self._lock:
            self._items.append(item)

    async def list(self, tenant: str, corpus: str) -> list[QuarantinedSource]:
        async with self._lock:
            return [
                item
                for item in self._items
                if item.tenant == TenantId(tenant) and item.corpus == CorpusId(corpus)
            ]


class FileQuarantineStore(MemoryQuarantineStore):
    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self._loaded = False

    async def _load(self) -> None:
        if not self._loaded and self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._items = [_decode(item) for item in payload]
        self._loaded = True

    async def add(self, item: QuarantinedSource) -> None:
        await self._load()
        await super().add(item)
        await self._persist()

    async def list(self, tenant: str, corpus: str) -> list[QuarantinedSource]:
        await self._load()
        return await super().list(tenant, corpus)

    async def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps([_encode(item) for item in self._items], indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


class PostgresQuarantineStore:
    """Shared quarantine adapter for multi-worker deployments."""

    def __init__(
        self,
        dsn: str,
        *,
        table: str = "ragdoll_quarantine",
        connect: Callable[..., Any] | None = None,
    ) -> None:
        if not dsn:
            raise ValueError("Postgres quarantine requires a DSN")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError("table must be a simple identifier")
        self.dsn = dsn
        self.table = table
        self._connect_override = connect
        self._initialized = False
        self._lock = asyncio.Lock()

    async def add(self, item: QuarantinedSource) -> None:
        await self._ensure_schema()
        await asyncio.to_thread(self._add_sync, item)

    async def list(self, tenant: str, corpus: str) -> list[QuarantinedSource]:
        await self._ensure_schema()
        return await asyncio.to_thread(self._list_sync, tenant, corpus)

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if not self._initialized:
                await asyncio.to_thread(self._initialize_sync)
                self._initialized = True

    def _connect(self):
        if self._connect_override:
            return self._connect_override(self.dsn)
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Postgres quarantine requires `pip install python-ragdoll[scaled]`"
            ) from exc
        return psycopg.connect(self.dsn)

    def _initialize_sync(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ("
                "job_id TEXT NOT NULL, tenant TEXT NOT NULL, corpus TEXT NOT NULL, "
                "source_id TEXT NOT NULL, code TEXT NOT NULL, detail TEXT NULL, "
                "quarantined_at TIMESTAMPTZ NOT NULL, "
                "PRIMARY KEY (job_id, source_id))"
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_scope_idx "
                f"ON {self.table} (tenant, corpus, quarantined_at)"
            )

    def _add_sync(self, item: QuarantinedSource) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {self.table} "
                "(job_id, tenant, corpus, source_id, code, detail, quarantined_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (job_id, source_id) DO UPDATE "
                "SET code = EXCLUDED.code, detail = EXCLUDED.detail, "
                "quarantined_at = EXCLUDED.quarantined_at",
                (
                    str(item.job_id),
                    str(item.tenant),
                    str(item.corpus),
                    item.source_id,
                    item.code,
                    item.detail,
                    item.quarantined_at,
                ),
            )

    def _list_sync(self, tenant: str, corpus: str) -> list[QuarantinedSource]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT job_id, tenant, corpus, source_id, code, detail, quarantined_at "
                f"FROM {self.table} WHERE tenant = %s AND corpus = %s "
                "ORDER BY quarantined_at",
                (tenant, corpus),
            )
            rows = cursor.fetchall()
        return [
            QuarantinedSource(
                job_id=JobId(row[0]),
                tenant=TenantId(row[1]),
                corpus=CorpusId(row[2]),
                source_id=row[3],
                code=row[4],
                detail=row[5],
                quarantined_at=row[6],
            )
            for row in rows
        ]


def quarantined_source(
    *,
    job_id: JobId,
    tenant: TenantId,
    corpus: CorpusId,
    source_id: str,
    code: str,
    detail: str | None,
) -> QuarantinedSource:
    return QuarantinedSource(
        job_id=job_id,
        tenant=tenant,
        corpus=corpus,
        source_id=source_id,
        code=code,
        detail=detail,
        quarantined_at=datetime.now(timezone.utc),
    )


def _encode(item: QuarantinedSource) -> dict:
    data = asdict(item)
    data.update(
        job_id=str(item.job_id),
        tenant=str(item.tenant),
        corpus=str(item.corpus),
        quarantined_at=item.quarantined_at.isoformat(),
    )
    return data


def _decode(data: dict) -> QuarantinedSource:
    return QuarantinedSource(
        job_id=JobId(data["job_id"]),
        tenant=TenantId(data["tenant"]),
        corpus=CorpusId(data["corpus"]),
        source_id=data["source_id"],
        code=data["code"],
        detail=data.get("detail"),
        quarantined_at=datetime.fromisoformat(data["quarantined_at"]),
    )
