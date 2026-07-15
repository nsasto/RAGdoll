"""Shared generation state for atomic corpus publication and rollback."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from ragdoll.contracts import CorpusId, GenerationId, TenantId
from ragdoll.errors import (
    GenerationActiveError,
    GenerationNotFoundError,
    TenantIsolationError,
)


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    id: GenerationId
    tenant: TenantId
    corpus: CorpusId
    vector_ids: tuple[str, ...]
    document_count: int
    checksum: str


class GenerationStateStore(Protocol):
    async def put(self, record: GenerationRecord) -> None: ...
    async def get(self, generation_id: GenerationId) -> GenerationRecord: ...
    async def promote(
        self,
        generation_id: GenerationId,
        *,
        tenant: str | None = None,
        corpus: str | None = None,
    ) -> GenerationId: ...
    async def active(self, tenant: str, corpus: str) -> GenerationId | None: ...
    async def rollback(self, tenant: str, corpus: str) -> GenerationId: ...
    async def list_scope(self, tenant: str, corpus: str) -> list[GenerationRecord]: ...
    async def delete_generation(self, generation_id: GenerationId) -> None: ...
    async def delete_scope(self, tenant: str, corpus: str) -> None: ...


class MemoryGenerationStateStore:
    def __init__(self) -> None:
        self._generations: dict[GenerationId, GenerationRecord] = {}
        self._active: dict[tuple[TenantId, CorpusId], GenerationId] = {}
        self._history: dict[tuple[TenantId, CorpusId], list[GenerationId]] = {}
        self._lock = asyncio.Lock()

    async def put(self, record: GenerationRecord) -> None:
        async with self._lock:
            self._generations[record.id] = record

    async def get(self, generation_id: GenerationId) -> GenerationRecord:
        async with self._lock:
            try:
                return self._generations[generation_id]
            except KeyError as exc:
                raise GenerationNotFoundError(
                    f"Unknown generation: {generation_id}"
                ) from exc

    async def promote(
        self,
        generation_id: GenerationId,
        *,
        tenant: str | None = None,
        corpus: str | None = None,
    ) -> GenerationId:
        async with self._lock:
            record = self._generations.get(generation_id)
            if record is None:
                raise GenerationNotFoundError(f"Unknown generation: {generation_id}")
            _assert_scope(record, tenant, corpus)
            key = (record.tenant, record.corpus)
            current = self._active.get(key)
            if current != generation_id:
                if current is not None:
                    self._history.setdefault(key, []).append(current)
                self._active[key] = generation_id
            return generation_id

    async def active(self, tenant: str, corpus: str) -> GenerationId | None:
        async with self._lock:
            return self._active.get((TenantId(tenant), CorpusId(corpus)))

    async def rollback(self, tenant: str, corpus: str) -> GenerationId:
        async with self._lock:
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

    async def list_scope(self, tenant: str, corpus: str) -> list[GenerationRecord]:
        key = (TenantId(tenant), CorpusId(corpus))
        async with self._lock:
            return [
                record
                for record in self._generations.values()
                if (record.tenant, record.corpus) == key
            ]

    async def delete_scope(self, tenant: str, corpus: str) -> None:
        key = (TenantId(tenant), CorpusId(corpus))
        async with self._lock:
            ids = [
                item_id
                for item_id, record in self._generations.items()
                if (record.tenant, record.corpus) == key
            ]
            for item_id in ids:
                del self._generations[item_id]
            self._active.pop(key, None)
            self._history.pop(key, None)

    async def delete_generation(self, generation_id: GenerationId) -> None:
        async with self._lock:
            record = self._generations.get(generation_id)
            if record is None:
                return
            key = (record.tenant, record.corpus)
            if self._active.get(key) == generation_id:
                raise GenerationActiveError("The active generation cannot be discarded")
            del self._generations[generation_id]
            history = self._history.get(key)
            if history is not None:
                self._history[key] = [item for item in history if item != generation_id]


class FileGenerationStateStore(MemoryGenerationStateStore):
    """Atomic single-host generation state adapter."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._decode(payload)
        self._loaded = True

    async def put(self, record: GenerationRecord) -> None:
        await self._ensure_loaded()
        await super().put(record)
        await self._persist()

    async def get(self, generation_id: GenerationId) -> GenerationRecord:
        await self._ensure_loaded()
        return await super().get(generation_id)

    async def promote(self, generation_id: GenerationId, **scope: Any) -> GenerationId:
        await self._ensure_loaded()
        result = await super().promote(generation_id, **scope)
        await self._persist()
        return result

    async def active(self, tenant: str, corpus: str) -> GenerationId | None:
        await self._ensure_loaded()
        return await super().active(tenant, corpus)

    async def rollback(self, tenant: str, corpus: str) -> GenerationId:
        await self._ensure_loaded()
        result = await super().rollback(tenant, corpus)
        await self._persist()
        return result

    async def list_scope(self, tenant: str, corpus: str) -> list[GenerationRecord]:
        await self._ensure_loaded()
        return await super().list_scope(tenant, corpus)

    async def delete_scope(self, tenant: str, corpus: str) -> None:
        await self._ensure_loaded()
        await super().delete_scope(tenant, corpus)
        await self._persist()

    async def delete_generation(self, generation_id: GenerationId) -> None:
        await self._ensure_loaded()
        await super().delete_generation(generation_id)
        await self._persist()

    async def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(self._encode(), indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def _encode(self) -> dict[str, Any]:
        return {
            "generations": {
                str(item_id): {
                    **asdict(record),
                    "id": str(record.id),
                    "tenant": str(record.tenant),
                    "corpus": str(record.corpus),
                    "vector_ids": list(record.vector_ids),
                }
                for item_id, record in self._generations.items()
            },
            "active": {
                _scope_key(*key): str(value) for key, value in self._active.items()
            },
            "history": {
                _scope_key(*key): [str(item) for item in value]
                for key, value in self._history.items()
            },
        }

    def _decode(self, payload: dict[str, Any]) -> None:
        self._generations = {
            GenerationId(item_id): _record(value)
            for item_id, value in payload.get("generations", {}).items()
        }
        self._active = {
            _parse_scope(key): GenerationId(value)
            for key, value in payload.get("active", {}).items()
        }
        self._history = {
            _parse_scope(key): [GenerationId(item) for item in value]
            for key, value in payload.get("history", {}).items()
        }


class PostgresGenerationStateStore:
    """Shared state adapter with transactional promotion and rollback."""

    def __init__(
        self,
        dsn: str,
        *,
        table_prefix: str = "ragdoll_corpus",
        connect: Callable[..., Any] | None = None,
    ) -> None:
        if not dsn:
            raise ValueError("Postgres generation state requires a DSN")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_prefix):
            raise ValueError("table_prefix must be a simple identifier")
        self.dsn = dsn
        self.generations = f"{table_prefix}_generations"
        self.corpora = f"{table_prefix}_active"
        self._connect_override = connect
        self._initialized = False
        self._lock = asyncio.Lock()

    async def put(self, record: GenerationRecord) -> None:
        await self._ensure_schema()
        await asyncio.to_thread(self._put_sync, record)

    async def get(self, generation_id: GenerationId) -> GenerationRecord:
        await self._ensure_schema()
        return await asyncio.to_thread(self._get_sync, generation_id)

    async def promote(self, generation_id: GenerationId, **scope: Any) -> GenerationId:
        await self._ensure_schema()
        return await asyncio.to_thread(self._promote_sync, generation_id, scope)

    async def active(self, tenant: str, corpus: str) -> GenerationId | None:
        await self._ensure_schema()
        return await asyncio.to_thread(self._active_sync, tenant, corpus)

    async def rollback(self, tenant: str, corpus: str) -> GenerationId:
        await self._ensure_schema()
        return await asyncio.to_thread(self._rollback_sync, tenant, corpus)

    async def list_scope(self, tenant: str, corpus: str) -> list[GenerationRecord]:
        await self._ensure_schema()
        return await asyncio.to_thread(self._list_sync, tenant, corpus)

    async def delete_scope(self, tenant: str, corpus: str) -> None:
        await self._ensure_schema()
        await asyncio.to_thread(self._delete_sync, tenant, corpus)

    async def delete_generation(self, generation_id: GenerationId) -> None:
        await self._ensure_schema()
        await asyncio.to_thread(self._delete_generation_sync, generation_id)

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
                "Postgres state requires `pip install python-ragdoll[scaled]`"
            ) from exc
        return psycopg.connect(self.dsn)

    def _initialize_sync(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {self.generations} ("
                "id TEXT PRIMARY KEY, tenant TEXT NOT NULL, corpus TEXT NOT NULL, "
                "vector_ids JSONB NOT NULL, document_count INTEGER NOT NULL, checksum TEXT NOT NULL)"
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {self.generations}_scope_idx "
                f"ON {self.generations} (tenant, corpus)"
            )
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {self.corpora} ("
                "tenant TEXT NOT NULL, corpus TEXT NOT NULL, active_generation TEXT NULL, "
                "history JSONB NOT NULL DEFAULT '[]'::jsonb, PRIMARY KEY (tenant, corpus))"
            )

    def _put_sync(self, record: GenerationRecord) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {self.generations} "
                "(id, tenant, corpus, vector_ids, document_count, checksum) "
                "VALUES (%s, %s, %s, %s::jsonb, %s, %s) ON CONFLICT (id) DO NOTHING",
                (
                    str(record.id),
                    str(record.tenant),
                    str(record.corpus),
                    json.dumps(record.vector_ids),
                    record.document_count,
                    record.checksum,
                ),
            )

    def _get_sync(self, generation_id: GenerationId) -> GenerationRecord:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT id, tenant, corpus, vector_ids, document_count, checksum "
                f"FROM {self.generations} WHERE id = %s",
                (str(generation_id),),
            )
            row = cursor.fetchone()
        if row is None:
            raise GenerationNotFoundError(f"Unknown generation: {generation_id}")
        return _row_record(row)

    def _promote_sync(
        self, generation_id: GenerationId, scope: dict[str, Any]
    ) -> GenerationId:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT tenant, corpus FROM {self.generations} WHERE id = %s",
                (str(generation_id),),
            )
            row = cursor.fetchone()
            if row is None:
                raise GenerationNotFoundError(f"Unknown generation: {generation_id}")
            tenant, corpus = row
            _assert_values(tenant, corpus, scope.get("tenant"), scope.get("corpus"))
            cursor.execute(
                f"INSERT INTO {self.corpora} (tenant, corpus) VALUES (%s, %s) "
                "ON CONFLICT (tenant, corpus) DO NOTHING",
                (tenant, corpus),
            )
            cursor.execute(
                f"SELECT active_generation, history FROM {self.corpora} "
                "WHERE tenant = %s AND corpus = %s FOR UPDATE",
                (tenant, corpus),
            )
            active, history = cursor.fetchone()
            history = list(_json(history))
            if active != str(generation_id):
                if active:
                    history.append(active)
                cursor.execute(
                    f"UPDATE {self.corpora} SET active_generation = %s, history = %s::jsonb "
                    "WHERE tenant = %s AND corpus = %s",
                    (str(generation_id), json.dumps(history), tenant, corpus),
                )
        return generation_id

    def _active_sync(self, tenant: str, corpus: str) -> GenerationId | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT active_generation FROM {self.corpora} WHERE tenant = %s AND corpus = %s",
                (tenant, corpus),
            )
            row = cursor.fetchone()
        return GenerationId(row[0]) if row and row[0] else None

    def _rollback_sync(self, tenant: str, corpus: str) -> GenerationId:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT active_generation, history FROM {self.corpora} "
                "WHERE tenant = %s AND corpus = %s FOR UPDATE",
                (tenant, corpus),
            )
            row = cursor.fetchone()
            if row is None or not _json(row[1]):
                raise GenerationNotFoundError("No previous generation is available")
            active, history_value = row
            history = list(_json(history_value))
            previous = history.pop()
            if active:
                history.append(active)
            cursor.execute(
                f"UPDATE {self.corpora} SET active_generation = %s, history = %s::jsonb "
                "WHERE tenant = %s AND corpus = %s",
                (previous, json.dumps(history), tenant, corpus),
            )
        return GenerationId(previous)

    def _list_sync(self, tenant: str, corpus: str) -> list[GenerationRecord]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT id, tenant, corpus, vector_ids, document_count, checksum "
                f"FROM {self.generations} WHERE tenant = %s AND corpus = %s",
                (tenant, corpus),
            )
            rows = cursor.fetchall()
        return [_row_record(row) for row in rows]

    def _delete_sync(self, tenant: str, corpus: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"DELETE FROM {self.corpora} WHERE tenant = %s AND corpus = %s",
                (tenant, corpus),
            )
            cursor.execute(
                f"DELETE FROM {self.generations} WHERE tenant = %s AND corpus = %s",
                (tenant, corpus),
            )

    def _delete_generation_sync(self, generation_id: GenerationId) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT tenant, corpus FROM {self.generations} WHERE id = %s FOR UPDATE",
                (str(generation_id),),
            )
            row = cursor.fetchone()
            if row is None:
                return
            cursor.execute(
                f"SELECT active_generation FROM {self.corpora} "
                "WHERE tenant = %s AND corpus = %s FOR UPDATE",
                row,
            )
            active = cursor.fetchone()
            if active and active[0] == str(generation_id):
                raise GenerationActiveError("The active generation cannot be discarded")
            cursor.execute(
                f"DELETE FROM {self.generations} WHERE id = %s",
                (str(generation_id),),
            )
            cursor.execute(
                f"UPDATE {self.corpora} "
                "SET history = (SELECT COALESCE(jsonb_agg(value), '[]'::jsonb) "
                "FROM jsonb_array_elements_text(history) AS item(value) "
                "WHERE item.value <> %s) "
                "WHERE tenant = %s AND corpus = %s",
                (str(generation_id), row[0], row[1]),
            )


def _assert_scope(
    record: GenerationRecord, tenant: str | None, corpus: str | None
) -> None:
    _assert_values(str(record.tenant), str(record.corpus), tenant, corpus)


def _assert_values(
    actual_tenant: str, actual_corpus: str, tenant: str | None, corpus: str | None
) -> None:
    if tenant is not None and tenant != actual_tenant:
        raise TenantIsolationError("Generation belongs to another tenant")
    if corpus is not None and corpus != actual_corpus:
        raise TenantIsolationError("Generation belongs to another corpus")


def _scope_key(tenant: TenantId, corpus: CorpusId) -> str:
    return json.dumps([str(tenant), str(corpus)], separators=(",", ":"))


def _parse_scope(value: str) -> tuple[TenantId, CorpusId]:
    tenant, corpus = json.loads(value)
    return TenantId(tenant), CorpusId(corpus)


def _record(value: dict[str, Any]) -> GenerationRecord:
    return GenerationRecord(
        id=GenerationId(value["id"]),
        tenant=TenantId(value["tenant"]),
        corpus=CorpusId(value["corpus"]),
        vector_ids=tuple(value["vector_ids"]),
        document_count=value["document_count"],
        checksum=value["checksum"],
    )


def _row_record(row: Sequence[Any]) -> GenerationRecord:
    return GenerationRecord(
        id=GenerationId(row[0]),
        tenant=TenantId(row[1]),
        corpus=CorpusId(row[2]),
        vector_ids=tuple(_json(row[3])),
        document_count=row[4],
        checksum=row[5],
    )


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value
