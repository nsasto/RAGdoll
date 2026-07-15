import asyncio

import pytest
from langchain_core.documents import Document

from ragdoll.contracts import IngestionSpec, ItemOutcome, ItemStatus, JobId
from ragdoll.corpus import InMemoryCorpusIndex
from ragdoll.ingestion.jobs import (
    CeleryExecutionAdapter,
    DurableIngestion,
    FileJobStore,
    JobStatus,
    JobRecord,
    MemoryJobStore,
    PreparationResult,
)
from ragdoll.observability import RecordingEventSink
from ragdoll.errors import RejectedSourceError
from ragdoll.quarantine import MemoryQuarantineStore


@pytest.mark.asyncio
async def test_completed_job_publishes_generation_and_reports_every_source():
    async def prepare(spec):
        return [
            Document(page_content=f"content from {source}", metadata={"source": source})
            for source in spec.sources
        ]

    index = InMemoryCorpusIndex()
    ingestion = DurableIngestion(index=index, prepare=prepare)

    job = await ingestion.submit(
        IngestionSpec(corpus="docs", sources=["one.txt", "two.txt"])
    )
    result = await job.wait()

    assert result.status is JobStatus.COMPLETED
    assert [item.status for item in result.items] == [
        ItemStatus.INDEXED,
        ItemStatus.INDEXED,
    ]
    assert all(item.document_id for item in result.items)
    assert all(item.chunk_ids for item in result.items)
    assert len(await index.query("default", "docs", "content")) == 2


@pytest.mark.asyncio
async def test_same_idempotency_key_returns_original_job_without_reprocessing():
    calls = 0

    async def prepare(spec):
        nonlocal calls
        calls += 1
        return [Document(page_content="same content")]

    ingestion = DurableIngestion(
        index=InMemoryCorpusIndex(), prepare=prepare, store=MemoryJobStore()
    )
    spec = IngestionSpec(
        corpus="docs", sources=["one.txt"], idempotency_key="release-42"
    )

    first = await ingestion.submit(spec)
    await first.wait()
    second = await ingestion.submit(spec)
    await second.wait()

    assert second.id == first.id
    assert calls == 1


@pytest.mark.asyncio
async def test_same_idempotency_key_retries_a_retryable_failed_job():
    calls = 0

    async def prepare(spec):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary loader failure")
        return [
            Document(page_content="recovered", metadata={"source": "one.txt"})
        ]

    ingestion = DurableIngestion(index=InMemoryCorpusIndex(), prepare=prepare)
    spec = IngestionSpec(
        corpus="docs", sources=["one.txt"], idempotency_key="retry-safe"
    )

    first = await ingestion.submit(spec)
    assert (await first.wait()).status is JobStatus.FAILED
    retried = await ingestion.submit(spec)

    assert retried.id == first.id
    assert (await retried.wait()).status is JobStatus.COMPLETED
    assert calls == 2


@pytest.mark.asyncio
async def test_cancelling_job_prevents_generation_from_becoming_visible():
    started = asyncio.Event()
    release = asyncio.Event()

    async def prepare(spec):
        started.set()
        await release.wait()
        return [Document(page_content="must stay hidden")]

    index = InMemoryCorpusIndex()
    ingestion = DurableIngestion(index=index, prepare=prepare)
    job = await ingestion.submit(IngestionSpec(corpus="docs", sources=["slow.txt"]))
    await started.wait()

    await job.cancel()
    release.set()
    result = await job.wait()

    assert result.status is JobStatus.CANCELLED
    assert await index.query("default", "docs", "hidden") == []


@pytest.mark.asyncio
async def test_failure_is_recorded_as_retryable_instead_of_success():
    async def prepare(spec):
        raise TimeoutError("loader timed out")

    ingestion = DurableIngestion(index=InMemoryCorpusIndex(), prepare=prepare)
    job = await ingestion.submit(IngestionSpec(corpus="docs", sources=["bad.txt"]))
    result = await job.wait()

    assert result.status is JobStatus.FAILED
    assert result.items[0].status is ItemStatus.RETRYABLE
    assert result.items[0].code == "TimeoutError"


@pytest.mark.asyncio
async def test_retryable_source_failure_does_not_publish_partial_corpus():
    index = InMemoryCorpusIndex()
    original = await index.stage(
        "default", "docs", [Document(page_content="active version")]
    )
    await index.promote(original)

    async def prepare(spec):
        return PreparationResult(
            documents=(
                Document(
                    page_content="partial replacement",
                    metadata={
                        "source": "good.txt",
                        "ingestion_source_id": "good.txt",
                    },
                ),
            ),
            failures=(
                ItemOutcome(
                    source_id="bad.txt",
                    status=ItemStatus.RETRYABLE,
                    code="loader_timeout",
                ),
            ),
        )

    ingestion = DurableIngestion(index=index, prepare=prepare)
    job = await ingestion.submit(
        IngestionSpec(corpus="docs", sources=["good.txt", "bad.txt"])
    )
    result = await job.wait()

    assert result.status is JobStatus.FAILED
    assert [item.status for item in result.items] == [
        ItemStatus.RETRYABLE,
        ItemStatus.RETRYABLE,
    ]
    assert [
        doc.page_content for doc in await index.query("default", "docs", "active")
    ] == ["active version"]


@pytest.mark.asyncio
async def test_file_job_store_survives_recreation(tmp_path):
    async def prepare(spec):
        return [Document(page_content="persisted", metadata={"source": "one.txt"})]

    first_store = FileJobStore(tmp_path)
    first = DurableIngestion(
        index=InMemoryCorpusIndex(), prepare=prepare, store=first_store
    )
    spec = IngestionSpec(
        tenant="acme",
        corpus="docs",
        sources=["one.txt"],
        idempotency_key="stable-request",
    )
    job = await first.submit(spec)
    expected = await job.wait()

    recreated_store = FileJobStore(tmp_path)
    record = await recreated_store.find_idempotent(spec)

    assert record is not None
    assert record.id == job.id
    assert record.result == expected


@pytest.mark.asyncio
async def test_incomplete_file_job_is_resumed_when_resubmitted(tmp_path):
    calls = 0

    async def prepare(spec):
        nonlocal calls
        calls += 1
        return [Document(page_content="resumed", metadata={"source": "one.txt"})]

    store = FileJobStore(tmp_path)
    original = DurableIngestion(
        index=InMemoryCorpusIndex(), prepare=prepare, store=store
    )
    spec = IngestionSpec(
        corpus="docs", sources=["one.txt"], idempotency_key="resume-me"
    )
    job = await original.submit(spec)
    await job.wait()

    record = await store.get(job.id)
    await store.save(
        type(record)(
            id=record.id,
            spec=record.spec,
            status=JobStatus.RUNNING,
            progress=record.progress,
        )
    )

    restarted = DurableIngestion(
        index=InMemoryCorpusIndex(), prepare=prepare, store=FileJobStore(tmp_path)
    )
    resumed = await restarted.submit(spec)
    result = await resumed.wait()

    assert resumed.id == job.id
    assert result.status is JobStatus.COMPLETED
    assert calls == 2


@pytest.mark.asyncio
async def test_staged_checkpoint_resumes_without_reloading_sources(tmp_path):
    index = InMemoryCorpusIndex()
    staged = await index.stage(
        "acme",
        "docs",
        [Document(page_content="checkpointed", metadata={"source": "one.txt"})],
    )
    store = FileJobStore(tmp_path)
    spec = IngestionSpec(tenant="acme", corpus="docs", sources=["one.txt"])
    checkpoint_items = (
        ItemOutcome(
            source_id="one.txt",
            status=ItemStatus.INDEXED,
            document_id=staged.documents[0].metadata["document_id"],
            chunk_ids=(staged.documents[0].metadata["chunk_id"],),
        ),
    )
    record = JobRecord(
        id=JobId("checkpointed-job"),
        spec=spec,
        status=JobStatus.RUNNING,
        staged_generation_id=staged.id,
        staged_checksum=staged.checksum,
        staged_document_count=staged.document_count,
        checkpoint_items=checkpoint_items,
    )
    await store.create(record)

    async def must_not_prepare(spec):
        raise AssertionError("staged checkpoint should bypass preparation")

    ingestion = DurableIngestion(index=index, prepare=must_not_prepare, store=store)
    await ingestion.execute(record.id)

    result = await index.query("acme", "docs", "checkpointed")
    assert [document.page_content for document in result] == ["checkpointed"]
    assert (await store.get(record.id)).result.items == checkpoint_items


@pytest.mark.asyncio
async def test_external_execution_adapter_can_run_job_by_id():
    class ExternalExecution:
        distributed = True

        def __init__(self):
            self.submitted = []

        async def submit(self, job_id, runner):
            self.submitted.append(job_id)

        async def cancel(self, job_id):
            pass

        async def wait(self, job_id):
            pass

    execution = ExternalExecution()

    async def prepare(spec):
        return [Document(page_content="queued", metadata={"source": "one.txt"})]

    ingestion = DurableIngestion(
        index=InMemoryCorpusIndex(), prepare=prepare, execution=execution
    )
    job = await ingestion.submit(IngestionSpec(corpus="docs", sources=["one.txt"]))

    assert execution.submitted == [job.id]
    assert (await job.status()).status is JobStatus.PENDING

    await ingestion.execute(job.id)
    assert (await job.wait()).status is JobStatus.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["memory", "file"])
async def test_job_store_claim_prevents_concurrent_worker_execution(adapter, tmp_path):
    store = MemoryJobStore() if adapter == "memory" else FileJobStore(tmp_path)
    record = JobRecord(
        id=JobId("job-one"),
        spec=IngestionSpec(corpus="docs", sources=["one.txt"]),
    )
    await store.create(record)

    first, second = await asyncio.gather(
        store.claim(record.id, "worker-one", 60),
        store.claim(record.id, "worker-two", 60),
    )

    assert (first is None) != (second is None)
    assert (await store.get(record.id)).attempt == 1


@pytest.mark.asyncio
async def test_celery_adapter_dispatches_named_task_and_revokes_it():
    class Result:
        id = "celery-task-1"

    class Control:
        def __init__(self):
            self.revoked = []

        def revoke(self, task_id, terminate=False):
            self.revoked.append((task_id, terminate))

    class App:
        def __init__(self):
            self.sent = []
            self.control = Control()

        def send_task(self, name, args):
            self.sent.append((name, args))
            return Result()

    app = App()
    adapter = CeleryExecutionAdapter(app, task_name="ragdoll.ingest")

    await adapter.submit("job-1", lambda _: None)
    await adapter.cancel("job-1")

    assert app.sent == [("ragdoll.ingest", ["job-1"])]
    assert app.control.revoked == [("celery-task-1", False)]


@pytest.mark.asyncio
async def test_ingestion_emits_job_lifecycle_events():
    events = RecordingEventSink()

    async def prepare(spec):
        return [Document(page_content="ok", metadata={"source": "one.txt"})]

    ingestion = DurableIngestion(
        index=InMemoryCorpusIndex(), prepare=prepare, events=events
    )
    job = await ingestion.submit(IngestionSpec(corpus="docs", sources=["one.txt"]))
    await job.wait()

    assert [event.name for event in events.events] == [
        "ingestion.submitted",
        "ingestion.started",
        "ingestion.staged",
        "ingestion.completed",
    ]


@pytest.mark.asyncio
async def test_reingesting_identical_content_reports_unchanged():
    async def prepare(spec):
        return [Document(page_content="same", metadata={"source": "one.txt"})]

    ingestion = DurableIngestion(index=InMemoryCorpusIndex(), prepare=prepare)
    first = await ingestion.submit(
        IngestionSpec(corpus="docs", sources=["one.txt"], idempotency_key="first")
    )
    await first.wait()
    second = await ingestion.submit(
        IngestionSpec(corpus="docs", sources=["one.txt"], idempotency_key="second")
    )

    result = await second.wait()

    assert result.items[0].status is ItemStatus.UNCHANGED


@pytest.mark.asyncio
async def test_rejected_source_is_quarantined_but_transient_failure_is_not():
    quarantine = MemoryQuarantineStore()

    async def reject(spec):
        raise RejectedSourceError("unsupported encrypted document")

    ingestion = DurableIngestion(
        index=InMemoryCorpusIndex(), prepare=reject, quarantine=quarantine
    )
    job = await ingestion.submit(
        IngestionSpec(tenant="acme", corpus="docs", sources=["secret.pdf"])
    )

    result = await job.wait()
    quarantined = await quarantine.list("acme", "docs")

    assert result.items[0].status is ItemStatus.REJECTED
    assert quarantined[0].source_id == "secret.pdf"
    assert quarantined[0].code == "rejected_source"
