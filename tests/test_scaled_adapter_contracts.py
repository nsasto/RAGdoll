import os
import uuid

import pytest

from ragdoll.contracts import CorpusId, GenerationId, IngestionSpec, JobId, TenantId
from ragdoll.generation_state import GenerationRecord, PostgresGenerationStateStore
from ragdoll.ingestion.jobs import JobRecord, PostgresJobStore
from ragdoll.config.base_config import VectorStoreConfig
from ragdoll.corpus import VectorCorpusIndex
from ragdoll.vector_stores import vector_store_from_config
from langchain_core.documents import Document

POSTGRES_DSN = os.environ.get("RAGDOLL_TEST_POSTGRES_DSN")
QDRANT_URL = os.environ.get("RAGDOLL_TEST_QDRANT_URL")


class ContractEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0]


@pytest.mark.asyncio
@pytest.mark.skipif(not POSTGRES_DSN, reason="requires RAGDOLL_TEST_POSTGRES_DSN")
async def test_postgres_adapters_pass_shared_state_contract():
    suffix = uuid.uuid4().hex[:10]
    tenant = f"tenant-{suffix}"
    corpus = "docs"
    generation_id = GenerationId(f"generation-{suffix}")
    state = PostgresGenerationStateStore(
        POSTGRES_DSN, table_prefix=f"ragdoll_test_{suffix}"
    )
    generation = GenerationRecord(
        id=generation_id,
        tenant=TenantId(tenant),
        corpus=CorpusId(corpus),
        vector_ids=("chunk-1",),
        document_count=1,
        checksum="checksum",
    )

    await state.put(generation)
    await state.promote(generation_id, tenant=tenant, corpus=corpus)

    assert await state.active(tenant, corpus) == generation_id

    jobs = PostgresJobStore(POSTGRES_DSN, table=f"ragdoll_jobs_{suffix}")
    spec = IngestionSpec(
        tenant=tenant,
        corpus=corpus,
        sources=["one.txt"],
        idempotency_key="same",
    )
    record = JobRecord(id=JobId(f"job-{suffix}"), spec=spec)

    created = await jobs.create(record)
    duplicate = await jobs.create(JobRecord(id=JobId(f"duplicate-{suffix}"), spec=spec))

    assert created.id == duplicate.id
    assert await jobs.claim(created.id, "worker-one", 60) is not None
    assert await jobs.claim(created.id, "worker-two", 60) is None


@pytest.mark.asyncio
@pytest.mark.skipif(
    not POSTGRES_DSN or not QDRANT_URL,
    reason="requires RAGDOLL_TEST_POSTGRES_DSN and RAGDOLL_TEST_QDRANT_URL",
)
async def test_qdrant_and_postgres_pass_corpus_visibility_contract():
    suffix = uuid.uuid4().hex[:10]
    collection = f"ragdoll_contract_{suffix}"
    state = PostgresGenerationStateStore(
        POSTGRES_DSN, table_prefix=f"ragdoll_qdrant_{suffix}"
    )
    store = vector_store_from_config(
        VectorStoreConfig(
            enabled=True,
            store_type="qdrant",
            params={"url": QDRANT_URL, "collection_name": collection},
        ),
        embedding=ContractEmbeddings(),
    )
    index = VectorCorpusIndex(store, state_store=state)
    generation = await index.stage(
        "acme",
        f"docs-{suffix}",
        [Document(page_content="production adapter contract")],
    )

    assert await index.query("acme", f"docs-{suffix}", "contract") == []
    await index.promote(generation)
    results = await index.query("acme", f"docs-{suffix}", "contract")

    assert [document.page_content for document in results] == [
        "production adapter contract"
    ]
    await index.delete("acme", f"docs-{suffix}")
