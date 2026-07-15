# Production-capable SDK

RAGdoll owns ingestion, corpus publication, retrieval, and failure semantics. Execution queues and databases remain replaceable adapters.

## One interface across deployments

```python
from ragdoll import Ragdoll

rag = Ragdoll.from_config("examples/deployment_local.yaml")

job = await rag.ingest(
    tenant="acme",
    corpus="product-docs",
    sources=["docs/", "s3://acme-manuals/"],
    idempotency_key="manuals-2026-07-15",
)
result = await job.wait()

answer = await rag.query(
    tenant="acme",
    corpus="product-docs",
    question="How does authentication work?",
)
```

For scripts without an event loop, use `rag.query_sync(..., corpus="product-docs")`. The original unscoped vector path remains available by omitting `corpus`, but new production code should always name a corpus.

RAGdoll 3 makes the scoped `query` method asynchronous. Applications upgrading
from 2.x should change `rag.query(...)` to `await rag.query(...)`; synchronous
callers can move to `rag.query_sync(...)`.

## Profiles

- `deployment_local.yaml`: inline jobs, atomic file state, persistent Chroma.
- `deployment_application.yaml`: the same interfaces on one application host.
- `deployment_scaled.yaml`: Celery workers, PostgreSQL job and generation state, and Qdrant vectors.

Install the scaled adapters with `pip install "python-ragdoll[scaled]"`. Start a worker with:

```bash
celery -A examples.celery_worker:celery_app worker --loglevel=INFO
```

Workers receive only a job ID. PostgreSQL is authoritative for job progress and active corpus generations; Celery result state is not used. Generation promotion is transactional, so queries continue using the previous generation until vector and optional graph staging succeeds.

## Reliability behavior

- Idempotency is scoped by tenant, corpus, and caller key.
- Content-derived document and chunk IDs remain stable across retries.
- Every source produces an indexed, unchanged, retryable, rejected, or cancelled outcome.
- Staged generations are invisible until promotion.
- Tenant, corpus, and active generation are mandatory backend filters.
- Partial vector or graph writes raise typed failures.
- Failed or cancelled unpublished generations are discarded without touching the active corpus.
- Embedding and relationship-extraction calls support concurrency and request-rate limits.
- Deadlines cancel retrieval and generation tasks.
- Structured events can be routed to logs or OpenTelemetry.

Production adapters run the same visibility and job-store contracts as local
adapters. Set `RAGDOLL_TEST_POSTGRES_DSN` to exercise PostgreSQL, and set both it
and `RAGDOLL_TEST_QDRANT_URL` to exercise the combined Qdrant/PostgreSQL corpus
contract.
