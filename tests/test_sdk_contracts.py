import pytest

from ragdoll.contracts import (
    CorpusId,
    IngestionSpec,
    ItemOutcome,
    ItemStatus,
    TenantId,
)


def test_ingestion_spec_requires_non_empty_corpus_and_sources():
    with pytest.raises(ValueError):
        IngestionSpec(corpus="", sources=["manual.pdf"])

    with pytest.raises(ValueError):
        IngestionSpec(corpus="docs", sources=[])


def test_ingestion_spec_defaults_to_single_tenant_local_mode():
    spec = IngestionSpec(corpus="docs", sources=["manual.pdf"])

    assert spec.tenant == TenantId("default")
    assert spec.corpus == CorpusId("docs")


def test_item_outcome_makes_retryability_explicit():
    outcome = ItemOutcome(
        source_id="manual.pdf",
        status=ItemStatus.RETRYABLE,
        code="embedding_rate_limited",
        detail="provider returned 429",
    )

    assert outcome.retryable is True
    assert outcome.succeeded is False
