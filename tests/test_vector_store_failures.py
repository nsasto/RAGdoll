from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from ragdoll.errors import BatchWriteError
from ragdoll.vector_stores.base_vector_store import BaseVectorStore


@pytest.mark.asyncio
async def test_permanent_vector_failure_is_not_returned_as_empty_success_ids():
    store = Mock()
    store.add_documents.side_effect = RuntimeError("backend unavailable")
    vector_store = BaseVectorStore(store)

    with pytest.raises(BatchWriteError) as exc_info:
        await vector_store.add_documents_parallel(
            [Document(page_content="one"), Document(page_content="two")],
            batch_size=2,
            max_concurrent=1,
            retry_failed=True,
        )

    assert exc_info.value.failed_count == 2
    assert exc_info.value.succeeded_count == 0
    assert exc_info.value.retryable is True
