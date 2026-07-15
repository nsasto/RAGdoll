from unittest.mock import Mock

import pytest

from ragdoll.errors import GraphWriteError
from ragdoll.pipeline import IngestionPipeline


def test_graph_store_false_result_is_a_write_failure():
    pipeline = object.__new__(IngestionPipeline)
    pipeline.graph_store = Mock()
    pipeline.graph_store.save_graph.return_value = False
    graph = Mock()

    with pytest.raises(GraphWriteError):
        pipeline._persist_graph_to_store(graph)


def test_graph_store_exception_is_a_write_failure():
    pipeline = object.__new__(IngestionPipeline)
    pipeline.graph_store = Mock()
    pipeline.graph_store.save_graph.side_effect = OSError("connection lost")
    graph = Mock()

    with pytest.raises(GraphWriteError) as exc_info:
        pipeline._persist_graph_to_store(graph)

    assert exc_info.value.retryable is True
