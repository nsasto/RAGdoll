import pytest
from langchain_core.documents import Document

from ragdoll.corpus import InMemoryCorpusIndex
from ragdoll.observability import RecordingEventSink
from ragdoll.query import QueryEngine


class LLM:
    async def call(self, prompt):
        return "four token answer"


@pytest.mark.asyncio
async def test_query_emits_structured_events_and_cost_trace():
    index = InMemoryCorpusIndex()
    generation = await index.stage(
        "acme", "docs", [Document(page_content="short context")]
    )
    await index.promote(generation)
    events = RecordingEventSink()
    engine = QueryEngine(
        index=index,
        llm_caller=LLM(),
        events=events,
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )

    result = await engine.query(tenant="acme", corpus="docs", question="question")

    assert [event.name for event in events.events] == [
        "query.started",
        "query.retrieved",
        "query.completed",
    ]
    assert result.trace.prompt_tokens > 0
    assert result.trace.completion_tokens > 0
    assert result.trace.estimated_cost_usd is not None
