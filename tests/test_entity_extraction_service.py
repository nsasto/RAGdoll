from __future__ import annotations

import pytest
from langchain_core.documents import Document

from ragdoll.entity_extraction.entity_extraction_service import EntityExtractionService


class DummyLLMCaller:
    def __init__(self, provider: str | None = None):
        self.provider = provider

    async def call(self, prompt: str) -> str:  # pragma: no cover - async contract
        return "[]"


@pytest.fixture(autouse=True)
def fake_spacy(monkeypatch):
    class DummyDoc:
        ents: list = []

    class DummyNLP:
        def __call__(self, text: str):
            return DummyDoc()

    class FakeSpacyModule:
        def __init__(self):
            self.cli = self

        def load(self, model_name: str):
            return DummyNLP()

        def download(self, model_name: str):
            return None

    monkeypatch.setattr(
        "ragdoll.entity_extraction.entity_extraction_service.spacy",
        FakeSpacyModule(),
    )


def test_relationship_prompt_provider_override():
    config = {
        "relationship_prompts": {
            "default": "relationship_extraction",
            "providers": {"openai": "relationship_extraction_openai"},
        },
        "relationship_parsing": {"preferred_format": "json"},
    }
    service = EntityExtractionService(config=config, llm_caller=DummyLLMCaller("openai"))

    prompt = service._build_relationship_prompt(
        Document(page_content="Alice founded Example Corp.", metadata={})
    )

    assert "OpenAI-powered knowledge graph analyst" in prompt


def test_relationship_prompt_falls_back_to_default():
    config = {
        "relationship_prompts": {
            "default": "relationship_extraction",
            "providers": {"openai": "relationship_extraction_openai"},
        },
        "relationship_parsing": {"preferred_format": "json"},
    }
    service = EntityExtractionService(config=config, llm_caller=DummyLLMCaller("other"))

    prompt = service._build_relationship_prompt(
        Document(page_content="Alice founded Example Corp.", metadata={})
    )

    assert "relationship extraction expert" in prompt
