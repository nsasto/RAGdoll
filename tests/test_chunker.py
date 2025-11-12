import pytest
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from ragdoll.chunkers import get_text_splitter, split_documents


def config_with(**chunker_values):
    return {"chunker": chunker_values}


def test_recursive_splitter_uses_config_values():
    splitter = get_text_splitter(
        config=config_with(
            default_splitter="recursive",
            chunk_size=256,
            chunk_overlap=64,
        )
    )
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter._chunk_size == 256
    assert splitter._chunk_overlap == 64


def test_markdown_splitter_has_expected_headers():
    splitter = get_text_splitter(
        config=config_with(
            default_splitter="markdown",
        )
    )
    assert isinstance(splitter, MarkdownTextSplitter)


def test_character_splitter_respects_separator():
    splitter = get_text_splitter(
        config=config_with(
            default_splitter="character",
            separator="--",
        )
    )
    assert isinstance(splitter, CharacterTextSplitter)
    assert splitter._separator == "--"


def test_code_splitter_uses_python_when_configured():
    splitter = get_text_splitter(
        config=config_with(
            default_splitter="code",
            language="python",
        )
    )
    assert isinstance(splitter, PythonCodeTextSplitter)


def test_token_splitter_uses_encoding_from_config():
    splitter = get_text_splitter(
        config=config_with(
            default_splitter="token",
            encoding_name="cl100k_base",
        )
    )
    assert isinstance(splitter, TokenTextSplitter)
    assert splitter._tokenizer.name == "cl100k_base"


def test_unknown_splitter_falls_back_to_recursive(caplog):
    splitter = get_text_splitter(
        config=config_with(
            default_splitter="does-not-exist",
        )
    )
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert any("Unknown splitter type" in record.msg for record in caplog.records)


def test_overlap_is_adjusted_when_too_large():
    splitter = get_text_splitter(
        config=config_with(
            default_splitter="recursive",
            chunk_size=100,
            chunk_overlap=150,
        )
    )
    # chunk_overlap should be reduced to 25% of chunk_size
    assert splitter._chunk_overlap == 25


def test_split_documents_with_custom_splitter():
    doc = Document(page_content="Paragraph one.\n\nParagraph two.", metadata={"id": 1})
    splitter = get_text_splitter(
        config=config_with(
            default_splitter="recursive",
            chunk_size=20,
            chunk_overlap=5,
        )
    )
    chunks = split_documents([doc], text_splitter=splitter)
    assert len(chunks) >= 2
    assert all(isinstance(chunk, Document) for chunk in chunks)


def test_split_documents_markdown_strategy_preserves_headers():
    doc = Document(
        page_content="# Title\n\n## Section\nContent line.",
        metadata={"source": "md"},
    )
    splitter = get_text_splitter(
        config=config_with(
            default_splitter="markdown",
        )
    )
    chunks = split_documents([doc], text_splitter=splitter)
    assert len(chunks) >= 1
    assert chunks[0].metadata["source"] == "md"
