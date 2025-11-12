import os
import pytest
import logging
from unittest.mock import patch, MagicMock
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from pathlib import Path
from ragdoll.ingestion import DocumentLoaderService as IngestionService, Source

# Get the directory of the current test file
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "test_data"


# Fixtures
@pytest.fixture
def ingestion_service():
    """Create an ingestion service with mocked metrics manager."""
    logging.basicConfig(level=logging.DEBUG)
    """Create an ingestion service with mocked metrics manager."""
    service = IngestionService(max_threads=2, batch_size=5)
    logging.info(
        f"Loaders on ingestion service fixture: {list(service.loaders.keys())}"
    )

    # Mock the metrics manager methods to avoid errors
    if hasattr(service, "metrics_manager") and service.metrics_manager is not None:
        service.metrics_manager.start_session = MagicMock(return_value={})
        service.metrics_manager.end_session = MagicMock(return_value={})
    return service


@pytest.fixture()
def sample_documents():
    return [{"page_content": "Test content", "metadata": {"source": "test"}}]


@pytest.fixture
def clean_ingestion_service(sample_documents):
    """Create an IngestionService with no caching and mocked network loaders."""
    service = IngestionService(use_cache=False)

    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = sample_documents
    service.loaders["website"] = MagicMock(return_value=mock_loader_instance)

    return service


# Test _build_sources method
class TestBuildSources:
    def test_build_sources_file(self, ingestion_service, tmp_path):
        """Test building sources from a file."""

        # Create a temporary test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        # Test file path input
        sources = ingestion_service._build_sources([str(test_file.absolute())])

        assert len(sources) == 1
        assert sources[0].is_file
        assert sources[0].extension == ".txt"
        assert str(test_file.absolute()) == sources[0].identifier

    def test_build_sources_glob(self, ingestion_service, tmp_path):
        """Test building sources from a glob pattern."""

        # Create a temporary directory for test files

        # Create multiple test files
        for i in range(3):
            test_file = tmp_path / f"test{i}.txt"
            test_file.write_text(f"Test content {i}")
        glob_pattern = str(tmp_path / "*.txt")
        sources = ingestion_service._build_sources([glob_pattern])

        assert len(sources) == 3
        assert all(s.is_file for s in sources)

    def test_build_sources_arxiv(self, ingestion_service):
        # Test arxiv
        sources = ingestion_service._build_sources(["https://arxiv.org/abs/1234.5678"])

        assert len(sources) == 1
        assert not sources[0].is_file
        assert sources[0].identifier == "https://arxiv.org/abs/1234.5678"

    def test_build_sources_website(self, ingestion_service):
        # Test website
        sources = ingestion_service._build_sources(["https://example.com"])
        assert len(sources) == 1
        assert not sources[0].is_file
        assert sources[0].identifier == "https://example.com"

    def test_build_sources_invalid(self, ingestion_service):
        # Test invalid source
        sources = ingestion_service._build_sources(["nonexistent_file.txt"])
        assert len(sources) == 0

    def test_build_sources_https_pdf(self, ingestion_service):
        sources = ingestion_service._build_sources(["https://example.com/file.pdf"])
        assert len(sources) == 1
        assert not sources[0].is_file
        assert sources[0].extension == ".pdf"
        assert sources[0].identifier == "https://example.com/file.pdf"


# Test _load_source method
class TestLoadSource:
    def test_load_text_file(self, ingestion_service, sample_documents):
        # Create a simple loader class that records the file_path it receives
        class DummyLoader:
            def __init__(self, file_path=None, path=None, web_path=None):
                self.received_path = file_path or path or web_path

            def load(self):
                return sample_documents

        ingestion_service.loaders = {".txt": DummyLoader}

        source = Source(is_file=True, identifier="test.txt", extension=".txt")

        docs = ingestion_service._load_source(source)
        assert docs == sample_documents
        assert docs[0]["page_content"] == sample_documents[0]["page_content"]

    def test_load_arxiv(self, clean_ingestion_service, sample_documents, monkeypatch):
        # Set up the mock that was already injected
        mock_config = MagicMock()
        mock_config.get_loader_mapping.return_value = {}

        with monkeypatch.context() as m:
            m.setattr(clean_ingestion_service, "config_manager", mock_config)
        # Create a source
        source = Source(is_file=False, identifier="1234.56789")

        # Test loading - using clean_ingestion_service which already has mocks
        with pytest.raises(ValueError, match="Unsupported source"):
            clean_ingestion_service._load_source(source)

    def test_load_website(self, clean_ingestion_service, sample_documents):
        # Create a source
        source = Source(
            is_file=False, identifier="https://example.com", extension="website"
        )
        # Test loading
        docs = clean_ingestion_service._load_source(source)
        assert docs[0]["page_content"] == sample_documents[0]["page_content"]

    def test_load_unsupported(self, ingestion_service):
        # Test that it raises a value error on load source
        # create a source with a file
        test_file = Path("test.unknown")
        test_file.touch()
        source = Source(is_file=True, identifier="test.unknown", extension=".unknown")
        with pytest.raises(ValueError, match=r"Unsupported source: ext=\.unknown"):
            ingestion_service.ingest_documents(["test.unknown"])


# Test ingest_documents method
class TestIngestDocuments:
    @patch.object(IngestionService, "_build_sources")
    @patch.object(IngestionService, "_load_source")
    def test_ingest_documents_success(
        self, mock_load_source, mock_build_sources, ingestion_service, sample_documents
    ):
        # Setup mocks
        mock_build_sources.return_value = [
            Source(is_file=True, identifier="test1.txt", extension=".txt"),
            Source(is_file=True, identifier="test2.txt", extension=".txt"),
        ]
        mock_load_source.return_value = sample_documents

        # Test ingestion
        result = ingestion_service.ingest_documents(["test1.txt", "test2.txt"])
        assert len(result) == 2 * len(sample_documents)
        assert mock_load_source.call_count == 2

    @patch.object(IngestionService, "_build_sources")
    def test_ingest_documents_no_sources(self, mock_build_sources, ingestion_service):
        # Setup mock
        mock_build_sources.return_value = []

        # Test empty sources
        with pytest.raises(ValueError, match="No valid sources found"):
            ingestion_service.ingest_documents(["nonexistent.txt"])

    @patch.object(IngestionService, "_build_sources")
    @patch.object(IngestionService, "_load_source")
    def test_ingest_documents_batching(
        self, mock_load_source, mock_build_sources, sample_documents
    ):
        # Create service with small batch size
        service = IngestionService(batch_size=2, max_threads=1)

        # Mock metrics manager methods for this service instance too
        if hasattr(service, "metrics_manager") and service.metrics_manager is not None:
            service.metrics_manager.start_session = MagicMock(return_value={})
            service.metrics_manager.end_session = MagicMock(return_value={})

        # Setup mocks
        mock_build_sources.return_value = [
            Source(is_file=True, identifier=f"test{i}.txt", extension=".txt")
            for i in range(5)  # 5 sources with batch size 2 should create 3 batches
        ]
        mock_load_source.return_value = sample_documents

        # Test batch processing
        service.ingest_documents(["test*.txt"])
        assert mock_load_source.call_count == 5

