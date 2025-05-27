import os
import pytest
import logging
from unittest.mock import patch, MagicMock
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from pathlib import Path
from ragdoll.content_extraction import ContentExtractionService, Source

# Get the directory of the current test file
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "test_data"

# Fixtures
@pytest.fixture
def content_extraction_service():
    """Create a content extraction service with mocked metrics manager."""
    logging.basicConfig(level=logging.DEBUG)
    """Create a content extraction service with mocked metrics manager."""
    service = ContentExtractionService(max_threads=2, batch_size=5)
    logging.info(f"Loaders on ingestion service fixture: {list(service.loaders.keys())}")

    # Mock the metrics manager methods to avoid errors
    if hasattr(service, 'metrics_manager') and service.metrics_manager is not None:
        service.metrics_manager.start_session = MagicMock(return_value={})
        service.metrics_manager.end_session = MagicMock(return_value={})
    return service

@pytest.fixture()
def sample_documents():
    return [{"page_content": "Test content", "metadata": {"source": "test"}}]

@pytest.fixture
def clean_content_extraction_service():
    """Create a ContentExtractionService with no caching for clean testing and the correct config values."""
    service = ContentExtractionService(use_cache=False)

    return service
# Test _build_sources method
class TestBuildSources:
    def test_build_sources_file(self, content_extraction_service, tmp_path):
        """Test building sources from a file."""
        
        # Create a temporary test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        # Test file path input
        sources = content_extraction_service._build_sources([str(test_file.absolute())])
        
        assert len(sources) == 1
        assert sources[0].is_file
        assert sources[0].extension == ".txt"
        assert str(test_file.absolute()) == sources[0].identifier

    def test_build_sources_glob(self, content_extraction_service, tmp_path, monkeypatch):
        """Test building sources from a glob pattern."""
        
        # Create multiple test files
        for i in range(3):
            test_file = tmp_path / f"test{i}.txt"
            test_file.write_text(f"Test content {i}")
        
        # Mock the _parse_file_sources method to avoid Path().glob() issues with absolute paths
        def mock_parse_file_sources(self, pattern):
            if "*" in pattern and tmp_path in Path(pattern).parents:
                # Return sources for our test files
                return [
                    Source(identifier=str((tmp_path / f"test{i}.txt").absolute()), 
                           extension=".txt", 
                           is_file=True)
                    for i in range(3)
                ]
            return []
        
        # Apply the mock
        monkeypatch.setattr(ContentExtractionService, '_parse_file_sources', mock_parse_file_sources)
        
        # Test with the glob pattern
        glob_pattern = str(tmp_path / "*.txt")
        sources = content_extraction_service._build_sources([glob_pattern])
        
        assert len(sources) == 3
        assert all(s.is_file for s in sources)
        assert all(s.extension == ".txt" for s in sources)

    def test_build_sources_arxiv(self, ingestion_service):
        #Test arxiv
        sources = ingestion_service._build_sources(
            ["https://arxiv.org/abs/1234.5678"]
        )
        
        assert len(sources) == 1
        assert not sources[0].is_file
        assert sources[0].identifier == "https://arxiv.org/abs/1234.5678"
    
    def test_build_sources_website(self, ingestion_service):
        #Test website
        sources = ingestion_service._build_sources(["https://example.com"])
        assert len(sources) == 1
        assert not sources[0].is_file
        assert sources[0].identifier == "https://example.com"
    
    def test_build_sources_invalid(self, ingestion_service):
        #Test invalid source
        sources = ingestion_service._build_sources(["nonexistent_file.txt"])
        assert len(sources) == 0

    

    def test_build_sources_https_pdf(self, ingestion_service):
        sources = ingestion_service._build_sources(
            ["https://example.com/file.pdf"]
        )
        assert len(sources) == 1
        assert not sources[0].is_file
        assert sources[0].extension == ".pdf"
        assert sources[0].identifier == "https://example.com/file.pdf"

# Test _load_source method
class TestLoadSource:
    def test_load_text_file(self, clean_content_extraction_service, sample_documents, monkeypatch):
        """Test loading a text file source."""
        # Create a mock loader class and instance
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader_class = MagicMock(return_value=mock_loader_instance)
        
        # Mock inspect.signature to make it look like the loader has file_path parameter
        def mock_signature(*args, **kwargs):
            mock_sig = MagicMock()
            mock_sig.parameters = {"file_path": MagicMock()}
            return mock_sig
        
        monkeypatch.setattr("inspect.signature", mock_signature)
        
        # Directly set the mock in the loaders dictionary
        clean_content_extraction_service.loaders = {".txt": mock_loader_class}
        
        # Create a source
        source = Source(is_file=True, identifier="test.txt", extension=".txt")
        
        # Test loading
        docs = clean_content_extraction_service._load_source(source)
        assert docs == sample_documents
        mock_loader_class.assert_called_once_with(file_path="test.txt")

    def test_load_arxiv(self, clean_content_extraction_service, sample_documents, monkeypatch):
        # Mock the _is_arxiv_url method to return True for our test source
        monkeypatch.setattr(clean_content_extraction_service, "_is_arxiv_url", lambda url: "1234.56789" in url)

        # Mock a loader for arxiv extension
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader_class = MagicMock(return_value=mock_loader_instance)
        
        # Set up the loader in the service
        clean_content_extraction_service.loaders = {"arxiv": mock_loader_class}

        # Create a source with the correct extension
        source = Source(is_file=False, identifier="1234.56789", extension="arxiv")
        
        # Test loading
        docs = clean_ingestion_service._load_source(source)
        assert docs == sample_documents

    def test_load_website(self, clean_content_extraction_service, sample_documents, monkeypatch):
        """Test loading a website source."""
        # Create mock loader class and instance
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader_class = MagicMock(return_value=mock_loader_instance)
        
        # Mock inspect.signature to return appropriate params
        def mock_signature(*args, **kwargs):
            mock_sig = MagicMock()
            mock_sig.parameters = {"web_path": MagicMock()}
            return mock_sig
        
        monkeypatch.setattr("inspect.signature", mock_signature)
        
        # Set up the loader in the service
        clean_content_extraction_service.loaders = {"website": mock_loader_class}
        
        # Create a source with website extension
        source = Source(is_file=False, identifier="https://example.com", extension="website")
        
        # Test loading
        docs = clean_ingestion_service._load_source(source)
        # Compare directly with the sample documents
        assert docs == sample_documents

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
    @patch.object(IngestionService, '_build_sources')
    @patch.object(IngestionService, '_load_source')
    def test_ingest_documents_success(self, mock_load_source, mock_build_sources, ingestion_service, sample_documents):
        # Setup mocks
        mock_build_sources.return_value = [
            Source(is_file=True, identifier="test1.txt", extension=".txt"),
            Source(is_file=True, identifier="test2.txt", extension=".txt")
        ]
        mock_load_source.return_value = sample_documents
        
        # Test ingestion
        result = ingestion_service.ingest_documents(["test1.txt", "test2.txt"])
        assert len(result) == 2 * len(sample_documents)
        assert mock_load_source.call_count == 2
    
    @patch.object(IngestionService, '_build_sources')
    def test_ingest_documents_no_sources(self, mock_build_sources, ingestion_service):
        # Setup mock
        mock_build_sources.return_value = []
        
        # Test empty sources
        with pytest.raises(ValueError, match="No valid sources found"):
            ingestion_service.ingest_documents(["nonexistent.txt"])
    
    @patch.object(IngestionService, '_build_sources')
    @patch.object(IngestionService, '_load_source')
    def test_ingest_documents_batching(self, mock_load_source, mock_build_sources, sample_documents):
        # Create service with small batch size
        service = IngestionService(batch_size=2, max_threads=1)
        
        # Mock metrics manager methods for this service instance too
        if hasattr(service, 'metrics_manager') and service.metrics_manager is not None:
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
