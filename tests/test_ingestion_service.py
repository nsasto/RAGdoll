import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from ragdoll.ingestion.ingestion_service import IngestionService, Source, SourceType

# Get the directory of the current test file
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "test_data"

# Fixtures
@pytest.fixture
def ingestion_service():
    return IngestionService(max_threads=2, batch_size=5)

@pytest.fixture
def sample_documents():
    return [{"page_content": "Test content", "metadata": {"source": "test"}}]

# Test _build_sources method
class TestBuildSources:
    def test_build_sources_file(self, ingestion_service, tmp_path):
        # Create a temporary test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        # Test file path input
        sources = ingestion_service._build_sources([str(test_file)])
        assert len(sources) == 1
        assert sources[0].type == SourceType.FILE
        assert sources[0].extension == ".txt"
        assert test_file.name in sources[0].identifier
    
    def test_build_sources_glob(self, ingestion_service, tmp_path, monkeypatch):
        # Create multiple test files
        for i in range(3):
            test_file = tmp_path / f"test{i}.txt"
            test_file.write_text(f"Test content {i}")
        
        # Patch Path.glob to work with the test directory
        original_glob = Path.glob
        def mock_glob(self, pattern):
            if str(self) == "." and "*" in pattern:
                return tmp_path.glob("*.txt")
            return original_glob(self, pattern)
        
        monkeypatch.setattr(Path, "glob", mock_glob)
        
        # Test glob pattern
        sources = ingestion_service._build_sources(["*.txt"])
        assert len(sources) == 3
        assert all(s.type == SourceType.FILE for s in sources)
    
    def test_build_sources_arxiv(self, ingestion_service):
        sources = ingestion_service._build_sources(["https://arxiv.org/abs/1234.56789"])
        assert len(sources) == 1
        assert sources[0].type == SourceType.ARXIV
        assert sources[0].identifier == "1234.56789"
    
    def test_build_sources_website(self, ingestion_service):
        sources = ingestion_service._build_sources(["https://example.com"])
        assert len(sources) == 1
        assert sources[0].type == SourceType.WEBSITE
        assert sources[0].identifier == "https://example.com"
    
    def test_build_sources_invalid(self, ingestion_service):
        sources = ingestion_service._build_sources(["nonexistent_file.txt"])
        assert len(sources) == 0

# Test _load_source method
class TestLoadSource:
    def test_load_text_file(self, ingestion_service, sample_documents):
        # Create a mock loader class and instance
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_documents
        mock_loader_class = MagicMock(return_value=mock_loader_instance)
        
        # Directly set the mock in the loaders dictionary
        ingestion_service.loaders = {".txt": mock_loader_class}
        
        # Create a source
        source = Source(type=SourceType.FILE, identifier="test.txt", extension=".txt")
        
        # Test loading
        docs = ingestion_service._load_source(source)
        assert docs == sample_documents
        mock_loader_class.assert_called_once_with(file_path="test.txt")
    
    @patch('ragdoll.ingestion.ingestion_service.ArxivRetriever')
    def test_load_arxiv(self, mock_arxiv_retriever, ingestion_service, sample_documents):
        # Mock the retriever
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.get_relevant_documents.return_value = sample_documents
        mock_arxiv_retriever.return_value = mock_retriever_instance
        
        # Create a source
        source = Source(type=SourceType.ARXIV, identifier="1234.56789")
        
        # Test loading
        docs = ingestion_service._load_source(source)
        assert docs == sample_documents
        mock_retriever_instance.get_relevant_documents.assert_called_once_with(query="1234.56789")
    
    @patch('ragdoll.ingestion.ingestion_service.WebLoader')
    def test_load_website(self, mock_web_loader, ingestion_service, sample_documents):
        # Mock the web loader
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = sample_documents
        mock_web_loader.return_value = mock_loader_instance
        
        # Create a source
        source = Source(type=SourceType.WEBSITE, identifier="https://example.com")
        
        # Test loading
        docs = ingestion_service._load_source(source)
        assert docs == sample_documents
        mock_loader_instance.load.assert_called_once_with("https://example.com")
    
    def test_load_unsupported(self, ingestion_service):
        source = Source(type=SourceType.FILE, identifier="test.unknown", extension=".unknown")
        docs = ingestion_service._load_source(source)
        assert docs == []

# Test ingest_documents method
class TestIngestDocuments:
    @patch.object(IngestionService, '_build_sources')
    @patch.object(IngestionService, '_load_source')
    def test_ingest_documents_success(self, mock_load_source, mock_build_sources, ingestion_service, sample_documents):
        # Setup mocks
        mock_build_sources.return_value = [
            Source(type=SourceType.FILE, identifier="test1.txt", extension=".txt"),
            Source(type=SourceType.FILE, identifier="test2.txt", extension=".txt")
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
        
        # Setup mocks
        mock_build_sources.return_value = [
            Source(type=SourceType.FILE, identifier=f"test{i}.txt", extension=".txt")
            for i in range(5)  # 5 sources with batch size 2 should create 3 batches
        ]
        mock_load_source.return_value = sample_documents
        
        # Test batch processing
        service.ingest_documents(["test*.txt"])
        assert mock_load_source.call_count == 5
