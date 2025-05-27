import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from ragdoll.vector_stores import get_vector_store, add_documents
from langchain_community.vectorstores import FAISS, Chroma, Pinecone


class MockEmbeddings(Embeddings):
    """Mock embedding model for testing."""
    def embed_documents(self, texts):
        # Return a fixed-size vector for each text
        return [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in texts]

    def embed_query(self, text):
        # Return a fixed-size vector for query
        return [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def embedding_model():
    """Fixture for mock embedding model."""
    return MockEmbeddings()


@pytest.fixture
def mock_texts():
    """Fixture for simple text examples."""
    return ["Document 1", "Document 2", "Document 3"]


@pytest.fixture
def mock_documents():
    """Fixture for Document objects."""
    return [
        Document(page_content="Document 1", metadata={"id": "1"}),
        Document(page_content="Document 2", metadata={"id": "2"}),
        Document(page_content="Document 3", metadata={"id": "3"})
    ]


@pytest.fixture
def mock_config_manager():
    """Fixture for mock config manager."""
    mock = MagicMock()
    mock._config = {
        "vector_store": {
            "store_type": "faiss",
            "persist_directory": "./test_vectors",
            "collection_name": "test_collection",
            "faiss": {
                "distance_strategy": "cosine"
            },
            "chroma": {
                "collection_name": "test_collection"
            },
            "pinecone": {
                "api_key": "fake_api_key",
                "environment": "fake_env",
                "index_name": "test_index"
            }
        }
    }
    return mock


def test_get_vector_store_default(embedding_model):
    """Test creating a vector store with default settings."""
    with patch("langchain_community.vectorstores.FAISS.from_texts") as mock_from_texts:
        mock_faiss = MagicMock(spec=FAISS)
        mock_from_texts.return_value = mock_faiss
        
        store = get_vector_store(embedding_model=embedding_model)
        
        mock_from_texts.assert_called_once()
        assert isinstance(store, MagicMock)


def test_get_vector_store_with_config_manager(mock_config_manager, embedding_model):
    """Test that get_vector_store uses the config manager."""
    with patch("ragdoll.vector_stores.ConfigManager", return_value=mock_config_manager):
        with patch("langchain_community.vectorstores.FAISS.from_texts") as mock_from_texts:
            mock_faiss = MagicMock(spec=FAISS)
            mock_from_texts.return_value = mock_faiss
            
            store = get_vector_store(config_manager=mock_config_manager, embedding_model=embedding_model)
            
            mock_from_texts.assert_called_once()
            assert store is not None


def test_get_vector_store_with_config_dict(embedding_model):
    """Test creating a vector store with a config dictionary."""
    config = {
        "vector_store": {
            "store_type": "faiss",
            "persist_directory": "./test_vectors"
        }
    }
    
    with patch("langchain_community.vectorstores.FAISS.from_texts") as mock_from_texts:
        mock_faiss = MagicMock(spec=FAISS)
        mock_from_texts.return_value = mock_faiss
        
        store = get_vector_store(config=config, embedding_model=embedding_model)
        
        mock_from_texts.assert_called_once()
        assert store is not None


def test_get_vector_store_faiss_with_texts(embedding_model, mock_texts):
    """Test creating FAISS store with texts."""
    with patch("langchain_community.vectorstores.FAISS.from_texts") as mock_from_texts:
        mock_faiss = MagicMock(spec=FAISS)
        mock_from_texts.return_value = mock_faiss
        
        store = get_vector_store(
            store_type="faiss",
            embedding_model=embedding_model,
            texts=mock_texts
        )
        
        # Check that from_texts was called with the mock texts
        mock_from_texts.assert_called_once()
        args, kwargs = mock_from_texts.call_args
        assert args[0] == mock_texts
        assert store == mock_faiss


def test_get_vector_store_faiss_with_documents(embedding_model, mock_documents):
    """Test creating FAISS store with documents."""
    with patch("langchain_community.vectorstores.FAISS.from_documents") as mock_from_docs:
        mock_faiss = MagicMock(spec=FAISS)
        mock_from_docs.return_value = mock_faiss
        
        store = get_vector_store(
            store_type="faiss",
            embedding_model=embedding_model,
            documents=mock_documents
        )
        
        # Check that from_documents was called with our mock documents
        mock_from_docs.assert_called_once()
        args, kwargs = mock_from_docs.call_args
        assert args[0] == mock_documents
        assert store == mock_faiss


def test_get_vector_store_faiss_load_local(embedding_model):
    """Test loading FAISS from disk."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a dummy file to simulate existing index
        os.makedirs(temp_dir, exist_ok=True)
        with open(os.path.join(temp_dir, "index.faiss"), "w") as f:
            f.write("dummy")
            
        with patch("langchain_community.vectorstores.FAISS.load_local") as mock_load_local:
            mock_faiss = MagicMock(spec=FAISS)
            mock_load_local.return_value = mock_faiss
            
            store = get_vector_store(
                store_type="faiss",
                embedding_model=embedding_model,
                persist_directory=temp_dir
            )
            
            # Check that load_local was called with the right directory
            mock_load_local.assert_called_once()
            args, kwargs = mock_load_local.call_args
            assert args[0] == temp_dir
            assert store == mock_faiss


def test_get_vector_store_chroma_with_documents(embedding_model, mock_documents):
    """Test creating Chroma store with documents."""
    with patch("langchain_community.vectorstores.Chroma.from_documents") as mock_from_docs:
        mock_chroma = MagicMock(spec=Chroma)
        mock_from_docs.return_value = mock_chroma
        
        store = get_vector_store(
            store_type="chroma",
            embedding_model=embedding_model,
            documents=mock_documents,
            collection_name="test_collection"
        )
        
        # Check that from_documents was called with our mock documents
        mock_from_docs.assert_called_once()
        args, kwargs = mock_from_docs.call_args
        assert args[0] == mock_documents
        assert kwargs.get("collection_name") == "test_collection"
        assert store == mock_chroma


def test_get_vector_store_chroma_persist(embedding_model, mock_documents):
    """Test creating persistent Chroma store."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("langchain_community.vectorstores.Chroma.from_documents") as mock_from_docs:
            mock_chroma = MagicMock(spec=Chroma)
            mock_from_docs.return_value = mock_chroma
            
            store = get_vector_store(
                store_type="chroma",
                embedding_model=embedding_model,
                documents=mock_documents,
                persist_directory=temp_dir,
                collection_name="test_collection"
            )
            
            # Check that from_documents was called with persistence directory
            mock_from_docs.assert_called_once()
            args, kwargs = mock_from_docs.call_args
            assert kwargs.get("persist_directory") == temp_dir
            assert store == mock_chroma


def test_get_vector_store_chroma_empty(embedding_model):
    """Test creating empty Chroma store."""
    # Create a more complete mock for chromadb with __version__ attribute
    chromadb_mock = MagicMock()
    chromadb_mock.__version__ = "0.4.0"  # Set a version
    
    # First, mock the chromadb import with our enhanced mock
    with patch.dict('sys.modules', {
        'chromadb': chromadb_mock, 
        'chromadb.config': MagicMock()
    }):
        # Then patch the Chroma class constructor directly
        with patch("langchain_community.vectorstores.Chroma.__init__", return_value=None) as mock_init:
            with patch("langchain_community.vectorstores.Chroma") as mock_chroma_class:
                # Create our return object
                mock_chroma = MagicMock(spec=Chroma)
                mock_chroma_class.return_value = mock_chroma
                
                # Call the function with chroma store type
                store = get_vector_store(
                    store_type="chroma",
                    embedding_model=embedding_model,
                    collection_name="test_collection"
                )
                
                # Check that constructor was called with expected args
                mock_init.assert_called()
                args, kwargs = mock_init.call_args
                assert kwargs.get("collection_name") == "test_collection"
                assert kwargs.get("embedding_function") == embedding_model


@pytest.mark.skip(reason="Pinecone tests require additional setup")
def test_get_vector_store_pinecone_with_documents(embedding_model, mock_documents):
    """Test creating Pinecone store with documents."""
    with patch("pinecone.init") as mock_init:
        with patch("pinecone.list_indexes", return_value=["test_index"]):
            with patch("langchain_community.vectorstores.Pinecone.from_documents") as mock_from_docs:
                mock_pinecone = MagicMock(spec=Pinecone)
                mock_from_docs.return_value = mock_pinecone
                
                store = get_vector_store(
                    store_type="pinecone",
                    embedding_model=embedding_model,
                    documents=mock_documents,
                    api_key="fake_api_key",
                    environment="fake_env",
                    index_name="test_index"
                )
                
                # Check that Pinecone was initialized and from_documents was called
                mock_init.assert_called_once_with(api_key="fake_api_key", environment="fake_env")
                mock_from_docs.assert_called_once()
                args, kwargs = mock_from_docs.call_args
                assert args[0] == mock_documents
                assert kwargs.get("index_name") == "test_index"
                assert store == mock_pinecone


def test_get_vector_store_unknown_type():
    """Test that unknown store type falls back to default."""
    with patch("langchain_community.vectorstores.FAISS.from_texts") as mock_from_texts:
        mock_faiss = MagicMock(spec=FAISS)
        mock_from_texts.return_value = mock_faiss
        
        with patch("ragdoll.vector_stores.logger.warning") as mock_logger:
            store = get_vector_store(store_type="unknown_type")
            
            # Check that a warning was logged and fallback to FAISS
            mock_logger.assert_called_once()
            mock_from_texts.assert_called_once()
            assert store is not None


def test_get_vector_store_exception_handling():
    """Test exception handling in vector store creation."""
    with patch("langchain_community.vectorstores.FAISS.from_texts", side_effect=Exception("Test error")):
        with patch("ragdoll.vector_stores.logger.error") as mock_logger:
            with pytest.raises(Exception):
                get_vector_store(store_type="faiss")
            
            # Check that error was logged
            mock_logger.assert_called_once()


def test_add_documents_to_existing_store():
    """Test adding documents to an existing vector store."""
    mock_store = MagicMock()
    mock_store.add_documents = MagicMock()
    
    mock_docs = [Document(page_content="New document")]
    
    add_documents(mock_store, mock_docs)
    
    # Check that add_documents was called on the store
    mock_store.add_documents.assert_called_once_with(mock_docs)


def test_add_documents_using_from_documents():
    """Test adding documents using from_documents when add_documents isn't available."""
    mock_store = MagicMock()
    # Set add_documents to None
    delattr(mock_store, "add_documents") if hasattr(mock_store, "add_documents") else None
    # But has from_documents
    mock_store.from_documents = MagicMock()
    
    mock_docs = [Document(page_content="New document")]
    
    add_documents(mock_store, mock_docs)
    
    # Check that from_documents was called as fallback
    mock_store.from_documents.assert_called_once()


def test_add_documents_no_support_raises_error():
    """Test that an error is raised when a store doesn't support adding documents."""
    mock_store = MagicMock()
    # No add_documents or from_documents methods
    delattr(mock_store, "add_documents") if hasattr(mock_store, "add_documents") else None
    delattr(mock_store, "from_documents") if hasattr(mock_store, "from_documents") else None
    
    mock_docs = [Document(page_content="New document")]
    
    with pytest.raises(ValueError):
        add_documents(mock_store, mock_docs)


def test_embedding_model_auto_creation():
    """Test that an embedding model is automatically created if not provided."""
    with patch("ragdoll.vector_stores.get_embedding_model") as mock_get_model:
        mock_embeddings = MagicMock(spec=Embeddings)
        mock_get_model.return_value = mock_embeddings
        
        with patch("langchain_community.vectorstores.FAISS.from_texts") as mock_from_texts:
            mock_faiss = MagicMock(spec=FAISS)
            mock_from_texts.return_value = mock_faiss
            
            store = get_vector_store()
            
            # Check that get_embedding_model was called
            mock_get_model.assert_called_once()
            # And the returned model was used for the vector store
            assert mock_from_texts.called
            args, kwargs = mock_from_texts.call_args
            # Fix keyword argument name
            assert kwargs.get("embedding") == mock_embeddings


if __name__ == "__main__":
    pytest.main(["-xvs", "test_vector_store.py"])