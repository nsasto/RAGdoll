import pytest
from ragdoll.ingestion.ingestion_service import IngestionService

def test_ingestion_service():
    ingestion_service = IngestionService()

    test_sources = [
        "https://www.langchain.com",
        "test_docs/dummy.txt",
        "test_docs/ukpga_20070003_en.pdf",
    ]

    ingested_documents = ingestion_service.ingest_documents(test_sources)

    # Check if documents were ingested successfully
    assert isinstance(ingested_documents, list)
    assert len(ingested_documents) > 0

    # Verify document content
    for document in ingested_documents:
        assert document.page_content != ""

        if document.metadata.get("source") == "tests/dummy.txt":
            assert "This is a dummy text file for testing." in document.page_content

if __name__ == "__main__":
    pytest.main()