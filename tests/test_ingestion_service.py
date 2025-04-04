import unittest
from ragdoll.ingestion.ingestion_service import IngestionService

class TestIngestionService(unittest.TestCase):

    def test_ingestion_service(self):
        ingestion_service = IngestionService()

        test_sources = [
            "https://www.langchain.com",
            "test_docs/dummy.txt",
            "test_docs/ukpga_20070003_en.pdf",
          ]

        ingested_documents = ingestion_service.ingest_documents(test_sources)

        self.assertIsInstance(ingested_documents, list)
        self.assertTrue(len(ingested_documents) > 0)

        for document in ingested_documents:
            self.assertTrue(document.page_content != "")

            if document.metadata.get("source") == "tests/dummy.txt":
                self.assertTrue("This is a dummy text file for testing." in document.page_content)

if __name__ == "__main__":
    unittest.main()