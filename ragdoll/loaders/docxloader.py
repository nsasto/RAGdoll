import sys
import io
from typing import List, Union, BinaryIO
from langchain_core.documents import Document
from datetime import datetime
import mammoth
import html2text
import os

class DocxLoader:
    def __init__(self, file_path: Union[str, BinaryIO]):
        self.file_path = file_path
        self.metadata = self._extract_file_metadata()

    def _extract_file_metadata(self) -> dict:
        metadata = {
            "source": self.file_path.name if hasattr(self.file_path, 'name') else str(self.file_path),
            "file_extension": ".docx",
        }
        return metadata

    def load(self) -> List[Document]:
        if isinstance(self.file_path, str):
            with open(self.file_path, "rb") as file:
                file_bytes = file.read()
        else:
            file_bytes = self.file_path.read()

        file_stream = io.BytesIO(file_bytes)

        # Step 1: Extract HTML from DOCX
        result = mammoth.convert_to_html(file_stream)
        html_content = result.value

        # Step 2: Convert HTML to Markdown (preserve tables)
        markdown_converter = html2text.HTML2Text()
        markdown_converter.ignore_links = False
        markdown_converter.bypass_tables = False  # Important: preserve tables
        markdown_content = markdown_converter.handle(html_content)

        # Step 3: Optionally split by Markdown headings (##, ###, etc.)
        sections = markdown_content.split("## ")

        documents = []
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            if i > 0:
                section = "## " + section  # Add back heading

            doc_metadata = self.metadata.copy()
            doc_metadata.update({
                "section": i,
                "loaded_at": datetime.utcnow().isoformat(),
            })

            documents.append(Document(page_content=section, metadata=doc_metadata))

        return documents
