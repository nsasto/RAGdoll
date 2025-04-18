import sys
import pandas as pd
from typing import List, Union, BinaryIO
from langchain_core.documents import Document
from datetime import datetime
import openpyxl
import xlrd
import os

class ExcelLoader:
    def __init__(self, file_path: Union[str, BinaryIO]):
        self.file_path = file_path
        self.extension = self._get_extension(file_path)
        self.metadata = self._extract_file_metadata()

    def _get_extension(self, path: Union[str, BinaryIO]) -> str:
        if isinstance(path, str):
            return os.path.splitext(path)[-1].lower()
        elif hasattr(path, 'name'):
            return os.path.splitext(path.name)[-1].lower()
        return ".xlsx"  # default assumption

    def _extract_file_metadata(self) -> dict:
        # Add basic file-level metadata
        metadata = {
            "source": self.file_path.name if hasattr(self.file_path, 'name') else str(self.file_path),
            "file_extension": self.extension,
        }

        try:
            if self.extension == ".xlsx":
                wb = openpyxl.load_workbook(self.file_path, read_only=True)
                props = wb.properties
                metadata.update({
                    "author": props.creator,
                    "created": props.created.isoformat() if props.created else None,
                    "modified": props.modified.isoformat() if props.modified else None,
                    "title": props.title,
                    "description": props.description,
                })
            elif self.extension == ".xls":
                # xlrd doesn't provide author/created info from .xls
                metadata.update({
                    "author": None,
                    "created": None,
                    "modified": None,
                    "title": None,
                    "description": None,
                })
        except Exception as e:
            metadata.update({"metadata_error": str(e)})

        return metadata

    def load(self) -> List[Document]:
        if self.extension == ".xlsx":
            sheets = pd.read_excel(self.file_path, sheet_name=None, engine="openpyxl")
        elif self.extension == ".xls":
            sheets = pd.read_excel(self.file_path, sheet_name=None, engine="xlrd")
        else:
            raise ValueError(f"Unsupported file extension: {self.extension}")

        documents = []
        for sheet_name, df in sheets.items():
            content = df.to_markdown(index=False)
            doc_metadata = self.metadata.copy()
            doc_metadata.update({
                "sheet_name": sheet_name,
                "num_rows": df.shape[0],
                "num_columns": df.shape[1],
                "columns": list(df.columns),
                "loaded_at": datetime.utcnow().isoformat(),
            })
            documents.append(Document(page_content=content, metadata=doc_metadata))

        return documents
