try:
    from ragdoll.ingestion import DocumentLoaderService, Source
    print("✅ Import from ragdoll.ingestion successful")
except ImportError as e:
    print(f"❌ Import from ragdoll.ingestion failed: {e}")

try:
    from langchain_markitdown.loaders import DocxLoader
    print("✅ Import from langchain_markitdown successful")
except ImportError as e:
    print(f"❌ Import from langchain_markitdown failed: {e}")

try:
    from langchain_markitdown import DocxLoader
    print("✅ Import from langchain_markitdown successful")
except ImportError as e:
    print(f"❌ Import from langchain_markitdown failed: {e}")
