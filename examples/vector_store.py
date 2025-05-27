"""
Example showing how to use RAGdoll's vector store factory with FAISS.
"""
import os
from pathlib import Path
import sys

# Add the parent directory to the path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from ragdoll.embeddings import get_embedding_model
from ragdoll.vector_stores import get_vector_store

print("🔍 RAGdoll Vector Store Example")
print("=" * 50)

# Store whether we're using real or fake embeddings
using_fake_embeddings = False

# Get embedding model - prefer a lightweight model for testing
try:
    embedding_model = get_embedding_model("text-embedding-3-small")
    print(f"✅ Embedding model created: {embedding_model.__class__.__name__}")
except Exception as e:
    print(f"❌ Error creating embedding model: {e}")
    print("Falling back to fake embeddings for testing")
    embedding_model = get_embedding_model("fake")
    print(f"✅ Fallback embedding model created: {embedding_model.__class__.__name__}")
    using_fake_embeddings = True

# Create some sample documents
documents = [
    Document(page_content="This is a test document about AI", metadata={"source": "test1"}),
    Document(page_content="Vector databases store embeddings for semantic search", metadata={"source": "test2"}),
    Document(page_content="RAG systems combine retrieval with generation", metadata={"source": "test3"})
]

# If using fake embeddings, make sure we clear any existing index
# to avoid dimension mismatch with previously created real embeddings
if using_fake_embeddings and os.path.exists(persist_dir):
    print(f"\n⚠️ Using fake embeddings but found existing vector store at {persist_dir}")
    print("🗑️ Removing existing vector store to avoid dimension mismatch")
    import shutil
    shutil.rmtree(persist_dir, ignore_errors=True)

# Ensure the directory exists
persist_dir = "./data/vector_stores/my_faiss"
os.makedirs(os.path.dirname(persist_dir), exist_ok=True)

print("\n📝 Creating vector store...")
try:
    vector_store = get_vector_store(
        store_type="faiss", 
        embedding_model=embedding_model,
        documents=documents,
        persist_directory=persist_dir,
        allow_dangerous_deserialization=True  # Add this parameter
    )
    print(f"✅ Vector store created successfully")
except Exception as e:
    print(f"❌ Error creating vector store: {e}")
    sys.exit(1)

# Save it - Note: With persist_directory, FAISS may have already saved the index
if vector_store is not None and hasattr(vector_store, "save_local"):
    print(f"\n💾 Saving vector store to {persist_dir}")
    vector_store.save_local(persist_dir)
    print("✅ Vector store saved successfully")

# Later, load it back
print(f"\n🔄 Loading vector store from {persist_dir}")
try:
    # Create an identical embedding model to ensure dimensions match
    query_embedding_model = get_embedding_model("text-embedding-3-small")
    
    loaded_store = get_vector_store(
        store_type="faiss",
        embedding_model=query_embedding_model,  # Use this explicitly created model
        persist_directory=persist_dir,
        allow_dangerous_deserialization=True  # Required for loading pickled data
    )
    print(f"✅ Vector store loaded successfully")
except Exception as e:
    print(f"❌ Error loading vector store: {e}")
    sys.exit(1)

# You can now perform similarity searches
if loaded_store is not None:
    query = "How do RAG systems work?"
    print(f"\n🔎 Searching with query: \"{query}\"")
    try:
        results = loaded_store.similarity_search(query, k=2)
        print(f"✅ Found {len(results)} matching documents:")
        
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} " + "-" * 40)
            print(f"📄 Content: {doc.page_content}")
            print(f"📋 Metadata: {doc.metadata}")
            print("-" * 50)
    except AssertionError as e:
        print(f"❌ Dimension mismatch error: {e}")
        print("This typically happens when the embedding model used to create the vector store")
        print("is different from the one used now. Try recreating the vector store or using")
        print("the same embedding model that was used to create it.")
    except Exception as e:
        print(f"❌ Error during search: {e}")
else:
    print("❌ Failed to load vector store. Please check the error messages above.")

print("\n✨ Example completed successfully!")