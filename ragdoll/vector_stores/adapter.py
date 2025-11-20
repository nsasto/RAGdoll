"""
Vector Store Adapter for embedding retrieval.

Provides a unified interface for retrieving embeddings by ID from different
vector store backends (Chroma, FAISS, etc.).
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class VectorStoreAdapter:
    """
    Adapter for retrieving embeddings from vector stores by document ID.

    Handles backend-specific APIs for Chroma, FAISS, and provides
    generic fallback for other vector stores.
    """

    def __init__(self, vector_store):
        """
        Initialize the adapter with a vector store instance.

        Args:
            vector_store: LangChain vector store or wrapped vector store instance
        """
        self.vector_store = vector_store
        self._backend = self._detect_backend()
        logger.debug(f"VectorStoreAdapter initialized with backend: {self._backend}")

    def _detect_backend(self) -> str:
        """Detect the vector store backend type."""
        # Check for wrapped vector store
        if hasattr(self.vector_store, "_store"):
            store = self.vector_store._store
        else:
            store = self.vector_store

        class_name = store.__class__.__name__.lower()
        module_name = store.__class__.__module__.lower()

        if "chroma" in class_name or "chroma" in module_name:
            return "chroma"
        elif "faiss" in class_name or "faiss" in module_name:
            return "faiss"
        elif "pinecone" in class_name or "pinecone" in module_name:
            return "pinecone"
        elif "weaviate" in class_name or "weaviate" in module_name:
            return "weaviate"
        else:
            return "generic"

    def get_embeddings_by_ids(
        self,
        vector_ids: List[str],
        validate_dimension: bool = True,
        expected_dimension: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Retrieve embeddings for given vector IDs.

        Args:
            vector_ids: List of vector store document IDs
            validate_dimension: Whether to validate embedding dimensions
            expected_dimension: Expected embedding dimension (for validation)

        Returns:
            Dictionary mapping vector_id to embedding array (numpy)

        Raises:
            ValueError: If embeddings cannot be retrieved or dimensions mismatch
        """
        if not vector_ids:
            return {}

        try:
            if self._backend == "chroma":
                return self._get_chroma_embeddings(
                    vector_ids, validate_dimension, expected_dimension
                )
            elif self._backend == "faiss":
                return self._get_faiss_embeddings(
                    vector_ids, validate_dimension, expected_dimension
                )
            else:
                return self._get_generic_embeddings(
                    vector_ids, validate_dimension, expected_dimension
                )
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings by IDs: {e}")
            return {}

    def _get_chroma_embeddings(
        self,
        vector_ids: List[str],
        validate_dimension: bool,
        expected_dimension: Optional[int],
    ) -> Dict[str, np.ndarray]:
        """Retrieve embeddings from Chroma backend."""
        # Access the underlying Chroma collection
        store = (
            self.vector_store._store
            if hasattr(self.vector_store, "_store")
            else self.vector_store
        )

        try:
            # Chroma's get() method with include embeddings
            if hasattr(store, "_collection"):
                collection = store._collection
            elif hasattr(store, "get"):
                # Direct Chroma store access
                results = store.get(ids=vector_ids, include=["embeddings"])
                embeddings_dict = {}

                if results and "embeddings" in results and results["embeddings"]:
                    for i, vec_id in enumerate(results.get("ids", [])):
                        embedding = np.array(results["embeddings"][i])

                        if validate_dimension and expected_dimension:
                            if len(embedding) != expected_dimension:
                                logger.warning(
                                    f"Embedding dimension mismatch for {vec_id}: "
                                    f"expected {expected_dimension}, got {len(embedding)}"
                                )
                                continue

                        embeddings_dict[vec_id] = embedding

                return embeddings_dict
            else:
                logger.warning(
                    "Chroma backend detected but cannot access .get() method"
                )
                return {}
        except Exception as e:
            logger.error(f"Error retrieving Chroma embeddings: {e}")
            return {}

    def _get_faiss_embeddings(
        self,
        vector_ids: List[str],
        validate_dimension: bool,
        expected_dimension: Optional[int],
    ) -> Dict[str, np.ndarray]:
        """Retrieve embeddings from FAISS backend."""
        store = (
            self.vector_store._store
            if hasattr(self.vector_store, "_store")
            else self.vector_store
        )

        try:
            # FAISS stores ID->index mapping in docstore
            if not hasattr(store, "index") or not hasattr(
                store, "index_to_docstore_id"
            ):
                logger.warning("FAISS backend missing index or docstore mapping")
                return {}

            embeddings_dict = {}
            index_to_id = store.index_to_docstore_id
            id_to_index = {doc_id: idx for idx, doc_id in index_to_id.items()}

            for vec_id in vector_ids:
                if vec_id not in id_to_index:
                    logger.debug(f"Vector ID {vec_id} not found in FAISS index")
                    continue

                idx = id_to_index[vec_id]

                # Reconstruct embedding from FAISS index
                try:
                    embedding = store.index.reconstruct(int(idx))
                    embedding = np.array(embedding)

                    if validate_dimension and expected_dimension:
                        if len(embedding) != expected_dimension:
                            logger.warning(
                                f"Embedding dimension mismatch for {vec_id}: "
                                f"expected {expected_dimension}, got {len(embedding)}"
                            )
                            continue

                    embeddings_dict[vec_id] = embedding
                except Exception as e:
                    logger.warning(
                        f"Failed to reconstruct FAISS embedding for {vec_id}: {e}"
                    )
                    continue

            return embeddings_dict

        except Exception as e:
            logger.error(f"Error retrieving FAISS embeddings: {e}")
            return {}

    def _get_generic_embeddings(
        self,
        vector_ids: List[str],
        validate_dimension: bool,
        expected_dimension: Optional[int],
    ) -> Dict[str, np.ndarray]:
        """
        Generic fallback for vector stores without direct embedding access.

        This attempts to use similarity_search_by_vector with dummy query,
        but is not guaranteed to work for all backends.
        """
        logger.warning(
            f"Using generic fallback for {self._backend} backend. "
            "Direct embedding retrieval may not be supported."
        )

        # Generic vector stores typically don't expose embeddings directly
        # Return empty dict and let caller fall back to fuzzy matching
        return {}

    def get_backend_type(self) -> str:
        """Return the detected backend type."""
        return self._backend
