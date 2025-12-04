"""
Reranker Retriever

Wraps any retriever with relevance-based reranking to improve result quality.
Supports multiple reranking strategies: LLM-based, Cohere, and cross-encoder.
"""

from typing import List, Optional, Dict, Any, Literal, Tuple
import logging
from langchain_core.documents import Document

from ragdoll.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class RerankerRetriever(BaseRetriever):
    """
    Retriever wrapper that reranks results by relevance.

    Wraps any base retriever and reranks its results using one of three strategies:
    - LLM: Uses a language model to score document relevance (flexible, but slower)
    - Cohere: Uses Cohere's rerank API (fast, specialized for reranking)
    - Cross-encoder: Uses sentence-transformers cross-encoder models (fast, local)

    Strategy:
    1. Over-retrieve documents from base retriever (fetch N * top_k)
    2. Score each document's relevance to the query
    3. Sort by relevance score
    4. Return top_k highest scoring documents

    Args:
        base_retriever: The underlying retriever to wrap
        reranker_llm: Optional LLM instance for LLM-based reranking
        app_config: AppConfig instance for accessing configuration
        config_manager: Config manager for LLM initialization
        provider: Reranking strategy ("llm", "cohere", or "cross-encoder")
        top_k: Number of documents to return after reranking
        over_retrieve_multiplier: Fetch N*top_k documents before reranking
        score_threshold: Minimum relevance score (0-1) to include documents
        batch_size: Number of documents to rerank in parallel (for LLM provider)
        log_scores: Whether to log reranking scores for debugging
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker_llm: Optional[Any] = None,
        app_config: Optional[Any] = None,
        config_manager: Optional[Any] = None,
        provider: Literal["llm", "cohere", "cross-encoder"] = "llm",
        top_k: int = 5,
        over_retrieve_multiplier: int = 2,
        score_threshold: float = 0.0,
        batch_size: int = 10,
        log_scores: bool = False,
    ):
        self.base_retriever = base_retriever
        self.provider = provider.lower()
        self.top_k = top_k
        self.over_retrieve_multiplier = over_retrieve_multiplier
        self.score_threshold = score_threshold
        self.batch_size = batch_size
        self.log_scores = log_scores
        self.app_config = app_config
        self.config_manager = config_manager

        # Lazy-loaded reranker components
        self._reranker_llm = reranker_llm
        self._cohere_client = None
        self._cross_encoder = None

        # Validate provider
        if self.provider not in ["llm", "cohere", "cross-encoder"]:
            raise ValueError(
                f"Unknown reranker provider: {provider}. "
                f"Must be 'llm', 'cohere', or 'cross-encoder'."
            )

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Retrieve and rerank documents.

        Args:
            query: Query string
            **kwargs: Override parameters (top_k, etc.)

        Returns:
            List of reranked Document objects
        """
        # Allow runtime override of top_k
        final_top_k = kwargs.get("top_k", self.top_k)

        # Over-retrieve from base retriever
        over_retrieve_k = final_top_k * self.over_retrieve_multiplier

        # Remove top_k from kwargs to avoid passing it twice
        retriever_kwargs = {k: v for k, v in kwargs.items() if k != "top_k"}
        documents = self.base_retriever.get_relevant_documents(
            query, top_k=over_retrieve_k, **retriever_kwargs
        )

        if not documents:
            return []

        # If we got fewer documents than requested and no threshold, no need to rerank
        if len(documents) <= final_top_k and self.score_threshold == 0.0:
            return documents[:final_top_k]

        # Rerank based on provider
        try:
            if self.provider == "llm":
                reranked = self._rerank_with_llm(query, documents)
            elif self.provider == "cohere":
                reranked = self._rerank_with_cohere(query, documents)
            else:  # cross-encoder
                reranked = self._rerank_with_cross_encoder(query, documents)

            # Filter by threshold and return top_k
            filtered = [doc for doc, score in reranked if score >= self.score_threshold]
            return filtered[:final_top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # Fallback: return original documents without reranking
            return documents[:final_top_k]

    async def aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """
        Async version of get_relevant_documents.

        Note: Currently delegates to sync version. Override for true async support.
        """
        return self.get_relevant_documents(query, **kwargs)

    def _rerank_with_llm(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Score documents using LLM.

        Args:
            query: Query string
            documents: Documents to rerank

        Returns:
            List of (document, score) tuples sorted by score (descending)
        """
        # Initialize LLM if not already done
        if self._reranker_llm is None:
            self._reranker_llm = self._init_llm_reranker()

        if self._reranker_llm is None:
            logger.warning("No LLM available for reranking, returning unsorted")
            return [(doc, 0.5) for doc in documents]

        prompt_template = """Rate the relevance of the following document to the query on a scale of 0-10.
Only respond with a single number between 0 and 10.

Query: {query}

Document: {document}

Relevance (0-10):"""

        scored = []
        for doc in documents:
            # Truncate document content to avoid token limits
            content = doc.page_content[:1000]
            prompt = prompt_template.format(query=query, document=content)

            try:
                # Try to call LLM (supports both LLMCaller and direct LLM)
                if hasattr(self._reranker_llm, "invoke"):
                    response = self._reranker_llm.invoke(prompt)
                elif callable(self._reranker_llm):
                    response = self._reranker_llm(prompt)
                else:
                    logger.warning(f"Unknown LLM interface: {type(self._reranker_llm)}")
                    response = "5"

                # Extract numeric score
                score_str = str(response).strip()
                # Handle responses like "8" or "Score: 8" or "8/10"
                import re

                match = re.search(r"(\d+(?:\.\d+)?)", score_str)
                if match:
                    score = float(match.group(1))
                else:
                    logger.warning(
                        f"Could not extract score from response: {score_str}"
                    )
                    score = 5.0

                # Normalize to 0-1 range
                score = max(0.0, min(10.0, score)) / 10.0

                # Add rerank score to metadata
                doc.metadata["rerank_score"] = score
                scored.append((doc, score))

                if self.log_scores:
                    logger.debug(
                        f"Rerank score {score:.2f} for doc: {content[:100]}..."
                    )

            except Exception as e:
                logger.warning(f"Failed to score document: {e}")
                # Default mid-score on error
                doc.metadata["rerank_score"] = 0.5
                scored.append((doc, 0.5))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _rerank_with_cohere(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Score documents using Cohere's rerank API.

        Args:
            query: Query string
            documents: Documents to rerank

        Returns:
            List of (document, score) tuples sorted by score (descending)
        """
        # Initialize Cohere client if not already done
        if self._cohere_client is None:
            self._cohere_client = self._init_cohere_reranker()

        if self._cohere_client is None:
            logger.warning("Cohere client not available, returning unsorted")
            return [(doc, 0.5) for doc in documents]

        try:
            # Prepare documents for Cohere API
            texts = [doc.page_content for doc in documents]

            # Call Cohere rerank API
            results = self._cohere_client.rerank(
                query=query, documents=texts, top_n=len(documents)
            )

            # Map results back to documents
            scored = []
            for result in results.results:
                doc = documents[result.index]
                score = result.relevance_score
                doc.metadata["rerank_score"] = score
                scored.append((doc, score))

                if self.log_scores:
                    logger.debug(
                        f"Cohere rerank score {score:.2f} for doc: {doc.page_content[:100]}..."
                    )

            # Already sorted by Cohere
            return scored

        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}", exc_info=True)
            return [(doc, 0.5) for doc in documents]

    def _rerank_with_cross_encoder(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Score documents using sentence-transformers cross-encoder.

        Args:
            query: Query string
            documents: Documents to rerank

        Returns:
            List of (document, score) tuples sorted by score (descending)
        """
        # Initialize cross-encoder if not already done
        if self._cross_encoder is None:
            self._cross_encoder = self._init_cross_encoder()

        if self._cross_encoder is None:
            logger.warning("Cross-encoder not available, returning unsorted")
            return [(doc, 0.5) for doc in documents]

        try:
            # Prepare query-document pairs
            pairs = [[query, doc.page_content[:1000]] for doc in documents]

            # Get scores from cross-encoder
            scores = self._cross_encoder.predict(pairs)

            # Normalize scores to 0-1 range using sigmoid
            import numpy as np

            normalized_scores = 1 / (1 + np.exp(-np.array(scores)))

            # Combine documents with scores
            scored = []
            for doc, score in zip(documents, normalized_scores):
                doc.metadata["rerank_score"] = float(score)
                scored.append((doc, float(score)))

                if self.log_scores:
                    logger.debug(
                        f"Cross-encoder score {score:.2f} for doc: {doc.page_content[:100]}..."
                    )

            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}", exc_info=True)
            return [(doc, 0.5) for doc in documents]

    def _init_llm_reranker(self) -> Optional[Any]:
        """
        Initialize LLM reranker, preferring explicit > config > default.

        Returns:
            LLM caller instance or None
        """
        # Already provided explicitly
        if self._reranker_llm is not None:
            return self._reranker_llm

        # Try to load from config
        if self.app_config and self.config_manager:
            raw_config = getattr(self.config_manager, "_config", {})
            reranker_cfg = raw_config.get("retriever", {}).get("reranker", {})
            llm_cfg = reranker_cfg.get("llm", {})

            if llm_cfg.get("model_name"):
                try:
                    from ragdoll.llms import get_llm_caller

                    return get_llm_caller(
                        model_name_or_config=llm_cfg.get("model_name"),
                        config_manager=self.config_manager,
                        app_config=self.app_config,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize reranker LLM from config: {e}"
                    )

        # Try to use default fast model
        try:
            from ragdoll.llms import get_llm_caller

            return get_llm_caller(
                model_name_or_config="gpt-3.5-turbo",
                config_manager=self.config_manager,
                app_config=self.app_config,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize default reranker LLM: {e}")
            return None

    def _init_cohere_reranker(self) -> Optional[Any]:
        """
        Initialize Cohere rerank client.

        Returns:
            Cohere client or None
        """
        try:
            import cohere

            # Try to get API key from config
            api_key = None
            if self.config_manager:
                raw_config = getattr(self.config_manager, "_config", {})
                reranker_cfg = raw_config.get("retriever", {}).get("reranker", {})
                cohere_cfg = reranker_cfg.get("cohere", {})
                api_key_env = cohere_cfg.get("api_key_env", "COHERE_API_KEY")

                # Resolve environment variable
                import os

                api_key = os.environ.get(api_key_env)

            if not api_key:
                logger.warning(
                    "COHERE_API_KEY not found in environment, Cohere reranking unavailable"
                )
                return None

            return cohere.Client(api_key)

        except ImportError:
            logger.warning(
                "cohere package not installed. Install with: pip install cohere"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")
            return None

    def _init_cross_encoder(self) -> Optional[Any]:
        """
        Initialize sentence-transformers cross-encoder.

        Returns:
            CrossEncoder instance or None
        """
        try:
            from sentence_transformers import CrossEncoder

            # Try to get model name from config
            model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
            if self.config_manager:
                raw_config = getattr(self.config_manager, "_config", {})
                reranker_cfg = raw_config.get("retriever", {}).get("reranker", {})
                ce_cfg = reranker_cfg.get("cross_encoder", {})
                model_name = ce_cfg.get("model_name", model_name)

            logger.info(f"Loading cross-encoder model: {model_name}")
            return CrossEncoder(model_name)

        except ImportError:
            logger.warning(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the reranker.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "provider": self.provider,
            "top_k": self.top_k,
            "over_retrieve_multiplier": self.over_retrieve_multiplier,
            "score_threshold": self.score_threshold,
        }

        # Include base retriever stats
        if hasattr(self.base_retriever, "get_stats"):
            stats["base_retriever_stats"] = self.base_retriever.get_stats()

        return stats
