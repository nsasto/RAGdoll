from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain.schema import Document
from ragdoll.chunkers import get_text_splitter, split_documents
from ragdoll.embeddings import get_embedding_model
from ragdoll.entity_extraction import get_entity_extractor
from ragdoll.vector_stores import get_vector_store
from ragdoll.graph_stores import get_graph_store
from ragdoll.ingestion import ContentExtractionService
from ragdoll.config import ConfigManager

logger = logging.getLogger("ragdoll.ingestion")

@dataclass
class IngestionOptions:
    """Options for the ingestion pipeline."""
    batch_size: int = 10
    parallel_extraction: bool = False
    max_workers: int = 4
    skip_vector_store: bool = False
    skip_graph_store: bool = False
    extract_entities: bool = True
    collect_metrics: bool = True
    
    # Additional options for sub-components
    chunking_options: Dict[str, Any] = None
    embedding_options: Dict[str, Any] = None
    vector_store_options: Dict[str, Any] = None
    graph_store_options: Dict[str, Any] = None
    entity_extraction_options: Dict[str, Any] = None
    llm: Any = None  # Add LLM parameter to options

class IngestionPipeline:
    """
    Flexible ingestion pipeline for Graph RAG.
    
    Coordinates document extraction, chunking, embedding, entity extraction, 
    and storage in both vector and graph stores.
    """
    
    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        content_extraction_service: Optional[ContentExtractionService] = None,
        text_splitter = None,
        embedding_model = None,
        entity_extractor = None,
        vector_store = None,
        graph_store = None,
        options: Optional[IngestionOptions] = None
    ):
        """
        Initialize the ingestion pipeline with component factories.
        
        Components can be pre-initialized or will be created using factories.
        """
        self.config_manager = config_manager or ConfigManager()
        self.options = options or IngestionOptions()
        
        # Initialize content extraction (for loading documents)
        self.content_extraction_service = content_extraction_service or ContentExtractionService(
            collect_metrics=self.options.collect_metrics
        )
        
        # Initialize text splitter
        self.text_splitter = text_splitter or get_text_splitter(
            config_manager=self.config_manager,
            **(self.options.chunking_options or {})
        )
        
        # Initialize embedding model
        self.embedding_model = embedding_model or get_embedding_model(
            config_manager=self.config_manager,
            **(self.options.embedding_options or {})
        )
        
        # Initialize entity extractor if needed
        if self.options.extract_entities:
            extraction_options = self.options.entity_extraction_options or {}
            # Pass LLM if provided in options
            if hasattr(self.options, 'llm') and self.options.llm is not None:
                extraction_options['llm'] = self.options.llm
            
            # Instead of passing to extract method, set it in the config
            extraction_options['config'] = extraction_options.get('config', {})
            extraction_options['config']['chunking_strategy'] = "none"
            
            self.entity_extractor = entity_extractor or get_entity_extractor(
                config_manager=self.config_manager,
                **extraction_options
            )
        else:
            self.entity_extractor = None
            
        # Initialize vector store if needed
        if not self.options.skip_vector_store:
            self.vector_store = vector_store or get_vector_store(
                embedding_model=self.embedding_model,
                config_manager=self.config_manager,
                **(self.options.vector_store_options or {})
            )
        else:
            self.vector_store = None
            
        # Initialize graph store if needed
        if not self.options.skip_graph_store:
            self.graph_store = graph_store or get_graph_store(
                config_manager=self.config_manager,
                **(self.options.graph_store_options or {})
            )
        else:
            self.graph_store = None
            
        # Statistics for tracking progress
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "vector_entries_added": 0,
            "graph_entries_added": 0,
            "errors": []
        }
        
    async def ingest(self, sources: List[Union[str, Path, Document]]) -> Dict[str, Any]:
        """
        Process sources through the entire ingestion pipeline.
        
        Args:
            sources: List of source identifiers (file paths, URLs) or Document objects
            
        Returns:
            Dictionary containing ingestion statistics
        """
        logger.info(f"Starting ingestion of {len(sources)} sources")
        
        # Reset statistics
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "vector_entries_added": 0,
            "graph_entries_added": 0,
            "errors": []
        }
        
        try:
            # Step 1: Extract content from sources
            documents = await self._extract_documents(sources)
            if not documents:
                logger.warning("No documents extracted from sources")
                return self.stats
                
            self.stats["documents_processed"] = len(documents)
            logger.info(f"Extracted {len(documents)} documents")
            
            # Step 2: Chunk documents
            chunks = self._chunk_documents(documents)
            if not chunks:
                logger.warning("No chunks created from documents")
                return self.stats
                
            self.stats["chunks_created"] = len(chunks)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Process chunks in batches
            await self._process_chunks_in_batches(chunks)
            
            logger.info(f"Ingestion complete: {self.stats}")
            return self.stats
            
        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
            self.stats["errors"].append(str(e))
            return self.stats
    
    async def _extract_documents(self, sources: List[Union[str, Path, Document]]) -> List[Document]:
        """Extract documents from various sources."""
        # If sources are already Document objects, return them directly
        if all(isinstance(source, Document) for source in sources):
            logger.info(f"Sources are already Document objects, returning directly")
            return sources
        
        # Otherwise extract documents using content extraction service
        try:
            # This will be a list of document lists
            nested_documents = self.content_extraction_service.ingest_documents([str(s) for s in sources])
            
            # Flatten the list while preserving source information
            flattened_documents = []
            for doc_list in nested_documents:
                if isinstance(doc_list, list):
                    flattened_documents.extend(doc_list)
                else:
                    # Sometimes might get a single document instead of a list
                    flattened_documents.append(doc_list)
                
            logger.info(f"Content extraction returned {len(flattened_documents)} documents from {len(nested_documents)} sources")
        
            # Validate all documents
            valid_documents = []
            for i, doc in enumerate(flattened_documents):
                if not isinstance(doc, Document):
                    logger.error(f"Extraction returned non-Document at index {i}: {type(doc)}")
                    continue
                
                if not isinstance(doc.page_content, str):
                    logger.error(f"Document at index {i} has non-string content: {type(doc.page_content)}")
                    # Try to fix if it's a nested Document
                    if isinstance(doc.page_content, Document):
                        nested_doc = doc.page_content
                        logger.warning(f"Fixing nested Document detected at index {i}")
                        # Create new document with nested content and combined metadata
                        combined_metadata = {**doc.metadata, **nested_doc.metadata}
                        valid_documents.append(Document(
                            page_content=nested_doc.page_content,
                            metadata=combined_metadata
                        ))
                    continue
                
                valid_documents.append(doc)
            
            return valid_documents
        except Exception as e:
            logger.error(f"Error extracting documents: {str(e)}", exc_info=True)
            self.stats["errors"].append(f"Document extraction error: {str(e)}")
            return []
    
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        if not documents:
            return []
        
        # Fix any nested Document objects before chunking
        fixed_documents = []
        for i, doc in enumerate(documents):
            try:
                if not isinstance(doc, Document):
                    logger.error(f"Non-Document object found at position {i}: {type(doc)}")
                    continue
                    
                if not isinstance(doc.page_content, str):
                    logger.error(f"Document at position {i} has non-string page_content: {type(doc.page_content)}")
                    # Fix nested Document
                    if isinstance(doc.page_content, Document):
                        nested_doc = doc.page_content
                        logger.warning(f"Fixing nested Document detected at position {i}")
                        # Create new document with nested content and combined metadata
                        combined_metadata = {**doc.metadata, **nested_doc.metadata}
                        fixed_documents.append(Document(
                            page_content=nested_doc.page_content,
                            metadata=combined_metadata
                        ))
                    continue
                
                # If we got here, document seems OK
                fixed_documents.append(doc)
            except Exception as e:
                logger.error(f"Error examining document at position {i}: {e}")
        
        # If we fixed any documents, log it
        if len(fixed_documents) != len(documents):
            logger.info(f"Fixed {len(documents) - len(fixed_documents)} problematic documents, proceeding with {len(fixed_documents)}")
        
        try:
            # Check what text_splitter we're using
            logger.info(f"Using text splitter: {self.text_splitter.__class__.__name__}")
            
            # Use your chunkers factory split_documents function with fixed documents
            chunked_docs = split_documents(
                fixed_documents,
                text_splitter=self.text_splitter,
                **(self.options.chunking_options or {})
            )
            return chunked_docs
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}", exc_info=True)
            self.stats["errors"].append(f"Document chunking error: {str(e)}")
            return []
    
    async def _process_chunks_in_batches(self, chunks: List[Document]) -> None:
        """Process chunks in batches for better memory management."""
        total_chunks = len(chunks)
        batch_size = self.options.batch_size
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
            
            # Process this batch
            await self._process_chunk_batch(batch)
    
    async def _process_chunk_batch(self, batch: List[Document]) -> None:
        """Process a single batch of chunks."""
        print(f"Processing batch of {len(batch)} chunks:\n{batch}\n")
        # Step 1: Add to vector store if enabled
        if self.vector_store and not self.options.skip_vector_store:
            try:
                self.vector_store.add_documents(batch)
                self.stats["vector_entries_added"] += len(batch)
                logger.info(f"Added {len(batch)} chunks to vector store")
            except Exception as e:
                logger.error(f"Error adding to vector store: {str(e)}")
                self.stats["errors"].append(f"Vector store error: {str(e)}")
        
        # Step 2: Extract entities and update graph store if enabled
        if self.entity_extractor and not self.options.skip_graph_store:
            try:
                # Use the LLM from options if available
                llm_to_use = self.options.llm if hasattr(self.options, 'llm') and self.options.llm is not None else None
                
                # Extract entities and relationships - pass the actual batch of documents
                # The issue was here - you were passing the batch name instead of the batch itself
                graph = await self.entity_extractor.extract(
                    documents=batch,  # Explicitly name the parameter to be clear
                    llm=llm_to_use,
                    #chunking_strategy="none"  # Explicitly prevent double chunking
                )
                
                # Update stats
                self.stats["entities_extracted"] += len(graph.nodes)
                self.stats["relationships_extracted"] += len(graph.edges)
                logger.info(f"Extracted {len(graph.nodes)} entities and {len(graph.edges)} relationships")
                
                # Add to graph store
                if self.graph_store:
                    self.graph_store.add_graph(graph)
                    self.stats["graph_entries_added"] += len(graph.nodes) + len(graph.edges)
                    logger.info(f"Added graph data to graph store")
            except Exception as e:
                logger.error(f"Error in entity extraction or graph storage: {str(e)}", exc_info=True)
                self.stats["errors"].append(f"Entity extraction error: {str(e)}")

# Simple function to create and use an ingestion pipeline
async def ingest_documents(
    sources: List[Union[str, Path, Document]], 
    config: Optional[Dict[str, Any]] = None,
    options: Optional[IngestionOptions] = None
) -> Dict[str, Any]:
    """
    Ingest documents through the pipeline.
    
    Args:
        sources: List of source identifiers or Document objects
        config: Optional configuration dictionary
        options: Optional ingestion options
        
    Returns:
        Dictionary containing ingestion statistics
    """
    # Create config manager if config provided
    config_manager = ConfigManager(config) if config else None
    
    # Create pipeline
    pipeline = IngestionPipeline(
        config_manager=config_manager,
        options=options or IngestionOptions()
    )
    
    # Run ingestion
    return await pipeline.ingest(sources)