from typing import Iterable
from langchain_openai import ChatOpenAI
import logging
from datetime import datetime
from langchain_community.utilities import GoogleSearchAPIWrapper
from colored import Fore, Style
from concurrent.futures.thread import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_openai import OpenAIEmbeddings


##peer load
from .scraper import Scraper
from .helpers import remove_set_duplicates
from .config import ragdoll_config


##peer load
from ragdoll.scraper import Scraper
from ragdoll.helpers import remove_set_duplicates
from ragdoll.config import ragdoll_config

class RagdollIndex:
    def __init__(self, config = {}):
        """
        Initializes a RagdollIndex object.

        Args:
            config (dict): Configuration options for the RagdollIndex. Default is an empty dictionary.
        """
        self.set_config(config)

        self.raw_documents = []
        self.document_chunks = []
        self.summaries = []
        self.search_terms = []
        self.search_results = []
        self.url_list = []
        self.retriever = None

    def set_config(self, config={}):
        """
        Set the configuration for the Ragdoll object.

        Parameters:
        - config (dict): A dictionary containing the configuration options.

        Returns:
        - None
        """

        cfg = ragdoll_config.copy()  # Copy the default config
        cfg.update(config)  # Merge user config into default config

        self.alternative_query_term_count = cfg.get('alternative_query_term_count')
        self.max_search_results_per_query = cfg.get('max_search_results_per_query')
        self.max_workers = cfg.get('max_workers')
        
        enable_logging = cfg.get('enable_logging', False)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if enable_logging else logging.INFO)
        
        if cfg.get('vector_store').lower() == "faiss":
            from langchain_community.vectorstores import FAISS

        if cfg.get('embeddings').lower() == "openaiembeddings":
            self.embeddings = OpenAIEmbeddings()

        self.text_splitter = self.get_text_splitter()

        self.cfg = cfg

    def get_config(self):
        return self.cfg

    def _get_sub_queries(self, query, query_count)->list:
        """Gets the sub-queries for the given question to be passed to the search engine.

        Args:
            question (str): The question to generate sub-queries for.
            max_iterations (int): The maximum number of sub-queries to generate.

        Returns:
            list: A list of sub-queries generated from the given question.
        """
        import ast

        prompt =(
            f'Write exactly {query_count} unique google search queries to search online that form an objective opinion from the following: "{query}"'
            f'Use the current date if needed: {datetime.now().strftime("%B %d, %Y")}.\n'
            f'You must respond with a list of strings in the following format: ["query 1", "query 2", "query 3", etc].'
        )
        self.logger.debug(f'👨‍💻 Generating potential search queries with prompt:\n {query}')
        # define the LLM
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, max_tokens=2056)
        result = llm.invoke(prompt)
        self.logger.info(f'👨‍💻 Generated potential search queries: {result.content}')
        return ast.literal_eval(result.content)
    

    def _google_search(self, query, n)->list:
        """Performs a Google search with the given query.

        Args:
            query (str): The search query.
            n (int): The number of search results to retrieve.

        Returns:
            list: A list of search results.
        """
        self.logger.debug(f"  🌐 Searching with query {query}...")

        googleSearch = GoogleSearchAPIWrapper()
        results = googleSearch.results(query, n)

        if results is None:
            return
        search_results = []
        
        for result in results:
            # skip youtube results
            if "youtube.com" in result["link"]:
                continue
            search_result = {
                "title": result["title"],
                "href": result["link"],
                "snippet": result["snippet"],
            }
            search_results.append(search_result)

        return search_results

    def get_scraped_content(self, urls=None):
        """Get site content for a given list of URLs or file path (if pdf).

        Args:
            urls (str): The URL for which to retrieve the site content.

        Returns:
            str: a list of langchain documents.
        """
        self.logger.debug('Fetching content URLs')
        urls = self.url_list if urls is None else urls
        documents = []
        try:
            documents = Scraper(urls).run()
        except Exception as e:
            self.logger.error(f"{Fore.RED}Error in get_scraped_content: {e}{Style.reset}")
        
        self.raw_documents = [documents[i]["raw_content"] for i in range(len(documents))]
        return self.raw_documents
    
    def get_suggested_search_terms(self, query: str):
        """Get appropriate web search terms for a query.

        Args:
            query (str): The query for which to retrieve suggested search terms.

        Returns:
            list: A list of suggested search terms.
        """
        self.logger.debug('Fetching suggested search terms for the query')
        self.search_terms = self._get_sub_queries(query, self.alternative_query_term_count)
        return self.search_terms

    def get_search_results(self, query_list):
        """
        Performs Google searches for each query in the query_list in parallel.

        Args:
            query_list: A list of search queries.
            n_results: The number of search results to retrieve for each query (default: 3).

        Returns:
            A list of dictionaries where each dictionary contains the search results for a
            specific query. The dictionary has the following keys:
            - query: The original search query.
            - results: A list of search results in the same format as returned by
                        `_google_search`.
        """
        n_results = self.max_search_results_per_query

        if isinstance(query_list, list):
            pass  # No processing needed for list
        elif isinstance(query_list, str):
            query_list = [query_list]  
        else:
            raise TypeError("Query must be a string or a list.")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._google_search, query, n_results) for query in query_list]

        # Wait for all tasks to finish and collect the results
        results = []

        for future, query in zip(futures, query_list):
            try:
                search_results = future.result()
                for item in search_results:
                    item['query'] = query
                results.extend(search_results)
            except Exception as e:
                print(f"Error processing query '{query}': {e}")

        urls = remove_set_duplicates(results, key='href')        
        self.search_results = urls
        self.url_list = [d['href'] for i, d in enumerate(urls)]

        return list(urls)


    def get_doc_summary(self, document: str):
        """Summarize a document.

        Args:
            document (str): The document to summarize.

        Returns:
            str: The summarized document.
        """
        self.logger.debug('Summarizing document')
        pass

    
    def get_text_splitter(self, chunk_size=1000, chunk_overlap=200, length_function=len, is_separator_regex=False):
        """
        Returns a RecursiveCharacterTextSplitter object with the specified parameters.

        Parameters:
        - chunk_size (int): The size of each text chunk.
        - chunk_overlap (int): The overlap between consecutive text chunks.
        - length_function (function): A function to calculate the length of the text.
        - is_separator_regex (bool): Whether the separator is a regular expression.

        Returns:
        - RecursiveCharacterTextSplitter: The text splitter object.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            is_separator_regex=is_separator_regex,
        )

        return self.text_splitter

    
    def get_split_documents(self, documents = None):
        """
        Splits the given documents into chunks using the text splitter.

        Args:
            documents (Iterable[Document]): The documents to be split into chunks.

        Returns:
            None
        """
        documents = self.raw_documents if documents is None else documents

        if self.text_splitter is None:
            self.get_text_splitter()
        self.logger.debug('Chunking document')
        
        self.document_chunks = self.text_splitter.split_documents(documents)
        return self.document_chunks

    def get_retriever(self, documents):
        """
        Returns a retriever object based on the specified vector store.

        Args:
            documents (list): List of documents to be used for creating the retriever.

        Returns:
            retriever: The retriever object based on the specified vector store.

        Raises:
            TypeError: If the vector store is not specified in the config dictionary.
        """
        if (self.cfg.get('vector_store').lower() == "faiss"):
            from langchain_community.vectorstores import FAISS  
            retriever = FAISS.from_documents(documents, OpenAIEmbeddings()).as_retriever()
        else:
            raise TypeError("Vector store not specified. Set this in the config dictionary")

        self.retriever = retriever
        return retriever
    
    def run_index_pipeline(self, query: str):
        """Run the entire process, taking a query as input.

        Args:
            query (str): The query to run the index pipeline on.
        """
        self.logger.debug('Running index pipeline')
        #get appropriate search queries for the question 
        search_queries = self.get_suggested_search_terms(query)
        #get google search results
        results=self.get_search_results(search_queries)
        #scrape the returned sites and return documents. 
        # results contains a little more metadata, the list of urls can be accessed via index.url_list which is used by default in the next call
        documents = self.get_scraped_content()
        #split docs
        split_docs = self.get_split_documents(documents)
        #create an in memory retriever
        retriever = self.get_retriever(split_docs)
        return retriever



if (__name__=='main'):
    print('RAGdoll Index is running...')