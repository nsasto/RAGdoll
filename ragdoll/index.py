from typing import Iterable
from langchain_openai import ChatOpenAI
import logging
from datetime import datetime
from langchain_google_community import GoogleSearchAPIWrapper
from colored import Fore, Style
from concurrent.futures.thread import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

#from langchain_openai import OpenAIEmbeddings

from .scraper import Scraper
from .helpers import remove_set_duplicates
from .config import Config
from .prompts import generate_search_queries_prompt
from .models import RagdollLLM

class RagdollIndex:
    def __init__(self, config = {}):
        """
        Initializes a RagdollIndex object.

        Args:
            config (dict): Configuration options for the RagdollIndex. Default is an empty dictionary.
        """
        self.cfg = Config(config)

        self.raw_documents = []
        self.document_chunks = []
        self.summaries = []
        self.search_terms = []
        self.search_results = []
        self.url_list = []

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.cfg.log_level)

        #initialize text splitter
        self.get_text_splitter()

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

        prompt = generate_search_queries_prompt(query, query_count)

        self.logger.info(f'🧠 Generating potential search queries with prompt:\n {query}')
        # define the LLM
        llm = RagdollLLM(self.cfg).llm

        result = llm.invoke(prompt)
        values = result.content if hasattr(result, 'content') else result
        self.logger.info(f'🧠 Generated potential search queries: {values}')
        return ast.literal_eval(values)
    

    def _google_search(self, query, n)->list:
        """Performs a Google search with the given query.

        Args:
            query (str): The search query.
            n (int): The number of search results to retrieve.

        Returns:
            list: A list of search results.
        """
        self.logger.info(f"  🌐 Searching with query {query}...")

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
        self.logger.info('🌐 Fetching raw source content')
        urls = self.url_list if urls is None else urls
        documents = []
        try:
            documents = Scraper(urls, user_agent=self.cfg.user_agent).run()
        except Exception as e:
            self.logger.error(f"{Fore.RED}Error in get_scraped_content: {e}{Style.reset}")
        
        self.raw_documents = [documents[i]["raw_content"] for i in range(len(documents))]
        return self.raw_documents
    
    def get_suggested_search_terms(self, query: str, n=None):
        """Get appropriate web search terms for a query.

        Args:
            query (str): The query for which to retrieve suggested search terms.
            n (int): number of alternative queries to return
        Returns:
            list: A list of suggested search terms.
        """
        n = self.cfg.alternative_query_term_count if n is None else n
        self.logger.info('Fetching suggested search terms for the query')
        self.search_terms = self._get_sub_queries(query, n)
        return self.search_terms

    def get_search_results(self, query_list, n_results=None):
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
        n_results = self.cfg.max_search_results_per_query if n_results is None else n_results

        if isinstance(query_list, list):
            pass  # No processing needed for list
        elif isinstance(query_list, str):
            query_list = [query_list]  
        else:
            raise TypeError("Query must be a string or a list.")

        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
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
        self.logger.info('Summarizing document')
        pass

    
    def get_text_splitter(self, chunk_size=1000, chunk_overlap=200, length_function=len, is_separator_regex=False, add_start_index=True ):
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
            add_start_index=add_start_index 

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
        self.logger.info('📰 Chunking document')
        
        self.document_chunks = self.text_splitter.split_documents(documents)
        return self.document_chunks
    
    def run_index_pipeline(self, query: str, **kwargs):
        """Run the entire process, taking a query as input.

        Args:
            query (str): The query to run the index pipeline on.
        """
        self.logger.info('Running index pipeline')
        #get appropriate search queries for the question 
        search_queries = self.get_suggested_search_terms(query)
        #get google search results
        results=self.get_search_results(search_queries)
        #scrape the returned sites and return documents. 
        # results contains a little more metadata, the list of urls can be accessed via index.url_list which is used by default in the next call
        documents = self.get_scraped_content()
        #split docs
        split_docs = self.get_split_documents(documents)

        return split_docs

    def run_document_pipeline(self, document_list: list, **kwargs):
        """Run the entire process, taking a list of pdf as input.

        Args:
            document_list (list): The list of pdfs to run the index pipeline on.
        """
        self.logger.info('Running document index pipeline')

        # scrape the pdf and return langchain documents. 
        # results contains a little more metadata, the list of urls can be accessed via index.url_list which is used by default in the next call
        documents = self.get_scraped_content(document_list)
        #split docs
        split_docs = self.get_split_documents(documents)

        return split_docs


if (__name__=='main'):
    print('RAGdoll Index...')