from concurrent.futures.thread import ThreadPoolExecutor
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import ArxivRetriever
from functools import partial
import requests
from bs4 import BeautifulSoup

from langchain.docstore.document import Document

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

class Scraper:
    """
    Scraper class to extract the content from the links
    """
  
    def __init__(self, urls, user_agent=DEFAULT_USER_AGENT):
        """
        Initialize the Scraper class.
        Args:
            urls:
        """
        self.urls = urls
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    
    def _build_metadata(self, soup, url) -> dict:
        """Build metadata from BeautifulSoup output."""
        metadata = {"source": url}
        if title := soup.find("title"):
            metadata["title"] = title.get_text()
        if description := soup.find("meta", attrs={"name": "description"}):
            metadata["description"] = description.get("content", "No description found.")
        if html := soup.find("html"):
            metadata["language"] = html.get("lang", "No language found.")
        return metadata
    
    def run(self):
        """
        Extracts the content from the links
        """
        partial_extract = partial(self.extract_data_from_link, session=self.session)
        with ThreadPoolExecutor(max_workers=20) as executor:
            contents = executor.map(partial_extract, self.urls)
        res = [content for content in contents if content["raw_content"] is not None]
        return res

    def extract_data_from_link(self, link, session):
        """
        Extracts the data from the given link.

        Args:
            link (str): The link from which to extract the data.
            session: The session object for making HTTP requests.

        Returns:
            dict: A dictionary containing the extracted data. The dictionary has the following keys:
                - "url" (str): The original link.
                - "raw_content" (Document or None): The extracted content as a Document object - single page combined for all content ready for post splitting, 
                                                    or None if an error occurred or the content is too short.
        """
        content = ""
        try:
            if link.endswith(".pdf"):
                content, metadata = self.scrape_pdf_with_pymupdf(link)
            elif "arxiv.org" in link:
                doc_num = link.split("/")[-1]
                content, metadata = self.scrape_pdf_with_arxiv(doc_num)
            elif link:
                content, metadata = self.scrape_text_with_bs(link, session)

            if len(content) < 100:
                return {"url": link, "raw_content": None}
            raw_content = Document(page_content=content, metadata=metadata)
            return {"url": link, "raw_content": raw_content}
        except Exception as e:
            print(f"error occurred: {e} ")
            return {"url": link, "raw_content": None}

    def scrape_text_with_bs(self, link, session):
        response = session.get(link, timeout=4)
        soup = BeautifulSoup(response.content, "lxml", from_encoding=response.encoding)

        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        raw_content = self.get_content_from_url(soup)
        lines = (line.strip() for line in raw_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        content = "\n".join(chunk for chunk in chunks if chunk)
        metadata = self._build_metadata(soup, link)
        return content, metadata
    
    def scrape_pdf_with_pymupdf(self, url) -> str:
        """Scrape a pdf with pymupdf

        Args:
            url (str): The url of the pdf to scrape

        Returns:
            str: The text scraped from the pdf
        """
        loader = PyMuPDFLoader(url)
        doc = loader.load()
        return str(doc), doc[0].metadata

    def scrape_pdf_with_arxiv(self, query) -> str:
        """Scrape a pdf with arxiv
        default document length of 70000 about ~15 pages or None for no limit

        Args:
            query (str): The query to search for

        Returns:
            str: The text scraped from the pdf
        """
        retriever = ArxivRetriever(load_max_docs=2, doc_content_chars_max=None)
        docs = retriever.get_relevant_documents(query=query)
        return docs[0].page_content, docs[0].metadata

    def get_content_from_url(self, soup):
        """Get the text from the soup

        Args:
            soup (BeautifulSoup): The soup to get the text from

        Returns:
            str: The text from the soup
        """
        text = ""
        tags = ["p", "h1", "h2", "h3", "h4", "h5"]
        for element in soup.find_all(tags):  # Find all the <p> elements
            text += element.text + "\n"
        return text


if __name__ == "__main__":
    print('.... starting scraper ....')
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
    print('  testing on prettier.io\n')
    urls = ["https://prettier.io/docs/en/webstorm"]
    try:
        html = Scraper(urls, user_agent).run()
    except Exception as e:
        print(f"Error in scrape_urls: {e}")
    print(html)

    print('  testing on pdf\n')
    urls = ['https://www.uou.ac.in/sites/default/files/slm/BHM-503T.pdf']
    try:
        pdf = Scraper(urls, user_agent).run()
    except Exception as e:
        print(f"Error in scrape_urls: {e}")
    print(pdf)

    print(' testing on arxiv\n')
    urls=['https://arxiv.org/pdf/2401.14418.pdf']
    try:
        arxiv = Scraper(urls, user_agent).run()
    except Exception as e:
        print(f"Error in scrape_urls: {e}")
    print(arxiv)