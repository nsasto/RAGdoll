import requests
from bs4 import BeautifulSoup

from ragdoll.loaders.base_loader import BaseLoader
from langchain.docstore.document import Document

class WebLoader(BaseLoader):    
    def load(self, link: str) -> list[Document]:
        """
        Loads the content from a web URL.
        :param link: The url to load from.
        :return: list[Document] The web page content as a list of Document.
        """
        try:
            session = requests.Session()
            response = session.get(link, timeout=4)
            soup = BeautifulSoup(response.content, "lxml", from_encoding=response.encoding)

            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()

            raw_content = self.get_content_from_url(soup)
            lines = (line.strip() for line in raw_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = "\n".join(chunk for chunk in chunks if chunk)

            metadata = self._build_metadata(soup, link)
            return [Document(page_content=content, metadata=metadata)]
        except requests.exceptions.Timeout:
            print(f"Timeout error occurred for link: {link}")
            return []
        except Exception as e:
            print(f"Error occurred for link: {link}, {e}")
            return []

    def get_content_from_url(self, soup):
        text = ""
        tags = ["p", "h1", "h2", "h3", "h4", "h5", "div"]
        for element in soup.find_all(tags):
            text += element.text + "\n"
        return text

    def _build_metadata(self, soup, url) -> dict:
        metadata = {"source": url}
        if title := soup.find("title"):
            metadata["title"] = title.get_text()
        if description := soup.find("meta", attrs={"name": "description"}):
            metadata["description"] = description.get("content", "No description found.")
        if html := soup.find("html"):
            metadata["language"] = html.get("lang", "No language found.")
        return metadata
