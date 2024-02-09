![Ragdoll](img/github-header-image.png)

# 🧭 Project Overview 

This project provides a set of helper classes that abstract some of the more common tasks of a typical RAG process including document loading/web scraping.  

It's based on local vector storage but can easily be extended to Pinecone using langchain. 

the default LLM and embedding model is OpenAI but there are also options to run a fully local LLM.

## 🚧 Prerequisites

* OpenAI API Key - For more information on how to create an OpenAI API key, visit the [OpenAI Platform Website](https://platform.openai.com/)
* Google API Keys - To set it up, create the GOOGLE_API_KEY in the Google Cloud credential console (https://console.cloud.google.com/apis/credentials) and a GOOGLE_CSE_ID using the Programmable Search Engine (https://programmablesearchengine.google.com/controlpanel/create). 

## 🎛 Project Setup

To set up the project on your local machine, follow these steps:

2. Clone the repository to your local machine.
3. Install the required dependencies using `pip install -r requirements.txt`.

## 📦 Project Structure

The project is structured as follows:
    
```
├── ragdoll_example.ipynb           # demo notebook.
├── ragdoll/                        # ragdoll files
├── README.md                       # This file.
├── requirements.txt                # List of dependencies.
└── img/                            # banner image above
```

## 🗄️ Data

The vector data used in this project is stored locally which is used to generate responses in the LLM Chat using a Retrieval Augmentation process. Be aware that if you are using OpenAI as your embeddings engine, that data will be sent to OpenAI. 

## Getting Started

Assumes you have the appropriate API keys for Google search and OpenAI in your environment variables or .env file. To load

```python
from dotenv import load_dotenv
load_dotenv(override=True)
```

#### 1. Create an Index from web content

```python
from ragdoll.index import RagdollIndex
index= RagdollIndex({'enable_logging':True})

question = "tell me more about langchain"
#get appropriate search queries for the question 
search_queries = index.get_suggested_search_terms(question)
#get google search results
results=index.get_search_results(search_queries)
#scrape the returned sites and return documents. 
# results contains a little more metadata, the list of urls can be accessed via index.url_list which is used by default in the next call
documents = index.get_scraped_content()
#split docs
split_docs = index.get_split_documents(documents)
#create an in memory retriever
retriever = index.get_retriever(split_docs)

```

Or, in one line as follows:

```python
retriever = index.run_index_pipeline(question)
```

#### 2. Retrieval 

And that's pretty much it. here's a quick test of retrieval:

```python
docs = retriever.get_relevant_documents('how does langchain work')

from ragdoll.helpers import pretty_print_docs
print("-" * 100)
print(f"The retriever had found {len(docs)} relevant documents")
print("-" * 100, "\n\n")
print(pretty_print_docs(docs, for_llm=False))
```


## 📚 References

The following resources were used in the development of this project:

- Langchain: https://www.langchain.com/
- FAISS: https://github.com/facebookresearch/faiss

## 🤝 Contributions

This project is a work in progress and there's plenty room for improvement - contributions are always welcome! If you have any ideas or suggestions, feel free to open an issue or submit a pull request.

## 🛡️ Disclaimer

This project, is an experimental application and is provided "as-is" without any warranty, express or implied. Code is shared for educational purposes under the MIT license.
