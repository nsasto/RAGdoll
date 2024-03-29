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

alternatively, install with pip:

`pip install git+https://github.com/nsasto/RAGdoll.git`

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
The super rapid version. 5 lines to build research and response generation:

```python
from ragdoll.index import RagdollIndex
from ragdoll.retriever import RagdollRetriever

index= RagdollIndex()
ragdoll = RagdollRetriever()

#ok, let's go
question = "tell me more about langchain"
split_docs = index.run_index_pipeline(question)
retriever = ragdoll.get_compression_retriever(retriever)
response = ragdoll.answer_me_this(question, cc_retriever)
print(response)
```

generates the following structured response (snippet included here only) :

```
LangChain is an artificial intelligence framework designed for programmers to develop applications using large language models. It offers several key features that make it versatile and useful for developers.

One of the main features of LangChain is its context-awareness capability. It allows applications to establish connections between a language model and various context sources. This means that developers can create applications that are aware of the context in which they are being used, making them more intelligent and responsive....
```

#### 1. Create an Index from web content

```python
from ragdoll.index import RagdollIndex
index= RagdollIndex()

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

```

Or, in one line as follows:

```python
split_docs = index.run_index_pipeline(question)
```

#### 2. Retrieval 

And that's pretty much it to load up our documents. To retrieve them using a langchain retriever is just as simple. 

```python
from ragdoll.retriever import RagdollRetriever

ragdoll = RagdollRetriever()
retriever = ragdoll.get_retriever(documents=split_docs) 
docs = retriever.get_relevant_documents('how does langchain work')

from ragdoll.helpers import pretty_print_docs
print("-" * 100)
print(f"The retriever had found {len(docs)} relevant documents")
print("-" * 100, "\n\n")
print(pretty_print_docs(docs, for_llm=False))
```
To use multi-query retrieval, use `get_mq_retriever`. Note that multi query will incur additional calls to your LLM. 
The Ragdoll MultiQuery class is a custom langchain retriever to resolve the native langchain bug as at version '0.1.6'. 

```python
retriever = ragdoll.get_mq_retriever(documents=split_docs) 
```

To use the Contextual Compression Retriever, you’ll need a base retriever (either the standard or multi query) - and then select the pipeline options which are all set to True by default but can be amended in the config params. The Contextual Compressor by default this refinement process:
embeddings_filter > splitter > redundant_filter > relevance_filter 

```python
cc_retriever = ragdoll.get_compression_retriever(retriever)
```

#### 3. Q&A 

Basic Q&A is pretty straight forward. Simply pass your question to the `answer_me_this` method:

```python
response = ragdoll.answer_me_this(question, cc_retriever)
print(response)
```

## 📚 References

The following resources were used in the development of this project:

- Langchain: https://www.langchain.com/
- FAISS: https://github.com/facebookresearch/faiss

## 🤝 Contributions

This project is a work in progress and there's plenty room for improvement - contributions are always welcome! If you have any ideas or suggestions, feel free to open an issue or submit a pull request.

## 🛡️ Disclaimer

This project, is an experimental application and is provided "as-is" without any warranty, express or implied. Code is shared for educational purposes under the MIT license.
