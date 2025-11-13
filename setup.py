from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent

install_requires = [
    # Conservative base install requirements (match `requirements.txt`)
    "langchain==1.0.5",
    "langchain-openai==1.0.2",
    "langchain-huggingface==1.0.1",
    "langchain-community==0.4.1",
    "langchain-chroma==1.0.0",
    "chromadb==1.3.4",
    "langchain-core==1.0.4",
    "langsmith==0.4.42",
    "langchain-google-community==3.0.0",
    "langchain-text-splitters==1.0.0",
    "langchain-markitdown==0.1.6",
    "openai>=1.40.0",
    "python-dotenv==1.0.1",
    "retry==0.9.2",
    "flask==3.0.0",
    "flask-cors==4.0.0",
]

setup(
    name="python-ragdoll",
    version="2.0.1",
    description="A set of helper classes that abstract some of the more common tasks of a typical RAG process including document loading/web scraping.",
    author="Nathan Sasto",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest==9.0.0",
            "coverage",
            "black",
            "isort",
            "flake8",
        ],
        "entity": [
            "spacy>=3.7.0",
            "spacy-transformers",
            "sentence_transformers>=2.2.2",
            "PyMuPDF>=1.25.5",
        ],
        "graph": [
            "neo4j>=5.11.0",
            "rdflib",
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: Markup :: Markdown',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    long_description=(this_directory / "README.md").read_text(encoding="utf-8"),
    long_description_content_type='text/markdown',
    license='MIT',
)
