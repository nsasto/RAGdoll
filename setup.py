from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="RAGdoll",
    version="2.0.1",
    description="A set of helper classes that abstract some of the more common tasks of a typical RAG process including document loading/web scraping.",
    author="Nathan Sasto",
    packages=find_packages(),
    install_requires=requirements,
)
