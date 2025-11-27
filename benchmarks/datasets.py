"""
Dataset loading and parsing for RAGdoll benchmarks.

Supports 2wikimultihopqa dataset with configurable subset sizes.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import xxhash


@dataclass
class Query:
    """A benchmark query with ground truth evidence."""

    question: str
    answer: str
    evidence: List[Tuple[str, int]]  # [(title, sentence_id), ...]
    query_type: str = "unknown"  # dataset-provided type label

    def __post_init__(self):
        """Convert evidence to list of tuples if needed."""
        if self.evidence and isinstance(self.evidence[0], list):
            self.evidence = [tuple(e) for e in self.evidence]
        # Normalize query type casing for downstream checks
        if self.query_type:
            self.query_type = self.query_type.lower()
        else:
            self.query_type = "unknown"

    @property
    def is_multihop(self) -> bool:
        """Check if this is a multi-hop query."""
        multihop_types = {
            "comparison",
            "bridge",
            "bridge_comparison",
            "compositional",
            "inference",
        }
        return self.query_type in multihop_types


def load_dataset(
    dataset_name: str, datasets_dir: Path, subset: int = 0
) -> List[Dict[str, Any]]:
    """
    Load a dataset from JSON file.

    Args:
        dataset_name: Name of dataset ("2wikimultihopqa")
        datasets_dir: Directory containing dataset files
        subset: Number of queries to use (0 = all)

    Returns:
        List of dataset entries

    Raises:
        FileNotFoundError: If dataset file doesn't exist
    """
    dataset_path = datasets_dir / f"{dataset_name}.json"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Please download from: https://github.com/circlemind-ai/fast-graphrag/tree/main/benchmarks/datasets"
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if subset > 0:
        return dataset[:subset]
    return dataset


def get_corpus(
    dataset: List[Dict[str, Any]], dataset_name: str
) -> Dict[int, Tuple[str, str]]:
    """
    Extract corpus (passages) from dataset.

    Args:
        dataset: Loaded dataset
        dataset_name: Name of dataset

    Returns:
        Dict mapping passage hash to (title, text)
    """
    if dataset_name not in ["2wikimultihopqa", "hotpotqa"]:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")

    passages: Dict[int, Tuple[str, str]] = {}

    for datapoint in dataset:
        context = datapoint["context"]

        for passage in context:
            title, sentences = passage
            title = title.encode("utf-8").decode()
            text = "\n".join(sentences).encode("utf-8").decode()

            # Hash text for unique ID
            hash_val = xxhash.xxh3_64_intdigest(text)

            if hash_val not in passages:
                passages[hash_val] = (title, text)

    return passages


def get_queries(dataset: List[Dict[str, Any]]) -> List[Query]:
    """
    Extract queries with ground truth from dataset.

    Args:
        dataset: Loaded dataset

    Returns:
        List of Query objects
    """
    queries: List[Query] = []

    for datapoint in dataset:
        queries.append(
            Query(
                question=datapoint["question"].encode("utf-8").decode(),
                answer=datapoint["answer"],
                evidence=list(datapoint["supporting_facts"]),
                query_type=datapoint.get("type", "unknown"),
            )
        )

    return queries


def get_multihop_questions(
    dataset_name: str, subset: int, datasets_dir: Path
) -> List[str]:
    """
    Load list of multi-hop questions if available.

    Args:
        dataset_name: Name of dataset
        subset: Subset size
        datasets_dir: Directory containing dataset files

    Returns:
        List of multi-hop question strings
    """
    questions_path = datasets_dir / "questions" / f"{dataset_name}_{subset}.json"

    if not questions_path.exists():
        return []

    with open(questions_path, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_costs(
    num_passages: int,
    num_queries: int,
    extract_entities: bool = False,
    avg_passage_tokens: int = 150,
    avg_entity_calls_per_passage: int = 1,
) -> Dict[str, float]:
    """
    Estimate API costs for benchmark run.

    Args:
        num_passages: Number of passages to ingest
        num_queries: Number of queries to run
        extract_entities: Whether entity extraction is enabled
        avg_passage_tokens: Average tokens per passage
        avg_entity_calls_per_passage: Average LLM calls for entity extraction

    Returns:
        Dict with cost estimates
    """
    # OpenAI pricing (as of 2024)
    EMBEDDING_COST_PER_1K = 0.00013  # text-embedding-3-small
    GPT4O_MINI_INPUT_COST_PER_1M = 0.150  # gpt-4o-mini input
    GPT4O_MINI_OUTPUT_COST_PER_1M = 0.600  # gpt-4o-mini output

    # Embedding costs
    embedding_tokens = (num_passages + num_queries) * avg_passage_tokens
    embedding_cost = (embedding_tokens / 1000) * EMBEDDING_COST_PER_1K

    # Entity extraction costs (if enabled)
    entity_cost = 0.0
    if extract_entities:
        llm_calls = num_passages * avg_entity_calls_per_passage
        input_tokens = llm_calls * avg_passage_tokens
        output_tokens = llm_calls * 100  # Estimated entity output

        entity_cost = (input_tokens / 1_000_000) * GPT4O_MINI_INPUT_COST_PER_1M + (
            output_tokens / 1_000_000
        ) * GPT4O_MINI_OUTPUT_COST_PER_1M

    return {
        "embedding_cost": embedding_cost,
        "entity_extraction_cost": entity_cost,
        "total_cost": embedding_cost + entity_cost,
        "embedding_tokens": embedding_tokens,
        "llm_calls": (
            num_passages * avg_entity_calls_per_passage if extract_entities else 0
        ),
    }
