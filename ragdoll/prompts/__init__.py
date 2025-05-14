"""Prompt management for RAGdoll."""
import os
from pathlib import Path
from typing import Dict, Optional, List

PROMPTS_DIR = Path(__file__).parent

def get_prompt(prompt_name: str) -> str:
    """
    Load a prompt from a file in the prompts directory.
    
    Args:
        prompt_name: The name of the prompt file (without .md extension)
        
    Returns:
        The prompt text
    """
    prompt_path = PROMPTS_DIR / f"{prompt_name}.md"
    if not prompt_path.exists():
        raise ValueError(f"Prompt '{prompt_name}' does not exist at {prompt_path}")
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def list_prompts() -> List[str]:
    """
    List all available prompts.
    
    Returns:
        List of prompt names (without .md extension)
    """
    return [f.stem for f in PROMPTS_DIR.glob("*.md")]