"""
RAGdoll package exports.

Expose the high-level orchestration class so consumers can rely on a stable
import path without triggering side effects.
"""

from .ragdoll import Ragdoll

__all__ = ["Ragdoll"]
