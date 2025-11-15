from __future__ import annotations

import os
from typing import Any, Callable, Optional


def resolve_env_reference(
    value: Any,
    *,
    label: Optional[str] = None,
    warn: Optional[Callable[[str], None]] = None,
) -> Any:
    """
    Interpret configuration values that reference environment variables.

    Supports both ``os.environ/VAR`` and legacy ``#VAR`` syntaxes. Returns the
    resolved environment value when available, otherwise the original value.
    """

    if not isinstance(value, str):
        return value

    env_var: Optional[str] = None
    if value.startswith("os.environ/"):
        env_var = value.split("/", 1)[1]
    elif value.startswith("#"):
        env_var = value[1:]

    if not env_var:
        return value

    resolved = os.environ.get(env_var)
    if resolved is None and warn:
        context = f" for {label}" if label else ""
        warn(f"Environment variable '{env_var}' not found{context}")

    return resolved
