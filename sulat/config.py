#!/usr/bin/env python

"""Configuration helpers for local caching.

This module ensures the library uses the SURUS_CACHE environment variable to
store local data. If SURUS_CACHE is not set by the user, it is automatically
initialized to ~/.cache/surus and the directory is created.
"""

from __future__ import annotations

import os
from typing import Optional


def _default_cache_dir() -> str:
    """Return the default cache directory path resolved at runtime.

    Resolving at runtime ensures changes to environment variables such as HOME
    (e.g., in tests) are respected.
    """
    return os.path.abspath(os.path.expanduser("~/.cache/surus"))


def _normalize_path(path: str) -> str:
    """Return a normalized absolute path for the cache directory."""
    return os.path.abspath(os.path.expanduser(path))


def ensure_cache_dir(path: Optional[str] = None) -> str:
    """Ensure the SURUS cache directory exists and environment is set.

    - If ``path`` is provided, use it; otherwise use ``SURUS_CACHE`` from the
      environment or the default path (``~/.cache/surus``).
    - Create the directory if it does not exist.
    - Set ``os.environ["SURUS_CACHE"]`` to the resolved path so child processes
      and other modules see the same value.

    Returns:
        The absolute path to the cache directory.
    """
    # Determine target path
    env_path = path or os.environ.get("SURUS_CACHE") or _default_cache_dir()
    target = _normalize_path(env_path)

    # Create directory if needed
    os.makedirs(target, exist_ok=True)

    # Set environment variable to ensure consistency
    os.environ["SURUS_CACHE"] = target
    return target


def get_cache_dir() -> str:
    """Get the cache directory path, ensuring it exists.

    This function always ensures the directory exists and the environment
    variable ``SURUS_CACHE`` is set before returning the path.
    """
    return ensure_cache_dir()
