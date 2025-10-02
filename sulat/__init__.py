#!/usr/bin/env python

__author__ = "surus"
__copyright__ = "SURUS AI"
__credits__ = ["SURUS AI"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "surus"
__email__ = "contacto@surus.dev"
__status__ = "Development"

from .config import ensure_cache_dir, get_cache_dir  # Ensure SURUS_CACHE is set on import
ensure_cache_dir()

from .transcribe import transcribe
from .extract import extract

__all__ = ["ensure_cache_dir", "get_cache_dir", "transcribe", "extract"]

