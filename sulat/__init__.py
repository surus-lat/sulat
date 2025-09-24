#!/usr/bin/env python

__author__ = "neosanma"
__copyright__ = "SURUS AI"
__credits__ = ["SURUS AI"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "neosanma"
__email__ = "contacto@neosanma.ai"
__status__ = "Development"

from .config import ensure_cache_dir, get_cache_dir  # Ensure SURUS_CACHE is set on import
ensure_cache_dir()

from .transcribe import transcribe
from .extract import extract

__all__ = ["ensure_cache_dir", "extract", "get_cache_dir", "transcribe"]