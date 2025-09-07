"""SURUS AI - Task-oriented AI library"""

from .client import transcribe, summarize, extract_to_json, chat, annotate

__version__ = "0.1.0"
__all__ = ["transcribe", "summarize", "extract_to_json", "chat", "annotate"]