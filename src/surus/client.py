"""SURUS main client interface"""

from typing import Union, BinaryIO, Optional
from .engines import _audio_engine, _text_engine


def transcribe(audio_input: Union[str, BinaryIO], high_performance: bool = False,
              source_lang: Optional[str] = None, target_lang: Optional[str] = None,
              response_format: str = "json", temperature: float = 0.0,
              custom_prompt: Optional[str] = None) -> str:
    """
    Transcribe audio to text using SURUS API.
    
    Args:
        audio_input: Path to audio file or file-like object
        high_performance: Use best model (nvidia/canary-1b-v2) for higher accuracy (default: False)
        source_lang: Source language for Canary model (e.g., 'es', 'en')
        target_lang: Target language for Canary model (e.g., 'es', 'en')
        response_format: Output format for Whisper models ('json', 'text', 'srt', 'verbose_json', 'vtt')
        temperature: Sampling temperature for Whisper models (0-1)
        custom_prompt: Custom prompt (future feature)
    
    Returns:
        Transcribed text
    """
    return _audio_engine.transcribe(
        audio_input=audio_input,
        high_performance=high_performance,
        source_lang=source_lang,
        target_lang=target_lang,
        response_format=response_format,
        temperature=temperature
    )


def summarize(text: str, high_performance: bool = False,
             length: str = "medium", style: str = "concise",
             custom_prompt: Optional[str] = None) -> str:
    """
    Summarize text.
    
    Args:
        text: Text to summarize
        high_performance: Use best model for higher quality (default: False)
        length: Summary length ('short', 'medium', 'long')
        style: Summary style ('concise', 'detailed', 'bullet-points')
        custom_prompt: Custom prompt to append to base summary prompt
    
    Returns:
        Summarized text
    """
    return _text_engine.summarize(
        text=text,
        high_performance=high_performance,
        length=length,
        style=style
    )


def extract_to_json(text: str, high_performance: bool = False,
                   schema: Optional[dict] = None, strict_mode: bool = True,
                   custom_prompt: Optional[str] = None) -> dict:
    """Extract structured data from text as JSON"""
    raise NotImplementedError("extract_to_json will be implemented in next phase")


def chat(message: str, high_performance: bool = False,
         context: Optional[list] = None, temperature: float = 0.7,
         custom_prompt: Optional[str] = None) -> str:
    """Chat with AI model"""
    raise NotImplementedError("chat will be implemented in next phase")


def annotate(image_input: Union[str, BinaryIO], high_performance: bool = False,
            annotation_type: str = "describe", custom_prompt: Optional[str] = None) -> str:
    """Annotate images with AI"""
    raise NotImplementedError("annotate will be implemented in next phase")