#!/usr/bin/env python

__author__ = "SURUS AI"
__copyright__ = "LLC"
__credits__ = ["SURUS AI"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "SURUS AI"
__email__ = "contact@surus.ai"
__status__ = "Development"

import requests
import os
from typing import Union, BinaryIO, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def transcribe(audio_input: Union[str, BinaryIO], 
               high_performance: bool = True,
               source_lang: Optional[str] = None, 
               target_lang: Optional[str] = None,
               response_format: str = "json", 
               temperature: float = 0.0) -> str:
    """
    Transcribe audio to text using SURUS API.
    
    Args:
        audio_input: Path to audio file or file-like object
        high_performance: Use best model (nvidia/canary-1b-v2) by default, set False for faster/cheaper Whisper
        source_lang: Source language for Canary model (e.g., 'es', 'en')  
        target_lang: Target language for Canary model (e.g., 'es', 'en')
        response_format: Output format for Whisper models ('json', 'text', 'srt', 'verbose_json', 'vtt')
        temperature: Sampling temperature for Whisper models (0-1)
    
    Returns:
        Transcribed text
    """
    api_key = os.getenv("SURUS_API_KEY")
    if not api_key:
        raise ValueError("SURUS_API_KEY environment variable not set")
    
    api_url = "https://api.surus.dev/functions/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Model selection
    model = 'nvidia/canary-1b-v2' if high_performance else 'surus-lat/whisper-large-v3-turbo-latam'
    
    # Prepare form data
    data = {'model': model}
    
    # Add model-specific parameters
    if model == 'nvidia/canary-1b-v2':
        if source_lang:
            data['source_lang'] = source_lang
        if target_lang:
            data['target_lang'] = target_lang
    else:
        data['response_format'] = response_format
        if temperature != 0.0:
            data['temperature'] = temperature
    
    # Both models actually use 'file' field (API docs were incorrect)
    file_field = 'file'
    
    # Handle file input
    if isinstance(audio_input, str):
        with open(audio_input, 'rb') as f:
            files = {file_field: f}
            response = requests.post(api_url, headers=headers, data=data, files=files)
    else:
        files = {file_field: audio_input}
        response = requests.post(api_url, headers=headers, data=data, files=files)
    
    response.raise_for_status()
    result = response.json()
    
    return result.get('text', str(result))