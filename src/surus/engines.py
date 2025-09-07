"""SURUS engines for different AI modalities"""

import requests
import os
from typing import Union, BinaryIO, Optional


class AudioEngine:
    """Handles audio-related AI tasks"""
    
    def __init__(self):
        self.api_url = "https://api.surus.dev/functions/v1/audio/transcriptions"
        self.api_key = os.getenv("SURUS_API_KEY")
        
        # Default and high-performance models
        self.models = {
            'default': 'surus-lat/whisper-large-v3-turbo-latam',
            'high_performance': 'nvidia/canary-1b-v2'
        }
    
    def transcribe(self, audio_input: Union[str, BinaryIO], high_performance: bool = False,
                  source_lang: Optional[str] = None, target_lang: Optional[str] = None,
                  response_format: str = "json", temperature: float = 0.0) -> str:
        """Transcribe audio to text using SURUS API"""
        if not self.api_key:
            raise ValueError("SURUS_API_KEY environment variable not set")
        
        model = self.models['high_performance' if high_performance else 'default']
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Prepare form data
        data = {'model': model}
        
        # Add model-specific parameters
        if model == 'nvidia/canary-1b-v2':
            # Canary model uses 'file' field and language parameters
            if source_lang:
                data['source_lang'] = source_lang
            if target_lang:
                data['target_lang'] = target_lang
        else:
            # Whisper models use 'audio' field and support more format options
            data['response_format'] = response_format
            if temperature != 0.0:
                data['temperature'] = temperature
        
        # Handle file input
        if isinstance(audio_input, str):
            with open(audio_input, 'rb') as f:
                file_field = 'file' if model == 'nvidia/canary-1b-v2' else 'audio'
                files = {file_field: f}
                response = requests.post(self.api_url, headers=headers, data=data, files=files)
        else:
            file_field = 'file' if model == 'nvidia/canary-1b-v2' else 'audio'
            files = {file_field: audio_input}
            response = requests.post(self.api_url, headers=headers, data=data, files=files)
        
        response.raise_for_status()
        result = response.json()
        
        return result.get('text', str(result))


class TextEngine:
    """Handles text-related AI tasks"""
    
    def __init__(self):
        # TODO: Implement text endpoints when available
        # For now, placeholder for future text API endpoints
        self.models = {
            'default': 'text-model-default',
            'high_performance': 'text-model-premium'
        }
    
    def summarize(self, text: str, high_performance: bool = False, 
                 length: str = "medium", style: str = "concise") -> str:
        """Summarize text - TODO: implement when text API is available"""
        raise NotImplementedError("Text summarization will be implemented when SURUS text API is available")


# Global engine instances
_audio_engine = AudioEngine()
_text_engine = TextEngine()