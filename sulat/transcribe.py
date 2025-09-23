#!/usr/bin/env python

__author__ = "SURUS AI"
__copyright__ = "LLC"
__credits__ = ["SURUS AI"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "SURUS AI"
__email__ = "contact@surus.ai"
__status__ = "Development"

import requests
import os
import tempfile
import warnings
from typing import Union, BinaryIO, Optional
from dotenv import load_dotenv
from .config import get_cache_dir

# Suppress pydub regex warnings in Python 3.12+
warnings.filterwarnings("ignore", message="invalid escape sequence", category=SyntaxWarning)

# Lazy import flags and holders for pydub
AudioSegment = None  # type: ignore
PYDUB_AVAILABLE = None  # type: ignore

def _ensure_pydub_imported() -> None:
    """Import pydub lazily and set availability flags.

    This avoids stale import-state issues when users install pydub after
    importing sulat within the same Python process.
    """
    global AudioSegment, PYDUB_AVAILABLE
    if PYDUB_AVAILABLE is not None:
        return
    try:
        from pydub import AudioSegment as _AudioSegment  # type: ignore
        AudioSegment = _AudioSegment
        PYDUB_AVAILABLE = True
    except Exception:
        PYDUB_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()


def _convert_to_mono(audio_path: str) -> str:
    """Convert audio file to mono and return path to converted file."""
    _ensure_pydub_imported()
    if not PYDUB_AVAILABLE:
        raise ImportError(
            "pydub is required for audio processing. Install with: pip install pydub\n"
            "Note: You may also need to install ffmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )
    
    # Load audio file
    audio = AudioSegment.from_file(audio_path)
    
    # Convert to mono if stereo
    if audio.channels > 1:
        print(f"Converting {audio.channels}-channel audio to mono...")
        audio = audio.set_channels(1)
    
    # Create temporary file for mono audio under SURUS_CACHE
    cache_dir = get_cache_dir()
    # Use NamedTemporaryFile with dir to place it under our cache
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=cache_dir)
    temp_path = temp_file.name
    temp_file.close()
    
    # Export as WAV (which is more reliable for API)
    audio.export(temp_path, format="wav")
    
    return temp_path


def transcribe(audio_input: Union[str, BinaryIO], 
               high_performance: bool = True,
               source_lang: Optional[str] = 'es', 
               target_lang: Optional[str] = 'es',
               response_format: str = "json", 
               temperature: float = 0.0) -> str:
    """
    Transcribe audio to text using SURUS API.
    
    Args:
        audio_input: Path to audio file or file-like object. Stereo files are automatically converted to mono.
        high_performance: Use best model (nvidia/canary-1b-v2) for higher accuracy
        source_lang: Source language for Canary model (e.g., 'es', 'en')  
        target_lang: Target language for Canary model (e.g., 'es', 'en')
        response_format: Output format for Whisper models ('json', 'text', 'srt', 'verbose_json', 'vtt')
        temperature: Sampling temperature for Whisper models (0-1)
    
    Returns:
        Transcribed text
        
    Behavior:
        - First, sends audio as-is to the SURUS API (server is expected to handle conversion).
        - If the API responds with a stereo/mono channel error, sulat will retry by converting
          locally to mono ONLY when pydub (and ffmpeg) are available.
        - If pydub is not available, a clear error message is raised with guidance.
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
    
    # Store debug info for error cases
    debug_info = {
        'api_url': api_url,
        'model': model,
        'data': data,
        'file_field': file_field
    }
    
    # First attempt: send audio as-is
    temp_file_path = None
    def _post_file(file_obj: BinaryIO):
        return requests.post(api_url, headers=headers, data=data, files={file_field: file_obj})

    if isinstance(audio_input, str):
        with open(audio_input, 'rb') as f:
            response = _post_file(f)
    else:
        response = _post_file(audio_input)

    # If not OK, optionally retry after local mono conversion (when available)
    if response.status_code != 200:
        # Try to extract server error JSON
        error_text = None
        error_json = None
        try:
            error_json = response.json()
        except ValueError:
            error_text = response.text

        stereo_error = False
        if error_json and isinstance(error_json, dict):
            detail = str(error_json.get('detail', ''))
            stereo_error = ('single channel' in detail.lower()) or ('mono' in detail.lower())

        if stereo_error and isinstance(audio_input, str):
            # Retry with local mono conversion if pydub is available
            try:
                mono_audio_path = _convert_to_mono(audio_input)
                temp_file_path = mono_audio_path if mono_audio_path != audio_input else None
                with open(mono_audio_path, 'rb') as f:
                    response = _post_file(f)
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        # If still not OK, raise with debug info
        if response.status_code != 200:
            print("🐛 Debug information:")
            print(f"   API URL: {debug_info['api_url']}")
            print(f"   Model: {debug_info['model']}")
            print(f"   Data: {debug_info['data']}")
            print(f"   File field: {debug_info['file_field']}")
            print()
            if error_json is not None:
                raise Exception(f"API Error {response.status_code}: {error_json}")
            else:
                raise Exception(f"API Error {response.status_code}: {error_text}")
    
    result = response.json()
    return result.get('text', str(result))