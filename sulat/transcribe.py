__author__ = "SURUS AI"
__copyright__ = "LLC"
__credits__ = ["SURUS AI"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "SURUS AI"
__email__ = "contact@surus.dev"
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
               **kwargs) -> str:
    """
    Transcribe audio to text using SURUS API via the new /transcribe endpoint.
    
    Args:
        audio_input: Path to audio file or file-like object.
        high_performance: Use high performance model (determines which model internally).
        source_lang: Source language for transcription (e.g., 'es', 'en').
        **kwargs: Additional parameters to pass to the API.
    
    Returns:
        Transcribed text
    """
    api_key = os.getenv("SURUS_API_KEY")
    if not api_key:
        raise ValueError("SURUS_API_KEY environment variable not set")
    
    api_url = "https://api.surus.dev/functions/v1/transcribe"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Prepare form data
    data = {'high_performance': high_performance}
    
    # Add optional parameters
    if source_lang:
        data['source_lang'] = source_lang
    
    # Add any additional parameters passed as kwargs
    data.update(kwargs)

    # Handle file upload
    if isinstance(audio_input, str):
        with open(audio_input, 'rb') as f:
            files = {'file': f}
            response = requests.post(api_url, headers=headers, data=data, files=files)
    else:
        # For file-like objects, we need to pass the object directly
        files = {'file': audio_input}
        response = requests.post(api_url, headers=headers, data=data, files=files)

    try:
        response.raise_for_status()
    except requests.HTTPError as err:
        try:
            error_json = response.json()
        except ValueError:
            error_json = {"error": response.text}
        raise Exception(f"SURUS API error {response.status_code}: {error_json}") from err

    result = response.json()
    return result.get('text', str(result))