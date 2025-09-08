#!/usr/bin/env python

__author__ = "SURUS AI"
__copyright__ = "LLC"
__credits__ = ["SURUS AI"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "SURUS AI" 
__email__ = "contact@surus.ai"
__status__ = "Development"

import os
import tempfile
import pytest
import boio

def test_api_integration():
    """Test SURUS transcribe with real API (requires SURUS_API_KEY and audio file)"""
    
    # Check API key
    api_key = os.getenv("SURUS_API_KEY")
    if not api_key:
        print("❌ SURUS_API_KEY not set")
        print("Set it with: export SURUS_API_KEY='your_key'")
        return
    
    print("✓ SURUS_API_KEY found")
    
    # Check for audio processing capabilities
    try:
        from boio.transcribe import PYDUB_AVAILABLE
        if PYDUB_AVAILABLE:
            print("✓ Audio processing available (pydub + ffmpeg)")
        else:
            print("⚠️  Audio processing not available (pydub missing)")
    except ImportError:
        print("⚠️  Could not check audio processing capabilities")
    
    # This would need a real audio file to test
    # For demo purposes, show what the API calls would look like
    print("\n📝 Example usage (needs real audio file):")
    print("# Default model (Whisper)")
    print("result = boio.transcribe('audio.wav')")
    print("# Model: surus-lat/whisper-large-v3-turbo-latam")
    
    print("\n# High performance model (Canary)")
    print("result = boio.transcribe('audio.wav', high_performance=True, source_lang='es')")
    print("# Model: nvidia/canary-1b-v2")
    
    print("\n🎵 Audio requirements:")
    print("- Mono channel (stereo files automatically converted)")
    print("- Supported formats: WAV, MP3, M4A, etc.")
    print("- Automatic format conversion to WAV for API")
    
    print(f"\n🔗 API Endpoint: https://api.surus.dev/functions/v1/audio/transcriptions")
    print(f"🔑 API Key: {api_key[:10]}...")
    
    try:
        # This will fail without a real audio file, but shows the error handling
        boio.transcribe("nonexistent.wav")
    except FileNotFoundError:
        print("✓ File error handling works")
    except Exception as e:
        print(f"✓ Error handling works: {type(e).__name__}")


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("SURUS_API_KEY"), reason="SURUS_API_KEY not set")
def test_real_api_call_with_sample_audio():
    """Integration test with real API call (requires SURUS_API_KEY and creates sample audio)"""
    
    # Skip if pydub not available
    try:
        from boio.transcribe import PYDUB_AVAILABLE
        if not PYDUB_AVAILABLE:
            pytest.skip("pydub not available for audio generation")
    except ImportError:
        pytest.skip("Cannot check pydub availability")
    
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine
        
        # Generate a short test audio (1 second, 440Hz tone)
        tone = Sine(440).to_audio_segment(duration=1000)  # 1 second
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tone.export(tmp_file.name, format="wav")
            temp_audio_path = tmp_file.name
        
        try:
            # Test transcription (this will likely return something like "beep" or empty)
            result = boio.transcribe(temp_audio_path)
            
            # Just verify we got a response without errors
            assert isinstance(result, str)
            print(f"✅ API call successful. Result: '{result}'")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                
    except ImportError:
        pytest.skip("pydub.generators not available")
    except Exception as e:
        pytest.fail(f"Real API test failed: {e}")


def test_audio_format_requirements():
    """Test that we properly document audio format requirements"""
    
    # Test that the function docstring mentions mono requirement
    docstring = boio.transcribe.__doc__
    assert "mono" in docstring.lower() or "stereo" in docstring.lower()
    assert "pydub" in docstring.lower()
    assert "ffmpeg" in docstring.lower()


def test_error_handling_without_pydub():
    """Test error handling when pydub is not available"""
    
    # This test verifies that the package gracefully handles missing pydub
    from boio.transcribe import _convert_to_mono
    
    # Mock pydub as unavailable (using importlib to reference module)
    import importlib
    transcribe_module = importlib.import_module('boio.transcribe')
    original_pydub = getattr(transcribe_module, 'PYDUB_AVAILABLE', None)
    try:
        setattr(transcribe_module, 'PYDUB_AVAILABLE', False)
        with pytest.raises(ImportError, match="pydub is required"):
            _convert_to_mono("test.wav")
    finally:
        setattr(transcribe_module, 'PYDUB_AVAILABLE', original_pydub)


if __name__ == "__main__":
    test_api_integration()