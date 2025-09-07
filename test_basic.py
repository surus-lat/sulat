#!/usr/bin/env python3
"""Basic test of SURUS transcribe functionality"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import surus

def test_transcribe_basic():
    """Test basic transcribe functionality"""
    print("Testing SURUS transcribe...")
    
    # Test imports and function existence
    assert hasattr(surus, 'transcribe'), "transcribe function not found"
    assert hasattr(surus, 'summarize'), "summarize function not found"
    
    print("✓ Basic imports working")
    print("✓ transcribe() function available") 
    print("✓ summarize() function available")
    
    # Test function signatures
    import inspect
    transcribe_sig = inspect.signature(surus.transcribe)
    expected_params = ['audio_input', 'high_performance', 'source_lang', 'target_lang', 'response_format', 'temperature', 'custom_prompt']
    
    for param in expected_params:
        assert param in transcribe_sig.parameters, f"Missing parameter: {param}"
    
    print("✓ transcribe() has correct SURUS API parameters")
    
    # Test models configuration
    from surus.engines import _audio_engine
    assert _audio_engine.models['default'] == 'surus-lat/whisper-large-v3-turbo-latam'
    assert _audio_engine.models['high_performance'] == 'nvidia/canary-1b-v2'
    print("✓ Audio engine configured with correct models")
    
    # Test API URL
    assert _audio_engine.api_url == "https://api.surus.dev/functions/v1/audio/transcriptions"
    print("✓ Audio engine pointing to SURUS API")
    
    print("\n🎉 Basic functionality test passed!")
    print("\nNext steps:")
    print("1. Set SURUS_API_KEY environment variable")
    print("2. Test with real audio file:")
    print("   # Default model (Whisper)")
    print("   surus.transcribe('audio.wav')")
    print("   # High performance model (Canary)")
    print("   surus.transcribe('audio.wav', high_performance=True, source_lang='es')")

if __name__ == "__main__":
    test_transcribe_basic()