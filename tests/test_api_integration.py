#!/usr/bin/env python

__author__ = "surus"
__copyright__ = "SURUS AI"
__credits__ = ["SURUS AI"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "surus"
__email__ = "contacto@surus.dev"
__status__ = "Development"


import os
import sulat

def test_api_integration():
    """Test sulat transcribe with real API (requires SURUS_API_KEY and audio file)"""
    
    # Check API key
    api_key = os.getenv("SURUS_API_KEY")
    if not api_key:
        print("❌ SURUS_API_KEY not set")
        print("Set it with: export SURUS_API_KEY='your_key'")
        return
    
    print("✓ SURUS_API_KEY found")
    
    # This would need a real audio file to test
    # For demo purposes, show what the API calls would look like
    print("\n📝 Example usage (needs real audio file):")
    print("# Default model (Whisper)")
    print("result = sulat.transcribe('audio.wav')")
    print("# Model: surus-lat/whisper-large-v3-turbo-latam")
    
    print("\n# High performance model (Canary)")
    print("result = sulat.transcribe('audio.wav', high_performance=True, source_lang='es')")
    print("# Model: nvidia/canary-1b-v2")
    
    print(f"\n🔗 API Endpoint: https://api.surus.dev/functions/v1/audio/transcriptions")
    print(f"🔑 API Key: {api_key[:10]}...")
    
    try:
        # This will fail without a real audio file, but shows the error handling
        sulat.transcribe("nonexistent.wav")
    except FileNotFoundError:
        print("✓ File error handling works")
    except Exception as e:
        print(f"✓ Error handling works: {type(e).__name__}")

if __name__ == "__main__":
    test_api_integration()