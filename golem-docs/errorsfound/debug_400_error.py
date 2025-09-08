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
import requests
from dotenv import load_dotenv

load_dotenv()

def debug_400_error():
    """Debug the 400 Bad Request error"""
    
    print("🔍 Debugging 400 Bad Request error...")
    
    # Check API key
    api_key = os.getenv("SURUS_API_KEY")
    if not api_key:
        print("❌ No SURUS_API_KEY found")
        return
    print(f"✓ API Key found: {api_key[:10]}...")
    
    # Check if audio file exists
    if not os.path.exists("audio_test.wav"):
        print("❌ audio_test.wav not found")
        return
    print(f"✓ File exists: {os.path.getsize('audio_test.wav')} bytes")
    
    # Test API call step by step
    api_url = "https://api.surus.dev/functions/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Try default Whisper model
    print("\n📡 Testing Whisper model...")
    with open('audio_test.wav', 'rb') as f:
        files = {'audio': f}
        data = {'model': 'surus-lat/whisper-large-v3-turbo-latam'}
        response = requests.post(api_url, headers=headers, data=data, files=files)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 400:
        print("\n💡 400 Bad Request analysis:")
        print("- Check if audio file is valid format")
        print("- File might be too small/corrupted")
        print("- Wrong field name (should be 'audio' for Whisper)")
        
        # Try Canary model
        print("\n📡 Testing Canary model...")
        with open('audio_test.wav', 'rb') as f:
            files = {'file': f}  # Canary uses 'file' not 'audio'
            data = {
                'model': 'nvidia/canary-1b-v2',
                'source_lang': 'es',
                'target_lang': 'es'
            }
            response2 = requests.post(api_url, headers=headers, data=data, files=files)
        
        print(f"Canary Status: {response2.status_code}")
        print(f"Canary Response: {response2.text}")

if __name__ == "__main__":
    debug_400_error()