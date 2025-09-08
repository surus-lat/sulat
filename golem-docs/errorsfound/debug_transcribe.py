#!/usr/bin/env python




import os
import boio
import requests

def debug_transcribe():
    """Debug transcription issues"""
    
    print("🔍 Debugging SURUS transcribe...")
    
    # Check API key
    api_key = os.getenv("SURUS_API_KEY")
    print(f"✓ API Key: {api_key[:10]}..." if api_key else "❌ No API key")
    
    # Check if file exists
    test_files = ["audio_test.mp3", "audio_test.wav", "test.mp3", "test.wav"]
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if existing_files:
        print(f"✓ Found audio files: {existing_files}")
        test_file = existing_files[0]
    else:
        print(f"❌ No audio files found. Looking for: {test_files}")
        print("\nTo test properly, you need an audio file. Try:")
        print("1. Record a short audio: say something and save as 'test.wav'")
        print("2. Download a sample: curl -o test.wav 'https://www.soundjay.com/misc/sounds/bell-ringing-05.wav'")
        return
    
    print(f"\n📁 Testing with file: {test_file}")
    print(f"📊 File size: {os.path.getsize(test_file)} bytes")
    
    try:
        # Test API directly first
        print("\n🔗 Testing API directly...")
        
        api_url = "https://api.surus.dev/functions/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        with open(test_file, 'rb') as f:
            files = {'audio': f}
            data = {'model': 'surus-lat/whisper-large-v3-turbo-latam'}
            response = requests.post(api_url, headers=headers, data=data, files=files)
        
        print(f"📡 Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"❌ Response text: {response.text}")
        else:
            print(f"✓ API works! Response: {response.json()}")
            
        # Now test through SURUS
        print(f"\n🎯 Testing through surus.transcribe()...")
        result = surus.transcribe(test_file)
        print(f"✅ Success! Result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if "400" in str(e):
            print("\n💡 400 error usually means:")
            print("- Unsupported audio format (try .wav instead of .mp3)")
            print("- Audio file corrupted or too small")
            print("- Missing required parameters")

if __name__ == "__main__":
    debug_transcribe()