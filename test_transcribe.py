#!/usr/bin/env python3

import boio
import os

print("🎯 Testing boio transcription with stereo-to-mono conversion...")
print()

# Check if audio file exists
audio_file = "audio.mp3"
if not os.path.exists(audio_file):
    audio_file = "audio_test.mp3"  # Try the other file name
    
if os.path.exists(audio_file):
    print(f"✅ Audio file found: {audio_file}")
    file_size = os.path.getsize(audio_file) / 1024 / 1024  # MB
    print(f"📁 File size: {file_size:.2f} MB")
    print()
    
    try:
        print("🔄 Starting transcription...")
        result = boio.transcribe(audio_file)
        print("✅ Transcription successful!")
        print()
        print("📝 Result:")
        print("-" * 50)
        print(result)
        print("-" * 50)
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("💡 Make sure you have:")
        print("  - SURUS_API_KEY environment variable set")
        print("  - ffmpeg installed (brew install ffmpeg)")
else:
    print(f"❌ Audio file not found: {audio_file}")
    print("Available files:")
    for f in os.listdir("."):
        if f.endswith((".mp3", ".wav", ".m4a")):
            print(f"  - {f}")