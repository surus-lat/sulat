import sulat
from dotenv import load_dotenv
load_dotenv()

print("🎯 Testing SURUS transcribe...")

# Test default model (Whisper)
result = sulat.transcribe("audio_test.wav")
print(f"✅ Whisper result: {result}")

# Test high performance model (Canary)
result2 = sulat.transcribe("audio_test.wav", high_performance=True, source_lang="es")
print(f"✅ Canary result: {result2}")


