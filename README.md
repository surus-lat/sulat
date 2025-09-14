<div align="center">
  <h1>sulat</h1>
  <img src="public/icon.png" alt="icon" width="80" height="80">
  <p><em>an Opinionated Way of doing AI - Task-oriented AI nodes </em></p>
</div>

## Installation

### From PyPI (Recommended)

```bash
pip install sulat
```

### From Source

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```python
import sulat

# Basic transcription (uses Whisper model)
text = sulat.transcribe("audio.wav")

# High performance transcription (uses Canary model)
text = sulat.transcribe("audio.wav", high_performance=True, source_lang="es")

# Control source and target languages
text = sulat.transcribe("audio.wav", source_lang="es", target_lang="en")

# Additional options
text = sulat.transcribe(
    "audio.wav",
    high_performance=False,  # Use Whisper (default) or Canary model
    source_lang="auto",      # Auto-detect or specify language
    target_lang="en",        # Target language for translation
    response_format="json",  # Response format
    temperature=0.0          # Temperature for generation
)
```

### Environment Setup

Create a `.env` file in your project root:

```bash
# .env
SURUS_API_KEY=your_api_key_here
```

Or set the environment variable directly:
```bash
export SURUS_API_KEY="your_api_key"
```

## Development

```bash
# Setup environment
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
pytest tests/
```