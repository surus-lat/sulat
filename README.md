<div align="center">
  <img src="public/icon.png" alt="icon" width="80" height="80">
  # surus
  
  *an opinionated way of doing ai*
</div>


## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```python
import surus

# Basic transcription (Whisper model)
text = surus.transcribe("audio.wav")

# High performance transcription (Canary model) 
text = surus.transcribe("audio.wav", high_performance=True, source_lang="es")
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