<div align="center">
  <h1>boi</h1>
  <img src="public/icon.png" alt="icon" width="80" height="80">
  <p><em>an opinionated way of doing ai</em></p>
</div>

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```python
import boi

# Basic transcription
text = boi.transcribe("audio.wav")

# control source and target langs
text = boi.transcribe("audio.wav", source_lang="es", target_lang="es")
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