<div align="center">
  <h1>boio</h1>
  <img src="public/icon.png" alt="icon" width="80" height="80">
  <p><em>an opinionated way of doing ai</em></p>
</div>

## Installation

```bash
# Install the package
pip install boio

# Optional: audio processing extras (only needed for local mono conversion fallback)
pip install "boio[audio]"
```

### Development Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage
export your surus api key: 

```bash
export SURUS_API_KEY=<your-api-key>
```


```python
import boio

# Basic transcription (server handles audio normalization)
text = boio.transcribe("audio.wav")

# control source and target langs
text = boio.transcribe("audio.wav", source_lang="es", target_lang="es")
```

### Audio handling behavior

- By default, boio sends your audio as-is. The SURUS backend performs necessary normalization (e.g., mono conversion).
- If the server responds with a mono-channel requirement error, boio will automatically retry by converting to mono locally IF the optional `boio[audio]` extras are installed (requires ffmpeg on your system).
- If local conversion is not available, boio raises a clear error message explaining how to enable the fallback or how to fix on the server.

### Environment Setup

Create a `.env` file in your project root:

```bash
# .env
SURUS_API_KEY=your_api_key_here
```

then load SURUS_API_KEY variable with
```bash
source .env
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