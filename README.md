<div align="center">
  <h1>sulat</h1>
  <img src="public/icon.png" alt="sulat icon" width="80" height="80">
  <p><em>Task-oriented AI nodes for transcription, translation, and structured extraction</em></p>
</div>

## Overview

`sulat` wraps SURUS AI endpoints to provide:
- **Speech transcription** with high-performance model option.
- **Structured extraction** guided by JSON schemas.
- **Text translation** between languages.

Python ≥ 3.9 is recommended.

## Installation

```bash
pip install sulat
```

From source:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Quickstart

```python
import sulat

# Whisper-style transcription
text = sulat.transcribe("audio.wav")

# High-performance transcription with source language
text_hp = sulat.transcribe(
    "audio.wav",
    high_performance=True,
    source_lang="es"
)
```

### Structured extraction

```python
schema = {
    "type": "object",
    "properties": {
        "animal": {"type": "string"},
        "action": {"type": "string"}
    }
}

result = sulat.extract(
    text="The quick brown fox jumps over the lazy dog.",
    json_schema=schema,
)
```

### Text Translation

```python
translated_text = sulat.translate(
    text="Hello, world!",
    target_lang="es"
)
```

## Environment

Set your SURUS API key:

```bash
export SURUS_API_KEY="your_api_key"
```

`sulat` caches intermediate files under `SURUS_CACHE` (defaults to `~/.cache/surus`). Override with:

```bash
export SURUS_CACHE="/custom/cache/dir"
```


## Development

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

Troubleshooting tips:
- Install `pydub` (and `ffmpeg`) for local stereo-to-mono conversion if needed.
- Ensure `SURUS_API_BASE` if you use a custom proxy.


## Run tests
pytest tests/
```
