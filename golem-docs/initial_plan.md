Fix and Test SURUS Transcribe

Step 1: Fix Package Configuration

- Update pyproject.toml to match current surus/ directory structure
- Remove incorrect src/ references
- Ensure proper package discovery

Step 2: Setup Development Environment

- Create UV virtual environment: uv venv
- Install package in development mode: uv pip install -e ".[dev]"
- Install dependencies (requests, pytest)

Step 3: Basic Function Tests

- Run pytest tests/ to verify imports and parameter validation
- Test error handling (missing API key)
- Verify function signatures match SURUS API requirements

Step 4: Integration Test (Optional)

- Create simple test script with real audio file
- Set SURUS_API_KEY environment variable
- Test both default (Whisper) and high_performance (Canary) models
- Verify API responses

Expected outcome: Working surus.transcribe() function that calls SURUS API correctly following 
task-first philosophy.