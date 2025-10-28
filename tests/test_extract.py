import pytest
import sulat
import os

# Test for the happy path

def test_extract_happy_path():
    # This test requires a valid SURUS_API_KEY to be set in the environment
    if not os.getenv("SURUS_API_KEY"):
        pytest.skip("SURUS_API_KEY not set, skipping API integration test")

    text = "The quick brown fox jumps over the lazy dog."
    json_schema = {
        "type": "object",
        "properties": {
            "animal": {"type": "string"},
            "action": {"type": "string"}
        }
    }

    try:
        result = sulat.extract(text, json_schema)
        assert isinstance(result, dict)
        assert "animal" in result
        assert "action" in result
    except Exception as e:
        pytest.fail(f"API call failed with exception: {e}")
