#!/usr/bin/env python

__author__ = "SURUS AI"
__copyright__ = "LLC"
__credits__ = ["SURUS AI"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "SURUS AI"
__email__ = "contact@surus.ai"
__status__ = "Development"

import inspect
import pytest
import requests
import importlib


def test_translate_function_exists():
    """Test that translate function is available in the translate module"""
    translate_module = importlib.import_module("sulat.translate")
    assert hasattr(translate_module, 'translate')


def test_translate_parameters():
    """Test translate function has correct parameters"""
    translate_module = importlib.import_module("sulat.translate")
    sig = inspect.signature(translate_module.translate)
    expected_params = ['text', 'target_lang', 'model', 'sampling_params']

    for param in expected_params:
        assert param in sig.parameters, f"Missing parameter: {param}"


def test_translate_missing_api_key(monkeypatch):
    """Test translate raises error when API key missing"""
    translate_module = importlib.import_module("sulat.translate")
    # Ensure the key is not present
    monkeypatch.delenv("SURUS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="SURUS_API_KEY environment variable not set"):
        translate_module.translate("dummy text", "es")


def test_translate_invalid_sampling_params_type(monkeypatch):
    """Test translate raises TypeError when sampling_params is not a dict"""
    translate_module = importlib.import_module("sulat.translate")
    # Ensure API key present so the sampling_params check is reached
    monkeypatch.setenv("SURUS_API_KEY", "fake-key")

    with pytest.raises(TypeError, match="sampling_params must be a dict"):
        translate_module.translate("hello", "es", sampling_params="not-a-dict")


def test_translate_calls_api_and_returns_translation(monkeypatch):
    """Test translate performs an API call and returns translated content (mocked)"""
    translate_module = importlib.import_module("sulat.translate")

    # Provide a fake API key
    monkeypatch.setenv("SURUS_API_KEY", "fake-key")

    # Dummy response object to emulate requests.Response behavior
    class DummyResponse:
        def __init__(self, json_data, status_code=200):
            self._json = json_data
            self.status_code = status_code
            self.text = str(json_data)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} Error")

        def json(self):
            return self._json

    # Patch the requests.post used in the translate module
    def fake_post(*args, **kwargs):
        return DummyResponse({"choices": [{"message": {"content": "hola mundo"}}]})

    # Patch the requests.post function used inside sulat.translate
    monkeypatch.setattr(translate_module.requests, "post", fake_post)

    translated = translate_module.translate("hello world", "es")
    assert translated == "hola mundo"