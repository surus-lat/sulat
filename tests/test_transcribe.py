#!/usr/bin/env python

__author__ = "SURUS AI"
__copyright__ = "LLC"
__credits__ = ["SURUS AI"]  
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "SURUS AI"
__email__ = "contact@surus.ai"
__status__ = "Development"

import pytest
import sulat


def test_transcribe_function_exists():
    """Test that transcribe function is available"""
    assert hasattr(sulat, 'transcribe')


def test_transcribe_parameters():
    """Test transcribe function has correct parameters"""
    import inspect
    sig = inspect.signature(sulat.transcribe)
    expected_params = ['audio_input', 'high_performance', 'source_lang', 'target_lang', 'response_format', 'temperature']
    
    for param in expected_params:
        assert param in sig.parameters, f"Missing parameter: {param}"


def test_transcribe_missing_api_key():
    """Test transcribe raises error when API key missing"""
    import os
    # Temporarily remove API key
    old_key = os.environ.get("SURUS_API_KEY")
    if "SURUS_API_KEY" in os.environ:
        del os.environ["SURUS_API_KEY"]
    
    try:
        with pytest.raises(ValueError, match="SURUS_API_KEY environment variable not set"):
            sulat.transcribe("dummy.wav")
    finally:
        # Restore API key
        if old_key:
            os.environ["SURUS_API_KEY"] = old_key