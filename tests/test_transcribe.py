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
import os
import tempfile
import io
from unittest.mock import patch, MagicMock, mock_open
import boio


def test_transcribe_function_exists():
    """Test that transcribe function is available"""
    assert hasattr(boio, 'transcribe')


def test_transcribe_parameters():
    """Test transcribe function has correct parameters"""
    import inspect
    sig = inspect.signature(boio.transcribe)
    expected_params = ['audio_input', 'high_performance', 'source_lang', 'target_lang', 'response_format', 'temperature']
    
    for param in expected_params:
        assert param in sig.parameters, f"Missing parameter: {param}"


def test_transcribe_missing_api_key():
    """Test transcribe raises error when API key missing"""
    # Temporarily remove API key
    old_key = os.environ.get("SURUS_API_KEY")
    if "SURUS_API_KEY" in os.environ:
        del os.environ["SURUS_API_KEY"]
    
    try:
        with pytest.raises(ValueError, match="SURUS_API_KEY environment variable not set"):
            boio.transcribe("dummy.wav")
    finally:
        # Restore API key
        if old_key:
            os.environ["SURUS_API_KEY"] = old_key


def test_pydub_import_handling():
    """Test that missing pydub dependency is handled gracefully"""
    from boio.transcribe import PYDUB_AVAILABLE
    
    # Test should pass regardless of whether pydub is available
    assert isinstance(PYDUB_AVAILABLE, bool)


@pytest.mark.skipif(getattr(__import__('boio.transcribe', fromlist=['']).transcribe, 'PYDUB_AVAILABLE', False) is False,
                    reason="pydub not available")
def test_convert_to_mono_function():
    """Test the mono conversion function with pydub available"""
    from boio.transcribe import _convert_to_mono
    
    # Test that the function exists and is callable
    assert callable(_convert_to_mono)


def test_missing_pydub_error():
    """Test that proper error is raised when pydub is missing but needed"""
    from boio.transcribe import _convert_to_mono
    
    # Mock PYDUB_AVAILABLE to False
    with patch('boio.transcribe.PYDUB_AVAILABLE', False):
        with pytest.raises(ImportError, match="pydub is required for audio processing"):
            _convert_to_mono("dummy.wav")


@patch('boio.transcribe.PYDUB_AVAILABLE', True)
@patch('boio.transcribe.AudioSegment')
@patch('boio.transcribe.tempfile.NamedTemporaryFile')
def test_mono_conversion_stereo_file(mock_temp_file, mock_audio_segment):
    """Test conversion of stereo file to mono"""
    from boio.transcribe import _convert_to_mono
    
    # Mock stereo audio (2 channels)
    mock_audio = MagicMock()
    mock_audio.channels = 2
    mock_audio.set_channels.return_value = mock_audio
    mock_audio_segment.from_file.return_value = mock_audio
    
    # Mock temporary file
    mock_temp = MagicMock()
    mock_temp.name = "/tmp/test_mono.wav"
    mock_temp_file.return_value = mock_temp
    
    result = _convert_to_mono("stereo_audio.mp3")
    
    # Verify stereo was converted to mono
    mock_audio.set_channels.assert_called_once_with(1)
    mock_audio.export.assert_called_once_with("/tmp/test_mono.wav", format="wav")
    assert result == "/tmp/test_mono.wav"


@patch('boio.transcribe.PYDUB_AVAILABLE', True)
@patch('boio.transcribe.AudioSegment')
@patch('boio.transcribe.tempfile.NamedTemporaryFile')
def test_mono_conversion_already_mono(mock_temp_file, mock_audio_segment):
    """Test that mono files are not unnecessarily converted"""
    from boio.transcribe import _convert_to_mono
    
    # Mock mono audio (1 channel)
    mock_audio = MagicMock()
    mock_audio.channels = 1
    mock_audio_segment.from_file.return_value = mock_audio
    
    # Mock temporary file
    mock_temp = MagicMock()
    mock_temp.name = "/tmp/test_mono.wav"
    mock_temp_file.return_value = mock_temp
    
    result = _convert_to_mono("mono_audio.wav")
    
    # Verify no conversion was attempted (set_channels not called)
    mock_audio.set_channels.assert_not_called()
    mock_audio.export.assert_called_once_with("/tmp/test_mono.wav", format="wav")
    assert result == "/tmp/test_mono.wav"


@patch.dict(os.environ, {'SURUS_API_KEY': 'test_key'})
@patch('boio.transcribe._convert_to_mono')
@patch('boio.transcribe.requests.post')
@patch('builtins.open', mock_open(read_data=b'fake audio data'))
def test_transcribe_with_file_path(mock_post, mock_convert):
    """Test transcribe function with file path input"""
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'text': 'Hello world'}
    mock_post.return_value = mock_response
    
    # Mock mono conversion
    mock_convert.return_value = '/tmp/mono_audio.wav'
    
    result = boio.transcribe('test_audio.mp3')
    
    # Verify mono conversion was called
    mock_convert.assert_called_once_with('test_audio.mp3')
    
    # Verify API was called
    mock_post.assert_called_once()
    
    # Verify result
    assert result == 'Hello world'


@patch.dict(os.environ, {'SURUS_API_KEY': 'test_key'})
@patch('boio.transcribe.requests.post')
def test_transcribe_with_file_object(mock_post):
    """Test transcribe function with file-like object input"""
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'text': 'Hello world'}
    mock_post.return_value = mock_response
    
    # Test with file-like object
    audio_data = io.BytesIO(b'fake audio data')
    result = boio.transcribe(audio_data)
    
    # Verify API was called
    mock_post.assert_called_once()
    
    # Verify result
    assert result == 'Hello world'


@patch.dict(os.environ, {'SURUS_API_KEY': 'test_key'})
@patch('boio.transcribe._convert_to_mono')
@patch('boio.transcribe.requests.post')
@patch('builtins.open', mock_open(read_data=b'fake audio data'))
def test_transcribe_error_with_debug_info(mock_post, mock_convert):
    """Test that debug info is shown on API errors"""
    # Mock API error response
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.json.return_value = {'detail': 'Server error'}
    mock_post.return_value = mock_response
    
    # Mock mono conversion
    mock_convert.return_value = '/tmp/mono_audio.wav'
    
    with pytest.raises(Exception, match="API Error 500"):
        boio.transcribe('test_audio.mp3')
    
    # Verify mono conversion was called
    mock_convert.assert_called_once_with('test_audio.mp3')


@patch.dict(os.environ, {'SURUS_API_KEY': 'test_key'})
@patch('boio.transcribe._convert_to_mono')
@patch('boio.transcribe.requests.post')
@patch('boio.transcribe.os.path.exists')
@patch('boio.transcribe.os.unlink')
@patch('builtins.open', mock_open(read_data=b'fake audio data'))
def test_temporary_file_cleanup(mock_unlink, mock_exists, mock_post, mock_convert):
    """Test that temporary files are properly cleaned up"""
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'text': 'Hello world'}
    mock_post.return_value = mock_response
    
    # Mock mono conversion creating a temp file
    mock_convert.return_value = '/tmp/mono_audio.wav'
    mock_exists.return_value = True
    
    result = boio.transcribe('stereo_audio.mp3')
    
    # Verify temp file was cleaned up
    mock_unlink.assert_called_once_with('/tmp/mono_audio.wav')
    assert result == 'Hello world'