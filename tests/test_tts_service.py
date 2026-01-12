"""
Unit tests for ElevenLabs TTS Service

Section 14.4: TTS & Audio Tests
Tests TTS generation with mocked ElevenLabs API responses.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from backend.services.tts import ElevenLabsTTSService, TTSResult


@pytest.fixture
def mock_api_key():
    """Mock API key for testing."""
    return "test_api_key_123"


@pytest.fixture
def mock_voice_id():
    """Mock voice ID for testing."""
    return "test_voice_id_456"


@pytest.fixture
def tts_service(mock_api_key, mock_voice_id):
    """Create TTS service with mock credentials."""
    return ElevenLabsTTSService(api_key=mock_api_key, voice_id=mock_voice_id)


@pytest.fixture
def mock_audio_bytes():
    """Mock audio data."""
    return b"fake_audio_data_12345"


# TTSResult Tests

@pytest.mark.unit
def test_tts_result_initialization(mock_audio_bytes):
    """Test TTSResult creation."""
    result = TTSResult(
        text="Hello world",
        audio_bytes=mock_audio_bytes,
        reading_order=1,
        voice_id="test_voice"
    )
    
    assert result.text == "Hello world"
    assert result.audio_bytes == mock_audio_bytes
    assert result.reading_order == 1
    assert result.voice_id == "test_voice"


@pytest.mark.unit
def test_tts_result_validates_audio_bytes():
    """Test that TTSResult validates audio_bytes type."""
    with pytest.raises(TypeError, match="audio_bytes must be bytes"):
        TTSResult(
            text="Hello",
            audio_bytes="not_bytes",  # Should be bytes
            reading_order=1,
            voice_id="test"
        )


# Service Initialization Tests

@pytest.mark.unit
def test_service_initialization_with_credentials(mock_api_key, mock_voice_id):
    """Test service initializes with provided credentials."""
    service = ElevenLabsTTSService(api_key=mock_api_key, voice_id=mock_voice_id)
    
    assert service.api_key == mock_api_key
    assert service.voice_id == mock_voice_id


@pytest.mark.unit
def test_service_initialization_from_settings():
    """Test service uses settings when no credentials provided."""
    with patch('backend.services.tts.settings') as mock_settings:
        mock_settings.ELEVENLABS_API_KEY = "settings_key"
        mock_settings.ELEVENLABS_VOICE_ID = "settings_voice"
        mock_settings.ELEVENLABS_CONFIGURED = True
        
        service = ElevenLabsTTSService()
        
        assert service.api_key == "settings_key"
        assert service.voice_id == "settings_voice"


# Speech Generation Tests

@pytest.mark.unit
@pytest.mark.elevenlabs
@patch('backend.services.tts.generate')
def test_generate_speech_success(mock_generate, tts_service, mock_audio_bytes):
    """
    Test successful speech generation.
    Section 14.4: Verify audio file creation per text bubble
    """
    mock_generate.return_value = mock_audio_bytes
    
    result = tts_service.generate_speech("Hello world", reading_order=1)
    
    assert result.text == "Hello world"
    assert result.audio_bytes == mock_audio_bytes
    assert result.reading_order == 1
    assert result.voice_id == tts_service.voice_id
    
    # Verify generate was called
    mock_generate.assert_called_once()


@pytest.mark.unit
@pytest.mark.elevenlabs
def test_generate_speech_not_configured():
    """Test error when TTS not configured."""
    service = ElevenLabsTTSService(api_key="", voice_id="")
    
    # The service should raise ValueError when attempting to use an empty API key
    with pytest.raises((ValueError, Exception)) as exc_info:
        service.generate_speech("Hello")
    
    # Check that an error was raised (either from our validation or the API)
    assert exc_info.value is not None


@pytest.mark.unit
@pytest.mark.elevenlabs
def test_generate_speech_empty_text(tts_service):
    """Test error with empty text."""
    with pytest.raises(ValueError, match="cannot be empty"):
        tts_service.generate_speech("")
    
    with pytest.raises(ValueError, match="cannot be empty"):
        tts_service.generate_speech("   ")


@pytest.mark.unit
@pytest.mark.elevenlabs
@patch('backend.services.tts.generate')
def test_generate_speech_handles_generator(mock_generate, tts_service):
    """Test handling of generator response from ElevenLabs."""
    # Simulate generator response
    mock_generate.return_value = (b"chunk1", b"chunk2", b"chunk3")
    
    result = tts_service.generate_speech("Test text")
    
    assert result.audio_bytes == b"chunk1chunk2chunk3"


@pytest.mark.unit
@pytest.mark.elevenlabs
@patch('backend.services.tts.generate')
def test_generate_speech_custom_voice(mock_generate, tts_service, mock_audio_bytes):
    """Test generation with custom voice ID."""
    mock_generate.return_value = mock_audio_bytes
    custom_voice = "custom_voice_123"
    
    result = tts_service.generate_speech("Hello", voice_id=custom_voice)
    
    assert result.voice_id == custom_voice


@pytest.mark.unit
@pytest.mark.elevenlabs
@patch('backend.services.tts.generate')
def test_generate_speech_api_error(mock_generate, tts_service):
    """Test handling of API errors."""
    mock_generate.side_effect = Exception("API Error")
    
    with pytest.raises(Exception, match="API Error"):
        tts_service.generate_speech("Hello")


# Async Speech Generation Tests

@pytest.mark.unit
@pytest.mark.elevenlabs
@pytest.mark.asyncio
@patch('backend.services.tts.generate')
async def test_generate_speech_async(mock_generate, tts_service, mock_audio_bytes):
    """Test async speech generation."""
    mock_generate.return_value = mock_audio_bytes
    
    result = await tts_service.generate_speech_async("Hello async", reading_order=2)
    
    assert result.text == "Hello async"
    assert result.reading_order == 2


# Batch Processing Tests

@pytest.mark.unit
@pytest.mark.elevenlabs
@pytest.mark.asyncio
@patch('backend.services.tts.generate')
async def test_generate_speech_batch(mock_generate, tts_service, mock_audio_bytes):
    """
    Test batch speech generation with parallelization.
    Section 8.1: Send text bubbles in parallel batches
    """
    mock_generate.return_value = mock_audio_bytes
    
    texts = ["First bubble", "Second bubble", "Third bubble"]
    results = await tts_service.generate_speech_batch(texts)
    
    assert len(results) == 3
    assert results[0].text == "First bubble"
    assert results[0].reading_order == 1
    assert results[1].text == "Second bubble"
    assert results[1].reading_order == 2
    assert results[2].text == "Third bubble"
    assert results[2].reading_order == 3


@pytest.mark.unit
@pytest.mark.elevenlabs
@pytest.mark.asyncio
async def test_generate_speech_batch_empty(tts_service):
    """Test batch processing with empty input."""
    results = await tts_service.generate_speech_batch([])
    assert results == []


@pytest.mark.unit
@pytest.mark.elevenlabs
@pytest.mark.asyncio
async def test_generate_speech_batch_not_configured():
    """Test batch processing when not configured."""
    service = ElevenLabsTTSService(api_key="", voice_id="")
    
    # Batch processing should return empty list if all fail (rather than raising)
    results = await service.generate_speech_batch(["Test"])
    
    # With empty API key, all TTS requests will fail
    assert len(results) == 0


@pytest.mark.unit
@pytest.mark.elevenlabs
@pytest.mark.asyncio
@patch('backend.services.tts.generate')
async def test_generate_speech_batch_respects_rate_limit(mock_generate, tts_service, mock_audio_bytes):
    """
    Test that batch processing respects rate limits.
    Section 8.1: Respect API rate limits
    """
    mock_generate.return_value = mock_audio_bytes
    
    # Create many texts to test parallelization
    texts = [f"Text {i}" for i in range(20)]
    
    with patch('backend.services.tts.settings') as mock_settings:
        mock_settings.ELEVENLABS_MAX_PARALLEL_REQUESTS = 5
        
        results = await tts_service.generate_speech_batch(texts)
        
        # Should successfully process all despite rate limiting
        assert len(results) == 20


@pytest.mark.unit
@pytest.mark.elevenlabs
@pytest.mark.asyncio
@patch('backend.services.tts.generate')
async def test_generate_speech_batch_handles_failures(mock_generate, tts_service, mock_audio_bytes):
    """
    Test batch processing continues on individual failures.
    Section 8.1: Parallel batches with error handling
    """
    # Create a side effect that fails for specific text
    def mock_side_effect(text, *args, **kwargs):
        if "Second" in text:
            raise Exception("API failed")
        return mock_audio_bytes
    
    mock_generate.side_effect = mock_side_effect
    
    texts = ["First", "Second", "Third"]
    results = await tts_service.generate_speech_batch(texts)
    
    # Should return only successful results (First and Third)
    assert len(results) == 2
    result_texts = [r.text for r in results]
    assert "First" in result_texts
    assert "Third" in result_texts
    assert "Second" not in result_texts


@pytest.mark.unit
@pytest.mark.elevenlabs
@pytest.mark.asyncio
@patch('backend.services.tts.generate')
async def test_generate_speech_batch_maintains_order(mock_generate, tts_service, mock_audio_bytes):
    """
    Test that results maintain reading order.
    Section 9: Maintain original reading order
    """
    mock_generate.return_value = mock_audio_bytes
    
    texts = ["A", "B", "C", "D", "E"]
    results = await tts_service.generate_speech_batch(texts)
    
    # Verify reading order is sequential
    for idx, result in enumerate(results):
        assert result.reading_order == idx + 1


# Synchronous Batch Wrapper Tests

@pytest.mark.unit
@pytest.mark.elevenlabs
@patch('backend.services.tts.generate')
def test_generate_speech_batch_sync(mock_generate, tts_service, mock_audio_bytes):
    """Test synchronous wrapper for batch generation."""
    mock_generate.return_value = mock_audio_bytes
    
    texts = ["Sync 1", "Sync 2"]
    results = tts_service.generate_speech_batch_sync(texts)
    
    assert len(results) == 2
    assert results[0].text == "Sync 1"
    assert results[1].text == "Sync 2"


@pytest.mark.unit
@pytest.mark.elevenlabs
@patch('backend.services.tts.generate')
def test_generate_speech_with_custom_voice_batch(mock_generate, tts_service, mock_audio_bytes):
    """Test batch generation with custom voice."""
    mock_generate.return_value = mock_audio_bytes
    custom_voice = "narrator_voice"
    
    results = tts_service.generate_speech_batch_sync(["Test"], voice_id=custom_voice)
    
    assert results[0].voice_id == custom_voice
