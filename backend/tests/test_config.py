"""
Unit tests for configuration management.

Section 14.1: Unit Tests - Configuration validation
"""

import pytest
from backend.config import Settings


@pytest.mark.unit
def test_settings_defaults():
    """Test configuration values are loaded correctly."""
    settings = Settings()
    
    # These should always have values (from .env or defaults)
    assert settings.AUDIO_OUTPUT_FORMAT == "wav"
    assert settings.AUDIO_PAUSE_DURATION_MS == 500
    # Updated values for improved text grouping (webtoon-optimized)
    assert settings.BUBBLE_MAX_VERTICAL_GAP == 100  # Increased for multi-line speech bubbles
    assert settings.BUBBLE_MAX_CENTER_SHIFT == 150  # Increased for varying line lengths
    assert isinstance(settings.DEBUG, bool)


@pytest.mark.unit
def test_google_vision_configured_check(tmp_path):
    """Test Google Vision configuration detection."""
    # Create temporary credentials file
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text('{"type": "service_account"}')
    
    settings = Settings(GOOGLE_APPLICATION_CREDENTIALS=str(creds_file))
    assert settings.GOOGLE_VISION_CONFIGURED is True
    
    # Test with non-existent file
    settings = Settings(GOOGLE_APPLICATION_CREDENTIALS="/nonexistent/file.json")
    assert settings.GOOGLE_VISION_CONFIGURED is False


@pytest.mark.unit
def test_elevenlabs_configured_check():
    """Test ElevenLabs configuration detection."""
    settings = Settings(
        ELEVENLABS_API_KEY="test_key",
        ELEVENLABS_VOICE_ID="test_voice"
    )
    assert settings.ELEVENLABS_CONFIGURED is True
    
    # Test with missing values
    settings = Settings(ELEVENLABS_API_KEY="", ELEVENLABS_VOICE_ID="")
    assert settings.ELEVENLABS_CONFIGURED is False


@pytest.mark.unit
def test_rate_limit_settings():
    """
    Test rate limiting configuration.
    Section 5.2: Secure-by-default includes rate limiting
    """
    settings = Settings()
    
    assert settings.RATE_LIMIT_PER_MINUTE > 0
    assert settings.MAX_IMAGES_PER_REQUEST > 0
    assert settings.MAX_IMAGE_SIZE_MB > 0
