"""
Pytest configuration and shared fixtures.

Section 14: Testing Strategy
Provides test client, mocked APIs, and common test utilities.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock

from backend.main import app
from backend.config import Settings


@pytest.fixture
def test_client():
    """
    FastAPI test client for integration tests.
    Section 14.1: Integration Tests
    """
    return TestClient(app)


@pytest.fixture
def mock_settings():
    """
    Mock settings for isolated testing.
    """
    return Settings(
        DEBUG=True,
        GOOGLE_APPLICATION_CREDENTIALS="/mock/credentials.json",
        ELEVENLABS_API_KEY="mock_elevenlabs_key",
        ELEVENLABS_VOICE_ID="mock_voice_id",
        ALLOWED_ORIGINS=["http://localhost:3000"]
    )


@pytest.fixture
def mock_google_vision_client():
    """
    Mock Google Vision API client.
    Section 14.2: OCR-Specific Test Cases with mocked responses
    """
    mock_client = MagicMock()
    
    # Mock response structure
    mock_response = Mock()
    mock_response.text_annotations = []
    mock_response.error.message = ""
    
    mock_client.text_detection.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_elevenlabs_client():
    """
    Mock ElevenLabs API client.
    Section 14.4: TTS & Audio Tests with mocked audio payloads
    """
    mock_client = MagicMock()
    
    # Mock audio generation returning fake audio bytes
    mock_client.generate.return_value = b"fake_audio_data"
    
    return mock_client


@pytest.fixture
def sample_image_bytes():
    """
    Sample image data for testing.
    Returns minimal valid PNG bytes.
    """
    # Minimal 1x1 PNG file
    return (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
        b'\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
        b'\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    )


@pytest.fixture
def sample_ocr_result():
    """
    Sample OCR result from Google Vision API.
    Section 14.2: OCR-Specific Test Cases
    """
    return [
        {
            "text": "Hello, world!",
            "bounding_box": {
                "vertices": [
                    {"x": 10, "y": 20},
                    {"x": 100, "y": 20},
                    {"x": 100, "y": 40},
                    {"x": 10, "y": 40}
                ]
            }
        }
    ]
