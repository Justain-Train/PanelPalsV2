"""
Integration Tests for /process/chapter Endpoint

Section 14.3: Integration Tests
Validates interactions between system components through the full pipeline.
"""

import pytest
import io
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import numpy as np
from scipy.io import wavfile

from backend.main import app
from backend.services import (
    BoundingBox,
    OCRResult,
    TextBubble,
    TTSResult
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_settings():
    """Mock settings to show all services configured."""
    with patch('backend.routers.process.settings') as mock:
        mock.GOOGLE_VISION_CONFIGURED = True
        mock.ELEVENLABS_CONFIGURED = True
        mock.AUDIO_SAMPLE_RATE = 44100
        mock.DEBUG = False
        yield mock


@pytest.fixture
def mock_image_files():
    """Create mock image files for upload."""
    # Create simple test image bytes (PNG header + minimal data)
    image_data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
    
    files = [
        ("images", ("panel1.png", io.BytesIO(image_data), "image/png")),
        ("images", ("panel2.png", io.BytesIO(image_data), "image/png")),
        ("images", ("panel3.png", io.BytesIO(image_data), "image/png")),
    ]
    return files


@pytest.fixture
def mock_ocr_results():
    """Create mock OCR results."""
    return [
        [  # Image 1 results
            OCRResult(
                text="Hello",
                bounding_box=BoundingBox([
                    {"x": 10, "y": 10}, {"x": 50, "y": 10},
                    {"x": 50, "y": 30}, {"x": 10, "y": 30}
                ]),
                confidence=0.99
            ),
            OCRResult(
                text="World",
                bounding_box=BoundingBox([
                    {"x": 10, "y": 40}, {"x": 50, "y": 40},
                    {"x": 50, "y": 60}, {"x": 10, "y": 60}
                ]),
                confidence=0.99
            ),
        ],
        [  # Image 2 results
            OCRResult(
                text="How",
                bounding_box=BoundingBox([
                    {"x": 20, "y": 20}, {"x": 60, "y": 20},
                    {"x": 60, "y": 40}, {"x": 20, "y": 40}
                ]),
                confidence=0.99
            ),
            OCRResult(
                text="are",
                bounding_box=BoundingBox([
                    {"x": 20, "y": 50}, {"x": 60, "y": 50},
                    {"x": 60, "y": 70}, {"x": 20, "y": 70}
                ]),
                confidence=0.99
            ),
            OCRResult(
                text="you",
                bounding_box=BoundingBox([
                    {"x": 20, "y": 80}, {"x": 60, "y": 80},
                    {"x": 60, "y": 100}, {"x": 20, "y": 100}
                ]),
                confidence=0.99
            ),
        ],
        [  # Image 3 results
            OCRResult(
                text="Good",
                bounding_box=BoundingBox([
                    {"x": 30, "y": 30}, {"x": 70, "y": 30},
                    {"x": 70, "y": 50}, {"x": 30, "y": 50}
                ]),
                confidence=0.99
            ),
            OCRResult(
                text="bye",
                bounding_box=BoundingBox([
                    {"x": 30, "y": 60}, {"x": 70, "y": 60},
                    {"x": 70, "y": 80}, {"x": 30, "y": 80}
                ]),
                confidence=0.99
            ),
        ],
    ]


@pytest.fixture
def mock_text_bubbles():
    """Create mock text bubbles."""
    return [
        TextBubble(
            text="Hello World",
            reading_order=1,
            bounding_box=BoundingBox([
                {"x": 10, "y": 10}, {"x": 50, "y": 10},
                {"x": 50, "y": 60}, {"x": 10, "y": 60}
            ])
        ),
        TextBubble(
            text="How are you",
            reading_order=2,
            bounding_box=BoundingBox([
                {"x": 20, "y": 20}, {"x": 60, "y": 20},
                {"x": 60, "y": 100}, {"x": 20, "y": 100}
            ])
        ),
        TextBubble(
            text="Good bye",
            reading_order=3,
            bounding_box=BoundingBox([
                {"x": 30, "y": 30}, {"x": 70, "y": 30},
                {"x": 70, "y": 80}, {"x": 30, "y": 80}
            ])
        ),
    ]


@pytest.fixture
def mock_tts_results():
    """Create mock TTS results with audio bytes."""
    def create_mock_audio(duration_ms=100):
        """Create mock MP3/WAV audio bytes."""
        sample_rate = 44100
        num_samples = int((duration_ms / 1000.0) * sample_rate)
        audio_data = np.zeros(num_samples, dtype=np.int16)
        
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_data)
        return buffer.getvalue()
    
    return [
        TTSResult(
            text="Hello World.",
            audio_bytes=create_mock_audio(500),
            reading_order=1,
            voice_id="test_voice"
        ),
        TTSResult(
            text="How are you.",
            audio_bytes=create_mock_audio(500),
            reading_order=2,
            voice_id="test_voice"
        ),
        TTSResult(
            text="Good bye.",
            audio_bytes=create_mock_audio(500),
            reading_order=3,
            voice_id="test_voice"
        ),
    ]


@pytest.fixture
def mock_wav_bytes():
    """Create mock final WAV file bytes."""
    sample_rate = 44100
    duration_seconds = 2
    audio_data = np.zeros(sample_rate * duration_seconds, dtype=np.int16)
    
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_data)
    return buffer.getvalue()


# Section 14.3: Integration Tests - Full Pipeline

@pytest.mark.integration
@patch('backend.routers.process.settings')
@patch('backend.routers.process.get_audio_stitcher')
@patch('backend.routers.process.get_tts_service')
@patch('backend.routers.process.get_text_grouper')
@patch('backend.routers.process.get_ocr_service')
def test_process_chapter_success(
    mock_get_ocr,
    mock_get_grouper,
    mock_get_tts,
    mock_get_stitcher,
    mock_settings,
    mock_ocr_results,
    mock_text_bubbles,
    mock_tts_results,
    mock_wav_bytes,
    mock_image_files,
    client
):
    """
    Test successful full pipeline execution.
    
    Section 14.3: Validate interactions between system components.
    """
    # Mock settings to show configured
    mock_settings.GOOGLE_VISION_CONFIGURED = True
    mock_settings.ELEVENLABS_CONFIGURED = True
    mock_settings.AUDIO_SAMPLE_RATE = 44100
    mock_settings.DEBUG = False
    
    # Setup mocks
    mock_ocr_service = Mock()
    mock_ocr_service.detect_text_batch = AsyncMock(return_value=mock_ocr_results)
    mock_get_ocr.return_value = mock_ocr_service
    
    mock_grouper = Mock()
    mock_grouper.group_into_bubbles.return_value = mock_text_bubbles
    mock_get_grouper.return_value = mock_grouper
    
    mock_tts_service = Mock()
    mock_tts_service.generate_speech_batch = AsyncMock(return_value=mock_tts_results)
    mock_get_tts.return_value = mock_tts_service
    
    mock_stitcher = Mock()
    mock_stitcher.stitch_audio_clips.return_value = mock_wav_bytes
    mock_stitcher.get_total_duration_ms.return_value = 2000
    mock_get_stitcher.return_value = mock_stitcher
    
    # Make request
    response = client.post(
        "/process/chapter",
        data={"chapter_id": "test_chapter_123"},
        files=mock_image_files
    )
    
    # Verify response
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert "chapter_test_chapter_123.wav" in response.headers["content-disposition"]
    assert response.headers["x-chapter-id"] == "test_chapter_123"
    assert response.headers["x-bubble-count"] == "3"
    assert response.headers["x-duration-ms"] == "2000"
    
    # Verify WAV file content
    assert response.content == mock_wav_bytes
    
    # Verify pipeline call order
    mock_ocr_service.detect_text_batch.assert_called_once()
    mock_grouper.group_into_bubbles.assert_called_once()
    mock_tts_service.generate_speech_batch.assert_called_once()
    mock_stitcher.stitch_audio_clips.assert_called_once()


@pytest.mark.integration
def test_process_chapter_no_images(client):
    """Test error when no images provided."""
    response = client.post(
        "/process/chapter",
        data={"chapter_id": "test_chapter"},
        files=[]
    )
    
    assert response.status_code == 422  # Validation error


@pytest.mark.integration
def test_process_chapter_too_many_images(client, mock_image_files):
    """Test error when too many images provided."""
    # Create 101 images (over the 100 limit)
    many_files = mock_image_files * 34  # 3 * 34 = 102
    
    response = client.post(
        "/process/chapter",
        data={"chapter_id": "test_chapter"},
        files=many_files
    )
    
    assert response.status_code == 400
    assert "Too many images" in response.json()["detail"]


@pytest.mark.integration
@patch('backend.routers.process.settings')
@patch('backend.routers.process.get_ocr_service')
def test_process_chapter_google_vision_not_configured(
    mock_get_ocr,
    mock_settings,
    mock_image_files,
    client
):
    """Test error when Google Vision not configured."""
    mock_settings.GOOGLE_VISION_CONFIGURED = False
    mock_settings.ELEVENLABS_CONFIGURED = True
    
    mock_ocr_service = Mock()
    mock_get_ocr.return_value = mock_ocr_service
    
    response = client.post(
        "/process/chapter",
        data={"chapter_id": "test_chapter"},
        files=mock_image_files
    )
    
    assert response.status_code == 503
    assert "Google Vision API not configured" in response.json()["detail"]


@pytest.mark.integration
@patch('backend.routers.process.settings')
@patch('backend.routers.process.get_ocr_service')
@patch('backend.routers.process.get_tts_service')
def test_process_chapter_elevenlabs_not_configured(
    mock_get_tts,
    mock_get_ocr,
    mock_settings,
    mock_image_files,
    client
):
    """Test error when ElevenLabs not configured."""
    mock_settings.GOOGLE_VISION_CONFIGURED = True
    mock_settings.ELEVENLABS_CONFIGURED = False
    
    mock_ocr_service = Mock()
    mock_get_ocr.return_value = mock_ocr_service
    
    mock_tts_service = Mock()
    mock_get_tts.return_value = mock_tts_service
    
    response = client.post(
        "/process/chapter",
        data={"chapter_id": "test_chapter"},
        files=mock_image_files
    )
    
    assert response.status_code == 503
    assert "ElevenLabs API not configured" in response.json()["detail"]


@pytest.mark.integration
@patch('backend.routers.process.get_ocr_service')
@patch('backend.routers.process.settings')
def test_process_chapter_ocr_failure(
    mock_settings,
    mock_get_ocr,
    mock_image_files,
    client
):
    """Test handling of OCR API failures."""
    # Mock settings to show configured
    mock_settings.GOOGLE_VISION_CONFIGURED = True
    mock_settings.ELEVENLABS_CONFIGURED = True
    
    mock_ocr_service = Mock()
    mock_ocr_service.detect_text_batch = AsyncMock(
        side_effect=Exception("Vision API error")
    )
    mock_get_ocr.return_value = mock_ocr_service
    
    response = client.post(
        "/process/chapter",
        data={"chapter_id": "test_chapter"},
        files=mock_image_files
    )
    
    assert response.status_code == 500
    assert "OCR processing failed" in response.json()["detail"]


@pytest.mark.integration
@patch('backend.routers.process.get_ocr_service')
@patch('backend.routers.process.settings')
def test_process_chapter_no_text_detected(
    mock_settings,
    mock_get_ocr,
    mock_image_files,
    client
):
    """Test handling when no text detected in images."""
    # Mock settings to show configured
    mock_settings.GOOGLE_VISION_CONFIGURED = True
    mock_settings.ELEVENLABS_CONFIGURED = True
    
    mock_ocr_service = Mock()
    mock_ocr_service.detect_text_batch = AsyncMock(return_value=[])
    mock_get_ocr.return_value = mock_ocr_service
    
    response = client.post(
        "/process/chapter",
        data={"chapter_id": "test_chapter"},
        files=mock_image_files
    )
    
    assert response.status_code == 400
    assert "No text detected" in response.json()["detail"]


@pytest.mark.integration
@patch('backend.routers.process.get_text_grouper')
@patch('backend.routers.process.get_ocr_service')
def test_process_chapter_text_grouping_failure(
    mock_get_ocr,
    mock_get_grouper,
    mock_ocr_results,
    mock_image_files,
    mock_settings,
    client
):
    """Test handling of text grouping failures."""
    mock_ocr_service = Mock()
    mock_ocr_service.detect_text_batch = AsyncMock(return_value=mock_ocr_results)
    mock_get_ocr.return_value = mock_ocr_service
    
    mock_grouper = Mock()
    mock_grouper.group_into_bubbles.side_effect = Exception("Grouping error")
    mock_get_grouper.return_value = mock_grouper
    
    response = client.post(
        "/process/chapter",
        data={"chapter_id": "test_chapter"},
        files=mock_image_files
    )
    
    assert response.status_code == 500
    assert "Text grouping failed" in response.json()["detail"]


@pytest.mark.integration
@patch('backend.routers.process.get_tts_service')
@patch('backend.routers.process.get_text_grouper')
@patch('backend.routers.process.get_ocr_service')
def test_process_chapter_tts_failure(
    mock_get_ocr,
    mock_get_grouper,
    mock_get_tts,
    mock_ocr_results,
    mock_text_bubbles,
    mock_image_files,
    mock_settings,
    client
):
    """Test handling of TTS API failures."""
    mock_ocr_service = Mock()
    mock_ocr_service.detect_text_batch = AsyncMock(return_value=mock_ocr_results)
    mock_get_ocr.return_value = mock_ocr_service
    
    mock_grouper = Mock()
    mock_grouper.group_into_bubbles.return_value = mock_text_bubbles
    mock_get_grouper.return_value = mock_grouper
    
    mock_tts_service = Mock()
    mock_tts_service.generate_speech_batch = AsyncMock(
        side_effect=Exception("TTS API error")
    )
    mock_get_tts.return_value = mock_tts_service
    
    response = client.post(
        "/process/chapter",
        data={"chapter_id": "test_chapter"},
        files=mock_image_files
    )
    
    assert response.status_code == 500
    assert "TTS generation failed" in response.json()["detail"]


@pytest.mark.integration
@patch('backend.routers.process.get_tts_service')
@patch('backend.routers.process.get_text_grouper')
@patch('backend.routers.process.get_ocr_service')
def test_process_chapter_tts_no_audio(
    mock_get_ocr,
    mock_get_grouper,
    mock_get_tts,
    mock_ocr_results,
    mock_text_bubbles,
    mock_image_files,
    mock_settings,
    client
):
    """Test handling when TTS generates no audio."""
    mock_ocr_service = Mock()
    mock_ocr_service.detect_text_batch = AsyncMock(return_value=mock_ocr_results)
    mock_get_ocr.return_value = mock_ocr_service
    
    mock_grouper = Mock()
    mock_grouper.group_into_bubbles.return_value = mock_text_bubbles
    mock_get_grouper.return_value = mock_grouper
    
    mock_tts_service = Mock()
    mock_tts_service.generate_speech_batch = AsyncMock(return_value=[])
    mock_get_tts.return_value = mock_tts_service
    
    response = client.post(
        "/process/chapter",
        data={"chapter_id": "test_chapter"},
        files=mock_image_files
    )
    
    assert response.status_code == 500
    assert "TTS generation produced no audio" in response.json()["detail"]


@pytest.mark.integration
@patch('backend.routers.process.get_audio_stitcher')
@patch('backend.routers.process.get_tts_service')
@patch('backend.routers.process.get_text_grouper')
@patch('backend.routers.process.get_ocr_service')
def test_process_chapter_audio_stitching_failure(
    mock_get_ocr,
    mock_get_grouper,
    mock_get_tts,
    mock_get_stitcher,
    mock_ocr_results,
    mock_text_bubbles,
    mock_tts_results,
    mock_image_files,
    mock_settings,
    client
):
    """Test handling of audio stitching failures."""
    mock_ocr_service = Mock()
    mock_ocr_service.detect_text_batch = AsyncMock(return_value=mock_ocr_results)
    mock_get_ocr.return_value = mock_ocr_service
    
    mock_grouper = Mock()
    mock_grouper.group_into_bubbles.return_value = mock_text_bubbles
    mock_get_grouper.return_value = mock_grouper
    
    mock_tts_service = Mock()
    mock_tts_service.generate_speech_batch = AsyncMock(return_value=mock_tts_results)
    mock_get_tts.return_value = mock_tts_service
    
    mock_stitcher = Mock()
    mock_stitcher.stitch_audio_clips.side_effect = Exception("Stitching error")
    mock_get_stitcher.return_value = mock_stitcher
    
    response = client.post(
        "/process/chapter",
        data={"chapter_id": "test_chapter"},
        files=mock_image_files
    )
    
    assert response.status_code == 500
    assert "Audio stitching failed" in response.json()["detail"]


@pytest.mark.integration
@patch('backend.routers.process.get_audio_stitcher')
@patch('backend.routers.process.get_tts_service')
@patch('backend.routers.process.get_text_grouper')
@patch('backend.routers.process.get_ocr_service')
def test_process_chapter_with_custom_voice(
    mock_get_ocr,
    mock_get_grouper,
    mock_get_tts,
    mock_get_stitcher,
    mock_ocr_results,
    mock_text_bubbles,
    mock_tts_results,
    mock_wav_bytes,
    mock_image_files,
    mock_settings,
    client
):
    """Test chapter processing with custom voice ID."""
    mock_ocr_service = Mock()
    mock_ocr_service.detect_text_batch = AsyncMock(return_value=mock_ocr_results)
    mock_get_ocr.return_value = mock_ocr_service
    
    mock_grouper = Mock()
    mock_grouper.group_into_bubbles.return_value = mock_text_bubbles
    mock_get_grouper.return_value = mock_grouper
    
    mock_tts_service = Mock()
    mock_tts_service.generate_speech_batch = AsyncMock(return_value=mock_tts_results)
    mock_get_tts.return_value = mock_tts_service
    
    mock_stitcher = Mock()
    mock_stitcher.stitch_audio_clips.return_value = mock_wav_bytes
    mock_stitcher.get_total_duration_ms.return_value = 2000
    mock_get_stitcher.return_value = mock_stitcher
    
    # Make request with custom voice
    response = client.post(
        "/process/chapter",
        data={
            "chapter_id": "test_chapter",
            "voice_id": "custom_voice_123"
        },
        files=mock_image_files
    )
    
    assert response.status_code == 200
    
    # Verify custom voice was passed to TTS
    call_args = mock_tts_service.generate_speech_batch.call_args
    assert call_args.kwargs.get('voice_id') == "custom_voice_123"


@pytest.mark.integration
@patch('backend.routers.process.get_audio_stitcher')
@patch('backend.routers.process.get_tts_service')
@patch('backend.routers.process.get_text_grouper')
@patch('backend.routers.process.get_ocr_service')
def test_process_chapter_text_preprocessing(
    mock_get_ocr,
    mock_get_grouper,
    mock_get_tts,
    mock_get_stitcher,
    mock_wav_bytes,
    mock_image_files,
    mock_settings,
    client
):
    """
    Test that text preprocessing is applied.
    
    Section 7: Text Preprocessing
    """
    # Create OCR results with text needing preprocessing
    ocr_results = [
        [  # Image 1
            OCRResult(
                text="He||o",
                bounding_box=BoundingBox([
                    {"x": 10, "y": 10}, {"x": 50, "y": 10},
                    {"x": 50, "y": 30}, {"x": 10, "y": 30}
                ]),
                confidence=0.99
            ),
            OCRResult(
                text="W0rld",
                bounding_box=BoundingBox([
                    {"x": 10, "y": 40}, {"x": 50, "y": 40},
                    {"x": 50, "y": 60}, {"x": 10, "y": 60}
                ]),
                confidence=0.99
            ),
        ],
    ]
    
    text_bubbles = [
        TextBubble(
            text="He||o W0rld",
            reading_order=1,
            bounding_box=BoundingBox([
                {"x": 10, "y": 10}, {"x": 50, "y": 10},
                {"x": 50, "y": 60}, {"x": 10, "y": 60}
            ])
        ),
    ]
    
    mock_ocr_service = Mock()
    mock_ocr_service.detect_text_batch = AsyncMock(return_value=ocr_results)
    mock_get_ocr.return_value = mock_ocr_service
    
    mock_grouper = Mock()
    mock_grouper.group_into_bubbles.return_value = text_bubbles
    mock_get_grouper.return_value = mock_grouper
    
    mock_tts_service = Mock()
    mock_tts_service.generate_speech_batch = AsyncMock(return_value=[
        TTSResult("Hello World.", b'audio', 1, "voice")
    ])
    mock_get_tts.return_value = mock_tts_service
    
    mock_stitcher = Mock()
    mock_stitcher.stitch_audio_clips.return_value = mock_wav_bytes
    mock_stitcher.get_total_duration_ms.return_value = 1000
    mock_get_stitcher.return_value = mock_stitcher
    
    response = client.post(
        "/process/chapter",
        data={"chapter_id": "test_chapter"},
        files=mock_image_files
    )
    
    assert response.status_code == 200
    
    # Verify preprocessed text was sent to TTS
    # "He||o W0rld" should become "HeIIo World." after preprocessing
    call_args = mock_tts_service.generate_speech_batch.call_args
    preprocessed_texts = call_args[0][0]
    assert len(preprocessed_texts) == 1
    # | → I, 0 → O, period added
    assert "HeIIo" in preprocessed_texts[0] or "Hello" in preprocessed_texts[0]
    assert preprocessed_texts[0].endswith('.')


@pytest.mark.integration
@patch('backend.routers.process.get_audio_stitcher')
@patch('backend.routers.process.get_tts_service')
@patch('backend.routers.process.get_text_grouper')
@patch('backend.routers.process.get_ocr_service')
def test_process_chapter_pipeline_call_order(
    mock_get_ocr,
    mock_get_grouper,
    mock_get_tts,
    mock_get_stitcher,
    mock_ocr_results,
    mock_text_bubbles,
    mock_tts_results,
    mock_wav_bytes,
    mock_image_files,
    mock_settings,
    client
):
    """
    Verify correct pipeline call order.
    
    Section 14.3: Verify correct call order
    """
    call_order = []
    
    mock_ocr_service = Mock()
    async def ocr_call(*args, **kwargs):
        call_order.append('ocr')
        return mock_ocr_results
    mock_ocr_service.detect_text_batch = ocr_call
    mock_get_ocr.return_value = mock_ocr_service
    
    mock_grouper = Mock()
    def grouper_call(*args, **kwargs):
        call_order.append('grouper')
        return mock_text_bubbles
    mock_grouper.group_into_bubbles = grouper_call
    mock_get_grouper.return_value = mock_grouper
    
    mock_tts_service = Mock()
    async def tts_call(*args, **kwargs):
        call_order.append('tts')
        return mock_tts_results
    mock_tts_service.generate_speech_batch = tts_call
    mock_get_tts.return_value = mock_tts_service
    
    mock_stitcher = Mock()
    def stitcher_call(*args, **kwargs):
        call_order.append('stitcher')
        return mock_wav_bytes
    mock_stitcher.stitch_audio_clips = stitcher_call
    mock_stitcher.get_total_duration_ms = Mock(return_value=2000)
    mock_get_stitcher.return_value = mock_stitcher
    
    response = client.post(
        "/process/chapter",
        data={"chapter_id": "test_chapter"},
        files=mock_image_files
    )
    
    assert response.status_code == 200
    
    # Verify correct order: OCR → Grouper → TTS → Stitcher
    assert call_order == ['ocr', 'grouper', 'tts', 'stitcher']
