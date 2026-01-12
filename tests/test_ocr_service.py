"""
Unit tests for Google Vision OCR Service

Section 14.2: OCR-Specific Test Cases
Tests text detection with mocked Google Vision API responses.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from google.api_core import exceptions

from backend.services.vision import (
    GoogleVisionOCRService,
    BoundingBox,
    OCRResult
)


@pytest.fixture
def mock_vision_client():
    """Mock Google Vision API client."""
    mock_client = MagicMock()
    mock_response = Mock()
    mock_response.error.message = ""
    mock_response.text_annotations = []
    mock_client.text_detection.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_vision_response():
    """
    Sample Google Vision API response.
    Section 14.2: Use curated dataset with expected OCR outputs
    """
    # Mock full text annotation (index 0 - will be skipped)
    full_text = Mock()
    full_text.description = "Hello world"
    full_text.bounding_poly.vertices = [
        Mock(x=10, y=20),
        Mock(x=100, y=20),
        Mock(x=100, y=40),
        Mock(x=10, y=40)
    ]
    
    # Mock individual word annotations
    word1 = Mock()
    word1.description = "Hello"
    word1.bounding_poly.vertices = [
        Mock(x=10, y=20),
        Mock(x=50, y=20),
        Mock(x=50, y=40),
        Mock(x=10, y=40)
    ]
    
    word2 = Mock()
    word2.description = "world"
    word2.bounding_poly.vertices = [
        Mock(x=60, y=20),
        Mock(x=100, y=20),
        Mock(x=100, y=40),
        Mock(x=60, y=40)
    ]
    
    mock_response = Mock()
    mock_response.error.message = ""
    mock_response.text_annotations = [full_text, word1, word2]
    
    return mock_response


# BoundingBox Tests

@pytest.mark.unit
def test_bounding_box_initialization():
    """Test BoundingBox calculates stats correctly."""
    vertices = [
        {"x": 10, "y": 20},
        {"x": 100, "y": 20},
        {"x": 100, "y": 40},
        {"x": 10, "y": 40}
    ]
    
    bbox = BoundingBox(vertices)
    
    assert bbox.left == 10
    assert bbox.right == 100
    assert bbox.top == 20
    assert bbox.bottom == 40
    assert bbox.width == 90
    assert bbox.height == 20
    assert bbox.center_x == 55.0
    assert bbox.center_y == 30.0


@pytest.mark.unit
def test_bounding_box_to_dict():
    """Test BoundingBox serialization."""
    vertices = [
        {"x": 10, "y": 20},
        {"x": 100, "y": 20},
        {"x": 100, "y": 40},
        {"x": 10, "y": 40}
    ]
    
    bbox = BoundingBox(vertices)
    result = bbox.to_dict()
    
    assert result["vertices"] == vertices
    assert result["left"] == 10
    assert result["width"] == 90
    assert "center_x" in result


# OCRResult Tests

@pytest.mark.unit
def test_ocr_result_initialization():
    """Test OCRResult creation."""
    vertices = [
        {"x": 10, "y": 20},
        {"x": 50, "y": 20},
        {"x": 50, "y": 40},
        {"x": 10, "y": 40}
    ]
    bbox = BoundingBox(vertices)
    
    result = OCRResult(text="Hello", bounding_box=bbox, confidence=0.95)
    
    assert result.text == "Hello"
    assert result.bounding_box == bbox
    assert result.confidence == 0.95


@pytest.mark.unit
def test_ocr_result_to_dict():
    """Test OCRResult serialization."""
    vertices = [{"x": 10, "y": 20}, {"x": 50, "y": 20}, {"x": 50, "y": 40}, {"x": 10, "y": 40}]
    bbox = BoundingBox(vertices)
    result = OCRResult(text="Hello", bounding_box=bbox)
    
    data = result.to_dict()
    
    assert data["text"] == "Hello"
    assert "bounding_box" in data
    assert data["confidence"] == 1.0


# GoogleVisionOCRService Tests

@pytest.mark.unit
@patch('backend.services.vision.vision.ImageAnnotatorClient')
def test_service_initialization_configured(mock_client_class, tmp_path):
    """Test service initializes when configured."""
    # Create a temporary credentials file
    creds_file = tmp_path / "credentials.json"
    creds_file.write_text('{"type": "service_account"}')
    
    with patch('backend.services.vision.settings') as mock_settings:
        mock_settings.GOOGLE_VISION_CONFIGURED = True
        mock_settings.GOOGLE_APPLICATION_CREDENTIALS = str(creds_file)
        
        service = GoogleVisionOCRService()
        assert service.client is not None
        mock_client_class.assert_called_once()


@pytest.mark.unit
@patch('backend.services.vision.vision.ImageAnnotatorClient')
def test_service_initialization_not_configured(mock_client_class):
    """Test service handles missing configuration gracefully."""
    with patch('backend.services.vision.settings') as mock_settings:
        mock_settings.GOOGLE_VISION_CONFIGURED = False
        
        service = GoogleVisionOCRService()
        assert service.client is None


@pytest.mark.unit
@pytest.mark.google_vision
def test_detect_text_success(sample_vision_response):
    """
    Test successful text detection.
    Section 14.2: OCR-Specific Test Cases
    """
    service = GoogleVisionOCRService()
    service.client = MagicMock()
    service.client.text_detection.return_value = sample_vision_response
    
    image_bytes = b"fake_image_data"
    results = service.detect_text(image_bytes)
    
    # Should skip first annotation (full text)
    assert len(results) == 2
    assert results[0].text == "Hello"
    assert results[1].text == "world"
    
    # Check bounding boxes
    assert results[0].bounding_box.left == 10
    assert results[0].bounding_box.right == 50
    assert results[1].bounding_box.left == 60


@pytest.mark.unit
@pytest.mark.google_vision
def test_detect_text_not_configured():
    """Test error handling when API not configured."""
    service = GoogleVisionOCRService()
    service.client = None
    
    with pytest.raises(ValueError, match="not configured"):
        service.detect_text(b"fake_image")


@pytest.mark.unit
@pytest.mark.google_vision
def test_detect_text_api_error():
    """
    Test handling of Vision API errors.
    Section 6.1: Enable retries and fallbacks
    """
    service = GoogleVisionOCRService()
    service.client = MagicMock()
    
    # Mock error response
    mock_response = Mock()
    mock_response.error.message = "Invalid image format"
    service.client.text_detection.return_value = mock_response
    
    with pytest.raises(Exception, match="Google Vision API error"):
        service.detect_text(b"invalid_image")


@pytest.mark.unit
@pytest.mark.google_vision
def test_detect_text_with_retry_logic():
    """
    Test retry logic for transient failures.
    Section 6.1: Enable retries and fallbacks
    """
    service = GoogleVisionOCRService()
    service.client = MagicMock()
    
    # Simulate transient failure then success
    mock_response = Mock()
    mock_response.error.message = ""
    mock_response.text_annotations = [Mock()]  # Just full text
    
    service.client.text_detection.side_effect = [
        exceptions.ServiceUnavailable("Temporary failure"),
        mock_response
    ]
    
    image_bytes = b"fake_image"
    results = service.detect_text(image_bytes)
    
    # Should succeed after retry
    assert results == []  # Empty because only full text annotation


@pytest.mark.unit
@pytest.mark.google_vision
def test_detect_text_batch_success(sample_vision_response):
    """
    Test batch processing of multiple images.
    Section 6.1: Process images in backend-controlled batches
    """
    service = GoogleVisionOCRService()
    service.client = MagicMock()
    service.client.text_detection.return_value = sample_vision_response
    
    images = [b"image1", b"image2", b"image3"]
    results = service.detect_text_batch(images, batch_size=2)
    
    assert len(results) == 3
    for result in results:
        assert len(result) == 2  # Two words per image
        assert result[0].text == "Hello"
        assert result[1].text == "world"


@pytest.mark.unit
@pytest.mark.google_vision
def test_detect_text_batch_empty():
    """Test batch processing with empty input."""
    service = GoogleVisionOCRService()
    service.client = MagicMock()
    
    results = service.detect_text_batch([])
    assert results == []


@pytest.mark.unit
@pytest.mark.google_vision
def test_detect_text_batch_not_configured():
    """Test batch processing when API not configured."""
    service = GoogleVisionOCRService()
    service.client = None
    
    with pytest.raises(ValueError, match="not configured"):
        service.detect_text_batch([b"image"])


@pytest.mark.unit
@pytest.mark.google_vision
def test_detect_text_batch_invalid_batch_size():
    """Test batch processing with invalid batch size."""
    service = GoogleVisionOCRService()
    service.client = MagicMock()
    
    # Test with negative batch size
    with pytest.raises(ValueError, match="Invalid batch size"):
        service.detect_text_batch([b"image"], batch_size=-1)


@pytest.mark.unit
@pytest.mark.google_vision
def test_detect_text_batch_handles_failures(sample_vision_response):
    """
    Test batch processing continues on individual image failures.
    Section 6.1: Reduce OCR failures with error handling
    """
    service = GoogleVisionOCRService()
    service.client = MagicMock()
    
    # First image succeeds, second fails, third succeeds
    service.client.text_detection.side_effect = [
        sample_vision_response,
        Exception("OCR failed"),
        sample_vision_response
    ]
    
    images = [b"image1", b"image2", b"image3"]
    results = service.detect_text_batch(images, batch_size=1)
    
    assert len(results) == 3
    assert len(results[0]) == 2  # Success
    assert len(results[1]) == 0  # Failed, returns empty
    assert len(results[2]) == 2  # Success


@pytest.mark.unit
def test_normalize_vertices():
    """Test vertex normalization from Vision API format."""
    service = GoogleVisionOCRService()
    service.client = MagicMock()
    
    # Mock Vision API vertices
    mock_vertices = [
        Mock(x=10, y=20),
        Mock(x=50, y=20),
        Mock(x=50, y=40),
        Mock(x=10, y=40)
    ]
    
    normalized = service._normalize_vertices(mock_vertices)
    
    assert normalized == [
        {"x": 10, "y": 20},
        {"x": 50, "y": 20},
        {"x": 50, "y": 40},
        {"x": 10, "y": 40}
    ]
