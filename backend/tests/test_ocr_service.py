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
    OCRResult,
    is_noise_token
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


# Noise Filter Tests

@pytest.mark.unit
@pytest.mark.parametrize("token", [
    "HUH", "?", "!", "...", "…", "—", "YOU'VE", "--", "*****", "D",
    "I", "A", "café",                        # letters incl. accented Latin
    "20", "999", "111", "222", "007",        # real numbers the lettering may show
    "1000", "5000", "10000", "5,000", "$100",
])
def test_is_noise_token_keeps_real_text(token):
    """Words, punctuation, and real numbers pass through."""
    assert not is_noise_token(token)


@pytest.mark.unit
@pytest.mark.parametrize("token", [
    "한다", "加", "リッツザ", "ין", "།",       # Korean, CJK, Japanese, Hebrew, Tibetan in the art
    "УЕАН", "о", "рор", "водоот",            # Cyrillic, incl. look-alikes of Latin letters
    "BEGлi",                                 # mixed Latin + Cyrillic
    "☐", "©", "|", "\\",                     # stray symbols
    "000", "0000", "00000", "00005",         # leading-zero runs from round shapes/textures
    "", "   ",
])
def test_is_noise_token_drops_noise(token):
    """Non-Latin script, stray symbols, and leading-zero runs are noise."""
    assert is_noise_token(token)


def _annotation(text, x):
    """Build a mock Vision word annotation at horizontal offset x."""
    annotation = Mock()
    annotation.description = text
    annotation.bounding_poly.vertices = [
        Mock(x=x, y=20), Mock(x=x + 30, y=20), Mock(x=x + 30, y=40), Mock(x=x, y=40)
    ]
    return annotation


@pytest.mark.unit
@pytest.mark.google_vision
def test_detect_text_filters_noise_tokens():
    """Noise words are dropped before results reach bubble grouping."""
    response = Mock()
    response.error.message = ""
    response.text_annotations = [
        _annotation("00000 HUH ? УЕАН 999", 0),  # full-text annotation, skipped
        _annotation("00000", 0),
        _annotation("HUH", 40),
        _annotation("?", 80),
        _annotation("УЕАН", 120),
        _annotation("999", 160),
    ]

    service = GoogleVisionOCRService()
    service.client = MagicMock()
    service.client.text_detection.return_value = response

    results = service.detect_text(b"fake_image_data")

    assert [r.text for r in results] == ["HUH", "?", "999"]
    assert results[0].bounding_box.left == 40


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
    results = service.detect_text_batch(images)

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
def test_detect_text_batch_invalid_batch_size(monkeypatch):
    """Test batch processing with invalid parallelism setting."""
    from backend.config import settings
    monkeypatch.setattr(settings, "OCR_MAX_PARALLEL_REQUESTS", -1)
    service = GoogleVisionOCRService()
    service.client = MagicMock()

    with pytest.raises(ValueError, match="Invalid OCR_MAX_PARALLEL_REQUESTS"):
        service.detect_text_batch([b"image"])


@pytest.mark.unit
@pytest.mark.google_vision
def test_detect_text_batch_handles_failures(sample_vision_response):
    """
    Test batch processing continues on individual image failures.
    Section 6.1: Reduce OCR failures with error handling
    """
    service = GoogleVisionOCRService()
    service.client = MagicMock()
    
    # Second image fails, the others succeed. Decided by image content rather
    # than call order, since calls run concurrently.
    def text_detection(image):
        if image.content == b"image2":
            raise Exception("OCR failed")
        return sample_vision_response
    service.client.text_detection.side_effect = text_detection

    images = [b"image1", b"image2", b"image3"]
    results = service.detect_text_batch(images)
    
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


# Parallel batch OCR

import threading
import time as _time

from backend.config import settings


def _batch_service(monkeypatch, fake_detect, parallel=4):
    """Service whose detect_text is replaced by `fake_detect(image_bytes)`."""
    monkeypatch.setattr(settings, "OCR_MAX_PARALLEL_REQUESTS", parallel)
    service = GoogleVisionOCRService()
    service.client = MagicMock()  # batch method only checks it's configured
    service.detect_text = fake_detect
    return service


@pytest.mark.unit
def test_batch_preserves_order_when_calls_finish_out_of_order(monkeypatch):
    """Later images finish first, but results still line up with input order."""
    def fake_detect(image_bytes):
        idx = int(image_bytes)
        _time.sleep(0.05 * (5 - idx))  # image 0 slowest, image 4 fastest
        return [f"result-{idx}"]

    service = _batch_service(monkeypatch, fake_detect, parallel=5)
    results = service.detect_text_batch([str(i).encode() for i in range(5)])
    assert results == [[f"result-{i}"] for i in range(5)]


@pytest.mark.unit
def test_batch_failed_image_gets_empty_result_others_continue(monkeypatch):
    def fake_detect(image_bytes):
        if image_bytes == b"bad":
            raise RuntimeError("Vision API error")
        return [image_bytes.decode()]

    service = _batch_service(monkeypatch, fake_detect)
    assert service.detect_text_batch([b"a", b"bad", b"c"]) == [["a"], [], ["c"]]


@pytest.mark.unit
def test_batch_never_exceeds_parallel_limit(monkeypatch):
    lock = threading.Lock()
    active, peak = [0], [0]

    def fake_detect(image_bytes):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        _time.sleep(0.02)
        with lock:
            active[0] -= 1
        return []

    service = _batch_service(monkeypatch, fake_detect, parallel=3)
    service.detect_text_batch([b"x"] * 12)
    assert peak[0] == 3


@pytest.mark.unit
def test_batch_runs_concurrently(monkeypatch):
    """10 calls of 0.1s with 10 workers take ~0.1s, not ~1s."""
    service = _batch_service(monkeypatch, lambda b: (_time.sleep(0.1), [])[1], parallel=10)
    start = _time.perf_counter()
    service.detect_text_batch([b"x"] * 10)
    assert _time.perf_counter() - start < 0.5


@pytest.mark.unit
def test_batch_rejects_invalid_parallelism(monkeypatch):
    service = _batch_service(monkeypatch, lambda b: [], parallel=0)
    with pytest.raises(ValueError, match="OCR_MAX_PARALLEL_REQUESTS"):
        service.detect_text_batch([b"x"])


# Panel stitching (OCR_STITCH_MAX_HEIGHT)

import io as _io
from PIL import Image as _Image

from backend.services.vision import OCRResult as _OCRResult, BoundingBox as _BoundingBox


def _png(height, width=100, color="white"):
    buf = _io.BytesIO()
    _Image.new("RGB", (width, height), color).save(buf, "PNG")
    return buf.getvalue()


def _word(text, y, height=20, x=10):
    return _OCRResult(text, _BoundingBox([
        {"x": x, "y": y}, {"x": x + 40, "y": y},
        {"x": x + 40, "y": y + height}, {"x": x, "y": y + height},
    ]))


def _stitch_service(monkeypatch, fake_detect, max_height):
    monkeypatch.setattr(settings, "OCR_STITCH_MAX_HEIGHT", max_height)
    monkeypatch.setattr(settings, "OCR_MAX_PARALLEL_REQUESTS", 4)
    service = GoogleVisionOCRService()
    service.client = MagicMock()
    service.detect_text = fake_detect
    return service


@pytest.mark.unit
def test_build_strips_packs_consecutive_panels_under_max_height():
    images = [_png(100), _png(100), _png(100), _png(300)]
    strips = GoogleVisionOCRService._build_strips(images, max_height=250)
    assert [s.members for s in strips] == [
        [(0, 0, 100), (1, 100, 100)],   # 200px strip
        [(2, 0, 100)],                  # adding panel 3 would exceed 250
        [(3, 0, 300)],                  # taller than max on its own
    ]
    assert _Image.open(_io.BytesIO(strips[0].image_bytes)).size == (100, 200)


@pytest.mark.unit
def test_single_panel_strip_sends_original_bytes():
    """No re-encoding when a strip holds one panel - identical to unstitched OCR."""
    images = [_png(300), _png(300)]
    strips = GoogleVisionOCRService._build_strips(images, max_height=400)
    assert [s.image_bytes for s in strips] == images


@pytest.mark.unit
def test_undecodable_image_is_sent_alone(monkeypatch):
    images = [_png(100), b"not an image", _png(100)]
    strips = GoogleVisionOCRService._build_strips(images, max_height=1000)
    assert [s.members[0][0] for s in strips] == [0, 1, 2]
    assert strips[1].image_bytes == b"not an image"


@pytest.mark.unit
def test_stitched_words_return_to_their_panel_in_panel_coordinates(monkeypatch):
    calls = []

    def fake_detect(image_bytes):
        calls.append(image_bytes)
        h = _Image.open(_io.BytesIO(image_bytes)).height
        if h == 200:   # strip of panels 0 and 1
            return [_word("TOP", y=40), _word("BOTTOM", y=140)]
        return [_word("ALONE", y=30)]

    service = _stitch_service(monkeypatch, fake_detect, max_height=250)
    results = service.detect_text_batch([_png(100), _png(100), _png(100)])

    assert len(calls) == 2                                   # 3 panels, 2 billable calls
    assert [[w.text for w in panel] for panel in results] == [["TOP"], ["BOTTOM"], ["ALONE"]]
    assert results[0][0].bounding_box.top == 40             # first panel: no offset
    assert results[1][0].bounding_box.top == 40             # 140 in strip - 100 offset
    assert results[1][0].bounding_box.center_y == 50
    assert results[2][0].bounding_box.top == 30


@pytest.mark.unit
def test_word_straddling_a_join_goes_to_panel_holding_its_centre(monkeypatch):
    def fake_detect(image_bytes):
        return [_word("SPLIT", y=95, height=20)]   # centre y=105 → second panel
    service = _stitch_service(monkeypatch, fake_detect, max_height=500)
    results = service.detect_text_batch([_png(100), _png(100)])
    assert [len(p) for p in results] == [0, 1]
    assert results[1][0].bounding_box.top == -5              # starts just above panel 2's top edge


@pytest.mark.unit
def test_failed_strip_gives_empty_results_for_its_panels(monkeypatch):
    def fake_detect(image_bytes):
        if _Image.open(_io.BytesIO(image_bytes)).height == 200:
            raise RuntimeError("Vision API error")
        return [_word("OK", y=10)]
    service = _stitch_service(monkeypatch, fake_detect, max_height=250)
    results = service.detect_text_batch([_png(100), _png(100), _png(100)])
    assert [[w.text for w in p] for p in results] == [[], [], ["OK"]]


@pytest.mark.unit
def test_stitching_disabled_makes_one_call_per_panel(monkeypatch):
    calls = []
    service = _stitch_service(monkeypatch, lambda b: (calls.append(b), [])[1], max_height=0)
    images = [_png(100), _png(100), _png(100)]
    service.detect_text_batch(images)
    assert sorted(calls) == sorted(images)
