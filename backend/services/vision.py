"""
Google Vision OCR Service

Section 6: OCR Pipeline (Backend Only – Google Vision API)
Performs text detection on images using Google Cloud Vision API.
"""

import io
import logging
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from PIL import Image
from google.cloud import vision
from google.api_core import retry, exceptions

from backend.config import settings

logger = logging.getLogger(__name__)


# Punctuation that appears in English webtoon lettering
ALLOWED_PUNCTUATION = set("'’\".,!?;:()-–—…*~&%$#@+/“”")

# Numbers starting with 3+ zeros (000, 00000, 00005). Real numbers never start
# this way, but OCR produces them from round shapes and textures in the art.
# 1000 / 5000 / 999 / 111 don't match and are kept.
LEADING_ZEROS = re.compile(r"^0{3,}\d*$")

# Sanity limits on Vision responses
MAX_WORD_CHARS = 100
MAX_COORDINATE = 100_000


def _is_allowed_char(char: str) -> bool:
    """Latin letters (incl. accented), ASCII digits, and webtoon punctuation."""
    if char in ALLOWED_PUNCTUATION or ("0" <= char <= "9"):
        return True
    # Unicode name check rejects Cyrillic/Greek look-alikes such as 'о' or 'р'
    return char.isalpha() and unicodedata.name(char, "").startswith("LATIN")


def is_noise_token(text: str) -> bool:
    """
    Check whether a single OCR word is noise that shouldn't reach grouping.

    A word is noise if:
    - it contains any character outside Latin letters, digits, and webtoon
      punctuation - e.g. Korean/CJK/Cyrillic text in the artwork, or symbols
      like ☐ ©
    - it's a number starting with 3+ zeros (000, 00005)
    """
    text = text.strip()
    if not text or len(text) > MAX_WORD_CHARS:
        return True
    if not all(_is_allowed_char(c) for c in text):
        return True
    return bool(LEADING_ZEROS.match(text))


class BoundingBox:
    """Normalized bounding box representation."""
    
    def __init__(self, vertices: List[Dict[str, int]]):
        """
        Initialize bounding box from Vision API vertices.
        
        Args:
            vertices: List of {"x": int, "y": int} dictionaries
        """
        self.vertices = vertices
        self._calculate_stats()
    
    def _calculate_stats(self):
        """Calculate derived statistics from vertices."""
        xs = [v.get("x", 0) for v in self.vertices]
        ys = [v.get("y", 0) for v in self.vertices]
        
        self.left = min(xs)
        self.right = max(xs)
        self.top = min(ys)
        self.bottom = max(ys)
        self.center_x = sum(xs) / len(xs)
        self.center_y = sum(ys) / len(ys)
        self.width = self.right - self.left
        self.height = self.bottom - self.top
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "vertices": self.vertices,
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "width": self.width,
            "height": self.height
        }


class OCRResult:
    """Structured OCR result for a single text detection."""
    
    def __init__(self, text: str, bounding_box: BoundingBox, confidence: float = 1.0):
        """
        Initialize OCR result.
        
        Args:
            text: Detected text string
            bounding_box: Normalized bounding box
            confidence: Detection confidence (0.0 to 1.0)
        """
        self.text = text
        self.bounding_box = bounding_box
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "bounding_box": self.bounding_box.to_dict(),
            "confidence": self.confidence
        }


class GoogleVisionOCRService:
    """
    Google Vision API OCR service.
    
    Section 6.1: OCR Strategy
    - Uses TEXT_DETECTION
    - Processes images in batches
    - Handles retries and failures
    """
    
    def __init__(self):
        """Initialize Google Vision client."""
        if not settings.GOOGLE_VISION_CONFIGURED:
            logger.warning(
                "Google Vision API not configured. "
                "Set GOOGLE_APPLICATION_CREDENTIALS in .env"
            )
            self.client = None
        else:
            self.client = vision.ImageAnnotatorClient()
            logger.info("Google Vision API client initialized")
    
    def _normalize_vertices(self, vertices) -> List[Dict[str, int]]:
        """
        Normalize Vision API vertices to standard format.
        
        Args:
            vertices: Vision API BoundingPoly vertices
            
        Returns:
            List of {"x": int, "y": int} dictionaries
        """
        # Clamped so a bad response can't produce absurd boxes downstream
        return [
            {"x": min(max(int(vertex.x), 0), MAX_COORDINATE), "y": min(max(int(vertex.y), 0), MAX_COORDINATE)}
            for vertex in vertices
        ]
    
    @retry.Retry(
        predicate=retry.if_exception_type(
            exceptions.ServiceUnavailable,
            exceptions.DeadlineExceeded,
            exceptions.InternalServerError
        ),
        initial=1.0,
        maximum=10.0,
        multiplier=2.0,
        deadline=60.0
    )
    def _detect_text_with_retry(self, image: vision.Image) -> Any:
        """Call Vision API with exponential-backoff retry."""
        if self.client is None:
            raise ValueError("Google Vision API client not initialized")
        
        logger.debug("Calling Google Vision API TEXT_DETECTION")
        return self.client.text_detection(image=image, timeout=settings.EXTERNAL_API_TIMEOUT_SECONDS)
    
    def detect_text(self, image_bytes: bytes) -> List[OCRResult]:
        """
        Detect text in a single image.
        
        Args:
            image_bytes: Image data as bytes (PNG/JPEG)
            
        Returns:
            List of OCRResult objects (excluding full-text annotation)
            
        Raises:
            ValueError: If Vision API not configured
            Exception: If OCR fails after retries
        """
        if self.client is None:
            raise ValueError(
                "Google Vision API not configured. "
                "Set GOOGLE_APPLICATION_CREDENTIALS in .env"
            )
        
        # Create Vision API image
        image = vision.Image(content=image_bytes)
        
        # Call API with retry logic
        start_time = time.time()
        response = self._detect_text_with_retry(image)
        elapsed = time.time() - start_time
        
        logger.info(f"OCR completed in {elapsed:.2f}s")
        
        # Check for errors
        if response.error.message:
            error_msg = (
                f"Google Vision API error: {response.error.message}\n"
                "For more info: https://cloud.google.com/apis/design/errors"
            )
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Parse results (skip first annotation which is full text)
        results = []
        noise = []
        annotations = response.text_annotations[1:]
        if len(annotations) > settings.OCR_MAX_WORDS_PER_IMAGE:
            logger.warning(
                f"Vision returned {len(annotations)} words; keeping the first {settings.OCR_MAX_WORDS_PER_IMAGE}"
            )
            annotations = annotations[:settings.OCR_MAX_WORDS_PER_IMAGE]
        for annotation in annotations:
            if is_noise_token(annotation.description):
                noise.append(annotation.description)
                continue

            vertices = self._normalize_vertices(annotation.bounding_poly.vertices)
            bbox = BoundingBox(vertices)
            
            result = OCRResult(
                text=annotation.description,
                bounding_box=bbox,
                confidence=1.0  # Vision API doesn't provide word-level confidence
            )
            results.append(result)

        if noise:
            logger.info(f"Filtered {len(noise)} OCR noise tokens: {noise}")
        logger.info(f"Detected {len(results)} text elements")
        return results
    
    def detect_text_batch(
        self,
        images: List[bytes],
    ) -> List[List[OCRResult]]:
        """
        Detect text in multiple images.

        With OCR_STITCH_MAX_HEIGHT > 0, consecutive panels are stacked into
        tall strips so one Vision call (one billable unit) covers several
        panels; results are split back per panel with panel-relative
        coordinates, so callers see the same shape as unstitched OCR.
        Calls run up to OCR_MAX_PARALLEL_REQUESTS at a time.

        Args:
            images: List of image bytes

        Returns:
            List of OCR results per image, in the same order as `images`.
            An image whose call fails gets [] and is logged; the rest continue.

        Raises:
            ValueError: If Vision API not configured or invalid parallelism
        """
        if self.client is None:
            raise ValueError("Google Vision API not configured")

        if not images:
            return []

        if settings.OCR_STITCH_MAX_HEIGHT <= 0:
            results = self._detect_many(images)
        else:
            strips = self._build_strips(images, settings.OCR_STITCH_MAX_HEIGHT)
            logger.info(
                f"Stitched {len(images)} panels into {len(strips)} strips "
                f"(≤{settings.OCR_STITCH_MAX_HEIGHT}px) for OCR"
            )
            strip_results = self._detect_many([strip.image_bytes for strip in strips])
            results = self._split_strip_results(strips, strip_results, len(images))

        logger.info(f"Batch processing complete: {len(results)} images processed")
        return results

    def _detect_many(self, images: List[bytes]) -> List[List[OCRResult]]:
        """
        Run detect_text on each image, up to OCR_MAX_PARALLEL_REQUESTS at once.

        Results are returned in input order; a failed image gets [].
        """
        max_workers = settings.OCR_MAX_PARALLEL_REQUESTS
        if max_workers <= 0:
            raise ValueError(f"Invalid OCR_MAX_PARALLEL_REQUESTS: {max_workers}")
        max_workers = min(max_workers, len(images))

        logger.info(f"Running OCR on {len(images)} images with {max_workers} parallel requests")

        # Pre-sized so each result lands in its image's slot, whatever order calls finish in
        all_results: List[List[OCRResult]] = [[] for _ in images]

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ocr") as pool:
            futures = {pool.submit(self.detect_text, image_bytes): idx
                       for idx, image_bytes in enumerate(images)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    all_results[idx] = future.result()
                    logger.debug(f"  Image {idx + 1}/{len(images)}: {len(all_results[idx])} detections")
                except Exception as e:
                    logger.error(f"  Image {idx + 1}/{len(images)} failed: {e}")

        return all_results

    @staticmethod
    def _build_strips(images: List[bytes], max_height: int) -> List["_Strip"]:
        """
        Pack consecutive panels into strips no taller than max_height.

        A panel that can't be decoded, or a strip holding a single panel
        (e.g. one taller than max_height), is sent as the panel's original
        bytes - identical to unstitched OCR for that panel.
        """
        decoded = []
        for image_bytes in images:
            try:
                img = Image.open(io.BytesIO(image_bytes))
                img.load()
                decoded.append(img)
            except Exception as e:
                logger.warning(f"Can't decode image for stitching, sending it alone: {e}")
                decoded.append(None)

        # Group panel indices; undecodable panels always stand alone
        groups: List[List[int]] = []
        current: List[int] = []
        current_height = 0
        for idx, img in enumerate(decoded):
            if img is None:
                if current:
                    groups.append(current)
                groups.append([idx])
                current, current_height = [], 0
                continue
            if current and current_height + img.height > max_height:
                groups.append(current)
                current, current_height = [], 0
            current.append(idx)
            current_height += img.height
        if current:
            groups.append(current)

        strips = []
        for group in groups:
            if len(group) == 1:
                idx = group[0]
                height = decoded[idx].height if decoded[idx] is not None else 0
                strips.append(_Strip(images[idx], [(idx, 0, height)]))
                continue

            members, y = [], 0
            for idx in group:
                members.append((idx, y, decoded[idx].height))
                y += decoded[idx].height
            canvas = Image.new("RGB", (max(decoded[i].width for i in group), y), "white")
            for idx, offset, _ in members:
                canvas.paste(decoded[idx].convert("RGB"), (0, offset))
            buffer = io.BytesIO()
            canvas.save(buffer, "JPEG", quality=settings.OCR_STITCH_JPEG_QUALITY)
            strips.append(_Strip(buffer.getvalue(), members))
        return strips

    @staticmethod
    def _split_strip_results(
        strips: List["_Strip"],
        strip_results: List[List[OCRResult]],
        num_images: int
    ) -> List[List[OCRResult]]:
        """
        Assign each word to the panel its box's vertical centre falls in, and
        shift its box by that panel's offset into panel coordinates.
        """
        results: List[List[OCRResult]] = [[] for _ in range(num_images)]
        for strip, words in zip(strips, strip_results):
            if len(strip.members) == 1:
                # Panel OCR'd on its own: words are already in its coordinates
                results[strip.members[0][0]] = list(words)
                continue
            for word in words:
                center_y = word.bounding_box.center_y
                idx, offset = next(
                    ((i, y) for i, y, h in strip.members if y <= center_y < y + h),
                    strip.members[-1][:2]  # below the last panel's edge → last panel
                )
                if offset:
                    shifted = [{"x": v["x"], "y": v["y"] - offset}
                               for v in word.bounding_box.vertices]
                    word = OCRResult(word.text, BoundingBox(shifted), word.confidence)
                results[idx].append(word)
        return results


class _Strip:
    """Stitched OCR input: image bytes plus where each panel sits in it."""

    def __init__(self, image_bytes: bytes, members: List[tuple]):
        self.image_bytes = image_bytes
        self.members = members  # [(panel_index, y_offset, height), ...] top to bottom
