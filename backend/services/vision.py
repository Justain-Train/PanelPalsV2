"""
Google Vision OCR Service

Section 6: OCR Pipeline (Backend Only – Google Vision API)
Performs text detection on images using Google Cloud Vision API.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from google.cloud import vision
from google.api_core import retry, exceptions

from backend.config import settings

logger = logging.getLogger(__name__)


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
        return [
            {"x": vertex.x, "y": vertex.y}
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
        """
        Call Vision API with retry logic.
        
        Section 6.1: Enable retries and fallbacks
        
        Args:
            image: Vision API Image object
            
        Returns:
            Vision API response
            
        Raises:
            Exception: If API call fails after retries
        """
        if self.client is None:
            raise ValueError("Google Vision API client not initialized")
        
        logger.debug("Calling Google Vision API TEXT_DETECTION")
        return self.client.text_detection(image=image)
    
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
        for annotation in response.text_annotations[1:]:
            vertices = self._normalize_vertices(annotation.bounding_poly.vertices)
            bbox = BoundingBox(vertices)
            
            result = OCRResult(
                text=annotation.description,
                bounding_box=bbox,
                confidence=1.0  # Vision API doesn't provide word-level confidence
            )
            results.append(result)
        
        logger.info(f"Detected {len(results)} text elements")
        return results
    
    def detect_text_batch(
        self, 
        images: List[bytes],
        batch_size: Optional[int] = None
    ) -> List[List[OCRResult]]:
        """
        Detect text in multiple images.
        
        Section 6.1: Process images in backend-controlled batches
        
        Args:
            images: List of image bytes
            batch_size: Override default batch size (from config)
            
        Returns:
            List of OCR results per image
            
        Raises:
            ValueError: If Vision API not configured or invalid batch size
        """
        if self.client is None:
            raise ValueError("Google Vision API not configured")
        
        if not images:
            return []
        
        # Use configured batch size
        batch_size = batch_size or settings.GOOGLE_VISION_MAX_BATCH_SIZE
        
        if batch_size <= 0:
            raise ValueError(f"Invalid batch size: {batch_size}")
        
        logger.info(f"Processing {len(images)} images in batches of {batch_size}")
        
        all_results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(images) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Process each image in batch
            batch_results = []
            for idx, image_bytes in enumerate(batch):
                try:
                    results = self.detect_text(image_bytes)
                    batch_results.append(results)
                    logger.debug(f"  Image {idx + 1}/{len(batch)}: {len(results)} detections")
                except Exception as e:
                    logger.error(f"  Image {idx + 1}/{len(batch)} failed: {e}")
                    # Add empty results for failed image
                    batch_results.append([])
            
            all_results.extend(batch_results)
        
        logger.info(f"Batch processing complete: {len(all_results)} images processed")
        return all_results
