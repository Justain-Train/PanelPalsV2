"""
Chapter Processing API Endpoint

POST /process/chapter - Full OCR → TTS pipeline
"""

import logging
import asyncio
import io
from typing import List, Optional
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.services import (
    GoogleVisionOCRService,
    TextBubbleGrouper,
    TextBoxClassifier,
    ElevenLabsTTSService,
    AudioStitcher,
    BubbleContinuationDetector,
    TextPreprocessor
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/process", tags=["processing"])


# Request/Response Models
class ProcessChapterResponse(BaseModel):
    """Response model for chapter processing."""
    
    chapter_id: str = Field(..., description="Chapter identifier")
    bubble_count: int = Field(..., description="Number of text bubbles processed")
    duration_ms: int = Field(..., description="Total audio duration in milliseconds")
    sample_rate: int = Field(..., description="Audio sample rate")
    format: str = Field(default="mp3", description="Audio format")


class ProcessingError(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    chapter_id: Optional[str] = Field(None, description="Chapter ID if available")


# Service initialization (lazy loaded on first request)
_ocr_service: Optional[GoogleVisionOCRService] = None
_text_grouper: Optional[TextBubbleGrouper] = None
_text_box_classifier: Optional[TextBoxClassifier] = None
_continuation_detector: Optional[BubbleContinuationDetector] = None
_tts_service: Optional[ElevenLabsTTSService] = None
_audio_stitcher: Optional[AudioStitcher] = None
_text_preprocessor: Optional[TextPreprocessor] = None


def get_ocr_service() -> GoogleVisionOCRService:
    """Lazy load OCR service."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = GoogleVisionOCRService()
        logger.info("Initialized Google Vision OCR service")
    return _ocr_service


def get_text_grouper() -> TextBubbleGrouper:
    """Lazy load text bubble grouper."""
    global _text_grouper
    if _text_grouper is None:
        _text_grouper = TextBubbleGrouper()
        logger.info("Initialized text bubble grouper")
    return _text_grouper


def get_text_box_classifier() -> TextBoxClassifier:
    """Lazy load text box classifier."""
    global _text_box_classifier
    if _text_box_classifier is None:
        _text_box_classifier = TextBoxClassifier()
        logger.info("Initialized text box classifier")
    return _text_box_classifier


def get_continuation_detector() -> BubbleContinuationDetector:
    """Lazy load bubble continuation detector."""
    global _continuation_detector
    if _continuation_detector is None:
        _continuation_detector = BubbleContinuationDetector()
        logger.info("Initialized bubble continuation detector")
    return _continuation_detector


def get_tts_service() -> ElevenLabsTTSService:
    """Lazy load TTS service."""
    global _tts_service
    if _tts_service is None:
        _tts_service = ElevenLabsTTSService()
        logger.info("Initialized ElevenLabs TTS service")
    return _tts_service


def get_audio_stitcher() -> AudioStitcher:
    """Lazy load audio stitcher."""
    global _audio_stitcher
    if _audio_stitcher is None:
        _audio_stitcher = AudioStitcher()
        logger.info("Initialized audio stitcher")
    return _audio_stitcher


def get_text_preprocessor() -> TextPreprocessor:
    """Lazy load text preprocessor."""
    global _text_preprocessor
    if _text_preprocessor is None:
        _text_preprocessor = TextPreprocessor()
        logger.info("Initialized text preprocessor")
    return _text_preprocessor


def preprocess_text(text: str) -> str:
    """Preprocess raw OCR text for TTS."""
    if not text:
        return ""
    
    preprocessor = get_text_preprocessor()
    return preprocessor.preprocess_for_tts(text)


@router.post(
    "/chapter",
    responses={
        200: {"description": "Chapter processed successfully, MP3 file returned", "content": {"audio/mpeg": {}}},
        400: {"model": ProcessingError, "description": "Invalid request"},
        500: {"model": ProcessingError, "description": "Processing failed"},
        503: {"model": ProcessingError, "description": "Service not configured"}
    },
    summary="Process Webtoon Chapter",
    description="Full pipeline: OCR → Text Grouping → TTS → Audio Stitching"
)
async def process_chapter(
    chapter_id: str = Form(..., description="Unique chapter identifier"),
    images: List[UploadFile] = File(..., description="Ordered chapter images"),
    voice_id: Optional[str] = Form(None, description="Custom voice ID (optional)")
) -> StreamingResponse:
    """
    Process a complete Webtoon chapter through the full OCR → TTS pipeline.
    Returns an MP3 audio file.
    """
    logger.info(f"Processing chapter {chapter_id} with {len(images)} images")
    
    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No images provided"
        )
    
    if len(images) > 500:  # Reasonable limit for a chapter
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many images: {len(images)}. Maximum is 150 per chapter."
        )
    
    try:
        # Initialize services
        ocr_service = get_ocr_service()
        text_box_classifier = get_text_box_classifier()
        text_grouper = get_text_grouper()
        continuation_detector = get_continuation_detector()
        tts_service = get_tts_service()
        audio_stitcher = get_audio_stitcher()
        
        logger.info(f"🎓 ML data collection active: {text_box_classifier.ml_data_collector.output_dir}")
        
        # Check service configuration
        if not settings.GOOGLE_VISION_CONFIGURED:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Google Vision API not configured. Set GOOGLE_APPLICATION_CREDENTIALS."
            )
        
        if not settings.ELEVENLABS_CONFIGURED:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ElevenLabs API not configured. Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID."
            )
    
        
        logger.info(f"Reading {len(images)} images for chapter {chapter_id}")
        image_bytes_list = []
        image_heights = []
        for idx, image_file in enumerate(images):
            try:
                image_bytes = await image_file.read()
                if not image_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Image {idx} is empty"
                    )
                image_bytes_list.append(image_bytes)
                
                # Extract image height for continuation detection
                img = Image.open(io.BytesIO(image_bytes))
                image_heights.append(img.height)
                
            except Exception as e:
                logger.error(f"Failed to read image {idx}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to read image {idx}: {str(e)}"
                )
        
        logger.info(f"Performing OCR on {len(image_bytes_list)} images")
        try:
            ocr_results = ocr_service.detect_text_batch(image_bytes_list)
        except Exception as e:
            logger.error(f"OCR failed for chapter {chapter_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OCR processing failed: {str(e)}"
            )
        
        if not ocr_results:
            logger.warning(f"No text detected in chapter {chapter_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text detected in images. Please verify images contain readable text."
            )
        
        logger.info(f"Detected text in {len(ocr_results)} images")
        bubble_groups = []
        
        for image_idx, image_ocr_results in enumerate(ocr_results):
            if not image_ocr_results:
                logger.warning(f"No text detected in panel {image_idx}")
                bubble_groups.append([])
                continue
            
            logger.info(f"Grouping {len(image_ocr_results)} OCR results from panel {image_idx}")
            try:
                image_bubbles = text_grouper.group_into_bubbles(
                    image_ocr_results,
                    panel_id=image_idx
                )
                logger.info(f"Formed {len(image_bubbles)} text bubbles from panel {image_idx}")
                bubble_groups.append(image_bubbles)
            except Exception as e:
                logger.error(f"Text grouping failed for panel {image_idx}: {e}")
                bubble_groups.append([])
                continue
        
        logger.info(
            f"Grouped text into {len(bubble_groups)} image groups, "
            f"total {sum(len(g) for g in bubble_groups)} bubbles before continuation detection"
        )
        
        logger.info("Classifying text bubbles (dialogue vs background)")
        filtered_bubble_groups = []
        
        for image_idx, image_bubbles in enumerate(bubble_groups):
            if not image_bubbles:
                filtered_bubble_groups.append([])
                continue
            
            # Get image dimensions for classification
            img = Image.open(io.BytesIO(image_bytes_list[image_idx]))
            image_width, image_height = img.size
            
            # Filter to only dialogue/narration bubbles
            filtered_bubbles = text_box_classifier.filter_text_bubbles(
                image_bubbles,
                image_width,
                image_height
            )
            
            filtered_bubble_groups.append(filtered_bubbles)
            logger.info(
                f"Panel {image_idx}: {len(image_bubbles)} bubbles → "
                f"{len(filtered_bubbles)} dialogue (filtered {len(image_bubbles) - len(filtered_bubbles)} background)"
            )
        
        # Use filtered bubble groups
        bubble_groups = filtered_bubble_groups

        logger.info("Detecting bubble continuations across images")

        try:
            logger.info("Heights of Images: " + ", ".join(str(h) for h in image_heights))
            text_bubbles = continuation_detector.detect_and_merge_continuations(bubble_groups, image_heights)
        except Exception as e:
            logger.error(f"Bubble continuation detection failed: {e}")
            text_bubbles = []
            for group in bubble_groups:
                text_bubbles.extend(group)
        
        if not text_bubbles:
            logger.warning(f"No text bubbles formed for chapter {chapter_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text bubbles could be formed from detected text."
            )
        
        logger.info(f"Formed {len(text_bubbles)} text bubbles")
        
        logger.info("Preprocessing text for TTS")
        preprocessed_texts = []
        for bubble in text_bubbles:

            preprocessed = preprocess_text(bubble.text)
            if preprocessed:  # Only include non-empty text
                preprocessed_texts.append(preprocessed)
        
        if not preprocessed_texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid text after preprocessing."
            )
        
        logger.info(f"Preprocessed {len(preprocessed_texts)} text bubbles")
        
        
        logger.info(f"Generating TTS for {len(preprocessed_texts)} bubbles")
        try:
            tts_results = await tts_service.generate_speech_batch(
                preprocessed_texts,
                voice_id = "onwK4e9ZLuTAKqWW03F9"
            )
        except Exception as e:
            logger.error(f"TTS generation failed for chapter {chapter_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"TTS generation failed: {str(e)}"
            )
        
        if not tts_results:
            logger.error(f"TTS generated no audio for chapter {chapter_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="TTS generation produced no audio. Please try again."
            )
        
        logger.info(f"Generated {len(tts_results)} audio clips")
        
        logger.info("Stitching audio clips")
        try:
            mp3_bytes = audio_stitcher.stitch_audio_clips(tts_results, output_format="mp3")
            duration_ms = audio_stitcher.get_total_duration_ms(tts_results)
        except Exception as e:
            logger.error(f"Audio stitching failed for chapter {chapter_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Audio stitching failed: {str(e)}"
            )
        
        logger.info(
            f"Successfully processed chapter {chapter_id}: "
            f"{len(text_bubbles)} bubbles, "
            f"{duration_ms}ms duration, "
            f"{len(mp3_bytes)} bytes"
        )
        
        if text_box_classifier.collect_ml_data and text_box_classifier.ml_data_collector:
            try:
                csv_path = text_box_classifier.ml_data_collector.save()
                if csv_path:
                    stats = text_box_classifier.ml_data_collector.get_stats()
                    logger.info(
                        f"💾 ML data saved to: {csv_path}\n"
                        f"   Total: {stats['total']}, "
                        f"Dialogue: {stats['dialogue']}, "
                        f"Background: {stats['background']}, "
                        f"Needs review: {stats['needs_review']}"
                    )
            except Exception as e:
                logger.warning(f"Failed to save ML data: {e}")
        
        return StreamingResponse(
            io.BytesIO(mp3_bytes),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f'attachment; filename="chapter_{chapter_id}.mp3"',
                "X-Chapter-ID": chapter_id,
                "X-Bubble-Count": str(len(text_bubbles)),
                "X-Duration-MS": str(duration_ms),
                "X-Format": "mp3"
            }
        )

    
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing chapter {chapter_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}" if settings.DEBUG else "Processing failed"
        )