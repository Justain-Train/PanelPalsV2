"""
Chapter Processing API Endpoint

POST /process/chapter - Full OCR → TTS pipeline

Section 4.1: API Design Principles
- RESTful endpoints
- Stateless requests
- Secure-by-default
"""

import logging
import asyncio
import io
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.services import (
    GoogleVisionOCRService,
    TextBubbleGrouper,
    ElevenLabsTTSService,
    AudioStitcher
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
    format: str = Field(default="wav", description="Audio format")


class ProcessingError(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    chapter_id: Optional[str] = Field(None, description="Chapter ID if available")


# Service initialization (lazy loaded on first request)
_ocr_service: Optional[GoogleVisionOCRService] = None
_text_grouper: Optional[TextBubbleGrouper] = None
_tts_service: Optional[ElevenLabsTTSService] = None
_audio_stitcher: Optional[AudioStitcher] = None


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


def preprocess_text(text: str) -> str:
    """
    Preprocess text before TTS.
    
    Section 7: Text Preprocessing
    - Clean OCR artifacts
    - Normalize punctuation
    - Fix common OCR errors
    
    Args:
        text: Raw OCR text
        
    Returns:
        Preprocessed text ready for TTS
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Normalize quotation marks
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    
    # Remove common OCR artifacts
    text = text.replace('|', 'I')  # Common OCR error
    text = text.replace('0', 'O')  # In dialogue context, 0 is often O
    
    # Ensure proper sentence ending
    if text and not text[-1] in '.!?':
        text += '.'
    
    return text.strip()


@router.post(
    "/chapter",
    response_model=ProcessChapterResponse,
    responses={
        200: {"description": "Chapter processed successfully, WAV file returned"},
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
    Process a complete Webtoon chapter through the full pipeline.
    
    Pipeline Steps:
    1. Validate request and check service configuration
    2. Perform OCR using Google Vision API
    3. Group text into ordered text bubbles
    4. Preprocess text for TTS
    5. Generate TTS audio in parallel
    6. Stitch audio with pauses
    7. Return WAV file
    
    Args:
        chapter_id: Unique identifier for the chapter
        images: List of chapter images in reading order
        voice_id: Optional custom voice ID (uses default if not provided)
        
    Returns:
        StreamingResponse with WAV audio file
        
    Raises:
        HTTPException: 400 for invalid input, 503 for unconfigured services
    """
    logger.info(f"Processing chapter {chapter_id} with {len(images)} images")
    
    # Step 1: Validate request
    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No images provided"
        )
    
    if len(images) > 100:  # Reasonable limit for a chapter
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many images: {len(images)}. Maximum is 100 per chapter."
        )
    
    try:
        # Initialize services
        ocr_service = get_ocr_service()
        text_grouper = get_text_grouper()
        tts_service = get_tts_service()
        audio_stitcher = get_audio_stitcher()
        
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
        
        # Step 2: Read and validate images
        logger.info(f"Reading {len(images)} images for chapter {chapter_id}")
        image_bytes_list = []
        for idx, image_file in enumerate(images):
            try:
                image_bytes = await image_file.read()
                if not image_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Image {idx} is empty"
                    )
                image_bytes_list.append(image_bytes)
            except Exception as e:
                logger.error(f"Failed to read image {idx}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to read image {idx}: {str(e)}"
                )
        
        # Step 3: Perform OCR using Google Vision API
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
        
        # Step 4: Group text bubbles per image to maintain reading order
        logger.info(f"Detected text in {len(ocr_results)} images")
        all_text_bubbles = []
        
        for image_idx, image_ocr_results in enumerate(ocr_results):
            if not image_ocr_results:
                logger.warning(f"No text detected in image {image_idx}")
                continue
            
            logger.info(f"Grouping {len(image_ocr_results)} OCR results from image {image_idx}")
            try:
                image_bubbles = text_grouper.group_into_bubbles(image_ocr_results)
                logger.info(f"Formed {len(image_bubbles)} text bubbles from image {image_idx}")
                # Add bubbles in order
                all_text_bubbles.extend(image_bubbles)
            except Exception as e:
                logger.error(f"Text grouping failed for image {image_idx}: {e}")
                # Continue with other images
                continue
        
        logger.info(f"Total text bubbles across all images: {len(all_text_bubbles)}")
        
        # Step 5: Use the combined text bubbles
        text_bubbles = all_text_bubbles
        
        if not text_bubbles:
            logger.warning(f"No text bubbles formed for chapter {chapter_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text bubbles could be formed from detected text."
            )
        
        logger.info(f"Formed {len(text_bubbles)} text bubbles")
        
        # Step 6: Preprocess text for TTS
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
        
        # Step 7: Generate TTS audio in parallel
        logger.info(f"Generating TTS for {len(preprocessed_texts)} bubbles")
        try:
            tts_results = await tts_service.generate_speech_batch(
                preprocessed_texts,
                voice_id=voice_id
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
        
        # Step 8: Stitch audio with pauses
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
        
        # Step 9: Return MP3 file as streaming response
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
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error processing chapter {chapter_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}" if settings.DEBUG else "Processing failed"
        )
