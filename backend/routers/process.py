"""
Chapter Processing API Endpoint

POST /process/chapter - Full OCR → TTS pipeline
"""

import logging
import asyncio
import io
import re
from typing import List, Optional
from PIL import Image
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.security import enforce_request_limits
from backend.services.audio_tags import tag_panel, apply_tags
from backend.services.timing import StageTimer
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


ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}
UPLOAD_CHUNK_BYTES = 1024 * 1024
# Used in response headers and filenames: keep it to safe characters
CHAPTER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def client_error_detail(message: str, error: Exception) -> str:
    """Generic message for clients; the exception detail only in DEBUG (it's always logged)."""
    return f"{message}: {error}" if settings.DEBUG else message


async def read_upload_limited(upload: UploadFile, max_bytes: int) -> Optional[bytes]:
    """
    Read an upload in chunks, stopping once it exceeds max_bytes.

    Returns the bytes, or None if the file is larger than max_bytes.
    Note: Starlette has already buffered the request body by this point, so a
    total request-size cap still belongs at the reverse proxy; this keeps
    oversized files out of image decoding, OCR stitching and the Vision API.
    """
    chunks, total = [], 0
    while True:
        chunk = await upload.read(UPLOAD_CHUNK_BYTES)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)


def preprocess_text(text: str) -> str:
    """Preprocess raw OCR text for TTS."""
    if not text:
        return ""
    
    preprocessor = get_text_preprocessor()
    return preprocessor.preprocess_for_tts(text, expressive=settings.TTS_SENTENCE_CASE)


@router.post(
    "/chapter",
    responses={
        200: {"description": "Chapter processed successfully, MP3 file returned", "content": {"audio/mpeg": {}}},
        400: {"model": ProcessingError, "description": "Invalid request"},
        500: {"model": ProcessingError, "description": "Processing failed"},
        503: {"model": ProcessingError, "description": "Service not configured"}
    },
    summary="Process Webtoon Chapter",
    description="Full pipeline: OCR → Text Grouping → TTS → Audio Stitching",
    dependencies=[Depends(enforce_request_limits)]
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
    if not CHAPTER_ID_PATTERN.match(chapter_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chapter_id must be 1-64 characters: letters, digits, '-' or '_'"
        )

    logger.info(f"Processing chapter {chapter_id} with {len(images)} images")
    timer = StageTimer()

    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No images provided"
        )
    
    if len(images) > settings.MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Too many images: {len(images)}. "
                f"Maximum is {settings.MAX_IMAGES_PER_REQUEST} per request."
            )
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
        timer.lap("init")

        logger.info(f"Reading {len(images)} images for chapter {chapter_id}")
        max_image_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
        image_bytes_list = []
        image_heights = []
        for idx, image_file in enumerate(images):
            try:
                image_bytes = await read_upload_limited(image_file, max_image_bytes)
                if image_bytes is None:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Image {idx} exceeds {settings.MAX_IMAGE_SIZE_MB} MB"
                    )
                if not image_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Image {idx} is empty"
                    )

                # Validate the format and get the height for continuation detection
                img = Image.open(io.BytesIO(image_bytes))
                if img.format not in ALLOWED_IMAGE_FORMATS:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Image {idx} must be PNG, JPEG or WebP (got {img.format})"
                    )
                if img.width * img.height > settings.MAX_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Image {idx} is too large ({img.width}x{img.height} pixels)"
                    )
                image_bytes_list.append(image_bytes)
                image_heights.append(img.height)

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to read image {idx}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=client_error_detail(f"Failed to read image {idx}", e)
                )
        timer.lap("read")

        logger.info(f"Performing OCR on {len(image_bytes_list)} images")
        try:
            # Off the event loop so other requests keep being served
            ocr_results = await asyncio.to_thread(ocr_service.detect_text_batch, image_bytes_list)
            timer.lap("ocr")
        except Exception as e:
            logger.error(f"OCR failed for chapter {chapter_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=client_error_detail("OCR processing failed", e)
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
        timer.lap("group")

        logger.info("Classifying text bubbles (dialogue vs background)")
        filtered_bubble_groups = []
        
        for image_idx, image_bubbles in enumerate(bubble_groups):
            if not image_bubbles:
                filtered_bubble_groups.append([])
                continue
            
            # Get image dimensions for classification
            img = Image.open(io.BytesIO(image_bytes_list[image_idx]))
            image_width, image_height = img.size
            
            # Split into dialogue/narration and background bubbles
            filtered_bubbles, background_bubbles = text_box_classifier.split_text_bubbles(
                image_bubbles,
                image_width,
                image_height,
                image=img  # background brightness features
            )

            # Sound words (SOB, GIGGLE...) in or next to dialogue → audio tags
            if settings.TTS_AUDIO_TAGS:
                tag_panel(filtered_bubbles, background_bubbles)

            filtered_bubble_groups.append(filtered_bubbles)
            logger.info(
                f"Panel {image_idx}: {len(image_bubbles)} bubbles → "
                f"{len(filtered_bubbles)} dialogue (filtered {len(image_bubbles) - len(filtered_bubbles)} background)"
            )
        
        # Use filtered bubble groups
        bubble_groups = filtered_bubble_groups
        timer.lap("classify")

        logger.info("Detecting bubble continuations across images")

        try:
            logger.info("Heights of Images: " + ", ".join(str(h) for h in image_heights))
            text_bubbles = continuation_detector.detect_and_merge_continuations(bubble_groups, image_heights)
        except Exception as e:
            logger.error(f"Bubble continuation detection failed: {e}")
            text_bubbles = []
            for group in bubble_groups:
                text_bubbles.extend(group)
        timer.lap("merge")

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
            # Tags go on after preprocessing, which would strip the brackets.
            # A bubble that was only a sound word ("HAHAHA") is sent as just its tag.
            if settings.TTS_AUDIO_TAGS and bubble.audio_tags:
                preprocessed = apply_tags(preprocessed, bubble.audio_tags)
            if preprocessed:  # Only include non-empty text
                preprocessed_texts.append(preprocessed)
        
        if not preprocessed_texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid text after preprocessing."
            )
        
        logger.info(f"Preprocessed {len(preprocessed_texts)} text bubbles")
        timer.lap("preprocess")

        # Caps ElevenLabs spend per request, whatever OCR returned
        total_chars = sum(len(text) for text in preprocessed_texts)
        if 0 < settings.TTS_MAX_CHARS_PER_CHAPTER < total_chars:
            logger.warning(f"Chapter {chapter_id} has {total_chars} characters of text; refusing TTS")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Chapter text exceeds {settings.TTS_MAX_CHARS_PER_CHAPTER} characters"
            )

        
        logger.info(f"Generating TTS for {len(preprocessed_texts)} bubbles")
        try:
            tts_results = await tts_service.generate_speech_batch(
                preprocessed_texts,
                voice_id = "onwK4e9ZLuTAKqWW03F9"
            )
            timer.lap("tts")
        except Exception as e:
            logger.error(f"TTS generation failed for chapter {chapter_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=client_error_detail("TTS generation failed", e)
            )
        
        if not tts_results:
            logger.error(f"TTS generated no audio for chapter {chapter_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="TTS generation produced no audio. Please try again."
            )
        
        logger.info(f"Generated {len(tts_results)} audio clips")

        # Lines still without audio after retries (1-based reading order). The
        # chapter is returned without them rather than failing, but reported.
        returned_orders = {result.reading_order for result in tts_results}
        missing_lines = [n for n in range(1, len(preprocessed_texts) + 1) if n not in returned_orders]
        if missing_lines:
            logger.warning(
                f"Chapter {chapter_id} is missing audio for {len(missing_lines)}/"
                f"{len(preprocessed_texts)} lines: {missing_lines}"
            )
        
        logger.info("Stitching audio clips")
        try:
            # One pass: the duration comes from the stitched audio, no second decode
            mp3_bytes, duration_ms = await asyncio.to_thread(
                audio_stitcher.stitch_audio_clips_with_duration, tts_results, output_format="mp3"
            )
            timer.lap("stitch")
        except Exception as e:
            logger.error(f"Audio stitching failed for chapter {chapter_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=client_error_detail("Audio stitching failed", e)
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
        timer.lap("ml_save")

        logger.info(
            f"⏱️ Chapter {chapter_id} timings ({len(images)} images, "
            f"{len(preprocessed_texts)} TTS lines): {timer.summary()}"
        )

        headers = {
            "Content-Disposition": f'attachment; filename="chapter_{chapter_id}.mp3"',
            "X-Chapter-ID": chapter_id,
            "X-Bubble-Count": str(len(text_bubbles)),
            "X-Duration-MS": str(duration_ms),
            "X-Format": "mp3",
            "Server-Timing": timer.server_timing_header()
        }
        if missing_lines:
            headers["X-Missing-Lines"] = ",".join(str(n) for n in missing_lines)

        return StreamingResponse(
            io.BytesIO(mp3_bytes),
            media_type="audio/mpeg",
            headers=headers
        )



    except HTTPException:
        logger.info(f"⏱️ Chapter {chapter_id} failed after: {timer.summary()}")
        raise
    except Exception as e:
        logger.info(f"⏱️ Chapter {chapter_id} failed after: {timer.summary()}")
        logger.error(f"Unexpected error processing chapter {chapter_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}" if settings.DEBUG else "Processing failed"
        )