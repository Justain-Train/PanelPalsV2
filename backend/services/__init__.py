"""
Backend Services Package

Contains service modules for OCR, TTS, and audio processing.
"""

# Vision OCR Service
from backend.services.vision import (
    GoogleVisionOCRService,
    BoundingBox,
    OCRResult
)

# Text Bubble Grouping Service
from backend.services.text_grouping import (
    TextBubbleGrouper,
    TextBubble
)

# TTS Service
from backend.services.tts import (
    ElevenLabsTTSService,
    TTSResult
)

# Audio Stitching Service
from backend.services.audio import (
    AudioStitcher
)

__all__ = [
    'GoogleVisionOCRService',
    'BoundingBox',
    'OCRResult',
    'TextBubbleGrouper',
    'TextBubble',
    'ElevenLabsTTSService',
    'TTSResult',
    'AudioStitcher',
]
