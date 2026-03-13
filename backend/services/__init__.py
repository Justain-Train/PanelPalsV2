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

# Text Box Classification Service
from backend.services.text_box_classifier import (
    TextBoxClassifier,
    ClassificationResult
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

# Bubble Continuation Detection Service
from backend.services.bubble_continuation import (
    BubbleContinuationDetector,
    ContinuationMatch
)

# Text Preprocessing Service
from backend.services.text_preprocessing import (
    TextPreprocessor,
    normalize_text,
    preprocess_for_classification,
    preprocess_for_tts
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
    'BubbleContinuationDetector',
    'ContinuationMatch',
    'TextBoxClassifier',
    'ClassificationResult',
    'TextPreprocessor',
    'normalize_text',
    'preprocess_for_classification',
    'preprocess_for_tts',
]
