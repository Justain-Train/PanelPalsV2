"""
Configuration management for PanelPals V2 Backend

Handles environment variables, API keys, and application settings.
Section 5.2: Secure-by-default API design.
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    """Application configuration using Pydantic BaseSettings."""
    
    # Application Settings
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000"],
        description="CORS allowed origins"
    )
    
    # Google Vision API
    GOOGLE_APPLICATION_CREDENTIALS: str = Field(
        default="",
        description="Path to Google Cloud credentials JSON"
    )
    GOOGLE_VISION_MAX_BATCH_SIZE: int = Field(
        default=16,
        description="Max images per Vision API batch request"
    )
    OCR_MAX_PARALLEL_REQUESTS: int = Field(
        default=10,
        description="Max concurrent Vision API calls per chapter (OCR is network-bound)"
    )
    OCR_STITCH_MAX_HEIGHT: int = Field(
        default=8000,
        description="Stack consecutive panels into strips up to this height (px) so each "
                    "Vision call covers several panels (~8x fewer billable units). 0 = one call per panel"
    )
    OCR_STITCH_JPEG_QUALITY: int = Field(
        default=85,
        description="JPEG quality for stitched strips (higher = bigger upload, no accuracy gain measured)"
    )

    # ElevenLabs API
    ELEVENLABS_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        description="ElevenLabs API key (masked in logs and errors)"
    )
    ELEVENLABS_VOICE_ID: str = Field(
        default="G17SuINrv2H9FC6nvetn",
        description="Default narrator voice ID for MVP"
    )
    ELEVENLABS_MAX_PARALLEL_REQUESTS: int = Field(
        default=5,
        description="Maximum parallel TTS requests - set to your ElevenLabs plan's concurrency limit"
    )
    TTS_MAX_RETRIES: int = Field(
        default=3,
        description="Retries per clip for rate-limit / busy / 5xx / network errors (exponential backoff)"
    )
    
    # Audio Processing
    AUDIO_PAUSE_DURATION_MS: int = Field(
        default=700,
        description="Pause duration between text bubbles in milliseconds"
    )
    AUDIO_OUTPUT_FORMAT: str = Field(
        default="mp3",
        description="Output audio format"
    )
    AUDIO_SAMPLE_RATE: int = Field(
        default=44100,
        description="Audio sample rate in Hz"
    )
    AUDIO_FFMPEG_TIMEOUT_SECONDS: int = Field(
        default=30,
        description="Timeout for ffmpeg conversion operations in seconds"
    )
    AUDIO_STITCH_TOTAL_TIMEOUT_SECONDS: int = Field(
        default=120,
        description="Total timeout for entire audio stitching operation in seconds"
    )
    
    # Text Bubble Grouping
    BUBBLE_MAX_VERTICAL_GAP: int = Field(
        default=100,
        description="Maximum vertical gap (pixels) to group lines into same bubble"
    )
    BUBBLE_MAX_CENTER_SHIFT: int = Field(
        default=150,
        description="Maximum horizontal center shift for same bubble"
    )

    # Dialogue/Background Classifier
    CLASSIFIER_MODE: str = Field(
        default="heuristic",
        description="'model' uses the trained ML classifier, 'heuristic' the weighted formula"
    )
    ML_MODEL_PATH: str = Field(
        default="models/best_model.joblib",
        description="Trained classifier (model_metadata.json must sit next to it)"
    )
    ML_THRESHOLD: float = Field(
        default=0.4,
        description="Minimum model P(dialogue) to keep a bubble; below 0.5 favours keeping dialogue"
    )

    # ML training data collection
    ML_COLLECT_DATA: bool = Field(
        default=False,
        description="Save every classified bubble to backend/ml/ml_data/raw during requests. Off by default: "
                    "it stores users' text and fills the training folder with unreviewed rows. "
                    "backend.ml.collect_ml_data always collects regardless of this setting"
    )

    # Expressive TTS (ElevenLabs v3/v4)
    TTS_SENTENCE_CASE: bool = Field(
        default=True,
        description="Convert all-caps bubbles to sentence case and keep ellipses (caps read as shouting)"
    )
    TTS_AUDIO_TAGS: bool = Field(
        default=True,
        description="Turn sound words (SOB, GIGGLE, SIGH...) into audio tags like [crying] on nearby dialogue"
    )

    # Security & Rate Limiting
    MAX_IMAGE_SIZE_MB: int = Field(
        default=10,
        description="Maximum image upload size in megabytes"
    )
    MAX_IMAGES_PER_REQUEST: int = Field(
        default=200,
        description="Maximum images per request (a whole chapter; longest seen so far is 175 panels)"
    )
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=10,
        description="API rate limit per client per minute (0 disables)"
    )
    API_KEYS: List[SecretStr] = Field(
        default=[],
        description='Accepted X-API-Key values, as a JSON list: ["key1","key2"]. Required unless DEBUG'
    )
    MAX_IMAGE_PIXELS: int = Field(
        default=40_000_000,
        description="Max width x height per uploaded image (guards against decompression bombs)"
    )
    
    @property
    def GOOGLE_VISION_CONFIGURED(self) -> bool:
        """Check if Google Vision API is configured."""
        return bool(self.GOOGLE_APPLICATION_CREDENTIALS and 
                os.path.exists(self.GOOGLE_APPLICATION_CREDENTIALS))
    
    @property
    def ELEVENLABS_CONFIGURED(self) -> bool:
        """Check if ElevenLabs API is configured."""
        return bool(self.ELEVENLABS_API_KEY.get_secret_value() and self.ELEVENLABS_VOICE_ID)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
